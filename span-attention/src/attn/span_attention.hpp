/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    span_attention.hpp
 */

#pragma once

#include <span_attn.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include "attn/common/config.h"
#include "attn/common/tile_mapping.hpp"
#include "attn/common/utils.hpp"
#include "attn/qk/qk.hpp"
#include "attn/qkv/qkv.hpp"
#include "attn/softmax/softmax.hpp"
#include "common/check_cuda.h"
#include "common/logger.h"

namespace span {

template <typename FType>
class SpanAttn {
  template <int HEAD_SIZE>
  using QKGemvConfig = QKGemvConfig<HEAD_SIZE, FType>;
  template <int HEAD_SIZE>
  using QKVGemvConfig = QKVGemvConfig<HEAD_SIZE, FType>;
  using SftmxConfig = SoftmaxConfig<FType>;

 public:
  SpanAttn(int headSize, int nHead, int nGroups, const int* seqLengths,
           int chunkSize, int nChunk, int batch, QuantMode kvQuantMode,
           const cudaDeviceProp& deviceProp, bool forceSimt)
      : seqLengths_(seqLengths, seqLengths + batch),
        maxLength_(*std::max_element(seqLengths, seqLengths + batch)),
        //* NOTE: QK workspace is aligned to chunk size for QKV requires padding
        qkStride_(
            static_cast<int>(U32DivRU(maxLength_, chunkSize) * chunkSize)),
        headSize_(headSize),
        nHead_(nHead),
        nGroups_(nGroups),
        headsPerGroup_(nHead / nGroups),
        chunkSize_(chunkSize),
        nChunk_(nChunk),
        batch_(batch),
        kvQuantMode_(kvQuantMode),
        smCount_(deviceProp.multiProcessorCount),
        smMajor_(deviceProp.major),
        useGEMM_(false) {
    // TODO: tune to decide GEMV/GEMM boundary
    constexpr int GEMM_MIN_KV_HEADS = 4;

    // whether use GEMM for QK/QKV
    // GEMM only applies to 16-bit FP types
    if (!std::is_same_v<FType, float>) {
      if (kvQuantMode_ == QuantMode::NONE) {
        if (headsPerGroup_ >= GEMM_MIN_KV_HEADS) {
          useGEMM_ = true;
        }
      }
    }

    if (forceSimt) {
      useGEMM_ = false;
    }

    qkUseGEMM_ = false;
    qkvUseGEMM_ = useGEMM_;

    // tile mapping
    int qkvSeqTileSize;
    SaStatus err;
    switch (headSize_) {
      case 128: {
        err = computeQKQKVMappings<128>(&qkvSeqTileSize);
        break;
      }
      default:
        LOG(ERROR) << "unsupported head size: " << headSize_ << std::endl;
        err = SaStatus::PARAM_ERROR;
        break;
    }
    SA_CHECK(err);

    if (maxLength_ >
        SftmxConfig::SOFTMAX_MAX_BLOCK * SftmxConfig::SOFTMAX_MAX_UNROLL) {
      constexpr int SOFTMAX_TILE_SIZE =
          SftmxConfig::TILED_SOFTMAX_SEQ_TILE_SIZE;
      SA_CHECK(BuildTileMapping(softmaxMappings_, seqLengths_.begin(),
                                SOFTMAX_TILE_SIZE, batch_, maxLength_));
    }

    LOG(DEBUG) << "qkMappings_.size(): " << qkMappings_.size() << std::endl
               << "qkvMappings_.size(): " << qkvMappings_.size() << std::endl
               << "softmaxMappings_.size(): " << softmaxMappings_.size()
               << std::endl;

    seqLengthsWsSize_ = AlignedMemSize<MEM_ALIGN>(batch_ * sizeof(uint32_t));
    qkMappingsWsSize_ =
        AlignedMemSize<MEM_ALIGN>(qkMappings_.size() * sizeof(TileMapping));
    qkvMappingsWsSize_ =
        AlignedMemSize<MEM_ALIGN>(qkvMappings_.size() * sizeof(TileMapping));
    softmaxMappingsWsSize_ = AlignedMemSize<MEM_ALIGN>(softmaxMappings_.size() *
                                                       sizeof(TileMapping));

    QKSize_ =
        AlignedMemSize<MEM_ALIGN>(batch_ * nHead_ * qkStride_ * sizeof(FType));
    softmaxWsSize_ =
        SoftmaxInplaceWorkspaceBytes<FType>(batch_, nHead_, maxLength_);
    SA_CHECK(dispatchGemmArch(qkvUseGEMM_,
                              // func SIMT
                              QKVWorkspaceBytes<SaArch::SIMT, FType>(),
                              // func SM80
                              QKVWorkspaceBytes<SaArch::SM80, FType>(),
                              // args
                              &QKVReduceWsSize_, batch_, nHead_, qkStride_,
                              headSize_, qkvSeqTileSize));
  }

  size_t GetDeviceWorkspaceSize() const {
    return GetHostWorkspaceSize() + QKSize_ + softmaxWsSize_ + QKVReduceWsSize_;
  }

  size_t GetHostWorkspaceSize() const {
    return seqLengthsWsSize_ + qkMappingsWsSize_ + qkvMappingsWsSize_ +
           softmaxMappingsWsSize_;
  }

  /**
   *  Q: device memory
   *  KCachePtrs, VCachePtrs: device memory
   *  O: device memory
   *  ws: device memory
   *
   * return value:
   *     0: success
   *    -1: program error
   *    -2: cuda runtime error
   */
  SaStatus Run(const FType* Q, const void* const* KCachePtrs,
               const void* const* VCachePtrs, FType* O, void* deviceWs,
               size_t deviceWsSize, void* hostWs, size_t hostWsSize,
               float QKScale, cudaStream_t stream) const {
    if (deviceWsSize < GetDeviceWorkspaceSize()) {
      LOG(ERROR) << "device workspace size too small: got " << deviceWsSize
                 << " bytes, should be at least " << GetDeviceWorkspaceSize()
                 << " bytes" << std::endl;
      return SaStatus::PARAM_ERROR;
    }

    if (hostWsSize < GetHostWorkspaceSize()) {
      LOG(ERROR) << "host workspace size too small: got " << hostWsSize
                 << " bytes, should be at least " << GetHostWorkspaceSize()
                 << " bytes" << std::endl;
      return SaStatus::PARAM_ERROR;
    }

    char* hostCpPtr = static_cast<char*>(hostWs);
    memcpy(hostCpPtr, seqLengths_.data(),
           seqLengths_.size() * sizeof(uint32_t));
    memcpy(hostCpPtr + seqLengthsWsSize_, qkMappings_.data(),
           qkMappings_.size() * sizeof(TileMapping));
    memcpy(hostCpPtr + seqLengthsWsSize_ + qkMappingsWsSize_,
           qkvMappings_.data(), qkvMappings_.size() * sizeof(TileMapping));
    memcpy(
        hostCpPtr + seqLengthsWsSize_ + qkMappingsWsSize_ + qkvMappingsWsSize_,
        softmaxMappings_.data(), softmaxMappings_.size() * sizeof(TileMapping));

    // prepare workspace
    SA_CHECK_CUDA_RET(cudaMemcpyAsync(deviceWs, hostWs, GetHostWorkspaceSize(),
                                      cudaMemcpyHostToDevice, stream));

    // workspace ptrs
    char* wsPool = static_cast<char*>(deviceWs);
    uint32_t* seqLengthsWs = reinterpret_cast<uint32_t*>(wsPool);
    void* qkMappingWs = wsPool + seqLengthsWsSize_;
    void* qkvMappingWs = wsPool + seqLengthsWsSize_ + qkMappingsWsSize_;
    void* softmaxMappingWs =
        wsPool + seqLengthsWsSize_ + qkMappingsWsSize_ + qkvMappingsWsSize_;
    FType* QK = reinterpret_cast<FType*>(
        wsPool + seqLengthsWsSize_ + qkMappingsWsSize_ + qkvMappingsWsSize_ +
        softmaxMappingsWsSize_);
    void* softmaxWs = wsPool + seqLengthsWsSize_ + qkMappingsWsSize_ +
                      qkvMappingsWsSize_ + softmaxMappingsWsSize_ + QKSize_;
    FType* QKVReduceWs = reinterpret_cast<FType*>(
        wsPool + seqLengthsWsSize_ + qkMappingsWsSize_ + qkvMappingsWsSize_ +
        softmaxMappingsWsSize_ + QKSize_ + softmaxWsSize_);

    LOG(DEBUG) << "Q:" << Q << " KCachePtrs:" << KCachePtrs
               << " VCachePtrs:" << VCachePtrs << " O:" << O
               << " seqLengthsWs:" << seqLengthsWs
               << " qkMappingWs:" << qkMappingWs
               << " qkvMappingWs:" << qkvMappingWs
               << " softmaxMappingWs:" << softmaxMappingWs << " QK:" << QK
               << " softmaxWs:" << softmaxWs << " QKVReduceWs:" << QKVReduceWs
               << std::endl;

    SA_CHECK_RET(launchQK(Q, KCachePtrs, QK, seqLengthsWs, qkMappingWs, QKScale,
                          stream));
    SA_CHECK_RET(
        launchSoftmax(QK, seqLengthsWs, softmaxMappingWs, softmaxWs, stream));
    SA_CHECK_RET(launchQKV(QK, VCachePtrs, QKVReduceWs, O, seqLengthsWs,
                           qkvMappingWs, qkStride_, stream));

    return SaStatus::SUCCESS;
  }

 private:
  // ================== Helpers ==================

  template <typename FuncSimt, typename FuncSm80, typename... Args>
  [[nodiscard]] SaStatus dispatchGemmArch(bool gemmPred, FuncSimt&& funcSimt,
                                          FuncSm80&& funcSm80,
                                          Args&&... args) const {
    if (gemmPred) {
      if (smMajor_ >= 8) {
        LOG(DEBUG) << "dispatch SM80" << std::endl;
        return funcSm80(std::forward<Args>(args)...);
      } else {
        LOG(DEBUG) << "dispatch SIMT" << std::endl;
        return funcSimt(std::forward<Args>(args)...);
      }
    } else {
      LOG(DEBUG) << "dispatch SIMT" << std::endl;
      return funcSimt(std::forward<Args>(args)...);
    }
  }

  template <int HEAD_SIZE>
  SaStatus computeQKQKVMappings(int* qkvSeqTileSizePtr) {
    auto buildQkMappingFunc = [&](int qkTileSize) -> SaStatus {
      SA_CHECK_RET(BuildTileMapping(qkMappings_, seqLengths_.begin(),
                                    qkTileSize, batch_, maxLength_));
      return SaStatus::SUCCESS;
    };

    auto buildQkvMappingFunc = [&](int qkvTileSize) -> SaStatus {
      SA_CHECK_RET(BuildTileMapping(qkvMappings_, seqLengths_.begin(),
                                    qkvTileSize, batch_, maxLength_));
      return SaStatus::SUCCESS;
    };

    SA_CHECK_RET(dispatchGemmArch(
        qkUseGEMM_,
        // func SIMT
        [&]() -> SaStatus {
          return buildQkMappingFunc(QKGemvConfig<HEAD_SIZE>::SEQ_TILE_SIZE);
        },
        // func SM80
        [&]() -> SaStatus { return buildQkMappingFunc(chunkSize_); }));

    SA_CHECK_RET(dispatchGemmArch(
        qkvUseGEMM_,
        // func SIMT
        [&, qkvSeqTileSizePtr]() -> SaStatus {
          *qkvSeqTileSizePtr = QKVGemvConfig<HEAD_SIZE>::SEQ_TILE_SIZE;
          return buildQkvMappingFunc(QKVGemvConfig<HEAD_SIZE>::SEQ_TILE_SIZE);
        },
        // func SM80
        [&, qkvSeqTileSizePtr]() -> SaStatus {
          *qkvSeqTileSizePtr = chunkSize_;
          return buildQkvMappingFunc(chunkSize_);
        }));
    return SaStatus::SUCCESS;
  }

  // ================== Launchers ==================

  SaStatus launchQK(const FType* Q, const void* const* KCachePtrs, FType* QK,
                    const uint32_t* seqLengths, const void* qkMappingWs,
                    float QKScale, cudaStream_t stream) const {
    if (qkMappings_.size() > std::numeric_limits<uint32_t>::max()) {
      LOG(ERROR) << "number of CTAs required for QK exceeds uint32_t: "
                 << qkMappings_.size() << std::endl;
      return SaStatus::EXCEED_LIMIT_ERROR;
    }
    uint32_t seqBlocks = static_cast<uint32_t>(qkMappings_.size());

    SA_CHECK_RET(dispatchGemmArch(
        useGEMM_,
        // func SIMT (nop)
        []() { return SaStatus::SUCCESS; },
        // func SM80
        //* NOTE: QK needs to be zeroed out because QKV requires padding along
        //* sequence length (K-dim of QKV GEMM)
        [&]() -> SaStatus {
          SA_CHECK_CUDA_RET(cudaMemsetAsync(QK, 0, QKSize_, stream));
          return SaStatus::SUCCESS;
        }));
    SA_CHECK_RET(dispatchGemmArch(qkUseGEMM_,
                                  // func SIMT
                                  QKLauncher<SaArch::SIMT, FType>(),
                                  // func SM80
                                  QKLauncher<SaArch::SM80, FType>(),
                                  // args
                                  Q, KCachePtrs, QK, seqLengths, qkMappingWs,
                                  kvQuantMode_, QKScale, qkStride_, nGroups_,
                                  nChunk_, chunkSize_, headSize_,
                                  headsPerGroup_, seqBlocks, stream));
    return SaStatus::SUCCESS;
  }

  SaStatus launchQKV(const FType* QK, const void* const* VCachePtrs,
                     FType* QKVReduceWs, FType* O, const uint32_t* seqLengths,
                     const void* qkvMappingWs, int stride,
                     cudaStream_t stream) const {
    if (qkvMappings_.size() > std::numeric_limits<uint32_t>::max()) {
      LOG(ERROR) << "number of CTAs required for QKV exceeds uint32_t: "
                 << qkvMappings_.size() << std::endl;
      return SaStatus::EXCEED_LIMIT_ERROR;
    }
    uint32_t seqBlocks = static_cast<uint32_t>(qkvMappings_.size());

    return dispatchGemmArch(qkvUseGEMM_,
                            // func SIMT
                            QKVLauncher<SaArch::SIMT, FType>(),
                            // func SM80
                            QKVLauncher<SaArch::SM80, FType>(),
                            // args
                            QK, VCachePtrs, QKVReduceWs, O, seqLengths,
                            qkvMappingWs, kvQuantMode_, qkStride_, batch_,
                            nGroups_, nChunk_, chunkSize_, headSize_,
                            headsPerGroup_, seqBlocks, stream);
  }

  SaStatus launchSoftmax(FType* QK, const uint32_t* seqLengths,
                         const void* softmaxMappingWs, void* softmaxWs,
                         cudaStream_t stream) const {
    if (softmaxMappings_.size() > std::numeric_limits<uint32_t>::max()) {
      LOG(ERROR) << "number of CTAs required for Softmax exceeds uint32_t: "
                 << softmaxMappings_.size() << std::endl;
      return SaStatus::EXCEED_LIMIT_ERROR;
    }
    uint32_t totalTiles = static_cast<uint32_t>(softmaxMappings_.size());
    return SoftmaxInplace(QK, seqLengths, softmaxMappingWs, softmaxWs, batch_,
                          nHead_, qkStride_, maxLength_, totalTiles, smCount_,
                          stream);
  }

  // ================== Members ==================
  std::vector<uint32_t> seqLengths_;
  std::vector<TileMapping> qkMappings_;
  std::vector<TileMapping> qkvMappings_;
  std::vector<TileMapping> softmaxMappings_;
  int maxLength_;
  int qkStride_;
  int headSize_;
  int nHead_;
  int nGroups_;
  int headsPerGroup_;
  int chunkSize_;
  int nChunk_;
  int batch_;
  QuantMode kvQuantMode_;
  int smCount_;
  int smMajor_;
  bool useGEMM_;

  bool qkUseGEMM_;
  bool qkvUseGEMM_;

  // workspace size in byte
  size_t seqLengthsWsSize_;
  size_t qkMappingsWsSize_;
  size_t qkvMappingsWsSize_;
  size_t softmaxMappingsWsSize_;
  size_t QKSize_;
  size_t softmaxWsSize_;
  size_t QKVReduceWsSize_;
};

}  // namespace span
