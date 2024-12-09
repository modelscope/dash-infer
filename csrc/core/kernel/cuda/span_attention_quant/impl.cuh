/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    impl.cuh
 */

#pragma once

#include "kernels.cuh"
#include "utility/check_cuda.h"

namespace allspark {
namespace cuda {
namespace span_attention_quant {

#define SPAN_ATTN_CHECK_FUNC(EXPR)   \
  {                                  \
    auto err = (EXPR);               \
    if (err != RetConfig::Success) { \
      return err;                    \
    }                                \
  }

template <QuantMode QMODE,       // quant cache mode
          typename FType,        // datatype of the activation
          typename QType,        // datatype of the quantized kv-cache
          typename QParamType,   // scaling and zero point parameter datatype
          typename ComputeType>  // computation datatype
class QuantizedSpanAttn {
 public:
  QuantizedSpanAttn(int headSize, int nHead, int seqLength, int chunkSize,
                    int nChunk, int batch, int deviceId)
      : headSize_(headSize),
        nHead_(nHead),
        seqLength_(seqLength),
        chunkSize_(chunkSize),
        nChunk_(nChunk),
        batch_(batch) {
    const int MEM_ALIGN = 128;
    // QK
    QKSize_ =
        AlignedMemSize<MEM_ALIGN>(batch_ * nHead_ * seqLength_ * sizeof(FType));
    // QKVGemv reduction workspace
    uint32_t seqBlocks;
    switch (headSize_) {
      case 64:
        seqBlocks = U32DivRU(seqLength_, QKVGemvConfig<64>::SEQ_TILE_SIZE);
        break;
      case 128:
        seqBlocks = U32DivRU(seqLength_, QKVGemvConfig<128>::SEQ_TILE_SIZE);
        break;
      default:
        seqBlocks = 0;
        break;
    }
    QKVReduceWsSize_ = AlignedMemSize<MEM_ALIGN>(batch_ * nHead_ * seqBlocks *
                                                 headSize_ * sizeof(FType));

    // tiled softmax workspace
    if (seqLength_ >
        SoftmaxConfig::SOFTMAX_MAX_BLOCK * SoftmaxConfig::SOFTMAX_MAX_UNROLL) {
      uint32_t softmaxTileSize = SoftmaxConfig::TILED_SOFTMAX_BLOCK *
                                 SoftmaxConfig::TILED_SOFTMAX_UNROLL;
      uint32_t softmaxNTiles = U32DivRU(seqLength_, softmaxTileSize);
      uint32_t softmaxBatch = batch_ * nHead_;

      softmaxReduceWsSize_ = AlignedMemSize<MEM_ALIGN>(
          softmaxBatch * softmaxNTiles * sizeof(ComputeType));
      softmaxReduceFinishCountSize_ =
          AlignedMemSize<MEM_ALIGN>(softmaxBatch * sizeof(uint32_t));
    } else {
      softmaxReduceWsSize_ = 0;
      softmaxReduceFinishCountSize_ = 0;
    }

    AS_CHECK_CUDA(cudaDeviceGetAttribute(
        &smCount_, cudaDevAttrMultiProcessorCount, deviceId));
  }

  size_t GetWorkspaceSize() const {
    return QKSize_ + 2 * softmaxReduceWsSize_ +
           2 * softmaxReduceFinishCountSize_ + QKVReduceWsSize_;
  }

  /**
   *  Q: device memory
   *  KCachePtrs, VCachePtrs: host memory
   *  O: device memory
   * ws: device memory
   *
   * QKScale: scaling factor for QK (the common value is 1 / sqrt(headSize))
   *
   * return value:
   *     0: success
   *    -1: program error
   *    -2: cuda runtime error
   */
  int Run(const FType* Q, const QType* const* KCachePtrs,
          const QType* const* VCachePtrs, FType* O, void* ws, size_t wsSize,
          ComputeType QKScale, cudaStream_t stream) const {
    if (wsSize < GetWorkspaceSize()) {
      return RetConfig::ProgramError;
    }

    // workspace
    char* wsPool = static_cast<char*>(ws);
    FType* QK = reinterpret_cast<FType*>(wsPool);
    void* softmaxWs = wsPool + QKSize_;
    FType* QKVReduceWs =
        reinterpret_cast<FType*>(wsPool + QKSize_ + 2 * softmaxReduceWsSize_ +
                                 2 * softmaxReduceFinishCountSize_);

    // dispatch chunkSize
    switch (chunkSize_) {
      case 16:
        return DispatchHeadSize<16>(Q, KCachePtrs, VCachePtrs, QK, softmaxWs,
                                    QKVReduceWs, O, QKScale, stream);
      case 32:
        return DispatchHeadSize<32>(Q, KCachePtrs, VCachePtrs, QK, softmaxWs,
                                    QKVReduceWs, O, QKScale, stream);
      case 64:
        return DispatchHeadSize<64>(Q, KCachePtrs, VCachePtrs, QK, softmaxWs,
                                    QKVReduceWs, O, QKScale, stream);
      case 128:
        return DispatchHeadSize<128>(Q, KCachePtrs, VCachePtrs, QK, softmaxWs,
                                     QKVReduceWs, O, QKScale, stream);
      default:
        return RetConfig::ProgramError;
    }
  }

 private:
  template <int ALIGN>
  size_t AlignedMemSize(size_t s) const {
    return (s + ALIGN - 1) / ALIGN * ALIGN;
  }

  template <int CHUNK_SIZE>
  int DispatchHeadSize(const FType* Q, const QType* const* KCachePtrs,
                       const QType* const* VCachePtrs, FType* QK,
                       void* softmaxWs, FType* QKVReduceWs, FType* O,
                       ComputeType QKScale, cudaStream_t stream) const {
    switch (headSize_) {
      case 64:
        return LaunchKernel<CHUNK_SIZE, 64>(Q, KCachePtrs, VCachePtrs, QK,
                                            softmaxWs, QKVReduceWs, O, QKScale,
                                            stream);
      case 128:
        return LaunchKernel<CHUNK_SIZE, 128>(Q, KCachePtrs, VCachePtrs, QK,
                                             softmaxWs, QKVReduceWs, O, QKScale,
                                             stream);
      default:
        return RetConfig::ProgramError;
    }
  }

  template <int CHUNK_SIZE, int HEAD_SIZE>
  int LaunchKernel(const FType* Q, const QType* const* KCachePtrs,
                   const QType* const* VCachePtrs, FType* QK, void* softmaxWs,
                   FType* QKVReduceWs, FType* O, ComputeType QKScale,
                   cudaStream_t stream) const {
    // QK GEMV
    SPAN_ATTN_CHECK_FUNC(
        (QKGemv<CHUNK_SIZE, HEAD_SIZE>(Q, KCachePtrs, QK, QKScale, stream)));

    // softmax
    SPAN_ATTN_CHECK_FUNC(SoftmaxInplace(QK, softmaxWs, stream));

    // QKV GEMV
    SPAN_ATTN_CHECK_FUNC((QKVGemv<CHUNK_SIZE, HEAD_SIZE>(
        QK, VCachePtrs, QKVReduceWs, O, stream)));

    return RetConfig::Success;
  }

  template <int CHUNK_SIZE, int HEAD_SIZE>
  int QKGemv(const FType* Q, const QType* const* KCachePtrs, FType* QK,
             ComputeType QKScale, cudaStream_t stream) const {
    static_assert(sizeof(FType) >= sizeof(QType), "");
    // 16Byte LDG instruction
    const int PACK_SIZE = 16 / sizeof(FType);

    static_assert((CHUNK_SIZE & (CHUNK_SIZE - 1)) == 0,
                  "CHUNK_SIZE must be power of 2");
    static_assert((HEAD_SIZE & (HEAD_SIZE - 1)) == 0,
                  "HEAD_SIZE must be power of 2");
    static_assert(HEAD_SIZE <= 32 * PACK_SIZE && HEAD_SIZE >= PACK_SIZE,
                  "invalid HEAD_SIZE");

    const int BLOCK = 256;
    const int BLOCK_X = HEAD_SIZE / PACK_SIZE;
    const int BLOCK_Y = BLOCK / BLOCK_X;
    static_assert(BLOCK >= BLOCK_X && BLOCK % BLOCK_X == 0, "");
    static_assert(CHUNK_SIZE % BLOCK_Y == 0 || BLOCK_Y % CHUNK_SIZE == 0, "");

    const int UNROLL = 4;
    const int INNER_UNROLL = CHUNK_SIZE <= BLOCK_Y ? 1
                             : CHUNK_SIZE >= UNROLL * BLOCK_Y
                                 ? UNROLL
                                 : CHUNK_SIZE / BLOCK_Y;
    const int OUTER_UNROLL = INNER_UNROLL < UNROLL ? UNROLL / INNER_UNROLL : 1;

    uint32_t seqBlocks = U32DivRU(seqLength_, BLOCK_Y * UNROLL);

    using QParamT = QuantizeParam<QParamType>;
    using QConfT = qcache::QCacheConfig<QMODE, FType>;
    QuantizedQKGemvKernel<CHUNK_SIZE, HEAD_SIZE, BLOCK_X, BLOCK_Y, PACK_SIZE,
                          INNER_UNROLL, OUTER_UNROLL, FType, QType, QParamT,
                          ComputeType, QConfT>
        <<<batch_ * nHead_ * seqBlocks, BLOCK, 0, stream>>>(
            Q, KCachePtrs, QK, QKScale, U32DivMod(seqBlocks), U32DivMod(nHead_),
            seqLength_, nChunk_);

    AS_CHECK_CUDA(cudaGetLastError());
    return RetConfig::Success;
  }

  template <int HEAD_SIZE>
  struct QKVGemvConfig {
    static_assert(sizeof(FType) >= sizeof(QType), "");
    static const int PACK_SIZE = 16 / sizeof(FType);

    static const int BLOCK_X = HEAD_SIZE / PACK_SIZE;
    static const int BLOCK_Y = 256 / BLOCK_X;
    static const int UNROLL = 8;
    static const int SEQ_TILE_SIZE = BLOCK_Y * UNROLL;
  };

  template <int CHUNK_SIZE, int HEAD_SIZE>
  int QKVGemv(const FType* QK, const QType* const* VCachePtrs,
              FType* QKVReduceWs, FType* O, cudaStream_t stream) const {
    static_assert((CHUNK_SIZE & (CHUNK_SIZE - 1)) == 0,
                  "CHUNK_SIZE must be power of 2");
    static_assert((HEAD_SIZE & (HEAD_SIZE - 1)) == 0,
                  "HEAD_SIZE must be power of 2");

    using Config = QKVGemvConfig<HEAD_SIZE>;
    const int PACK_SIZE = Config::PACK_SIZE;
    const int BLOCK_X = Config::BLOCK_X;
    const int BLOCK_Y = Config::BLOCK_Y;
    const int UNROLL = Config::UNROLL;
    const int INNER_UNROLL = CHUNK_SIZE <= BLOCK_Y ? 1
                             : CHUNK_SIZE >= UNROLL * BLOCK_Y
                                 ? UNROLL
                                 : CHUNK_SIZE / BLOCK_Y;
    const int OUTER_UNROLL = INNER_UNROLL < UNROLL ? UNROLL / INNER_UNROLL : 1;
    using QParamT = QuantizeParam<QParamType>;
    using QConfT = qcache::QCacheConfig<QMODE, FType>;

    uint32_t seqBlocks = U32DivRU(seqLength_, BLOCK_Y * UNROLL);
    if (seqBlocks == 1) {
      QuantizedQKVGemvKernel<CHUNK_SIZE, HEAD_SIZE, BLOCK_X, BLOCK_Y, PACK_SIZE,
                             INNER_UNROLL, OUTER_UNROLL, false, FType, QType,
                             QParamT, ComputeType, QConfT>
          <<<batch_ * nHead_ * seqBlocks, BLOCK_X * BLOCK_Y, 0, stream>>>(
              QK, VCachePtrs, O, QKVReduceWs, U32DivMod(seqBlocks),
              U32DivMod(nHead_), seqLength_, nChunk_);
    } else {
      QuantizedQKVGemvKernel<CHUNK_SIZE, HEAD_SIZE, BLOCK_X, BLOCK_Y, PACK_SIZE,
                             INNER_UNROLL, OUTER_UNROLL, true, FType, QType,
                             QParamT, ComputeType, QConfT>
          <<<batch_ * nHead_ * seqBlocks, BLOCK_X * BLOCK_Y, 0, stream>>>(
              QK, VCachePtrs, O, QKVReduceWs, U32DivMod(seqBlocks),
              U32DivMod(nHead_), seqLength_, nChunk_);
    }
    AS_CHECK_CUDA(cudaGetLastError());

    // QKV GEMV reduction
    if (seqBlocks > 1) {
      const int RED_PACK_SIZE = 16 / sizeof(FType);
      const int BLOCK = 256;
      const int RED_BLOCK_X = HEAD_SIZE / RED_PACK_SIZE;
      const int RED_BLOCK_Y = BLOCK / RED_BLOCK_X;
      const int RED_UNROLL = 4;

      QKVReduceKernel<HEAD_SIZE, RED_BLOCK_X, RED_BLOCK_Y, RED_PACK_SIZE,
                      RED_UNROLL, FType, ComputeType>
          <<<batch_ * nHead_, RED_BLOCK_X * RED_BLOCK_Y, 0, stream>>>(
              QKVReduceWs, O, seqBlocks);
    }
    AS_CHECK_CUDA(cudaGetLastError());
    return RetConfig::Success;
  }

  struct SoftmaxConfig {
    // 1-CTA single pass softmax configuration
    static const int SOFTMAX_MAX_BLOCK = 1024;
    static const int SOFTMAX_MAX_UNROLL = 16;

    // multi-CTA single pass softmax configuration
    static const int TILED_SOFTMAX_BLOCK = 256;
    static const int TILED_SOFTMAX_UNROLL = 32;
    static const int TILED_SOFTMAX_MAX_NTILES = 32;
  };

  int SoftmaxInplace(FType* QK, void* softmaxWs, cudaStream_t stream) const {
    using Config = SoftmaxConfig;
    uint32_t softmaxBatch = batch_ * nHead_;
    uint32_t softmaxLength = seqLength_;

    if (softmaxLength <= 16) {
      return LaunchSoftmaxKernel<4, 32, 4>(QK, softmaxLength, softmaxBatch,
                                           stream);
    } else if (softmaxLength <= 32) {
      return LaunchSoftmaxKernel<8, 16, 4>(QK, softmaxLength, softmaxBatch,
                                           stream);
    } else if (softmaxLength <= 64) {
      return LaunchSoftmaxKernel<16, 8, 4>(QK, softmaxLength, softmaxBatch,
                                           stream);
    } else if (softmaxLength <= 128) {
      return LaunchSoftmaxKernel<32, 4, 4>(QK, softmaxLength, softmaxBatch,
                                           stream);
    } else if (softmaxLength <= 256) {
      return LaunchSoftmaxKernel<32, 4, 8>(QK, softmaxLength, softmaxBatch,
                                           stream);
    } else if (softmaxLength <= 512) {
      return LaunchSoftmaxKernel<64, 2, 8>(QK, softmaxLength, softmaxBatch,
                                           stream);
    } else if (softmaxLength <= 1024) {
      return LaunchSoftmaxKernel<128, 1, 8>(QK, softmaxLength, softmaxBatch,
                                            stream);
    } else if (softmaxLength <= 2048) {
      return LaunchSoftmaxKernel<256, 1, 8>(QK, softmaxLength, softmaxBatch,
                                            stream);
    } else if (softmaxLength <= 4096) {
      return LaunchSoftmaxKernel<512, 1, 8>(QK, softmaxLength, softmaxBatch,
                                            stream);
    } else if (softmaxLength <= 8192) {
      return LaunchSoftmaxKernel<512, 1, 16>(QK, softmaxLength, softmaxBatch,
                                             stream);
    } else if (softmaxLength <=
               Config::SOFTMAX_MAX_BLOCK * Config::SOFTMAX_MAX_UNROLL) {
      return LaunchSoftmaxKernel<Config::SOFTMAX_MAX_BLOCK, 1,
                                 Config::SOFTMAX_MAX_UNROLL>(
          QK, softmaxLength, softmaxBatch, stream);
    } else {
      return LaunchTiledSoftmaxKernel<Config::TILED_SOFTMAX_BLOCK,
                                      Config::TILED_SOFTMAX_UNROLL,
                                      Config::TILED_SOFTMAX_MAX_NTILES>(
          QK, softmaxWs, softmaxLength, softmaxBatch, stream);
    }
  }

  template <int BLOCK_X, int BLOCK_Y, int UNROLL>
  int LaunchSoftmaxKernel(FType* QK, uint32_t length, uint32_t batch,
                          cudaStream_t stream) const {
    uint32_t grid = U32DivRU(batch, BLOCK_Y);
    uint32_t block = BLOCK_X * BLOCK_Y;
    InplaceSoftmaxKernel<BLOCK_X, BLOCK_Y, UNROLL, FType, ComputeType>
        <<<grid, block, 0, stream>>>(QK, length, batch);
    AS_CHECK_CUDA(cudaGetLastError());
    return RetConfig::Success;
  }

  template <int BLOCK, int UNROLL, int MAX_NTILES>
  int LaunchTiledSoftmaxKernel(FType* QK, void* softmaxWs, uint32_t length,
                               uint32_t batch, cudaStream_t stream) const {
    uint32_t nTiles = U32DivRU(length, BLOCK * UNROLL);
    int maxBlocksPerSM;
    AS_CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxBlocksPerSM,
        TiledInplaceSoftmaxKernel<BLOCK, UNROLL, MAX_NTILES, FType,
                                  ComputeType>,
        BLOCK, 0));

    if (nTiles > MAX_NTILES || nTiles > maxBlocksPerSM * smCount_) {
      return RetConfig::ProgramError;
    }

    // workspace
    char* wsPtr = static_cast<char*>(softmaxWs);
    ComputeType* maxReduceWs = reinterpret_cast<ComputeType*>(wsPtr);
    ComputeType* sumReduceWs =
        reinterpret_cast<ComputeType*>(wsPtr + softmaxReduceWsSize_);
    uint32_t* maxReduceFinishCount =
        reinterpret_cast<uint32_t*>(wsPtr + 2 * softmaxReduceWsSize_);
    uint32_t* sumReduceFinishCount = reinterpret_cast<uint32_t*>(
        wsPtr + 2 * softmaxReduceWsSize_ + softmaxReduceFinishCountSize_);

    // set the finish count workspace to 0
    AS_CHECK_CUDA(cudaMemsetAsync(wsPtr + 2 * softmaxReduceWsSize_, 0,
                                  2 * softmaxReduceFinishCountSize_, stream));

    TiledInplaceSoftmaxKernel<BLOCK, UNROLL, MAX_NTILES, FType, ComputeType>
        <<<dim3(nTiles, batch), BLOCK, 0, stream>>>(
            QK, maxReduceWs, maxReduceFinishCount, sumReduceWs,
            sumReduceFinishCount, length, batch);
    AS_CHECK_CUDA(cudaGetLastError());
    return RetConfig::Success;
  }

  int headSize_;
  int nHead_;
  int seqLength_;
  int chunkSize_;
  int nChunk_;
  int batch_;

  // workspace size in byte
  size_t QKSize_;
  size_t softmaxReduceWsSize_;
  size_t softmaxReduceFinishCountSize_;
  size_t QKVReduceWsSize_;

  int smCount_;
};

#undef SPAN_ATTN_CHECK_FUNC

}  // namespace span_attention_quant
}  // namespace cuda
}  // namespace allspark
