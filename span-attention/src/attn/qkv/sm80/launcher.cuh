/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    launcher.cuh
 */

#pragma once

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/default_gemm_configuration.h>
#include <cutlass/gemm/device/gemm_batched.h>

#include "attn/common/cutlass_type.h"
#include "attn/common/device_kernel.cuh"
#include "attn/common/tile_mapping.hpp"
#include "attn/common/utils.hpp"
#include "attn/qkv/qkv_gemv.cuh"
#include "attn/qkv/sm80/kernel.cuh"
#include "cache_quant/qcache.cuh"
#include "common/check_cuda.h"
#include "common/logger.h"
#include "utils/pack.cuh"
#include "utils/shuffle.cuh"

namespace span {

namespace detail {

template <QuantMode QMODE, int CHUNK_SIZE, int HEAD_SIZE, int GROUP_SIZE,
          typename FType>
class QKVGemmSm80 {
 public:
  SaStatus operator()(const FType* QK, const void* const* VCachePtrs,
                      FType* QKVReduceWs, FType* O, const uint32_t* seqLengths,
                      const void* qkvMappingWs, int stride, int batch,
                      int nGroups, int nChunk, int headsPerGroup,
                      uint32_t seqBlocks, cudaStream_t stream) const {
    LOG(ERROR) << "QKVGemmSm80 only support QuantMode::NONE, got "
               << to_string(QMODE) << std::endl;
    return SaStatus::INTERNAL_ERROR;
  }
};

template <int CHUNK_SIZE, int HEAD_SIZE, int GROUP_SIZE, typename FType>
class QKVGemmSm80<QuantMode::NONE, CHUNK_SIZE, HEAD_SIZE, GROUP_SIZE, FType> {
  // ==============================
  // span attn config
  // ==============================
  static constexpr QuantMode QMODE = QuantMode::NONE;
  using QCacheConfig = qcache::QCacheConfig<QMODE, FType>;
  using ComputeType = typename QCacheConfig::ComputeT;

  // ==============================
  // cutlass config
  // ==============================
  using ElementAccumulator = typename CutlassType<ComputeType>::Type;
  using ElementComputeEpilogue = ElementAccumulator;
  using ElementQK = typename CutlassType<FType>::Type;
  using ElementV = typename CutlassType<FType>::Type;
  using ElementOut = typename CutlassType<FType>::Type;

  // static CheckSameType<ElementQK, ElementV> __check;

  using LayoutQK = cutlass::layout::RowMajor;
  using LayoutV = cutlass::layout::RowMajor;
  using LayoutOut = cutlass::layout::RowMajor;

  static constexpr int MIN_TILE_M = GROUP_SIZE <= 16 ? 16 : 32;
  static constexpr int MIN_TILE_N = GROUP_SIZE <= 16 ? 128 : 64;
  static constexpr int TILE_K = CHUNK_SIZE <= 32 ? 16 : 32;

  static_assert(IsPowerOf2(GROUP_SIZE), "GROUP_SIZE must be power of 2");
  static_assert(IsPowerOf2(CHUNK_SIZE), "GROUP_SIZE must be power of 2");
  static_assert(IsPowerOf2(HEAD_SIZE), "HEAD_SIZE must be power of 2");

  // input alignment
  static_assert(HEAD_SIZE * sizeof(FType) >= 16,
                "requires HEAD_SIZE * sizeof(FType) >= 16");
  static_assert((HEAD_SIZE * sizeof(FType)) % 16 == 0,
                "requires (HEAD_SIZE * sizeof(FType)) % 16 == 0");

  static_assert((GROUP_SIZE / MIN_TILE_M) * (HEAD_SIZE / MIN_TILE_N) <=
                    CTA_MAX_WARP,
                "requires (GROUP_SIZE / MIN_TILE_M) * (HEAD_SIZE / MIN_TILE_N) "
                "<= CTA_MAX_WARP");

  //! NOTE: M needs padding
  // threadblock tile M, N, K
  // LD SMEM: (16 + 128) * 16 * sizeof(FType) * #stages
  // where #stages = CHUNK_SIZE / K = 32 / 16 = 2
  // 16-bit FP, 2 stages: 2304 * 2 B * 2 = 9 kB
  // A100: 192 kB/SM, 192 / 9 = 21 blocks/SM
  using ShapeBlockTile =
      cutlass::gemm::GemmShape<std::max(MIN_TILE_M, GROUP_SIZE),
                               std::max(MIN_TILE_N, HEAD_SIZE), TILE_K>;
  using ShapeWarpTile =
      cutlass::gemm::GemmShape<MIN_TILE_M, MIN_TILE_N, TILE_K>;
  // mma.m16n8k16
  using ShapeTensorCore = cutlass::gemm::GemmShape<16, 8, TILE_K / 2>;

  // Number of pipelines you want to use
  static constexpr int kStages = std::max(2, std::min(5, CHUNK_SIZE / TILE_K));

  // This code section describes how threadblocks are scheduled on GPU
  using ThreadblockSwizzle =
      cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle;

  // This code section describes the epilogue part of the kernel
  using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
      // data type of output matrix
      ElementOut,
      // the number of elements per vectorized memory access. For a byte, it's
      // 16 elements. This becomes the vector width of math instructions in the
      // epilogue too
      //! NOT directly 128 / cutlass::sizeof_bits<ElementOut>::value, due to
      //! potential CUTLASS internal bug which resulting in illy ordered output,
      //! e.g., a b c d e f g h -> e f g h a b c d
      std::min(4, 128 / cutlass::sizeof_bits<ElementOut>::value),
      // data type of accumulator
      ElementAccumulator,
      // data type for alpha/beta in linear combination function
      ElementComputeEpilogue,
      // no scaling needed
      cutlass::epilogue::thread::ScaleType::Kind::Nothing>;

  using GemmKernel =
      QKVGemmSm80Kernel<CHUNK_SIZE, HEAD_SIZE, ElementQK, LayoutQK, ElementV,
                        LayoutV, ElementOut, LayoutOut, ElementAccumulator,
                        ShapeBlockTile, ShapeWarpTile, ShapeTensorCore,
                        EpilogueOutputOp, ThreadblockSwizzle, kStages>;

  using GemmParams = typename GemmKernel::Params;

 public:
  /// Runs the kernel using initialized state.
  SaStatus operator()(const FType* QK, const void* const* VCachePtrs,
                      FType* QKVReduceWs, FType* O, const uint32_t* seqLengths,
                      const void* qkvMappingWs, int stride, int batch,
                      int nGroups, int nChunk, int headsPerGroup,
                      uint32_t seqBlocks, cudaStream_t stream) const {
    const auto& maxSeqLength = stride;
    const uint32_t maxQKVSeqBlocks = U32DivRU(maxSeqLength, CHUNK_SIZE);
    FType* const gemmOutput = maxQKVSeqBlocks == 1 ? O : QKVReduceWs;

    LOG(DEBUG) << "QK: " << QK << " QKVReduceWs: " << QKVReduceWs << " O: " << O
               << " gemmOutput: " << gemmOutput << std::endl;
    LOG(DEBUG) << "maxSeqLength: " << maxSeqLength << std::endl;
    LOG(DEBUG) << "maxQKVSeqBlocks: " << maxQKVSeqBlocks << std::endl;
    LOG(DEBUG) << "QK stride: " << stride << std::endl;

    cutlass::gemm::GemmCoord perProblemShape{headsPerGroup, HEAD_SIZE,
                                             CHUNK_SIZE};

    cutlass::TensorRef<ElementQK const, LayoutQK> refQK{
        reinterpret_cast<const ElementQK*>(QK), stride};
    int64_t strideQK = headsPerGroup * stride;
    cutlass::TensorRef<ElementV const, LayoutV> refV{
        static_cast<const ElementV*>(nullptr), HEAD_SIZE};
    int64_t strideV = CHUNK_SIZE * HEAD_SIZE;
    cutlass::TensorRef<ElementOut, LayoutOut> refOut{
        reinterpret_cast<ElementOut*>(gemmOutput), maxQKVSeqBlocks * HEAD_SIZE};
    int64_t strideOut = headsPerGroup * maxQKVSeqBlocks * HEAD_SIZE;

    GemmParams gemmParams{perProblemShape,
                          refQK.non_const_ref(),
                          strideQK,
                          refV.non_const_ref(),
                          strideV,
                          refOut.non_const_ref(),
                          strideOut,
                          seqLengths,
                          static_cast<const TileMapping*>(qkvMappingWs),
                          reinterpret_cast<const ElementV* const*>(VCachePtrs),
                          nChunk,
                          maxSeqLength,
                          headsPerGroup,
                          static_cast<int>(maxQKVSeqBlocks)};

    dim3 grid = dim3(seqBlocks, 1, nGroups);
    dim3 block(GemmKernel::kThreadCount, 1, 1);

    int smemSize = int(sizeof(typename GemmKernel::SharedStorage));
    if (smemSize >= (48 << 10)) {
      SA_CHECK_CUDA_RET(cudaFuncSetAttribute(
          span::Kernel<GemmKernel>, cudaFuncAttributeMaxDynamicSharedMemorySize,
          smemSize));
    }

    LOG(DEBUG) << "kStages: " << kStages << std::endl;
    LOG(DEBUG) << "grid.x: " << grid.x << " block.x: " << block.x << std::endl;

    span::Kernel<GemmKernel><<<grid, block, smemSize, stream>>>(gemmParams);
    SA_CHECK_KERNEL_RET();

    // QKV GEMV reduction
    if (maxQKVSeqBlocks > 1) {
      LOG(DEBUG) << "Reduce QKV" << std::endl;
      constexpr int RED_PACK_SIZE = 16 / sizeof(FType);
      constexpr int BLOCK = 256;
      constexpr int RED_BLOCK_X = HEAD_SIZE / RED_PACK_SIZE;
      constexpr int RED_BLOCK_Y = BLOCK / RED_BLOCK_X;
      constexpr int RED_UNROLL = 4;
      const int nHead = nGroups * headsPerGroup;

      QKVReduceKernel<HEAD_SIZE, RED_BLOCK_X, RED_BLOCK_Y, RED_PACK_SIZE,
                      RED_UNROLL, CHUNK_SIZE, FType, ComputeType>
          <<<batch * nHead, RED_BLOCK_X * RED_BLOCK_Y, 0, stream>>>(
              QKVReduceWs, O, seqLengths, maxQKVSeqBlocks, U32DivMod(nHead));
    }
    SA_CHECK_KERNEL_RET();
    return SaStatus::SUCCESS;
  }
};

}  // namespace detail

}  // namespace span
