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
#include "attn/qk/sm80/kernel.cuh"
#include "cache_quant/qcache.cuh"
#include "common/check_cuda.h"
#include "common/logger.h"
#include "utils/pack.cuh"
#include "utils/shuffle.cuh"

namespace span {

namespace detail {

template <QuantMode QMODE, int CHUNK_SIZE, int HEAD_SIZE, int GROUP_SIZE,
          typename FType>
class QKGemmSm80 {
 public:
  SaStatus operator()(const FType* Q, const void* const* KCachePtrs, FType* QK,
                      const uint32_t* seqLengths, const void* qkMappingWs,
                      float QKScale, int stride, int nGroups, int nChunk,
                      int headsPerGroup, uint32_t seqBlocks,
                      cudaStream_t stream) const {
    LOG(ERROR) << "QKGemmSm80 only support QuantMode::NONE, got "
               << to_string(QMODE) << std::endl;
    return SaStatus::INTERNAL_ERROR;
  }
};

template <int CHUNK_SIZE, int HEAD_SIZE, int GROUP_SIZE, typename FType>
class QKGemmSm80<QuantMode::NONE, CHUNK_SIZE, HEAD_SIZE, GROUP_SIZE, FType> {
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
  using ElementQ = typename CutlassType<FType>::Type;
  using ElementK = typename CutlassType<FType>::Type;
  using ElementOut = typename CutlassType<FType>::Type;

  // static CheckSameType<ElementQ, ElementK> __check;

  using LayoutQ = cutlass::layout::RowMajor;
  using LayoutK = cutlass::layout::ColumnMajor;
  using LayoutOut = cutlass::layout::RowMajor;

  static constexpr int MIN_TILE_M = GROUP_SIZE <= 16 ? 16 : 32;
  static constexpr int MIN_TILE_N = CHUNK_SIZE <= 32 ? 32 : 64;
  static constexpr int TILE_K = 32;

  static_assert(HEAD_SIZE >= TILE_K, "requires HEAD_SIZE >= TILE_K");
  static_assert(HEAD_SIZE % TILE_K == 0, "requires HEAD_SIZE % TILE_K == 0");
  static_assert(IsPowerOf2(GROUP_SIZE), "GROUP_SIZE must be power of 2");
  static_assert(IsPowerOf2(CHUNK_SIZE), "GROUP_SIZE must be power of 2");

  // Number of pipelines you want to use
  static constexpr int kStages = HEAD_SIZE / TILE_K;

  //! NOTE: M needs padding
  // threadblock tile M, N, K
  // LD SMEM: (16 + 32) * 32 * sizeof(FType) * #stages
  // where #stages = HEAD_SIZE / K = 128 / 32 = 4
  // 16-bit FP, 4 stages: 1536 * 2 B * 4 = 12 kB
  // A100: 192 kB/SM, 192 / 12 = 6 blocks/SM
  using ShapeBlockTile =
      cutlass::gemm::GemmShape<std::max(MIN_TILE_M, GROUP_SIZE),
                               std::max(MIN_TILE_N, CHUNK_SIZE), TILE_K>;
  // only 1 warp per block
  using ShapeWarpTile =
      cutlass::gemm::GemmShape<MIN_TILE_M, MIN_TILE_N, TILE_K>;
  // mma.m16n8k16
  using ShapeTensorCore = cutlass::gemm::GemmShape<16, 8, 16>;

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
      //! NOT 128 / cutlass::sizeof_bits<ElementOut>::value,
      //! otherwise will pollute the empty slots after the last token when
      //! seqlen is not a multiple of CHUNK_SIZE
      1,
      // data type of accumulator
      ElementAccumulator,
      // data type for alpha/beta in linear combination function
      ElementComputeEpilogue,
      // scaling accumulator only
      cutlass::epilogue::thread::ScaleType::Kind::OnlyAlphaScaling>;

  using GemmKernel =
      QKGemmSm80Kernel<CHUNK_SIZE, HEAD_SIZE, ElementQ, LayoutQ, ElementK,
                       LayoutK, ElementOut, LayoutOut, ElementAccumulator,
                       ShapeBlockTile, ShapeWarpTile, ShapeTensorCore,
                       EpilogueOutputOp, ThreadblockSwizzle, kStages>;

  using GemmParams = typename GemmKernel::Params;

 public:
  /// Runs the kernel using initialized state.
  SaStatus operator()(const FType* Q, const void* const* KCachePtrs, FType* QK,
                      const uint32_t* seqLengths, const void* qkMappingWs,
                      float QKScale, int stride, int nGroups, int nChunk,
                      int headsPerGroup, uint32_t seqBlocks,
                      cudaStream_t stream) const {
    const auto& maxSeqLength = stride;
    cutlass::gemm::GemmCoord perProblemShape{headsPerGroup, CHUNK_SIZE,
                                             HEAD_SIZE};

    cutlass::TensorRef<ElementQ const, LayoutQ> refQ{
        reinterpret_cast<const ElementQ*>(Q), HEAD_SIZE};
    int64_t strideQ = headsPerGroup * HEAD_SIZE;
    cutlass::TensorRef<ElementK const, LayoutK> refK{
        static_cast<const ElementK*>(nullptr), HEAD_SIZE};
    int64_t strideK = CHUNK_SIZE * HEAD_SIZE;
    cutlass::TensorRef<ElementOut, LayoutOut> refOut{
        reinterpret_cast<ElementOut*>(QK), stride};
    int64_t strideOut = headsPerGroup * stride;

    GemmParams gemmParams{perProblemShape,
                          refQ.non_const_ref(),
                          strideQ,
                          refK.non_const_ref(),
                          strideK,
                          refOut.non_const_ref(),
                          strideOut,
                          seqLengths,
                          static_cast<const TileMapping*>(qkMappingWs),
                          reinterpret_cast<const ElementK* const*>(KCachePtrs),
                          nChunk,
                          maxSeqLength,
                          headsPerGroup,
                          QKScale};

    dim3 grid = dim3(seqBlocks, 1, nGroups);
    dim3 block(GemmKernel::kThreadCount, 1, 1);

    int smemSize = int(sizeof(typename GemmKernel::SharedStorage));
    if (smemSize >= (48 << 10)) {
      SA_CHECK_CUDA_RET(cudaFuncSetAttribute(
          span::Kernel<GemmKernel>, cudaFuncAttributeMaxDynamicSharedMemorySize,
          smemSize));
    }

    span::Kernel<GemmKernel><<<grid, block, smemSize, stream>>>(gemmParams);
    SA_CHECK_KERNEL_RET();
    return SaStatus::SUCCESS;
  }
};

}  // namespace detail

}  // namespace span
