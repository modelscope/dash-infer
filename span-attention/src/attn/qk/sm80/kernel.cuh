/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    kernel.cuh
 */

#pragma once

#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/linear_combination.h>

#include "attn/common/config.h"
#include "attn/common/tile_mapping.hpp"
#include "attn/common/utils.hpp"

namespace span {

namespace detail {

template <int CHUNK_SIZE, int HEAD_SIZE, typename ElementQ, typename LayoutQ,
          typename ElementK, typename LayoutK, typename ElementOut,
          typename LayoutOut, typename ElementAccumulator,
          typename ShapeMMAThreadBlock, typename ShapeMMAWarp,
          typename ShapeMMAOp, typename EpilogueOutputOp,
          typename ThreadblockSwizzle, int kStages>
class QKGemmSm80Kernel {
  using MMAOp = cutlass::arch::OpClassTensorOp;

  using SmArch = cutlass::arch::Sm80;

  static constexpr int kAlignmentQ =
      128 / cutlass::sizeof_bits<ElementQ>::value;
  static constexpr int kAlignmentK =
      128 / cutlass::sizeof_bits<ElementK>::value;

  using Operator = cutlass::arch::OpMultiplyAdd;

  static constexpr bool kSplitKSerial = false;

  // cutlass::gemm::kernel::Gemm
  using DefaultGemmKernel = typename cutlass::gemm::kernel::DefaultGemm<
      ElementQ, LayoutQ, kAlignmentQ, ElementK, LayoutK, kAlignmentK,
      ElementOut, LayoutOut, ElementAccumulator, MMAOp, SmArch,
      ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp, EpilogueOutputOp,
      ThreadblockSwizzle, kStages, kSplitKSerial, Operator>::GemmKernel;

  // ==============================
  // cutlass threadblock MMA and Epilogue
  // ==============================

  // cutlass::gemm::threadblock::MmaPipelined
  // from cutlass::gemm::threadblock::DefaultMma
  using Mma = typename DefaultGemmKernel::Mma;
  // cutlass::epilogue::threadblock::Epilogue
  using Epilogue = typename DefaultGemmKernel::Epilogue;
  // cutlass::epilogue::thread::LinearCombination
  using OutputOp = typename Epilogue::OutputOp;
  /// Warp count (concept: GemmShape)
  using WarpCount = typename Mma::WarpCount;

 public:
  static constexpr int kThreadCount = WARP_SIZE * WarpCount::kCount;

  /// Shared memory storage structure
  union SharedStorage {
    typename Mma::SharedStorage mainLoop;
    // cutlass::epilogue::threadblock::EpilogueBase<>::SharedStorage
    typename Epilogue::SharedStorage epilogue;
  };

  struct Params {
    cutlass::gemm::GemmCoord problemSize;
    typename Mma::IteratorA::Params paramsQ;
    typename Mma::IteratorA::TensorRef refQ;
    int64_t strideQ;
    typename Mma::IteratorB::Params paramsK;
    typename Mma::IteratorB::TensorRef refK;
    int64_t strideK;
    typename Epilogue::OutputTileIterator::Params paramsOut;
    typename Epilogue::OutputTileIterator::TensorRef refOut;
    int64_t strideOut;
    const uint32_t* seqLengths;
    const TileMapping* mappings;
    const ElementK* const* spanPtrArray;
    int spanPtrStride;
    int maxSeqLength;
    int headsPerGroup;
    ElementAccumulator alpha;
    int gemmKIterations;

    //
    // Methods
    //

    HOST_DEVICE_FUNC Params() {};

    HOST_DEVICE_FUNC Params(
        const cutlass::gemm::GemmCoord& problemSize_,
        typename Mma::IteratorA::TensorRef refQ_, int64_t strideQ_,
        typename Mma::IteratorB::TensorRef refK_, int64_t strideK_,
        typename Epilogue::OutputTileIterator::TensorRef refOut_,
        int64_t strideOut_, const uint32_t* seqLengths_,
        const TileMapping* mappings_, const ElementK* const* spanPtrArray_,
        int spanPtrStride_, int maxSeqLength_, int headsPerGroup_,
        ElementAccumulator alpha_)
        : problemSize(problemSize_),
          paramsQ(refQ_.layout()),
          refQ(refQ_),
          strideQ(strideQ_),
          paramsK(refK_.layout()),
          refK(refK_),
          strideK(strideK_),
          paramsOut(refOut_.layout()),
          refOut(refOut_),
          strideOut(strideOut_),
          seqLengths(seqLengths_),
          mappings(mappings_),
          spanPtrArray(spanPtrArray_),
          spanPtrStride(spanPtrStride_),
          maxSeqLength(maxSeqLength_),
          headsPerGroup(headsPerGroup_),
          alpha(alpha_),
          gemmKIterations((HEAD_SIZE + Mma::Shape::kK - 1) / Mma::Shape::kK) {}
  };

  HOST_DEVICE_FUNC QKGemmSm80Kernel() {};

  //* From cutlass/include/cutlass/gemm/kernel/gemm_batched.h:L146
  /// Executes one GEMM
  DEVICE_FUNC void operator()(const Params& params,
                              SharedStorage& sharedStorage) const {
#if __CUDA_ARCH__ < 800
    if (blockIdx.x == 0 && threadIdx.x == 0) {
#ifndef NDEBUG
      printf(
          "QKGemmSm80Kernel error: unsupported CUDA arch %d; "
          "hint: check `-arch' compiler flag\n",
          __CUDA_ARCH__);
#else
      printf(
          "QKGemmSm80Kernel error: does not support CUDA arch < 800; "
          "hint: check `-arch' compiler flag\n");
#endif  // NDEBUG
    }
    // just abort
    __trap();
    return;
#endif  // __CUDA_ARCH__

    // batching
    const uint32_t globalTileId = blockIdx.x;
    const uint32_t nGroup = gridDim.z;
    const uint32_t headId = blockIdx.z;

    const uint32_t attnBatchId = params.mappings[globalTileId].batchId;
    const uint32_t seqBlockId = params.mappings[globalTileId].tileId;

    const uint32_t seqLength = params.seqLengths[attnBatchId];
    const uint32_t tileSeqIdx0 = seqBlockId * CHUNK_SIZE;

    // adjust shapes and pointers
    auto problemSize = params.problemSize;
    bool fullTile = tileSeqIdx0 + CHUNK_SIZE <= seqLength;
    if (!fullTile) {
      problemSize[problemSize.kN] = seqLength - tileSeqIdx0;
    };

    auto refQ = params.refQ;
    refQ.add_pointer_offset(attnBatchId * nGroup * params.headsPerGroup *
                            HEAD_SIZE);

    auto refK = params.refK;
    const ElementK* const* span_ptr =
        params.spanPtrArray + attnBatchId * params.spanPtrStride + seqBlockId;
    refK.reset(const_cast<ElementK*>(*span_ptr));

    auto refOut = params.refOut;
    refOut.add_pointer_offset(attnBatchId * nGroup * params.headsPerGroup *
                                  params.maxSeqLength +
                              seqBlockId * CHUNK_SIZE);

    // Compute position within threadblock
    int tid = threadIdx.x;

    // Construct iterators to A and B operands
    typename Mma::IteratorA iteratorQ(params.paramsQ, refQ.data(),
                                      problemSize.mk(), tid);
    iteratorQ.add_pointer_offset(params.strideQ * headId);

    typename Mma::IteratorB iteratorK(params.paramsK, refK.data(),
                                      problemSize.kn(), tid);
    iteratorK.add_pointer_offset(params.strideK * headId);

    //
    // Main loop
    //
    // Broadcast the warp_id computed by lane 0 to ensure dependent code
    // is compiled as warp-uniform.
    int warpId = cutlass::canonical_warp_idx_sync();
    int laneId = cutlass::canonical_lane_idx();

    Mma mma(sharedStorage.mainLoop, tid, warpId, laneId);

    typename Mma::FragmentC accumulators;
    accumulators.clear();
    // Compute threadblock-scoped matrix multiply-add
    mma(params.gemmKIterations, accumulators, iteratorQ, iteratorK,
        accumulators);

    //
    // Epilogue
    //
    typename OutputOp::Params epilogueParam{params.alpha};
    OutputOp outputOp(epilogueParam);

    // Tile iterator writing to output tile
    typename Epilogue::OutputTileIterator iteratorOut(
        params.paramsOut, refOut.data(), problemSize.mn(), tid);
    iteratorOut.add_pointer_offset(params.strideOut * headId);

    Epilogue epilogue(sharedStorage.epilogue, tid, warpId, laneId);
    // run efficient epilogue
    epilogue(outputOp, iteratorOut, accumulators, iteratorOut);

    return;
  }
};

}  // namespace detail

}  // namespace span
