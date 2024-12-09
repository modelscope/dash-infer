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

template <int CHUNK_SIZE, int HEAD_SIZE, typename ElementQK, typename LayoutQK,
          typename ElementV, typename LayoutV, typename ElementOut,
          typename LayoutOut, typename ElementAccumulator,
          typename ShapeMMAThreadBlock, typename ShapeMMAWarp,
          typename ShapeMMAOp, typename EpilogueOutputOp,
          typename ThreadblockSwizzle, int kStages>
class QKVGemmSm80Kernel {
  using MMAOp = cutlass::arch::OpClassTensorOp;

  using SmArch = cutlass::arch::Sm80;

  static constexpr int kAlignmentQK =
      128 / cutlass::sizeof_bits<ElementQK>::value;
  static constexpr int kAlignmentV =
      128 / cutlass::sizeof_bits<ElementV>::value;

  using Operator = cutlass::arch::OpMultiplyAdd;

  static constexpr bool kSplitKSerial = false;

  // cutlass::gemm::kernel::Gemm
  using DefaultGemmKernel = typename cutlass::gemm::kernel::DefaultGemm<
      ElementQK, LayoutQK, kAlignmentQK, ElementV, LayoutV, kAlignmentV,
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

  static constexpr int kFullTileGemmKIterations =
      (CHUNK_SIZE + Mma::Shape::kK - 1) / Mma::Shape::kK;

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
    typename Mma::IteratorA::Params paramsQK;
    typename Mma::IteratorA::TensorRef refQK;
    int64_t strideQK;
    typename Mma::IteratorB::Params paramsV;
    typename Mma::IteratorB::TensorRef refV;
    int64_t strideV;
    typename Epilogue::OutputTileIterator::Params paramsOut;
    typename Epilogue::OutputTileIterator::TensorRef refOut;
    int64_t strideOut;
    const uint32_t* seqLengths;
    const TileMapping* mappings;
    const ElementV* const* spanPtrArray;
    int spanPtrStride;
    int maxSeqLength;
    int headsPerGroup;
    int maxQKVSeqBlocks;

    //
    // Methods
    //

    HOST_DEVICE_FUNC Params() {};

    HOST_DEVICE_FUNC Params(
        const cutlass::gemm::GemmCoord& problemSize_,
        typename Mma::IteratorA::TensorRef refQK_, int64_t strideQK_,
        typename Mma::IteratorB::TensorRef refV_, int64_t strideV_,
        typename Epilogue::OutputTileIterator::TensorRef refOut_,
        int64_t strideOut_, const uint32_t* seqLengths_,
        const TileMapping* mappings_, const ElementV* const* spanPtrArray_,
        int spanPtrStride_, int maxSeqLength_, int headsPerGroup_,
        int maxQKVSeqBlocks_)
        : problemSize(problemSize_),
          paramsQK(refQK_.layout()),
          refQK(refQK_),
          strideQK(strideQK_),
          paramsV(refV_.layout()),
          refV(refV_),
          strideV(strideV_),
          paramsOut(refOut_.layout()),
          refOut(refOut_),
          strideOut(strideOut_),
          seqLengths(seqLengths_),
          mappings(mappings_),
          spanPtrArray(spanPtrArray_),
          spanPtrStride(spanPtrStride_),
          maxSeqLength(maxSeqLength_),
          headsPerGroup(headsPerGroup_),
          maxQKVSeqBlocks(maxQKVSeqBlocks_) {}
  };

  HOST_DEVICE_FUNC QKVGemmSm80Kernel() {};

  //* From cutlass/include/cutlass/gemm/kernel/gemm_batched.h:L146
  /// Executes one GEMM
  DEVICE_FUNC void operator()(const Params& params,
                              SharedStorage& sharedStorage) const {
#if __CUDA_ARCH__ < 800
    if (blockIdx.x == 0 && threadIdx.x == 0) {
#ifndef NDEBUG
      printf(
          "QKVGemmSm80Kernel error: unsupported CUDA arch %d; "
          "hint: check `-arch' compiler flag\n",
          __CUDA_ARCH__);
#else
      printf(
          "QKVGemmSm80Kernel error: does not support CUDA arch < 800; "
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

    // adjust shapes and pointers
    auto problemSize = params.problemSize;
    auto gemmKIterations = kFullTileGemmKIterations;
#if 0
    const uint32_t seqLength = params.seqLengths[attnBatchId];
    const uint32_t tileSeqIdx0 = seqBlockId * CHUNK_SIZE;
    bool fullTile = tileSeqIdx0 + CHUNK_SIZE <= seqLength;
    if (!fullTile) {
      if (threadIdx.x == 0) {
        printf("batch %d block %d: partial tile\n", attnBatchId, seqBlockId);
      }
      problemSize[problemSize.kK] = seqLength - tileSeqIdx0;
      constexpr int kMmaK = Mma::Shape::kK;
      gemmKIterations = (problemSize[problemSize.kK] + kMmaK - 1) / kMmaK;
    };
#endif

    auto refQK = params.refQK;
    refQK.add_pointer_offset(attnBatchId * nGroup * params.headsPerGroup *
                                 params.maxSeqLength +
                             seqBlockId * CHUNK_SIZE);

    auto refV = params.refV;
    const ElementV* const* span_ptr =
        params.spanPtrArray + attnBatchId * params.spanPtrStride + seqBlockId;
    refV.reset(const_cast<ElementV*>(*span_ptr));

    auto refOut = params.refOut;
    refOut.add_pointer_offset(attnBatchId * nGroup * params.headsPerGroup *
                                  params.maxQKVSeqBlocks * HEAD_SIZE +
                              seqBlockId * HEAD_SIZE);

    // Compute position within threadblock
    int tid = threadIdx.x;

    // Construct iterators to A and B operands
    typename Mma::IteratorA iteratorQK(params.paramsQK, refQK.data(),
                                       problemSize.mk(), tid);
    iteratorQK.add_pointer_offset(params.strideQK * headId);

    typename Mma::IteratorB iteratorV(params.paramsV, refV.data(),
                                      problemSize.kn(), tid);
    iteratorV.add_pointer_offset(params.strideV * headId);

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
    mma(gemmKIterations, accumulators, iteratorQK, iteratorV, accumulators);

    //
    // Epilogue
    //
    OutputOp outputOp(typename OutputOp::Params{});

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
