/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    sgmv.cu
 */

/*
 * SgmvCutlass source code copied from:
 * Punica https://github.com/punica-ai/punica
 */

#include <interface/allspark_check.h>
#include <utility/check.h>

#include <string>

#include "cuda_kernel.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"

namespace allspark {
namespace cuda {

template <typename T>
struct cutlass_dtype {
  using type = T;
};

template <>
struct cutlass_dtype<half> {
  using type = cutlass::half_t;
};

template <>
struct cutlass_dtype<hie::bfloat16> {
  using type = cutlass::bfloat16_t;
};

template <typename T>
__global__ void precompute_sgmv_args(
    cutlass::gemm::GemmCoord* all_problems, T** ptr_y, T** ptr_x, T** ptr_w,
    int64_t* ld_y, int64_t* ld_x, int64_t* ld_w, T* y, T* x, T** w,
    const int32_t* s, const int32_t* ranks, bool is_k_tensor, bool is_n_tensor,
    int d_in, int d_out, bool unsplit, int unsplit_n, int max_rank) {
  int i = blockIdx.x;
  int m = s[i * 2 + 1] - s[i * 2];
  int k = d_in;
  int n = ranks[i];
  if (is_k_tensor == true) {
    // lora_B
    k = ranks[i];
    n = d_out;
    if (unsplit == true) {
      // attention.self
      ptr_x[i] = x + s[i * 2] * (max_rank / 3);
      ptr_y[i] = y + s[i * 2] * unsplit_n;
    } else {
      ptr_x[i] = x + s[i * 2] * max_rank;
      ptr_y[i] = y + s[i * 2] * n;
    }
  } else if (is_n_tensor == true) {
    // lora_A
    k = d_in;
    n = ranks[i];
    ptr_x[i] = x + s[i * 2] * k;
    ptr_y[i] = y + s[i * 2] * max_rank;
  }
  // 已经在父函数处理了else的异常情况

  all_problems[i] = cutlass::gemm::GemmCoord(m, n, k);
  ptr_w[i] = w[i];
  if (unsplit == true) {
    ld_x[i] = k;
    ld_w[i] = unsplit_n;
    ld_y[i] = unsplit_n;
  } else {
    ld_x[i] = k;
    ld_w[i] = n;
    ld_y[i] = n;
  }
}

size_t sgmv_tmp_size(int num_problems) {
  constexpr auto sz = sizeof(void*) * 3 + sizeof(int64_t) * 3 +
                      sizeof(cutlass::gemm::GemmCoord);
  return sz * num_problems;
}

template <typename T>
inline T* alloc_from_buf(void** buf, int n) {
  auto* p = (T*)*buf;
  *buf = (void*)(p + n);
  return p;
}

template <typename DType>
bool SgmvCutlass(DType* y, const DType* x, const DType** w, const int32_t* s,
                 const int32_t* ranks, void* tmp_d, bool is_k_tensor,
                 bool is_n_tensor, int num_problems, int d_in, int d_out,
                 bool unsplit, int unsplit_n, int max_rank, int CC,
                 cudaStream_t stream) {
  using cutlass_t = typename cutlass_dtype<DType>::type;

  auto ptr_Y = alloc_from_buf<cutlass_t*>(&tmp_d, num_problems);
  auto ptr_X = alloc_from_buf<cutlass_t*>(&tmp_d, num_problems);
  auto ptr_W = alloc_from_buf<cutlass_t*>(&tmp_d, num_problems);
  auto ld_Y = alloc_from_buf<int64_t>(&tmp_d, num_problems);
  auto ld_X = alloc_from_buf<int64_t>(&tmp_d, num_problems);
  auto ld_W = alloc_from_buf<int64_t>(&tmp_d, num_problems);
  auto all_problems =
      alloc_from_buf<cutlass::gemm::GemmCoord>(&tmp_d, num_problems);
  AS_ENFORCE((is_k_tensor == true || is_n_tensor == true) &&
             (is_k_tensor != is_n_tensor));
  AS_ENFORCE((unsplit == false) || (is_k_tensor == unsplit));

  precompute_sgmv_args<<<num_problems, 1, 0, stream>>>(
      all_problems, ptr_Y, ptr_X, ptr_W, ld_Y, ld_X, ld_W, (cutlass_t*)y,
      (cutlass_t*)x, (cutlass_t**)w, s, ranks, is_k_tensor, is_n_tensor, d_in,
      d_out, unsplit, unsplit_n, max_rank);

  using cutlass::epilogue::thread::LinearCombination;
  using cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle;
  using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
      cutlass_t,                                      // Element A
      cutlass::layout::RowMajor,                      // Layout A
      cutlass::ComplexTransform::kNone,               //
      8,                                              // Granularity A
      cutlass_t,                                      // Element B
      cutlass::layout::RowMajor,                      // Layout B
      cutlass::ComplexTransform::kNone,               //
      8,                                              // Granularity B
      cutlass_t,                                      // Element C&D
      cutlass::layout::RowMajor,                      // Layout C&D
      float,                                          // Element Accumulator
      cutlass::arch::OpClassTensorOp,                 // Operator Class Tag
      cutlass::arch::Sm80,                            // Architecture
      cutlass::gemm::GemmShape<32, 64, 64>,           // Thread Block Shape
      cutlass::gemm::GemmShape<16, 32, 64>,           // Warp Shape
      cutlass::gemm::GemmShape<16, 8, 16>,            // Instruction Shape
      LinearCombination<cutlass_t, 1, float, float>,  // Epilogue
      GemmIdentityThreadblockSwizzle<>,               // Swizzling Operator
      6                                               // Stages
      >::GemmKernel;

  using EpilogueOutputOp = typename GemmKernel::Epilogue::OutputOp;
  typename EpilogueOutputOp::Params epilogue_op(1.0, 0.0);
  // alpha: 1.0, beta: 0.0
  // cutlass gemm operation: ptr_Y = alpha * ptr_X * ptr_W + beta * ptr_Y
  // ptr_Y[i] is not set to zero so beta must be 0.0

  using GemmGrouped = cutlass::gemm::device::GemmGrouped<GemmKernel>;
  typename GemmGrouped::Arguments args(all_problems, num_problems, 512,
                                       epilogue_op, ptr_X, ptr_W, ptr_Y, ptr_Y,
                                       ld_X, ld_W, ld_Y, ld_Y);

  GemmGrouped gemm;

  if (CC >= 80) {
    auto status = gemm.initialize(args, nullptr, stream);
    if (status != cutlass::Status::kSuccess) {
      throw AsException("SgmvCutlass gemm.initialize failed: " +
                        std::string(cutlassGetStatusString(status)) + "\n");
    }
    status = gemm.run(stream);
    if (status != cutlass::Status::kSuccess) {
      throw AsException("SgmvCutlass gemm.run failed: " +
                        std::string(cutlassGetStatusString(status)) + "\n");
    }
  } else {
    throw AsException("Compute capability not supported for SgmvCutlass");
  }
  return true;
}

template bool SgmvCutlass<half>(half* y, const half* x, const half** w,
                                const int32_t* s, const int32_t* ranks,
                                void* tmp_d, bool is_k_tensor, bool is_n_tensor,
                                int num_problems, int d_in, int d_out,
                                bool unsplit, int unsplit_n, int max_rank,
                                int CC, cudaStream_t stream);

template bool SgmvCutlass<hie::bfloat16>(
    hie::bfloat16* y, const hie::bfloat16* x, const hie::bfloat16** w,
    const int32_t* s, const int32_t* ranks, void* tmp_d, bool is_k_tensor,
    bool is_n_tensor, int num_problems, int d_in, int d_out, bool unsplit,
    int unsplit_n, int max_rank, int CC, cudaStream_t stream);

template <typename T>
__global__ void do_split_qkv(T** out_ptrs, T* in, const int32_t* s,
                             const int32_t* lora_B_ranks, int max_rank) {
  int problem_idx = blockIdx.x / 3;
  int qkv_idx = blockIdx.x % 3;
  int rank = lora_B_ranks[problem_idx];
  T* out = out_ptrs[qkv_idx] + (max_rank / 3) * s[problem_idx * 2];
  T* in_qkv = in + max_rank * s[problem_idx * 2] + qkv_idx * rank;
  int batch = s[problem_idx * 2 + 1] - s[problem_idx * 2];

  for (int i = 0; i < batch; i++) {
    for (int j = 0; j < rank; j++) {
      *out = *in_qkv;
      out++;
      in_qkv++;
    }
    in_qkv += 2 * rank;  // skip to next q/k/v
  }
}

template <typename DType>
bool SgmvSplitQKV(DType** out_ptrs, const DType* in, const int32_t* s,
                  const int32_t* lora_B_ranks, int max_rank, int num_problems,
                  cudaStream_t stream) {
  using cutlass_t = typename cutlass_dtype<DType>::type;
  do_split_qkv<<<num_problems * 3, 1, 0, stream>>>(
      (cutlass_t**)out_ptrs, (cutlass_t*)in, s, lora_B_ranks, max_rank);
  return true;
}

template bool SgmvSplitQKV<float>(float** out_ptrs, const float* in,
                                  const int32_t* s, const int32_t* lora_B_ranks,
                                  int max_rank, int num_problems,
                                  cudaStream_t stream);

template bool SgmvSplitQKV<half>(half** out_ptrs, const half* in,
                                 const int32_t* s, const int32_t* lora_B_ranks,
                                 int max_rank, int num_problems,
                                 cudaStream_t stream);

template bool SgmvSplitQKV<hie::bfloat16>(hie::bfloat16** out_ptrs,
                                          const hie::bfloat16* in,
                                          const int32_t* s,
                                          const int32_t* lora_B_ranks,
                                          int max_rank, int num_problems,
                                          cudaStream_t stream);

}  // namespace cuda
}  // namespace allspark
