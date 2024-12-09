/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    xformer_mha.A.fp16.cu
 */

#ifdef ENABLE_CUDA
#ifndef __HGGCCC__
#include "xformer_kernel.cuh"

namespace allspark {
namespace cuda {

#ifdef ENABLE_FP16
XFORMER_PREFILL_ATTENTION_IMPL_A(cutlass::half_t, cutlass::arch::Sm70);
XFORMER_PREFILL_ATTENTION_IMPL_A(cutlass::half_t, cutlass::arch::Sm75);
XFORMER_PREFILL_ATTENTION_IMPL_A(cutlass::half_t, cutlass::arch::Sm80);
#endif  // ENABLE_FP16

}  // namespace cuda
}  // namespace allspark

#endif  // __HGGCCC__
#endif  // ENABLE_CUDA
