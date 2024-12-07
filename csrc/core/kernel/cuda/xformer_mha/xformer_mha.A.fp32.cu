/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    xformer_mha.A.fp32.cu
 */

#ifdef ENABLE_CUDA
#ifndef __HGGCCC__
#include "xformer_kernel.cuh"

namespace allspark {
namespace cuda {

XFORMER_PREFILL_ATTENTION_IMPL_A(float, cutlass::arch::Sm70);
XFORMER_PREFILL_ATTENTION_IMPL_A(float, cutlass::arch::Sm75);
XFORMER_PREFILL_ATTENTION_IMPL_A(float, cutlass::arch::Sm80);

}  // namespace cuda
}  // namespace allspark

#endif  // __HGGCCC__
#endif  // ENABLE_CUDA
