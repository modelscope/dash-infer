/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    xformer_mha.B.bf16.cu
 */

#ifdef ENABLE_CUDA
#ifndef __HGGCCC__
#include "xformer_kernel.cuh"

namespace allspark {
namespace cuda {

#ifdef ENABLE_BF16
/* no bf16 support for v100 (cutlass::bfloat16_t, cutlass::arch::Sm70); */
// XFORMER_PREFILL_ATTENTION_IMPL_B(cutlass::bfloat16_t, cutlass::arch::Sm70);
XFORMER_PREFILL_ATTENTION_IMPL_B(cutlass::bfloat16_t, cutlass::arch::Sm75);
XFORMER_PREFILL_ATTENTION_IMPL_B(cutlass::bfloat16_t, cutlass::arch::Sm80);
#endif  // ENABLE_BF16

}  // namespace cuda
}  // namespace allspark

#endif  // __HGGCCC__
#endif  // ENABLE_CUDA
