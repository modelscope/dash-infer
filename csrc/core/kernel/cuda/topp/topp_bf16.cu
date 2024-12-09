/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    topp_bf16.cu
 */

#include "topp.cuh"

#if ENABLE_BF16
#include <cuda_runtime.h>
#if (__CUDACC_VER_MAJOR__ >= 11 || CUDA_VERSION >= 11000)
#include <cub/cub.cuh>
namespace cub {
template <>
struct NumericTraits<hie::bfloat16>
    : BaseTraits<cub::FLOATING_POINT, true, false, unsigned short,
                 __nv_bfloat16> {};
}  // namespace cub
#else
static_assert(false,
              "CUB requires support for native __nv_bfloat16, "
              "please check your CUDA version");
#endif  // (__CUDACC_VER_MAJOR__ >= 11 || CUDA_VERSION >= 11000)
#endif  // ENABLE_BF16

namespace allspark {
namespace cuda {

#ifdef ENABLE_BF16
template void TopPSoftmaxGetWorkspaceSize<hie::bfloat16>(size_t* sizeInBytes,
                                                         int batch_size,
                                                         int length,
                                                         bool input_is_sorted);

template void TopPSoftmaxLauncher<hie::bfloat16>(
    int* topp_count, hie::bfloat16* topp_probs, int* topp_indices,
    const hie::bfloat16* input_logits, const float* p_values,
    const float* temperatures, hie::bfloat16* temp_probs, void* workspace,
    size_t ws_size_in_bytes, int batch_size, int length, bool input_is_sorted,
    hiednnCudaHandle_t handle, cudaStream_t stream);
#endif

}  // namespace cuda
}  // namespace allspark
