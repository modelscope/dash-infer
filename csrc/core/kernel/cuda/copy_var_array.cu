/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    copy_var_array.cu
 */

#include <cmath>

#include "cuda_common.h"
#include "cuda_kernel.h"

namespace allspark {
namespace cuda {

template <typename T>
__global__ void CopyToArray(T* array, const T** vars, int count) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    array[index] = *vars[index];
  }
}

template <typename T>
void CopyToArrayKernelLauncher(T* array, const T** vars, int count,
                               cudaStream_t stream) {
  int N = count;
  const int block_num = (N + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
  CopyToArray<<<block_num, THREAD_PER_BLOCK, 0, stream>>>(array, vars, count);
}

template void CopyToArrayKernelLauncher<int>(int* array, const int** vars,
                                             int count, cudaStream_t stream);

template void CopyToArrayKernelLauncher<int64_t>(int64_t* array,
                                                 const int64_t** vars,
                                                 int count,
                                                 cudaStream_t stream);

template void CopyToArrayKernelLauncher<float>(float* array, const float** vars,
                                               int count, cudaStream_t stream);

#ifdef ENABLE_FP16
template void CopyToArrayKernelLauncher<half>(half* array, const half** vars,
                                              int count, cudaStream_t stream);
#endif

template void CopyToArrayKernelLauncher<hie::bfloat16>(
    hie::bfloat16* array, const hie::bfloat16** vars, int count,
    cudaStream_t stream);

//

template <typename T>
__global__ void CopyToVars(T** vars, const T* array, int count) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    *vars[index] = array[index];
  }
}

template <typename T>
void CopyToVarsKernelLauncher(T** vars, const T* array, int count,
                              cudaStream_t stream) {
  int N = count;
  const int block_num = (N + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
  CopyToVars<<<block_num, THREAD_PER_BLOCK, 0, stream>>>(vars, array, count);
}

template void CopyToVarsKernelLauncher<int>(int** vars, const int* array,
                                            int count, cudaStream_t stream);

template void CopyToVarsKernelLauncher<int64_t>(int64_t** vars,
                                                const int64_t* array, int count,
                                                cudaStream_t stream);

template void CopyToVarsKernelLauncher<float>(float** vars, const float* array,
                                              int count, cudaStream_t stream);

#ifdef ENABLE_FP16
template void CopyToVarsKernelLauncher<half>(half** vars, const half* array,
                                             int count, cudaStream_t stream);
#endif

template void CopyToVarsKernelLauncher<hie::bfloat16>(
    hie::bfloat16** vars, const hie::bfloat16* array, int count,
    cudaStream_t stream);

}  // namespace cuda
}  // namespace allspark
