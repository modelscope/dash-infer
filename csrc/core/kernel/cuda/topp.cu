/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    topp.cu
 */

#include "cuda_common.h"  // NOLINT
#include "cuda_kernel.h"
namespace allspark {
namespace cuda {
template <typename T>
__global__ void topp_kernel(int N, T* input, int* k_arr, float* p_arr,
                            int length) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < N) {
    if (p_arr[tid] > 1e-7) {
      T* input_now = input + tid * length;
      float sum = 0;
      int last = k_arr[tid];
      for (int i = 0; i < k_arr[tid]; i++) {
        sum += (float)input_now[i];
        if (sum > p_arr[tid]) {
          last = i + 1;
          break;
        }
      }
      k_arr[tid] = last;
    }
  }
}
template <typename T>
void CalcTopP(const T* input, int* k_arr, float* p_arr, int batch, int length,
              cudaStream_t stream) {
  int N = batch;
  const int block_num = (N + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
  topp_kernel<<<block_num, THREAD_PER_BLOCK, 0, stream>>>(N, input, k_arr,
                                                          p_arr, length);
}

template void CalcTopP<float>(const float* input, int* k_arr, float* p_arr,
                              int batch, int length, cudaStream_t cu_stream);
#ifdef ENABLE_FP16
template void CalcTopP<half>(const half* input, int* k_arr, float* p_arr,
                             int batch, int length, cudaStream_t cu_stream);
#endif
template void CalcTopP<hie::bfloat16>(const hie::bfloat16* input, int* k_arr,
                                      float* p_arr, int batch, int length,
                                      cudaStream_t cu_stream);
}  // namespace cuda
}  // namespace allspark
