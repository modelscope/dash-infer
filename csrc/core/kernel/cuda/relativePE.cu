/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    relativePE.cu
 */

#include "cmath"
#include "cuda_common.h"  // NOLINT
#include "cuda_kernel.h"
namespace allspark {
namespace cuda {
const float e = 2.7182818284;
template <typename T>
__global__ void relativePE_kernel(T* out, const T* attention_bias,
                                  int batch_size, int seq_length, int k,
                                  int N) {
  // return [batch,seq_length,k,seq_length]
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    int batch = idx / k;
    int head = idx % k;
    for (int i = 0; i < seq_length; i++) {
      for (int j = 0; j < seq_length; j++) {
        int result0 = std::min(1, std::max(0, j - i)) * 16;
        int result1 = std::abs(j - i);
        if (result1 >= 8) {
          result1 = std::min(int(std::log((float)result1 / 8) / e * 8 + 8), 15);
        }
        out[batch * k * seq_length * seq_length + i * k * seq_length +
            head * seq_length + j] =
            attention_bias[(result0 + result1) * k + head];
      }
    }
  }
}
template <typename T>
__global__ void relativePE_decoder_kernel(T* out, const T* attention_bias,
                                          int batch_size, int seq_length, int k,
                                          int N) {
  // return [batch,1,k,seq_length],i=seq_length-1
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    int batch = idx / k;
    int head = idx % k;
    int i = seq_length - 1;
    for (int j = 0; j < seq_length; j++) {
      int result1 = i - j;
      if (result1 >= 16) {
        result1 = std::min(
            int(std::log((float)result1 / 16) / std::log(8) * 16 + 16), 31);
      }
      out[batch * 1 * k * seq_length + head * seq_length + j] =
          attention_bias[(result1)*k + head];
    }
  }
}
template <typename T>
void RelativePEKernelLauncher(T* out, const T* attention_bias, int batch_size,
                              int seq_length, int k, int step, bool is_decoder,
                              cudaStream_t stream) {
  int N = batch_size * k;
  const int block_num = (N + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
  if (!is_decoder) {
    relativePE_kernel<<<block_num, THREAD_PER_BLOCK, 0, stream>>>(
        out, attention_bias, batch_size, seq_length, k, N);
  } else {
    relativePE_decoder_kernel<<<block_num, THREAD_PER_BLOCK, 0, stream>>>(
        out, attention_bias, batch_size, step, k, N);
  }
}

template void RelativePEKernelLauncher<float>(float* out,
                                              const float* attention_bias,
                                              int batch_size, int seq_length,
                                              int k, int step, bool is_decoder,
                                              cudaStream_t stream);
#ifdef ENABLE_FP16
template void RelativePEKernelLauncher<half>(half* out,
                                             const half* attention_bias,
                                             int batch_size, int seq_length,
                                             int k, int step, bool is_decoder,
                                             cudaStream_t stream);
#endif
template void RelativePEKernelLauncher<hie::bfloat16>(
    hie::bfloat16* out, const hie::bfloat16* attention_bias, int batch_size,
    int seq_length, int k, int step, bool is_decoder, cudaStream_t stream);
}  // namespace cuda
}  // namespace allspark