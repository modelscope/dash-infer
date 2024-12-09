/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    layernorm.cu
 */

#include "cuda_common.h"  // NOLINT
#include "cuda_kernel.h"
#include "reduce.cuh"  // NOLINT

namespace allspark {
namespace cuda {

template <bool ADDBIAS, int ite, typename T>
__global__ static void layernorm_kernel(T* out, const T* input, const T* bias,
                                        const T* gamma, const T* beta, int n,
                                        float eps) {
  int tid = threadIdx.x;
  __shared__ float s_mean;
  __shared__ float s_variance;

  float local_out[ite];
  float mean = 0.f;
  float mean_2 = 0.f;
#pragma unroll ite
  for (int i = 0; i < ite; ++i) {
    int idx = blockIdx.x * n + i * blockDim.x + tid;
    if (i * blockDim.x + tid >= n) {
      local_out[i] = 0;
    } else {
      if (ADDBIAS) {
        local_out[i] = (float)(__ldg(&input[idx]) + __ldg(&bias[idx]));
      } else {
        local_out[i] = (float)(__ldg(&input[idx]));
      }
    }
    mean += local_out[i];
    mean_2 += local_out[i] * local_out[i];
  }
  blockReduce<float, ReduceOp::kSUM>(&mean);
  __syncthreads();
  blockReduce<float, ReduceOp::kSUM>(&mean_2);
  if (tid == 0) {
    mean /= n;
    mean_2 /= n;
    s_mean = mean;
    s_variance = rsqrtf(mean_2 - mean * mean + (float)eps);
  }
  __syncthreads();
#pragma unroll ite
  for (int i = 0; i < ite; ++i) {
    int col_id = i * blockDim.x + tid;
    if (col_id >= n) {
      continue;
    }
    int idx = blockIdx.x * n + col_id;
    out[idx] = (float)((local_out[i] - s_mean) * s_variance *
                           (float)(__ldg(&gamma[col_id])) +
                       (float)(__ldg(&beta[col_id])));
  }
}
template <typename T>
__global__ void layernorm_kernel_base(int N, T* out, const T* input,
                                      const T* gamma, const T* beta, int n,
                                      float eps) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  double mean = 0;
  double mean_2 = 0;
  double variance = 0;
  if (tid < N) {
    for (int i = 0; i < n; i++) {
      float now = (float)(input[tid * n + i]);
      mean += now;
      mean_2 += now * now;
    }
    mean /= n;
    mean_2 /= n;
    variance = rsqrtf(mean_2 - mean * mean + (float)eps);
    for (int i = 0; i < n; i++) {
      float now = (float)(input[tid * n + i]);
      out[tid * n + i] =
          (float)(now - mean) * variance * (float)gamma[i] + (float)beta[i];
    }
  }
}
template <bool ADDBIAS, int ite, typename T>
__global__ static void layernorm_nobeta_kernel(T* out, const T* input,
                                               const T* bias, const T* gamma,
                                               int n, float eps) {
  int tid = threadIdx.x;
  __shared__ float s_variance;

  float local_out[ite];
  float mean_2 = 0.f;
#pragma unroll ite
  for (int i = 0; i < ite; ++i) {
    int idx = blockIdx.x * n + i * blockDim.x + tid;
    if (i * blockDim.x + tid >= n) {
      local_out[i] = 0;
    } else {
      if (ADDBIAS) {
        local_out[i] = (float)(__ldg(&input[idx]) + __ldg(&bias[idx]));
      } else {
        local_out[i] = (float)(__ldg(&input[idx]));
      }
    }

    mean_2 += local_out[i] * local_out[i];
  }
  __syncthreads();
  blockReduce<float, ReduceOp::kSUM>(&mean_2);
  if (tid == 0) {
    mean_2 /= n;
    s_variance = rsqrtf(mean_2 + (float)eps);
  }
  __syncthreads();
#pragma unroll ite
  for (int i = 0; i < ite; ++i) {
    int col_id = i * blockDim.x + tid;
    if (col_id >= n) {
      continue;
    }
    int idx = blockIdx.x * n + col_id;
    out[idx] =
        (float)(local_out[i]) * s_variance * (float)(__ldg(&gamma[col_id]));
  }
}
template <typename T>
void LayerNormKernelLauncher(T* out, const T* input, const T* bias,
                             const T* gamma, const T* beta, int m, int n,
                             float eps, cudaStream_t stream) {
  if (n / 8 <= 1024) {
    const int ite = 8;
    // const int thread_num = (n/ite + 32 -1)/32*32;
    const int thread_num = 1024;
    // assert((n / ite) <= 1024 && (n / ite) % 32 == 0);
    if (bias != nullptr) {
      layernorm_kernel<true, ite, T>
          <<<m, thread_num, 0, stream>>>(out, input, bias, gamma, beta, n, eps);
    } else {
      layernorm_kernel<false, ite, T>
          <<<m, thread_num, 0, stream>>>(out, input, bias, gamma, beta, n, eps);
    }
  } else if (n / 16 <= 1024) {
    const int ite = 16;
    // const int thread_num = (n/ite + 32 -1)/32*32;
    const int thread_num = 1024;
    // assert((n / ite) <= 1024 && (n / ite) % 32 == 0);
    if (bias != nullptr) {
      layernorm_kernel<true, ite, T>
          <<<m, thread_num, 0, stream>>>(out, input, bias, gamma, beta, n, eps);
    } else {
      layernorm_kernel<false, ite, T>
          <<<m, thread_num, 0, stream>>>(out, input, bias, gamma, beta, n, eps);
    }
  } else if (n / 32 <= 1024) {
    const int ite = 32;
    const int thread_num = (n / ite + 32 - 1) / 32 * 32;
    // assert((n / ite) <= 1024 && (n / ite) % 32 == 0);
    if (bias != nullptr) {
      layernorm_kernel<true, ite, T>
          <<<m, thread_num, 0, stream>>>(out, input, bias, gamma, beta, n, eps);
    } else {
      layernorm_kernel<false, ite, T>
          <<<m, thread_num, 0, stream>>>(out, input, bias, gamma, beta, n, eps);
    }
  } else if (n / 64 <= 1024) {
    const int ite = 64;
    const int thread_num = (n / ite + 32 - 1) / 32 * 32;
    // assert((n / ite) <= 1024 && (n / ite) % 32 == 0);
    if (bias != nullptr) {
      layernorm_kernel<true, ite, T>
          <<<m, thread_num, 0, stream>>>(out, input, bias, gamma, beta, n, eps);
    } else {
      layernorm_kernel<false, ite, T>
          <<<m, thread_num, 0, stream>>>(out, input, bias, gamma, beta, n, eps);
    }
  }
}
template void LayerNormKernelLauncher<float>(float* out, const float* input,
                                             const float* bias,
                                             const float* gamma,
                                             const float* beta, int m, int n,
                                             float eps, cudaStream_t stream);
#ifdef ENABLE_FP16
template void LayerNormKernelLauncher<half>(half* out, const half* input,
                                            const half* bias, const half* gamma,
                                            const half* beta, int m, int n,
                                            float eps, cudaStream_t stream);
#endif
template void LayerNormKernelLauncher<hie::bfloat16>(
    hie::bfloat16* out, const hie::bfloat16* input, const hie::bfloat16* bias,
    const hie::bfloat16* gamma, const hie::bfloat16* beta, int m, int n,
    float eps, cudaStream_t stream);
template <typename T>
void LayerNormNoBetaKernelLauncher(T* out, const T* input, const T* bias,
                                   const T* gamma, int m, int n, float eps,
                                   cudaStream_t stream) {
  if (n / 8 <= 1024) {
    const int ite = 8;
    // const int thread_num = (n/ite + 32 -1)/32*32;
    const int thread_num = 1024;
    // assert((n / ite) <= 1024 && (n / ite) % 32 == 0);
    if (bias != nullptr) {
      layernorm_nobeta_kernel<true, ite, T>
          <<<m, thread_num, 0, stream>>>(out, input, bias, gamma, n, eps);
    } else {
      layernorm_nobeta_kernel<false, ite, T>
          <<<m, thread_num, 0, stream>>>(out, input, bias, gamma, n, eps);
    }
  } else if (n / 16 <= 1024) {
    const int ite = 16;
    // const int thread_num = (n/ite + 32 -1)/32*32;
    const int thread_num = 1024;
    // assert((n / ite) <= 1024 && (n / ite) % 32 == 0);
    if (bias != nullptr) {
      layernorm_nobeta_kernel<true, ite, T>
          <<<m, thread_num, 0, stream>>>(out, input, bias, gamma, n, eps);
    } else {
      layernorm_nobeta_kernel<false, ite, T>
          <<<m, thread_num, 0, stream>>>(out, input, bias, gamma, n, eps);
    }
  } else if (n / 32 <= 1024) {
    const int ite = 32;
    const int thread_num = (n / ite + 32 - 1) / 32 * 32;
    // assert((n / ite) <= 1024 && (n / ite) % 32 == 0);
    if (bias != nullptr) {
      layernorm_nobeta_kernel<true, ite, T>
          <<<m, thread_num, 0, stream>>>(out, input, bias, gamma, n, eps);
    } else {
      layernorm_nobeta_kernel<false, ite, T>
          <<<m, thread_num, 0, stream>>>(out, input, bias, gamma, n, eps);
    }
  } else if (n / 64 <= 1024) {
    const int ite = 64;
    const int thread_num = (n / ite + 32 - 1) / 32 * 32;
    // assert((n / ite) <= 1024 && (n / ite) % 32 == 0);
    if (bias != nullptr) {
      layernorm_nobeta_kernel<true, ite, T>
          <<<m, thread_num, 0, stream>>>(out, input, bias, gamma, n, eps);
    } else {
      layernorm_nobeta_kernel<false, ite, T>
          <<<m, thread_num, 0, stream>>>(out, input, bias, gamma, n, eps);
    }
  }
}
template void LayerNormNoBetaKernelLauncher<float>(
    float* out, const float* input, const float* bias, const float* gamma,
    int m, int n, float eps, cudaStream_t stream);
#ifdef ENABLE_FP16
template void LayerNormNoBetaKernelLauncher<half>(half* out, const half* input,
                                                  const half* bias,
                                                  const half* gamma, int m,
                                                  int n, float eps,
                                                  cudaStream_t stream);
#endif
template void LayerNormNoBetaKernelLauncher<hie::bfloat16>(
    hie::bfloat16* out, const hie::bfloat16* input, const hie::bfloat16* bias,
    const hie::bfloat16* gamma, int m, int n, float eps, cudaStream_t stream);
}  // namespace cuda
}  // namespace allspark
