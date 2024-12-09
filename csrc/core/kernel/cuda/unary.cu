/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    unary.cu
 */

#include <cmath>

#include "allspark.pb.h"
#include "cuda_common.h"  // NOLINT
#include "cuda_kernel.h"
#include "elementwise.cuh"

namespace allspark {
namespace cuda {

template <typename T>
struct TanhFunctor {
  __device__ __host__ __forceinline__ T operator()(T x) const {
    return tanhf(x);
  }
};

template <typename T>
struct GeluFunctor {
  __device__ __host__ __forceinline__ T operator()(T x) const {
    T cdf = 0.5f * (1.0f + erff(x * (T)0.70710678f));
    return x * cdf;
  }
};

template <typename T>
struct GeluTanFunctor {
  __device__ __host__ __forceinline__ T operator()(T x) const {
    T cdf = 0.5f * (1.0f + tanhf(((T)0.7978845608028654f *
                                  (T)(x + (T)0.044715f * x * x * x))));
    return x * cdf;
  }
};
template <typename T>
struct ReluFunctor {
  __device__ __host__ __forceinline__ T operator()(T x) const {
    return std::max(x, (T)0);
  }
};
template <typename T>
struct SiluFunctor {
  __device__ __host__ __forceinline__ T operator()(T x) const {
    return x * (1.0f / (1.0f + expf((T)(-x))));
  }
};
template <typename T>
struct SigmoidFunctor {
  __device__ __host__ __forceinline__ T operator()(T x) const {
    return (1.0f / (1.0f + expf((T)(-x))));
  }
};
/* To be deleted:
template <typename T>
__global__ void geglu_kernel(int N, T* out, const T* in) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        T x = in[tid+N];
        T cdf = 0.5f * (1.0f + tanhf(((T)0.7978845608028654f *
                                      (T)(x + (T)0.044715f * x * x * x))));
        x = x * cdf;
        out[tid] = in[tid] * x;
    }
}
*/

template <typename T>
void UnaryKernelLauncher(T* out, const T* in, int64_t count, int type,
                         cudaStream_t stream) {
  switch (type) {
    case UnaryType::TANH:
      elementwise::Unary(TanhFunctor<T>(), count, out, in, stream);
      break;
    case UnaryType::GELU_ERF:
      elementwise::Unary(GeluFunctor<T>(), count, out, in, stream);
      break;
    case UnaryType::GELU_TANH:
      elementwise::Unary(GeluTanFunctor<T>(), count, out, in, stream);
      break;
    case UnaryType::RELU:
      elementwise::Unary(ReluFunctor<T>(), count, out, in, stream);
      break;
    case UnaryType::SILU:
      elementwise::Unary(SiluFunctor<T>(), count, out, in, stream);
      break;
    case UnaryType::SIGMOID:
      elementwise::Unary(SigmoidFunctor<T>(), count, out, in, stream);
      break;
    /* To be deleted:
    case 5:
        {
            int N = count / 2;
            const int block_num = (N + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
            geglu_kernel<<<block_num, THREAD_PER_BLOCK, 0, stream>>>(
                N, out ,in);
        break;
        }
    */
    default:
      return;
  }
}

template void UnaryKernelLauncher<float>(float* out, const float* in,
                                         int64_t count, int type,
                                         cudaStream_t stream);

#ifdef ENABLE_FP16
template void UnaryKernelLauncher<half>(half* out, const half* in,
                                        int64_t count, int type,
                                        cudaStream_t stream);
#endif
template void UnaryKernelLauncher<hie::bfloat16>(hie::bfloat16* out,
                                                 const hie::bfloat16* in,
                                                 int64_t count, int type,
                                                 cudaStream_t stream);

template <typename FactoryT, typename T>
__global__ void unary_glu_kernel(size_t N, FactoryT f, int outer_size,
                                 int inner_size, T* out, const T* in) {
  size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < N) {
    size_t row = tid / inner_size;
    size_t col = tid % inner_size;
    out[row * inner_size + col] = f(in[row * inner_size * 2 + col]) *
                                  in[row * inner_size * 2 + inner_size + col];
  }
}
template <typename T>
void UnaryGLUKernelLauncher(T* out, const T* in, size_t outer_size,
                            size_t inner_size, int type, cudaStream_t stream) {
  size_t N = (size_t)outer_size * inner_size;
  const size_t block_num = (N + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
  switch (type) {
    case UnaryType::TANH:
      unary_glu_kernel<<<block_num, THREAD_PER_BLOCK, 0, stream>>>(
          N, TanhFunctor<T>(), outer_size, inner_size, out, in);
      break;
    case UnaryType::GELU_ERF:
      unary_glu_kernel<<<block_num, THREAD_PER_BLOCK, 0, stream>>>(
          N, GeluFunctor<T>(), outer_size, inner_size, out, in);
      break;
    case UnaryType::GELU_TANH:
      unary_glu_kernel<<<block_num, THREAD_PER_BLOCK, 0, stream>>>(
          N, GeluTanFunctor<T>(), outer_size, inner_size, out, in);
      break;
    case UnaryType::RELU:
      unary_glu_kernel<<<block_num, THREAD_PER_BLOCK, 0, stream>>>(
          N, ReluFunctor<T>(), outer_size, inner_size, out, in);
      break;
    case UnaryType::SILU:
      unary_glu_kernel<<<block_num, THREAD_PER_BLOCK, 0, stream>>>(
          N, SiluFunctor<T>(), outer_size, inner_size, out, in);
      break;
    case UnaryType::SIGMOID:
      unary_glu_kernel<<<block_num, THREAD_PER_BLOCK, 0, stream>>>(
          N, SigmoidFunctor<T>(), outer_size, inner_size, out, in);
      break;
    /* To be deleted:
    case 5:
        {
            int N = count / 2;
            const int block_num = (N + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
            geglu_kernel<<<block_num, THREAD_PER_BLOCK, 0, stream>>>(
                N, out ,in);
        break;
        }
    */
    default:
      return;
  }
}
template void UnaryGLUKernelLauncher<float>(float* out, const float* in,
                                            size_t outer_size,
                                            size_t inner_size, int type,
                                            cudaStream_t stream);

#ifdef ENABLE_FP16
template void UnaryGLUKernelLauncher<half>(half* out, const half* in,
                                           size_t outer_size, size_t inner_size,
                                           int type, cudaStream_t stream);
#endif
template void UnaryGLUKernelLauncher<hie::bfloat16>(hie::bfloat16* out,
                                                    const hie::bfloat16* in,
                                                    size_t outer_size,
                                                    size_t inner_size, int type,
                                                    cudaStream_t stream);

}  // namespace cuda
}  // namespace allspark
