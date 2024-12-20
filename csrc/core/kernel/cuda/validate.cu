/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    validate.cu
 */
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>

#include <cmath>
#include <stdexcept>

#include "cuda_common.h"  // NOLINT

namespace allspark {
namespace cuda_util {

// 模板函数 ToFloat
template <typename T>
__device__ float ToFloat(T x);

// 特化 __half
template <>
__device__ float ToFloat<__half>(__half x) {
  return __half2float(x);
}

// 特化 __nv_bfloat16
#ifdef ENABLE_BF16
template <>
__device__ float ToFloat<__nv_bfloat16>(__nv_bfloat16 x) {
  return __bfloat162float(x);
}
#endif

// 检查 inf 和 nan 的函数
template <typename T>
void CheckInfNan(const T* d_data, size_t size_in_bytes) {
  size_t size = size_in_bytes / 2;
  size_t batch_size = 1024 * 1024;
  size_t num_batches = (size + batch_size - 1) / batch_size;
  for (size_t i = 0; i < num_batches; ++i) {
    size_t start = i * batch_size;
    size_t end = std::min(start + batch_size, size);
    // 分批处理数据
    thrust::device_vector<T> d_vec(d_data + start, d_data + end);
    // 使用 thrust::transform 和 thrust::reduce 来检查 inf 和 nan
    thrust::device_vector<bool> has_inf(end - start);
    thrust::device_vector<bool> has_nan(end - start);
    thrust::transform(d_vec.begin(), d_vec.end(), has_inf.begin(),
                      [] __device__(T x) {
                        float f = ToFloat<T>(x);
                        return isinf(f);
                      });
    thrust::transform(d_vec.begin(), d_vec.end(), has_nan.begin(),
                      [] __device__(T x) {
                        float f = ToFloat<T>(x);
                        return isnan(f);
                      });
    bool batch_has_inf = thrust::reduce(has_inf.begin(), has_inf.end(), false,
                                        thrust::logical_or<bool>());
    bool batch_has_nan = thrust::reduce(has_nan.begin(), has_nan.end(), false,
                                        thrust::logical_or<bool>());
    if (batch_has_inf) throw std::domain_error("INF number exists!");
    if (batch_has_nan) throw std::domain_error("NAN exists!");
  }
}

// 显式实例化
template void CheckInfNan<__half>(const __half* d_data, size_t size_in_bytes);

#ifdef ENABLE_BF16
template void CheckInfNan<__nv_bfloat16>(const __nv_bfloat16* d_data,
                                         size_t size_in_bytes);
#endif

}  // namespace cuda_util
}  // namespace allspark
