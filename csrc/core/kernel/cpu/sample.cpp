/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    sample.cpp
 */
#include <cfloat>
#include <random>

#include "cpu_common.h"  // NOLINT
#include "cpu_kernel.h"
namespace allspark {
namespace cpu {

void SampleKernelInitLauncher(void* states, unsigned long long seed,
                              int batch_size) {
  assert(batch_size == 1);
  for (int i = 0; i < batch_size; i++) {
    ((std::mt19937*)states)[i] = std::mt19937(seed);
  }
}

template <typename T>
T exponential_func(T val) {
  const float lambda = 1.0;
  auto f_val = static_cast<float>(val);
  return static_cast<T>(-1.0) / lambda * std::log1p(-f_val);
}

template <typename T>
T multinomial_distri(T in, std::mt19937& rng) {
  std::uniform_real_distribution<T> dist(0.0, 1.0);
  T rand = dist(rng);

  float q = exponential_func(rand);               // q ~ Exp(1), float
  T logits_prob = static_cast<T>(float(in) / q);  //  ~ Multinomial
  return logits_prob;
}

template <typename T>
void sample_kernel(int64_t* out, T* in, const int64_t* indice,
                   std::mt19937& state, int num) {
  for (int i = 0; i < num; i++) {
    in[i] = multinomial_distri(in[i], state);
  }

  T max = in[0];
  out[0] = indice[0];
  for (int i = 1; i < num; i++) {
    if (max < in[i]) {
      max = in[i];
      out[0] = indice[i];
    }
  }
}

template <typename T>
void SampleKernel(int64_t* out, void* states, T* in, const int64_t* indice,
                  int batch_size, int* num_arr, int stride) {
  assert(batch_size == 1);
  parallel_for(batch_size, [&](int b) {
    sample_kernel(out + b, in + b * stride, indice + b * stride,
                  ((std::mt19937*)states)[b], num_arr[b]);
  });
}
template void SampleKernel<float>(int64_t* out, void* states, float* in,
                                  const int64_t* indice, int batch_size,
                                  int* num_arr, int stride);

}  // namespace cpu
}  // namespace allspark
