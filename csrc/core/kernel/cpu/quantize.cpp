/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    quantize.cpp
 */

#include "cpu_common.h"
#include "cpu_kernel.h"
#ifdef ENABLE_FP16
#include <common/float16.h>
#endif
#include <math.h>
namespace allspark {
namespace cpu {

template <typename T>
void quantize(int m, int n, const T* data, int8_t* q_data, float* scale,
              int8_t* zero, int* redsum) {
  parallel_for(n, [&](int j) {
    float qmax = 127;
    float qmin = -128;
    int redsum_ = 0;
    T fmax = data[j];
    T fmin = data[j];
    for (int i = 0; i < m; i++) {
      fmax = std::max(fmax, data[i * n + j]);
      fmin = std::min(fmin, data[i * n + j]);
    }
    float scale_ = (float)(fmax - fmin) / (qmax - qmin);
    float init_zero = qmin - fmin / scale_;
    init_zero = std::max(init_zero, qmin);
    init_zero = std::min(init_zero, qmax);
    zero[j] = (int8_t)(round(init_zero));
    parallel_for(m, [&](int i) {
      float q_data_ = (float)data[i * n + j] / scale_ + init_zero;
      q_data_ = std::max(q_data_, qmin);
      q_data_ = std::min(q_data_, qmax);
      q_data[i * n + j] = (int8_t)(round(q_data_));
      redsum_ += (int)q_data[i * n + j];
    });
    scale[j] = (float)scale_;
    redsum[j] = (int)redsum_;
  });
}
template void quantize<float>(int m, int n, const float* data, int8_t* q_data,
                              float* scale, int8_t* zero, int* redsum);
#ifdef ENABLE_FP16
template void quantize<half>(int m, int n, const half* data, int8_t* q_data,
                             float* scale, int8_t* zero, int* redsum);
#endif
}  // namespace cpu
}  // namespace allspark
