/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    relativePE.cpp
 */

#include <cmath>

#include "cpu_common.h"
#include "cpu_kernel.h"
namespace allspark {
namespace cpu {
const float e = 2.7182818284;
template <>
void RelativePEKernel<float>(float* out, const float* attention_bias,
                             int batch_size, int seq_length, int k, int step,
                             bool is_decoder) {
  int N = batch_size * k;
  if (!is_decoder) {
    parallel_for(N, [&](int idx) {
      int batch = idx / k;
      int head = idx % k;
      for (int i = 0; i < seq_length; i++) {
        for (int j = 0; j < seq_length; j++) {
          int result0 = std::min(1, std::max(0, j - i)) * 16;
          int result1 = std::abs(j - i);
          if (result1 >= 8) {
            result1 =
                std::min(int(std::log((float)result1 / 8) / e * 8 + 8), 15);
          }
          // batch,seq,num,seq
          out[batch * k * seq_length * seq_length + i * k * seq_length +
              head * seq_length + j] =
              attention_bias[(result0 + result1) * k + head];
        }
      }
    });
  } else {
    parallel_for(N, [&](int idx) {
      int batch = idx / k;
      int head = idx % k;
      int i = step - 1;
      for (int j = 0; j < step; j++) {
        int result1 = i - j;
        if (result1 >= 16) {
          result1 = std::min(
              int(std::log((float)result1 / 16) / std::log(8) * 16 + 16), 31);
        }
        out[batch * 1 * k * step + head * step + j] =
            attention_bias[(result1)*k + head];
      }
    });
  }
}
}  // namespace cpu
}  // namespace allspark