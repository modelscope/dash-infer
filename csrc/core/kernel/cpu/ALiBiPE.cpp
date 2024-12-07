/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    ALiBiPE.cpp
 */

#include <cmath>

#include "cpu_common.h"
#include "cpu_kernel.h"
#ifdef ENABLE_FP16
#include <common/float16.h>
#endif
namespace allspark {
namespace cpu {
static float get_ALiBiPE_slope(int head, int num_heads, int ori_num_heads,
                               int rank) {
  float closest_power_of_2 = std::pow(2, floor(log2(ori_num_heads)));
  float pos = rank * num_heads + head;

  float power, slope;
  if (pos < closest_power_of_2)
    power = closest_power_of_2;
  else
    power = closest_power_of_2 * 2;
  float base = std::pow(2, (-(std::pow(2, -(log2f(power) - 3)))));

  if (pos < closest_power_of_2)
    slope = std::pow(base, pos + 1);
  else
    slope = std::pow(base, (pos - closest_power_of_2) * 2 + 1);

  return slope;
}

template <typename T>
void ALiBiPE_kernel(T* out, int* batch_offset, int batch_size, int seq_length,
                    int num_heads, int ori_num_heads, int rank, int N) {
  // return [batch,seq_length,num_heads,seq_length]
  parallel_for(N, [&](int idx) {
    int batch = idx / num_heads;
    int head = idx % num_heads;
    int offset = batch_offset[batch];
    float slope = get_ALiBiPE_slope(head, num_heads, ori_num_heads, rank);
    for (int i = 0; i < seq_length; i++) {
      for (int j = 0; j < seq_length; j++) {
        out[batch * num_heads * seq_length * seq_length +
            i * num_heads * seq_length + head * seq_length + j] =
            slope * (j - offset);
      }
    }
  });
}
template <typename T>
void ALiBiPE_decoder_kernel(T* out, int* batch_offset, int batch_size,
                            int seq_length, int num_heads, int ori_num_heads,
                            int rank, int N) {
  // return [batch,1,num_heads,seq_length],i=seq_length-1
  parallel_for(N, [&](int idx) {
    int batch = idx / num_heads;
    int head = idx % num_heads;
    int offset = batch_offset[batch];
    float slope = get_ALiBiPE_slope(head, num_heads, ori_num_heads, rank);
    for (int j = 0; j < seq_length; j++) {
      out[batch * num_heads * 1 * seq_length + head * seq_length + j] =
          slope * (j - offset);
    }
  });
}
template <typename T>
void ALiBiPEKernelLauncher(T* out, int* batch_offset, int batch_size,
                           int seq_length, int num_heads, int ori_num_heads,
                           int step, int rank) {
  int N = batch_size * num_heads;
  if (step - 1 == 0) {
    ALiBiPE_kernel(out, batch_offset, batch_size, seq_length, num_heads,
                   ori_num_heads, rank, N);
  } else {
    ALiBiPE_decoder_kernel(out, batch_offset, batch_size, step, num_heads,
                           ori_num_heads, rank, N);
  }
}

template void ALiBiPEKernelLauncher<float>(float* out, int* batch_offset,
                                           int batch_size, int seq_length,
                                           int num_heads, int ori_num_heads,
                                           int step, int rank);
#ifdef ENABLE_FP16
template void ALiBiPEKernelLauncher<half>(half* out, int* batch_offset,
                                          int batch_size, int seq_length,
                                          int num_heads, int ori_num_heads,
                                          int step, int rank);
#endif
}  // namespace cpu
}  // namespace allspark
