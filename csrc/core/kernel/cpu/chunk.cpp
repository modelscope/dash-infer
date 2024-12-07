/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    chunk.cpp
 */

#include <math.h>

#include "allspark.pb.h"
#include "cpu_common.h"
#include "cpu_kernel.h"

namespace allspark {
namespace cpu {

template <typename T>
void ChunkKernelLauncher(T* out, T* in, int batch, int seq_len, int hidden_size,
                         int chunk_split) {
  // OPT TBD
  int N = batch * seq_len * hidden_size;
  parallel_for(N, [&](int tid) {
    if (tid < N) {
      int bs = tid / hidden_size;
      int hid = tid % hidden_size;
      int split_hid = hidden_size / chunk_split;
      int split_part = hid / split_hid;
      int now = hid % split_hid;
      if (now < split_hid / 2) {
        T x = in[bs * hidden_size + hid + split_hid / 2];
        T cdf = 0.5f * (1.0f + tanhf(((T)0.7978845608028654f *
                                      (T)(x + (T)0.044715f * x * x * x))));
        x = x * cdf;
        out[bs * hidden_size / 2 + split_part * split_hid / 2 + now] =
            in[bs * hidden_size + hid] * x;
      }
    }
  });
}
template void ChunkKernelLauncher<float>(float* output, float* input, int batch,
                                         int seq_len, int hidden_size,
                                         int chunk_split);
template <typename T>
void ChunkBinary(T* out, T* in, int batch, int seq_len, int hidden_size,
                 int chunk_split, int type) {
  int N = batch * seq_len * hidden_size;
  switch (type) {
    case BinaryType::SWIGLU:
      parallel_for(N, [&](int tid) {
        int bs = tid / hidden_size;
        int hid = tid % hidden_size;
        int split_hid = hidden_size / chunk_split;
        int split_part = hid / split_hid;
        int now = hid % split_hid;
        if (now < split_hid / 2) {
          T x = in[bs * hidden_size + hid];
          x = x * (1.0f / (1.0f + expf((T)(-x))));
          out[bs * hidden_size / 2 + split_part * split_hid / 2 + now] =
              x * in[bs * hidden_size + hid + split_hid / 2];
        }
      });
      break;
    case BinaryType::GEGLU:
      // elementwise::Binary(GeGLUFunctor<T>(), count, out, in1, in2,
      // stream);
      break;

    default:
      return;
  }
}
template void ChunkBinary<float>(float* output, float* input, int batch,
                                 int seq_len, int hidden_size, int chunk_split,
                                 int type);
}  // namespace cpu
}  // namespace allspark