/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    transpose.cpp
 */

#include "cpu_common.h"
#include "cpu_kernel.h"

namespace allspark {
namespace cpu {
template <>
void TransposeAxis01KernelLauncher<float>(float* out, const float* in, int dim0,
                                          int dim1, int dim2) {
  int N = dim0 * dim1 * dim2;
  parallel_for(N, [&](int index) {
    const int input_dim2_index = index % dim2;
    index /= dim2;
    const int input_dim1_index = index % dim1;
    index = index / dim1;
    const int input_dim0_index = index % dim0;

    out[input_dim1_index * dim0 * dim2 + input_dim0_index * dim2 +
        input_dim2_index] = in[input_dim0_index * dim1 * dim2 +
                               input_dim1_index * dim2 + input_dim2_index];
  });
}

}  // namespace cpu
}  // namespace allspark