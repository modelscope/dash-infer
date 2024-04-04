/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    cast.cpp
 */

#include "cpu_common.h"
#include "cpu_kernel.h"
namespace allspark {
namespace cpu {

template <typename T0, typename T1>
void CastKernelLauncher(const T0* in, T1* out, int size) {
  parallel_for(size, [&](int i) { out[i] = static_cast<T1>(in[i]); });
}
template void CastKernelLauncher<int64_t, float>(const int64_t* in, float* out,
                                                 int size);
}  // namespace cpu
}  // namespace allspark
