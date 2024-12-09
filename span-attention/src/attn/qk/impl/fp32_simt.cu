/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    fp32_simt.cu
 */

#include "attn/qk/impl_simt.cuh"

namespace span {
template struct QKLauncher<SaArch::SIMT, float>;
}  // namespace span
