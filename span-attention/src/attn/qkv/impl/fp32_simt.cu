/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    fp32_simt.cu
 */

#include "attn/qkv/impl_simt.cuh"

namespace span {
template struct QKVWorkspaceBytes<SaArch::SIMT, float>;
template struct QKVLauncher<SaArch::SIMT, float>;
}  // namespace span
