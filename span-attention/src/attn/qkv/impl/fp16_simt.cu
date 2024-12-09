/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    fp16_simt.cu
 */

#ifdef ENABLE_FP16
#include "attn/qkv/impl_simt.cuh"
#include "common/data_type.h"

namespace span {
template struct QKVWorkspaceBytes<SaArch::SIMT, half_t>;
template struct QKVLauncher<SaArch::SIMT, half_t>;
}  // namespace span
#endif
