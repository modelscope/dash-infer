/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    fp16_sm80.cu
 */

#ifdef ENABLE_FP16
#include "attn/qkv/impl_sm80.cuh"
#include "common/data_type.h"

namespace span {
template struct QKVWorkspaceBytes<SaArch::SM80, half_t>;
template struct QKVLauncher<SaArch::SM80, half_t>;
}  // namespace span
#endif
