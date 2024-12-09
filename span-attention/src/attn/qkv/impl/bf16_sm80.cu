/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    bf16_sm80.cu
 */

#ifdef ENABLE_BF16
#include "attn/qkv/impl_sm80.cuh"
#include "common/data_type.h"

namespace span {
template struct QKVWorkspaceBytes<SaArch::SM80, bfloat16_t>;
template struct QKVLauncher<SaArch::SM80, bfloat16_t>;
}  // namespace span
#endif
