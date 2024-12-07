/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    bf16_sm80.cu
 */

#ifdef ENABLE_BF16
#include "attn/qk/impl_sm80.cuh"
#include "common/data_type.h"

namespace span {
template struct QKLauncher<SaArch::SM80, bfloat16_t>;
}  // namespace span
#endif
