/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    bf16_simt.cu
 */

#ifdef ENABLE_BF16
#include "attn/qk/impl_simt.cuh"
#include "common/data_type.h"

namespace span {
template struct QKLauncher<SaArch::SIMT, bfloat16_t>;
}  // namespace span
#endif
