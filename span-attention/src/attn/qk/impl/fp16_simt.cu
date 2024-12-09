/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    fp16_simt.cu
 */

#ifdef ENABLE_FP16
#include "attn/qk/impl_simt.cuh"
#include "common/data_type.h"

namespace span {
template struct QKLauncher<SaArch::SIMT, half_t>;
}  // namespace span
#endif
