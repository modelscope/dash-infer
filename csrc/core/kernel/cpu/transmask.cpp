/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    transmask.cpp
 */

#include "cpu_common.h"
#include "cpu_kernel.h"

namespace allspark {
namespace cpu {

template <>
void TransMaskKernel<float>(float* out, const int64_t* mask, int batch_size,
                            int seq_length, bool seq_mask, bool blank) {
  int N = batch_size * seq_length * seq_length;
  if (seq_mask) {
    parallel_for(N, [&](int idx) {
      const int idx1 = idx % seq_length;  // seq_id
      const int tmp_idx = idx / seq_length;
      const int idx2 = tmp_idx % seq_length;
      const int idx3 = tmp_idx / seq_length;  // batch_id
      const int mask_idx = idx3 * seq_length + idx1;
      int64_t mask_val = mask == nullptr ? 1 : mask[mask_idx];
      out[idx] = ((int)(idx2 < idx1 ? 0 : 1)) & (int)mask_val;
    });
  } else if (blank) {
    parallel_for(N, [&](int idx) {
      const int idx1 = idx % seq_length;  // seq_id
      const int tmp_idx = idx / seq_length;
      const int idx2 = tmp_idx % seq_length;
      const int idx3 = tmp_idx / seq_length;  // batch_id
      const int mask_idx = idx3 * seq_length + idx1;
      int64_t mask_val = mask == nullptr ? 1 : mask[mask_idx];
      if (idx1 >= seq_length - 1) {
        out[idx] = ((int)(idx2 < idx1 ? 0 : 1)) & (int)mask_val;
      } else {
        out[idx] = (int)mask_val;
      }
    });
  } else {
    parallel_for(N, [&](int idx) {
      const int idx1 = idx % seq_length;  // seq_id
      const int tmp_idx = idx / seq_length;
      const int idx2 = tmp_idx % seq_length;
      const int idx3 = tmp_idx / seq_length;  // batch_id
      const int mask_idx = idx3 * seq_length + idx1;
      int64_t mask_val = mask == nullptr ? 1 : mask[mask_idx];
      out[idx] = (int)mask_val;
    });
  }
}

}  // namespace cpu
}  // namespace allspark