/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    transmask.cu
 */

#include "cuda_common.h"  // NOLINT
#include "cuda_kernel.h"

namespace allspark {
namespace cuda {

template <bool DECODER, typename T>
__global__ void trans_mask_kernel(T* out, const int64_t* mask, int seq_length,
                                  int N) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    const int idx1 = idx % seq_length;  // seq_id
    const int tmp_idx = idx / seq_length;
    const int idx2 = tmp_idx % seq_length;
    const int idx3 = tmp_idx / seq_length;  // batch_id
    const int mask_idx = idx3 * seq_length + idx1;
    int64_t mask_val = mask == nullptr ? 1 : (int)mask[mask_idx];
    if (DECODER) {
      out[idx] = ((int)(idx2 < idx1 ? 0 : 1)) & (int)mask_val;
    } else {
      out[idx] = (int)mask_val;
    }
  }
}
template <typename T>
__global__ void trans_mask_kernel_blank(T* out, const int64_t* mask,
                                        int seq_length, int N) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    const int idx1 = idx % seq_length;  // seq_id
    const int tmp_idx = idx / seq_length;
    const int idx2 = tmp_idx % seq_length;
    const int idx3 = tmp_idx / seq_length;  // batch_id
    const int mask_idx = idx3 * seq_length + idx1;
    int64_t mask_val = mask == nullptr ? 1 : (int)mask[mask_idx];
    if (idx1 >= seq_length - 1) {
      out[idx] = ((int)(idx2 < idx1 ? 0 : 1)) & (int)mask_val;
    } else {
      out[idx] = (int)mask_val;
    }
  }
}
template <typename T>
void TransMaskKernelLauncher(T* out, const int64_t* mask, int batch_size,
                             int seq_length, bool seq_mask, bool blank,
                             cudaStream_t stream) {
  int N = batch_size * seq_length * seq_length;
  const int block_num = (N + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
  if (seq_mask) {
    trans_mask_kernel<true, T>
        <<<block_num, THREAD_PER_BLOCK, 0, stream>>>(out, mask, seq_length, N);
  } else if (blank) {
    trans_mask_kernel_blank<T>
        <<<block_num, THREAD_PER_BLOCK, 0, stream>>>(out, mask, seq_length, N);
  } else {
    trans_mask_kernel<false, T>
        <<<block_num, THREAD_PER_BLOCK, 0, stream>>>(out, mask, seq_length, N);
  }
}

template void TransMaskKernelLauncher<float>(float* out, const int64_t* mask,
                                             int batch_size, int seq_length,
                                             bool seq_mask, bool blank,
                                             cudaStream_t stream);
#ifdef ENABLE_FP16
template void TransMaskKernelLauncher<half>(half* out, const int64_t* mask,
                                            int batch_size, int seq_length,
                                            bool seq_mask, bool blank,
                                            cudaStream_t stream);
#endif
template void TransMaskKernelLauncher<hie::bfloat16>(
    hie::bfloat16* out, const int64_t* mask, int batch_size, int seq_length,
    bool seq_mask, bool blank, cudaStream_t stream);
}  // namespace cuda
}  // namespace allspark