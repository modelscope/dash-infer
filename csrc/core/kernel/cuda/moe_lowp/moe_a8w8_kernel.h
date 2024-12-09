/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    moe_a8w8_kernel.h
 */

#pragma once
#include <cstdint>
#include <cstdlib>

#include "../cuda_common.h"
#include "../hie/cuda_activation.hpp"

namespace allspark {
namespace cuda {

void GetReorderQData(int8_t* reorder_qdata, float* reorder_scale,
                     int32_t* reorder_red_sum, int8_t* input_q,
                     float* input_scale, int32_t* input_red_sum,
                     int64_t* experts_idx, int64_t* experts_seq,
                     int* total_tokens_post_pad, int max_total_tokens,
                     int padding_val, int topk, int hidden_size, int block_size,
                     cudaStream_t stream);
template <typename T>
void GetReorderQWeightData(int64_t* experts_idx, int* total_tokens_post_pad,
                           T* scale, T* scale_in, T* zero, T* zero_in,
                           int max_block, int layout_size, int block_size,
                           cudaStream_t stream);
template <typename FT, typename QT>
void MOEQWeightGetBatchArrayLauncher(
    int64_t* experts_idx, int* total_tokens_post_pad, QT* qdata,
    void** qdata_array, FT* scale, void** scale_array, FT* zero,
    void** zero_array, int max_block, int layout_size_qdata,
    int layout_size_scale, int block_size, cudaStream_t stream);
}  // namespace cuda
}  // namespace allspark