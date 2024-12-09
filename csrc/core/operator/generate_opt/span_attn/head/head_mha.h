/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    head_mha.h
 */
#pragma once

#include "head_base.h"

namespace allspark {

class HeadMHA : public AttentionHead {
 public:
  HeadMHA(int num_heads, int num_groups, int size_per_head, bool is_multi_node,
          int nranks)
      : AttentionHead(num_heads, num_groups, size_per_head) {
    if (num_groups_ != 0 && num_groups_ != num_heads_) {
      LOG(ERROR) << "HeadMHA: invalid attribute, num_groups should "
                 << "pass in 0 or " << num_heads_ << ", got " << num_groups_;
      AS_THROW(AsStatus::ALLSPARK_PARAM_ERROR);
    }

    if (is_multi_node) {
      if (num_heads_ < nranks || num_heads_ % nranks != 0) {
        LOG(ERROR) << "HeadMHA: invalid attribute, num_heads: " << num_heads_
                   << ", nranks: " << nranks;
        AS_THROW(AsStatus::ALLSPARK_PARAM_ERROR);
      }
      num_heads_ /= nranks;
    }
    num_groups_ = num_heads_;
    kv_stride_ = size_per_head_ * num_groups_;
  }

  /* for Reshape */
  AsStatus UpdateShape(int qkv_stride) override {
    qkv_stride_ = qkv_stride;
    // for MHA, QKV share the same size
    if (qkv_stride_ % 3 != 0) {
      LOG(ERROR) << "HeadMHA: invalid shape, qkv_stride should be a "
                 << "multiple of 3, got " << qkv_stride_;
      return AsStatus::ALLSPARK_PARAM_ERROR;
    }
    hidden_size_ = qkv_stride_ / 3;
    return AsStatus::ALLSPARK_SUCCESS;
  }
};

}  // namespace allspark
