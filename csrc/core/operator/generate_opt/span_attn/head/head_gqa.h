/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    head_gqa.h
 */
#pragma once

#include "head_base.h"

namespace allspark {

class HeadGQA : public AttentionHead {
 public:
  HeadGQA(int num_heads, int num_groups, int size_per_head, bool is_multi_node,
          int nranks)
      : AttentionHead(num_heads, num_groups, size_per_head) {
    if (num_groups_ <= 0) {
      LOG(ERROR) << "HeadGQA: invalid attribute, num_groups should be a "
                    "positive integer";
      AS_THROW(AsStatus::ALLSPARK_PARAM_ERROR);
    }

    if (num_heads_ % num_groups_ != 0) {
      LOG(ERROR) << "HeadGQA: num_heads should be a multiple of num_groups, "
                    "num_heads: "
                 << num_heads_ << ", num_groups: " << num_groups_;
      AS_THROW(AsStatus::ALLSPARK_PARAM_ERROR);
    }

    if (is_multi_node) {
      if (num_heads_ < nranks || num_heads_ % nranks != 0) {
        LOG(ERROR) << "HeadGQA: invalid attribute, num_heads: " << num_heads_
                   << ", nranks: " << nranks;
        AS_THROW(AsStatus::ALLSPARK_PARAM_ERROR);
      }
      num_heads_ /= nranks;

      /*
       * For num_groups_ == 1, this is MQA, the same K/V head is copied to
       * every worker.
       * Otherwise, the K/V heads should be properly split to each worker.
       */
      if (num_groups_ != 1) {
        if (num_groups_ < nranks || num_groups_ % nranks != 0) {
          LOG(ERROR) << "HeadGQA: invalid attribute, num_groups: "
                     << num_groups_ << ", nranks: " << nranks;
          AS_THROW(AsStatus::ALLSPARK_PARAM_ERROR);
        }
        num_groups_ /= nranks;
      }
    }

    kv_stride_ = size_per_head_ * num_groups_;
  }

  /* for Reshape */
  AsStatus UpdateShape(int qkv_stride) override {
    qkv_stride_ = qkv_stride;
    // for GQA, KV share the same size
    hidden_size_ = qkv_stride_ - kv_stride_ * 2;
    return AsStatus::ALLSPARK_SUCCESS;
  }
};

}  // namespace allspark
