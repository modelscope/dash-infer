/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    head_base.h
 */
#pragma once

#include <glog/logging.h>

#include "allspark_check.h"

namespace allspark {

class AttentionHead {
 public:
  AttentionHead(int num_heads, int num_groups, int size_per_head)
      : num_heads_(num_heads),
        num_groups_(num_groups),
        size_per_head_(size_per_head),
        hidden_size_(1),
        qkv_stride_(1),
        kv_stride_(1) {}
  virtual ~AttentionHead() {}

  int NumHeads() const { return num_heads_; }
  int NumGroups() const { return num_groups_; }
  int KVStride() const { return kv_stride_; }
  int QKVStride() const { return qkv_stride_; }
  int HiddenSize() const { return hidden_size_; }
  int SizePerHead() const { return size_per_head_; }

  /* for Reshape */
  virtual AsStatus UpdateShape(int qkv_stride) = 0;

 protected:
  // load from model config
  /// @brief number of Q channels
  int num_heads_;
  /// @brief number of K / V channels
  int num_groups_;
  /// @brief size of each channel
  int size_per_head_;

  /// @brief size of Q / output
  int hidden_size_;

  // deduced from input shape
  int qkv_stride_;

  // computed from attributes
  /// @brief size of K / V
  int kv_stride_;
};

}  // namespace allspark
