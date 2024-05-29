/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    rotary_op.h
 */

#pragma once

#include <core/operator/operator.h>

namespace allspark {
enum RotaryType : int {
  base = 0,
  position_embedding_2D = 1,  // use for chatglm_v1
  rotary_pct = 2,             // use for dolly_v2
  half_inner = 3,             // use for chatglm_v2
};
class RotaryOp : public AsOperator {
 public:
  explicit RotaryOp(const std::string& op_type = "")
      : AsOperator(op_type),
        batch_size_(1),
        seq_len_(1),
        hidden_size_(4096),
        num_heads_(32),
        size_per_head_(128),
        gemm_batch_(1),
        score_size_(0),
        base_(10000.f),
        pos_embedding_(false),
        first_beam_(false),
        use_weight_(false),
        rotary_type_(0),
        rotary_pct_(1.0),
        seqlen_extrapolation_(1.f) {}
  AsStatus Init(const OperatorProto& op_proto, const DeviceContext& ctx,
                const TensorMap& weights_map, TensorMap* tensor_map);
  AsStatus Reshape(RuntimeContext* runtime_ctx) override;
  AsStatus Forward(RuntimeContext* runtime_ctx) override;
  AsStatus RunContext(RuntimeContext* runtime_ctx);
  AsStatus RunDecoder(RuntimeContext* runtime_ctx);
  AsStatus RunRotary(int run_batch_size, AsTensor* rotary_step,
                     AsTensor* rotary_inv_freq);

 private:
  std::vector<float> calculate_invfreq(float base, int query_length) {
    auto alpha = [&](float query_length) -> float {
      float context_value = log2(query_length / float(ntk_model_embed_)) + 1;
      float calculate_alpha = std::pow(2, ceil(context_value)) - 1;
      return calculate_alpha >= 1 ? calculate_alpha : 1.f;
    };

    // enable dynamic ntk. and not decoder.
    if (ntk_model_embed_ != -1) {
      base *= std::pow(alpha(float(query_length)),
                       float(size_per_head_) / (float(size_per_head_) - 2.f));
    }

    int invfreq_len = size_per_head_ / 2;
    std::vector<float> invfreq(invfreq_len);
    for (int i = 0; i < invfreq_len; i++) {
      float exponent = float(i * 2) / float(size_per_head_);
      invfreq[i] = 1.f / (std::pow(base, exponent) * seqlen_extrapolation_);
    }
    return invfreq;
  }

 private:
  DataType dtype_ = DATATYPE_UNDEFINED;
  int xlogn_ = -1;  // model base sequence embedding length.
  int batch_size_;
  int seq_len_;
  int hidden_size_;
  int num_heads_;
  int size_per_head_;
  int gemm_batch_;
  int64_t score_size_;
  float base_;
  bool pos_embedding_;
  bool first_beam_;
  int rotary_type_;
  bool use_weight_;
  float rotary_pct_;
  float seqlen_extrapolation_;
  int ntk_model_embed_;
  int group_num_ = 0;
  int qkv_stride_ = 0;
  int kv_stride_ = 0;
};

}  // namespace allspark
