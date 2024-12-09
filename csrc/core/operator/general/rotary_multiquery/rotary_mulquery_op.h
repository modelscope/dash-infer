/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    rotary_mulquery_op.h
 */

#pragma once

#include <core/operator/operator.h>

namespace allspark {

/* @brief: assumpt seq_len = 1
 * inputs:
      qkv_fuse: (batch * beam, 1, hidden_size)
      mask: (batch * beam, 1, step) (只有第一次解码用得到这东西)
      beam_idx : [batch * beam]
 * outputs:
      attn_out: (batch * beam, 1, hidden_size)
 */
enum RotaryType : int {
  base = 0,
  position_embedding_2D = 1,  // use for chatglm_v1
  rotary_pct = 2,             // use for dolly_v2
  half_inner = 3,             // use for chatglm_v2/3
  mrope = 4,                  // use for qwen2_vl
};
class RotaryMulQueryOp : public AsOperator {
 public:
  explicit RotaryMulQueryOp(const std::string& op_type = "")
      : AsOperator(op_type),
        batch_size_(1),
        seq_len_(1),
        hidden_size_(768),
        num_heads_(16),
        size_per_head_(64),
        base_(10000.f),
        group_num_(0),
        use_weight_(true),
        rotary_type_(0),
        rope_ratio_(1.0),
        invfreq_type_(0),
        ntk_model_embed_(-1) {}
  AsStatus Init(const OperatorProto& op_proto, const DeviceContext& ctx,
                const TensorMap& weights_map, TensorMap* tensor_map);
  AsStatus Reshape(RuntimeContext* runtime_ctx) override;
  AsStatus Forward(RuntimeContext* runtime_ctx) override;
  AsStatus RunContext(RuntimeContext* runtime_ctx);
  AsStatus RunDecoder(RuntimeContext* runtime_ctx);
  AsStatus RunRotary(int run_batch_size);

 private:
  std::vector<float> calculate_invfreq(float base, int query_length, int type) {
    std::vector<float> invfreq;
    switch (type) {
      case RotaryInvFreqType::base_rotary: {
        auto alpha = [&](float query_length) -> float {
          float context_value =
              log2(query_length / float(ntk_model_embed_)) + 1;
          float calculate_alpha = std::pow(2, ceil(context_value)) - 1;
          return calculate_alpha >= 1 ? calculate_alpha : 1.f;
        };

        // enable dynamic ntk. and not decoder.
        if (ntk_model_embed_ != -1) {
          base *=
              std::pow(alpha(float(query_length)),
                       float(size_per_head_) / (float(size_per_head_) - 2.f));
        }

        int invfreq_len = size_per_head_ / 2;
        for (int i = 0; i < invfreq_len; i++) {
          float exponent = float(i * 2) / float(size_per_head_);
          float theta =
              1.f / (std::pow(base, exponent) * seqlen_extrapolation_);
          invfreq.emplace_back(theta);
        }
        break;
      }
      case RotaryInvFreqType::chatglm_v2: {
        int step = 2;
        int rotary_dim = size_per_head_ / 2;
        for (int i = 0; i < rotary_dim; i += step) {
          float theta =
              1.0 / std::pow(base, (float)i / rotary_dim) / rope_ratio_;
          invfreq.emplace_back(theta);
        }
        break;
      }
      case RotaryInvFreqType::chatglm_v3: {
        int step = 2;
        int rotary_dim = size_per_head_ / 2;
        for (int i = 0; i < rotary_dim; i += step) {
          float theta =
              1.0 / std::pow(base * rope_ratio_, (float)i / rotary_dim);
          invfreq.emplace_back(theta);
        }
        break;
      }
      default: {
        LOG(ERROR) << "RotaryMulQueryOp: unsupported invfreq_type" << std::endl;
        break;
      }
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
  float base_;
  int rotary_type_;  // refert to RotaryType
  bool use_weight_;
  float seqlen_extrapolation_;
  int ntk_model_embed_;
  int group_num_ = 0;
  int qkv_stride_ = 0;
  int kv_stride_ = 0;
  float rope_ratio_;
  int invfreq_type_;  // refer to RotaryInvFreqType
  std::unique_ptr<AsTensor> inv_freq_;
  std::unique_ptr<AsTensor> run_step_;
  std::unique_ptr<AsTensor> run_step_host_;
};

}  // namespace allspark
