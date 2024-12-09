/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    rotary_op.h
 */

#pragma once

#include <core/operator/operator.h>

#include <cmath>
namespace allspark {
enum RotaryType : int {
  base = 0,
  position_embedding_2D = 1,  // use for chatglm_v1
  rotary_pct = 2,             // use for dolly_v2
  half_inner = 3,             // use for chatglm_v2/3
  mrope = 4,                  // use for qwen2_vl
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
                     AsTensor* rotary_inv_freq, bool use_positions);
  float yarn_find_correction_dim(int num_rotations, int dim, float base,
                                 int max_position_embeddings) {
    return (dim *
            std::log(max_position_embeddings / (num_rotations * 2 * M_PI))) /
           (2 * std::log(base));
  }

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
      case RotaryInvFreqType::yarn: {
        float scaling_factor =
            (float)query_length / float(original_max_position_embeddings_);
        if (scaling_factor < 1) {
          scaling_factor = 1.f;
        }
        float mscale = 0.1 * std::log(scaling_factor) + 1.f;
        int invfreq_len = size_per_head_ / 2;
        std::vector<float> invfreq_extrapolation;
        std::vector<float> invfreq_interpolation;
        for (int i = 0; i < invfreq_len; i++) {
          float exponent = float(i * 2) / float(size_per_head_);
          float theta = 1.f / (std::pow(base, exponent));
          invfreq_extrapolation.emplace_back(theta);
          invfreq_interpolation.emplace_back(theta / scaling_factor);
        }
        if (query_length < original_max_position_embeddings_) {
          return invfreq_extrapolation;
        }
        int beta_fast = 32;
        int beta_slow = 1;
        float low = std::floor(
            yarn_find_correction_dim(beta_fast, size_per_head_, base,
                                     original_max_position_embeddings_));
        float high = std::ceil(
            yarn_find_correction_dim(beta_slow, size_per_head_, base,
                                     original_max_position_embeddings_));
        low = std::max(low, 0.f);
        high = std::min(high, float(size_per_head_ - 1));
        if (low == high) {
          high += 0.001;
        }
        for (int i = 0; i < invfreq_len; i++) {
          float x = (i - low) / (high - low);
          x = std::clamp(x, 0.f, 1.f);
          x = 1.f - x;
          invfreq.emplace_back((invfreq_interpolation[i] * (1 - x) +
                                invfreq_extrapolation[i] * x) *
                               mscale);
        }
        break;
      }
      default: {
        LOG(ERROR) << "RotaryOp: unsupported invfreq_type" << std::endl;
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

  // about shape
  int group_num_ = 0;
  int qkv_stride_ = 0;
  int kv_stride_ = 0;
  int hidden_size_;
  int num_heads_;
  int size_per_head_;
  int gemm_batch_;
  int64_t score_size_;
  // about calc_infreq
  float base_;
  bool first_beam_;
  int rotary_type_;
  bool use_weight_;
  float rotary_pct_;
  float seqlen_extrapolation_;
  int ntk_model_embed_;
  float rope_ratio_;
  int invfreq_type_;  // refer to RotaryInvFreqType
  int original_max_position_embeddings_;
  int mrope_size_ = 0;
  std::unique_ptr<AsTensor> mrope_section_;
  std::unique_ptr<AsTensor> mrope_position_;
};

}  // namespace allspark
