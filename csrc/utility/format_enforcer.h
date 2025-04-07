/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    format_enforcer.h
 */
// using third_party/from_source/lmfe-cpp
#pragma once

#include <common/common.h>
#include <common/device_context.h>
#include <core/tensor/tensor.h>

#include <limits>
#include <lmfe/jsonschemaparser.hpp>
#include <lmfe/tokenenforcer.hpp>

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#endif

namespace allspark {
namespace util {

// 负责建出前缀树以及generate过程中decode
class AsTokenizerData : public TokenEnforcerTokenizerData {
 public:
  AsTokenizerData(VocabType type, int eos_token_id);

  std::string decode(const std::vector<int>& tokens) const override {
    return decode_tokens(tokens);
  }

 protected:
  std::vector<std::tuple<int, std::string, bool>> get_regular_tokens()
      const override;

  int get_eos_token_id() const override {
    // not used
    return 0;
  }

  std::string decode_token(int token) const;
  std::string decode_tokens(const std::vector<int>& tokens) const;

 public:
  VocabType vocab_type_;
  std::map<int, std::string> id2token_;
  std::map<std::string, uint8_t> utf8_to_byte_;
};

// 负责对一条request进行format enforce
class FormatEnforcer {
 public:
  FormatEnforcer(std::map<std::string, int>& vocab, std::string& schema_str,
                 VocabType vocab_type, int eos_token_id);

  std::unique_ptr<TokenEnforcer> token_enforcer_;
  static std::shared_ptr<AsTokenizerData> tokenizer_data_;
  std::vector<int> gen_sequence;

  static std::map<std::string, int> vocab_;
  static VocabType vocab_type_;
  std::vector<int8_t> scores_mask_;

#ifdef ENABLE_CUDA
  // 存储从GPU拷贝来的logits，类型未必是float，但用float占位可以避免整个类写成模板类
  // 在GenerateOp::Reshape进行空间申请
  float* scores_buf_ = nullptr;

  ~FormatEnforcer() {
    if (scores_buf_ != nullptr) {
      cudaFreeHost(scores_buf_);
      scores_buf_ = nullptr;
    }
  }
#endif

  // logits processor
  template <typename T>
  AsStatus process_logits(FrozenTokenVector& allowed_tokens, T* in_ptr,
                          const DeviceContext* ctx, size_t dtype_size,
                          int model_vocab_size) {
    scores_mask_.assign(model_vocab_size, 0);
    T* scores = in_ptr;
#ifdef ENABLE_CUDA
    if (ctx->GetDeviceType() == DeviceType::CUDA) {
      CopyData((void*)scores_buf_, DeviceType::CPU, (void*)(in_ptr),
               DeviceType::CUDA, dtype_size * model_vocab_size, ctx);
      ctx->Synchronize();
      scores = (T*)scores_buf_;
    }
#endif
    for (auto& token : allowed_tokens) {
      AS_ENFORCE(token < scores_mask_.size());
      scores_mask_[token] = 1;
    }
    for (int i = 0; i < scores_mask_.size(); i++) {
      if (scores_mask_[i] == 0) {
        scores[i] = std::numeric_limits<float>::lowest();
      }
    }
#ifdef ENABLE_CUDA
    if (ctx->GetDeviceType() == DeviceType::CUDA) {
      CopyData((void*)(in_ptr), DeviceType::CUDA, (void*)scores,
               DeviceType::CPU, dtype_size * model_vocab_size, ctx);
    }
#endif
    return AsStatus::ALLSPARK_SUCCESS;
  }

 private:
  static std::unordered_map<std::string, std::shared_ptr<JsonSchemaParser>>
      parser_map_;
};

}  // namespace util
}  // namespace allspark
