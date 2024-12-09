/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    format_enforcer.h
 */
// using third_party/from_source/lmfe-cpp
#pragma once

#include <common/common.h>

#include <lmfe/jsonschemaparser.hpp>
#include <lmfe/tokenenforcer.hpp>

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

 private:
  static std::unordered_map<std::string, std::shared_ptr<JsonSchemaParser>>
      parser_map_;
};

}  // namespace util
}  // namespace allspark
