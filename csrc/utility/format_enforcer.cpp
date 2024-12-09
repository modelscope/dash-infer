/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    format_enforcer.cpp
 */

#include "format_enforcer.h"

#include <cstdarg>

namespace allspark {

std::ostream& operator<<(std::ostream& os, const VocabType& vt) {
  switch (vt) {
    case VocabType::VOCAB_TYPE_WPM:
      return os << "VOCAB_TYPE_WPM";
    case VocabType::VOCAB_TYPE_SPM:
      return os << "VOCAB_TYPE_SPM";
    case VocabType::VOCAB_TYPE_UGM:
      return os << "VOCAB_TYPE_UGM";
    case VocabType::VOCAB_TYPE_BPE:
      return os << "VOCAB_TYPE_BPE";
    default:
      LOG(ERROR) << "Operator \'<<\' got invalid VocabType!";
      AS_THROW(AsStatus::ALLSPARK_RUNTIME_ERROR)
  }
}

namespace util {

std::map<int, std::string> token2piece_cache_;

// taken from llama.cpp/src/llama-vocab.cpp::format()
__attribute__((format(printf, 1, 2))) static std::string format(const char* fmt,
                                                                ...) {
  va_list ap;
  va_list ap2;
  va_start(ap, fmt);
  va_copy(ap2, ap);
  int size = vsnprintf(NULL, 0, fmt, ap);
  AS_ENFORCE(size >= 0 && size < INT_MAX);  // NOLINT
  std::vector<char> buf(size + 1);
  int size2 = vsnprintf(buf.data(), size + 1, fmt, ap2);
  AS_ENFORCE(size2 == size);
  va_end(ap2);
  va_end(ap);
  return std::string(buf.data(), size);
}

// taken from llama.cpp/src/unicode.cpp::unicode_cpt_to_utf8()
std::string unicode_cpt_to_utf8(uint32_t cp) {
  std::string result;

  if (/* 0x00 <= cp && */ cp <= 0x7f) {
    result.push_back(cp);
    return result;
  }
  if (0x80 <= cp && cp <= 0x7ff) {
    result.push_back(0xc0 | ((cp >> 6) & 0x1f));
    result.push_back(0x80 | (cp & 0x3f));
    return result;
  }
  if (0x800 <= cp && cp <= 0xffff) {
    result.push_back(0xe0 | ((cp >> 12) & 0x0f));
    result.push_back(0x80 | ((cp >> 6) & 0x3f));
    result.push_back(0x80 | (cp & 0x3f));
    return result;
  }
  if (0x10000 <= cp && cp <= 0x10ffff) {
    result.push_back(0xf0 | ((cp >> 18) & 0x07));
    result.push_back(0x80 | ((cp >> 12) & 0x3f));
    result.push_back(0x80 | ((cp >> 6) & 0x3f));
    result.push_back(0x80 | (cp & 0x3f));
    return result;
  }

  LOG(ERROR) << "unicode_cpt_to_utf8 failed to convert "
                "codepoint to utf8!\n"
             << "input codepoint(uint32): " << cp << "\n";
  AS_THROW(AsStatus::ALLSPARK_RUNTIME_ERROR)
}

// basically taken from llama.cpp/src/unicode.cpp::unicode_utf8_to_byte_map()
void unicode_utf8_to_byte_map(std::map<std::string, uint8_t>& utf8_to_byte) {
  for (int ch = 0x21; ch <= 0x7E; ++ch) {  // u'!' to u'~'
    AS_ENFORCE(0 <= ch && ch < 256);
    utf8_to_byte[unicode_cpt_to_utf8(ch)] = ch;
  }
  for (int ch = 0xA1; ch <= 0xAC; ++ch) {  // u'¡' to u'¬'
    AS_ENFORCE(0 <= ch && ch < 256);
    utf8_to_byte[unicode_cpt_to_utf8(ch)] = ch;
  }
  for (int ch = 0xAE; ch <= 0xFF; ++ch) {  // u'®' to u'ÿ'
    AS_ENFORCE(0 <= ch && ch < 256);
    utf8_to_byte[unicode_cpt_to_utf8(ch)] = ch;
  }
  auto n = 0;
  for (int ch = 0; ch < 256; ++ch) {
    if (utf8_to_byte.find(unicode_cpt_to_utf8(ch)) == utf8_to_byte.end()) {
      utf8_to_byte[unicode_cpt_to_utf8(256 + n)] = ch;
      ++n;
    }
  }
}

std::vector<std::tuple<int, std::string, bool>>
AsTokenizerData::get_regular_tokens() const {
  std::vector<std::tuple<int, std::string, bool>> regular_tokens(
      FormatEnforcer::vocab_.size());
  for (int i = 0; i < FormatEnforcer::vocab_.size(); ++i) {
    auto token_str = decode_token(i);
    const std::vector<int> tokens = {i};
    auto token_sequence_str = decode_tokens(tokens);
    bool is_new_word = token_str.size() > token_sequence_str.size();
    regular_tokens[i] = std::make_tuple(i, token_str, is_new_word);
  }
  return regular_tokens;
};

void unescape_whitespace(std::string& s) {
  std::string search = "\xe2\x96\x81";
  std::string replace = " ";
  if (search.empty()) {
    return;
  }
  std::string builder;
  builder.reserve(s.length());
  size_t pos = 0;
  size_t last_pos = 0;
  while ((pos = s.find(search, last_pos)) != std::string::npos) {
    builder.append(s, last_pos, pos - last_pos);
    builder.append(replace);
    last_pos = pos + search.length();
  }
  builder.append(s, last_pos, std::string::npos);
  s = std::move(builder);
}

// taken from llama.cpp/src/unicode.cpp::unicode_cpt_from_utf8
uint32_t unicode_cpt_from_utf8(const std::string& utf8, size_t& offset) {
  AS_ENFORCE(offset < utf8.size());
  if (!(utf8[offset + 0] & 0x80)) {
    auto result = utf8[offset + 0];
    offset += 1;
    return result;
  }
  if (!(utf8[offset + 0] & 0x40)) {
    LOG(ERROR) << "unicode_cpt_from_utf8 got invalid character!\n"
               << "current character: " << utf8[offset + 0] << "\n";
    AS_THROW(AsStatus::ALLSPARK_RUNTIME_ERROR)
  }
  if (!(utf8[offset + 0] & 0x20)) {
    if (offset + 1 >= utf8.size() || !((utf8[offset + 1] & 0xc0) == 0x80)) {
      LOG(ERROR) << "unicode_cpt_from_utf8 got invalid character!\n"
                 << "current character: " << utf8[offset + 0] << "\n";
      AS_THROW(AsStatus::ALLSPARK_RUNTIME_ERROR)
    }
    auto result = ((utf8[offset + 0] & 0x1f) << 6) | (utf8[offset + 1] & 0x3f);
    offset += 2;
    return result;
  }
  if (!(utf8[offset + 0] & 0x10)) {
    if (offset + 2 >= utf8.size() || !((utf8[offset + 1] & 0xc0) == 0x80) ||
        !((utf8[offset + 2] & 0xc0) == 0x80)) {
      LOG(ERROR) << "unicode_cpt_from_utf8 got invalid character!\n"
                 << "current character: " << utf8[offset + 0] << "\n";
      AS_THROW(AsStatus::ALLSPARK_RUNTIME_ERROR)
    }
    auto result = ((utf8[offset + 0] & 0x0f) << 12) |
                  ((utf8[offset + 1] & 0x3f) << 6) | (utf8[offset + 2] & 0x3f);
    offset += 3;
    return result;
  }
  if (!(utf8[offset + 0] & 0x08)) {
    if (offset + 3 >= utf8.size() || !((utf8[offset + 1] & 0xc0) == 0x80) ||
        !((utf8[offset + 2] & 0xc0) == 0x80) ||
        !((utf8[offset + 3] & 0xc0) == 0x80)) {
      LOG(ERROR) << "unicode_cpt_from_utf8 got invalid character!\n"
                 << "current character: " << utf8[offset + 0] << "\n";
      AS_THROW(AsStatus::ALLSPARK_RUNTIME_ERROR)
    }
    auto result = ((utf8[offset + 0] & 0x07) << 18) |
                  ((utf8[offset + 1] & 0x3f) << 12) |
                  ((utf8[offset + 2] & 0x3f) << 6) | (utf8[offset + 3] & 0x3f);
    offset += 4;
    return result;
  }
  LOG(ERROR) << "unicode_cpt_from_utf8 failed to convert utf8 "
                "to codepoint!\n"
             << "input text: " << utf8 << "\n";
  AS_THROW(AsStatus::ALLSPARK_RUNTIME_ERROR)
}

// taken from llama.cpp/src/llama-vocab.cpp::unicode_cpts_from_utf8()
std::vector<uint32_t> unicode_cpts_from_utf8(const std::string& utf8) {
  std::vector<uint32_t> result;
  result.reserve(utf8.size());
  size_t offset = 0;
  while (offset < utf8.size()) {
    result.push_back(unicode_cpt_from_utf8(utf8, offset));
  }
  return result;
}

// taken from llama.cpp/src/llama-vocab.cpp::llama_decode_text()
std::string bpe_decode_text(
    const std::string& token_text,
    const std::map<std::string, uint8_t>& utf8_to_byte) {
  std::string decoded_text;
  const auto cpts = unicode_cpts_from_utf8(token_text);
  for (const auto cpt : cpts) {
    const auto utf8 = unicode_cpt_to_utf8(cpt);
    try {
      decoded_text += utf8_to_byte.at(utf8);
    } catch (const std::out_of_range& /*e*/) {
      decoded_text += "[UNK_BYTE_0x";
      for (const auto c : utf8) {
        decoded_text += format("%02x", (uint8_t)c);
      }
      decoded_text += token_text + "]";
    }
  }
  return decoded_text;
}

/*
@function token_to_piece
    decoded text written to 'buf'
    return: decoded text size(in byte)

    mostly taken from llama.cpp/src/llama-vocab.cpp::llama_token_to_piece_impl()
*/
int token_to_piece(int token, char* buf, int length, int lstrip,
                   const AsTokenizerData& token_data) {
  // copy piece chars to output text buffer
  // skip up to 'lstrip' leading spaces before copying
  auto _try_copy = [=](const char* token, size_t size) -> int32_t {
    for (int32_t i = 0; i < lstrip && size && *token == ' '; ++i) {
      token++;
      size--;
    }
    if (length < (int32_t)size) {
      return -(int32_t)size;
    }
    memcpy(buf, token, size);
    return (int32_t)size;
  };

  // if we have a cache - use it
  if (token2piece_cache_.find(token) != token2piece_cache_.end()) {
    std::string& result = token2piece_cache_[token];
    return _try_copy(result.data(), result.size());
  }

  std::string token_text;
  try {
    token_text = token_data.id2token_.at(token);
    switch (token_data.vocab_type_) {
      case VocabType::VOCAB_TYPE_WPM:
      case VocabType::VOCAB_TYPE_SPM:
      case VocabType::VOCAB_TYPE_UGM: {
        std::string result = token_text;
        unescape_whitespace(result);
        token2piece_cache_[token] = result;
        return _try_copy(result.data(), result.size());
        break;
      }
      case VocabType::VOCAB_TYPE_BPE: {
        std::string result =
            bpe_decode_text(token_text, token_data.utf8_to_byte_);
        token2piece_cache_[token] = result;
        return _try_copy(result.data(), result.size());
        break;
      }
      default:
        LOG(ERROR) << "token_to_piece unknown VocabType!\n"
                   << "Got type: " << token_data.vocab_type_ << "\n";
        AS_THROW(AsStatus::ALLSPARK_RUNTIME_ERROR)
    }
  } catch (const std::out_of_range& /*e*/) {
    LOG(ERROR) << "token_to_piece token ID doesn't exist in "
                  "model vocab!\n"
               << "Got ID: " << token << "\n";
    AS_THROW(AsStatus::ALLSPARK_RUNTIME_ERROR)
  }

  return 0;
}

AsTokenizerData::AsTokenizerData(VocabType type, int eos_token_id)
    : vocab_type_(type) {
  this->eos_token_id = eos_token_id;
  for (auto& itr : FormatEnforcer::vocab_) {
    id2token_[itr.second] = itr.first;
  }
  unicode_utf8_to_byte_map(utf8_to_byte_);
}

std::string AsTokenizerData::decode_token(int token) const {
  std::vector<char> result(8, 0);
  const int n_tokens =
      token_to_piece(token, result.data(), result.size(), 0, *this);
  if (n_tokens < 0) {
    result.resize(-n_tokens);
    int check = token_to_piece(token, result.data(), result.size(), 0, *this);
    AS_ENFORCE(check == -n_tokens);
  } else {
    result.resize(n_tokens);
  }
  return std::string(result.data(), result.size());
}

std::string AsTokenizerData::decode_tokens(
    const std::vector<int>& tokens) const {
  std::string piece;
  std::string result;
  result.reserve(tokens.size() * 4);
  for (size_t i = 0; i < tokens.size(); ++i) {
    piece = decode_token(tokens[i]);

    // remove the leading space of the first non-BOS token
    if (i == 0 && piece[0] == ' ') {
      piece = piece.substr(1);
    }

    result.append(piece);
  }

  return result;
}

/* Caution: members below defined as static because currently only one model can
 * be loaded into the engine instance, so these members can be shared between
 * all requests  */
std::shared_ptr<AsTokenizerData> FormatEnforcer::tokenizer_data_ = nullptr;
std::unordered_map<std::string, std::shared_ptr<JsonSchemaParser>>
    FormatEnforcer::parser_map_;
std::map<std::string, int> FormatEnforcer::vocab_;
VocabType FormatEnforcer::vocab_type_;

FormatEnforcer::FormatEnforcer(std::map<std::string, int>& vocab,
                               std::string& schema_str, VocabType vocab_type,
                               int eos_token_id) {
  if (vocab_.empty() == true) {
    vocab_ = std::move(vocab);
    vocab_type_ = vocab_type;
  }
  if (tokenizer_data_ == nullptr) {
    tokenizer_data_ =
        std::make_shared<AsTokenizerData>(vocab_type_, eos_token_id);
    tokenizer_data_->initialize();
  }
  if (parser_map_.find(schema_str) == parser_map_.end()) {
    parser_map_[schema_str] =
        std::make_shared<JsonSchemaParser>(schema_str, nullptr);
  }
  token_enforcer_ =
      std::make_unique<TokenEnforcer>(tokenizer_data_, parser_map_[schema_str]);
}

}  // namespace util
}  // namespace allspark