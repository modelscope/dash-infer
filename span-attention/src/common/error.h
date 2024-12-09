/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    error.h
 */

#pragma once

#include <span_attn.h>

#include <stdexcept>

namespace span {

class SpanAttnError : public std::runtime_error {
  SaStatus code_;

 public:
  SpanAttnError(SaStatus code) : SpanAttnError(code, GetErrorString(code)) {}
  SpanAttnError(SaStatus code, const std::string& msg)
      : std::runtime_error(msg), code_(code) {}
  SpanAttnError(SaStatus code, const char* msg)
      : std::runtime_error(msg), code_(code) {}

  virtual ~SpanAttnError() = default;

  SaStatus code() const { return code_; }
};
}  // namespace span
