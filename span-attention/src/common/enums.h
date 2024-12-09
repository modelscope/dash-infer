/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    enums.h
 */

#pragma once

#include <span_attn.h>

#include <string>

namespace span {

std::string to_string(DataType type);

std::string to_string(SaStatus code);

std::string to_string(QuantMode mode);

}  // namespace span
