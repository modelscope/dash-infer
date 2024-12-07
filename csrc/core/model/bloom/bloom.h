/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    bloom.h
 */

#pragma once

#include <core/model/model.h>

#include <string>

namespace allspark {

class BloomModel : public AsModel {
 public:
  explicit BloomModel(const std::string& model_type = "")
      : AsModel(model_type) {}
  AsStatus Init(const TransformerProto& model_proto,
                const DeviceContext& ctx) override;
};
}  // namespace allspark
