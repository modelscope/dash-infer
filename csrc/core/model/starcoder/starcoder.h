/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    starcoder.h
 */

#pragma once

#include <core/model/model.h>

#include <string>

namespace allspark {

class StarCoderModel : public AsModel {
 public:
  explicit StarCoderModel(const std::string& model_type = "")
      : AsModel(model_type) {}
  AsStatus Init(const TransformerProto& model_proto,
                const DeviceContext& ctx) override;
};
}  // namespace allspark
