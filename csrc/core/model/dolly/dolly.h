/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    dolly.h
 */

#pragma once

#include <core/model/model.h>

#include <string>

namespace allspark {

class DollyModel : public AsModel {
 public:
  explicit DollyModel(const std::string& model_type = "")
      : AsModel(model_type) {}
  AsStatus Init(const TransformerProto& model_proto,
                const DeviceContext& ctx) override;
};
}  // namespace allspark
