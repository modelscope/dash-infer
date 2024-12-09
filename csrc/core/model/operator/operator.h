/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    operator.h
 */

#pragma once

#include <core/model/model.h>

#include <string>

namespace allspark {

class OperatorModel : public AsModel {
 public:
  explicit OperatorModel(const std::string& model_type = "")
      : AsModel(model_type) {}
  AsStatus Init(const TransformerProto& model_proto,
                const DeviceContext& ctx) override;

 private:
  int num_layers_ = 1;
};
}  // namespace allspark
