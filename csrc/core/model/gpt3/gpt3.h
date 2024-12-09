/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    gpt3.h
 */

#pragma once

#include <core/model/model.h>

#include <string>

namespace allspark {

class GPT3Model : public AsModel {
 public:
  explicit GPT3Model(const std::string& model_type = "")
      : AsModel(model_type) {}
  AsStatus Init(const TransformerProto& model_proto,
                const DeviceContext& ctx) override;
};
}  // namespace allspark
