/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    gpt2.h
 */

#pragma once

#include <core/model/model.h>

#include <string>

namespace allspark {

class GPT2Model : public AsModel {
 public:
  explicit GPT2Model(const std::string& model_type = "")
      : AsModel(model_type) {}
  AsStatus Init(const TransformerProto& model_proto,
                const DeviceContext& ctx) override;
};
}  // namespace allspark
