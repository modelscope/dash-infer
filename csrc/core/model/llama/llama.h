/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    llama.h
 */

#pragma once

#include <core/model/model.h>

#include <string>

namespace allspark {

class LLaMAModel : public AsModel {
 public:
  explicit LLaMAModel(const std::string& model_type = "")
      : AsModel(model_type) {}
  AsStatus Init(const TransformerProto& model_proto,
                const DeviceContext& ctx) override;
};

class LLaMA_v2Model : public LLaMAModel {
 public:
  explicit LLaMA_v2Model(const std::string& model_type = "")
      : LLaMAModel(model_type){};
};
}  // namespace allspark
