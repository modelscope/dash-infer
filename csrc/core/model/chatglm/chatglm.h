/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    chatglm.h
 */

#pragma once

#include <core/model/model.h>

#include <string>

namespace allspark {

class ChatGLMModel : public AsModel {
 public:
  explicit ChatGLMModel(const std::string& model_type = "")
      : AsModel(model_type) {}
  AsStatus Init(const TransformerProto& model_proto,
                const DeviceContext& ctx) override;
  AsStatus Forward(const TensorMap& inputs, TensorMap* outputs);
};

class ChatGLM_v2Model : public ChatGLMModel {
 public:
  explicit ChatGLM_v2Model(const std::string& model_type = "")
      : ChatGLMModel(model_type){};
};

class ChatGLM_v3Model : public ChatGLMModel {
 public:
  explicit ChatGLM_v3Model(const std::string& model_type = "")
      : ChatGLMModel(model_type){};
};
}  // namespace allspark
