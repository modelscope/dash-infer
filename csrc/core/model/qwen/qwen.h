/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    qwen.h
 */

#pragma once

#include <core/model/model.h>

#include <string>

namespace allspark {

class QwenModel : public AsModel {
 public:
  explicit QwenModel(const std::string& model_type = "")
      : AsModel(model_type) {}
  AsStatus Init(const TransformerProto& model_proto,
                const DeviceContext& ctx) override;
};

class QwenModel_v10 : public QwenModel {
 public:
  explicit QwenModel_v10(const std::string& model_type = "")
      : QwenModel(model_type){};
};

class QwenModel_v15 : public QwenModel {
 public:
  explicit QwenModel_v15(const std::string& model_type = "")
      : QwenModel(model_type){};
};

class QwenModel_v20 : public QwenModel {
 public:
  explicit QwenModel_v20(const std::string& model_type = "")
      : QwenModel(model_type){};
};

class QwenCodeModel_v20 : public QwenModel {
 public:
  explicit QwenCodeModel_v20(const std::string& model_type = "")
      : QwenModel(model_type){};
};

class QwenModel_v20_MOE : public QwenModel {
 public:
  explicit QwenModel_v20_MOE(const std::string& model_type = "")
      : QwenModel(model_type){};
};
}  // namespace allspark
