/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    baichuan.h
 */

#pragma once

#include <core/model/model.h>

#include <string>

namespace allspark {

class BaichuanModel : public AsModel {
 public:
  explicit BaichuanModel(const std::string& model_type = "")
      : AsModel(model_type) {}
  AsStatus Init(const TransformerProto& model_proto,
                const DeviceContext& ctx) override;
};

class BaichuanModel_v1 : public BaichuanModel {
 public:
  explicit BaichuanModel_v1(const std::string& model_type = "")
      : BaichuanModel(model_type){};
};

class BaichuanModel_v2 : public BaichuanModel {
 public:
  explicit BaichuanModel_v2(const std::string& model_type = "")
      : BaichuanModel(model_type){};
};
}  // namespace allspark
