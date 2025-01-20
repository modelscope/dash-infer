/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    chatglm.cpp
 */

#include "chatglm.h"  // NOLINT

namespace allspark {
AsStatus ChatGLMModel::Init(const TransformerProto& model_proto,
                            const DeviceContext& ctx) {
  DLOG(INFO) << "ChatGLMModel::Init()" << std::endl;
  AS_CHECK_STATUS(AsModel::Init(model_proto, ctx));
  topo_ops_.clear();
  // parse graph
  for (auto& op : graph_ops_["decoder"]) {
    topo_ops_.emplace_back(op.get());
  }
  for (auto& op : graph_ops_["gen_graph"]) {
    topo_ops_.emplace_back(op.get());
  }
  return AsStatus::ALLSPARK_SUCCESS;
}

REGISTER_MODEL("ChatGLM_v1", ChatGLMModel)
REGISTER_MODEL("ChatGLM_v2", ChatGLM_v2Model)
REGISTER_MODEL("ChatGLM_v3", ChatGLM_v3Model)
REGISTER_MODEL("ChatGLM_v4", ChatGLM_v4Model)
}  // namespace allspark
