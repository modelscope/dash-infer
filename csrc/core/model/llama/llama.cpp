/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    llama.cpp
 */

#include "llama.h"  // NOLINT

namespace allspark {
AsStatus LLaMAModel::Init(const TransformerProto& model_proto,
                          const DeviceContext& ctx) {
  DLOG(INFO) << "LLaMAModel::Init()" << std::endl;
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

REGISTER_MODEL("LLaMA", LLaMAModel)
REGISTER_MODEL("LLaMA_v2", LLaMA_v2Model)
REGISTER_MODEL("LLaMA_v3", LLaMA_v3Model)
}  // namespace allspark
