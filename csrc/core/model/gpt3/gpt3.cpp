/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    gpt3.cpp
 */

#include "gpt3.h"  // NOLINT

namespace allspark {
AsStatus GPT3Model::Init(const TransformerProto& model_proto,
                         const DeviceContext& ctx) {
  DLOG(INFO) << "GPT3Model::Init()" << std::endl;
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
REGISTER_MODEL("GPT3", GPT3Model)
}  // namespace allspark
