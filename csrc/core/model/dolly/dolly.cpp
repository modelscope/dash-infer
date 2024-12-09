/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    dolly.cpp
 */

#include "dolly.h"  // NOLINT

namespace allspark {
AsStatus DollyModel::Init(const TransformerProto& model_proto,
                          const DeviceContext& ctx) {
  DLOG(INFO) << "DollyModel::Init()" << std::endl;
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

REGISTER_MODEL("Dolly_v2", DollyModel)
}  // namespace allspark
