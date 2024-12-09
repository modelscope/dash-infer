/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    opt.cpp
 */

#include "opt.h"  // NOLINT

namespace allspark {
AsStatus OPTModel::Init(const TransformerProto& model_proto,
                        const DeviceContext& ctx) {
  DLOG(INFO) << "OPTModel::Init()" << std::endl;
  AS_CHECK_STATUS(AsModel::Init(model_proto, ctx));
  // parse graph
  for (auto& op : graph_ops_["decoder"]) {
    topo_ops_.emplace_back(op.get());
  }
  for (auto& op : graph_ops_["gen_graph"]) {
    topo_ops_.emplace_back(op.get());
  }
  return AsStatus::ALLSPARK_SUCCESS;
}
REGISTER_MODEL("OPT", OPTModel)
}  // namespace allspark
