/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    starcoder.cpp
 */

#include "starcoder.h"  // NOLINT

namespace allspark {
AsStatus StarCoderModel::Init(const TransformerProto& model_proto,
                              const DeviceContext& ctx) {
  DLOG(INFO) << "StarCoderModel::Init()" << std::endl;
  AS_CHECK_STATUS(AsModel::Init(model_proto, ctx));
  topo_ops_.clear();
  // parse graph
  for (auto& op : graph_ops_["decoder"]) {
    topo_ops_.emplace_back(op.get());
  }
  if (model_proto.model_conf().is_generate())
    for (auto& op : graph_ops_["gen_graph"]) {
      topo_ops_.emplace_back(op.get());
    }
  return AsStatus::ALLSPARK_SUCCESS;
}

REGISTER_MODEL("StarCoder", StarCoderModel)
}  // namespace allspark
