/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    operator.cpp
 */

#include "operator.h"  // NOLINT

namespace allspark {
AsStatus OperatorModel::Init(const TransformerProto& model_proto,
                             const DeviceContext& ctx) {
  AS_CHECK_STATUS(AsModel::Init(model_proto, ctx));
  // parse graph
  for (auto& op : graph_ops_["operator"]) {
    topo_ops_.emplace_back(op.get());
  }
  return AsStatus::ALLSPARK_SUCCESS;
}

REGISTER_MODEL("Operator", OperatorModel)
}  // namespace allspark
