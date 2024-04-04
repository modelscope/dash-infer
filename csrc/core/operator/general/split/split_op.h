/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    split_op.h
 */

#pragma once

#include <core/operator/operator.h>
namespace allspark {

class WeightSplitter;

class SplitOp : public AsOperator {
 public:
  explicit SplitOp(const std::string& op_type = "") : AsOperator(op_type) {}
  AsStatus Init(const OperatorProto& op_proto, const DeviceContext& ctx,
                const TensorMap& weights_map, TensorMap* tensor_map);
  AsStatus Reshape() override;
  AsStatus Forward() override;

 private:
  SplitMode split_type_ = NOSPLIT;
  std::unique_ptr<WeightSplitter> splitter_;
  RankInfo rank_info_ = RankInfo(1, 1);
};

}  // namespace allspark
