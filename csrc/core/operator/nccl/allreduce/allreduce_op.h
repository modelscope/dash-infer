/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    allreduce_op.h
 */

#pragma once

#include <core/operator/operator.h>
#include <mpi.h>

namespace allspark {
class AllReduceOp : public AsOperator {
 public:
  explicit AllReduceOp(const std::string& op_type = "")
      : AsOperator(op_type),
        nranks_(1),
        rank_id_(0),
        mpi_dtype_(MPI_FLOAT),
        count_(0) {}

  AsStatus Init(const OperatorProto& op_proto, const DeviceContext& ctx,
                const TensorMap& weights_map, TensorMap* tensor_map);
  AsStatus Reshape() override;
  AsStatus Forward() override;
  AsStatus Reshape(RuntimeContext* runtime_ctx) override {
    return this->Reshape();
  }
  AsStatus Forward(RuntimeContext* runtime_ctx) override {
    return this->Forward();
  }

 private:
  size_t nranks_;
  size_t rank_id_;
  MPI_Datatype mpi_dtype_;

  size_t count_;
};
}  // namespace allspark
