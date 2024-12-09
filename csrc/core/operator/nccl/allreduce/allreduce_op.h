/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    allreduce_op.h
 */

#pragma once

#include <core/operator/operator.h>
#ifdef ENABLE_CUDA
#include <nccl.h>
#endif

#ifdef ENABLE_MULTINUMA
#include <mpi.h>
#endif

namespace allspark {

class AllReduceOp : public AsOperator {
 public:
  explicit AllReduceOp(const std::string& op_type = "")
      : AsOperator(op_type), count_(0) {}
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
#ifdef ENABLE_CUDA
  ncclDataType_t nccl_dtype_ = ncclFloat32;
#endif
#ifdef ENABLE_MULTINUMA
  MPI_Datatype mpi_dtype_ = MPI_FLOAT;
#endif

  size_t count_;
};
}  // namespace allspark
