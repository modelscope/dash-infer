/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    allgather_op.h
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

class AllGatherOp : public AsOperator {
 public:
  explicit AllGatherOp(const std::string& op_type = "")
      : AsOperator(op_type), count_(0), m_(1), n_(1), nranks_(1) {}
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
  size_t count_;
  int m_;
  int n_;
  void (*kernel_launcher)(DataType dtype, void* out, void* in, void* tmp_data,
                          int count, int batch_size, int hidden_size,
                          int nranks, const DeviceContext* ctx) = nullptr;
};
}  // namespace allspark
