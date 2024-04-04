/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    allreduce_op.cpp
 */

#include "allreduce_op.h"  // NOLINT

#include <coodinator/worker_coodinator.h>
#include <cpu/cpu_context.h>

#include "cpu/mpi_utils.hpp"

namespace allspark {

AsStatus AllReduceOp::Init(const OperatorProto& op_proto,
                           const DeviceContext& ctx,
                           const TensorMap& weights_map,
                           TensorMap* tensor_map) {
  AS_CHECK_STATUS(AsOperator::Init(op_proto, ctx, weights_map, tensor_map));
  DataType dtype = tensor_map_->at(in_names_[0])->GetDataType();
  DeviceType backend = ctx.GetDeviceType();
  switch (backend) {
    case DeviceType::CPU: {
      const CPUContext* cpu_ctx = static_cast<const CPUContext*>(ctx_);
      nranks_ = cpu_ctx->GetNranks();
      rank_id_ = cpu_ctx->GetRank();

      if (dtype != DataType::FLOAT32) {
        LOG(ERROR) << op_type_
                   << " not supported in DataType:" << DataType_Name(dtype)
                   << std::endl;
        return AsStatus::ALLSPARK_RUNTIME_ERROR;
      }
      mpi_dtype_ = GetMpiType(dtype);
      break;
    }
    default:
      LOG(ERROR) << "AllReduce Operator does not support "
                 << DeviceType_Name(backend) << " device type" << std::endl;
      return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }
  tensor_map_->at(out_names_[0])->SetDataType(dtype);
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus AllReduceOp::Reshape() {
  Shape out_shape = tensor_map_->at(in_names_[0])->GetShape();
  count_ = out_shape.Count();
  AS_CHECK_STATUS(
      tensor_map_->at(out_names_[0])->SetShape(std::move(out_shape)));
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus AllReduceOp::Forward() {
  void* in = tensor_map_->at(in_names_[0])->GetDataPtr();
  void* out = tensor_map_->at(out_names_[0])->GetDataPtr();
  DeviceType backend = ctx_->GetDeviceType();

  auto coodinator = WorkerCoodinator(nranks_, rank_id_,
                                     WorkerCoodinator::GetDefaultTimeout());

  int ret = coodinator.StateSyncWithTimeout();
  if (ret) {
    LOG(ERROR) << "AllReduce: Sync state timeout, something wrong..." << ret;
    coodinator.ResetCounter();
    return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }

  if (nranks_ == 1) {
    return AsStatus::ALLSPARK_SUCCESS;
  }

  switch (backend) {
    case DeviceType::CPU:
      if (in == out) {
        // in place all reduce
        MPI_Allreduce(MPI_IN_PLACE, out, count_, mpi_dtype_, MPI_SUM,
                      MPI_COMM_WORLD);
      } else {
        MPI_Allreduce(in, out, count_, mpi_dtype_, MPI_SUM, MPI_COMM_WORLD);
      }
      break;
    default:
      LOG(ERROR) << "AllReduce Operator does not support "
                 << DeviceType_Name(backend) << " device type" << std::endl;
      return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }

  coodinator.ResetCounter();

  return AsStatus::ALLSPARK_SUCCESS;
}

REGISTER_OP("AllReduce", CPU, AllReduceOp)
}  // namespace allspark
