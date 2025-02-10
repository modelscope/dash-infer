/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    allreduce_op.cpp
 */

#include "allreduce_op.h"  // NOLINT
#ifdef ENABLE_CUDA
#include <check_cuda.h>
#include <cuda/cuda_context.h>

#include <cuda/nccl_utils.hpp>
#endif
#include <coodinator/worker_coodinator.h>
#include <cpu/cpu_context.h>

#ifdef ENABLE_MULTINUMA
#include "cpu/mpi_utils.hpp"
#endif

namespace allspark {

AsStatus AllReduceOp::Init(const OperatorProto& op_proto,
                           const DeviceContext& ctx,
                           const TensorMap& weights_map,
                           TensorMap* tensor_map) {
  AS_CHECK_STATUS(AsOperator::Init(op_proto, ctx, weights_map, tensor_map));
  DataType dtype = tensor_map_->at(in_names_[0])->GetDataType();
  DeviceType backend = ctx.GetDeviceType();
  switch (backend) {
#ifdef ENABLE_CUDA
    case DeviceType::CUDA: {
      const CUDAContext* gpu_ctx = static_cast<const CUDAContext*>(ctx_);
      nranks_ = gpu_ctx->GetNranks();
      rank_id_ = gpu_ctx->GetRank();
      nccl_dtype_ = GetNcclType(dtype);
      break;
    }
#endif
    case DeviceType::CPU: {
#ifdef ENABLE_MULTINUMA
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
#else
      // Single CPU support is not require any setting.
#endif
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

  if (nranks_ == 1) {
    return AsStatus::ALLSPARK_SUCCESS;
  }

  auto coodinator = WorkerCoodinator(nranks_, rank_id_,
                                     WorkerCoodinator::GetDefaultTimeout());

  int ret = coodinator.StateSyncWithTimeout();
  if (ret) {
    LOG(ERROR) << "AllReduce: Sync state timeout, something wrong..." << ret;
    coodinator.ResetCounter();
    return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }

  switch (backend) {
#ifdef ENABLE_CUDA
    case DeviceType::CUDA: {
      const CUDAContext* cuda_ctx = static_cast<const CUDAContext*>(ctx_);
      // add sync test, if there is some cuda error, abort here.
      AS_CHECK_NCCL(ncclAllReduce(in, out, count_, nccl_dtype_, ncclSum,
                                  cuda_ctx->GetNCCLComm(),
                                  cuda_ctx->GetStream()));
      break;
    }
#endif
    case DeviceType::CPU:
#ifdef ENABLE_MULTINUMA
      if (in == out) {
        // in place all reduce
        MPI_Allreduce(MPI_IN_PLACE, out, count_, mpi_dtype_, MPI_SUM,
                      MPI_COMM_WORLD);
      } else {
        MPI_Allreduce(in, out, count_, mpi_dtype_, MPI_SUM, MPI_COMM_WORLD);
      }
#else
      // single numa cpu all reduce don't require operation.
      return AsStatus::ALLSPARK_SUCCESS;
#endif
      break;
    default:
      LOG(ERROR) << "AllReduce Operator does not support "
                 << DeviceType_Name(backend) << " device type" << std::endl;
      return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }

  coodinator.ResetCounter();

  return AsStatus::ALLSPARK_SUCCESS;
}

REGISTER_OP(AllReduce, CUDA, AllReduceOp)
REGISTER_OP(AllReduce, CPU, AllReduceOp)
}  // namespace allspark
