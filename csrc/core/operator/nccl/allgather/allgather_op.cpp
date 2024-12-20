/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    allgather_op.cpp
 */

#include "allgather_op.h"  // NOLINT

#include <check_cuda.h>
#ifdef ENABLE_CUDA
#include <check_cuda.h>
#include <cuda/cuda_context.h>

#include <cuda/nccl_utils.hpp>
#endif
#include <coodinator/worker_coodinator.h>
#include <core/kernel/kernel.h>
#include <cpu/cpu_context.h>
#include <utility/datatype_dispatcher.h>

#ifdef ENABLE_MULTINUMA
#include "cpu/mpi_utils.hpp"
#endif

namespace allspark {

#ifdef ENABLE_CUDA
void nccl_allgather_launcher(DataType dtype, void* out, void* in,
                             void* tmp_data, int count, int batch_size,
                             int hidden_size, int nranks,
                             const DeviceContext* ctx) {
#ifdef CONFIG_DEBUG_OP
  DLOG(INFO) << "nccl_allgather_launcher" << std::endl;
#endif
  const CUDAContext* cuda_ctx = static_cast<const CUDAContext*>(ctx);

  if (nranks > 1) {
    ncclDataType_t nccl_dtype = GetNcclType(dtype);
    // sync to fix hang issue in some devices, no impact on performance
    cuda_ctx->Synchronize();
    AS_CHECK_NCCL(ncclAllGather(in, tmp_data, count, nccl_dtype,
                                cuda_ctx->GetNCCLComm(),
                                cuda_ctx->GetStream()));
    auto functor = [&]<typename T>() {
      T* typed_in = static_cast<T*>(tmp_data);
      T* typed_out = static_cast<T*>(out);
      cuda::transpose_axis_01_kernelLauncher(typed_out, typed_in, nranks,
                                             batch_size, hidden_size,
                                             cuda_ctx->GetStream());
    };
    DispatchCUDA(dtype, functor);
    // sync to fix hang issue in some devices, no impact on performance
    cuda_ctx->Synchronize();
  } else {
    AS_CHECK_CUDA(cudaMemcpyAsync(out, in, count * SizeofType(dtype),
                                  cudaMemcpyDeviceToDevice,
                                  cuda_ctx->GetStream()));
  }
}
#endif

void mpi_allgather_launcher(DataType dtype, void* out, void* in, void* tmp_data,
                            int count, int batch_size, int hidden_size,
                            int nranks, const DeviceContext* ctx) {
#ifdef ENABLE_MULTINUMA
  DLOG(INFO) << "mpi_allgather_launcher" << std::endl;
  MPI_Datatype mpi_dtype = GetMpiType(dtype);

  if (nranks > 1) {
    if (in == tmp_data) {
      // in place allgather
      MPI_Allgather(MPI_IN_PLACE, count, mpi_dtype, tmp_data, count, mpi_dtype,
                    MPI_COMM_WORLD);
    } else {
      MPI_Allgather(in, count, mpi_dtype, tmp_data, count, mpi_dtype,
                    MPI_COMM_WORLD);
    }
    auto functor = [&]<typename T>() {
      T* typed_in = static_cast<T*>(tmp_data);
      T* typed_out = static_cast<T*>(out);
      cpu::TransposeAxis01KernelLauncher(typed_out, typed_in, nranks,
                                         batch_size, hidden_size);
    };
    DispatchCPU(dtype, functor);
  } else {
    memcpy(out, in, count * SizeofType(dtype));
  }
#else
  LOG(ERROR) << "Multi-NUMA codes are not compiled" << std::endl;
  AS_THROW(AsStatus::ALLSPARK_RUNTIME_ERROR);
#endif
}

AsStatus AllGatherOp::Init(const OperatorProto& op_proto,
                           const DeviceContext& ctx,
                           const TensorMap& weights_map,
                           TensorMap* tensor_map) {
  AS_CHECK_STATUS(AsOperator::Init(op_proto, ctx, weights_map, tensor_map));
  DataType dtype = tensor_map_->at(in_names_[0])->GetDataType();
  DeviceType backend = ctx.GetDeviceType();
  switch (backend) {
#ifdef ENABLE_CUDA
    case DeviceType::CUDA: {
      kernel_launcher = nccl_allgather_launcher;
      const CUDAContext* gpu_ctx = static_cast<const CUDAContext*>(ctx_);
      nranks_ = gpu_ctx->GetNranks();
      rank_id_ = gpu_ctx->GetRank();
      break;
    }
#endif
    case DeviceType::CPU: {
#ifdef ENABLE_MULTINUMA
      kernel_launcher = mpi_allgather_launcher;
      const CPUContext* cpu_ctx = static_cast<const CPUContext*>(ctx_);
      nranks_ = cpu_ctx->GetNranks();
      rank_id_ = cpu_ctx->GetRank();

      if (dtype != DataType::FLOAT32) {
        LOG(ERROR) << op_type_
                   << " not supported in DataType:" << DataType_Name(dtype)
                   << std::endl;
        return AsStatus::ALLSPARK_RUNTIME_ERROR;
      }
#else
      LOG(ERROR) << "Multi-NUMA codes are not compiled" << std::endl;
      return AsStatus::ALLSPARK_RUNTIME_ERROR;
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
AsStatus AllGatherOp::Reshape() {
  Shape in_shape = tensor_map_->at(in_names_[0])->GetShape();
  m_ = in_shape.Count(0, in_shape.Size() - 1);
  n_ = in_shape[in_shape.Size() - 1];
  count_ = in_shape.Count();

  Shape out_shape(in_shape);
  out_shape[out_shape.Size() - 1] *= nranks_;

  DataType dtype = tensor_map_->at(in_names_[0])->GetDataType();

  int64_t size = out_shape.Count() * SizeofType(dtype);
  AS_CHECK_STATUS(
      tensor_map_->at(out_names_[0])->SetShape(std::move(out_shape)));

  AS_CHECK_STATUS(tensor_map_->at("workspace")->SetShape(Shape{size}));
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus AllGatherOp::Forward() {
  void* in = tensor_map_->at(in_names_[0])->GetDataPtr();
  void* out = tensor_map_->at(out_names_[0])->GetDataPtr();
  void* tmp_data = tensor_map_->at("workspace")->GetDataPtr();
  DataType dtype = tensor_map_->at(in_names_[0])->GetDataType();

  auto coodinator = WorkerCoodinator(nranks_, rank_id_,
                                     WorkerCoodinator::GetDefaultTimeout());

  int ret = coodinator.StateSyncWithTimeout();

  if (ret == 0) {
    kernel_launcher(dtype, out, in, tmp_data, count_, m_, n_, nranks_, ctx_);
    coodinator.ResetCounter();
    return AsStatus::ALLSPARK_SUCCESS;
  } else {
    LOG(ERROR) << "AllGather: Sync state timeout, something wrong..." << ret;
    coodinator.ResetCounter();
    return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }
}

REGISTER_OP(AllGather, CUDA, AllGatherOp)
REGISTER_OP(AllGather, CPU, AllGatherOp)
}  // namespace allspark
