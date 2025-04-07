/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    ALiBiPE_op.cpp
 */

#include "ALiBiPE_op.h"  // NOLINT

#include <core/kernel/kernel.h>
#include <utility/datatype_dispatcher.h>
#ifdef ENABLE_CUDA
#include <cuda/cuda_context.h>
#endif
#include <cpu/cpu_context.h>

namespace allspark {

#ifdef ENABLE_CUDA
AsStatus gpu_ALiBiPE(DataType dtype, void* out, int* batch_offset, int batch,
                     int seq_len, int num_heads, int ori_num_heads, int step,
                     const DeviceContext* ctx) {
  DLOG(INFO) << "gpu_ALiBiPE" << std::endl;
  const CUDAContext* gpu_ctx = static_cast<const CUDAContext*>(ctx);
  auto functor = [&]<typename T>() {
    T* typed_out = static_cast<T*>(out);
    cuda::ALiBiPEKernelLauncher(typed_out, batch_offset, batch, seq_len,
                                num_heads, ori_num_heads, step,
                                gpu_ctx->GetRank(), gpu_ctx->GetStream());
  };
  DispatchCUDA(dtype, functor);
  return AsStatus::ALLSPARK_SUCCESS;
}
#endif

AsStatus cpu_ALiBiPE(DataType dtype, void* out, int* batch_offset, int batch,
                     int seq_len, int num_heads, int ori_num_heads, int step,
                     const DeviceContext* ctx) {
  DLOG(INFO) << "cpu_ALiBiPE" << std::endl;
  const CPUContext* cpu_ctx = static_cast<const CPUContext*>(ctx);
  auto functor = [&]<typename T>() {
    T* typed_out = static_cast<T*>(out);
    cpu::ALiBiPEKernelLauncher(typed_out, batch_offset, batch, seq_len,
                               num_heads, ori_num_heads, step,
                               cpu_ctx->GetRank());
  };
  DispatchCPU(dtype, functor);
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus ALiBiPEOp::Init(const OperatorProto& op_proto,
                         const DeviceContext& ctx, const TensorMap& weights_map,
                         TensorMap* tensor_map) {
  AS_CHECK_STATUS(AsOperator::Init(op_proto, ctx, weights_map, tensor_map));
  auto& attr_map = op_proto.attr();
  if (attr_map.find("num_heads") == attr_map.end()) {
    LOG(ERROR) << "ALiBiPEOp : can't find num_heads attribute." << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
  num_heads_ = *(int*)(attr_map.at("num_heads").c_str());
  ori_num_heads_ = num_heads_;
  DataType dtype = ctx_->GetDtype();
  tensor_map_->at(out_names_[0])->SetDataType(dtype);
  DeviceType backend = ctx.GetDeviceType();
  switch (backend) {
#ifdef ENABLE_CUDA
    case DeviceType::CUDA: {
      kernel_launcher = gpu_ALiBiPE;
      const CUDAContext* gpu_ctx = static_cast<const CUDAContext*>(ctx_);
      num_heads_ /= gpu_ctx->GetNranks();
      break;
    }
#endif
    case DeviceType::CPU: {
      kernel_launcher = cpu_ALiBiPE;
      const CPUContext* cpu_ctx = static_cast<const CPUContext*>(ctx_);
      num_heads_ /= cpu_ctx->GetNranks();
      break;
    }
    default:
      LOG(ERROR) << "RelativePE Operator does not support "
                 << DeviceType_Name(backend) << " device type" << std::endl;
      return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus ALiBiPEOp::Reshape() {
  LOG(ERROR) << "NOT SUPPORT ALiBiPEOp in allspark2.x";
  return AsStatus::ALLSPARK_RUNTIME_ERROR;

#if 0
  Shape in_shape = tensor_map_->at(in_names_[0])->GetShape();
  batch_size_ = in_shape[0];
  if (gen_ctx_->step == 0) {
    seq_length_ = in_shape[1];
    Shape out_shape = Shape{batch_size_, seq_length_, num_heads_, seq_length_};
    AS_CHECK_STATUS(
        tensor_map_->at(out_names_[0])->SetShape(std::move(out_shape)));
  } else {
    seq_length_ = 1;
  }

  return AsStatus::ALLSPARK_SUCCESS;
#endif
}

AsStatus ALiBiPEOp::Forward() {
  LOG(ERROR) << "NOT SUPPORT ALiBiPEOp in allspark2.x";
  return AsStatus::ALLSPARK_RUNTIME_ERROR;

#if 0
  // int* batch_offset = (int*)tensor_map_->at(in_names_[1])->GetDataPtr();
  int* batch_offset = nullptr;
  AsTensor* out_tensor = tensor_map_->at(out_names_[0]).get();
  if (gen_ctx_->step != 0) {
    Shape out_shape = Shape{batch_size_, 1, num_heads_, gen_ctx_->step + 1};
    AS_CHECK_STATUS(
        tensor_map_->at(out_names_[0])->SetShape(std::move(out_shape)));
  }
  kernel_launcher(out_tensor->GetDataType(), out_tensor->GetDataPtr(),
                  batch_offset, batch_size_, seq_length_, num_heads_,
                  ori_num_heads_, (gen_ctx_->step + 1), ctx_);
  return AsStatus::ALLSPARK_SUCCESS;
#endif
}

REGISTER_OP(ALiBiPE, CPU, ALiBiPEOp)
REGISTER_OP(ALiBiPE, CUDA, ALiBiPEOp)
}  // namespace allspark
