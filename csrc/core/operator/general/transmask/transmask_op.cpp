/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    transmask_op.cpp
 */
#include "transmask_op.h"  // NOLINT

#include <core/kernel/kernel.h>
#include <utility/datatype_dispatcher.h>
#ifdef ENABLE_CUDA
#include <cuda/cuda_context.h>
#endif
#include <cpu/cpu_context.h>

namespace allspark {

#ifdef ENABLE_CUDA
AsStatus gpu_transmask(DataType dtype, void* out, const int64_t* in, int batch,
                       int seq_len, bool seq_mask, bool blank,
                       const DeviceContext* ctx) {
  DLOG(INFO) << "gpu_transmask" << std::endl;
  const CUDAContext* gpu_ctx = static_cast<const CUDAContext*>(ctx);
  auto functor = [&]<typename T>() {
    T* typed_out = static_cast<T*>(out);
    cuda::TransMaskKernelLauncher(typed_out, in, batch, seq_len, seq_mask,
                                  blank, gpu_ctx->GetStream());
  };
  DispatchCUDA(dtype, functor);
  return AsStatus::ALLSPARK_SUCCESS;
}

#endif

AsStatus cpu_transmask(DataType dtype, void* out, const int64_t* in, int batch,
                       int seq_len, bool seq_mask, bool blank,
                       const DeviceContext* ctx) {
  DLOG(INFO) << "cpu_transmask" << std::endl;
  const CPUContext* cpu_ctx = static_cast<const CPUContext*>(ctx);
  auto functor = [&]<typename T>() {
    T* typed_out = static_cast<T*>(out);
    cpu::TransMaskKernel(typed_out, in, batch, seq_len, seq_mask, blank);
  };
  DispatchCPU(dtype, functor);
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus TransMaskOp::Init(const OperatorProto& op_proto,
                           const DeviceContext& ctx,
                           const TensorMap& weights_map,
                           TensorMap* tensor_map) {
  AS_CHECK_STATUS(AsOperator::Init(op_proto, ctx, weights_map, tensor_map));
  auto& attr_map = op_proto.attr();
  if (attr_map.find("sequence_mask") != attr_map.end()) {
    seq_mask_ = *(bool*)(attr_map.at("sequence_mask").c_str());
  }
  if (attr_map.find("blank") != attr_map.end()) {
    blank_ = *(bool*)(attr_map.at("blank").c_str());
  }
  if (seq_mask_ && blank_) {
    LOG(ERROR) << "TransMask Operator does not support (seq_mask + blank) type"
               << std::endl;
    return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }
  tensor_map_->at(out_names_[0])->SetDataType(DataType::FLOAT32);
  if (out_names_.size() > 1) {
    tensor_map_->at(out_names_[1])->SetDataType(DataType::INT32);
  }
  DeviceType backend = ctx.GetDeviceType();
  switch (backend) {
#ifdef ENABLE_CUDA
    case DeviceType::CUDA:
      kernel_launcher = gpu_transmask;

      // TODO(zhangyufei): this is a fastfix, please remove this in future ..
      int device_id;
      cudaGetDevice(&device_id);
      cudaGetDeviceProperties(&dprop_, device_id);
      break;
#endif
    case DeviceType::CPU:
      kernel_launcher = cpu_transmask;
      break;
    default:
      LOG(ERROR) << "TransMask Operator does not support "
                 << DeviceType_Name(backend) << " device type" << std::endl;
      return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }

  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus TransMaskOp::Reshape() {
  std::pair<bool, AsMHAPrefill> prefill_mode_pair = GetPrefillMode();
  if (!prefill_mode_pair.first) {
    LOG(ERROR) << "TransMaskOp get prefill mode error. " << std::endl;
    return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }

#ifdef ENABLE_CUDA
  if (ctx_->GetDeviceType() == DeviceType::CUDA) {
    if (prefill_mode_pair.second == allspark::AsMHAPrefill::AsPrefillFlashV2 ||
        prefill_mode_pair.second == allspark::AsMHAPrefill::AsPrefillXformer) {
      return AsStatus::ALLSPARK_SUCCESS;
    }
  }
#endif  // ENABLE_CUDA
  Shape out_shape = tensor_map_->at(in_names_[0])->GetShape();
  batch_size_ = out_shape[0];
  seq_length_ = out_shape[1];
  out_shape.Append(seq_length_);
  AS_CHECK_STATUS(
      tensor_map_->at(out_names_[0])->SetShape(std::move(out_shape)));
  if (out_names_.size() > 1) {
    AS_CHECK_STATUS(
        tensor_map_->at(out_names_[1])->SetShape(Shape{batch_size_}));
  }
  AsTensor* out_tensor = tensor_map_->at(out_names_[0]).get();
  DLOG(INFO) << "out_tensor type: " << (int)out_tensor->GetDataType()
             << " shape: " << out_tensor->GetShape().ToString() << std::endl;
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus TransMaskOp::Forward() {
  std::pair<bool, AsMHAPrefill> prefill_mode_pair = GetPrefillMode();
  if (!prefill_mode_pair.first) {
    LOG(ERROR) << "TransMaskOp get prefill mode error. " << std::endl;
    return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }

#ifdef ENABLE_CUDA
  if (ctx_->GetDeviceType() == DeviceType::CUDA) {
    if (prefill_mode_pair.second == allspark::AsMHAPrefill::AsPrefillFlashV2 ||
        prefill_mode_pair.second == allspark::AsMHAPrefill::AsPrefillXformer) {
      return AsStatus::ALLSPARK_SUCCESS;
    }
  }
#endif  // ENABLE_CUDA
  AsTensor* out_tensor = tensor_map_->at(out_names_[0]).get();
  // use nullptr as mask,default all mask = 1
  kernel_launcher(out_tensor->GetDataType(), out_tensor->GetDataPtr(), nullptr,
                  batch_size_, seq_length_, seq_mask_, blank_, ctx_);
  return AsStatus::ALLSPARK_SUCCESS;
}

REGISTER_OP(TransMask, CUDA, TransMaskOp)
REGISTER_OP(TransMask, CPU, TransMaskOp)
}  // namespace allspark
