/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    qk_layernorm_nobeta_op.cpp
 */

#include "qk_layernorm_nobeta_op.h"  // NOLINT

#include <core/kernel/kernel.h>
#include <utility/datatype_dispatcher.h>
#ifdef ENABLE_CUDA
#include <cuda/cuda_context.h>
#endif
#include <cpu/cpu_context.h>
namespace allspark {

#ifdef ENABLE_CUDA
AsStatus gpu_qk_layernorm(DataType dtype, void* output, const void* input,
                          const void* gamma, void* output_points,
                          void* input_points, void* host_output_points,
                          void* host_input_points, int batch_size,
                          int num_heads, int hidden_size, int head_dim,
                          float eps, const DeviceContext* ctx) {
#ifdef CONFIG_DEBUG_OP
  DLOG(INFO) << "gpu_layernorm" << std::endl;
#endif
  const CUDAContext* gpu_ctx = static_cast<const CUDAContext*>(ctx);
  auto functor = [&]<typename T>() {
    // LOG(INFO) << "gpu_qk_layernorm start";
    // LOG(INFO) << "batch_size = " << batch_size << "num_heads = " << num_heads
    //           << "hidden_size = " << hidden_size << "head_dim = "<< head_dim;

    const T** typed_input_point = static_cast<const T**>(input_points);
    const T* typed_input = static_cast<const T*>(input);
    T** typed_output_point = static_cast<T**>(output_points);
    T* typed_output = static_cast<T*>(output);
    int m = batch_size * num_heads;
    T** h_input_point = (T**)host_input_points;
    T** h_output_point = (T**)host_output_points;

    int size_dtype = SizeofType(dtype);
    for (int batch = 0; batch < batch_size; batch++)
      for (int head = 0; head < num_heads; head++) {
        int now = batch * num_heads + head;
        int offset = batch * hidden_size + head * head_dim;

        // 直接使用指针算术
        h_input_point[now] = (T*)(typed_input + offset);
        h_output_point[now] = typed_output + offset;
      }
    const T* typed_gamma = static_cast<const T*>(gamma);
    const T* typed_bias = static_cast<const T*>(nullptr);
    // const T* typed_gamma = static_cast<const T*>(gamma);
    // const T* typed_beta = static_cast<const T*>(beta);
    cudaMemcpyAsync(typed_input_point, h_input_point, m * sizeof(T*),
                    cudaMemcpyHostToDevice, gpu_ctx->GetStream());
    cudaMemcpyAsync(typed_output_point, h_output_point, m * sizeof(T*),
                    cudaMemcpyHostToDevice, gpu_ctx->GetStream());
    // ctx->Synchronize();
    // LOG(INFO) << "before BatchLayerNormNoBetaKernelLauncher";
    cuda::BatchLayerNormNoBetaKernelLauncher(
        typed_output_point, typed_input_point, typed_bias, typed_gamma, m,
        head_dim, eps, gpu_ctx->GetStream());
  };
  DispatchCUDA(dtype, functor);
  return AsStatus::ALLSPARK_SUCCESS;
}
#endif
AsStatus QKLayerNormNoBetaOp::Init(const OperatorProto& op_proto,
                                   const DeviceContext& ctx,
                                   const TensorMap& weights_map,
                                   TensorMap* tensor_map) {
  AS_CHECK_STATUS(AsOperator::Init(op_proto, ctx, weights_map, tensor_map));
  // check weight
  if (weights_.size() != 2) {
    LOG(ERROR) << "QKLayerNormNoBetaOp has 2 weights [gamma]" << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
  head_dim_ = weights_[0]->GetShape()[0];
  // type inference
  DataType dtype = weights_[0]->GetDataType();
  tensor_map_->at(out_names_[0])->SetDataType(dtype);
  // attr
  auto& attr_map = op_proto.attr();
  if (attr_map.find("eps") == attr_map.end()) {
    LOG(ERROR) << "QKLayerNormNoBetaOp : can't find eps attribute."
               << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }

  eps_ = *(float*)(attr_map.at("eps").c_str());
  if (attr_map.find("num_heads") == attr_map.end()) {
    LOG(ERROR) << "QKLayerNormNoBetaOp : can't find num_heads attribute."
               << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
  num_heads_ = *(int*)(attr_map.at("num_heads").c_str());
  if (attr_map.find("multi_query_group_num") == attr_map.end()) {
    LOG(ERROR)
        << "QKLayerNormNoBetaOp : can't find multi_query_group_num attribute."
        << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
  multi_query_group_num_ =
      *(int*)(attr_map.at("multi_query_group_num").c_str());
  DeviceType backend = ctx.GetDeviceType();
  switch (backend) {
#ifdef ENABLE_CUDA
    case DeviceType::CUDA: {
      const CUDAContext* gpu_ctx = static_cast<const CUDAContext*>(ctx_);
      num_heads_ /= gpu_ctx->GetNranks();
      if (multi_query_group_num_ != 1) {
        multi_query_group_num_ /= gpu_ctx->GetNranks();
      }

      break;
    }
#endif
    // case DeviceType::CPU:
    //   kernel_launcher = cpu_qk_layernorm;
    //   break;
    default:
      LOG(ERROR) << "LayerNorm Operator does not support "
                 << DeviceType_Name(backend) << " device type" << std::endl;
      return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }
  input_points_ = std::make_unique<AsTensor>(
      "input_points_", backend, DataType::POINTER, DataMode::DENSE, Shape{1});
  output_points_ = std::make_unique<AsTensor>(
      "output_points_", backend, DataType::POINTER, DataMode::DENSE, Shape{1});
  q_host_input_points_ =
      std::make_unique<AsTensor>("q_host_input_points_", DeviceType::CPU,
                                 DataType::POINTER, DataMode::DENSE, Shape{1});
  q_host_output_points_ =
      std::make_unique<AsTensor>("q_host_output_points_", DeviceType::CPU,
                                 DataType::POINTER, DataMode::DENSE, Shape{1});
  k_host_input_points_ =
      std::make_unique<AsTensor>("k_host_input_points_", DeviceType::CPU,
                                 DataType::POINTER, DataMode::DENSE, Shape{1});
  k_host_output_points_ =
      std::make_unique<AsTensor>("k_host_output_points_", DeviceType::CPU,
                                 DataType::POINTER, DataMode::DENSE, Shape{1});
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus QKLayerNormNoBetaOp::Reshape() {
  Shape out_shape = tensor_map_->at(in_names_[0])->GetShape();
  AS_CHECK_STATUS(
      tensor_map_->at(out_names_[0])->SetShape(std::move(out_shape)));
  batch_size_ = out_shape[0] * out_shape[1];
  hidden_size_ = out_shape[2];
  // num_heads_ = hidden_size_ / 3 / head_dim_;
  input_points_->SetShape(Shape{batch_size_, num_heads_});
  output_points_->SetShape(Shape{batch_size_, num_heads_});
  q_host_input_points_->SetShape(Shape{batch_size_, num_heads_});
  q_host_output_points_->SetShape(Shape{batch_size_, num_heads_});
  k_host_input_points_->SetShape(Shape{batch_size_, multi_query_group_num_});
  k_host_output_points_->SetShape(Shape{batch_size_, multi_query_group_num_});
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus QKLayerNormNoBetaOp::Forward() {
  AsTensor* in_tensor = tensor_map_->at(in_names_[0]).get();
  void* in = in_tensor->GetDataPtr();
  void* out = tensor_map_->at(out_names_[0])->GetDataPtr();
  // void* bias = in_names_.size() == 2
  //                  ? tensor_map_->at(in_names_[1])->GetDataPtr()
  //                  : nullptr;
  // int64_t m = in_tensor->GetShape().Count() / hidden_size_;
  DeviceType backend = ctx_->GetDeviceType();
  DataType dtype = in_tensor->GetDataType();
  switch (backend) {
#ifdef ENABLE_CUDA
    case DeviceType::CUDA: {
      // kernel_launcher = gpu_qk_layernorm;
      const CUDAContext* gpu_ctx = static_cast<const CUDAContext*>(ctx_);
      cudaMemcpyAsync(out, in, batch_size_ * hidden_size_ * SizeofType(dtype),
                      cudaMemcpyDeviceToDevice, gpu_ctx->GetStream());
      int pos_bias = num_heads_ * head_dim_ * SizeofType(dtype);
      gpu_qk_layernorm(dtype, out, in, weights_[0]->GetDataPtr(),
                       output_points_->GetDataPtr(),
                       input_points_->GetDataPtr(),
                       q_host_output_points_->GetDataPtr(),
                       q_host_input_points_->GetDataPtr(), batch_size_,
                       num_heads_, hidden_size_, head_dim_, eps_, ctx_);
      gpu_qk_layernorm(
          dtype, out + pos_bias, in + pos_bias, weights_[1]->GetDataPtr(),
          output_points_->GetDataPtr(), input_points_->GetDataPtr(),
          k_host_output_points_->GetDataPtr(),
          k_host_input_points_->GetDataPtr(), batch_size_,
          multi_query_group_num_, hidden_size_, head_dim_, eps_, ctx_);
      break;
    }
#endif
    case DeviceType::CPU: {
      // kernel_launcher = cpu_qk_layernorm;
      break;
    }
    default:
      LOG(ERROR) << "LayerNorm Operator does not support "
                 << DeviceType_Name(backend) << " device type" << std::endl;
      return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }
  // PrintInformation();
  return AsStatus::ALLSPARK_SUCCESS;
}

REGISTER_OP(QKLayerNormNoBeta, CUDA, QKLayerNormNoBetaOp)
REGISTER_OP(QKLayerNormNoBeta, CPU, QKLayerNormNoBetaOp)
}  // namespace allspark
