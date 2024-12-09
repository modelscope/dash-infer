/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    moe_op.cpp
 */

#include "moe_op.h"  // NOLINT

#include <core/kernel/kernel.h>
#include <utility/datatype_dispatcher.h>
#ifdef ENABLE_CUDA
#include <cuda/cuda_context.h>
#endif
#include <cpu/cpu_context.h>

namespace allspark {
AsStatus MoeOp::Init(const OperatorProto& op_proto, const DeviceContext& ctx,
                     const TensorMap& weights_map, TensorMap* tensor_map) {
  AS_CHECK_STATUS(AsOperator::Init(op_proto, ctx, weights_map, tensor_map));
  // type inference
  dtype_ = tensor_map_->at(in_names_[0])->GetDataType();
  DeviceType backend = ctx.GetDeviceType();
  tensor_map_->at(out_names_[0])->SetDataType(dtype_);
  // attr
  auto& attr_map = op_proto.attr();
  if (attr_map.find("num_experts") == attr_map.end()) {
    LOG(ERROR) << "MoeOp : can't find num_expert attribute." << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
  num_expert_ = *(int*)(attr_map.at("num_experts").c_str());
  if (attr_map.find("num_experts_per_tok") == attr_map.end()) {
    LOG(ERROR) << "MoeOp : can't find num_expert_per_tok attribute."
               << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
  num_expert_pertoken_ = *(int*)(attr_map.at("num_experts_per_tok").c_str());
  first_moe_ = true;
  // default
  float_gate_score_ = std::make_unique<AsTensor>(
      "topk_value_", backend, DataType::FLOAT32, DataMode::DENSE,
      Shape{ctx_->GetModelMaxLength(), num_expert_});
  topk_value_ = std::make_unique<AsTensor>(
      "topk_value_", backend, DataType::FLOAT32, DataMode::DENSE,
      Shape{ctx_->GetModelMaxLength(), num_expert_});
  experts_score_ = std::make_unique<AsTensor>(
      "experts_score_", backend, DataType::FLOAT32, DataMode::DENSE,
      Shape{ctx_->GetModelMaxLength(), num_expert_pertoken_});
  topk_indice_ = std::make_unique<AsTensor>(
      "topk_indice_", backend, DataType::INT32, DataMode::DENSE,
      Shape{ctx_->GetModelMaxLength(), num_expert_pertoken_});
  mid_row_indices_ = std::make_unique<AsTensor>(
      "mid_row_indices_", backend, DataType::INT32, DataMode::DENSE,
      Shape{ctx_->GetModelMaxLength(), num_expert_pertoken_});
  mid_expert_indices_ = std::make_unique<AsTensor>(
      "mid_expert_indices_", backend, DataType::INT32, DataMode::DENSE,
      Shape{ctx_->GetModelMaxLength(), num_expert_pertoken_});
  final_row_indices_ = std::make_unique<AsTensor>(
      "final_row_indices_", backend, DataType::INT32, DataMode::DENSE,
      Shape{ctx_->GetModelMaxLength(), num_expert_pertoken_});
  hidden_size_ = weights_[0]->GetShape()[1];
  proj_size_ = weights_[0]->GetShape()[2] / 2;
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus MoeOp::Reshape() {
  Shape out_shape = tensor_map_->at(in_names_[0])->GetShape();
  AS_CHECK_STATUS(
      tensor_map_->at(out_names_[0])->SetShape(std::move(out_shape)));

  if (first_moe_) {
    // only reshape once,for warmup
    first_moe_ = false;
    total_token_ = ctx_->GetModelMaxLength();
    int64_t max_total_tokens = total_token_ * num_expert_pertoken_;
    expert_size_ = (int64_t)hidden_size_ * proj_size_;
    float_gate_score_->SetShape(Shape{total_token_, num_expert_});
    topk_value_->SetShape(Shape{total_token_, num_expert_});
    experts_score_->SetShape(Shape{total_token_, num_expert_pertoken_});
    topk_indice_->SetShape(Shape{total_token_, num_expert_pertoken_});
    ws_size_ = 0;

#ifdef ENABLE_CUDA
    if (ctx_->GetDeviceType() == DeviceType::CUDA) {
      size_t softmax_workspace = 0;
      cuda::StridedSoftmaxGetWorkspaceSize<float>(
          &softmax_workspace, ctx_->GetModelMaxLength(), num_expert_);
      AS_CHECK_STATUS(
          tensor_map_->at("workspace")
              ->SetShape(Shape{static_cast<dim_t>(softmax_workspace)}));
      ws_size_ += softmax_workspace;
    }
#endif
    ws_size_ += max_total_tokens * proj_size_ * 2 *
                SizeofType(dtype_);  // up_gate_proj_out
    ws_size_ +=
        max_total_tokens * proj_size_ * SizeofType(dtype_);  // mid_result
    ws_size_ +=
        max_total_tokens * hidden_size_ * SizeofType(dtype_);  // final_result
    AS_CHECK_STATUS(tensor_map_->at("workspace")->SetShape(Shape{(ws_size_)}));
    AsTensor* in_tensor = tensor_map_->at(in_names_[0]).get();
    AsTensor* expert_weight_tensor = tensor_map_->at(in_names_[1]).get();
    AsTensor* gate_up_proj_weight_tensor = weights_[0];
    AsTensor* down_proj_weight_tensor = weights_[1];
    AsTensor* out_tensor = tensor_map_->at(out_names_[0]).get();

    switch (ctx_->GetDeviceType()) {
#ifdef ENABLE_CUDA
      case DeviceType::CUDA: {
        int top_k = num_expert_pertoken_;
        const CUDAContext* gpu_ctx = static_cast<const CUDAContext*>(ctx_);
        cublasHandle_t cublas_handle = gpu_ctx->GetCublasHandle();
        cudaStream_t cu_stream =
            static_cast<const CUDAContext*>(ctx_)->GetStream();
        cuda::GetWorkspaceSize(&hWsSize, &dWsSize, total_token_, num_expert_);
        cudaMallocHost(&hWs, hWsSize);
        AS_CHECK_STATUS(
            tensor_map_->at("workspace")->SetShape(Shape{(dWsSize)}));
        auto functor = [&]<typename T>() {
          void* ws_ptr = tensor_map_->at("workspace")->GetDataPtr();
          reorder_data = ws_ptr;
          gate_up_proj_out = (char*)reorder_data + max_total_tokens *
                                                       hidden_size_ *
                                                       SizeofType(dtype_);
          mid_result = (char*)gate_up_proj_out +
                       max_total_tokens * proj_size_ * 2 * SizeofType(dtype_);
          final_result = (char*)mid_result +
                         max_total_tokens * proj_size_ * SizeofType(dtype_);
        };
        DispatchCUDA(dtype_, functor);
        break;
      }
#endif
      case DeviceType::CPU: {
        LOG(ERROR) << "MOE Operator does not support "
                   << "CPU"
                   << " device type" << std::endl;
        return AsStatus::ALLSPARK_RUNTIME_ERROR;
      }
      default:
        break;
    }
  }
  total_token_ = out_shape[0] * out_shape[1];
  topk_indice_->SetShape(Shape{total_token_, num_expert_pertoken_});
  mid_row_indices_->SetShape(Shape{total_token_, num_expert_pertoken_});
  mid_expert_indices_->SetShape(Shape{total_token_ * num_expert_pertoken_, 1});
  final_row_indices_->SetShape(Shape{total_token_ * num_expert_pertoken_, 1});
  return AsStatus::ALLSPARK_SUCCESS;
}
#if 0
// dubug code
static void print_info(void* input, const DeviceContext* ctx,
                       size_t layout_size = 0) {
  const int print_count = 10;
  cudaStream_t cu_stream = static_cast<const CUDAContext*>(ctx)->GetStream();
  std::vector<char> host_out(print_count);
  cudaMemcpyAsync(host_out.data(), input, print_count, cudaMemcpyDeviceToHost,
                  cu_stream);
  ctx->Synchronize();
  void* data_ptr = host_out.data();
  half* ptr = static_cast<half*>(data_ptr);
  for (int i = 0; i < print_count; i++) {
    LOG(INFO) << (float)(ptr[i]) << ",";
  }
  LOG(INFO) << std::endl;
}
#endif
AsStatus MoeOp::Forward() {
  AsTensor* in_tensor = tensor_map_->at(in_names_[0]).get();
  AsTensor* expert_weight_tensor = tensor_map_->at(in_names_[1]).get();
  AsTensor* gate_up_proj_weight_tensor = weights_[0];
  AsTensor* down_proj_weight_tensor = weights_[1];
  AsTensor* out_tensor = tensor_map_->at(out_names_[0]).get();
  void* ws_ptr = tensor_map_->at("workspace")->GetDataPtr();
  switch (ctx_->GetDeviceType()) {
#ifdef ENABLE_CUDA
    case DeviceType::CUDA: {
      int top_k = num_expert_pertoken_;
      const CUDAContext* gpu_ctx = static_cast<const CUDAContext*>(ctx_);
      cublasHandle_t cublas_handle = gpu_ctx->GetCublasHandle();
      cudaStream_t cu_stream =
          static_cast<const CUDAContext*>(ctx_)->GetStream();
      auto functor = [&]<typename T>() {
        cuda::CastKernelLauncher((T*)expert_weight_tensor->GetDataPtr(),
                                 (float*)float_gate_score_->GetDataPtr(),
                                 expert_weight_tensor->GetShape().Count(),
                                 cu_stream);
        // cuda::StridedSoftmaxLauncher((float*)topk_value_->GetDataPtr(),
        //                              (float*)float_gate_score_->GetDataPtr(),
        //                              nullptr, nullptr, ws_ptr, ws_size_,
        //                              total_token_, num_expert_, cu_stream);
        cuda::SoftmaxLowReduceKernelLauncher(
            (float*)float_gate_score_->GetDataPtr(),
            (float*)topk_value_->GetDataPtr(), total_token_, num_expert_,
            cu_stream);
        cuda::TopKKernelLauncher((float*)experts_score_->GetDataPtr(),
                                 (int*)topk_indice_->GetDataPtr(),
                                 (float*)topk_value_->GetDataPtr(),
                                 total_token_, num_expert_, top_k, cu_stream);
        cuda::MoeBatchedGemmLauncher<T>(
            (T*)in_tensor->GetDataPtr(),
            (T*)gate_up_proj_weight_tensor->GetDataPtr(),
            (uint32_t*)topk_indice_->GetDataPtr(), (T*)gate_up_proj_out,
            (uint32_t*)mid_row_indices_->GetDataPtr(), hWs, hWsSize, ws_ptr,
            dWsSize, total_token_, proj_size_ * 2, hidden_size_, num_expert_,
            top_k, cu_stream);

        cuda::UnaryGLUKernelLauncher((T*)mid_result, (T*)gate_up_proj_out,
                                     total_token_ * top_k, proj_size_,
                                     UnaryType::SILU, cu_stream);

        cuda::GetExpertByIndice((int*)mid_expert_indices_->GetDataPtr(),
                                (int*)topk_indice_->GetDataPtr(),
                                (int*)mid_row_indices_->GetDataPtr(),
                                total_token_, top_k, num_expert_, cu_stream);

        cuda::MoeBatchedGemmLauncher<T>(
            (T*)mid_result, (T*)down_proj_weight_tensor->GetDataPtr(),
            (uint32_t*)mid_expert_indices_->GetDataPtr(), (T*)final_result,
            (uint32_t*)final_row_indices_->GetDataPtr(), hWs, hWsSize, ws_ptr,
            dWsSize, total_token_ * top_k, hidden_size_, proj_size_,
            num_expert_, 1, cu_stream);
        cuda::FinalizeMoeRoutingNewKernelLauncher(
            (T*)out_tensor->GetDataPtr(), (T*)final_result,
            (float*)experts_score_->GetDataPtr(),
            (int*)mid_row_indices_->GetDataPtr(),
            (int*)final_row_indices_->GetDataPtr(), total_token_, top_k,
            hidden_size_, cu_stream);
      };
      DispatchCUDA(dtype_, functor);
      break;
    }
#endif
    case DeviceType::CPU: {
      LOG(ERROR) << "MOE Operator does not support "
                 << "CPU"
                 << " device type" << std::endl;
      return AsStatus::ALLSPARK_RUNTIME_ERROR;
    }
    default:
      break;
  }
  return AsStatus::ALLSPARK_SUCCESS;
}

}  // namespace allspark
