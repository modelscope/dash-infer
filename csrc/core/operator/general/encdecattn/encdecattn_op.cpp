/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    encdecattn_op.cpp
 */

#include "encdecattn_op.h"  // NOLINT

#include <core/kernel/kernel.h>
#include <cpu/cpu_context.h>
#include <utility/datatype_dispatcher.h>

#include <cmath>
namespace allspark {

AsStatus EncdecAttentionOp::Init(const OperatorProto& op_proto,
                                 const DeviceContext& ctx,
                                 const TensorMap& weights_map,
                                 TensorMap* tensor_map) {
  AS_CHECK_STATUS(AsOperator::Init(op_proto, ctx, weights_map, tensor_map));
  // type inference
  dtype_ = tensor_map_->at(in_names_[0])->GetDataType();
  tensor_map_->at(out_names_[0])->SetDataType(dtype_);
  // attr
  auto& attr_map = op_proto.attr();
  if (attr_map.find("num_heads") == attr_map.end()) {
    LOG(ERROR) << "EncdecAttentionOp : can't find num_heads attribute."
               << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
  if (attr_map.find("alpha") != attr_map.end()) {
    alpha_ = *(float*)(attr_map.at("alpha").c_str());
  }
  num_heads_ = *(int*)(attr_map.at("num_heads").c_str());
  DeviceType backend = ctx.GetDeviceType();
  switch (backend) {
    case DeviceType::CPU: {
      const CPUContext* cpu_ctx = static_cast<const CPUContext*>(ctx_);
      num_heads_ /= cpu_ctx->GetNranks();
      break;
    }
  }
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus EncdecAttentionOp::Reshape() {
  Shape x_shape = tensor_map_->at(in_names_[0])->GetShape();
  batch_size_ = tensor_map_->at(in_names_[1])->GetShape()[0];
  enc_seq_len_ = tensor_map_->at(in_names_[1])->GetShape()[1];
  hidden_size_ = x_shape[2];
  beam_size_ = x_shape[0] / batch_size_;
  // set variable
  if (hidden_size_ % num_heads_) {
    LOG(ERROR) << "Invalid attribute in EncdecAttentionOp. hidden_size : "
               << hidden_size_ << ", num_heads : " << num_heads_ << std::endl;
    return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }
  size_per_head_ = hidden_size_ / num_heads_;
  gemm_batch_ = batch_size_ * beam_size_ * num_heads_;
  if (alpha_ < 0) {  // set default alpha
    alpha_ = 1.0f / std::sqrt(size_per_head_ * 1.0f);
  }
  score_size_ =
      round32((int64_t)gemm_batch_ * enc_seq_len_ * SizeofType(dtype_));
  int64_t ws_size =
      score_size_ + (int64_t)sizeof(void*) * round32(gemm_batch_) * 5;
  tensor_map_->at("workspace")->SetShape(Shape({ws_size}));
  tensor_map_->at(out_names_[0])->SetShape(std::move(x_shape));
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus EncdecAttentionOp::Forward() {
  AsTensor* in_tensor = tensor_map_->at(in_names_[0]).get();
  int offset = hidden_size_ * SizeofType(dtype_);
  void* q_buf = in_tensor->GetDataPtr();
  void* k_buf = tensor_map_->at(in_names_[1]).get()->GetDataPtr();
  void* v_buf = (char*)k_buf + offset;
  float* mask_buf = in_names_.size() > 2
                        ? (float*)tensor_map_->at(in_names_[2])->GetDataPtr()
                        : nullptr;
  char* score_buf = (char*)(tensor_map_->at("workspace")->GetDataPtr());
  void** q_array = (void**)(score_buf + score_size_);
  void** k_array = q_array + round32(gemm_batch_);
  void** v_array = k_array + round32(gemm_batch_);
  void** score_array = v_array + round32(gemm_batch_);
  void** out_array = score_array + round32(gemm_batch_);
  DeviceType backend = ctx_->GetDeviceType();
  switch (backend) {
    case DeviceType::CPU: {
      auto functor = [&]<typename T>() {
        int q_stride = hidden_size_;
        int kv_stride = 2 * hidden_size_;
        int score_stride = enc_seq_len_ * num_heads_;
        int out_stride = hidden_size_;
        T* out = (T*)(tensor_map_->at(out_names_[0])->GetDataPtr());
        T* score = (T*)score_buf;
        T* query = (T*)q_buf;
        T* key = (T*)k_buf;
        T* value = (T*)v_buf;
        cpu::GetBatchArrayLauncher(
            query, key, value, score, out, (T**)q_array, (T**)k_array,
            (T**)v_array, (T**)score_array, (T**)out_array, batch_size_,
            beam_size_, num_heads_, size_per_head_, enc_seq_len_, q_stride,
            kv_stride * enc_seq_len_, score_stride, out_stride);
        cpu::BatchGemmWraper<T>(score_array, q_array, k_array, 1, enc_seq_len_,
                                size_per_head_, false, true, alpha_, 0.0f,
                                q_stride, kv_stride, score_stride, gemm_batch_);
        cpu::BatchSoftmax<T>(score, mask_buf, batch_size_ * beam_size_,
                             beam_size_, num_heads_, 1, enc_seq_len_);
        cpu::BatchGemmWraper<T>(out_array, score_array, v_array, 1,
                                size_per_head_, enc_seq_len_, false, false,
                                1.0f, 0.0f, score_stride, kv_stride, out_stride,
                                gemm_batch_);
      };
      DispatchCPU(dtype_, functor);
      break;
    }
    default:
      LOG(ERROR) << "EncdecAttention Operator does not support "
                 << DeviceType_Name(backend) << " device type" << std::endl;
      return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }
  return AsStatus::ALLSPARK_SUCCESS;
}

REGISTER_OP("EncdecAttention", CPU, EncdecAttentionOp)
}  // namespace allspark
