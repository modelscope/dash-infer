/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    mha_op.cpp
 */

#include "mha_op.h"  // NOLINT

#include <core/kernel/kernel.h>
#include <utility/datatype_dispatcher.h>

#include <cmath>
#ifdef ENABLE_CUDA
#include <cuda/cuda_context.h>
#endif
#include <cpu/cpu_context.h>
namespace allspark {

#ifdef ENABLE_CUDA
AsStatus gpu_mha(DataType dtype, void* out, void* score, const void* query,
                 const void* key, const void* value, const float* mask,
                 const void* position_embedding, void** q_array, void** k_array,
                 void** v_array, void** score_array, void** out_array,
                 int batch_size, int seq_len, int hidden_size, int num_heads,
                 int size_per_head, int gemm_batch, float alpha,
                 const DeviceContext* ctx) {
  DLOG(INFO) << "gpu_mha" << std::endl;
  const CUDAContext* gpu_ctx = static_cast<const CUDAContext*>(ctx);
  cublasHandle_t cublas_handle = gpu_ctx->GetCublasHandle();
  if (gpu_ctx->GetMatmulPrecision() == PrecisionLevel::HIGH &&
      dtype == FLOAT32) {
    cublasSetMathMode(cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH);
  }
  cudaStream_t cu_stream = (cudaStream_t)(gpu_ctx->GetStream());
  int qkv_stride = hidden_size * 3;
  int out_stride = hidden_size;
  int score_stride = seq_len * num_heads;
  auto functor = [&]<typename T>() {
    cuda::GetBatchArrayLauncher(
        (T*)query, (T*)key, (T*)value, (T*)score, (T*)out, (T**)q_array,
        (T**)k_array, (T**)v_array, (T**)score_array, (T**)out_array,
        batch_size, 1, num_heads, size_per_head, seq_len, qkv_stride * seq_len,
        qkv_stride * seq_len, score_stride * seq_len, out_stride * seq_len,
        cu_stream);
    // batch gemm 1
    cuda::BatchGemmWraper<T>(score_array, q_array, k_array, seq_len, seq_len,
                             size_per_head, false, true, alpha, 0.0f,
                             qkv_stride, qkv_stride, score_stride, gemm_batch,
                             cublas_handle);
    if (position_embedding) {
      cuda::BinaryKernelLauncher((T*)score, (T*)score, (T*)position_embedding,
                                 batch_size * num_heads * seq_len * seq_len, 1,
                                 cu_stream);
    }
    cuda::SoftmaxKernelLauncher((T*)score, mask, batch_size, 1, num_heads,
                                seq_len, seq_len, cu_stream);
    // batch gemm 2
    cuda::BatchGemmWraper<T>(out_array, score_array, v_array, seq_len,
                             size_per_head, seq_len, false, false, 1.0f, 0.0f,
                             score_stride, qkv_stride, out_stride, gemm_batch,
                             cublas_handle);
  };
  DispatchCUDA(dtype, functor);
  return AsStatus::ALLSPARK_SUCCESS;
}
#endif
AsStatus cpu_mha(DataType dtype, void* out, void* score, const void* query,
                 const void* key, const void* value, const float* mask,
                 const void* position_embedding, void** q_array, void** k_array,
                 void** v_array, void** score_array, void** out_array,
                 int batch_size, int seq_len, int hidden_size, int num_heads,
                 int size_per_head, int gemm_batch, float alpha,
                 const DeviceContext* ctx) {
  DLOG(INFO) << "cpu_mha" << std::endl;
  int qkv_stride = hidden_size * 3;
  int out_stride = hidden_size;
  int score_stride = seq_len * num_heads;
  auto functor = [&]<typename T>() {
    cpu::GetBatchArrayLauncher(
        (T*)query, (T*)key, (T*)value, (T*)score, (T*)out, (T**)q_array,
        (T**)k_array, (T**)v_array, (T**)score_array, (T**)out_array,
        batch_size, 1, num_heads, size_per_head, seq_len, qkv_stride * seq_len,
        qkv_stride * seq_len, score_stride * seq_len, out_stride * seq_len);
    cpu::BatchGemmWraper<T>(score_array, q_array, k_array, seq_len, seq_len,
                            size_per_head, false, true, alpha, 0.0f, qkv_stride,
                            qkv_stride, score_stride, gemm_batch);
    if (position_embedding) {
      cpu::SimpleAdd((T*)score, (T*)score, (T*)position_embedding,
                     batch_size * num_heads * seq_len * seq_len);
    }
    cpu::BatchSoftmax<T>((T*)score, mask, batch_size, 1, num_heads, seq_len,
                         seq_len);
    cpu::BatchGemmWraper<T>(out_array, score_array, v_array, seq_len,
                            size_per_head, seq_len, false, false, 1.0f, 0.0f,
                            score_stride, qkv_stride, out_stride, gemm_batch);
  };
  DispatchCPU(dtype, functor);
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus MultiHeadAttentionOp::Init(const OperatorProto& op_proto,
                                    const DeviceContext& ctx,
                                    const TensorMap& weights_map,
                                    TensorMap* tensor_map) {
  AS_CHECK_STATUS(AsOperator::Init(op_proto, ctx, weights_map, tensor_map));
  // type inference
  DataType dtype = tensor_map_->at(in_names_[0])->GetDataType();
  tensor_map_->at(out_names_[0])->SetDataType(dtype);
  // attr
  auto& attr_map = op_proto.attr();
  if (attr_map.find("num_heads") == attr_map.end()) {
    LOG(ERROR) << "MultiHeadAttentionOp : can't find num_heads attribute."
               << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
  num_heads_ = *(int*)(attr_map.at("num_heads").c_str());
  if (attr_map.find("position_embedding") != attr_map.end()) {
    pos_embedding_ = true;
  }
  if (attr_map.find("alpha") != attr_map.end()) {
    alpha_ = *(float*)(attr_map.at("alpha").c_str());
  }
  DeviceType backend = ctx.GetDeviceType();
  switch (backend) {
#ifdef ENABLE_CUDA
    case DeviceType::CUDA: {
      kernel_launcher = gpu_mha;
      const CUDAContext* gpu_ctx = static_cast<const CUDAContext*>(ctx_);
      num_heads_ /= gpu_ctx->GetNranks();
      break;
    }
#endif
    case DeviceType::CPU: {
      kernel_launcher = cpu_mha;
      const CPUContext* cpu_ctx = static_cast<const CPUContext*>(ctx_);
      num_heads_ /= cpu_ctx->GetNranks();
      break;
    }
    default:
      LOG(ERROR) << "MultiHeadAttention Operator does not support "
                 << DeviceType_Name(backend) << " device type" << std::endl;
      return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus MultiHeadAttentionOp::Reshape() {
  const AsTensor* x = tensor_map_->at(in_names_[0]).get();
  const Shape& x_shape = x->GetShape();
  Shape y_shape(x_shape);
  y_shape[2] /= 3;
  batch_size_ = y_shape[0];
  seq_len_ = y_shape[1];
  hidden_size_ = y_shape[2];
  // set variable
  int dtype_size = SizeofType(x->GetDataType());
  if (hidden_size_ % num_heads_) {
    LOG(ERROR) << "Invalid attribute in MultiHeadAttentionOp. hidden_size : "
               << hidden_size_ << ", num_heads : " << num_heads_ << std::endl;
    return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }
  size_per_head_ = hidden_size_ / num_heads_;
  gemm_batch_ = batch_size_ * num_heads_;
  if (alpha_ < 0) {
    alpha_ = 1.0f / std::sqrt(size_per_head_ * 1.0f);
  }
  score_size_ = round32((int64_t)batch_size_ * seq_len_ * num_heads_ *
                        seq_len_ * dtype_size);
  int64_t ws_size =
      score_size_ + (int64_t)sizeof(void*) * round32(gemm_batch_) * 5;
  tensor_map_->at("workspace")->SetShape(Shape({ws_size}));
  tensor_map_->at(out_names_[0])->SetShape(std::move(y_shape));

#ifdef ENABLE_CUDA
  if (ctx_->GetDeviceType() == DeviceType::CUDA) {
    const CUDAContext* gpu_ctx = static_cast<const CUDAContext*>(ctx_);
    cublasHandle_t cublas_handle = gpu_ctx->GetCublasHandle();
    cublasSetWorkspace(cublas_handle,
                       tensor_map_->at("cublas_workspace")->GetDataPtr(),
                       tensor_map_->at("cublas_workspace")->GetSizeInByte());
  }
#endif
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus MultiHeadAttentionOp::Forward() {
  AsTensor* in_tensor = tensor_map_->at(in_names_[0]).get();
  int offset = hidden_size_ * SizeofType(in_tensor->GetDataType());
  void* q_buf = in_tensor->GetDataPtr();
  void* k_buf = (char*)q_buf + offset;
  void* v_buf = (char*)k_buf + offset;
  float* mask_buf = (float*)(tensor_map_->at(in_names_[1])->GetDataPtr());
  // float* mask_buf = nullptr; for PLUG2
  void* position_embedding =
      pos_embedding_ ? tensor_map_->at(in_names_[2])->GetDataPtr() : nullptr;
  char* score_buf = (char*)(tensor_map_->at("workspace")->GetDataPtr());
  void** q_array = (void**)(score_buf + score_size_);
  void** k_array = q_array + round32(gemm_batch_);
  void** v_array = k_array + round32(gemm_batch_);
  void** score_array = v_array + round32(gemm_batch_);
  void** out_array = score_array + round32(gemm_batch_);
  kernel_launcher(
      in_tensor->GetDataType(), tensor_map_->at(out_names_[0])->GetDataPtr(),
      score_buf, q_buf, k_buf, v_buf, mask_buf, position_embedding, q_array,
      k_array, v_array, score_array, out_array, batch_size_, seq_len_,
      hidden_size_, num_heads_, size_per_head_, gemm_batch_, alpha_, ctx_);
  return AsStatus::ALLSPARK_SUCCESS;
}

REGISTER_OP(MultiHeadAttention, CUDA, MultiHeadAttentionOp)
REGISTER_OP(MultiHeadAttention, CPU, MultiHeadAttentionOp)
}  // namespace allspark
