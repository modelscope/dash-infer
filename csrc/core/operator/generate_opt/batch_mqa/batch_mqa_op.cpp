/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    batch_mqa_op.cpp
 */

#include "batch_mqa_op.h"  // NOLINT

#include <core/kernel/kernel.h>
#include <cpu/cpu_context.h>
#include <utility/datatype_dispatcher.h>

#include <cmath>
namespace allspark {
void cpu_dec_single_mqa(DataType dtype, void* out, void* score,
                        const void* query, const void* key, const void* value,
                        const float* mask, const void* position_embedding,
                        void* k_cache, void* v_cache, void** q_array,
                        void** k_array, void** v_array, void** score_array,
                        void** out_array, int batch_size, int beam_size,
                        int seq_len, int step, int cache_max_len,
                        int hidden_size, int num_heads, int size_per_head,
                        int group_num, int gemm_batch, float alpha,
                        const DeviceContext* ctx) {
  auto functor = [&]<typename T>() {
    int kv_stride = size_per_head * group_num;
    int q_stride = hidden_size + 2 * kv_stride;
    cpu::UpdateKVLauncher((T*)k_cache, (T*)v_cache, (const T*)key,
                          (const T*)value, batch_size, step - 1, cache_max_len,
                          kv_stride, seq_len, q_stride);
    if (seq_len != 1) {
      step = seq_len;
    }
    int out_stride = hidden_size;
    int score_stride = step * num_heads;
    cpu::MultiQueryGetBatchArrayLauncher(
        (T*)query, (T*)k_cache, (T*)v_cache, (T*)score, (T*)out, (T**)q_array,
        (T**)k_array, (T**)v_array, (T**)score_array, (T**)out_array,
        batch_size, 1, num_heads, size_per_head, group_num, step,
        q_stride * seq_len, kv_stride * cache_max_len, score_stride * seq_len,
        out_stride * seq_len);
    cpu::BatchGemmWraper<T>(score_array, q_array, k_array, seq_len, step,
                            size_per_head, false, true, alpha, 0.0f, q_stride,
                            kv_stride, score_stride, gemm_batch);
    if (position_embedding) {
      cpu::SimpleAdd((T*)score, (T*)score, (T*)position_embedding,
                     batch_size * num_heads * step);
    }
    cpu::BatchSoftmax<T>((T*)score, mask, batch_size, beam_size, num_heads,
                         seq_len, step);
    cpu::BatchGemmWraper<T>(out_array, score_array, v_array, seq_len,
                            size_per_head, step, false, false, 1.0f, 0.0f,
                            score_stride, kv_stride, out_stride, gemm_batch);
  };
  DispatchCPU(dtype, functor);
}

int get_mqa_layer_num(std::string str) {
  std::stringstream ss(str);
  std::string temp;
  while (std::getline(ss, temp, '.')) {
    bool flag = true;
    for (char c : temp) {
      if (!std::isdigit(c)) {
        flag = false;
        break;
      }
    }
    if (flag) {
      return std::stoi(temp);
    }
  }
  return -1;
}
AsStatus BatchMQAOp::Init(const OperatorProto& op_proto,
                          const DeviceContext& ctx,
                          const TensorMap& weights_map, TensorMap* tensor_map) {
  AS_CHECK_STATUS(AsOperator::Init(op_proto, ctx, weights_map, tensor_map));
  // layer num
  layer_num_ = get_mqa_layer_num(this->op_name_);
  if (layer_num_ < 0) {
    LOG(ERROR) << "BatchMQAOp : can't find layer_num_" << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
  // type inference
  dtype_ = tensor_map_->at(in_names_[0])->GetDataType();
  tensor_map_->at(out_names_[0])->SetDataType(dtype_);

  // attr
  auto& attr_map = op_proto.attr();
  if (attr_map.find("multinode") != attr_map.end()) {
    multi_nodes_ = *(bool*)(attr_map.at("multinode").c_str());
  } else {
    multi_nodes_ = true;
  }

  if (attr_map.find("num_heads") == attr_map.end()) {
    LOG(ERROR) << "BatchMQAOp : can't find num_heads attribute." << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
  num_heads_ = *(int*)(attr_map.at("num_heads").c_str());

  size_per_head_ = ctx.GetSizePerHead();
  if (attr_map.find("multi_query_group_num") != attr_map.end()) {
    group_num_ = *(int*)(attr_map.at("multi_query_group_num").c_str());
  } else {
    group_num_ = num_heads_;
  }

  DeviceType backend = ctx.GetDeviceType();
  switch (backend) {
    case DeviceType::CPU: {
      kernel_launcher = cpu_dec_single_mqa;
      if (multi_nodes_) {
        const CPUContext* cpu_ctx = static_cast<const CPUContext*>(ctx_);
        num_heads_ /= cpu_ctx->GetNranks();
        if (group_num_ != 1) {
          group_num_ /= cpu_ctx->GetNranks();
        }
      }
      break;
    }
    default:
      LOG(ERROR) << op_type_ << " Operator does not support "
                 << DeviceType_Name(backend) << " device type" << std::endl;
      return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }
  kv_stride_ = size_per_head_ * group_num_;
  hidden_size_ = size_per_head_ * num_heads_;
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus BatchMQAOp::Reshape(RuntimeContext* runtime_ctx) {
  const AsTensor* x = tensor_map_->at(in_names_[0]).get();
  const Shape& x_shape = x->GetShape();
  Shape y_shape(x_shape);
  int single_batch = 1;
  batch_size_ = y_shape[0];
  seq_len_ = y_shape[1];
  qkv_stride_ = y_shape[2];
  if (qkv_stride_ != hidden_size_ + 2 * kv_stride_) {
    LOG(ERROR) << "Invalid qkv_stride_ in BatchMQAOp"
               << ", qkv_strde = " << qkv_stride_
               << ", hidden_size = " << hidden_size_
               << ", kv_stride = " << kv_stride_ << std::endl;
    return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }

  // set variable
  gemm_batch_ = single_batch * num_heads_;
  if (alpha_ < 0) {
    alpha_ = 1.0f / std::sqrt(size_per_head_ * 1.0f);
  }
  score_size_ =
      round32((int64_t)single_batch * ctx_->GetModelMaxLength() * num_heads_ *
              (ctx_->GetModelMaxLength()) * SizeofType(dtype_));
  int64_t ws_size =
      score_size_ + (int64_t)sizeof(void*) * round32(gemm_batch_) * 5;
  tensor_map_->at("workspace")->SetShape(Shape({ws_size}));
  y_shape[2] = hidden_size_;
  tensor_map_->at(out_names_[0])->SetShape(std::move(y_shape));

  if (runtime_ctx->is_context) {
    int64_t per_size = ctx_->GetKVcacheSize() * kv_stride_ * SizeofType(dtype_);
    runtime_ctx->GetContextGenCtx()->k_cache_list.push_back(
        std::make_unique<CacheMemory>(ctx_->GetDeviceType(), per_size));
    runtime_ctx->GetContextGenCtx()->v_cache_list.push_back(
        std::make_unique<CacheMemory>(ctx_->GetDeviceType(), per_size));
  }
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus BatchMQAOp::RunOneBatch(GenerateContext* gen_ctx, int current_batch) {
  AsTensor* in_tensor = tensor_map_->at(in_names_[0]).get();
  AsTensor* out_tensor = tensor_map_->at(out_names_[0]).get();

  // alloc kv_cache
  int64_t kv_size =
      (int64_t)(gen_ctx->step + seq_len_) * kv_stride_ * SizeofType(dtype_);
  gen_ctx->k_cache_list[layer_num_]->Alloc(kv_size);
  gen_ctx->v_cache_list[layer_num_]->Alloc(kv_size);
  void* k_cache_buf = gen_ctx->k_cache_list[layer_num_]->GetData();
  void* v_cache_buf = gen_ctx->v_cache_list[layer_num_]->GetData();

  // prepare input buf
  // qkv
  int offset = hidden_size_ * SizeofType(dtype_);
  void* out_ptr =
      (char*)out_tensor->GetDataPtr() + current_batch * seq_len_ * offset;
  void* q_buf = (char*)in_tensor->GetDataPtr() +
                current_batch * seq_len_ * (qkv_stride_ * SizeofType(dtype_));
  void* k_buf = (char*)q_buf + hidden_size_ * SizeofType(dtype_);
  void* v_buf = (char*)k_buf + kv_stride_ * SizeofType(dtype_);
  // mask
  float* mask_buf = gen_ctx->step == 0
                        ? (float*)(tensor_map_->at(in_names_[1])->GetDataPtr())
                        : nullptr;
  if (tensor_map_->at(in_names_[1])->GetShape().Count() == 0) {
    mask_buf = nullptr;
  }
  // position
  void* position_embedding =
      pos_embedding_ ? tensor_map_->at(in_names_[2])->GetDataPtr() : nullptr;

  // workspace
  char* score_buf = (char*)(tensor_map_->at("workspace")->GetDataPtr());
  void** q_array = (void**)(score_buf + score_size_);
  void** k_array = q_array + round32(gemm_batch_);
  void** v_array = k_array + round32(gemm_batch_);
  void** score_array = v_array + round32(gemm_batch_);
  void** out_array = score_array + round32(gemm_batch_);

  kernel_launcher(dtype_, out_ptr, score_buf, q_buf, k_buf, v_buf, mask_buf,
                  position_embedding, k_cache_buf, v_cache_buf, q_array,
                  k_array, v_array, score_array, out_array, 1,
                  gen_ctx->num_beams, seq_len_, (gen_ctx->step + 1),
                  ctx_->GetModelMaxLength(), hidden_size_, num_heads_,
                  size_per_head_, group_num_, gemm_batch_, alpha_, ctx_);

  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus BatchMQAOp::RunContext(RuntimeContext* runtime_ctx) {
  if (batch_size_ != 1) {
    LOG(ERROR) << "BatchMQAOp only support multibatch in decoder pharse, "
                  "not context pharse."
               << std::endl;
    return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }
  GenerateContext* gen_ctx = runtime_ctx->GetContextGenCtx();
  RunOneBatch(gen_ctx, 0);
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus BatchMQAOp::RunDecoder(RuntimeContext* runtime_ctx) {
  for (int batch = 0; batch < batch_size_; batch++) {
    GenerateContext* gen_ctx = runtime_ctx->GetGenCtx(batch);
    RunOneBatch(gen_ctx, batch);
  }
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus BatchMQAOp::Forward(RuntimeContext* runtime_ctx) {
  AsStatus status;
  if (runtime_ctx->is_context) {
    status = RunContext(runtime_ctx);
  } else {
    status = RunDecoder(runtime_ctx);
  }
  return status;
}

AsStatus BatchMQAOp::ResetCache() { return AsStatus::ALLSPARK_SUCCESS; }

REGISTER_OP("DecOptMQA", CPU, BatchMQAOp)
}  // namespace allspark
