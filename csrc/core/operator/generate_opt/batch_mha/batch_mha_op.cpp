/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    batch_mha_op.cpp
 */

#include "batch_mha_op.h"  // NOLINT

#include <core/kernel/kernel.h>
#include <cpu/cpu_context.h>
#include <utility/datatype_dispatcher.h>

#include <cmath>

namespace allspark {
void cpu_dec_single_mha(DataType dtype, void* out, void* score,
                        const void* query, const void* key, const void* value,
                        const float* mask, const void* position_embedding,
                        void* k_cache, void* v_cache, void** q_array,
                        void** k_array, void** v_array, void** score_array,
                        void** out_array, int batch_size, int beam_size,
                        int seq_len, int step, int cache_max_len,
                        int hidden_size, int num_heads, int size_per_head,
                        int gemm_batch, float alpha, bool xlogn_enable,
                        int xlogn_len, const DeviceContext* ctx) {
  DLOG(INFO) << "cpu_dec_single_mha" << std::endl;
  auto functor = [&]<typename T>() {
    cpu::UpdateKVLauncher((T*)k_cache, (T*)v_cache, (const T*)key,
                          (const T*)value, batch_size, step - 1, cache_max_len,
                          hidden_size, seq_len, 3 * hidden_size);
    if (seq_len != 1) {
      step = seq_len;
    }
    int q_stride = hidden_size * 3;
    int kv_stride = hidden_size;
    int out_stride = hidden_size;
    int score_stride = step * num_heads;
    cpu::GetBatchArrayLauncher((T*)query, (T*)k_cache, (T*)v_cache, (T*)score,
                               (T*)out, (T**)q_array, (T**)k_array,
                               (T**)v_array, (T**)score_array, (T**)out_array,
                               batch_size, 1, num_heads, size_per_head, step,
                               q_stride * seq_len, kv_stride * cache_max_len,
                               score_stride * seq_len, out_stride * seq_len);
    cpu::BatchGemmWraper<T>(score_array, q_array, k_array, seq_len, step,
                            size_per_head, false, true, alpha, 0.0f, q_stride,
                            kv_stride, score_stride, gemm_batch);
    if (position_embedding) {
      cpu::SimpleAdd((T*)score, (T*)score, (T*)position_embedding,
                     batch_size * num_heads * step * seq_len);
    }
    cpu::BatchSoftmax<T>((T*)score, mask, batch_size, beam_size, num_heads,
                         seq_len, step);
    cpu::BatchGemmWraper<T>(out_array, score_array, v_array, seq_len,
                            size_per_head, step, false, false, 1.0f, 0.0f,
                            score_stride, kv_stride, out_stride, gemm_batch);
  };
  DispatchCPU(dtype, functor);
}

int get_layer_num(std::string str) {
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

AsStatus BatchMHAOp::Init(const OperatorProto& op_proto,
                          const DeviceContext& ctx,
                          const TensorMap& weights_map, TensorMap* tensor_map) {
  AS_CHECK_STATUS(AsOperator::Init(op_proto, ctx, weights_map, tensor_map));
  layer_num_ = get_layer_num(this->op_name_);
  if (layer_num_ < 0) {
    LOG(ERROR) << "BatchMHAOp : can't find layer_num_" << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }

  // type inference
  dtype_ = tensor_map_->at(in_names_[0])->GetDataType();
  tensor_map_->at(out_names_[0])->SetDataType(dtype_);

  // attr
  auto& attr_map = op_proto.attr();

  if (attr_map.find("num_heads") == attr_map.end()) {
    LOG(ERROR) << "BatchMHAOp : can't find num_heads attribute." << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
  num_heads_ = *(int*)(attr_map.at("num_heads").c_str());

  if (attr_map.find("multinode") != attr_map.end()) {
    multi_nodes_ = *(bool*)(attr_map.at("multinode").c_str());
  } else {
    multi_nodes_ = true;
  }

  if (attr_map.find("position_embedding") != attr_map.end()) {
    pos_embedding_ = true;
  }

  if (attr_map.find("alpha") != attr_map.end()) {
    alpha_ = *(float*)(attr_map.at("alpha").c_str());
  }
  AS_CHECK_STATUS(lognFromAttributes(op_proto));

  causal_mask_ = true;
  DeviceType backend = ctx.GetDeviceType();

  switch (backend) {
    case DeviceType::CPU: {
      kernel_launcher = cpu_dec_single_mha;
      if (multi_nodes_) {
        const CPUContext* cpu_ctx = static_cast<const CPUContext*>(ctx_);
        num_heads_ /= cpu_ctx->GetNranks();
      }
      break;
    }
    default:
      LOG(ERROR) << op_type_ << " Operator does not support "
                 << DeviceType_Name(backend) << " device type" << std::endl;
      return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }

  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus BatchMHAOp::Reshape(RuntimeContext* runtime_ctx) {
  constexpr int single_batch = 1;

  const AsTensor* x = tensor_map_->at(in_names_[0]).get();
  const Shape& x_shape = x->GetShape();
  Shape y_shape(x_shape);
  y_shape[2] /= 3;
  batch_size_ = y_shape[0];
  seq_len_ = y_shape[1];
  hidden_size_ = y_shape[2];

  gemm_batch_ = single_batch * num_heads_;

  DLOG(INFO) << "BatchMHAOp::Reshape: batch_size_ = " << batch_size_
             << ", seq_len_ = " << seq_len_
             << ", hidden_size_ = " << hidden_size_;

  // set variable
  if (hidden_size_ % num_heads_) {
    LOG(ERROR) << "Invalid attribute in BatchMHAOp. hidden_size : "
               << hidden_size_ << ", num_heads : " << num_heads_ << std::endl;
    return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }
  size_per_head_ = hidden_size_ / num_heads_;
  if (alpha_ < 0) {
    alpha_ = 1.0f / std::sqrt(size_per_head_ * 1.0f);
  }
  score_size_ =
      round32((int64_t)single_batch * ctx_->GetModelMaxLength() * num_heads_ *
              (ctx_->GetModelMaxLength()) * SizeofType(dtype_));

  tensor_map_->at(out_names_[0])->SetShape(std::move(y_shape));

  // not using flash attention, set fallback MHA workspace.
  int64_t ws_size =
      score_size_ + (int64_t)sizeof(void*) * round32(gemm_batch_) * 5;
  tensor_map_->at("workspace")->SetShape(Shape({ws_size}));
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus BatchMHAOp::runOneBatch(GenerateContext* gen_ctx, int current_batch) {
  AsTensor* in_tensor = tensor_map_->at(in_names_[0]).get();
  AsTensor* out_tensor = tensor_map_->at(out_names_[0]).get();
  AsTensor* wss_tensor = tensor_map_->at("workspace").get();
  void* k_cache_buf = gen_ctx->k_cache_list[layer_num_]->GetData();
  void* v_cache_buf = gen_ctx->v_cache_list[layer_num_]->GetData();
  int offset = hidden_size_ * SizeofType(dtype_);
  void* out_ptr =
      (char*)out_tensor->GetDataPtr() + current_batch * seq_len_ * offset;
  void* q_buf =
      (char*)in_tensor->GetDataPtr() + current_batch * seq_len_ * offset * 3;
  void* k_buf = (char*)q_buf + offset;
  void* v_buf = (char*)k_buf + offset;
  float* mask_buf = gen_ctx->step == 0
                        ? (float*)(tensor_map_->at(in_names_[1])->GetDataPtr())
                        : nullptr;
  if (tensor_map_->at(in_names_[1])->GetShape().Count() == 0) {
    mask_buf = nullptr;
  }
  void* position_embedding =
      pos_embedding_ ? tensor_map_->at(in_names_[2])->GetDataPtr() : nullptr;
  char* score_buf = (char*)(tensor_map_->at("workspace")->GetDataPtr());
  void** q_array = (void**)(score_buf + score_size_);
  void** k_array = q_array + round32(gemm_batch_);
  void** v_array = k_array + round32(gemm_batch_);
  void** score_array = v_array + round32(gemm_batch_);
  void** out_array = score_array + round32(gemm_batch_);

  std::pair<bool, AsMHAPrefill> prefill_mode_pair = GetPrefillMode();
  if (!prefill_mode_pair.first) {
    LOG(ERROR) << "BatchMHAOp get prefill mode error. " << std::endl;
    return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }

  kernel_launcher(
      dtype_, out_ptr, score_buf, q_buf, k_buf, v_buf, mask_buf,
      position_embedding, k_cache_buf, v_cache_buf, q_array, k_array, v_array,
      score_array, out_array, 1, gen_ctx->num_beams, seq_len_,
      (gen_ctx->step + 1), ctx_->GetModelMaxLength(), hidden_size_, num_heads_,
      size_per_head_, gemm_batch_, alpha_, enable_logn_, xlogn_, ctx_);
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus BatchMHAOp::runContext(RuntimeContext* runtime_ctx) {
  if (batch_size_ != 1) {
    LOG(ERROR) << "BatchMHAOp only support multibatch in decoder pharse, "
                  "not context pharse."
               << std::endl;
    return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }
  GenerateContext* gen_ctx = runtime_ctx->GetContextGenCtx();
  DLOG(INFO) << "BatchMHAOp::runContext [" << gen_ctx->request->request_id
             << "][layer " << layer_num_ << "]";
  runOneBatch(gen_ctx, 0);
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus BatchMHAOp::runDecoder(RuntimeContext* runtime_ctx) {
  DLOG(INFO) << "BatchMHAOp::runDecoder: batch size=" << batch_size_;

  for (int batch = 0; batch < batch_size_; batch++) {
    GenerateContext* gen_ctx = runtime_ctx->GetGenCtx(batch);
    DLOG(INFO) << "BatchMHAOp::runDecoder (batch " << batch << ")["
               << gen_ctx->request->request_id << "][step " << gen_ctx->step
               << "][layer " << layer_num_ << "]";
    runOneBatch(gen_ctx, batch);
  }
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus BatchMHAOp::Alloc(RuntimeContext* runtime_ctx) {
  if (runtime_ctx->is_context) {
    int64_t per_size =
        ctx_->GetKVcacheSize() * hidden_size_ * SizeofType(dtype_);
    GenerateContext* gen_ctx = runtime_ctx->GetContextGenCtx();
    gen_ctx->k_cache_list.push_back(
        std::make_unique<CacheMemory>(ctx_->GetDeviceType(), per_size));
    gen_ctx->v_cache_list.push_back(
        std::make_unique<CacheMemory>(ctx_->GetDeviceType(), per_size));
  }
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus BatchMHAOp::Forward(RuntimeContext* runtime_ctx) {
  AsStatus status;
  if (runtime_ctx->is_context) {
    status = runContext(runtime_ctx);
  } else {
    status = runDecoder(runtime_ctx);
  }
  return status;
}

REGISTER_OP("DecOptMHA", CPU, BatchMHAOp)
}  // namespace allspark
