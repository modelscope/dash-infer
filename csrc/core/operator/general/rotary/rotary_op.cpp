/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    rotary_op.cpp
 */

#include "rotary_op.h"  // NOLINT

#include <common/float16.h>
#include <core/kernel/kernel.h>
#include <cpu/cpu_context.h>
#include <utility/datatype_dispatcher.h>

#include <cmath>
namespace allspark {

void rotary_launcher(DataType dtype, void* out, void* in, float* inv_freq,
                     int* batch_offset, int batch_size, int seq_len,
                     int* step_list, int hidden_size, int num_heads,
                     int size_per_head, int input_len, int qkv_stride,
                     int rotary_type, float rotary_pct, int xlogn,
                     const DeviceContext* ctx) {
  DeviceType backend = ctx->GetDeviceType();
  if (backend == DeviceType::CPU) {
    switch (rotary_type) {
      case RotaryType::base: {
        auto functor = [&]<typename T>() {
          cpu::RotaryKernelLauncher<T>(
              (T*)out, (T*)in, inv_freq, batch_offset, batch_size, seq_len,
              num_heads, size_per_head, step_list, qkv_stride, xlogn);
        };
        DispatchCPU(dtype, functor);
        break;
      }
      default: {
        LOG(ERROR) << "RotaryOp (CPU): not support rotary_type" << std::endl;
        break;
      }
    }
  }
}

AsStatus RotaryOp::Init(const OperatorProto& op_proto, const DeviceContext& ctx,
                        const TensorMap& weights_map, TensorMap* tensor_map) {
  AS_CHECK_STATUS(AsOperator::Init(op_proto, ctx, weights_map, tensor_map));
  // type inference
  dtype_ = tensor_map_->at(in_names_[0])->GetDataType();
  tensor_map_->at(out_names_[0])->SetDataType(dtype_);
  // attr.1 num_heads
  auto& attr_map = op_proto.attr();
  if (attr_map.find("num_heads") == attr_map.end()) {
    LOG(ERROR) << "RotaryOp : can't find num_heads attribute." << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
  num_heads_ = *(int*)(attr_map.at("num_heads").c_str());

  // attr.2 rotary_type
  rotary_type_ = RotaryType::base;
  if (attr_map.find("rotary_type") != attr_map.end()) {
    rotary_type_ = *(int*)(attr_map.at("rotary_type").c_str());
  }

  // attr.3 use_weight
  use_weight_ = false;
  if (attr_map.find("use_weight") != attr_map.end()) {
    use_weight_ = *(bool*)(attr_map.at("use_weight").c_str());
  }

  // attr.4 dynamic ntk enable / param
  ntk_model_embed_ = -1;
  if (attr_map.find("ntk_model_embed") != attr_map.end()) {
    ntk_model_embed_ = *(int*)(attr_map.at("ntk_model_embed").c_str());
  }

  // attr.5 rotary_pct
  rotary_pct_ = 1.0;
  if (attr_map.find("rotary_pct") != attr_map.end()) {
    rotary_pct_ = *(float*)(attr_map.at("rotary_pct").c_str());
  }

  // attr.6 sequence length coe.
  seqlen_extrapolation_ = 1.f;
  if (attr_map.find("seqlen_extrapolation") != attr_map.end()) {
    seqlen_extrapolation_ =
        *(float*)(attr_map.at("seqlen_extrapolation").c_str());
    if (seqlen_extrapolation_ < 0.5f || seqlen_extrapolation_ > 128.f) {
      LOG(ERROR) << "RotaryOp : "
                 << "Incoming sequence length extrapolation value = "
                 << seqlen_extrapolation_ << ". "
                 << "out of valid range = [0.5, 128]. "
                 << "please check seqlen_extrapolation attributes again."
                 << std::endl;
      return AsStatus::ALLSPARK_PARAM_ERROR;
    }
  }

  // attr.7 base
  base_ = 10000.f;
  if (attr_map.find("rotary_base") != attr_map.end()) {
    base_ = *(float*)(attr_map.at("rotary_base").c_str());
  }
  xlogn_ = -1;
  if (attr_map.find("logn_model_embedding") != attr_map.end()) {
    xlogn_ = *(int*)(attr_map.at("logn_model_embedding").c_str());
  }

  // backend switch
  DeviceType backend = ctx.GetDeviceType();
  switch (backend) {
    case DeviceType::CPU: {
      const CPUContext* cpu_ctx = static_cast<const CPUContext*>(ctx_);
      num_heads_ /= cpu_ctx->GetNranks();
      break;
    }
    default:
      LOG(ERROR) << op_type_ << " Operator does not support "
                 << DeviceType_Name(backend) << " device type" << std::endl;
      return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus RotaryOp::Reshape(RuntimeContext* runtime_ctx) {
  const AsTensor* x = tensor_map_->at(in_names_[0]).get();
  const Shape& x_shape = x->GetShape();
  Shape y_shape(x_shape);
  batch_size_ = y_shape[0];
  seq_len_ = y_shape[1];
  hidden_size_ = y_shape[2] / 3;
  tensor_map_->at(out_names_[0])->SetShape(std::move(y_shape));
  // set variable
  if (hidden_size_ % num_heads_) {
    LOG(ERROR) << "Invalid attribute in RotaryOp. hidden_size : "
               << hidden_size_ << ", num_heads : " << num_heads_ << std::endl;
    return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }
  size_per_head_ = hidden_size_ / num_heads_;
  gemm_batch_ = batch_size_ * num_heads_;
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus RotaryOp::RunRotary(int run_batch_size, AsTensor* rotary_step,
                             AsTensor* rotary_inv_freq) {
  int* run_step = (int*)rotary_step->GetDataPtr();
  float* inv_freq = (float*)rotary_inv_freq->GetDataPtr();
  int qkv_stride = 3 * hidden_size_;
  int* batch_offset = nullptr;
  int offset = hidden_size_ * SizeofType(dtype_);
  void* q_buf = (char*)tensor_map_->at(in_names_[0])->GetDataPtr();
  void* k_buf = (char*)q_buf + offset;
  void* v_buf = (char*)k_buf + offset;
  void* outq_buf = (char*)tensor_map_->at(out_names_[0])->GetDataPtr();
  void* outk_buf = (char*)outq_buf + offset;
  void* outv_buf = (char*)outk_buf + offset;

  rotary_launcher(dtype_, outq_buf, q_buf, inv_freq, batch_offset,
                  run_batch_size, seq_len_, run_step, hidden_size_, num_heads_,
                  size_per_head_, 0, qkv_stride, rotary_type_, rotary_pct_,
                  xlogn_, ctx_);
  rotary_launcher(dtype_, outk_buf, k_buf, inv_freq, batch_offset,
                  run_batch_size, seq_len_, run_step, hidden_size_, num_heads_,
                  size_per_head_, 0, qkv_stride, rotary_type_, rotary_pct_, -1,
                  ctx_);
  rotary_launcher(dtype_, outv_buf, v_buf, nullptr, batch_offset,
                  run_batch_size, seq_len_, run_step, hidden_size_, num_heads_,
                  size_per_head_, 0, qkv_stride, rotary_type_, rotary_pct_, -1,
                  ctx_);
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus RotaryOp::RunContext(RuntimeContext* runtime_ctx) {
  if (batch_size_ != 1) {
    LOG(ERROR) << "BatchMHAOp only support multibatch in decoder pharse, "
                  "not context pharse."
               << std::endl;
    return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }

  std::shared_ptr<LayerCacheManager> layer_cache_manager =
      runtime_ctx->GetLayerCacheManager();
  AsTensor* rotary_step = layer_cache_manager->GetCache("rotary_step");
  AsTensor* rotary_inv_freq = layer_cache_manager->GetCache("rotary_inv_freq");
  if (!layer_cache_manager->IsCacheSet("rotary_step") &&
      !layer_cache_manager->IsCacheSet("rotary_inv_freq")) {
    layer_cache_manager->SetCache("rotary_step");
    layer_cache_manager->SetCache("rotary_inv_freq");
    int freq_size = size_per_head_ / 2;
    int batch_size = 1;
    GenerateContext* gen_ctx = runtime_ctx->GetContextGenCtx();
    std::vector<float> inv_freq_tmp =
        calculate_invfreq(base_, gen_ctx->input_len);
    rotary_inv_freq->SetShape(Shape{batch_size * freq_size});
    rotary_inv_freq->CopyDataFrom(inv_freq_tmp.data(),
                                  sizeof(float) * batch_size * freq_size,
                                  DeviceType::CPU, ctx_);
    std::vector<int> run_step_tmp(batch_size);
    run_step_tmp[0] = gen_ctx->step;
    rotary_step->SetShape(Shape{batch_size});
    rotary_step->CopyDataFrom(run_step_tmp.data(), sizeof(int) * batch_size,
                              DeviceType::CPU, ctx_);
  }
  RunRotary(1, rotary_step, rotary_inv_freq);
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus RotaryOp::RunDecoder(RuntimeContext* runtime_ctx) {
  std::shared_ptr<LayerCacheManager> layer_cache_manager =
      runtime_ctx->GetLayerCacheManager();
  AsTensor* rotary_step = layer_cache_manager->GetCache("rotary_step");
  AsTensor* rotary_inv_freq = layer_cache_manager->GetCache("rotary_inv_freq");
  if (!layer_cache_manager->IsCacheSet("rotary_step") &&
      !layer_cache_manager->IsCacheSet("rotary_inv_freq")) {
    layer_cache_manager->SetCache("rotary_step");
    layer_cache_manager->SetCache("rotary_inv_freq");
    int freq_size = size_per_head_ / 2;
    int batch_size = runtime_ctx->GetGenCtxListSize();
    std::vector<float> inv_freq_tmp(batch_size * freq_size);
    std::vector<int> run_step_tmp(batch_size);
    for (int batch = 0; batch < batch_size; batch++) {
      GenerateContext* gen_ctx = runtime_ctx->GetGenCtx(batch);
      std::vector<float> inv_freq_one =
          calculate_invfreq(base_, gen_ctx->input_len);
      for (int j = 0; j < freq_size; j++) {
        inv_freq_tmp[batch * freq_size + j] = inv_freq_one[j];
      }
      run_step_tmp[batch] = gen_ctx->step;
    }
    rotary_inv_freq->SetShape(Shape{batch_size * freq_size});
    rotary_inv_freq->CopyDataFrom(inv_freq_tmp.data(),
                                  sizeof(float) * batch_size * freq_size,
                                  DeviceType::CPU, ctx_);
    rotary_step->SetShape(Shape{batch_size});
    rotary_step->CopyDataFrom(run_step_tmp.data(), sizeof(int) * batch_size,
                              DeviceType::CPU, ctx_);
  }
  RunRotary(runtime_ctx->GetGenCtxListSize(), rotary_step, rotary_inv_freq);
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus RotaryOp::Forward(RuntimeContext* runtime_ctx) {
  AsStatus status = AsStatus::ALLSPARK_SUCCESS;
  if (runtime_ctx->is_context) {
    status = RunContext(runtime_ctx);
  } else {
    status = RunDecoder(runtime_ctx);
  }
  return status;
}

REGISTER_OP("Rotary", CPU, RotaryOp)
}  // namespace allspark
