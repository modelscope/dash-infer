/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    rotary_mulquery_op.cpp
 */

#include "rotary_mulquery_op.h"  // NOLINT

#include <common/float16.h>
#include <core/kernel/kernel.h>
#include <cpu/cpu_context.h>
#include <utility/datatype_dispatcher.h>

#include <cmath>
namespace allspark {

template <typename T>
static void rotary_inv_freq_convert(AsTensor* inv_freq_tensor, T* data_ptr,
                                    int inv_size, DeviceType device_type,
                                    const DeviceContext* ctx) {
  // 先取值到CPU上，再强转为float
  std::vector<T> inv_freq_0(inv_size);
  std::vector<float> inv_freq_tmp(inv_size);
  if (device_type == DeviceType::CPU) {
    memcpy(inv_freq_0.data(), data_ptr, inv_size * sizeof(T));
  }
  for (int i = 0; i < inv_size; ++i) {
    inv_freq_tmp[i] = (float)(inv_freq_0[i]);
  }
  inv_freq_tensor->CopyDataFrom(inv_freq_tmp.data(),
                                inv_freq_tmp.size() * sizeof(float),
                                DeviceType::CPU, ctx);
  return;
}

void rotary_multiquery_launcher(DataType dtype, void* out, void* in,
                                float* inv_freq, int* batch_offset,
                                int batch_size, int seq_len, int* step_list,
                                int hidden_size, int num_heads,
                                int size_per_head, int qkv_stride,
                                int rotary_type, int xlogn,
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
      case RotaryType::half_inner: {
        auto functor = [&]<typename T>() {
          cpu::RotaryEmbeddingHalfInner<T>(
              (T*)out, (T*)in, inv_freq, batch_offset, batch_size, seq_len,
              num_heads, size_per_head, step_list, qkv_stride);
        };
        DispatchCPU(dtype, functor);
        break;
      }
      default: {
        LOG(ERROR) << "RotaryMulQueryOp (CPU): not support rotary_type"
                   << std::endl;
        break;
      }
    }
  }
}

AsStatus RotaryMulQueryOp::Init(const OperatorProto& op_proto,
                                const DeviceContext& ctx,
                                const TensorMap& weights_map,
                                TensorMap* tensor_map) {
  AS_CHECK_STATUS(AsOperator::Init(op_proto, ctx, weights_map, tensor_map));
  // type inference
  dtype_ = tensor_map_->at(in_names_[0])->GetDataType();
  tensor_map_->at(out_names_[0])->SetDataType(dtype_);

  // attr.1 num_heads
  auto& attr_map = op_proto.attr();
  if (attr_map.find("num_heads") == attr_map.end()) {
    LOG(ERROR) << "RotaryMulQueryOp : can't find num_heads attribute."
               << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
  num_heads_ = *(int*)(attr_map.at("num_heads").c_str());

  // attr.2 rotary_type
  rotary_type_ = RotaryType::base;
  if (attr_map.find("rotary_type") != attr_map.end()) {
    // 找不到默认为0,即default类型，适配老版本
    rotary_type_ = *(int*)(attr_map.at("rotary_type").c_str());
  }

  // attr.3 use_weight
  use_weight_ = true;
  if (attr_map.find("use_weight") != attr_map.end()) {
    use_weight_ = *(bool*)(attr_map.at("use_weight").c_str());
  }

  // attr.4 dynamic ntk enable / param
  ntk_model_embed_ = -1;
  if (attr_map.find("ntk_model_embed") != attr_map.end()) {
    ntk_model_embed_ = *(int*)(attr_map.at("ntk_model_embed").c_str());
  }

  // attr.5 sequence length coe.
  seqlen_extrapolation_ = 1.f;
  if (attr_map.find("seqlen_extrapolation") != attr_map.end()) {
    seqlen_extrapolation_ =
        *(float*)(attr_map.at("seqlen_extrapolation").c_str());
    if (use_weight_) {
      LOG(ERROR) << "RotaryMulQueryOp : "
                 << "only online inv-freq calculation support "
                 << "extrapolation coe." << std::endl;
    }
    if (seqlen_extrapolation_ < 0.5f || seqlen_extrapolation_ > 128.f) {
      LOG(ERROR) << "RotaryMulQueryOp : "
                 << "Incoming sequence length extrapolation value = "
                 << seqlen_extrapolation_ << ". "
                 << "out of valid range = [0.5, 128]. "
                 << "please check seqlen_extrapolation attributes again."
                 << std::endl;
      return AsStatus::ALLSPARK_PARAM_ERROR;
    }
  }

  // attr.6 base
  base_ = 10000.f;
  if (attr_map.find("rotary_base") != attr_map.end()) {
    base_ = *(float*)(attr_map.at("rotary_base").c_str());
    if (use_weight_) {
      LOG(ERROR) << "RotaryMulQueryOp : "
                 << "only online inv-freq calculation require base as "
                 << "attributes." << std::endl;
    }
  }
  xlogn_ = -1;
  if (attr_map.find("logn_model_embedding") != attr_map.end()) {
    xlogn_ = *(int*)(attr_map.at("logn_model_embedding").c_str());
  }

  // attr.7 hidden_size
  if (attr_map.find("hidden_size") == attr_map.end()) {
    LOG(ERROR) << "RotaryMulQueryOp : can't find hidden_size attribute."
               << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
  hidden_size_ = *(int*)(attr_map.at("hidden_size").c_str());

  if (hidden_size_ % num_heads_) {
    LOG(ERROR) << "Invalid attribute in RotaryMulQueryOp. hidden_size : "
               << hidden_size_ << ", num_heads : " << num_heads_ << std::endl;
    return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }
  size_per_head_ = hidden_size_ / num_heads_;

  // attr.8 multi_query_group_num
  if (attr_map.find("multi_query_group_num") == attr_map.end()) {
    LOG(ERROR) << "RotaryAOp : can't find multi_query_group_num attribute."
               << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
  group_num_ = *(int*)(attr_map.at("multi_query_group_num").c_str());

  // attr.9 rope_ratio
  if (attr_map.find("rope_ratio") != attr_map.end()) {
    rope_ratio_ = *(float*)(attr_map.at("rope_ratio").c_str());
  }

  if (attr_map.find("invfreq_type") != attr_map.end()) {
    invfreq_type_ = *(int*)(attr_map.at("invfreq_type").c_str());
  }

  // backend switch
  DeviceType backend = ctx.GetDeviceType();
  switch (backend) {
    case DeviceType::CPU: {
      const CPUContext* cpu_ctx = static_cast<const CPUContext*>(ctx_);
      hidden_size_ /= cpu_ctx->GetNranks();
      num_heads_ /= cpu_ctx->GetNranks();
      if (group_num_ != 1) {
        group_num_ /= cpu_ctx->GetNranks();
      }
      break;
    }
    default:
      LOG(ERROR) << op_type_ << " Operator does not support "
                 << DeviceType_Name(backend) << " device type" << std::endl;
      return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }
  kv_stride_ = size_per_head_ * group_num_;
  int32_t flags = static_cast<int32_t>(AsTensorFlags::empty_flag);
  run_step_host_ = std::make_unique<AsTensor>(
      "run_step", DeviceType::CPU, DataType::INT32, DataMode::DENSE,
      Shape{ctx_->GetModelMaxBatch()}, flags);
  inv_freq_ = std::make_unique<AsTensor>(
      "inv_freq", backend, DataType::FLOAT32, DataMode::DENSE,
      Shape{ctx_->GetModelMaxBatch() * size_per_head_ / 2});
  run_step_ = std::make_unique<AsTensor>("run_step", backend, DataType::INT32,
                                         DataMode::DENSE,
                                         Shape{ctx_->GetModelMaxBatch()});
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus RotaryMulQueryOp::Reshape(RuntimeContext* runtime_ctx) {
  const AsTensor* x = tensor_map_->at(in_names_[0]).get();
  const Shape& x_shape = x->GetShape();
  Shape y_shape(x_shape);
  batch_size_ = y_shape[0];
  seq_len_ = y_shape[1];
  qkv_stride_ = y_shape[2];
  if (qkv_stride_ != hidden_size_ + 2 * kv_stride_) {
    LOG(ERROR) << "Invalid qkv_stride_ in DecOptMQAOp"
               << "qkv_strde = " << qkv_stride_
               << ", hidden_size = " << hidden_size_
               << ", kv_stride = " << kv_stride_ << std::endl;
    return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }
  tensor_map_->at(out_names_[0])->SetShape(std::move(y_shape));
  int batch_size = ctx_->GetModelMaxBatch();
  run_step_host_->SetShape(Shape{batch_size});
  run_step_->SetShape(Shape{batch_size});

  tensor_map_->at(out_names_[0])->SetShape(std::move(y_shape));
  if (!use_weight_) {
    std::vector<float> inv_freq_tmp =
        calculate_invfreq(base_, seq_len_, invfreq_type_);
    int inv_size = inv_freq_tmp.size();
    inv_freq_->SetShape(Shape({inv_size}));
    inv_freq_->CopyDataFrom(inv_freq_tmp.data(),
                            sizeof(float) * inv_freq_tmp.size(),
                            DeviceType::CPU, ctx_);
  } else {
    int inv_size = weights_[0]->GetShape().Count();
    inv_freq_->SetShape(Shape({inv_size}));
    if (weights_[0]->GetDataType() == DataType::FLOAT32) {
      inv_freq_->CopyDataFrom(weights_[0]->GetDataPtr(),
                              inv_size * sizeof(float),
                              weights_[0]->GetDeviceType(), ctx_);
    } else if (weights_[0]->GetDataType() == DataType::FLOAT16) {
      rotary_inv_freq_convert(inv_freq_.get(), (half*)weights_[0]->GetDataPtr(),
                              inv_size, weights_[0]->GetDeviceType(), ctx_);
    } else if (weights_[0]->GetDataType() == DataType::BFLOAT16) {
      rotary_inv_freq_convert(inv_freq_.get(),
                              (hie::bfloat16*)weights_[0]->GetDataPtr(),
                              inv_size, weights_[0]->GetDeviceType(), ctx_);
    } else {
      LOG(ERROR) << "RotaryMulQueryOp::Reshape: Not support DataType"
                 << weights_[0]->GetDataType() << std::endl;
      return AsStatus::ALLSPARK_RUNTIME_ERROR;
    }
  }
  ctx_->Synchronize();

  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus RotaryMulQueryOp::RunRotary(int run_batch_size) {
  float* inv_freq = (float*)inv_freq_->GetDataPtr();
  if (inv_freq_->GetShape().Count() == 0) inv_freq = nullptr;

  int* run_step = (int*)run_step_->GetDataPtr();
  int qkv_stride = qkv_stride_;
  int* batch_offset = nullptr;
  int offset = hidden_size_ * SizeofType(dtype_);
  void* q_buf = (char*)tensor_map_->at(in_names_[0])->GetDataPtr();
  void* k_buf = (char*)q_buf + hidden_size_ * SizeofType(dtype_);
  void* v_buf = (char*)k_buf + kv_stride_ * SizeofType(dtype_);
  void* outq_buf = (char*)tensor_map_->at(out_names_[0])->GetDataPtr();
  void* outk_buf = (char*)outq_buf + hidden_size_ * SizeofType(dtype_);
  void* outv_buf = (char*)outk_buf + kv_stride_ * SizeofType(dtype_);

  rotary_multiquery_launcher(dtype_, outq_buf, q_buf, inv_freq, batch_offset,
                             run_batch_size, seq_len_, run_step, hidden_size_,
                             num_heads_, size_per_head_, qkv_stride,
                             rotary_type_, xlogn_, ctx_);
  rotary_multiquery_launcher(dtype_, outk_buf, k_buf, inv_freq, batch_offset,
                             run_batch_size, seq_len_, run_step, hidden_size_,
                             group_num_, size_per_head_, qkv_stride,
                             rotary_type_, xlogn_, ctx_);
  rotary_multiquery_launcher(dtype_, outv_buf, v_buf, nullptr, batch_offset,
                             run_batch_size, seq_len_, run_step, hidden_size_,
                             group_num_, size_per_head_, qkv_stride,
                             rotary_type_, xlogn_, ctx_);
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus RotaryMulQueryOp::RunContext(RuntimeContext* runtime_ctx) {
  if (batch_size_ != 1) {
    LOG(ERROR) << "BatchMHAOp only support multibatch in decoder pharse, "
                  "not context pharse."
               << std::endl;
    return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }
  int freq_size = size_per_head_ / 2;
  int batch_size = 1;
  GenerateContext* gen_ctx = runtime_ctx->GetContextGenCtx();

  std::vector<int> run_step_tmp(batch_size);
  run_step_tmp[0] = gen_ctx->step;
  run_step_host_->SetShape(Shape{batch_size});
  run_step_->SetShape(Shape{batch_size});
  run_step_host_->CopyDataFrom(run_step_tmp.data(), sizeof(int) * batch_size,
                               DeviceType::CPU, ctx_);
  TensorUtils::DeepCopyWholeAsync(*run_step_, *run_step_host_, ctx_);
  RunRotary(batch_size);
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus RotaryMulQueryOp::RunDecoder(RuntimeContext* runtime_ctx) {
  int freq_size = size_per_head_ / 2;
  int batch_size = runtime_ctx->GetGenCtxListSize();
  std::vector<int> run_step_tmp(batch_size);
  for (int batch = 0; batch < batch_size; batch++) {
    GenerateContext* gen_ctx = runtime_ctx->GetGenCtx(batch);
    run_step_tmp[batch] = gen_ctx->step;
  }

  run_step_host_->SetShape(Shape{batch_size});
  run_step_->SetShape(Shape{batch_size});

  run_step_host_->CopyDataFrom(run_step_tmp.data(), sizeof(int) * batch_size,
                               DeviceType::CPU, ctx_);
  TensorUtils::DeepCopyWholeAsync(*run_step_, *run_step_host_, ctx_);
  RunRotary(batch_size);
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus RotaryMulQueryOp::Forward(RuntimeContext* runtime_ctx) {
  AsStatus status = AsStatus::ALLSPARK_SUCCESS;
  if (runtime_ctx->is_context) {
    status = RunContext(runtime_ctx);
  } else {
    status = RunDecoder(runtime_ctx);
  }
  return status;
}

REGISTER_OP("RotaryMulQuery", CPU, RotaryMulQueryOp)
}  // namespace allspark
