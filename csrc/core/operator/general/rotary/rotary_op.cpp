/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    rotary_op.cpp
 */

#include "rotary_op.h"  // NOLINT

#include <core/kernel/kernel.h>
#include <utility/datatype_dispatcher.h>

#include <cmath>
#ifdef ENABLE_CUDA
#include <cuda/cuda_context.h>
#else
#include <common/float16.h>
#endif
#include <cpu/cpu_context.h>
namespace allspark {
void rotary_launcher(DataType dtype, void* out, void* in, float* inv_freq,
                     int* batch_offset, int batch_size, int seq_len,
                     int* step_list, int hidden_size, int num_heads,
                     int size_per_head, int input_len, int qkv_stride,
                     int rotary_type, float rotary_pct, int xlogn,
                     int* positions, int mrope_size, int* mrope_section,
                     const DeviceContext* ctx) {
  DeviceType backend = ctx->GetDeviceType();
  if (backend == DeviceType::CUDA) {
#ifdef ENABLE_CUDA
    const CUDAContext* gpu_ctx = static_cast<const CUDAContext*>(ctx);
    cudaStream_t cu_stream = gpu_ctx->GetStream();
    switch (rotary_type) {
      case RotaryType::base: {
        if ((hidden_size * SizeofType(dtype)) % 16 == 0) {
          auto functor = [&]<typename T>() {
            cuda::RotaryOptEmbedding<T>(
                (T*)out, (T*)in, inv_freq, batch_offset, batch_size, seq_len,
                num_heads, size_per_head, step_list, qkv_stride, xlogn,
                positions, mrope_size, mrope_section, cu_stream);
          };
          DispatchCUDA(dtype, functor);
        } else {
          LOG(ERROR) << "rotary_launcher (hidden_size * "
                        "SizeofType(dtype)) % 16 != 0,abort()"
                     << std::endl;
          abort();
        }
        break;
      }
      // case RotaryType::mrope: {
      //   // only for context
      //   LOG(INFO) << "RotaryType::mrope";
      //   auto functor = [&]<typename T>() {
      //     cuda::RotaryMultimodalSections<T>(
      //         (T*)out, (T*)in, inv_freq, batch_size, seq_len, num_heads,
      //         size_per_head, qkv_stride, positions, mrope_size,
      //         mrope_section, cu_stream);
      //   };
      //   DispatchCUDA(dtype, functor);
      //   break;
      // }
      default: {
        LOG(ERROR) << "RotaryOp (GPU): not support rotary_type" << std::endl;
        break;
      }
    }
#endif
  } else if (backend == DeviceType::CPU) {
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
    // 找不到默认为0,即default类型，适配老版本
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
    if (use_weight_) {
      // LOG(INFO) << "RotaryOp : "
      //           << "only online inv-freq calculation support
      //           extrapolation coe. "
      //           << std::endl;
    }
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
    if (use_weight_) {
      // LOG(INFO) << "RotaryOp : "
      //           << "only online inv-freq calculation require base as
      //           attributes. "
      //           << std::endl;
    }
  }
  xlogn_ = -1;
  if (attr_map.find("logn_model_embedding") != attr_map.end()) {
    xlogn_ = *(int*)(attr_map.at("logn_model_embedding").c_str());
  }
  // attr.8 multi_query_group_num
  if (attr_map.find("multi_query_group_num") != attr_map.end()) {
    group_num_ = *(int*)(attr_map.at("multi_query_group_num").c_str());
  } else {
    group_num_ = num_heads_;
  }

  // attr.9 rope_ratio
  if (attr_map.find("rope_ratio") != attr_map.end()) {
    rope_ratio_ = *(float*)(attr_map.at("rope_ratio").c_str());
  }
  invfreq_type_ = base_rotary;
  if (attr_map.find("invfreq_type") != attr_map.end()) {
    invfreq_type_ = *(int*)(attr_map.at("invfreq_type").c_str());
  }
  if (attr_map.find("original_max_position_embeddings") != attr_map.end()) {
    original_max_position_embeddings_ =
        *(int*)(attr_map.at("original_max_position_embeddings").c_str());
  }
  std::vector<int> mrope_section;
  mrope_size_ = 0;
  if (attr_map.find("mrope_section_size") != attr_map.end()) {
    mrope_size_ = *(int*)(attr_map.at("mrope_section_size").c_str());
    mrope_section.resize(mrope_size_ + 1);
    mrope_section[0] = 0;
    for (int i = 1; i <= mrope_size_; i++) {
      std::string mrope_section_name = "mrope_section_" + std::to_string(i - 1);
      if (attr_map.find(mrope_section_name) == attr_map.end()) {
        LOG(ERROR) << "RotaryOp : "
                   << " can't find " << mrope_section_name << std::endl;
      }
      mrope_section[i] = *(int*)(attr_map.at(mrope_section_name).c_str());
    }
    for (int i = 1; i <= mrope_size_; i++) {
      mrope_section[i] += mrope_section[i - 1];
    }
    if (mrope_section[mrope_size_] != size_per_head_ / 2) {
      LOG(ERROR) << "RotaryOp : "
                 << " sum(mrope_section) = " << mrope_section[mrope_size_]
                 << " != " << size_per_head_ / 2 << std::endl;
    }
  }
  size_per_head_ = ctx.GetSizePerHead();
  // backend switch
  DeviceType backend = ctx.GetDeviceType();

  // mrope for vl
  mrope_section_ =
      std::make_unique<AsTensor>("mrope_section_", backend, DataType::INT32,
                                 DataMode::DENSE, Shape{mrope_size_ + 1});
  mrope_position_ = std::make_unique<AsTensor>(
      "mrope_position_", backend, DataType::INT32, DataMode::DENSE, Shape{0});
  switch (backend) {
#ifdef ENABLE_CUDA
    case DeviceType::CUDA: {
      const CUDAContext* gpu_ctx = static_cast<const CUDAContext*>(ctx_);
      num_heads_ /= gpu_ctx->GetNranks();
      if (group_num_ != 1) {
        group_num_ /= gpu_ctx->GetNranks();
      }
      if (mrope_size_ != 0) {
        cudaMemcpyAsync((char*)mrope_section_->GetDataPtr(),
                        (char*)mrope_section.data(),
                        (mrope_size_ + 1) * sizeof(int), cudaMemcpyHostToDevice,
                        gpu_ctx->GetStream());
      }
      break;
    }
#endif
    case DeviceType::CPU: {
      const CPUContext* cpu_ctx = static_cast<const CPUContext*>(ctx_);
      num_heads_ /= cpu_ctx->GetNranks();
      if (group_num_ != 1) {
        group_num_ /= cpu_ctx->GetNranks();
      }
      if (mrope_size_ != 0) {
        memcpy(mrope_section_->GetDataPtr(), mrope_section.data(),
               sizeof(int) * (mrope_size_ + 1));
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
AsStatus RotaryOp::Reshape(RuntimeContext* runtime_ctx) {
  const AsTensor* x = tensor_map_->at(in_names_[0]).get();
  const Shape& x_shape = x->GetShape();
  Shape y_shape(x_shape);
  batch_size_ = y_shape[0];
  seq_len_ = y_shape[1];
  qkv_stride_ = y_shape[2];
  if (qkv_stride_ != hidden_size_ + 2 * kv_stride_) {
    LOG(ERROR) << "Invalid qkv_stride_ in RotaryOp"
               << "qkv_strde = " << qkv_stride_
               << ", hidden_size = " << hidden_size_
               << ", kv_stride = " << kv_stride_ << std::endl;
    return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }
  tensor_map_->at(out_names_[0])->SetShape(std::move(y_shape));
  mrope_position_->SetShape(Shape{mrope_size_, seq_len_});
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus RotaryOp::RunRotary(int run_batch_size, AsTensor* rotary_step,
                             AsTensor* rotary_inv_freq,
                             bool use_positions = false) {
  int* run_step = (int*)rotary_step->GetDataPtr();
  float* inv_freq = (float*)rotary_inv_freq->GetDataPtr();
  int qkv_stride = qkv_stride_;
  int* batch_offset = nullptr;
  void* q_buf = (char*)tensor_map_->at(in_names_[0])->GetDataPtr();
  void* k_buf = (char*)q_buf + hidden_size_ * SizeofType(dtype_);
  void* v_buf = (char*)k_buf + kv_stride_ * SizeofType(dtype_);
  void* outq_buf = (char*)tensor_map_->at(out_names_[0])->GetDataPtr();
  void* outk_buf = (char*)outq_buf + hidden_size_ * SizeofType(dtype_);
  void* outv_buf = (char*)outk_buf + kv_stride_ * SizeofType(dtype_);
  int* positions =
      use_positions ? (int*)mrope_position_->GetDataPtr() : nullptr;
  int* mrope_section = (int*)mrope_section_->GetDataPtr();
  rotary_launcher(dtype_, outq_buf, q_buf, inv_freq, batch_offset,
                  run_batch_size, seq_len_, run_step, hidden_size_, num_heads_,
                  size_per_head_, 0, qkv_stride, rotary_type_, rotary_pct_,
                  xlogn_, positions, mrope_size_, mrope_section, ctx_);
  rotary_launcher(dtype_, outk_buf, k_buf, inv_freq, batch_offset,
                  run_batch_size, seq_len_, run_step, hidden_size_, group_num_,
                  size_per_head_, 0, qkv_stride, rotary_type_, rotary_pct_, -1,
                  positions, mrope_size_, mrope_section, ctx_);
  rotary_launcher(dtype_, outv_buf, v_buf, nullptr, batch_offset,
                  run_batch_size, seq_len_, run_step, hidden_size_, group_num_,
                  size_per_head_, 0, qkv_stride, rotary_type_, rotary_pct_, -1,
                  positions, mrope_size_, mrope_section, ctx_);
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus RotaryOp::RunContext(RuntimeContext* runtime_ctx) {
  if (batch_size_ != 1) {
    LOG(ERROR) << "BatchMHAOp only support multibatch in decoder pharse, "
                  "not context pharse."
               << std::endl;
    return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }
  std::shared_ptr<GenerateContext> gen_ctx = runtime_ctx->GetContextGenCtx();

  seq_len_ += gen_ctx->prefix_len;
  TensorListMap extra_embedding = gen_ctx->request->extra_embedding;
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
    std::shared_ptr<GenerateContext> gen_ctx = runtime_ctx->GetContextGenCtx();
    std::vector<float> inv_freq_tmp =
        calculate_invfreq(base_, gen_ctx->input_len, invfreq_type_);
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
  if (extra_embedding.empty() || mrope_size_ == 0) {
    // No richembedding or don't need mrope
    RunRotary(1, rotary_step, rotary_inv_freq);
  } else {
    if (extra_embedding.count("positions") == 0) {
      LOG(ERROR) << "Not find positions in mm_info, please check gen_cfg "
                 << std::endl;
      return AsStatus::ALLSPARK_RUNTIME_ERROR;
    }

    AsTensor* pos_tensor = extra_embedding["positions"][0].get();
    gen_ctx->real_input_len =
        ((int*)pos_tensor->GetDataPtr())[(gen_ctx->input_len - 1)] +
        1;  // last toekn index
    DeviceType backend = ctx_->GetDeviceType();
    if (backend == DeviceType::CUDA) {
#ifdef ENABLE_CUDA
      AS_CHECK_CUDA(cudaMemcpyAsync(
          (char*)mrope_position_->GetDataPtr(), (char*)pos_tensor->GetDataPtr(),
          pos_tensor->GetShape().Count() * sizeof(int), cudaMemcpyHostToDevice,
          static_cast<const CUDAContext*>(ctx_)->GetStream()));
#endif
    } else if (backend == DeviceType::CPU) {
      memcpy(mrope_position_->GetDataPtr(), pos_tensor->GetDataPtr(),
             pos_tensor->GetShape().Count() * sizeof(int));
    }
    RunRotary(1, rotary_step, rotary_inv_freq, true);
  }

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
      std::shared_ptr<GenerateContext> gen_ctx = runtime_ctx->GetGenCtx(batch);
      std::vector<float> inv_freq_one =
          calculate_invfreq(base_, gen_ctx->step, invfreq_type_);
      for (int j = 0; j < freq_size; j++) {
        inv_freq_tmp[batch * freq_size + j] = inv_freq_one[j];
      }
      run_step_tmp[batch] =
          gen_ctx->real_input_len + (gen_ctx->step - gen_ctx->input_len);
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
// REGISTER_OP(RotaryMulQuery, CUDA, RotaryOp)
// REGISTER_OP(RotaryMulQuery, CPU, RotaryOp)
REGISTER_OP(Rotary, CUDA, RotaryOp)
REGISTER_OP(Rotary, CPU, RotaryOp)
}  // namespace allspark
