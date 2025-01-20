/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    gemm_lora_op_gpu.cpp
 */

#ifdef ENABLE_CUDA
#include "gemm_lora_op_gpu.h"

#include <core/kernel/cuda/gemm_lowp/gemm_a16w8_kernel.h>
#include <core/kernel/kernel.h>
#include <cuda/cuda_context.h>
#include <utility/check_cuda.h>
#include <utility/datatype_dispatcher.h>

#include "runtime/weight/weight_manager_lora.h"

namespace allspark {

extern AsStatus dense_gemm(DataType dtype, void* out, const void* in,
                           const void* bias, const AsTensor* weight, int m,
                           int n, int k, int lda, int ldb, int ldc, bool transA,
                           bool transB, int batch, float alpha,
                           const void* binary_in, UnaryType activation,
                           const DeviceContext* ctx);

AsStatus GemmLoraOpGPU::InitV2(const OperatorProto& op_proto,
                               const DeviceContext& ctx,
                               const TensorMap& weights_map,
                               TensorMap& weights_buffer,
                               TensorMap* tensor_map) {
  DLOG(INFO) << "GemmLoraOpGPU::InitV2" << std::endl;
  op_proto_.CopyFrom(op_proto);
  AS_CHECK_STATUS(GemmOpBase::InitV2(op_proto, ctx, weights_map, weights_buffer,
                                     tensor_map));

  // 计算QKV相关维度信息
  int nslice = ctx.GetNranks();
  assert(ctx.GetNumberHeads() * ctx.GetSizePerHead() % nslice == 0);
  assert(ctx.GetNumberGroups() * ctx.GetSizePerHead() % nslice == 0);
  q_outdim_size_ = ctx.GetNumberHeads() * ctx.GetSizePerHead() / nslice;
  k_outdim_size_ = ctx.GetNumberGroups() * ctx.GetSizePerHead() / nslice;
  v_outdim_size_ = k_outdim_size_;
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus GemmLoraOpGPU::Reshape(RuntimeContext* runtime_ctx) {
  DLOG(INFO) << "GemmLoraOpGPU::Reshape" << std::endl;
  //  根据batch里req用到的lora，把不在cuda中的lora换入
  auto batchsize =
      runtime_ctx->is_context ? 1 : runtime_ctx->GetGenCtxListSize();

  DLOG(INFO) << "batchsize=" << batchsize << std::endl;
  std::string lora_name_of_max_r = "";  // choose lora with max rank
  int max_lora_r = 0;
  for (auto i = 0; i < batchsize; i++) {
    // DLOG(INFO) <<  "runtime_ctx->is_context=" << std::boolalpha <<
    // runtime_ctx->is_context;
    std::shared_ptr<GenerateContext> gen_ctx =
        runtime_ctx->is_context ? runtime_ctx->GetContextGenCtx()
                                : runtime_ctx->GetGenCtx(i);
    auto lora_name = gen_ctx->gen_cfg.lora_name;
    DLOG(INFO) << i << ":r " << lora_name << std::endl;
    if (lora_name.empty())  // 加载了lora权重 但请求中可以不使用
      continue;
    DLOG(INFO) << "prepare to lora_manager_->GetHandleByName lora " << lora_name
               << std::endl;
    auto lora_weight_handle = lora_manager_->GetHandleByName(lora_name);
    DLOG(INFO) << "done lora_manager_->GetHandleByName lora " << lora_name
               << std::endl;
    auto& t_name = weight_names_[0];
    auto weight_tensor_p =
        lora_manager_->GetLoraTensorByName(lora_name, t_name);
    auto lora_r = std::min(weight_tensor_p->GetShape()[0],
                           weight_tensor_p->GetShape()[1]);
    DLOG(INFO) << "bs=" << batchsize << " i=" << i
               << " runtime_ctx->is_context=" << runtime_ctx->is_context
               << " lora_name=" << lora_name << " lora_r=" << lora_r;
    if (lora_r >= max_lora_r) {
      max_lora_r = lora_r;
      lora_name_of_max_r = lora_name;
    }
  }
  if (lora_name_of_max_r.empty()) {  // no lora
    DLOG(INFO) << "no lora, reshape and return" << std::endl;
    AS_CHECK_STATUS(GemmOpGPU::Reshape(runtime_ctx));
    DLOG(INFO) << "reshape done" << std::endl;
    return AsStatus::ALLSPARK_SUCCESS;
  }
  assert(lora_manager_);
  // assumes that key-names are identical between different LoRAs
  auto lora_weight_handle = lora_manager_->GetHandleByName(lora_name_of_max_r);
  weights_.clear();
  auto& t_name = weight_names_[0];
  DLOG(INFO) << "It's a real lora weight for lora " << lora_name_of_max_r
             << ", op " << op_name_ << ", lora_weight: " << t_name;
  auto weight_tensor_p =
      lora_manager_->GetLoraTensorByName(lora_name_of_max_r, t_name);
  weights_.emplace_back(weight_tensor_p.get());

  // now we have real lora weights, so re-init & reshape Base, to get the max
  // shapes for batch about GemmLora
  TensorMap stub_weight;
  AS_CHECK_STATUS(GemmOpBase::InitV2(op_proto_, *ctx_, stub_weight, stub_weight,
                                     tensor_map_));
  AS_CHECK_STATUS(GemmOpGPU::Reshape(runtime_ctx));
  kernel_launcher = dense_gemm;  // only support dense_gemm for Lora

  // AsTensor* batch_out_tensor = tensor_map_->at(out_names_[0]).get();
  // DLOG(INFO) << "reshape: batch_out_tensor:" << batch_out_tensor->ToString();
  DLOG(INFO) << "after reshape" << std::endl;
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus GemmLoraOpGPU::Forward(RuntimeContext* runtime_ctx) {
  DLOG(INFO) << "GemmLoraOpGPU::Forward" << std::endl;
  AsTensor* batch_in_tensor = tensor_map_->at(in_names_[0]).get();
  AsTensor* batch_out_tensor = tensor_map_->at(out_names_[0]).get();
  // DLOG(INFO) << "forward: batch_out_tensor:" << batch_out_tensor->ToString();
  void* in = batch_in_tensor->GetDataPtr();
  void* out = batch_out_tensor->GetDataPtr();

  const auto& shape_in = batch_in_tensor->GetShape();
  const auto& max_shape_out = batch_out_tensor->GetShape();
  assert(shape_in[0] == max_shape_out[0]);
  auto ndims_in = shape_in.Size();
  auto ndims_out = max_shape_out.Size();
  assert(ndims_in == 3 && ndims_in == ndims_out);
  auto input_batchsize = shape_in[0];

  const CUDAContext* gpu_ctx = static_cast<const CUDAContext*>(ctx_);
  cudaStream_t cu_stream = gpu_ctx->GetStream();
  cudaMemsetAsync(out, 0, batch_out_tensor->GetSizeInByte(),
                  cu_stream);  // clear whole output

  auto batch_in_stride = batch_in_tensor->GetShape().Count(1) *
                         SizeofType(batch_in_tensor->GetDataType());
  auto batch_out_stride = batch_out_tensor->GetShape().Count(1) *
                          SizeofType(batch_out_tensor->GetDataType());
  auto out_size_per_batch = batch_out_tensor->GetSizeInByte() / input_batchsize;

  auto in_bytes_per_data = SizeofType(batch_in_tensor->GetDataType());
  for (auto i = 0; i < input_batchsize; i++) {
    char* in_ptr = (char*)batch_in_tensor->GetDataPtr() + i * batch_in_stride;
    void* out_ptr =
        (char*)batch_out_tensor->GetDataPtr() + i * batch_out_stride;
    // GemmLora 不使用weights_, 使用lora_weight
    std::shared_ptr<GenerateContext> gen_ctx =
        runtime_ctx->is_context ? runtime_ctx->GetContextGenCtx()
                                : runtime_ctx->GetGenCtx(i);
    auto lora_name = gen_ctx->gen_cfg.lora_name;
    DLOG(INFO) << i << ":f " << lora_name << std::endl;
    if (lora_name.empty()) {
      continue;
    }
    assert(weight_names_.size() > 0);
    auto lora_weight_name = weight_names_[0];
    auto lora_weight =
        lora_manager_->GetLoraTensorByName(lora_name, lora_weight_name);
    bool has_lora_bias =
        lora_manager_->HasLoraBias(lora_name, lora_weight_name);
    DLOG(INFO) << lora_weight_name << " has_bias=" << has_lora_bias
               << std::endl;
    std::shared_ptr<AsTensor> lora_bias;
    if (has_lora_bias) {
      auto lora_bias_name = lora_manager_->GetBiasName(lora_weight_name);
      lora_bias = lora_manager_->GetLoraTensorByName(lora_name, lora_bias_name);
    }
    const auto& shape_w = lora_weight->GetShape();
    auto ndims_w = shape_w.Size();
    assert(ndims_w == 2);
    if (has_lora_bias) {
      const auto& shape_bias = lora_bias->GetShape();
      auto ndims_bias = shape_bias.Size();
      assert(ndims_bias == 1);
      assert(shape_w[1] == shape_bias[0]);
    }
    // 获取lora weight/bias OK

    if (lora_weight_name.rfind(".attention.self.lora_B") != std::string::npos) {
      // process qkv weight & bias cache
      for (auto iter = qkv_weight_cache_.begin();
           iter != qkv_weight_cache_.end();) {
        const auto& lora_name = iter->first.first;
        if (GetTaintedStatus(lora_name))
          iter = qkv_weight_cache_.erase(iter);
        else
          iter++;
      }
      for (auto iter = qkv_bias_cache_.begin();
           iter != qkv_bias_cache_.end();) {
        const auto& lora_name = iter->first.first;
        if (GetTaintedStatus(lora_name))
          iter = qkv_bias_cache_.erase(iter);
        else
          iter++;
      }

      // 已支持GROUP_VSPLIT
      assert(shape_w[1] == q_outdim_size_ + k_outdim_size_ + v_outdim_size_);
      assert(shape_w[1] == max_shape_out[2]);
      auto lora_r = shape_w[0];
      // shape_in[2] is not accurate (due to padding for LoRAs with different r
      // in one batch), use lora_r instead! assert(shape_in[2] == lora_r ||
      // shape_in[2] == lora_r * 3);

      // 复原成qkv 3个输入
      auto m = shape_in[1];
      auto k = lora_r;
      using QKV_DimType = dim_t[3];
      QKV_DimType n = {q_outdim_size_, k_outdim_size_,
                       v_outdim_size_};  // 支持MHA、QGA、MQA
      QKV_DimType n_prefix_sum = {0, q_outdim_size_,
                                  q_outdim_size_ + k_outdim_size_};
      auto lda = k * 3;
      QKV_DimType ldb = {n[0], n[1], n[2]};
      if (transB_) ldb[0] = ldb[1] = ldb[2] = k;
      QKV_DimType& ldc = n;

      std::vector<std::shared_ptr<AsTensor>> output_parts;
      for (int qkv_idx = 0; qkv_idx < 3; qkv_idx++) {
        std::shared_ptr<AsTensor> weight_t_part = nullptr;
        if (qkv_weight_cache_.count({lora_name, qkv_idx}) == 0) {
          weight_t_part = std::make_shared<AsTensor>(
              lora_weight->GetName() + "." + std::to_string(qkv_idx),
              lora_weight->GetDeviceType(), lora_weight->GetDataType(),
              lora_weight->GetDataMode(), Shape{k, n[qkv_idx]});
          TensorUtils::DeepCopyMatrix2D(*weight_t_part, *lora_weight,
                                        n_prefix_sum[qkv_idx], 0, ctx_);
          qkv_weight_cache_[std::make_pair(lora_name, qkv_idx)] = weight_t_part;
        }
        weight_t_part = qkv_weight_cache_.at({lora_name, qkv_idx});
        void* bias = nullptr;
        std::shared_ptr<AsTensor> bias_t_part = nullptr;
        if (has_lora_bias) {
          if (qkv_bias_cache_.count({lora_name, qkv_idx}) == 0) {
            bias_t_part = std::make_shared<AsTensor>(
                lora_bias->GetName() + "." + std::to_string(qkv_idx),
                lora_bias->GetDeviceType(), lora_bias->GetDataType(),
                lora_bias->GetDataMode(), Shape{n[qkv_idx]});
            TensorUtils::DeepCopyMatrix2D(*bias_t_part, *lora_bias,
                                          n_prefix_sum[qkv_idx], 0, ctx_);
            qkv_bias_cache_[std::make_pair(lora_name, qkv_idx)] = bias_t_part;
          }
          bias_t_part = qkv_bias_cache_.at({lora_name, qkv_idx});
          bias = bias_t_part->GetDataPtr();
        }
        auto out_t_part = std::make_shared<AsTensor>(
            batch_out_tensor->GetName() + "." + std::to_string(qkv_idx),
            batch_out_tensor->GetDeviceType(), batch_out_tensor->GetDataType(),
            batch_out_tensor->GetDataMode(), Shape{m, n[qkv_idx]});
        if (!use_quant_) {  // fp16, fp32
          DLOG(INFO) << lora_weight_name << " alpha_=" << alpha_ << std::endl;
          kernel_launcher(
              batch_in_tensor->GetDataType(), out_t_part->GetDataPtr(), in_ptr,
              bias, weight_t_part.get(), m, n[qkv_idx], k, lda, ldb[qkv_idx],
              ldc[qkv_idx], false, transB_, 1, alpha_ /* aka. lora_scaling */,
              nullptr, UNARYTYPE_UNDEFINED, ctx_);
        } else {  // use quantization
          throw AsException("lora quant not implemented");
        }
        output_parts.emplace_back(out_t_part);
        in_ptr += lora_r * in_bytes_per_data;
      }

      TensorUtils::ConcatMatrix2DColWise(*batch_out_tensor, i, output_parts,
                                         ctx_);
      // ctx_->Synchronize();

      // now qkv weights & bias cache loaded, clear tag
      RemoveTaintedStatus(lora_name);

    } else {
      auto m = shape_in[1];
      auto k = shape_w[0];
      auto n = shape_w[1];

      if (!use_quant_) {  // fp16 , fp32
        auto lda = k;
        auto ldb = transB_ ? k : n;
        auto ldc = n;
        void* bias = has_lora_bias ? lora_bias->GetDataPtr() : nullptr;
        assert(shape_w[1] <= max_shape_out[2]);  // maybe padding
        DLOG(INFO) << lora_weight_name << " alpha_=" << alpha_ << std::endl;
        kernel_launcher(batch_in_tensor->GetDataType(), out_ptr, in_ptr, bias,
                        lora_weight.get(), m, n, k, lda, ldb, ldc, false,
                        transB_, 1, alpha_ /* aka. lora_scaling */, nullptr,
                        UNARYTYPE_UNDEFINED, ctx_);
      } else {  // use quantization
        throw AsException("lora quant not implemented");
      }
    }
  }
  DLOG(INFO) << "after forward" << std::endl;
  return AsStatus::ALLSPARK_SUCCESS;
}

REGISTER_OP(GemmLora, CUDA, GemmLoraOpGPU)
}  // namespace allspark
#endif
