/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    gemm_op_x86_spr.cpp
 */

#if defined(__x86_64__) || defined(_M_X64)
#include "gemm_op_x86_spr.h"

#include <core/kernel/kernel.h>
#include <cpu/cpu_common.h>
#include <cpu/cpu_context.h>
#include <utility/datatype_dispatcher.h>

#include "bgemm_f32bf16f32_simple.h"
#include "hgemm_f32f16f32_simple.h"

using dnnl::memory;
using tag = memory::format_tag;
#define USE_ONEDNN_BF16_GEMM 1
namespace allspark {

template <typename T1, typename T2>
static void convert_datatype(T1* input, T2* output, int64_t num_elements,
                             const DeviceContext& ctx) {
  const CPUContext* cpu_ctx = static_cast<const CPUContext*>(&ctx);
  int num_threads = cpu_ctx->GetNumThread();
  int64_t num_elem_per_thread = std::ceil(num_elements * 1.0 / num_threads);
  cpu::parallel_for(num_threads, [&](int n) {
    int64_t min_idx = n * num_elem_per_thread;
    int64_t max_idx = std::min((n + 1) * num_elem_per_thread, num_elements);
    for (int64_t i = min_idx; i < max_idx; i++) {
      output[i] = (T2)input[i];
    }
  });
}

AsStatus GemmOpSpr::Init(const OperatorProto& op_proto,
                         const DeviceContext& ctx, const TensorMap& weights_map,
                         TensorMap* tensor_map) {
  LOG(ERROR) << "GemmOpARM only support InitV2()" << std::endl;
  return AsStatus::ALLSPARK_INVALID_CALL_ERROR;
}

AsStatus GemmOpSpr::InitV2(const OperatorProto& op_proto,
                           const DeviceContext& ctx,
                           const TensorMap& weights_map,
                           TensorMap& weights_buffer, TensorMap* tensor_map) {
  DLOG(INFO) << "GemmOpSpr::InitV2()" << std::endl;

  if (!is_spr_ && ctx.GetMatmulPrecision() == PrecisionLevel::MEDIUM_FP16) {
    LOG(WARNING) << "Current CPU does not support fp16 GEMM";
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }

  if (ctx.GetMatmulPrecision() == PrecisionLevel::MEDIUM_BF16) {
    weight_data_type_ = DataType::BFLOAT16;
  } else if (ctx.GetMatmulPrecision() == PrecisionLevel::MEDIUM_FP16) {
    weight_data_type_ = DataType::FLOAT16;
  } else {
    weight_data_type_ = DataType::FLOAT32;
  }

  if (weight_data_type_ == DataType::FLOAT32) {
    AS_CHECK_STATUS(GemmOpCPU::InitV2(op_proto, ctx, weights_map,
                                      weights_buffer, tensor_map));
  } else if (weight_data_type_ == DataType::BFLOAT16) {
#if USE_ONEDNN_BF16_GEMM
    // use onednn bf16 gemm
    AS_CHECK_STATUS(GemmOpCPU::InitV2(op_proto, ctx, weights_map,
                                      weights_buffer, tensor_map));
#else
#ifdef ENABLE_BF16
    AS_CHECK_STATUS(GemmOpBase::InitV2(op_proto, ctx, weights_map,
                                       weights_buffer, tensor_map));

    // use intrinsic bf16 gemm
    AsTensor* mutable_weight = const_cast<AsTensor*>(weights_[0]);
    Shape weight_shape = weights_[0]->GetShape();

    // fp32 -> bf16
    auto as_weight_bf16 = std::make_unique<AsTensor>(
        weights_[0]->GetName() + "_bf16", weights_[0]->GetDeviceType(),
        DataType::BFLOAT16, weights_[0]->GetDataMode(), weight_shape);

    float* wei_fp32_ptr = (float*)mutable_weight->GetDataPtr();
    bfloat16_t* wei_bf16_ptr = (bfloat16_t*)as_weight_bf16->GetDataPtr();
    convert_datatype(wei_fp32_ptr, wei_bf16_ptr, weight_shape.Count(), ctx);

    auto as_weight_pack = std::make_unique<AsTensor>(
        weights_[0]->GetName() + "_bf16_pack", *as_weight_bf16);
    ig_bgemm_f32bf16f32_packb(transB_, n_, k_,
                              (const bfloat16_t*)as_weight_bf16->GetDataPtr(),
                              ldb_, (bfloat16_t*)as_weight_pack->GetDataPtr());

    mutable_weight->Free();
    mutable_weight->SetName(as_weight_pack->GetName());
    mutable_weight->SetDataType(DataType::BFLOAT16);
    mutable_weight->SetShape(std::move(weight_shape));
    TensorUtils::DeepCopyWholeAsync(*mutable_weight, *as_weight_pack, ctx);
#endif
#endif
  } else if (weight_data_type_ == DataType::FLOAT16) {
#ifdef ENABLE_FP16
    AS_CHECK_STATUS(GemmOpBase::InitV2(op_proto, ctx, weights_map,
                                       weights_buffer, tensor_map));

    // intel gemm to perform gemm op
    AsTensor* mutable_weight = const_cast<AsTensor*>(weights_[0]);
    Shape weight_shape = weights_[0]->GetShape();

    // fp32 -> fp16
    auto as_weight_fp16 = std::make_unique<AsTensor>(
        weights_[0]->GetName() + "_fp16", weights_[0]->GetDeviceType(),
        DataType::FLOAT16, weights_[0]->GetDataMode(), weight_shape);
    TensorUtils::Memset(*as_weight_fp16, 0);

    float* wei_fp32_ptr = (float*)mutable_weight->GetDataPtr();
    float16_t* wei_fp16_ptr = (float16_t*)as_weight_fp16->GetDataPtr();
    convert_datatype(wei_fp32_ptr, wei_fp16_ptr,
                     mutable_weight->GetShape().Count(), ctx);

    auto as_weight_pack = std::make_unique<AsTensor>(
        weights_[0]->GetName() + "_fp16_pack", *as_weight_fp16);
    ig_hgemm_f32f16f32_packb(transB_, n_, k_,
                             (const float16_t*)as_weight_fp16->GetDataPtr(),
                             ldb_, (float16_t*)as_weight_pack->GetDataPtr());

    mutable_weight->Free();
    mutable_weight->SetName(as_weight_pack->GetName());
    mutable_weight->SetDataType(DataType::FLOAT16);
    mutable_weight->SetShape(std::move(weight_shape));
    TensorUtils::DeepCopyWholeAsync(*mutable_weight, *as_weight_pack, ctx_);
#endif
  }

  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus GemmOpSpr::Reshape() {
  if (weight_data_type_ == DataType::FLOAT32) {
    AS_CHECK_STATUS(GemmOpCPU::Reshape());
  } else if (weight_data_type_ == DataType::BFLOAT16) {
#if USE_ONEDNN_BF16_GEMM
    // use onednn bf16 gemm
    AS_CHECK_STATUS(GemmOpCPU::Reshape());
#else
    // use intrinsic bf16 gemm
    AS_CHECK_STATUS(GemmOpBase::Reshape(n_));
#endif
  } else if (weight_data_type_ == DataType::FLOAT16) {
    AS_CHECK_STATUS(GemmOpBase::Reshape(n_));
  } else {
    LOG(ERROR) << "Unsupported matmul precision";
    return AsStatus::ALLSPARK_INVALID_CALL_ERROR;
  }
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus GemmOpSpr::Forward() {
  // DLOG(INFO) << "GemmOpSpr::Forward, m: " << m_
  //            << ", GetMatmulPrecision: " << ctx_->GetMatmulPrecision();

  AsTensor* in_tensor = tensor_map_->at(in_names_[0]).get();
  void* in = in_tensor->GetDataPtr();
  void* out = tensor_map_->at(out_names_[0])->GetDataPtr();
  void* bin_in = (in_names_.size() == 2)
                     ? tensor_map_->at(in_names_[1])->GetDataPtr()
                     : nullptr;
  if (is_split_k_) {
    in = (char*)in + k_ * rank_id_ * SizeofType(dtype_);
  }
  if (weight_data_type_ == DataType::FLOAT32) {
    // use onednn fp32 gemm
    AS_CHECK_STATUS(GemmOpCPU::Forward());
  } else if (weight_data_type_ == DataType::BFLOAT16) {
#if USE_ONEDNN_BF16_GEMM
    // use onednn bf16 gemm
    AS_CHECK_STATUS(GemmOpCPU::Forward());
#else
#ifdef ENABLE_BF16
    // use intrinsic bf16 gemm
    const AsTensor* weight = static_cast<const AsTensor*>(weights_[0]);
    void* bias = (weights_.size() == 2) ? weights_[1]->GetDataPtr() : nullptr;
    if (bias) {
      // either activation or binary_type is supported
      // bias is not supported with binary_type MUL
      if (activation_ == UnaryType::RELU) {
        ig_bgemm_f32bf16f32_compute_biasadd_relu(
            transB_, m_, n_, k_, 1.0f, (const float*)in, lda_,
            (const bfloat16_t*)weight->GetDataPtr(), 0.0f, (float*)out, ldc_,
            (const float*)bias);
      } else if (binary_type_ == BinaryType::ADD) {
        ig_bgemm_f32bf16f32_compute_residential(
            transB_, m_, n_, k_, 1.0f, (const float*)in, lda_,
            (const float16_t*)weight->GetDataPtr(), 0.0f, (float*)out, ldc_,
            (const float*)bias, (const float*)bin_in, ldc_);
      } else if (binary_type_ == BinaryType::MUL) {
        LOG(ERROR) << "Unsupported matmul precision with binary_type MUL";
        return AsStatus::ALLSPARK_INVALID_CALL_ERROR;
      } else {
        ig_bgemm_f32bf16f32_compute_biasadd(
            transB_, m_, n_, k_, 1.0f, (const float*)in, lda_,
            (const bfloat16_t*)weight->GetDataPtr(), 0.0f, (float*)out, ldc_,
            (const float*)bias);
      }
    } else {
      // either activation or binary_type is supported
      if (activation_ == UnaryType::SILU) {
        ig_bgemm_f32bf16f32_compute_silu(
            transB_, m_, n_, k_, 1.0f, (const float*)in, lda_,
            (const bfloat16_t*)weight->GetDataPtr(), 0.0f, (float*)out, ldc_);
      } else if (binary_type_ == BinaryType::ADD) {
        ig_bgemm_f32bf16f32_compute_residential(
            transB_, m_, n_, k_, 1.0f, (const float*)in, lda_,
            (const float16_t*)weight->GetDataPtr(), 0.0f, (float*)out, ldc_,
            (const float*)bias, (const float*)bin_in, ldc_);
      } else if (binary_type_ == BinaryType::MUL) {
        ig_bgemm_f32bf16f32_compute_resmul(
            transB_, m_, n_, k_, 1.0f, (const float*)in, lda_,
            (const float16_t*)weight->GetDataPtr(), 0.0f, (float*)out, ldc_,
            (const float*)bin_in, ldc_);
      } else {
        ig_bgemm_f32bf16f32_compute(
            transB_, m_, n_, k_, 1.0f, (const float*)in, lda_,
            (const bfloat16_t*)weight->GetDataPtr(), 0.0f, (float*)out, ldc_);
      }
    }
#endif
#endif
  } else if (weight_data_type_ == DataType::FLOAT16) {
#ifdef ENABLE_FP16
    // use intrinsic fp16 gemm
    const AsTensor* weight = static_cast<const AsTensor*>(weights_[0]);
    void* bias = (weights_.size() == 2) ? weights_[1]->GetDataPtr() : nullptr;
    if (bias) {
      // either activation or binary_type is supported
      // bias is not supported with binary_type MUL
      if (activation_ == UnaryType::RELU) {
        ig_hgemm_f32f16f32_compute_biasadd_relu(
            transB_, m_, n_, k_, 1.0f, (const float*)in, lda_,
            (const float16_t*)weight->GetDataPtr(), 0.0f, (float*)out, ldc_,
            (const float*)bias);
      } else if (binary_type_ == BinaryType::ADD) {
        ig_hgemm_f32f16f32_compute_residential(
            transB_, m_, n_, k_, 1.0f, (const float*)in, lda_,
            (const float16_t*)weight->GetDataPtr(), 0.0f, (float*)out, ldc_,
            (const float*)bias, (const float*)bin_in, ldc_);
      } else if (binary_type_ == BinaryType::MUL) {
        LOG(ERROR) << "Unsupported matmul precision with binary_type MUL";
        return AsStatus::ALLSPARK_INVALID_CALL_ERROR;
      } else {
        ig_hgemm_f32f16f32_compute_biasadd(
            transB_, m_, n_, k_, 1.0f, (const float*)in, lda_,
            (const float16_t*)weight->GetDataPtr(), 0.0f, (float*)out, ldc_,
            (const float*)bias);
      }
    } else {
      // either activation or binary_type is supported
      if (activation_ == UnaryType::SILU) {
        ig_hgemm_f32f16f32_compute_silu(
            transB_, m_, n_, k_, 1.0f, (const float*)in, lda_,
            (const float16_t*)weight->GetDataPtr(), 0.0f, (float*)out, ldc_);
      } else if (binary_type_ == BinaryType::ADD) {
        ig_hgemm_f32f16f32_compute_residential(
            transB_, m_, n_, k_, 1.0f, (const float*)in, lda_,
            (const float16_t*)weight->GetDataPtr(), 0.0f, (float*)out, ldc_,
            (const float*)bias, (const float*)bin_in, ldc_);
      } else if (binary_type_ == BinaryType::MUL) {
        ig_hgemm_f32f16f32_compute_resmul(
            transB_, m_, n_, k_, 1.0f, (const float*)in, lda_,
            (const float16_t*)weight->GetDataPtr(), 0.0f, (float*)out, ldc_,
            (const float*)bin_in, ldc_);
      } else {
        ig_hgemm_f32f16f32_compute(transB_, m_, n_, k_, 1.0f, (const float*)in,
                                   lda_, (const float16_t*)weight->GetDataPtr(),
                                   0.0f, (float*)out, ldc_);
      }
    }
#endif
  } else {
    LOG(ERROR) << "Unsupported matmul precision";
    return AsStatus::ALLSPARK_INVALID_CALL_ERROR;
  }
  return AsStatus::ALLSPARK_SUCCESS;
}

REGISTER_OP(Gemm, CPU, GemmOpSpr)
}  // namespace allspark
#endif
