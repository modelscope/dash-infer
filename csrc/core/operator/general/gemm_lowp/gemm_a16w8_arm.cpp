/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    gemm_a16w8_arm.cpp
 */

#ifdef ENABLE_ARM_V84_V9
#include "gemm_a16w8_arm.h"

#include <core/kernel/cpu/gemm_lowp/gemm_a16w8_kernel.h>
#include <core/kernel/kernel.h>
#include <cpu/cpu_context.h>
#include <utility/datatype_dispatcher.h>

namespace allspark {
AsStatus GemmA16W8ARM::Init(const OperatorProto& op_proto,
                            const DeviceContext& ctx,
                            const TensorMap& weights_map,
                            TensorMap* tensor_map) {
  LOG(ERROR) << "GemmA16W8ARM only support InitV2()" << std::endl;
  return AsStatus::ALLSPARK_INVALID_CALL_ERROR;
}

AsStatus GemmA16W8ARM::InitV2(const OperatorProto& op_proto,
                              const DeviceContext& ctx,
                              const TensorMap& weights_map,
                              TensorMap& weights_buffer,
                              TensorMap* tensor_map) {
  DLOG(INFO) << "GemmA16W8ARM::InitV2()" << std::endl;
  AS_CHECK_STATUS(GemmA16W8Base::InitV2(op_proto, ctx, weights_map,
                                        weights_buffer, tensor_map));

  if (alpha_ != 1.0 || beta_ != 1.0) {
    LOG(ERROR) << "GemmA16W8ARM only support alpha=1 and beta==1."
               << " alpha: " << alpha_ << " beta: " << beta_ << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }

  K_pack_ = std::ceil(k_ / 8.0) * 8;
  int width = K_pack_ * 2;
  int height = n_ / 2 + n_ % 2;

  if (group_size_ == -1) {
    group_size_ = K_pack_;
  }

  const uint8_t* orig_weight_ptr =
      reinterpret_cast<uint8_t*>(weights_[0]->GetDataPtr());

  int scale_type = weights_[1]->GetDataType();
  switch (scale_type) {
    case DataType::FLOAT32: {
      float* orig_scale_ptr =
          reinterpret_cast<float*>(weights_[1]->GetDataPtr());
      float* orig_zero_ptr =
          reinterpret_cast<float*>(weights_[2]->GetDataPtr());

      const Shape& scale_shape = weights_[1]->GetShape();

      weight_bf16_packed_ = std::make_shared<AsTensor>(
          weights_[0]->GetName() + "_packed_bf16", DeviceType::CPU,
          DataType::BFLOAT16, weights_[0]->GetDataMode(),
          Shape({height, width}));
      TensorUtils::Memset(*weight_bf16_packed_, 0);

      hie::bfloat16* pack_weight_bf16_ptr =
          static_cast<hie::bfloat16*>(weight_bf16_packed_->GetDataPtr());

      PackWeightBf16(n_, k_, K_pack_, orig_weight_ptr, pack_weight_bf16_ptr,
                     orig_scale_ptr, orig_zero_ptr, group_size_);

#if TEST_TYPE_CVT_FP16  // temporary macro for test
      int fp16_w = (scale_shape[1] + 3) / 4 * 4;
      int fp16_h = scale_shape[0] + scale_shape[0] % 2;
      int64_t fp16_size = fp16_w * fp16_h * 2;

      Shape scale_shape_new = Shape({fp16_h, fp16_w});
      std::shared_ptr<AsTensor> scale = std::make_shared<AsTensor>(
          weights_[1]->GetName() + "_packed_fp16", DeviceType::CPU,
          DataType::FLOAT16, weights_[1]->GetDataMode(), scale_shape_new);
      TensorUtils::Memset(*scale, 0);

      std::shared_ptr<AsTensor> scaleXzp = std::make_shared<AsTensor>(
          weights_[2]->GetName() + "_packed_fp16", DeviceType::CPU,
          DataType::FLOAT16, weights_[2]->GetDataMode(), scale_shape_new);
      TensorUtils::Memset(*scaleXzp, 0);

      float16_t* fp16_scale_ptr = static_cast<float16_t*>(scale->GetDataPtr());
      float16_t* fp16_scaleXzp_ptr =
          static_cast<float16_t*>(scaleXzp->GetDataPtr());

      ProcessQuantParamFp16(orig_scale_ptr, orig_zero_ptr, scale_shape,
                            fp16_scale_ptr, fp16_scaleXzp_ptr);

      weights_[1]->Free();
      weights_[1]->SetName(scale->GetName());
      weights_[1]->SetDataType(DataType::FLOAT16);
      weights_[1]->SetShape(std::move(scale_shape_new));
      TensorUtils::DeepCopyWholeAsync(*weights_[1], *scale, ctx_);

      weights_[2]->Free();
      weights_[2]->SetName(scaleXzp->GetName());
      weights_[2]->SetDataType(DataType::FLOAT16);
      weights_[2]->SetShape(std::move(scale_shape_new));
      TensorUtils::DeepCopyWholeAsync(*weights_[2], *scaleXzp, ctx_);
#else
      std::shared_ptr<AsTensor> scale = std::make_shared<AsTensor>(
          weights_[1]->GetName() + "_packed", DeviceType::CPU,
          DataType::FLOAT32, weights_[1]->GetDataMode(), scale_shape);

      std::shared_ptr<AsTensor> scaleXzp = std::make_shared<AsTensor>(
          weights_[2]->GetName() + "_packed", DeviceType::CPU,
          DataType::FLOAT32, weights_[2]->GetDataMode(), scale_shape);

      float* new_scale_ptr = static_cast<float*>(scale->GetDataPtr());
      float* new_scaleXzp_ptr = static_cast<float*>(scaleXzp->GetDataPtr());

      ProcessQuantParam(orig_scale_ptr, orig_zero_ptr, scale_shape,
                        new_scale_ptr, new_scaleXzp_ptr);

      weights_[1]->SetName(scale->GetName());
      TensorUtils::DeepCopyWholeAsync(*weights_[1], *scale, ctx_);

      weights_[2]->SetName(scaleXzp->GetName());
      TensorUtils::DeepCopyWholeAsync(*weights_[2], *scaleXzp, ctx_);
#endif
      break;
    }
    default:
      LOG(ERROR) << "GemmA16W8CPU scale/zeropoint DataType Error\n";
      return AsStatus::ALLSPARK_PARAM_ERROR;
      break;
  }

  Shape weight_packed_shape = Shape({height, width});
  std::shared_ptr<AsTensor> weight_packed = std::make_shared<AsTensor>(
      weights_[0]->GetName() + "_packed", DeviceType::CPU, DataType::UINT8,
      weights_[0]->GetDataMode(), weight_packed_shape);
  TensorUtils::Memset(*weight_packed, 0);

  uint8_t* pack_weight_ptr = static_cast<uint8_t*>(weight_packed->GetDataPtr());

  PackWeight(n_, k_, K_pack_, orig_weight_ptr, pack_weight_ptr);

  weights_[0]->Free();
  weights_[0]->SetName(weight_packed->GetName());
  weights_[0]->SetDataType(weight_packed->GetDataType());
  weights_[0]->SetShape(std::move(weight_packed_shape));
  TensorUtils::DeepCopyWholeAsync(*weights_[0], *weight_packed, ctx_);

  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus GemmA16W8ARM::Reshape() {
  DLOG(INFO) << "GemmA16W8ARM::Reshape()" << std::endl;
  int yn = n_;
  AS_CHECK_STATUS(GemmA16W8Base::Reshape(yn));

  int64_t ws_size = 0;
  if (cpu::GemmA16W8Launcher::SelectKernel(m_, n_, k_, group_size_) != 0) {
    // Use fuse Kernel
    ws_size = cpu::GemmA16W8Launcher::GetWorkSpaceSize(m_, n_, k_, group_size_);
  }
  tensor_map_->at("workspace")->SetShape(Shape({ws_size}));
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus GemmA16W8ARM::Forward() {
  AsTensor* in_tensor = tensor_map_->at(in_names_[0]).get();
  void* in = in_tensor->GetDataPtr();
  void* rhs_qdata_ptr = weights_[0]->GetDataPtr();
  void* rhs_qdata_bf16_ptr = weight_bf16_packed_->GetDataPtr();
  void* rhs_scales_ptr = weights_[1]->GetDataPtr();
  void* rhs_scaleXzp_ptr = weights_[2]->GetDataPtr();
  void* bias = (weights_.size() == 4) ? weights_[3]->GetDataPtr() : nullptr;
  void* out = tensor_map_->at(out_names_[0])->GetDataPtr();
  void* ws_ptr = tensor_map_->at("workspace")->GetDataPtr();

  if (cpu::GemmA16W8Launcher::SelectKernel(m_, n_, k_, group_size_) != 0) {
    cpu::GemmA16W8Launcher::Run<float, uint8_t>(
        static_cast<const float*>(in),
        static_cast<const uint8_t*>(rhs_qdata_ptr),
        static_cast<const hie::bfloat16*>(rhs_qdata_bf16_ptr),
        static_cast<const void*>(rhs_scales_ptr),
        static_cast<const void*>(rhs_scaleXzp_ptr),
        static_cast<const float*>(bias), static_cast<float*>(out), uint32_t(m_),
        uint32_t(n_), uint32_t(k_), uint32_t(lda_), uint32_t(group_size_),
        static_cast<void*>(ws_ptr), activation_);
  }

  return AsStatus::ALLSPARK_SUCCESS;
}

/************************************/

void GemmA16W8ARM::PackWeight(const uint32_t N, const uint32_t K,
                              const uint32_t K_pack, const uint8_t* b_u8_unpack,
                              uint8_t* b_u8) {
  cpu::gemm_pack_weight_U8toU8_arm(N, K, K_pack, b_u8_unpack, b_u8);
  return;
}

void GemmA16W8ARM::PackWeightBf16(const uint32_t N, const uint32_t K,
                                  const uint32_t K_pack,
                                  const uint8_t* b_u8_unpack,
                                  hie::bfloat16* b_bf16, const float* scale,
                                  const float* zero, int group_size) {
  cpu::gemm_pack_weight_U8toBF16_arm(N, K, K_pack, b_u8_unpack, b_bf16, scale,
                                     zero, group_size);
  return;
}

void GemmA16W8ARM::ProcessQuantParam(float* scale, float* zero,
                                     Shape scale_shape, float* scale_new,
                                     float* zero_new) {
  int size = scale_shape.Count(0);
  for (int i = 0; i < size; i++) {
    scale_new[i] = scale[i];
    zero_new[i] = -scale[i] * zero[i];
  }
  return;
}

void GemmA16W8ARM::ProcessQuantParamFp16(float* scale, float* zero,
                                         Shape scale_shape,
                                         float16_t* scale_fp16,
                                         float16_t* scaleXzp_fp16) {
  cpu::parallel_for(scale_shape[1], [&](int n) {
    for (int subch_idx = 0; subch_idx < scale_shape[0]; subch_idx++) {
      int idx = subch_idx * scale_shape[1] + n;
      float scaleXzp = -scale[idx] * zero[idx];

      int fp16_w = (n / 4 * 2 + subch_idx % 2) * 4 + n % 4;
      int fp16_h = subch_idx / 2;
      int fp16_idx = fp16_h * ((scale_shape[1] + 3) / 4 * 4) * 2 + fp16_w;
      scale_fp16[fp16_idx] = (float16_t)scale[idx];
      scaleXzp_fp16[fp16_idx] = (float16_t)scaleXzp;
    }
  });
  return;
}

REGISTER_OP(GemmA16W8, CPU, GemmA16W8ARM)
}  // namespace allspark
#endif