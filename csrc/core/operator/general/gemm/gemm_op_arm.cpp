/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    gemm_op_arm.cpp
 */

#ifdef ENABLE_ARM_V84_V9
#include "gemm_op_arm.h"

#include <core/kernel/cpu/gemm_lowp/arm/gemm_kernel.h>
#include <core/kernel/kernel.h>
#include <cpu/cpu_context.h>
#include <utility/datatype_dispatcher.h>

using dnnl::memory;
namespace allspark {
AsStatus GemmOpARM::Init(const OperatorProto& op_proto,
                         const DeviceContext& ctx, const TensorMap& weights_map,
                         TensorMap* tensor_map) {
  LOG(ERROR) << "GemmOpARM only support InitV2()" << std::endl;
  return AsStatus::ALLSPARK_INVALID_CALL_ERROR;
}

AsStatus GemmOpARM::InitV2(const OperatorProto& op_proto,
                           const DeviceContext& ctx,
                           const TensorMap& weights_map,
                           TensorMap& weights_buffer, TensorMap* tensor_map,
                           RuntimeContext* runtime_ctx) {
  AS_CHECK_STATUS(GemmOpCPU::InitV2(op_proto, ctx, weights_map, weights_buffer,
                                    tensor_map, runtime_ctx));

  if (alpha_ != 1.0) {
    LOG(ERROR) << "GemmOpARM only support alpha=1."
               << " alpha: " << alpha_ << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }

  K_pack_ = std::ceil(k_ / 8.0) * 8;
  int width = K_pack_ * 2;
  int height = n_ / 2 + n_ % 2;

  if (ctx_->GetMatmulPrecision() == PrecisionLevel::MEDIUM_BF16) {
    const float* orig_weight_ptr =
        reinterpret_cast<float*>(weights_[0]->GetDataPtr());

    Shape weight_packed_shape = Shape({height, width});
    std::shared_ptr<AsTensor> weight_bf16_packed = std::make_shared<AsTensor>(
        weights_[0]->GetName() + "_packed_bf16", DeviceType::CPU,
        DataType::BFLOAT16, weights_[0]->GetDataMode(), weight_packed_shape);

    hie::bfloat16* pack_weight_bf16_ptr =
        static_cast<hie::bfloat16*>(weight_bf16_packed->GetDataPtr());

    PackWeightBf16(n_, k_, K_pack_, orig_weight_ptr, pack_weight_bf16_ptr);

    weights_[0]->Free();
    weights_[0]->SetName(weight_bf16_packed->GetName());
    weights_[0]->SetDataType(DataType::BFLOAT16);
    weights_[0]->SetShape(std::move(weight_packed_shape));
    TensorUtils::DeepCopyWholeAsync(*weights_[0], *weight_bf16_packed, ctx_);
  }
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus GemmOpARM::Reshape(RuntimeContext* runtime_ctx) {
  if (ctx_->GetMatmulPrecision() == PrecisionLevel::MEDIUM_BF16) {
    int yn = n_;
    AS_CHECK_STATUS(GemmOpBase::Reshape(yn));

    int bf16_elem_size = 2;
    int a_bf16_size = (m_ * K_pack_ + m_ % 2 * K_pack_) * bf16_elem_size;
    int64_t ws_size = a_bf16_size;
    tensor_map_->at("workspace")->SetShape(Shape({ws_size}));
  } else {
    AS_CHECK_STATUS(GemmOpCPU::Reshape(runtime_ctx));
  }
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus GemmOpARM::Forward(RuntimeContext* runtime_ctx) {
  if (ctx_->GetMatmulPrecision() == PrecisionLevel::MEDIUM_BF16) {
    AsTensor* in_tensor = tensor_map_->at(in_names_[0]).get();
    void* in = in_tensor->GetDataPtr();
    void* out = tensor_map_->at(out_names_[0])->GetDataPtr();
    void* bias = (weights_.size() == 2) ? weights_[1]->GetDataPtr() : nullptr;
    void* ws_ptr = tensor_map_->at("workspace")->GetDataPtr();

    if (is_split_k_) {
      in = (char*)in + k_ * rank_id_ * SizeofType(dtype_);
    }

    hie::bfloat16* pack_weight_bf16_ptr =
        static_cast<hie::bfloat16*>(weights_[0]->GetDataPtr());

    cpu::gemm_kernel_arm(m_, n_, k_, lda_, (float*)in, pack_weight_bf16_ptr,
                         (float*)out, (float*)bias, activation_,
                         static_cast<void*>(ws_ptr));
  } else {
    AS_CHECK_STATUS(GemmOpCPU::Forward(runtime_ctx));
  }

  return AsStatus::ALLSPARK_SUCCESS;
}

/************************************/

void GemmOpARM::PackWeightBf16(const uint32_t N, const uint32_t K,
                               const uint32_t K_pack, const float* b_fp32,
                               hie::bfloat16* b_bf16) {
  cpu::gemm_pack_weight_FP32toBF16_arm(N, K, K_pack, b_fp32, b_bf16);
}

REGISTER_OP(Gemm, CPU, GemmOpARM)
}  // namespace allspark
#endif
