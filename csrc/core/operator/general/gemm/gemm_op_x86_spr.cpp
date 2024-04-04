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

using dnnl::memory;
using tag = memory::format_tag;

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

  if (ctx.GetMatmulPrecision() == PrecisionLevel::MEDIUM_BF16) {
    weight_data_type_ = DataType::BFLOAT16;
  } else {
    weight_data_type_ = DataType::FLOAT32;
  }

  if (weight_data_type_ == DataType::FLOAT32) {
    AS_CHECK_STATUS(GemmOpCPU::InitV2(op_proto, ctx, weights_map,
                                      weights_buffer, tensor_map));
  } else if (weight_data_type_ == DataType::BFLOAT16) {
    // use onednn bf16 gemm
    AS_CHECK_STATUS(GemmOpCPU::InitV2(op_proto, ctx, weights_map,
                                      weights_buffer, tensor_map));
  }

  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus GemmOpSpr::Reshape() {
  if (weight_data_type_ == DataType::FLOAT32) {
    AS_CHECK_STATUS(GemmOpCPU::Reshape());
  } else if (weight_data_type_ == DataType::BFLOAT16) {
    AS_CHECK_STATUS(GemmOpCPU::Reshape());
  } else if (weight_data_type_ == DataType::FLOAT16) {
    AS_CHECK_STATUS(GemmOpBase::Reshape(n_));
  } else {
    LOG(ERROR) << "Unsupported matmul precision";
    return AsStatus::ALLSPARK_INVALID_CALL_ERROR;
  }
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus GemmOpSpr::Forward() {
  DLOG(INFO) << "GemmOpSpr::Forward, m: " << m_
             << ", GetMatmulPrecision: " << ctx_->GetMatmulPrecision();

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
    AS_CHECK_STATUS(GemmOpCPU::Forward());
  } else {
    LOG(ERROR) << "Unsupported matmul precision";
    return AsStatus::ALLSPARK_INVALID_CALL_ERROR;
  }
  return AsStatus::ALLSPARK_SUCCESS;
}

REGISTER_OP("Gemm", CPU, GemmOpSpr)
}  // namespace allspark
#endif
