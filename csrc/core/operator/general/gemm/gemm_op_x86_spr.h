/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    gemm_op_x86_spr.h
 */

#pragma once
#if defined(__x86_64__) || defined(_M_X64)
#include <common/float16.h>
#include <core/operator/operator.h>

#include <common/hie_bfloat16.hpp>

#include "gemm_op_cpu.h"

namespace allspark {
class GemmOpSpr : public GemmOpCPU {
 public:
  GemmOpSpr(const std::string& op_type = "") : GemmOpCPU(op_type) {
    dnnl::cpu_isa eff_isa = dnnl::get_effective_cpu_isa();
    if (eff_isa >= dnnl::cpu_isa::avx512_core_amx) {
      is_spr_ = true;
    }
  }

  AsStatus Init(const OperatorProto& op_proto, const DeviceContext& ctx,
                const TensorMap& weights_map, TensorMap* tensor_map) override;
  AsStatus InitV2(const OperatorProto& op_proto, const DeviceContext& ctx,
                  const TensorMap& weights_map, TensorMap& weights_buffer,
                  TensorMap* tensor_map) override;
  AsStatus Reshape() override;
  AsStatus Forward() override;

 private:
  bool is_spr_ = false;
};
}  // namespace allspark
#endif
