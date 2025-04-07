/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    postprocess_id_op.cpp
 */

#include "postprocess_id_op.h"  // NOLINT

#include <core/kernel/kernel.h>
#include <utility/datatype_dispatcher.h>
#ifdef ENABLE_CUDA
#include <cuda/cuda_context.h>
#endif
#include <cpu/cpu_context.h>

namespace allspark {

AsStatus PostProcessIdOp::Init(const OperatorProto& op_proto,
                               const DeviceContext& ctx,
                               const TensorMap& weights_map,
                               TensorMap* tensor_map) {
  AS_CHECK_STATUS(AsOperator::Init(op_proto, ctx, weights_map, tensor_map));
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus PostProcessIdOp::Reshape(RuntimeContext* runtime_ctx) {
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus PostProcessIdOp::Forward(RuntimeContext* runtime_ctx) {
  return AsStatus::ALLSPARK_SUCCESS;
}

REGISTER_OP(PostProcessId, CUDA, PostProcessIdOp)
REGISTER_OP(PostProcessId, CPU, PostProcessIdOp)
}  // namespace allspark
