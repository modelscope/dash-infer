/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    preprocess_id_op.cpp
 */

#include "preprocess_id_op.h"  // NOLINT

#include <core/kernel/kernel.h>
#include <utility/datatype_dispatcher.h>
#ifdef ENABLE_CUDA
#include <cuda/cuda_context.h>
#endif
#include <cpu/cpu_context.h>

namespace allspark {

AsStatus PreProcessIdOp::Init(const OperatorProto& op_proto,
                              const DeviceContext& ctx,
                              const TensorMap& weights_map,
                              TensorMap* tensor_map) {
  AS_CHECK_STATUS(AsOperator::Init(op_proto, ctx, weights_map, tensor_map));
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus PreProcessIdOp::Reshape(RuntimeContext* runtime_ctx) {
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus PreProcessIdOp::Forward(RuntimeContext* runtime_ctx) {
  GenerateContext* gen_ctx = runtime_ctx->GetContextGenCtx();
  std::shared_ptr<Request> request = gen_ctx->request;

  {
    // create local generated_ids (own by each workder)
    std::string new_tensor_name = "generated_ids";
    auto generated_ids_tensor = std::make_shared<AsTensor>(
        new_tensor_name, DeviceType::CPU, DataType::INT64, DataMode::DENSE,
        Shape{1, ctx_->GetModelMaxLength()});

    // copy input_ids to generated_ids, cpu to cpu
    generated_ids_tensor->SetShape(
        Shape(request->inputs.at("input_ids")->GetShape()));
    AS_CHECK_EXCEPTION(TensorUtils::DeepCopyWholeAsync(
        *generated_ids_tensor, *request->inputs.at("input_ids"), ctx_));
    request->interim.insert({new_tensor_name, generated_ids_tensor});
  }

#if ENABLE_CUDA
  if (ctx_->GetDeviceType() == DeviceType::CUDA) {
    std::string tensor_name, new_tensor_name;

    // copy generated_ids to gpu
    tensor_name = "generated_ids";
    new_tensor_name = tensor_name + "_gpu";
    auto generated_ids_gpu_tensor = std::make_shared<AsTensor>(
        new_tensor_name, DeviceType::CUDA,
        request->interim.at(tensor_name)->GetDataType(), DataMode::DENSE,
        Shape{1, ctx_->GetModelMaxLength()});
    generated_ids_gpu_tensor->SetShape(
        Shape(request->interim.at(tensor_name)->GetShape()));
    request->interim.insert({new_tensor_name, generated_ids_gpu_tensor});

    AS_CHECK_EXCEPTION(TensorUtils::DeepCopyWholeAsync(
        *request->interim.at(new_tensor_name),
        *request->interim.at(tensor_name), ctx_));

    // copy new_input_ids to gpu
    tensor_name = "new_input_ids";
    new_tensor_name = tensor_name + "_gpu";
    auto new_input_ids_gpu_tensor = std::make_shared<AsTensor>(
        new_tensor_name, DeviceType::CUDA,
        request->interim.at(tensor_name)->GetDataType(), DataMode::DENSE,
        request->interim.at(tensor_name)->GetShape());
    request->interim.insert({new_tensor_name, new_input_ids_gpu_tensor});

    AS_CHECK_EXCEPTION(TensorUtils::DeepCopyWholeAsync(
        *request->interim.at(new_tensor_name),
        *request->interim.at(tensor_name), ctx_));
  }
#endif

  return AsStatus::ALLSPARK_SUCCESS;
}

REGISTER_OP(PreProcessId, CUDA, PreProcessIdOp)
REGISTER_OP(PreProcessId, CPU, PreProcessIdOp)
}  // namespace allspark
