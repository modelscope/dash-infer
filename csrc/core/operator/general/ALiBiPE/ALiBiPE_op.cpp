/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    ALiBiPE_op.cpp
 */

#include "ALiBiPE_op.h"  // NOLINT

#include <core/kernel/kernel.h>
#include <cpu/cpu_context.h>
#include <utility/datatype_dispatcher.h>

namespace allspark {
AsStatus cpu_ALiBiPE(DataType dtype, void* out, int* batch_offset, int batch,
                     int seq_len, int num_heads, int ori_num_heads,
                     const DeviceContext* ctx, bool is_context,
                     std::vector<int>& step_list) {
  DLOG(INFO) << "cpu_ALiBiPE" << std::endl;
  const CPUContext* cpu_ctx = static_cast<const CPUContext*>(ctx);
  auto functor = [&]<typename T>() {
    T* typed_out = static_cast<T*>(out);
    cpu::ALiBiPEKernelLauncher(typed_out, batch_offset, batch, seq_len,
                               num_heads, ori_num_heads, cpu_ctx->GetRank(),
                               is_context, step_list);
  };
  DispatchCPU(dtype, functor);
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus ALiBiPEOp::Init(const OperatorProto& op_proto,
                         const DeviceContext& ctx, const TensorMap& weights_map,
                         TensorMap* tensor_map) {
  AS_CHECK_STATUS(AsOperator::Init(op_proto, ctx, weights_map, tensor_map));
  auto& attr_map = op_proto.attr();
  if (attr_map.find("num_heads") == attr_map.end()) {
    LOG(ERROR) << "ALiBiPEOp : can't find num_heads attribute." << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
  num_heads_ = *(int*)(attr_map.at("num_heads").c_str());
  ori_num_heads_ = num_heads_;
  DataType dtype = ctx_->GetDtype();
  tensor_map_->at(out_names_[0])->SetDataType(dtype);
  DeviceType backend = ctx.GetDeviceType();
  switch (backend) {
    case DeviceType::CPU: {
      kernel_launcher = cpu_ALiBiPE;
      const CPUContext* cpu_ctx = static_cast<const CPUContext*>(ctx_);
      num_heads_ /= cpu_ctx->GetNranks();
      break;
    }
    default:
      LOG(ERROR) << "RelativePE Operator does not support "
                 << DeviceType_Name(backend) << " device type" << std::endl;
      return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus ALiBiPEOp::Reshape(RuntimeContext* runtime_ctx) {
  Shape in_shape = tensor_map_->at(in_names_[0])->GetShape();
  if (runtime_ctx->is_context == true) {
    batch_size_ = in_shape[0];
    seq_length_ = in_shape[1];
    Shape out_shape = Shape{batch_size_, seq_length_, num_heads_, seq_length_};
    AS_CHECK_STATUS(
        tensor_map_->at(out_names_[0])->SetShape(std::move(out_shape)));
  }
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus ALiBiPEOp::runContext(RuntimeContext* runtime_ctx) {
  int* batch_offset = nullptr;
  AsTensor* out_tensor = tensor_map_->at(out_names_[0]).get();
  std::vector<int> step_list;
  kernel_launcher(out_tensor->GetDataType(), out_tensor->GetDataPtr(),
                  batch_offset, batch_size_, seq_length_, num_heads_,
                  ori_num_heads_, ctx_, runtime_ctx->is_context, step_list);
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus ALiBiPEOp::runDecode(RuntimeContext* runtime_ctx) {
  int* batch_offset = nullptr;
  AsTensor* out_tensor = tensor_map_->at(out_names_[0]).get();
  int batch_size = runtime_ctx->GetGenCtxListSize();
  std::vector<int> step_list(batch_size);
  int max_step = 1;
  for (int i = 0; i < batch_size; i++) {
    GenerateContext* gen_ctx = runtime_ctx->GetGenCtx(i);
    if (gen_ctx->step + 1 > max_step) {
      max_step = gen_ctx->step + 1;
    }
    step_list[i] = gen_ctx->step + 1;
  }
  Shape out_shape = Shape{batch_size, 1, num_heads_, max_step};
  AS_CHECK_STATUS(
      tensor_map_->at(out_names_[0])->SetShape(std::move(out_shape)));
  kernel_launcher(out_tensor->GetDataType(), out_tensor->GetDataPtr(),
                  batch_offset, batch_size, max_step, num_heads_,
                  ori_num_heads_, ctx_, runtime_ctx->is_context, step_list);
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus ALiBiPEOp::Forward(RuntimeContext* runtime_ctx) {
  if (runtime_ctx->is_context == true) {
    runContext(runtime_ctx);
  } else {
    runDecode(runtime_ctx);
  }
  return AsStatus::ALLSPARK_SUCCESS;
}

REGISTER_OP("ALiBiPE", CPU, ALiBiPEOp)
}  // namespace allspark
