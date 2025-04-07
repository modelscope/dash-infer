/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    binary_op.cpp
 */

#include "binary_op.h"  // NOLINT

#include <core/kernel/kernel.h>
#include <utility/datatype_dispatcher.h>
#ifdef ENABLE_CUDA
#include <cuda/cuda_context.h>
#endif
#include <cpu/cpu_context.h>
using dnnl::memory;

#define NO_INPLACE_BINARY 0
#define USE_ONEDNN_BINARY 1

namespace allspark {
AsStatus BinaryOp::Init(const OperatorProto& op_proto, const DeviceContext& ctx,
                        const TensorMap& weights_map, TensorMap* tensor_map) {
  AS_CHECK_STATUS(AsOperator::Init(op_proto, ctx, weights_map, tensor_map));
  // type inference
  DataType dtype = tensor_map_->at(in_names_[0])->GetDataType();
  tensor_map_->at(out_names_[0])->SetDataType(dtype);
  // attr
  auto& attr_map = op_proto.attr();
  if (attr_map.find("binary_type") == attr_map.end()) {
    LOG(ERROR) << "BinaryOp : can't find binary_type attribute." << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
  binary_type_ = *(BinaryType*)(attr_map.at("binary_type").c_str());
  DeviceType backend = ctx.GetDeviceType();
  switch (backend) {
    case DeviceType::CPU: {
      dnnl_op_ctx_ = std::make_unique<DNNLOpContext>();
      auto& algo_map = DNNLOpContext::binary_algo_map_;
      if (algo_map.find(binary_type_) == algo_map.end()) {
        LOG(ERROR) << "Unsupported binary type:"
                   << BinaryType_Name(binary_type_) << std::endl;
        return AsStatus::ALLSPARK_PARAM_ERROR;
      }
      dnnl_op_ctx_->algo_ = algo_map[binary_type_];
      if ((binary_type_ == BinaryType::GEGLU) ||
          (binary_type_ == BinaryType::SWIGLU)) {
        dnnl_op_ctx_->pr_fwd_.resize(2);
      } else {
        dnnl_op_ctx_->pr_fwd_.resize(1);
      }
      dnnl_op_ctx_->ins_.resize(2);
      dnnl_op_ctx_->outs_.resize(1);
      break;
    }
#ifdef ENABLE_CUDA
    case DeviceType::CUDA: {
      break;
    }
#endif
    default:
      LOG(ERROR) << "Binary Operator does not support "
                 << DeviceType_Name(backend) << " device type" << std::endl;
      return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus BinaryOp::Reshape(RuntimeContext* runtime_ctx) {
  Shape out_shape = tensor_map_->at(in_names_[0])->GetShape();
  AS_CHECK_STATUS(
      tensor_map_->at(out_names_[0])->SetShape(std::move(out_shape)));
  if (ctx_->GetDeviceType() == DeviceType::CPU) {
    auto eng = DNNLEngine::GetInstance().GetEngine();
    memory::desc data_desc({out_shape.Count()}, memory::data_type::f32,
                           memory::format_tag::x);
    dnnl_op_ctx_->ins_[0] = std::make_unique<memory>(data_desc, eng, nullptr);
    dnnl_op_ctx_->ins_[1] = std::make_unique<memory>(data_desc, eng, nullptr);
    dnnl_op_ctx_->outs_[0] = std::make_unique<memory>(data_desc, eng, nullptr);
    dnnl_op_ctx_->pr_fwd_[0] =
        std::make_unique<dnnl::binary>(dnnl::binary::primitive_desc{
            eng, dnnl_op_ctx_->algo_, data_desc, data_desc, data_desc});

    if (binary_type_ == BinaryType::GEGLU) {
      // do gelu first
      dnnl_op_ctx_->pr_fwd_[1] = std::make_unique<dnnl::eltwise_forward>(
          dnnl::eltwise_forward::primitive_desc{
              eng, dnnl::prop_kind::forward_inference,
              dnnl::algorithm::eltwise_gelu_tanh, data_desc, data_desc, 0.f,
              0.f});
    }
    if (binary_type_ == BinaryType::SWIGLU) {
      // do gelu first
      dnnl_op_ctx_->pr_fwd_[1] = std::make_unique<dnnl::eltwise_forward>(
          dnnl::eltwise_forward::primitive_desc{
              eng, dnnl::prop_kind::forward_inference,
              dnnl::algorithm::eltwise_swish, data_desc, data_desc, 1.f, 0.f});
    }
  }
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus BinaryOp::Forward(RuntimeContext* runtime_ctx) {
  AsTensor* x_tensor = tensor_map_->at(in_names_[0]).get();
  AsTensor* y_tensor = tensor_map_->at(in_names_[1]).get();
  AsTensor* z_tensor = tensor_map_->at(out_names_[0]).get();
  int64_t count = x_tensor->GetShape().Count();

  switch (ctx_->GetDeviceType()) {
#ifdef ENABLE_CUDA
    case DeviceType::CUDA: {
#ifdef CONFIG_DEBUG_OP
      DLOG(INFO) << "gpu_binary" << std::endl;
#endif
      const CUDAContext* gpu_ctx = static_cast<const CUDAContext*>(ctx_);
      auto functor = [&]<typename T>() {
        T* typed_out = static_cast<T*>(z_tensor->GetDataPtr());
        const T* typed_in1 = static_cast<const T*>(x_tensor->GetDataPtr());
        const T* typed_in2 = static_cast<const T*>(y_tensor->GetDataPtr());
        cuda::BinaryKernelLauncher(typed_out, typed_in1, typed_in2, count,
                                   binary_type_, gpu_ctx->GetStream());
      };
      DispatchCUDA(x_tensor->GetDataType(), functor);
      break;
    }
#endif
    case DeviceType::CPU: {
      dnnl::memory& in0_mem = *(dnnl_op_ctx_->ins_[0]);
      dnnl::memory& in1_mem = *(dnnl_op_ctx_->ins_[1]);
      dnnl::memory& out_mem = *(dnnl_op_ctx_->outs_[0]);

#if (NO_INPLACE_BINARY & USE_ONEDNN_BINARY)
      auto eng = DNNLEngine::GetInstance().GetEngine();
      dnnl::memory temp_mem =
          dnnl::memory(dnnl_op_ctx_->ins_[1]->get_desc(), eng);
#endif

      const CPUContext* cpu_ctx = static_cast<const CPUContext*>(ctx_);
      in0_mem.set_data_handle(x_tensor->GetDataPtr());
      in1_mem.set_data_handle(y_tensor->GetDataPtr());
      out_mem.set_data_handle(z_tensor->GetDataPtr());
      if ((binary_type_ == BinaryType::GEGLU ||
           binary_type_ == BinaryType::SWIGLU) &&
          dnnl_op_ctx_->pr_fwd_.size() > 1) {
#if (NO_INPLACE_BINARY & USE_ONEDNN_BINARY)
        std::unordered_map<int, memory> args1{{DNNL_ARG_SRC_0, in1_mem},
                                              {DNNL_ARG_DST, temp_mem}};
        dnnl_op_ctx_->pr_fwd_[1]->execute(cpu_ctx->GetStream(), args1);

        std::unordered_map<int, memory> args2{{DNNL_ARG_SRC_0, in0_mem},
                                              {DNNL_ARG_SRC_1, temp_mem},
                                              {DNNL_ARG_DST, out_mem}};
        dnnl_op_ctx_->pr_fwd_[0]->execute(cpu_ctx->GetStream(), args2);
#elif USE_ONEDNN_BINARY
        std::unordered_map<int, memory> args1{{DNNL_ARG_SRC_0, in1_mem},
                                              {DNNL_ARG_DST, in1_mem}};
        dnnl_op_ctx_->pr_fwd_[1]->execute(cpu_ctx->GetStream(), args1);

        std::unordered_map<int, memory> args2{{DNNL_ARG_SRC_0, in0_mem},
                                              {DNNL_ARG_SRC_1, in1_mem},
                                              {DNNL_ARG_DST, out_mem}};
        dnnl_op_ctx_->pr_fwd_[0]->execute(cpu_ctx->GetStream(), args2);
#else
        // naive impl
        for (size_t idx = 0; idx < y_tensor->GetShape().Count(); idx++) {
          float y = ((float*)y_tensor->GetDataPtr())[idx];
          float x = ((float*)x_tensor->GetDataPtr())[idx];
          float tmp = y * (1.0f / (1.0f + expf(-y)));  // swiglu result
          float out = tmp * x;                         // mul
          *(((float*)z_tensor->GetDataPtr()) + idx) = out;
        }
#endif
      } else {
#if USE_ONEDNN_BINARY
        std::unordered_map<int, memory> args{{DNNL_ARG_SRC_0, in0_mem},
                                             {DNNL_ARG_SRC_1, in1_mem},
                                             {DNNL_ARG_DST, out_mem}};
        dnnl_op_ctx_->pr_fwd_[0]->execute(cpu_ctx->GetStream(), args);
#else
        // naive impl
        for (size_t idx = 0; idx < y_tensor->GetShape().Count(); idx++) {
          float y = ((float*)y_tensor->GetDataPtr())[idx];
          float x = ((float*)x_tensor->GetDataPtr())[idx];
          *(((float*)z_tensor->GetDataPtr()) + idx) = x + y;
        }
#endif
      }
      break;
    }
    default:
      break;
  }
  return AsStatus::ALLSPARK_SUCCESS;
}
REGISTER_OP(Binary, CUDA, BinaryOp)
REGISTER_OP(Binary, CPU, BinaryOp)
}  // namespace allspark
