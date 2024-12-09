/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    quantize.cpp
 */

#include "quantize.h"  // NOLINT

#include <core/kernel/kernel.h>
#include <utility/datatype_dispatcher.h>
#ifdef ENABLE_CUDA
#include <cuda/cuda_context.h>
#endif
#include <cpu/cpu_context.h>
using dnnl::memory;
namespace allspark {
AsStatus QuantizeOp::Init(const OperatorProto& op_proto,
                          const DeviceContext& ctx,
                          const TensorMap& weights_map, TensorMap* tensor_map) {
  AS_CHECK_STATUS(AsOperator::Init(op_proto, ctx, weights_map, tensor_map));
  // type inference
  dtype_ = tensor_map_->at(in_names_[0])->GetDataType();
  tensor_map_->at(out_names_[0])->SetDataType(DataType::INT8);
  tensor_map_->at(out_names_[1])->SetDataType(DataType::FLOAT32);
  tensor_map_->at(out_names_[2])->SetDataType(DataType::INT8);
  tensor_map_->at(out_names_[3])->SetDataType(DataType::INT32);
  DeviceType backend = ctx.GetDeviceType();
  switch (backend) {
#ifdef ENABLE_CUDA
    case DeviceType::CUDA:
      break;
#endif
    case DeviceType::CPU: {
      break;
    }
    default:
      LOG(ERROR) << "Operator does not support " << DeviceType_Name(backend)
                 << " device type" << std::endl;
      return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus QuantizeOp::Reshape() {
  const Shape& x_shape = tensor_map_->at(in_names_[0])->GetShape();
  int x_ndims = x_shape.Size();
  Shape base_shape = x_shape;
  Shape y_shape;
  m_ = x_shape.Count(0, x_ndims - 1);

  for (int i = 0; i < x_ndims - 2; ++i) {
    y_shape.Append(x_shape[i]);
  }
  y_shape.Append(1);
  y_shape.Append(x_shape[x_ndims - 2]);
  tensor_map_->at(out_names_[0])->SetShape(std::move(base_shape));
  tensor_map_->at(out_names_[1])->SetShape(std::move(y_shape));
  tensor_map_->at(out_names_[2])->SetShape(std::move(y_shape));
  tensor_map_->at(out_names_[3])->SetShape(std::move(y_shape));
  // Lhs isn`t weight and datatype is float.
  // That need workspace to store qdata, scale, zero and redsum.
  // if (tensor_map_->at(in_names_[0])->GetDataType() != DataType::INT8) {
  //     ws_size += aligne_size(lhs_cnt_) * sizeof(int8_t) +
  //                aligne_size(lhs_reduce_cnt_) * SizeofType(dtype_) +
  //                aligne_size(lhs_reduce_cnt_) * sizeof(int8_t) +
  //                aligne_size(lhs_reduce_cnt_) * sizeof(int);
  // }
  // tensor_map_->at("workspace")->SetShape(Shape({ws_size}));
  return AsStatus::ALLSPARK_SUCCESS;
}
template <typename FType, typename QType>
AsStatus QuantizeOp::Process() {
  AsTensor* in_tensor = tensor_map_->at(in_names_[0]).get();
  DataType dtype = in_tensor->GetDataType();
  switch (ctx_->GetDeviceType()) {
    case DeviceType::CUDA: {
#ifdef ENABLE_CUDA
      const CUDAContext* gpu_ctx = static_cast<const CUDAContext*>(ctx_);
      cudaStream_t cu_stream = gpu_ctx->GetStream();
      if (dtype != DataType::INT8) {
        const FType* lhs_fdata = static_cast<FType*>(in_tensor->GetDataPtr());
        // workspace store qdata scale zero and redsum
        lhs_qdata_ = (int8_t*)tensor_map_->at(out_names_[0])->GetDataPtr();
        lhs_scale_ = (float*)tensor_map_->at(out_names_[1])->GetDataPtr();
        lhs_zero_ = (int8_t*)tensor_map_->at(out_names_[2])->GetDataPtr();
        lhs_redsum_ = (int*)tensor_map_->at(out_names_[3])->GetDataPtr();
        // Quantize Lhs
        bool do_per_channel = (per_channel_ & 0x2) ? true : false;
        Shape lhs_shape = in_tensor->GetShape();
        const int lhs_ndim = lhs_shape.Size();
        int inner_len = lhs_shape[lhs_ndim - 1];
        int outer_len = lhs_shape.Count() / inner_len;
        cuda::QuantizePerChannelImp<FType, QType>(
            lhs_fdata, lhs_qdata_, lhs_scale_, lhs_zero_, lhs_redsum_,
            inner_len, outer_len, cu_stream);
      }
#endif
      break;
    }

    case DeviceType::CPU: {
      break;
    }
  }
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus QuantizeOp::Forward() {
  Process<float, int8_t>();
  return AsStatus::ALLSPARK_SUCCESS;
}

REGISTER_OP(Quantize, CUDA, QuantizeOp)
// REGISTER_OP(Quantize, CPU, QuantizeOp)
}  // namespace allspark
