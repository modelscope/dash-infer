/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    gemm_op.cpp
 */

#include "gemm_op.h"  // NOLINT

#include <core/kernel/kernel.h>
#include <cpu/cpu_context.h>
#ifdef ENABLE_CUDA
#include <check_cuda.h>
#include <cuda/cuda_context.h>
#endif
#include <utility/datatype_dispatcher.h>

namespace allspark {
AsStatus GemmOpBase::Init(const OperatorProto& op_proto,
                          const DeviceContext& ctx,
                          const TensorMap& weights_map, TensorMap* tensor_map) {
  LOG(ERROR) << "GemmOpBase only support InitV2()" << std::endl;
  return AsStatus::ALLSPARK_INVALID_CALL_ERROR;
}

AsStatus GemmOpBase::InitV2(const OperatorProto& op_proto,
                            const DeviceContext& ctx,
                            const TensorMap& weights_map,
                            TensorMap& weights_buffer, TensorMap* tensor_map) {
  AS_CHECK_STATUS(AsOperator::Init(op_proto, ctx, weights_map, tensor_map));
  // check weight
  // if (weights_.size() != 2 && weights_.size() != 1) {
  //     LOG(ERROR) << "GemmOpBase has 1~2 weights: [weight], (optional)
  //     [bias]."
  //                << std::endl;
  //     return AsStatus::ALLSPARK_PARAM_ERROR;
  // }
  // type inference
  DataType dtype = tensor_map_->at(in_names_[0])->GetDataType();
  // tensor_map_->at(out_names_[0])->SetDataType(dtype);
  // attr
  DeviceType backend = ctx.GetDeviceType();
  switch (backend) {
#ifdef ENABLE_CUDA
    case DeviceType::CUDA: {
      const CUDAContext* gpu_ctx = static_cast<const CUDAContext*>(ctx_);
      nranks_ = gpu_ctx->GetNranks();
      rank_id_ = gpu_ctx->GetRank();
      break;
    }
#endif
    case DeviceType::CPU: {
      const CPUContext* cpu_ctx = static_cast<const CPUContext*>(ctx_);
      nranks_ = cpu_ctx->GetNranks();
      rank_id_ = cpu_ctx->GetRank();
      break;
    }
    default:
      LOG(ERROR) << "Gemm Operator does not support "
                 << DeviceType_Name(backend) << " device type" << std::endl;
      return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }
  tensor_map_->at(out_names_[0])->SetDataType(dtype);
  auto& attr_map = op_proto.attr();
  if (attr_map.find("transB") != attr_map.end()) {
    transB_ = *(bool*)(attr_map.at("transB").c_str());
  }
  if (attr_map.find("is_pooler") != attr_map.end()) {
    is_pooler_ = *(bool*)(attr_map.at("is_pooler").c_str());
  }
  if (attr_map.find("activation") != attr_map.end()) {
    activation_ = *(UnaryType*)(attr_map.at("activation").c_str());
  }
  if (attr_map.find("binary_type") != attr_map.end()) {
    binary_type_ = *(BinaryType*)(attr_map.at("binary_type").c_str());
  }
  if (attr_map.find("alpha") != attr_map.end()) {
    alpha_ = *(float*)(attr_map.at("alpha").c_str());
  }
  if (attr_map.find("splitk") != attr_map.end()) {
    is_split_k_ = *(bool*)(attr_map.at("splitk").c_str());
  }

  // set k_, n_, batch
  const Shape& w_shape = weights_[0]->GetShape();
  int ndims_w = w_shape.Size();
  if (ndims_w < 2) {
    LOG(ERROR) << "GemmOpBase : Invalid weight shape." << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
  k_ = transB_ ? w_shape[ndims_w - 1] : w_shape[ndims_w - 2];
  n_ = transB_ ? w_shape[ndims_w - 2] : w_shape[ndims_w - 1];
  batch_ = w_shape.Count(0, ndims_w - 2);
  lda_ = k_;
  ldb_ = transB_ ? k_ : n_;
  ldc_ = n_;
  if (is_split_k_) {
    lda_ = k_ * nranks_;
  }
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus GemmOpBase::Reshape(int yn) {
  const Shape& x_shape = tensor_map_->at(in_names_[0])->GetShape();
  int x_ndims = x_shape.Size();
  Shape y_shape;
  // int yn = is_npad_ ? n_padded_before_ : n_;
  if (is_pooler_) {
    if (x_shape.Size() != 3) {
      return AsStatus::ALLSPARK_RUNTIME_ERROR;
    }
    m_ = x_shape[0];
    lda_ = k_ * x_shape[1];
    y_shape = Shape({m_, yn});
  } else {
    m_ = x_shape.Count(0, x_ndims - 1);
    if (batch_ != 1) {
      y_shape.Append(batch_);
    }
    for (int i = 0; i < x_ndims - 1; ++i) {
      y_shape.Append(x_shape[i]);
    }
    y_shape.Append(yn);
  }
  dtype_ = tensor_map_->at(in_names_[0])->GetDataType();
  tensor_map_->at(out_names_[0])->SetDataType(dtype_);
  tensor_map_->at(out_names_[0])->SetShape(std::move(y_shape));

  const Shape& w_shape = weights_[0]->GetShape();
  if (weights_[0]->GetDataType() == DataType::INT8) {
    long size = weights_[0]->GetShape().Count() * SizeofType(dtype_);
    tensor_map_->at("workspace")->SetShape(Shape{size});
  }
  // binary input shape is known in reshape phase
  if (rank_info_.rank_id != 0 && binary_type_ == ADD) {
    // force set binary_type_ to BINARYTYPE_UNDEFINED in case all workers add
    // bias
    binary_type_ = BINARYTYPE_UNDEFINED;
  }
  return AsStatus::ALLSPARK_SUCCESS;
}
}  // namespace allspark
