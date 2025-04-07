/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    gemm_a16w8.cpp
 */

#include "gemm_a16w8.h"

#include <core/kernel/kernel.h>
#include <cpu/cpu_context.h>
#include <utility/datatype_dispatcher.h>

namespace allspark {
AsStatus GemmA16W8Base::Init(const OperatorProto& op_proto,
                             const DeviceContext& ctx,
                             const TensorMap& weights_map,
                             TensorMap* tensor_map) {
  LOG(ERROR) << "GemmA16W8Base only support InitV2()" << std::endl;
  return AsStatus::ALLSPARK_INVALID_CALL_ERROR;
}

AsStatus GemmA16W8Base::InitV2(const OperatorProto& op_proto,
                               const DeviceContext& ctx,
                               const TensorMap& weights_map,
                               TensorMap& weights_buffer, TensorMap* tensor_map,
                               RuntimeContext* runtime_ctx) {
  AS_CHECK_STATUS(AsOperator::Init(op_proto, ctx, weights_map, tensor_map));

  // check weight
  if (weights_.size() != 3 && weights_.size() != 4) {
    LOG(ERROR) << "GemmA16W8Base has 3-4 weights: [weight], [scales], [zeros], "
                  "(optional) [bias]."
               << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }

  // Type inference
  DataType dtype = tensor_map_->at(in_names_[0])->GetDataType();
  tensor_map_->at(out_names_[0])->SetDataType(dtype);

  // Attr
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
  if (attr_map.find("alpha") != attr_map.end()) {
    alpha_ = *(float*)(attr_map.at("alpha").c_str());
  }
  if (attr_map.find("GroupSize") != attr_map.end()) {
    group_size_ = *(int*)(attr_map.at("GroupSize").c_str());
    if (group_size_ != 512 && group_size_ != 256 && group_size_ != 128 &&
        group_size_ != 64) {
      LOG(ERROR) << "GEMM_A16W8: SubChannel only support GroupSize [512, "
                    "256, 128, 64]."
                 << std::endl;
      return AsStatus::ALLSPARK_PARAM_ERROR;
    }
  }

  // Set k_, n_, batch
  const Shape& w_shape = weights_[0]->GetShape();
  const int ndims_w = w_shape.Size();
  if (ndims_w != 2) {
    LOG(ERROR) << "GemmA16W8Base : Invalid weight shape." << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }

  // Check Trans.
  // TODO: More
  if (transA_ == 1 || transB_ == 1) {
    LOG(ERROR) << "GemmA16W8Base only support transA=0 and transB==0."
               << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
  if (is_pooler_ == true) {
    LOG(ERROR) << "GemmA16W8Base: is_pooler error." << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }

  k_ = transB_ ? w_shape[ndims_w - 1] : w_shape[ndims_w - 2];
  n_ = transB_ ? w_shape[ndims_w - 2] : w_shape[ndims_w - 1];
  batch_ = w_shape.Count(0, ndims_w - 2);

  if (batch_ != 1) {
    LOG(ERROR) << "GemmA16W8Base : Only support batch_ = 1." << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }

  lda_ = k_;
  ldb_ = transB_ ? k_ : n_;
  ldc_ = n_;

  // Quantize datatype, int8 or uint8
  qtype_ = weights_[0]->GetDataType();
  if (qtype_ != DataType::INT8 && qtype_ != DataType::UINT8) {
    LOG(ERROR) << "GemmA16W8Base : Only support int8/uint8, got : " << qtype_
               << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }

  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus GemmA16W8Base::Reshape(int yn) {
  const Shape& x_shape = tensor_map_->at(in_names_[0])->GetShape();
  int x_ndims = x_shape.Size();
  Shape y_shape;

  m_ = x_shape.Count(0, x_ndims - 1);

  for (int i = 0; i < x_ndims - 1; ++i) {
    y_shape.Append(x_shape[i]);
  }
  y_shape.Append(yn);

  ftype_ = tensor_map_->at(in_names_[0])->GetDataType();
  tensor_map_->at(out_names_[0])->SetDataType(ftype_);
  tensor_map_->at(out_names_[0])->SetShape(std::move(y_shape));

  return AsStatus::ALLSPARK_SUCCESS;
}
}  // namespace allspark
