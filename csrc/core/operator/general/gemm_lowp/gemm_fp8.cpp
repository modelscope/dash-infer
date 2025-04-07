/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    gemm_fp8.cpp
 */

#include "gemm_fp8.h"

#include <core/kernel/kernel.h>
#include <cpu/cpu_context.h>
#include <utility/datatype_dispatcher.h>

namespace allspark {

AsStatus GemmFP8Base::Init(const OperatorProto& op_proto,
                           const DeviceContext& ctx,
                           const TensorMap& weights_map,
                           TensorMap* tensor_map) {
  LOG(ERROR) << "GemmFP8Base only support InitV2()" << std::endl;
  return AsStatus::ALLSPARK_INVALID_CALL_ERROR;
}

AsStatus GemmFP8Base::InitV2(const OperatorProto& op_proto,
                             const DeviceContext& ctx,
                             const TensorMap& weights_map,
                             TensorMap& weights_buffer, TensorMap* tensor_map,
                             RuntimeContext* runtime_ctx) {
  DLOG(INFO) << "GemmFP8Base::Init()" << std::endl;
  AS_CHECK_STATUS(AsOperator::Init(op_proto, ctx, weights_map, tensor_map));

  // check weight
  if (weights_.size() != 3 && weights_.size() != 4) {
    LOG(ERROR) << "GemmFP8Base has 3-4 weights: [weight], [scales], [zeros], "
                  "(optional) [bias]."
               << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }

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

  // Check Trans.
  if (transA_ != 0 || transB_ != 0) {
    LOG(ERROR) << "GemmFP8Base only support transA=0 and transB=0."
               << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
  if (is_pooler_ == true) {
    LOG(ERROR) << "GemmFP8Base: is_pooler error." << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }

  // Set k_, n_, batch_
  const Shape& w_shape = weights_[0]->GetShape();
  const int ndims_w = w_shape.Size();
  if (ndims_w != 2) {
    LOG(ERROR) << "GemmFP8Base : Invalid weight shape." << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }

  k_ = transB_ ? w_shape[ndims_w - 1] : w_shape[ndims_w - 2];
  n_ = transB_ ? w_shape[ndims_w - 2] : w_shape[ndims_w - 1];
  batch_ = w_shape.Count(0, ndims_w - 2);

  if (batch_ != 1) {
    LOG(ERROR) << "GemmFP8Base : Only support batch_ = 1." << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }

  lda_ = k_;
  ldb_ = transB_ ? k_ : n_;
  ldc_ = n_;

  // weight datatype
  wtype_ = weights_[0]->GetDataType();
  if (wtype_ != DataType::FLOAT8E4M3 && wtype_ != DataType::FLOAT8E5M2 &&
      wtype_ != DataType::FLOAT16 && wtype_ != DataType::BFLOAT16) {
    LOG(ERROR) << "GemmFP8Base : weight data type only support float8_e4m3, "
                  "float8_e5m2, float16 or bfloat16 ."
               << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }

  // Type inference
  atype_ = tensor_map_->at(in_names_[0])->GetDataType();
  tensor_map_->at(out_names_[0])->SetDataType(atype_);
  if (atype_ != DataType::FLOAT16 && atype_ != DataType::BFLOAT16) {
    LOG(ERROR)
        << "GemmFP8Base : input data type only support float16 or bfloat16."
        << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
  return AsStatus::ALLSPARK_SUCCESS;
}

}  // namespace allspark