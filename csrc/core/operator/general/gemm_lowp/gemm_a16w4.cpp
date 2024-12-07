/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    gemm_a16w4.cpp
 */

#include "gemm_a16w4.h"

#include <core/kernel/kernel.h>
#include <cpu/cpu_context.h>
#include <utility/datatype_dispatcher.h>

namespace allspark {

AsStatus GemmA16W4Base::Init(const OperatorProto& op_proto,
                             const DeviceContext& ctx,
                             const TensorMap& weights_map,
                             TensorMap* tensor_map) {
  LOG(ERROR) << "GemmA16W4Base only support InitV2()" << std::endl;
  return AsStatus::ALLSPARK_INVALID_CALL_ERROR;
}

AsStatus GemmA16W4Base::InitV2(const OperatorProto& op_proto,
                               const DeviceContext& ctx,
                               const TensorMap& weights_map,
                               TensorMap& weights_buffer,
                               TensorMap* tensor_map) {
  DLOG(INFO) << "GemmA16W4GPU::Init()" << std::endl;
  AS_CHECK_STATUS(AsOperator::Init(op_proto, ctx, weights_map, tensor_map));

  // check weight
  if (weights_.size() != 3 && weights_.size() != 4) {
    LOG(ERROR) << "GemmA16W4Base has 3-4 weights: [weight], [scales], [zeros], "
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
    if (group_size_ % 8 != 0 || group_size_ < 32) {
      LOG(ERROR) << "GemmA16W4Base: SubChannel only support GroupSize >= 32 "
                    "and GroupSize must be divisible by 8."
                 << std::endl;
      return AsStatus::ALLSPARK_PARAM_ERROR;
    }
  }

  // Check Trans.
  if (transA_ == 1 || transB_ == 1) {
    LOG(ERROR) << "GemmA16W4Base only support transA=0 and transB==0."
               << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
  if (is_pooler_ == true) {
    LOG(ERROR) << "GemmA16W4Base: is_pooler error." << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }

  // Set k_, n_, n_pack_, batch
  const Shape& w_shape = weights_[0]->GetShape();
  const Shape& param_shape = weights_[1]->GetShape();
  const int ndims_w = w_shape.Size();
  if (ndims_w != 2) {
    LOG(ERROR) << "GemmA16W4Base : Invalid weight shape." << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
  k_ = transB_ ? w_shape[ndims_w - 1] : w_shape[ndims_w - 2];
  n_pack_ = transB_ ? w_shape[ndims_w - 2] : w_shape[ndims_w - 1];
  n_ = param_shape[ndims_w - 1];  // quantize param`s N is the real N. weight`s
                                  // n_pack_ is (N+1)/2
  if (n_pack_ != (n_ + 1) / 2) {
    LOG(ERROR) << "GemmA16W4Base : N_PACK Size Error.";
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }

  batch_ = w_shape.Count(0, ndims_w - 2);
  if (batch_ != 1) {
    LOG(ERROR) << "GemmA16W4Base : Only support batch_ = 1." << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }

  lda_ = k_;
  ldb_ = transB_ ? k_ : n_;
  ldc_ = n_;

  // weight datatype, uint8
  // int8 mean int4x2,  uint8 mean uint4x2
  qtype_ = weights_[0]->GetDataType();
  if (qtype_ != DataType::UINT8) {
    LOG(ERROR)
        << "GemmA16W4Base : Pack weight data type only support uint8(uint4x2)."
        << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
  return AsStatus::ALLSPARK_SUCCESS;
}

}  // namespace allspark