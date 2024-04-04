/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    cast_op.h
 */

#pragma once

#include <core/operator/operator.h>

namespace allspark {

class CastOp : public AsOperator {
 public:
  explicit CastOp(const std::string& op_type = "") : AsOperator(op_type) {}
  AsStatus Init(const OperatorProto& op_proto, const DeviceContext& ctx,
                const TensorMap& weights_map, TensorMap* tensor_map);
  AsStatus Reshape() override;
  AsStatus Forward() override;

 private:
  AsStatus (*kernel_launcher)(DataType dtype0, DataType dtype1, const void* in,
                              void* out, int size,
                              const DeviceContext* ctx) = nullptr;
  DataType src_datatype_ = DATATYPE_UNDEFINED,
           dst_datatype_ = DATATYPE_UNDEFINED;
};

}  // namespace allspark
