/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    preprocess_id_op.h
 */

#pragma once

#include <core/operator/operator.h>

namespace allspark {

/* @brief: 构建decoder最初的dec_id, 一共有两种方式:
 *  1）从input_ids broadcast获得, 输入输出如下:
 *     inputs :
 *       > input_ids [batch, seq_len]
 *     outputs :
 *       > dec_ids [batch * beam, seq_len]
 *  2）从start_ids broadcast获得, 输入输出如下:
 *     op_attr : {"start_id" : xxx}
 *     outputs :
 *       > input_ids [batch, seq_len] (仅用来获取batch)
 *       > dec_ids [batch * beam, 1]
 */
class PreProcessIdOp : public AsOperator {
 public:
  explicit PreProcessIdOp(const std::string& op_type = "")
      : AsOperator(op_type),
        batch_size_(1),
        num_beam_(1),
        seq_len_(1),
        max_len_(1),
        start_id_(-1) {}
  AsStatus Init(const OperatorProto& op_proto, const DeviceContext& ctx,
                const TensorMap& weights_map, TensorMap* tensor_map);
  AsStatus Reshape() override;
  AsStatus Forward() override;
  AsStatus Reshape(RuntimeContext* runtime_ctx) override {
    return this->Reshape();
  }
  AsStatus Forward(RuntimeContext* runtime_ctx) override {
    return this->Forward();
  }

 private:
  int batch_size_;
  int num_beam_ = 1;
  int seq_len_;
  int max_len_;
  int64_t start_id_;
};

}  // namespace allspark
