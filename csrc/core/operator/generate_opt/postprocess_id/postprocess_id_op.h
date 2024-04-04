/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    postprocess_id_op.h
 */

#pragma once

#include <core/operator/operator.h>

namespace allspark {

class PostProcessIdOp : public AsOperator {
 public:
  explicit PostProcessIdOp(const std::string& op_type = "")
      : AsOperator(op_type) {}
  AsStatus Init(const OperatorProto& op_proto, const DeviceContext& ctx,
                const TensorMap& weights_map, TensorMap* tensor_map);
  AsStatus Reshape(RuntimeContext* runtime_ctx) override;
  AsStatus Forward(RuntimeContext* runtime_ctx) override;
  AsStatus RunContext(RuntimeContext* runtime_ctx);
  AsStatus RunDecoder(RuntimeContext* runtime_ctx);
  AsStatus RunOneBatch(GenerateContext* gen_ctx, int current_batch);
  inline std::string GetInTensorName() { return in_name_; }

 private:
  std::string in_name_;
  int batch_size_ = 1;
  int in_stride_ = 1;
  int out_stride_ = 1;
  int rank_ = 0;
  std::unique_ptr<AsTensor> output_host_;
};

}  // namespace allspark
