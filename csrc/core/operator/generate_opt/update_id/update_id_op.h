/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    update_id_op.h
 */

#pragma once

#include <core/operator/operator.h>

namespace allspark {

/*
 * inputs :
 *    > dec_ids [batch, 1]
 * outputs :
 *    > max_dec_ids [batch, max_dec_len]
 */

class UpdateIdOp : public AsOperator {
 public:
  explicit UpdateIdOp(const std::string& op_type = "")
      : AsOperator(op_type), batch_size_(1), beam_size_(1), seq_len_(1) {}
  AsStatus Init(const OperatorProto& op_proto, const DeviceContext& ctx,
                const TensorMap& weights_map, TensorMap* tensor_map);
  AsStatus Reshape(RuntimeContext* runtime_ctx) override;
  AsStatus Forward(RuntimeContext* runtime_ctx) override;
  AsStatus RunContext(RuntimeContext* runtime_ctx);
  AsStatus RunDecoder(RuntimeContext* runtime_ctx);

 private:
  AsStatus (*kernel_launcher)(int64_t* max_dec_ids, const int64_t* dec_ids,
                              const int* beam_idx, int64_t* tmp_id,
                              int batch_size, int beam_size, int max_dec_len,
                              int* step_list, int seq_len,
                              const DeviceContext* ctx) = nullptr;

  int batch_size_;
  int seq_len_;
  int beam_size_;
  std::unique_ptr<AsTensor> tmp_id_;
  std::unique_ptr<AsTensor> tmp_step_;
};

}  // namespace allspark
