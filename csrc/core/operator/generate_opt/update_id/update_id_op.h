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
  explicit UpdateIdOp(const std::string& op_type = "") : AsOperator(op_type) {}
  AsStatus Init(const OperatorProto& op_proto, const DeviceContext& ctx,
                const TensorMap& weights_map, TensorMap* tensor_map);
  AsStatus Reshape(RuntimeContext* runtime_ctx) override;
  AsStatus Forward(RuntimeContext* runtime_ctx) override;
  AsStatus RunContext(RuntimeContext* runtime_ctx);
  AsStatus RunDecoder(RuntimeContext* runtime_ctx);

 private:
  bool check_stop_words(
      const int batch_size, const int generated_len, const int max_len,
      int64_t* out_host, bool* gen_over,
      const std::vector<std::vector<int64_t>>& stop_words_ids);

  bool check_finish(std::shared_ptr<GenerateContext>& gen_ctx);
  AsStatus copy_generated_ids(std::shared_ptr<GenerateContext>& gen_ctx,
                              bool is_context);
};

}  // namespace allspark
