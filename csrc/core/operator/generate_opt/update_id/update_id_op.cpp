/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    update_id_op.cpp
 */

#include "update_id_op.h"  // NOLINT

#include <core/kernel/kernel.h>
#include <utility/datatype_dispatcher.h>
#ifdef ENABLE_CUDA
#include <check_cuda.h>
#include <cuda/cuda_context.h>
#endif
#include <cpu/cpu_context.h>

namespace allspark {
bool UpdateIdOp::check_stop_words(
    const int batch_size, const int generated_len, const int max_len,
    int64_t* out_host, bool* gen_over,
    const std::vector<std::vector<int64_t>>& stop_words_ids) {
  bool finish = true;
  // only support batch_size = 1
  for (size_t i = 0; i < batch_size; i++) {
    const auto current_data = out_host + i * max_len;
    bool stop_matched = false;
    for (const auto& stop_word : stop_words_ids) {
      if (generated_len > stop_word.size()) {
        const auto word_start = current_data + generated_len - stop_word.size();
        const auto compare_size = stop_word.size() * sizeof(int64_t);

        if (memcmp(word_start, stop_word.data(), compare_size) == 0) {
          stop_matched = true;
          break;
        }
      }
    }
    gen_over[i] |= stop_matched;
    finish &= gen_over[i];
  }
  return finish;
}

bool UpdateIdOp::check_finish(std::shared_ptr<GenerateContext>& gen_ctx) {
  if (gen_ctx->finish) {
    return gen_ctx->finish;
  }

  if (gen_ctx->step + gen_ctx->in_length_bias >=
      gen_ctx->gen_cfg.max_length - 1) {
    gen_ctx->finish = true;
    return true;
  }

  int64_t* generated_ids = static_cast<int64_t*>(
      gen_ctx->request->interim.at("generated_ids")->GetDataPtr());
  if (gen_ctx->gen_cfg.early_stopping) {
    if (generated_ids[gen_ctx->step + gen_ctx->in_length_bias] ==
        (int64_t)gen_ctx->gen_cfg.eos_token_id) {
      gen_ctx->finish = true;
      return true;
    }
  }

  if (gen_ctx->generate_method == 0 &&
      (!gen_ctx->gen_cfg.stop_words_ids.empty())) {
    const auto generated_len = gen_ctx->step + gen_ctx->in_length_bias + 1;
    auto* gen_over = gen_ctx->gen_over;
    if (check_stop_words(1, generated_len, gen_ctx->engine_max_length,
                         generated_ids, gen_over,
                         gen_ctx->gen_cfg.stop_words_ids)) {
      gen_ctx->finish = true;
      return true;
    }
  }

  return gen_ctx->finish;
}

AsStatus UpdateIdOp::copy_generated_ids(
    std::shared_ptr<GenerateContext>& gen_ctx, bool is_context) {
  if (rank_info_.rank_id == 0) {
    std::shared_ptr<AsTensor> local_tensor =
        gen_ctx->request->interim.at("generated_ids");
    std::shared_ptr<AsTensor> global_tensor =
        gen_ctx->request->outputs.at("generated_ids_global");
    global_tensor->SetShape(Shape(local_tensor->GetShape()));

    if (is_context) {
      int generated_len = gen_ctx->step + gen_ctx->in_length_bias + 1;
      memcpy(global_tensor->GetDataPtr(), local_tensor->GetDataPtr(),
             generated_len * sizeof(int64_t));
    } else {
      memcpy(static_cast<int64_t*>(global_tensor->GetDataPtr()) + gen_ctx->step,
             static_cast<int64_t*>(local_tensor->GetDataPtr()) + gen_ctx->step,
             1 * sizeof(int64_t));
    }
  }

  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus UpdateIdOp::Init(const OperatorProto& op_proto,
                          const DeviceContext& ctx,
                          const TensorMap& weights_map, TensorMap* tensor_map) {
  AS_CHECK_STATUS(AsOperator::Init(op_proto, ctx, weights_map, tensor_map));
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus UpdateIdOp::Reshape(RuntimeContext* runtime_ctx) {
  return AsStatus::ALLSPARK_SUCCESS;
}

// clang-format off
/*
* |-------------------------------------------------------------------------------|
* |                 |      prefix_len == 0       |      prefix_len > 0            |
* |-------------------------------------------------------------------------------|
* | update_id_first | step = 0                   | step = prefix_len              |
* | (context)       | in_length_bias = 0         | in_length_bias = 0             |
* |                 | run_step = 0               | run_step = 0                   |
* |                 | seq_len = input_len        | seq_len = new_input_len        |
* |-------------------------------------------------------------------------------|
* | update_id       | step = 0                   | step = prefix_len              |
* | (context)       | in_length_bias = input_len | in_length_bias = new_input_len |
* |                 | run_step = input_len       | run_step = input_len           |
* |                 | seq_len = 1                | seq_len = 1                    |
* |-------------------------------------------------------------------------------|
* | update_id       | step = input_len + gen_len | step = input_len + gen_len     |
* | (decode)        | in_length_bias = 0         | in_length_bias = 0             |
* |                 | run_step = step            | run_step = step                |
* |                 | seq_len = 1                | seq_len = 1                    |
* |-------------------------------------------------------------------------------|
*/
// clang-format on
AsStatus UpdateIdOp::RunContext(RuntimeContext* runtime_ctx) {
  std::shared_ptr<GenerateContext> gen_ctx = runtime_ctx->GetContextGenCtx();
  check_finish(gen_ctx);
  copy_generated_ids(gen_ctx, true);
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus UpdateIdOp::RunDecoder(RuntimeContext* runtime_ctx) {
  int batch_size = runtime_ctx->GetGenCtxListSize();
  for (int i = 0; i < batch_size; i++) {
    std::shared_ptr<GenerateContext> gen_ctx = runtime_ctx->GetGenCtx(i);
    check_finish(gen_ctx);
    copy_generated_ids(gen_ctx, false);
  }

  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus UpdateIdOp::Forward(RuntimeContext* runtime_ctx) {
  if (this->GetOpName() == "update_id_first") return AsStatus::ALLSPARK_SUCCESS;

  AsStatus status;
  if (runtime_ctx->is_context) {
    status = RunContext(runtime_ctx);
  } else {
    status = RunDecoder(runtime_ctx);
  }
  return status;
}

REGISTER_OP(UpdateId, CUDA, UpdateIdOp)
REGISTER_OP(UpdateId, CPU, UpdateIdOp)
}  // namespace allspark
