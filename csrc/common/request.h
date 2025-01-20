
/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    request.h
 */

#pragma once
#include <core/tensor/tensor.h>

#include <chrono>
#include <string>

namespace allspark {

#ifdef ENABLE_JSON_MODE
namespace util {
class FormatEnforcer;
}
#endif

struct Request {
  std::string request_id;
  TensorMap inputs;
  const TensorMap outputs;  // created in AsEngineImpl, shared by all workers
  TensorMap interim;        // intermediate tensors
  GenerateConfig gen_cfg;
  std::vector<std::vector<std::pair<int, float>>> log_probs_list;
  std::vector<float> token_logprobs_list;
  bool finish = false;
  int input_len = 0;
  int origin_len = 0;
  int prefill_chunk_len = 0;
  int prefix_len = 0;
  int prefix_len_gpu = 0;
  size_t generated_len = 0;
  AsEngine::GenerateRequestStatus status =
      AsEngine::GenerateRequestStatus::Init;
  TensorListMap extra_embedding;
#ifdef ENABLE_JSON_MODE
  std::shared_ptr<util::FormatEnforcer> format_enforcer;
#endif
  std::chrono::time_point<std::chrono::steady_clock> enqueue_ts;
  const std::chrono::time_point<std::chrono::steady_clock> start_ts;
  std::chrono::time_point<std::chrono::steady_clock> context_ts;
  std::chrono::time_point<std::chrono::steady_clock> generate_ts;

  Request(const std::string& request_id_, const TensorMap& inputs_,
          const TensorMap& outputs_, const GenerateConfig& gen_cfg,
          const TensorMap& interim_ = {})
      : request_id(request_id_),
        inputs(inputs_),
        outputs(outputs_),
        interim(interim_),
        gen_cfg(gen_cfg),
        finish(false),
        status(AsEngine::GenerateRequestStatus::Init),
        start_ts(std::chrono::steady_clock::now()) {}

  Request(std::shared_ptr<Request> source_request)
      : request_id(source_request->request_id),
        inputs(source_request->inputs),
        outputs(source_request->outputs),
        interim(source_request->interim),
        gen_cfg(source_request->gen_cfg),
        log_probs_list(source_request->log_probs_list),
        token_logprobs_list(source_request->token_logprobs_list),
        finish(source_request->finish),
        input_len(source_request->input_len),
        prefill_chunk_len(source_request->prefill_chunk_len),
        prefix_len(source_request->prefix_len),
        status(source_request->status),
        extra_embedding(source_request->extra_embedding) {}
};

}  // namespace allspark
