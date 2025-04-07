
/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    request.h
 */

#pragma once
#include <core/tensor/tensor.h>

#include <chrono>
#include <string>

#if ENABLE_SPAN_ATTENTION
#include <cache/prefix_cache_manager.h>
#endif

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
  std::shared_ptr<moodycamel::ConcurrentQueue<int64_t>>
      generated_ids_queue;  // created in AsEngineImpl, shared by all workers
  GenerateConfig gen_cfg;
  std::vector<std::vector<std::pair<int, float>>> log_probs_list;
  std::vector<float> token_logprobs_list;
  std::unordered_map<std::string, std::vector<std::shared_ptr<AsTensor>>>
      tensors_from_model_inference_list;
  bool finish = false;
  int input_len = 0;
  int origin_len = 0;
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

#if ENABLE_SPAN_ATTENTION
  std::vector<PrefixCacheManager::PrefixNodePtr> prefix_cache_node_list;
#endif

  Request(
      const std::string& request_id, const TensorMap& inputs,
      const TensorMap& outputs_, const GenerateConfig& gen_cfg,
      std::shared_ptr<moodycamel::ConcurrentQueue<int64_t>> generated_ids_queue,
      const std::chrono::time_point<std::chrono::steady_clock> start_ts,
      const TensorMap& interim = {})
      : request_id(request_id),
        inputs(inputs),
        outputs(outputs_),
        interim(interim),
        gen_cfg(gen_cfg),
        generated_ids_queue(generated_ids_queue),
        finish(false),
        status(AsEngine::GenerateRequestStatus::Init),
        start_ts(start_ts) {}

  Request(std::shared_ptr<Request> source_request)
      : request_id(source_request->request_id),
        inputs(source_request->inputs),
        outputs(source_request->outputs),
        interim(source_request->interim),
        gen_cfg(source_request->gen_cfg),
        generated_ids_queue(source_request->generated_ids_queue),
        log_probs_list(source_request->log_probs_list),
        token_logprobs_list(source_request->token_logprobs_list),
        tensors_from_model_inference_list(
            source_request->tensors_from_model_inference_list),
        finish(source_request->finish),
        input_len(source_request->input_len),
        prefix_len(source_request->prefix_len),
        status(source_request->status),
        extra_embedding(source_request->extra_embedding) {}
};

}  // namespace allspark
