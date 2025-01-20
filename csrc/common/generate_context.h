/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    generate_context.h
 */

#pragma once

#if ENABLE_SPAN_ATTENTION
#include <cache/prefix_cache_manager.h>
#include <cache/virtual_cache.h>
#endif
#include <common/common.h>
#include <common/request.h>
#include <core/tensor/cache_memory.h>

#include <memory>
#include <vector>

namespace allspark {

#ifdef ENABLE_JSON_MODE
namespace util {
class FormatEnforcer;
}
#endif

enum class AsyncStage : int {
  INIT = 0,
  PROCESSING = 1,
  DONE = 2,
};

struct GenerateContext {
  int step = 0;
  int in_length_bias = 0;
  int max_length = 0;
  int num_beams = 1;
  int batch_size = 1;
  bool need_init = false;
  bool only_decoder = false;
  bool finish = false;
  int current_batch = 0;
  int generate_method = 0;  // sample=0,beamsearch=1
  bool async = false;
  bool gen_over[1024];  // max_batch = 1024
  int input_len = 0;
  int real_input_len = 0;  // for rich_embedding
  int prefix_len = 0;
  bool is_first_round = true;  // 一串异步请求里的第一个，即"首包"
  AsyncStage async_stage = AsyncStage::INIT;
  std::string uuid = "default-gen-ctx-uuid";
  GenerateConfig gen_cfg;
  std::vector<std::vector<int>> bad_words_ids;
  std::shared_ptr<Request> request;
  int engine_max_length = 0;

  /// @deprecated Use virtual cache instead.
  std::vector<std::unique_ptr<CacheMemory>> k_cache_list, v_cache_list;

#if ENABLE_SPAN_ATTENTION
  std::unique_ptr<VirtualCache> virtual_k_cache;
  std::unique_ptr<VirtualCache> virtual_v_cache;
  std::vector<PrefixCacheManager::PrefixNodePtr> prefix_cache_node_list;
#endif

#ifdef ENABLE_JSON_MODE
  std::shared_ptr<util::FormatEnforcer> format_enforcer;
#endif

  // for random sampling algo of SampleOp,only for cpu or cuda PhiloxCudaState
  std::unique_ptr<AsTensor> sample_state = nullptr;
};

using GenContextList = std::vector<std::shared_ptr<GenerateContext>>;
class LayerCacheManager {
 public:
  AsTensor* GetCache(std::string cache_name) {
    return cache_map_[cache_name].get();
  }
  void CreateCache(std::string cache_name, std::unique_ptr<AsTensor> cache) {
    cache_map_[cache_name] = std::move(cache);
    cache_set_map[cache_name] = false;
  }
  bool IsCacheSet(std::string cache_name) {
    if (cache_set_map.count(cache_name))
      return cache_set_map[cache_name];
    else
      return false;
  }
  void SetCache(std::string cache_name) { cache_set_map[cache_name] = true; }
  void ResetCache(std::string cache_name) { cache_set_map[cache_name] = false; }

 private:
  TensorMap cache_map_;
  std::map<std::string, bool> cache_set_map;
};
class RuntimeContext {
 public:
  bool is_context = false;
  int current_batch = 0;
  int generate_method = 0;
  std::vector<int> logprobs_indice_host;
  std::vector<float> logprobs_value_host;
  std::vector<float> token_logprobs_host;

  std::shared_ptr<GenerateContext> GetContextGenCtx() const {
    return gen_ctx_list[current_batch];
  }
  std::shared_ptr<GenerateContext> GetGenCtx(int index) const {
    return gen_ctx_list[index];
  }
  int GetGenCtxListSize() const { return gen_ctx_list.size(); }
  void PushBackGenCtx(std::shared_ptr<GenerateContext> gen_ctx) {
    gen_ctx_list.push_back(gen_ctx);
    gen_ctx_list[gen_ctx_list.size() - 1]->current_batch =
        gen_ctx_list.size() - 1;
  }
  void PopBackGenCtx() { gen_ctx_list.pop_back(); }
  void FinishRequest(int index) {
    int batch_size = gen_ctx_list.size();
    gen_ctx_list[index]->request->finish = true;
    gen_ctx_list[index]->request->status =
        AsEngine::GenerateRequestStatus::GenerateFinished;
    gen_ctx_list[index] = std::move(gen_ctx_list[batch_size - 1]);
    gen_ctx_list[index]->current_batch = index;
    gen_ctx_list.pop_back();
  }
  void CreateLayerCacheManager() {
    layer_cache_manager = std::make_shared<LayerCacheManager>();
  }
  std::shared_ptr<LayerCacheManager> GetLayerCacheManager() {
    return layer_cache_manager;
  }

 private:
  GenContextList gen_ctx_list = std::vector<std::shared_ptr<GenerateContext>>();
  std::shared_ptr<LayerCacheManager> layer_cache_manager;
};

}  // namespace allspark
