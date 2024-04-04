/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    generate_context.h
 */

#pragma once
#include <common/common.h>
#include <common/request.h>
#include <core/tensor/cache_memory.h>

#include <memory>
#include <vector>

namespace allspark {

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
  bool is_first_round = true;  // 一串异步请求里的第一个，即"首包"
  AsyncStage async_stage = AsyncStage::INIT;
  std::string uuid = "default-gen-ctx-uuid";
  GenerateConfig gen_cfg;
  std::vector<std::vector<int>> bad_words_ids;
  std::shared_ptr<Request> request;
  int engine_max_length = 0;

  /// @deprecated Use virtual cache instead.
  std::vector<std::unique_ptr<CacheMemory>> k_cache_list, v_cache_list;

  // for random sampling algo of SampleOp
  std::unique_ptr<AsTensor> sample_state = nullptr;
};

using GenContextList = std::vector<std::unique_ptr<GenerateContext>>;
class LayerCacheManager {
 public:
  AsTensor* GetCache(std::string cache_name) {
    return cache_map_[cache_name].get();
  }
  void CreateCache(std::string cache_name, std::unique_ptr<AsTensor> cache) {
    cache_map_[cache_name] = std::move(cache);
    cache_set_map[cache_name] = false;
  }
  bool IsCacheSet(std::string cache_name) { return cache_set_map[cache_name]; }
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
  std::vector<int64_t> logprobs_indice_host;
  std::vector<float> logprobs_value_host;
  GenerateContext* GetContextGenCtx() const {
    return gen_ctx_list[current_batch].get();
  }
  GenerateContext* GetGenCtx(int index) const {
    return gen_ctx_list[index].get();
  }
  int GetGenCtxListSize() const { return gen_ctx_list.size(); }
  void PushBackGenCtx(std::unique_ptr<GenerateContext> gen_ctx) {
    gen_ctx_list.push_back(std::move(gen_ctx));
    gen_ctx_list[gen_ctx_list.size() - 1]->current_batch =
        gen_ctx_list.size() - 1;
  }
  void PopBackGenCtx() { gen_ctx_list.pop_back(); }
  void FinishRequest(int index) {
    int batch_size = gen_ctx_list.size();
    gen_ctx_list[index]->request->finish = true;
    gen_ctx_list[index]->request->status =
        AsEngine::GenerateRequestStatus::GenerateFinished;
    // LOG(INFO) << "uuid = " << gen_ctx_list[index]->request->request_id <<
    // "finish = "<< gen_ctx_list[index]->request->finish<<"FinishRequest "
    // <<std::endl;
    gen_ctx_list[index] = std::move(gen_ctx_list[batch_size - 1]);
    gen_ctx_list[index]->current_batch = index;
    gen_ctx_list.pop_back();
  }
  std::shared_ptr<LayerCacheManager> CreateLayerCacheManager() {
    layer_cache_manager = std::make_shared<LayerCacheManager>();
    return layer_cache_manager;
  }
  std::shared_ptr<LayerCacheManager> GetLayerCacheManager() {
    return layer_cache_manager;
  }

 private:
  GenContextList gen_ctx_list = std::vector<std::unique_ptr<GenerateContext>>();
  std::shared_ptr<LayerCacheManager> layer_cache_manager;
};

}  // namespace allspark
