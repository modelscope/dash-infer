/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    prefix_cache_manager.cpp
 */

#if ENABLE_SPAN_ATTENTION
#include "prefix_cache_manager.h"

#include <iomanip>
#include <sstream>

#include "core/tensor/tensor.h"
#include "cuda/cuda_context.h"
#include "prefix_cache_manager_member.hpp"

#define ENABLE_PREFIX_CACHE_LOG 0
#define PREFIX_DLOG(level) \
  if (ENABLE_PREFIX_CACHE_LOG) LOG(level)

namespace allspark {

PrefixCacheManager::PrefixCacheManager(
    const CacheSpanManager::Ptr& span_manager,
    const CacheFrameManager::Ptr& frame_manager,
    const PrefixCacheCoordinator::Ptr& coordinator, const TensorMap* tensor_map,
    const DeviceContext* ctx, const int ttl = 300)
    : coordinator_(coordinator), tensor_map_(tensor_map) {
  const int nranks = ctx->GetNranks();
  const int max_length = ctx->GetModelMaxLength();
  const int max_batch = ctx->GetModelMaxBatch();
  const AsCacheMode cache_mode = ctx->GetCacheMode();
  layer_num_ = ctx->GetDecoderLayer();
  data_type_ = ctx->GetDtype();
  token_per_span_ = ctx->GetCacheSpanSize();
  stream_ = static_cast<const CUDAContext*>(ctx)->GetStream();
  ctx_ = ctx;
  time_to_live_sec_ = (float)ttl;

  int span_per_node = 2 * layer_num_;
  int capacity = (frame_manager->CountFrame() /
                  span_per_node);  // warm-up stage, use all spans

  gpu_union_ = std::make_shared<CacheUnion>(DeviceType::CUDA, capacity,
                                            span_manager, frame_manager);
  LOG(INFO) << "[PrefixCacheManager] gpu prefix cache capacity: " << capacity
            << ", able to cache: " << capacity * token_per_span_ << " tokens";

  quant_mode_ = CacheUtils::toQuantMode(cache_mode);

  const int max_num_spans =
      (max_length + token_per_span_ - 1) / token_per_span_;
  spanptr_tensor_device_ = std::make_shared<AsTensor>(
      "prefix_cache_spanptr_tensor_device", DeviceType::CUDA, DataType::POINTER,
      DataMode::DENSE, Shape{max_num_spans * span_per_node});

  min_capacity_ = max_num_spans * max_batch;

  init_cpu_union(nranks, max_length, max_batch);
  threadpool_ = std::make_unique<ThreadPool>(threadpool_size_);
}

/***************************************
 * public functions
 ***************************************/

void PrefixCacheManager::UpdateCapacity() {
  int span_per_node = 2 * layer_num_;
  int capacity = prefix_cache_ratio_ *
                 (gpu_union_->frame_manager->CountFrame() / span_per_node);

  gpu_union_->capacity = capacity;

  if (capacity < min_capacity_) {
    LOG(ERROR) << __FUNCTION__ << ": gpu memory capacity"
               << " (" << capacity << ") "
               << " < minimum required capacity"
               << " (" << min_capacity_ << "). "
               << "Cache capacity must be large enough to store at least "
                  "max_batch * max_length tokens.";
    AS_THROW(AsStatus::ALLSPARK_RUNTIME_ERROR);
  }

  LOG(INFO) << "[PrefixCacheManager] gpu prefix cache capacity: " << capacity
            << ", able to cache: " << capacity * token_per_span_ << " tokens";
}

void PrefixCacheManager::SetSeqlenThre(int thre) {
  seq_len_thre_ = thre;
  LOG(INFO) << "[PrefixCacheManager] "
            << "seq_len_thre_: " << seq_len_thre_;
}

void PrefixCacheManager::Insert(
    const std::shared_ptr<AsTensor>& tokens, const int start_idx,
    const NodeTimestamp ts,
    const std::vector<std::unique_ptr<CacheArray>>& layer_cache_k,
    const std::vector<std::unique_ptr<CacheArray>>& layer_cache_v,
    std::vector<std::string>& hash_vec) {
  auto profiler = std::make_shared<Profiler>(__FUNCTION__, &timecost_table_);

  coordinator_->waitForOtherWorkers();

  int layer_num = layer_cache_k.size();
  int token_num = tokens->GetShape().Count();
  int new_hash_idx = hash_vec.size();
  std::vector<PrefixNodePtr> insert_vec;
  std::vector<PrefixNodePtr> swap_vec;
  std::vector<std::string> victim_hash_vec;

  for (int i = start_idx + (token_per_span_ - 1),
           cache_idx = start_idx / token_per_span_;
       i < token_num; i += token_per_span_, cache_idx++) {
    std::string hash_str = hash_tokens(
        tokens->GetDataPtr(), (i + 1) * SizeofType(tokens->GetDataType()));
    hash_vec.emplace_back(hash_str);

    std::vector<CacheSpan::Ptr> k_cache_tmp(layer_num);
    std::vector<CacheSpan::Ptr> v_cache_tmp(layer_num);
    for (int layer_idx = 0; layer_idx < layer_num; layer_idx++) {
      CacheSpan::Ptr k_cache =
          layer_cache_k[layer_idx]->GetCacheVector()[cache_idx];
      CacheSpan::Ptr v_cache =
          layer_cache_v[layer_idx]->GetCacheVector()[cache_idx];

      k_cache_tmp[layer_idx] = k_cache;
      v_cache_tmp[layer_idx] = v_cache;
    }
    PrefixNodePtr new_cuda_node = nullptr;
    create_node(new_cuda_node, k_cache_tmp, v_cache_tmp, i + 1, ts,
                DeviceType::CUDA, hash_str);
    set_node_ready(new_cuda_node);

    bool is_cached = transverse_hash_table(hash_str, DeviceType::CUDA);
    if (is_cached == true) {
      // kv cache is already in gpu hash table,
      // thus update node reference
      ref_node(hash_str, ts, DeviceType::CUDA);
      PREFIX_DLOG(INFO) << __FUNCTION__
                        << ": 1, node exist in gpu hash_table, hash: "
                        << hash_str;
    } else {
      PREFIX_DLOG(INFO) << __FUNCTION__
                        << ": 2, node not exist in gpu hash_table, hash: "
                        << hash_str;
      if (gpu_union_->hash_table.size() < gpu_union_->capacity) {
        insert_node(hash_str, new_cuda_node, DeviceType::CUDA);
        ref_node(hash_str, ts, DeviceType::CUDA);
        PREFIX_DLOG(INFO) << __FUNCTION__ << ": 2.1, insert to gpu hash_table, "
                          << "hash: " << hash_str << ", "
                          << gpu_union_->hash_table.size() << "/"
                          << gpu_union_->capacity;
      } else {
        PREFIX_DLOG(INFO) << __FUNCTION__ << ": 2.2, gpu hash_table is full, "
                          << "hash: " << hash_str;
        std::string victim_hash = gpu_union_->evictor.FindVictim();

        if (victim_hash != DEFAULT_HASH_STR) {
          victim_hash_vec.emplace_back(victim_hash);
          gpu_union_->evictor.Del(victim_hash);
          insert_vec.emplace_back(new_cuda_node);
          // clang-format off
          PREFIX_DLOG(INFO)
              << __FUNCTION__
              << ": 2.2.1, find victim, "
              << "insert to gpu hash_table later, "
              << "hash: " << hash_str << ", "
              << "swap victim to cpu hash_table later, "
              << "victim hash: " << victim_hash << ", "
              << gpu_union_->hash_table.size() << "/"
              << gpu_union_->capacity << ", "
              << cpu_union_->hash_table.size() << "/"
              << cpu_union_->capacity;
          // clang-format on
        } else {
          swap_vec.emplace_back(new_cuda_node);
          PREFIX_DLOG(INFO)
              << __FUNCTION__
              << ": 2.2.2, no victim, swap new node to cpu hash_table later, "
              << gpu_union_->hash_table.size() << "/" << gpu_union_->capacity;
        }
      }
    }

    is_cached = transverse_hash_table(hash_str, DeviceType::CPU);
    if (is_cached == true) {
      PREFIX_DLOG(INFO) << __FUNCTION__
                        << ": 3, node exist in cpu hash_table, hash: "
                        << hash_str;
    }
  }

  filter_timeout_hash(victim_hash_vec);
  swap_to_cpu_by_hashlist(victim_hash_vec);
  delete_multinodes(victim_hash_vec, DeviceType::CUDA);

  for (PrefixNodePtr& node : insert_vec) {
    insert_node(node->hash, node, DeviceType::CUDA);
    ref_node(node->hash, ts, DeviceType::CUDA);
  }

  swap_to_cpu_by_nodelist(swap_vec);

  for (int idx = new_hash_idx; idx < hash_vec.size(); idx++) {
    ref_node(hash_vec[idx], ts, DeviceType::CPU);
  }

  PREFIX_DLOG(INFO) << __FUNCTION__ << ": 4, after insert, "
                    << gpu_union_->hash_table.size() << "/"
                    << gpu_union_->capacity;
}

void PrefixCacheManager::RefOnly(const std::shared_ptr<AsTensor>& tokens,
                                 const NodeTimestamp ts, int& prefix_len,
                                 int& gpu_cached_len,
                                 std::vector<std::string>& hash_vec) {
  auto profiler = std::make_shared<Profiler>(__FUNCTION__, &timecost_table_);

  int start_idx = 0;
  int token_num = tokens->GetShape().Count();
  int uncached_idx = 0;
  gpu_cached_len = 0;

  for (int i = start_idx + (token_per_span_ - 1); i < token_num - 1;
       i += token_per_span_) {
    std::string hash_str = hash_tokens(
        tokens->GetDataPtr(), (i + 1) * SizeofType(tokens->GetDataType()));

    bool is_cached = transverse_hash_table(hash_str, DeviceType::CUDA);
    if (is_cached == true) {
      // found in gpu hash_table
      gpu_cached_len += token_per_span_;

      // prevent the cache from being released when gpu memory is not enough
      ref_node(hash_str, ts, DeviceType::CUDA);

      PREFIX_DLOG(INFO) << __FUNCTION__ << ": 1, ref gpu node, "
                        << gpu_union_->evictor.Size() << "/"
                        << gpu_union_->hash_table.size() << "/"
                        << gpu_union_->capacity << ", "
                        << "hash: " << hash_str << ", "
                        << "ref_cnt: "
                        << gpu_union_->hash_table[hash_str]->ref_cnt;
    }

    bool is_cached_cpu = transverse_hash_table(hash_str, DeviceType::CPU);
    if (is_cached_cpu == true) {
      // found in cpu hash_table
      ref_node(hash_str, ts, DeviceType::CPU);

      PREFIX_DLOG(INFO) << __FUNCTION__ << ": 2, ref cpu node, "
                        << cpu_union_->evictor.Size() << "/"
                        << cpu_union_->hash_table.size() << "/"
                        << cpu_union_->capacity << ", "
                        << "hash: " << hash_str << ", "
                        << "ref_cnt: "
                        << cpu_union_->hash_table[hash_str]->ref_cnt;
    }

    if (is_cached || is_cached_cpu) {
      uncached_idx += token_per_span_;
      hash_vec.emplace_back(hash_str);
    } else {
      break;
    }
  }

  prefix_len = uncached_idx;
}

void PrefixCacheManager::RefFill(
    const std::shared_ptr<AsTensor>& tokens,
    const std::shared_ptr<AsTensor>& tokens_for_hash,
    std::shared_ptr<AsTensor>& new_tokens, const NodeTimestamp ts,
    int& prefix_len, std::unique_ptr<VirtualCache>& virtual_k_cache,
    std::unique_ptr<VirtualCache>& virtual_v_cache,
    std::vector<std::string>& hash_vec) {
  auto profiler = std::make_shared<Profiler>(__FUNCTION__, &timecost_table_);

  int start_idx = 0;
  int token_num = tokens->GetShape().Count();
  int layer_num = virtual_k_cache->GetLayerNum();
  std::vector<std::string> hash_vec_cpu;
  hash_vec.clear();

  int uncached_idx = 0;
  for (int i = start_idx + (token_per_span_ - 1),
           cache_idx = start_idx / token_per_span_;
       i < token_num - 1; i += token_per_span_, cache_idx++) {
    std::string hash_str =
        hash_tokens(tokens_for_hash->GetDataPtr(),
                    (i + 1) * SizeofType(tokens_for_hash->GetDataType()));

    bool is_cached = transverse_hash_table(hash_str, DeviceType::CUDA);
    if (is_cached == true) {
      // found in gpu hash_table
      PREFIX_DLOG(INFO) << __FUNCTION__ << ": 1, ref gpu node, "
                        << "hash: " << hash_str << ", "
                        << "ref_cnt: "
                        << gpu_union_->hash_table[hash_str]->ref_cnt;
      uncached_idx += token_per_span_;
      ref_node(hash_str, ts, DeviceType::CUDA);
      ref_node(hash_str, ts, DeviceType::CPU);
      virtual_k_cache->FillCache(gpu_union_->hash_table[hash_str]->k_cache);
      virtual_v_cache->FillCache(gpu_union_->hash_table[hash_str]->v_cache);
      hash_vec.emplace_back(hash_str);
    } else {
      // not found in gpu hash_table
      PREFIX_DLOG(INFO) << __FUNCTION__
                        << ": 2, not found in gpu hash_table, hash: "
                        << hash_str;
      is_cached = transverse_hash_table(hash_str, DeviceType::CPU);
      if (is_cached == true) {
        // found in cpu hash_table
        PREFIX_DLOG(INFO) << __FUNCTION__
                          << ": 2.1, found in cpu hash_table, hash: "
                          << hash_str;
        uncached_idx += token_per_span_;
        PrefixNodePtr& cpu_node = cpu_union_->hash_table[hash_str];

        while (!check_node_ready(cpu_node)) {
          usleep(100);
          LOG(INFO) << "[PrefixCacheManager] " << __FUNCTION__
                    << " waiting for node ready, hash: " << hash_str;
        }
        hash_vec_cpu.emplace_back(hash_str);
      } else {
        // not found
        PREFIX_DLOG(INFO) << __FUNCTION__
                          << ": 3, cache not found, hash: " << hash_str;
        break;
      }
    }
  }

  prefix_len = uncached_idx;
  if (hash_vec_cpu.size() > 0) {
    int cpu_cached_len = hash_vec_cpu.size() * token_per_span_;
    if (cpu_cached_len >= seq_len_thre_) {
      int exceed_num = gpu_union_->hash_table.size() + hash_vec_cpu.size() -
                       gpu_union_->capacity;
      if (exceed_num > 0) {
        evict_unrefered_by_num(exceed_num);
        PREFIX_DLOG(INFO) << __FUNCTION__ << ": 2.2, gpu hash_table is full, "
                          << "evict node_num: " << exceed_num << ", "
                          << "hash_vec_cpu.size(): " << hash_vec_cpu.size()
                          << ", " << gpu_union_->hash_table.size() << "/"
                          << gpu_union_->capacity;
      }

      swap_to_gpu_by_hashlist(hash_vec_cpu, virtual_k_cache, virtual_v_cache);
      for (auto& hash_str : hash_vec_cpu) {
        ref_node(hash_str, ts, DeviceType::CUDA);
        ref_node(hash_str, ts, DeviceType::CPU);
        hash_vec.emplace_back(hash_str);
        PREFIX_DLOG(INFO) << __FUNCTION__ << ": 2.3, swap cpu node to gpu, "
                          << "hash: " << hash_str << ", "
                          << gpu_union_->hash_table.size() << "/"
                          << gpu_union_->capacity << ", "
                          << cpu_union_->hash_table.size() << "/"
                          << cpu_union_->capacity;
      }
    } else {
      prefix_len -= cpu_cached_len;
    }
  }

  new_tokens = std::make_shared<AsTensor>(
      "new_input_ids", tokens->GetDeviceType(), tokens->GetDataType(),
      tokens->GetDataMode(), Shape({1, token_num - prefix_len}));
  TensorUtils::DeepCopyMatrix2D(*new_tokens, *tokens, prefix_len, 0);
}

void PrefixCacheManager::UnRef(const std::vector<std::string>& hash_vec) {
  for (std::string hash_str : hash_vec) {
    bool is_cached = transverse_hash_table(hash_str, DeviceType::CUDA);
    if (is_cached) {
      unref_node(hash_str, DeviceType::CUDA);
      PREFIX_DLOG(INFO) << __FUNCTION__ << ": 1, unref gpu node, "
                        << gpu_union_->evictor.Size() << "/"
                        << gpu_union_->hash_table.size() << "/"
                        << gpu_union_->capacity << ", "
                        << "hash: " << hash_str << ", "
                        << "ref_cnt: "
                        << gpu_union_->hash_table[hash_str]->ref_cnt;
    }

    bool is_cached_cpu = transverse_hash_table(hash_str, DeviceType::CPU);
    if (is_cached_cpu) {
      unref_node(hash_str, DeviceType::CPU);
      PREFIX_DLOG(INFO) << __FUNCTION__ << ": 2, unref cpu node, "
                        << cpu_union_->evictor.Size() << "/"
                        << cpu_union_->hash_table.size() << "/"
                        << cpu_union_->capacity << ", "
                        << "hash: " << hash_str << ", "
                        << "ref_cnt: "
                        << cpu_union_->hash_table[hash_str]->ref_cnt;
    }
  }
  return;
}

void PrefixCacheManager::UpdateCnt(int hit_cnt, int miss_cnt) {
  hit_cnt_ += hit_cnt;
  miss_cnt_ += miss_cnt;
}

void PrefixCacheManager::Reset() {
  auto lambda = [&](CacheUnionPtr union_ptr) {
    for (auto it = union_ptr->hash_table.begin();
         it != union_ptr->hash_table.end(); ++it) {
      unref_cache_span(it->first, union_ptr->device_type);
      free_cache_span(it->first, union_ptr->device_type);
    }
    union_ptr->hash_table.clear();
    union_ptr->evictor.Reset();
  };

  wait_for_async_tasks();

  lambda(gpu_union_);
  lambda(cpu_union_);

  hit_cnt_ = 0;
  miss_cnt_ = 0;
  timecost_table_.clear();

  PrintPrefixCacheInfo("Reset");
}

void PrefixCacheManager::EvictUnrefered(int target) {
  auto profiler = std::make_shared<Profiler>(__FUNCTION__, &timecost_table_);

  int frame_num = target - gpu_union_->frame_manager->CountFreeFrame();
  int node_num = (frame_num + layer_num_ * 2 - 1) / (layer_num_ * 2);

  evict_unrefered_by_num(node_num);
}

void PrefixCacheManager::EvictAllUnrefered() {
  std::vector<std::string> hash_list =
      gpu_union_->evictor.GetAllCandidatesHash();
  filter_timeout_hash(hash_list);
  swap_to_cpu_by_hashlist(hash_list);
  delete_multinodes(hash_list, DeviceType::CUDA);
}

void PrefixCacheManager::UpdateEngineStat(AsEngineStat* as_stat) {
  int64_t total_cnt = hit_cnt_ + miss_cnt_;
  as_stat->prefix_cache_hit_token = hit_cnt_;
  as_stat->prefix_cache_miss_token = miss_cnt_;
  if (total_cnt > 0) {
    as_stat->prefix_cache_hit_rate = hit_cnt_ * 1.0 / total_cnt;
    as_stat->prefix_cache_miss_rate = miss_cnt_ * 1.0 / total_cnt;
  } else {
    as_stat->prefix_cache_hit_rate = 0;
    as_stat->prefix_cache_miss_rate = 0;
  }
}

/***************************************
 * public functions for debug
 ***************************************/

void PrefixCacheManager::PrintPrefixCacheInfo(std::string extra_info = "") {
  int64_t total_cnt = hit_cnt_ + miss_cnt_;
  float hit_rate = total_cnt == 0 ? 0 : hit_cnt_ * 1.0 / total_cnt;
  float miss_rate = total_cnt == 0 ? 0 : miss_cnt_ * 1.0 / total_cnt;

  // clang-format off
  std::stringstream ss;
  ss << "[PrefixCacheManager] "
     << "cached node cnt: " << gpu_union_->hash_table.size() << "/"
     << gpu_union_->capacity << ", "
     << "unref node cnt: " << gpu_union_->evictor.Size() << ", "
     << "cached cpu node cnt: " << cpu_union_->hash_table.size() << "/"
     << cpu_union_->capacity << ", "
     << "unref cpu node cnt: " << cpu_union_->evictor.Size() << ", "
     << "hit rate: " << hit_rate << ", "
     << "(" << hit_cnt_ << "/" << total_cnt << "), "
     << "miss rate: " << miss_rate << ", "
     << "(" << miss_cnt_ << "/" << total_cnt << ")";

  /*
   * frame_size calc: CacheUtils::GetSpanSizeInBytes
   * frame_size =
   *     span_size * num_kv_heads * (hidden_size / num_q_head) * sizeof(datatype)
   * occupied_frame_num = 2 * span_num * layer_num
   */
  ss << ", "
     << "occupied frame: " << gpu_union_->frame_manager->CountOccupiedFrame()
     << "/" << gpu_union_->frame_manager->CountFreeFrame()
     << "/" << gpu_union_->frame_manager->CountFrame();
  ss << ", "
     << "occupied cpu frame: "
     << cpu_union_->frame_manager->CountOccupiedFrame() << "/"
     << cpu_union_->frame_manager->CountFreeFrame() << "/"
     << cpu_union_->frame_manager->CountFrame();

  if (extra_info != "")
    ss << ", "
       << "extra info: " << extra_info;

  LOG(INFO) << ss.str();
  // clang-format on
}

void PrefixCacheManager::PrintTimecost() {
  std::stringstream ss;
  ss << "[PrefixCacheManager] timecost: ";
  for (auto it = timecost_table_.begin(); it != timecost_table_.end(); it++) {
    ss << it->first << ": " << it->second << " ms";
    if (std::next(it) != timecost_table_.end()) {
      ss << ", ";
    }
  }

  LOG(INFO) << ss.str();
}

#if ENABLE_PREFIX_CACHE_DEBUG_API
void PrefixCacheManager::PrintAllPrefixCache(DeviceType device_type) {
  CacheUnionPtr union_ptr = gpu_union_;
  std::string device = "GPU";
  if (device_type == DeviceType::CPU) {
    union_ptr = cpu_union_;
    device = "CPU";
  }

  std::stringstream ss;

  ss << "\n###################################################\n"
     << "# [PrefixCacheManager] " << __FUNCTION__ << " on " << device << "\n"
     << "###################################################\n";

  ss << "-------------------\n"
     << "| hash table "
     << "\n"
     << "-------------------\n"
     << to_string(device_type) << "\n";

  ss << "\n-------------------\n"
     << "| evictor table "
     << "\n"
     << "-------------------\n"
     << union_ptr->evictor.ToString() << "\n";

  LOG(INFO) << ss.str();
}

void PrefixCacheManager::CheckHash(const std::shared_ptr<AsTensor>& tokens,
                                   std::vector<std::string>& hash_vec,
                                   std::string request_id) {
  int start_idx = 0;
  int token_num = tokens->GetShape().Count();

  int expected_size = token_num / token_per_span_;
  if (expected_size != hash_vec.size()) {
    LOG(INFO) << "[PrefixCacheManager] CheckHash ERROR! "
              << "token_num: " << token_num << ", "
              << "expected hash size: " << token_num / token_per_span_ << ", "
              << "hash_vec.size(): " << hash_vec.size() << ", "
              << "request_id: " << request_id;
    return;
  }

  int cache_idx = start_idx / token_per_span_;
  for (int i = start_idx + (token_per_span_ - 1); i < token_num;
       i += token_per_span_, cache_idx++) {
    std::string hash_str = hash_tokens(
        tokens->GetDataPtr(), (i + 1) * SizeofType(tokens->GetDataType()));
    if (cache_idx < hash_vec.size()) {
      if (hash_vec[cache_idx] != hash_str) {
        LOG(INFO) << "[PrefixCacheManager] CheckHash ERROR! "
                  << "cache_idx: " << cache_idx << ", "
                  << "hash_vec.size: " << hash_vec.size() << ", "
                  << "hash_vec[cache_idx]: " << hash_vec[cache_idx] << ", "
                  << "hash_str: " << hash_str << ", "
                  << "request_id: " << request_id;
        // abort();
      }
    }
  }
}

void PrefixCacheManager::SwapByHashList(std::vector<std::string>& hash_vec) {
  auto profiler = std::make_shared<Profiler>(__FUNCTION__, &timecost_table_);
  swap_to_cpu_by_hashlist(hash_vec);
}

void PrefixCacheManager::RemoveByHashList(std::vector<std::string>& hash_vec) {
  auto lambda = [&](CacheUnionPtr union_ptr) {
    for (std::string& hash_str : hash_vec) {
      delete_node(hash_str, union_ptr->device_type);
    }
  };
  lambda(gpu_union_);
}
#endif  // ENABLE_PREFIX_CACHE_DEBUG_API
}  // namespace allspark
#endif
