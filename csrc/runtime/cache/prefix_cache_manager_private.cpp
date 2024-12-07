/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    prefix_cache_manager_private.cpp
 */

#if ENABLE_SPAN_ATTENTION
#include <iomanip>
#include <sstream>

#include "core/tensor/tensor.h"
#include "cuda/cuda_context.h"
#include "cuda/cuda_kernel_span_cache.h"
#include "prefix_cache_manager.h"
#include "prefix_cache_manager_member.hpp"
#include "utility/datatype_dispatcher.h"

namespace allspark {

/***************************************
 * privte functions
 ***************************************/

void PrefixCacheManager::init_cpu_union(const int nranks,
                                        const int engine_max_length,
                                        const int engine_max_batch) {
  float ratio = 0.0;
  const char* mem_ratio = std::getenv("CPU_CACHE_RATIO");
  if (mem_ratio) {
    try {
      float ratio_tmp = std::stof(std::string(mem_ratio));
      if (ratio_tmp < 0 || ratio_tmp > 1) {
        LOG(WARNING) << "invalid float range for env var CPU_CACHE_RATIO: "
                     << mem_ratio << ", will use CPU_CACHE_RATIO=" << ratio
                     << std::endl;
      } else {
        ratio = ratio_tmp;
      }
    } catch (std::exception& e) {
      LOG(WARNING) << "invalid float format for env var CPU_CACHE_RATIO: "
                   << mem_ratio << ", will use CPU_CACHE_RATIO=" << ratio
                   << std::endl;
    }
  }

  int kv_cnt = 2;
  size_t available_memory = coordinator_->getCpuAvailMem() * 1024;
  size_t cache_mem_bytes = (size_t)(available_memory * ratio);
  size_t frame_size = gpu_union_->span_manager->GetSpanSize();
  size_t frame_per_node =
      kv_cnt *
      layer_num_;  // number of frames needed per FIXED_SPAN_SIZE tokens
  size_t max_frame_num = 0;
  int capacity = 0;

  if (ratio > 0) {
    size_t cpu_workspace_size = MemcpyWorkspace::CalcWorkspaceSize(
        engine_max_length, engine_max_batch, token_per_span_, frame_per_node,
        frame_size);
    if (cache_mem_bytes < cpu_workspace_size * nranks) {
      LOG(ERROR) << __FUNCTION__ << ": cache_mem_bytes (" << cache_mem_bytes
                 << ") < cpu_workspace_size * nranks ("
                 << cpu_workspace_size * nranks << ")";
      AS_THROW(AsStatus::ALLSPARK_RUNTIME_ERROR);
    }

    cpu_workspace_ = std::make_shared<MemcpyWorkspace>(
        engine_max_length, engine_max_batch, token_per_span_, frame_per_node,
        frame_size);
    cache_mem_bytes -= cpu_workspace_size * nranks;

    size_t cache_mem_per_worker = (size_t)(cache_mem_bytes / nranks);
    max_frame_num = cache_mem_per_worker / frame_size;
    capacity = max_frame_num / frame_per_node;

    if (capacity < min_capacity_) {
      LOG(ERROR) << __FUNCTION__ << ": cpu memory capacity"
                 << " (" << capacity << ") "
                 << " < minimum required capacity"
                 << " (" << min_capacity_ << "). "
                 << "Cache capacity must be large enough to store at least "
                    "max_batch * max_length tokens.";
      AS_THROW(AsStatus::ALLSPARK_RUNTIME_ERROR);
    }

    const float to_GB = 1024 * 1024 * 1024;
    // clang-format off
    LOG(INFO) << "[PrefixCacheManager] "
              << "cpu available memory: " << available_memory / to_GB << " GB, "
              << "memory for workspace: " << cpu_workspace_size * nranks / to_GB << " GB, "
              << "memory for cache: " << cache_mem_bytes / to_GB << " GB, "
              << "memory for cache (per worker): "
              << cache_mem_per_worker / to_GB << " GB, "
              << "worker number: " << nranks << ", "
              << "CPU_CACHE_RATIO: " << ratio << ", "
              << "frame_size: " << frame_size << ", "
              << "max_frame_num: " << max_frame_num;
    LOG(INFO) << "[PrefixCacheManager] cpu prefix cache capacity: " << capacity << ", "
              << "able to cache: " << capacity * token_per_span_ << " tokens";
    LOG(INFO) << "[PrefixCacheManager] "
              << "Allocating cpu memory for prefix cache takes some time...";
    // clang-format on
  }

#ifdef CONFIG_CONCURRENT_SPAN
  auto frame_manager_cpu = std::make_shared<ConcurrentCacheFrameManager>(
      DeviceType::CPU, max_frame_num);
  auto span_manager_cpu =
      std::make_shared<ConcurrentCacheSpanManager>(frame_manager_cpu);
#else
  int grow_num = 0;
  auto frame_manager_cpu = std::make_shared<DefaultCacheFrameManager>(
      DeviceType::CPU, max_frame_num, grow_num);
  auto span_manager_cpu =
      std::make_shared<DefaultCacheSpanManager>(frame_manager_cpu);
#endif
  span_manager_cpu->Init(frame_size);

  cpu_union_ = std::make_shared<CacheUnion>(
      DeviceType::CPU, capacity, span_manager_cpu, frame_manager_cpu);

  return;
}

std::string PrefixCacheManager::hash_tokens(void* data, int len) {
  uint32_t hash_value = 0;
  MurmurHash3_x86_32(data, len, 0, &hash_value);

  std::stringstream ss;
  ss << std::setw(10) << std::setfill('0') << hash_value;
  return ss.str();
}

bool PrefixCacheManager::transverse_hash_table(const std::string& hash,
                                               DeviceType device_type) {
  CacheUnionPtr union_ptr = gpu_union_;
  if (device_type == DeviceType::CPU) {
    union_ptr = cpu_union_;
  }

  if (union_ptr->hash_table.count(hash) > 0)
    return true;
  else
    return false;
}

void PrefixCacheManager::ref_node(const std::string& hash,
                                  const NodeTimestamp& ts,
                                  DeviceType device_type) {
  CacheUnionPtr union_ptr = gpu_union_;
  if (device_type == DeviceType::CPU) {
    union_ptr = cpu_union_;
  }

  if (union_ptr->hash_table.count(hash) <= 0) return;

  if (union_ptr->hash_table[hash]->ref_cnt == 0) {
    ref_cache_span(hash, union_ptr->device_type);
    union_ptr->evictor.Del(hash);
  }
  union_ptr->hash_table[hash]->ref_cnt += 1;
  if (union_ptr->hash_table[hash]->last_access_time < ts) {
    union_ptr->hash_table[hash]->last_access_time = ts;
  }
}

void PrefixCacheManager::unref_node(const std::string& hash,
                                    DeviceType device_type) {
  CacheUnionPtr union_ptr = gpu_union_;
  if (device_type == DeviceType::CPU) {
    union_ptr = cpu_union_;
  }

  union_ptr->hash_table[hash]->ref_cnt -= 1;
  if (union_ptr->hash_table[hash]->ref_cnt == 0) {
    /*
     * should NOT unref span here,
     * otherwise unreferenced span will be released
     * after StopRequest
     */
    // unref_cache_span(hash, union_ptr->device_type);
    union_ptr->evictor.Add(hash, union_ptr->hash_table[hash]);
  }
}

bool PrefixCacheManager::insert_node(const std::string& hash,
                                     PrefixNodePtr node,
                                     DeviceType device_type) {
  if (node == nullptr) {
    LOG(ERROR) << __FUNCTION__ << ": node is nullptr";
    return false;
  }

  CacheUnionPtr union_ptr = gpu_union_;
  if (device_type == DeviceType::CPU) {
    union_ptr = cpu_union_;
  }

  if (union_ptr->hash_table.size() < union_ptr->capacity) {
    union_ptr->hash_table[hash] = node;
    ref_cache_span(hash, device_type);
    if (node->ref_cnt == 0) {
      union_ptr->evictor.Add(hash, union_ptr->hash_table[hash]);
    }
    return true;
  }

  return false;
}

void PrefixCacheManager::delete_node(const std::string& hash,
                                     DeviceType device_type) {
  CacheUnionPtr union_ptr = gpu_union_;
  if (device_type == DeviceType::CPU) {
    union_ptr = cpu_union_;
  }

  unref_cache_span(hash, union_ptr->device_type);
  free_cache_span(hash, union_ptr->device_type);
  union_ptr->hash_table.erase(hash);
  union_ptr->evictor.Del(hash);
}

void PrefixCacheManager::delete_multinodes(
    const std::vector<std::string>& hash_vec, DeviceType device_type) {
  CacheUnionPtr union_ptr = gpu_union_;
  if (device_type == DeviceType::CPU) {
    union_ptr = cpu_union_;
  }

  for (std::string hash : hash_vec) {
    unref_cache_span(hash, union_ptr->device_type);
    free_cache_span(hash, union_ptr->device_type);
    union_ptr->hash_table.erase(hash);
    union_ptr->evictor.Del(hash);
  }
}

#if ENABLE_PREFIX_CACHE_DEBUG_API
bool PrefixCacheManager::copy_node(PrefixNodePtr& src_node,
                                   PrefixNodePtr& dst_node) {
  if (src_node == nullptr || dst_node == nullptr) {
    LOG(ERROR) << __FUNCTION__ << ": src or dst node is nullptr";
    return false;
  }

  if (src_node->cache_num != dst_node->cache_num) {
    LOG(ERROR) << __FUNCTION__ << ": cache_num not equal";
    return false;
  }

  PrefixNode::DeepCopy(src_node, dst_node);
  return true;
}
#endif

void PrefixCacheManager::create_node(PrefixNodePtr& new_node,
                                     std::vector<CacheSpan::Ptr>& k_cache_in,
                                     std::vector<CacheSpan::Ptr>& v_cache_in,
                                     int prefix_len, NodeTimestamp ts,
                                     DeviceType device_type,
                                     std::string hash = DEFAULT_HASH_STR) {
  // for debug
  if (hash == DEFAULT_HASH_STR) {
    LOG(ERROR) << __FUNCTION__ << ": forget to set node hash";
    AS_THROW(AsStatus::ALLSPARK_RUNTIME_ERROR);
  }

  new_node = std::make_shared<PrefixNode>(k_cache_in, v_cache_in, prefix_len,
                                          ts, device_type, hash);
}

#if ENABLE_PREFIX_CACHE_DEBUG_API
bool PrefixCacheManager::compare_node(PrefixNodePtr& node1,
                                      PrefixNodePtr& node2, int idx) {
  if (node1 == nullptr || node2 == nullptr) {
    LOG(INFO) << __FUNCTION__ << ": node1 or node2 is nullptr";
    return false;
  }

  auto cpu_node1 = node1;
  auto cpu_node2 = node2;

  if (node1->device_type != DeviceType::CPU) {
    cpu_node1 = swap_node_to_cpu(node1);
  }

  if (node2->device_type != DeviceType::CPU) {
    cpu_node2 = swap_node_to_cpu(node2);
  }

  return PrefixNode::isSame(cpu_node1, cpu_node2, idx);
}
#endif

void PrefixCacheManager::set_node_ready(PrefixNodePtr node) {
  node->is_ready = true;
}

bool PrefixCacheManager::check_node_ready(PrefixNodePtr& node) {
  return node->is_ready;
}

void PrefixCacheManager::filter_timeout_hash(
    std::vector<std::string>& hash_list) {
  auto current_ts = std::chrono::steady_clock::now();

  hash_list.erase(
      std::remove_if(
          hash_list.begin(), hash_list.end(),
          [this, current_ts](std::string hash_str) {
            PrefixNodePtr& node = gpu_union_->hash_table[hash_str];
            auto duration = current_ts - node->last_access_time;
            auto duration_in_s =
                std::chrono::duration_cast<std::chrono::milliseconds>(duration)
                    .count() /
                1000.0;
            if (duration_in_s > time_to_live_sec_) {
              delete_node(hash_str, DeviceType::CUDA);
              LOG(INFO) << "[PrefixCacheManager] "
                        << "time to last_access_time: " << duration_in_s
                        << " sec, time_to_live_sec_: " << time_to_live_sec_
                        << " sec";
              return true;
            }
            return false;
          }),
      hash_list.end());
}

void PrefixCacheManager::update_node_ts(const std::string& hash,
                                        const NodeTimestamp& ts,
                                        DeviceType device_type) {
  CacheUnionPtr union_ptr = gpu_union_;
  if (device_type == DeviceType::CPU) {
    union_ptr = cpu_union_;
  }

  if (union_ptr->hash_table[hash]->last_access_time < ts) {
    union_ptr->hash_table[hash]->last_access_time = ts;
  }
}

bool PrefixCacheManager::alloc_cpu_node(PrefixNodePtr& new_node,
                                        const int cache_num, const int ref_cnt,
                                        const int prefix_len,
                                        const NodeTimestamp ts,
                                        std::string hash = DEFAULT_HASH_STR) {
  if (cpu_union_->capacity <= 0) return false;
  if (cpu_union_->frame_manager->CountFreeFrame() <= 0) return false;

  if (cpu_union_->hash_table.size() >= cpu_union_->capacity) {
    std::string victim_hash = cpu_union_->evictor.FindVictim();
    if (victim_hash != DEFAULT_HASH_STR) {
      delete_node(victim_hash, DeviceType::CPU);
    } else {
      return false;
    }
  }

  std::string tag_k = "k_cache_cpu_span";
  std::string tag_v = "v_cache_cpu_span";

  size_t claimed_cnt = 0;
  std::vector<CacheSpan::Ptr> k_cache_tmp(cache_num);
  std::vector<CacheSpan::Ptr> v_cache_tmp(cache_num);
  claimed_cnt += cpu_union_->span_manager->ClaimSpan(
      k_cache_tmp, std::move(tag_k), cache_num);
  claimed_cnt += cpu_union_->span_manager->ClaimSpan(
      v_cache_tmp, std::move(tag_v), cache_num);
  if (claimed_cnt != (cache_num * 2)) return false;

  create_node(new_node, k_cache_tmp, v_cache_tmp, prefix_len, ts,
              DeviceType::CPU, hash);

  return true;
}

bool PrefixCacheManager::alloc_gpu_node(
    PrefixNodePtr& new_node, const int cache_num, const int ref_cnt,
    const int prefix_len, const NodeTimestamp ts,
    std::unique_ptr<VirtualCache>& virtual_k_cache,
    std::unique_ptr<VirtualCache>& virtual_v_cache,
    std::string hash = DEFAULT_HASH_STR) {
  std::vector<CacheSpan::Ptr> k_cache_tmp(layer_num_);
  std::vector<CacheSpan::Ptr> v_cache_tmp(layer_num_);
  for (int layer_idx = 0; layer_idx < layer_num_; layer_idx++) {
    virtual_k_cache->GetCache(layer_idx, token_per_span_);
    virtual_v_cache->GetCache(layer_idx, token_per_span_);
    CacheSpan::Ptr k_cache =
        virtual_k_cache->GetLayerCache()[layer_idx]->GetCacheVector().back();
    CacheSpan::Ptr v_cache =
        virtual_v_cache->GetLayerCache()[layer_idx]->GetCacheVector().back();
    k_cache_tmp[layer_idx] = k_cache;
    v_cache_tmp[layer_idx] = v_cache;
  }

  create_node(new_node, k_cache_tmp, v_cache_tmp, prefix_len, ts,
              DeviceType::CUDA, hash);
  return true;
}

void PrefixCacheManager::swap_to_cpu_by_nodelist(
    std::vector<PrefixNodePtr>& src_node_list) {
  std::vector<PrefixNodePtr> gpu_node_list;
  std::vector<PrefixNodePtr> new_cpu_node_list;

  // step 1: collect all uncached nodes, and allocate new cpu nodes
  for (PrefixNodePtr& src_node : src_node_list) {
    std::string hash = src_node->hash;
    bool is_cached = transverse_hash_table(hash, DeviceType::CPU);
    if (is_cached) {
      update_node_ts(hash, src_node->last_access_time, DeviceType::CPU);
    } else {
      PrefixNodePtr new_node_ptr = nullptr;
      bool success = alloc_cpu_node(new_node_ptr, src_node->cache_num,
                                    src_node->ref_cnt, src_node->prefix_len,
                                    src_node->last_access_time, hash);
      if (success == true) {
        gpu_node_list.emplace_back(src_node);
        new_cpu_node_list.emplace_back(new_node_ptr);
      } else {
        break;
      }
    }
  }

  if (new_cpu_node_list.size() <= 0) return;

  int vec_size = new_cpu_node_list.size();
  int max_span_num = spanptr_tensor_device_->GetShape()[0];
  int max_copy_size = max_span_num / (layer_num_ * 2);
  for (int start_idx = 0; start_idx < vec_size; start_idx += max_copy_size) {
    int end_idx = start_idx + max_copy_size;
    end_idx = end_idx > vec_size ? vec_size : end_idx;

    std::vector<PrefixNodePtr> sliced_gpu_node_list(
        gpu_node_list.begin() + start_idx, gpu_node_list.begin() + end_idx);
    std::vector<PrefixNodePtr> sliced_cpu_node_list(
        new_cpu_node_list.begin() + start_idx,
        new_cpu_node_list.begin() + end_idx);
    swap_nodelist_to_cpu_impl(sliced_gpu_node_list, sliced_cpu_node_list);
  }
}

void PrefixCacheManager::swap_to_cpu_by_hashlist(
    std::vector<std::string>& hash_list) {
  std::vector<PrefixNodePtr> gpu_node_list;
  std::vector<PrefixNodePtr> new_cpu_node_list;

  // step 1: collect all uncached nodes, and allocate new cpu nodes
  for (std::string hash : hash_list) {
    PrefixNodePtr& src_node = gpu_union_->hash_table[hash];

    bool is_cached = transverse_hash_table(hash, DeviceType::CPU);
    if (is_cached) {
      update_node_ts(hash, src_node->last_access_time, DeviceType::CPU);
    } else {
      PrefixNodePtr new_node_ptr = nullptr;
      bool success = alloc_cpu_node(new_node_ptr, src_node->cache_num,
                                    src_node->ref_cnt, src_node->prefix_len,
                                    src_node->last_access_time, hash);
      if (success == true) {
        gpu_node_list.emplace_back(src_node);
        new_cpu_node_list.emplace_back(new_node_ptr);
      } else {
        break;
      }
    }
  }

  if (new_cpu_node_list.size() <= 0) return;

  int vec_size = new_cpu_node_list.size();
  int max_span_num = spanptr_tensor_device_->GetShape()[0];
  int max_copy_size = max_span_num / (layer_num_ * 2);
  for (int start_idx = 0; start_idx < vec_size; start_idx += max_copy_size) {
    int end_idx = start_idx + max_copy_size;
    end_idx = end_idx > vec_size ? vec_size : end_idx;

    std::vector<PrefixNodePtr> sliced_gpu_node_list(
        gpu_node_list.begin() + start_idx, gpu_node_list.begin() + end_idx);
    std::vector<PrefixNodePtr> sliced_cpu_node_list(
        new_cpu_node_list.begin() + start_idx,
        new_cpu_node_list.begin() + end_idx);
    swap_nodelist_to_cpu_impl(sliced_gpu_node_list, sliced_cpu_node_list);
  }
}

void PrefixCacheManager::swap_nodelist_to_cpu_impl(
    std::vector<PrefixNodePtr>& gpu_node_list,
    std::vector<PrefixNodePtr>& new_cpu_node_list) {
  // step 2: copy span
  const int node_num = gpu_node_list.size();
  if (node_num <= 0) return;

  const int cache_num = gpu_node_list[0]->cache_num;
  const int span_num = node_num * cache_num * 2;
  if (span_num > spanptr_tensor_device_->GetShape()[0]) {
    LOG(ERROR) << __FUNCTION__ << ": too many span to cache";
    AS_THROW(AsStatus::ALLSPARK_RUNTIME_ERROR);
  }

  std::vector<void*> gpu_span_list(span_num);
  for (int i = 0; i < node_num; i++) {
    PrefixNodePtr& gpu_node = gpu_node_list[i];
    for (int j = 0; j < cache_num; j++) {
      int idx = (i * cache_num + j) * 2;
      gpu_span_list[idx + 0] = gpu_node->k_cache[j]->Data();
      gpu_span_list[idx + 1] = gpu_node->v_cache[j]->Data();
    }
  }

  void* gpu_workspace_ptr = tensor_map_->at("workspace")->GetDataPtr();
  const size_t gpu_workspace_size =
      tensor_map_->at("workspace")->GetSizeInByte();
  const size_t cpu_workspace_size = cpu_workspace_->GetBlockSize();
  std::shared_ptr<void> cpu_ws_block = cpu_workspace_->QueryBlock();
  void* cpu_workspace_ptr = cpu_ws_block.get();
  const int span_size = gpu_union_->span_manager->GetSpanSize();

  if (span_num * span_size > cpu_workspace_size) {
    LOG(ERROR) << __FUNCTION__ << ": cpu workspace is too small for swap";
    AS_THROW(AsStatus::ALLSPARK_RUNTIME_ERROR);
  }

  const int max_copy_num =
      std::min(gpu_workspace_size, cpu_workspace_size) / span_size;
  const int total_copy_num = gpu_span_list.size();
  const int copy_step = std::min(max_copy_num, total_copy_num);

  for (int copy_cnt = 0; copy_cnt < total_copy_num; copy_cnt += copy_step) {
    int copy_num = std::min(copy_step, total_copy_num - copy_cnt);

    // step 2.1: device to device, discrete to continuous
    TensorUtils::Memset(*spanptr_tensor_device_, 0);

    AS_CHECK_CUDA(cudaMemcpyAsync(
        spanptr_tensor_device_->GetDataPtr(), &gpu_span_list[copy_cnt],
        copy_num * sizeof(gpu_span_list[0]), cudaMemcpyHostToDevice, stream_));
    AS_CHECK_CUDA(cudaStreamSynchronize(stream_));

    DispatchCUDA(data_type_, [&]<typename T>() {
      cuda::SpanToContCopyLauncher<T>(
          gpu_workspace_ptr,
          static_cast<const void* const*>(spanptr_tensor_device_->GetDataPtr()),
          copy_num, span_size, data_type_, quant_mode_, stream_);
    });
    const CUDAContext* gpu_ctx = dynamic_cast<const CUDAContext*>(ctx_);
    gpu_ctx->Synchronize();

    // step 2.2: device to host, continuous to continuous
    AS_CHECK_CUDA(cudaMemcpyAsync(cpu_workspace_ptr + copy_cnt * span_size,
                                  gpu_workspace_ptr, copy_num * span_size,
                                  cudaMemcpyDeviceToHost, stream_));
    AS_CHECK_CUDA(cudaStreamSynchronize(stream_));
  }

  for (int i = 0; i < node_num; i++) {
    PrefixNodePtr& new_cpu_node = new_cpu_node_list[i];
    insert_node(new_cpu_node->hash, new_cpu_node, DeviceType::CPU);
  }
  for (int i = 0; i < node_num; i++) {
    PrefixNodePtr new_cpu_node = new_cpu_node_list[i];
    threadpool_->enqueue(
        [this, i, new_cpu_node, cache_num, span_size, cpu_ws_block]() {
          for (int j = 0; j < cache_num; j++) {
            int idx = (i * cache_num + j) * 2;
            memcpy(new_cpu_node->k_cache[j]->Data(),
                   cpu_ws_block.get() + (idx + 0) * span_size, span_size);
            memcpy(new_cpu_node->v_cache[j]->Data(),
                   cpu_ws_block.get() + (idx + 1) * span_size, span_size);
          }
          set_node_ready(new_cpu_node);
        });
  }
}

#if ENABLE_PREFIX_CACHE_DEBUG_API
PrefixCacheManager::PrefixNodePtr PrefixCacheManager::swap_node_to_cpu(
    PrefixNodePtr& src_node) {
  // step 1: create a cpu node
  PrefixNodePtr new_node_ptr = nullptr;
  bool rt = alloc_cpu_node(new_node_ptr, src_node->cache_num, src_node->ref_cnt,
                           src_node->prefix_len, src_node->last_access_time,
                           src_node->hash);

  // step 2: copy data from gpu node to cpu node
  if (rt == true) {
    rt &= copy_node(src_node, new_node_ptr);
  }

  // step 3: insert cpu node to hash table
  if (rt == true) {
    set_node_ready(new_node_ptr);
  }

  return new_node_ptr;
}

bool PrefixCacheManager::swap_node_to_cpu(PrefixNodePtr& src_node,
                                          const std::string& hash) {
  bool is_cached = transverse_hash_table(hash, DeviceType::CPU);
  if (is_cached) {
    update_node_ts(hash, src_node->last_access_time, DeviceType::CPU);
    return true;
  }

  // step 1: create a cpu node
  PrefixNodePtr new_node_ptr = nullptr;
  bool rt =
      alloc_cpu_node(new_node_ptr, src_node->cache_num, src_node->ref_cnt,
                     src_node->prefix_len, src_node->last_access_time, hash);

  // step 2: copy data from gpu node to cpu node
  if (rt == true) {
    rt &= copy_node(src_node, new_node_ptr);
  }

  // step 3: insert cpu node to hash table
  if (rt == true) {
    rt &= insert_node(hash, new_node_ptr, DeviceType::CPU);
    set_node_ready(new_node_ptr);
  }

  return rt;
}
#endif

void PrefixCacheManager::swap_to_gpu_by_hashlist(
    std::vector<std::string>& hash_list,
    std::unique_ptr<VirtualCache>& virtual_k_cache,
    std::unique_ptr<VirtualCache>& virtual_v_cache) {
  std::vector<PrefixNodePtr> cpu_node_list;
  std::vector<PrefixNodePtr> new_gpu_node_list;

  // step 1: collect all unhashed nodes, and allocate new gpu nodes
  for (std::string hash : hash_list) {
    PrefixNodePtr& src_node = cpu_union_->hash_table[hash];

    bool is_cached = transverse_hash_table(hash, DeviceType::CUDA);
    if (is_cached) {
      update_node_ts(hash, src_node->last_access_time, DeviceType::CUDA);
      virtual_k_cache->FillCache(gpu_union_->hash_table[hash]->k_cache);
      virtual_v_cache->FillCache(gpu_union_->hash_table[hash]->v_cache);
    } else {
      PrefixNodePtr new_node_ptr = nullptr;
      bool success =
          alloc_gpu_node(new_node_ptr, src_node->cache_num, src_node->ref_cnt,
                         src_node->prefix_len, src_node->last_access_time,
                         virtual_k_cache, virtual_v_cache, hash);
      if (success == true) {
        cpu_node_list.emplace_back(src_node);
        new_gpu_node_list.emplace_back(new_node_ptr);
      } else {
        break;
      }
    }
  }

  if (new_gpu_node_list.size() <= 0) return;

  int vec_size = new_gpu_node_list.size();
  int max_span_num = spanptr_tensor_device_->GetShape()[0];
  int max_copy_size = max_span_num / (layer_num_ * 2);
  for (int start_idx = 0; start_idx < vec_size; start_idx += max_copy_size) {
    int end_idx = start_idx + max_copy_size;
    end_idx = end_idx > vec_size ? vec_size : end_idx;

    std::vector<PrefixNodePtr> sliced_cpu_node_list(
        cpu_node_list.begin() + start_idx, cpu_node_list.begin() + end_idx);
    std::vector<PrefixNodePtr> sliced_gpu_node_list(
        new_gpu_node_list.begin() + start_idx,
        new_gpu_node_list.begin() + end_idx);
    swap_nodelist_to_gpu_impl(sliced_cpu_node_list, sliced_gpu_node_list);
  }
}

void PrefixCacheManager::swap_nodelist_to_gpu_impl(
    std::vector<PrefixNodePtr>& cpu_node_list,
    std::vector<PrefixNodePtr>& new_gpu_node_list) {
  // step 2: copy span
  const int node_num = cpu_node_list.size();
  if (node_num <= 0) return;

  const int cache_num = cpu_node_list[0]->cache_num;
  const int span_num = node_num * cache_num * 2;
  if (span_num > spanptr_tensor_device_->GetShape()[0]) {
    LOG(ERROR) << __FUNCTION__ << ": too many span to cache";
    AS_THROW(AsStatus::ALLSPARK_RUNTIME_ERROR);
  }

  std::vector<void*> gpu_span_list(span_num);
  std::vector<void*> cpu_span_list(span_num);
  for (int i = 0; i < node_num; i++) {
    PrefixNodePtr& new_gpu_node = new_gpu_node_list[i];
    PrefixNodePtr& cpu_node = cpu_node_list[i];
    for (int j = 0; j < cache_num; j++) {
      int idx = (i * cache_num + j) * 2;
      cpu_span_list[idx + 0] = cpu_node->k_cache[j]->Data();
      cpu_span_list[idx + 1] = cpu_node->v_cache[j]->Data();
      gpu_span_list[idx + 0] = new_gpu_node->k_cache[j]->Data();
      gpu_span_list[idx + 1] = new_gpu_node->v_cache[j]->Data();
    }
  }

  void* gpu_workspace_ptr = tensor_map_->at("workspace")->GetDataPtr();
  const size_t gpu_workspace_size =
      tensor_map_->at("workspace")->GetSizeInByte();
  const size_t cpu_workspace_size = cpu_workspace_->GetBlockSize();
  std::shared_ptr<void> cpu_ws_block = cpu_workspace_->QueryBlock();
  void* cpu_workspace_ptr = cpu_ws_block.get();
  const int span_size = gpu_union_->span_manager->GetSpanSize();

  if (span_num * span_size > cpu_workspace_size) {
    LOG(ERROR) << __FUNCTION__ << ": cpu workspace is too small for swap";
    AS_THROW(AsStatus::ALLSPARK_RUNTIME_ERROR);
  }

  const int max_copy_num =
      std::min(gpu_workspace_size, cpu_workspace_size) / span_size;
  const int total_copy_num = gpu_span_list.size();
  const int copy_step = std::min(max_copy_num, total_copy_num);

  // step 2.1: host to host, discrete to continuous
  for (int i = 0; i < total_copy_num; ++i) {
    memcpy(cpu_workspace_ptr + i * span_size, cpu_span_list[i], span_size);
  }

  for (int copy_cnt = 0; copy_cnt < total_copy_num; copy_cnt += copy_step) {
    int copy_num = std::min(copy_step, total_copy_num - copy_cnt);

    // step 2.2: host to device, continuous to continuous
    AS_CHECK_CUDA(cudaMemcpyAsync(
        gpu_workspace_ptr, cpu_workspace_ptr + copy_cnt * span_size,
        copy_num * span_size, cudaMemcpyHostToDevice, stream_));
    AS_CHECK_CUDA(cudaStreamSynchronize(stream_));

    // step 2.3: device to device, continuous to discrete
    TensorUtils::Memset(*spanptr_tensor_device_, 0);

    AS_CHECK_CUDA(cudaMemcpyAsync(
        spanptr_tensor_device_->GetDataPtr(), &gpu_span_list[copy_cnt],
        copy_num * sizeof(gpu_span_list[0]), cudaMemcpyHostToDevice, stream_));
    AS_CHECK_CUDA(cudaStreamSynchronize(stream_));

    DispatchCUDA(data_type_, [&]<typename T>() {
      cuda::ContToSpanCopyLauncher<T>(
          static_cast<void**>(spanptr_tensor_device_->GetDataPtr()),
          gpu_workspace_ptr, copy_num, span_size, data_type_, quant_mode_,
          stream_);
    });
    const CUDAContext* gpu_ctx = dynamic_cast<const CUDAContext*>(ctx_);
    gpu_ctx->Synchronize();
  }

  // step 3: insert cpu node to hash table
  for (int i = 0; i < node_num; i++) {
    PrefixNodePtr& new_gpu_node = new_gpu_node_list[i];
    insert_node(new_gpu_node->hash, new_gpu_node, DeviceType::CUDA);
    set_node_ready(new_gpu_node);
  }
}

#if ENABLE_PREFIX_CACHE_DEBUG_API
bool PrefixCacheManager::swap_node_to_gpu(PrefixNodePtr& src_node,
                                          PrefixNodePtr& dst_node,
                                          const std::string& hash) {
  bool rt = copy_node(src_node, dst_node);

  if (rt == true) {
    rt &= insert_node(hash, dst_node, DeviceType::CUDA);
    set_node_ready(dst_node);
  }
  return rt;
}
#endif

void PrefixCacheManager::evict_unrefered_by_num(int node_num) {
  std::vector<std::string> victim_hash_list =
      gpu_union_->evictor.FindMultiVictims(node_num);

  if (victim_hash_list.size() < node_num) {
    LOG(ERROR) << __FUNCTION__ << ": victim_num (" << victim_hash_list.size()
               << ") < required node_num (" << node_num << ")";
  }

  filter_timeout_hash(victim_hash_list);
  swap_to_cpu_by_hashlist(victim_hash_list);
  delete_multinodes(victim_hash_list, DeviceType::CUDA);
}

void PrefixCacheManager::ref_cache_span(const std::string& hash,
                                        DeviceType device_type) {
  CacheUnionPtr union_ptr = gpu_union_;
  if (device_type == DeviceType::CPU) {
    union_ptr = cpu_union_;
  }

  bool is_cached = transverse_hash_table(hash, device_type);
  if (is_cached) {
    int cache_num = union_ptr->hash_table[hash]->cache_num;
    for (int cache_idx = 0; cache_idx < cache_num; cache_idx++) {
      union_ptr->hash_table[hash]->k_cache[cache_idx]->RefCacheSpan();
      union_ptr->hash_table[hash]->v_cache[cache_idx]->RefCacheSpan();
    }
  } else {
    ;
  }
}

void PrefixCacheManager::unref_cache_span(const std::string& hash,
                                          DeviceType device_type) {
  CacheUnionPtr union_ptr = gpu_union_;
  if (device_type == DeviceType::CPU) {
    union_ptr = cpu_union_;
  }

  bool is_cached = transverse_hash_table(hash, device_type);
  if (is_cached) {
    int cache_num = union_ptr->hash_table[hash]->cache_num;
    for (int cache_idx = 0; cache_idx < cache_num; cache_idx++) {
      union_ptr->hash_table[hash]->k_cache[cache_idx]->UnrefCacheSpan();
      union_ptr->hash_table[hash]->v_cache[cache_idx]->UnrefCacheSpan();
    }
  } else {
    ;
  }
}

void PrefixCacheManager::free_cache_span(const std::string& hash,
                                         DeviceType device_type) {
  CacheUnionPtr union_ptr = gpu_union_;
  if (device_type == DeviceType::CPU) {
    union_ptr = cpu_union_;
  }

  bool is_cached = transverse_hash_table(hash, device_type);
  if (is_cached) {
    int cache_num = union_ptr->hash_table[hash]->cache_num;
    for (int cache_idx = 0; cache_idx < cache_num; cache_idx++) {
      union_ptr->span_manager->ReleaseSpan(
          union_ptr->hash_table[hash]->k_cache[cache_idx]);
      union_ptr->span_manager->ReleaseSpan(
          union_ptr->hash_table[hash]->v_cache[cache_idx]);
    }
  } else {
    ;
  }
}

void PrefixCacheManager::wait_for_async_tasks() {
  size_t running_tasks = threadpool_->hasTasksRunning();
  if (running_tasks > 0)
    LOG(INFO) << "[PrefixCacheManager] Waiting for all tasks in the thread "
                 "pool to finish executing...";

  while (running_tasks > 0) {
    usleep(1000);
    running_tasks = threadpool_->hasTasksRunning();
    LOG(INFO) << "[PrefixCacheManager] remaining tasks: " << running_tasks;
  }
  return;
}

std::string PrefixCacheManager::to_string(DeviceType device_type) {
  CacheUnionPtr union_ptr = gpu_union_;
  if (device_type == DeviceType::CPU) {
    union_ptr = cpu_union_;
  }

  std::stringstream ss;
  ss << "total cached prefix number: " << union_ptr->hash_table.size() << "\n";
  for (auto it = union_ptr->hash_table.begin();
       it != union_ptr->hash_table.end(); ++it) {
    ss << "hash: " << it->first << ", "
       << "ref_cnt: " << it->second->ref_cnt << ", "
       << "prefix_len: " << it->second->prefix_len << "\n";
  }
  return ss.str();
}

}  // namespace allspark
#endif
