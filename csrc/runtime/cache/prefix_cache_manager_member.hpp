/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    prefix_cache_manager_member.hpp
 */

#pragma once

#include <chrono>
#include <fstream>
#include <queue>
#include <string>
#include <unordered_map>

#include "common/common.h"
#include "cuda/cuda_context.h"
#include "env_config.h"
#include "prefix_cache_manager.h"
#include "span_manager.h"
#include "utility/check_cuda.h"
#include "virtual_cache.h"

namespace allspark {
#if ENABLE_SPAN_ATTENTION

/***************************************
 * privte struct, class definitions
 ***************************************/
class PrefixCacheManager::PrefixNode {
 public:
  std::vector<CacheSpan::Ptr> k_cache;
  std::vector<CacheSpan::Ptr> v_cache;
  int ref_cnt;
  int prefix_len;
  int cache_num;
  NodeTimestamp last_access_time;
  DeviceType device_type;
  std::string hash;
  bool is_ready;

 public:
  PrefixNode(std::vector<CacheSpan::Ptr>& k_cache_in,
             std::vector<CacheSpan::Ptr>& v_cache_in, int prefix_len,
             NodeTimestamp ts, DeviceType device_type,
             std::string hash_str = DEFAULT_HASH_STR)
      : ref_cnt(0),
        prefix_len(prefix_len),
        last_access_time(ts),
        device_type(device_type),
        is_ready(false),
        hash(hash_str) {
    k_cache = std::move(k_cache_in);
    v_cache = std::move(v_cache_in);
    cache_num = k_cache.size();
    if (k_cache.size() != v_cache.size()) {
      LOG(ERROR) << __FUNCTION__ << ": kv cache size not equal";
      AS_THROW(AsStatus::ALLSPARK_RUNTIME_ERROR);
    }
  }

  // smaller nodes are more likely to be evicted
  bool operator<(const PrefixNode& other) const {
    if (last_access_time != other.last_access_time) {
      return last_access_time < other.last_access_time;
    }
    if (prefix_len != other.prefix_len) {
      return prefix_len > other.prefix_len;
    }
    return hash < other.hash;
  }

#if ENABLE_PREFIX_CACHE_DEBUG_API
  static void DeepCopy(std::shared_ptr<PrefixNode>& src_node,
                       std::shared_ptr<PrefixNode>& dst_node) {
    if (src_node == nullptr || dst_node == nullptr) {
      LOG(ERROR) << __FUNCTION__ << ": src or dst node is nullptr";
      AS_THROW(AsStatus::ALLSPARK_RUNTIME_ERROR);
    }
    if (src_node->cache_num != dst_node->cache_num) {
      LOG(ERROR) << __FUNCTION__ << ": cache_num not equal";
      AS_THROW(AsStatus::ALLSPARK_RUNTIME_ERROR);
    }

    dst_node->ref_cnt = src_node->ref_cnt;
    dst_node->prefix_len = src_node->prefix_len;
    dst_node->last_access_time = src_node->last_access_time;
    dst_node->hash = src_node->hash;

    int cache_num = src_node->cache_num;
    for (int i = 0; i < cache_num; i++) {
      void* src_k_ptr = src_node->k_cache[i]->Frame()->Data();
      void* dst_k_ptr = dst_node->k_cache[i]->Frame()->Data();
      void* src_v_ptr = src_node->v_cache[i]->Frame()->Data();
      void* dst_v_ptr = dst_node->v_cache[i]->Frame()->Data();

      int64_t nbytes = dst_node->k_cache[i]->Frame()->Size();

      auto cpyKind =
          GetCudaMemcpyKind(src_node->device_type, dst_node->device_type);
      AS_CHECK_CUDA(cudaMemcpy(dst_k_ptr, src_k_ptr, nbytes, cpyKind));
      AS_CHECK_CUDA(cudaMemcpy(dst_v_ptr, src_v_ptr, nbytes, cpyKind));
    }
  }

  static void DeepCopyAysnc(std::shared_ptr<PrefixNode>& src_node,
                            std::shared_ptr<PrefixNode>& dst_node,
                            cudaStream_t cu_stream) {
    if (src_node == nullptr || dst_node == nullptr) {
      LOG(ERROR) << __FUNCTION__ << ": src or dst node is nullptr";
      AS_THROW(AsStatus::ALLSPARK_RUNTIME_ERROR);
    }
    if (src_node->cache_num != dst_node->cache_num) {
      LOG(ERROR) << __FUNCTION__ << ": cache_num not equal";
      AS_THROW(AsStatus::ALLSPARK_RUNTIME_ERROR);
    }

    dst_node->ref_cnt = src_node->ref_cnt;
    dst_node->prefix_len = src_node->prefix_len;
    dst_node->last_access_time = src_node->last_access_time;
    dst_node->hash = src_node->hash;

    int cache_num = src_node->cache_num;
    for (int i = 0; i < cache_num; i++) {
      void* src_k_ptr = src_node->k_cache[i]->Frame()->Data();
      void* dst_k_ptr = dst_node->k_cache[i]->Frame()->Data();
      void* src_v_ptr = src_node->v_cache[i]->Frame()->Data();
      void* dst_v_ptr = dst_node->v_cache[i]->Frame()->Data();

      int64_t nbytes = dst_node->k_cache[i]->Frame()->Size();

      auto cpyKind =
          GetCudaMemcpyKind(src_node->device_type, dst_node->device_type);
      AS_CHECK_CUDA(
          cudaMemcpyAsync(dst_k_ptr, src_k_ptr, nbytes, cpyKind, cu_stream));
      AS_CHECK_CUDA(
          cudaMemcpyAsync(dst_v_ptr, src_v_ptr, nbytes, cpyKind, cu_stream));
    }
  }

  static void PrintData(std::shared_ptr<PrefixNode>& node) {
    if (node == nullptr) {
      return;
    }

    if (node->device_type != DeviceType::CPU) {
      LOG(ERROR) << __FUNCTION__ << ": only support comparing cpu nodes";
      AS_THROW(AsStatus::ALLSPARK_RUNTIME_ERROR);
    }

    int span_size = node->k_cache[0]->Size();
    // for (int i = 0; i < node->cache_num; i++) {
    // for (int i = node->cache_num - 1; i < node->cache_num; i++) {
    for (int i = 0; i < 1; i++) {
      void* node_k_cache_ptr = node->k_cache[i]->Data();
      void* node_v_cache_ptr = node->v_cache[i]->Data();
      std::stringstream ss;
      int print_num = 10;
      uint16_t* data1 = static_cast<uint16_t*>(node_k_cache_ptr);
      ss << "hash_str: " << node->hash;
      ss << "\n";
      ss << "first " << print_num << " data in k_cache: ";
      for (int idx = 0; idx < print_num; idx++) {
        ss << data1[idx] << " ";
      }
      uint16_t* data2 = static_cast<uint16_t*>(node_v_cache_ptr);
      ss << "\n";
      ss << "first " << print_num << " data in v_cache: ";
      for (int idx = 0; idx < print_num; idx++) {
        ss << data2[idx] << " ";
      }
      LOG(INFO) << ss.str();
    }
  }

  static bool isSame(std::shared_ptr<PrefixNode>& node1,
                     std::shared_ptr<PrefixNode>& node2, int idx = -1) {
    if (node1 == nullptr || node2 == nullptr) {
      LOG(INFO) << __FUNCTION__ << ": node1 or node2 is nullptr";
      return false;
    }

    if (node1->device_type != node2->device_type) return false;

    if (node1->device_type != DeviceType::CPU) {
      LOG(ERROR) << __FUNCTION__ << ": only support comparing cpu nodes";
      AS_THROW(AsStatus::ALLSPARK_RUNTIME_ERROR);
    }

    if (node1->cache_num != node2->cache_num) return false;

    bool rt = true;
    int span_size = node1->k_cache[0]->Size();
    for (int i = 0; i < node1->cache_num; i++) {
      void* node1_k_cache_ptr = node1->k_cache[i]->Data();
      void* node1_v_cache_ptr = node1->v_cache[i]->Data();
      void* node2_k_cache_ptr = node2->k_cache[i]->Data();
      void* node2_v_cache_ptr = node2->v_cache[i]->Data();

      bool is_same = true;
      is_same &= (memcmp(node1_k_cache_ptr, node2_k_cache_ptr, span_size) == 0);
      is_same &= (memcmp(node1_v_cache_ptr, node2_v_cache_ptr, span_size) == 0);
      if (is_same != true) {
        std::stringstream ss;
        ss << __FUNCTION__ << ": two nodes have different data";
        if (idx >= 0) {
          ss << ", node no. " << idx;
        }
        ss << ", layer no. " << i;
        LOG(INFO) << ss.str();
        rt = false;
        break;
      }
    }

    if (rt == true) {
      LOG(INFO) << __FUNCTION__ << ": two nodes have the same data";
    }
    return rt;
  }
#endif
};

class PrefixCacheManager::LRUEvictor {
 private:
  struct NodeComparator {
    bool operator()(const PrefixNodePtr& lhs, const PrefixNodePtr& rhs) const {
      return (*lhs) < (*rhs);
    }
  };

  std::set<PrefixNodePtr, NodeComparator> candidates_set_;

 public:
  std::string FindVictim() {
    std::string victim_hash = DEFAULT_HASH_STR;

    if (!candidates_set_.empty()) {
      const PrefixNodePtr& victim_node = *candidates_set_.begin();
      victim_hash = victim_node->hash;
    }

    return victim_hash;
  }

  std::vector<std::string> FindMultiVictims(int k) {
    k = std::min(static_cast<int>(candidates_set_.size()), k);

    std::vector<std::string> victim_hash_list;
    victim_hash_list.reserve(k);
    auto itEnd = std::next(candidates_set_.begin(), k);
    std::transform(
        candidates_set_.begin(), itEnd, std::back_inserter(victim_hash_list),
        [](const PrefixNodePtr& node) -> std::string { return node->hash; });
    return std::move(victim_hash_list);
  }

  std::vector<std::string> GetAllCandidatesHash() {
    int size = candidates_set_.size();
    return std::move(FindMultiVictims(size));
  }

  void Add(PrefixNodePtr& kv_cache) { candidates_set_.insert(kv_cache); }
  void Del(PrefixNodePtr& kv_cache) { candidates_set_.erase(kv_cache); }
  void Reset() { candidates_set_.clear(); }

  int Size() { return candidates_set_.size(); }
  std::string ToString() {
    std::stringstream ss;
    ss << "total candidate number: " << candidates_set_.size() << "\n";
    for (auto it = candidates_set_.begin(); it != candidates_set_.end(); ++it) {
      ss << "hash: " << (*it)->hash << ", "
         << "ref_cnt: " << (*it)->ref_cnt << ", "
         << "prefix_len: " << (*it)->prefix_len << "\n";
    }
    return ss.str();
  }
};

class PrefixCacheManager::CacheUnion {
 public:
  CacheUnion(DeviceType device_type, int capacity,
             const CacheSpanManager::Ptr& span_manager,
             const CacheFrameManager::Ptr& frame_manager)
      : device_type(device_type),
        capacity(capacity),
        span_manager(span_manager),
        frame_manager(frame_manager) {}
  CacheUnion() = delete;
  const DeviceType device_type;
  int capacity;
  LRUEvictor evictor;
  CacheSpanManager::Ptr span_manager;
  CacheFrameManager::Ptr frame_manager;
  std::unordered_map<std::string, PrefixNodePtr> hash_table;
};

class PrefixCacheManager::MemcpyWorkspace {
 public:
  MemcpyWorkspace(int max_length, int max_batch, int token_per_span,
                  int frame_per_node, int frame_size,
                  DeviceType device_type = DeviceType::CPU)
      : device_type(device_type) {
    block_num = max_batch;
    block_size = CalcWorkspaceSize(max_length, max_batch, token_per_span,
                                   frame_per_node, frame_size);
    workspace_size = block_size * block_num;
    cudaMallocHost((void**)&workspace_ptr, workspace_size);
    for (int i = 0; i < block_num; i++) {
      available_blocks.push(workspace_ptr + i * block_size);
    }
  }
  MemcpyWorkspace() = delete;

  ~MemcpyWorkspace() { cudaFreeHost(workspace_ptr); }

  using WsBlock = std::pair<int, void*>;

  static size_t CalcWorkspaceSize(int max_length, int max_batch,
                                  int token_per_span, int frame_per_node,
                                  int frame_size) {
    size_t ws_block_size =
        static_cast<size_t>((max_length + token_per_span - 1) /
                            token_per_span) *
        max_batch * frame_per_node * frame_size;
    return ws_block_size;
  }

  void* GetWorkspacePtr() { return workspace_ptr; }

  size_t GetWorkspaceSize() { return workspace_size; }

  size_t GetBlockSize() { return block_size; }

  std::shared_ptr<void> QueryBlock() {
    void* block = pop();

    while (block == nullptr) {
      usleep(100);
      block = pop();
    }

    return std::shared_ptr<void>(block, [this](void* p) {
      std::lock_guard<std::mutex> guard(mtx);
      available_blocks.push(p);
    });
  }

 private:
  void* pop() {
    std::lock_guard<std::mutex> guard(mtx);
    void* block = nullptr;
    if (!available_blocks.empty()) {
      block = available_blocks.front();
      available_blocks.pop();
    }
    return block;
  }

 private:
  const DeviceType device_type;
  int block_num;
  size_t block_size;
  size_t workspace_size;
  void* workspace_ptr;
  std::queue<void*> available_blocks;
  std::mutex mtx;
};

class PrefixCacheManager::Profiler {
 public:
  Profiler(std::string info = "",
           std::unordered_map<std::string, float>* table = nullptr)
      : info(info),
        timecost_table(table),
        start_time_point(std::chrono::steady_clock::now()) {}

  ~Profiler() {
    auto end_time_point = std::chrono::steady_clock::now();
    auto duration = end_time_point - start_time_point;
    auto duration_in_ms =
        std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count() /
        1000000.0;

    if (timecost_table != nullptr) {
      if (timecost_table->count(info) > 0) {
        (*timecost_table)[info] += duration_in_ms;
      } else {
        (*timecost_table)[info] = duration_in_ms;
      }
    }

    if (EnvVarConfig::GetInt("ALLSPARK_TIME_LOG", 0) == 1) {
      LOG(INFO) << "[PrefixCacheManager] " << info << ", "
                << "timecost: " << duration_in_ms << " ms";
    }
  }

 private:
  const std::string info;
  const std::chrono::time_point<std::chrono::steady_clock> start_time_point;
  std::unordered_map<std::string, float>* timecost_table;
};
#endif
}  // namespace allspark
