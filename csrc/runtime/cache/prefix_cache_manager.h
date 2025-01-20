/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    prefix_cache_manager.h
 */

#pragma once

#include <chrono>
#include <condition_variable>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "MurmurHash3.h"
#include "common/common.h"
#include "common/device_context.h"
#include "common/thread_pool.h"
#include "core/tensor/tensor.h"
#include "span_manager.h"
#include "virtual_cache.h"

namespace allspark {
#if ENABLE_SPAN_ATTENTION
#ifndef DEFAULT_HASH_STR
#define DEFAULT_HASH_STR "not_hashed"
#endif

#define ENABLE_PREFIX_CACHE_DEBUG_API 0

typedef std::chrono::time_point<std::chrono::steady_clock> NodeTimestamp;

class PrefixCacheCoordinator {
 public:
  using Ptr = std::shared_ptr<PrefixCacheCoordinator>;
  using ConstPtr = std::shared_ptr<const PrefixCacheCoordinator>;

  PrefixCacheCoordinator() = delete;
  PrefixCacheCoordinator(const int nranks) : worker_num(nranks) {
    cpu_available_mem = get_cpu_available_mem();
  }

  void waitForOtherWorkers() {
    std::unique_lock<std::mutex> lock(mtx);
    ++worker_count;
    if (worker_count == worker_num) {
      worker_count = 0;      // Reset for next use
      wait_cv.notify_all();  // Notify all threads to proceed
    } else {
      wait_cv.wait(lock, [this] { return worker_count == 0; });
    }
  }

  size_t getCpuAvailMem() { return cpu_available_mem; }

 private:
  size_t get_cpu_available_mem() {
    std::ifstream meminfo("/proc/meminfo");
    std::string line;
    size_t availableMemory = 0;

    if (meminfo.is_open()) {
      while (std::getline(meminfo, line)) {
        if (line.find("MemAvailable:") == 0) {
          std::sscanf(line.c_str(), "MemAvailable: %zu kB", &availableMemory);
          break;
        }
      }
      meminfo.close();
    } else {
      LOG(ERROR) << "Unable to open /proc/meminfo";
    }

    return availableMemory;
  }

 private:
  std::mutex mtx;
  std::condition_variable wait_cv;
  int worker_count = 0;
  const int worker_num;
  size_t cpu_available_mem;  // kB
};

class PrefixCacheManager {
 public:
  using Ptr = std::shared_ptr<PrefixCacheManager>;
  using ConstPtr = std::shared_ptr<const PrefixCacheManager>;
  using CacheArray = cache::CacheArray;

  /***************************************
   * privte definitions
   ***************************************/
 private:
  class PrefixNode;
  class LRUEvictor;
  class CacheUnion;
  class MemcpyWorkspace;
  class Profiler;
  using CacheUnionPtr = std::shared_ptr<CacheUnion>;

 public:
  using PrefixNodePtr = std::shared_ptr<PrefixNode>;

 public:
  PrefixCacheManager() = delete;
  PrefixCacheManager(const CacheSpanManager::Ptr& span_manager,
                     const CacheFrameManager::Ptr& frame_manager,
                     const PrefixCacheCoordinator::Ptr& coordinator,
                     const TensorMap* tensor_map, const DeviceContext* ctx,
                     const int ttl);
  ~PrefixCacheManager() { PrintTimecost(); }

  void UpdateCapacity();
  void SetSeqlenThre(int thre);

  void Insert(const std::shared_ptr<AsTensor>& tokens, const int start_idx,
              const NodeTimestamp ts,
              const std::vector<std::unique_ptr<CacheArray>>& layer_cache_k,
              const std::vector<std::unique_ptr<CacheArray>>& layer_cache_v,
              std::vector<PrefixNodePtr>& node_vec);

  void RefOnly(const std::shared_ptr<AsTensor>& tokens, const NodeTimestamp ts,
               int& prefix_len, int& gpu_cached_len,
               std::vector<PrefixNodePtr>& node_vec);

  void RefFill(const std::shared_ptr<AsTensor>& tokens,
               const std::shared_ptr<AsTensor>& tokens_for_hash,
               std::shared_ptr<AsTensor>& new_tokens, const NodeTimestamp ts,
               int& prefix_len, std::unique_ptr<VirtualCache>& virtual_k_cache,
               std::unique_ptr<VirtualCache>& virtual_v_cache,
               std::vector<PrefixNodePtr>& node_vec);

  void UnRef(const std::vector<PrefixNodePtr>& node_vec);

  void UpdateCnt(int hit_cnt, int miss_cnt);
  void Reset();
  void EvictUnrefered(int target);
  void EvictAllUnrefered();
  void UpdateEngineStat(AsEngineStat* as_stat);
  // float GetFreeUnrefThre() { return free_unref_thre_; }

  /***************************************
   * public functions for debug
   ***************************************/
 public:
  void PrintPrefixCacheInfo(std::string extra_info);
  void PrintTimecost();
#if ENABLE_PREFIX_CACHE_DEBUG_API
  void PrintAllPrefixCache(DeviceType device_type);
  void CheckHash(const std::shared_ptr<AsTensor>& tokens,
                 std::vector<std::string>& hash_vec, std::string request_id);
  void SwapByNodeList(std::vector<PrefixNodePtr>& node_list);
  void SwapByHashList(std::vector<std::string>& hash_vec);
  void RemoveByHashList(std::vector<std::string>& hash_vec);
#endif

  /***************************************
   * privte functions
   ***************************************/
 private:
  void init_cpu_union(const int nranks, const int engine_max_length,
                      const int engine_max_batch);

  std::string hash_tokens(void* data, int len);
  bool transverse_hash_table(const std::string& hash, DeviceType device_type);

  void ref_node(const std::string& hash, const NodeTimestamp& ts,
                DeviceType device_type);
  void unref_node(const std::string& hash, DeviceType device_type);
  bool insert_node(const std::string& hash, PrefixNodePtr node,
                   DeviceType device_type);
  void delete_node(const std::string& hash, DeviceType device_type);
  void delete_multinodes(const std::vector<std::string>& hash_vec,
                         DeviceType device_type);

  void create_node(PrefixNodePtr& new_node,
                   std::vector<CacheSpan::Ptr>& k_cache_in,
                   std::vector<CacheSpan::Ptr>& v_cache_in, int prefix_len,
                   NodeTimestamp ts, DeviceType device_type, std::string hash);

  void set_node_ready(PrefixNodePtr node);
  bool check_node_ready(PrefixNodePtr& node);
  void filter_timeout_hash(std::vector<std::string>& hash_list);

  void update_node_ts(const std::string& hash, const NodeTimestamp& ts,
                      DeviceType device_type);

  bool alloc_cpu_node(PrefixNodePtr& new_node, const int cache_num,
                      const int ref_cnt, const int prefix_len,
                      const NodeTimestamp ts, std::string hash);

  bool alloc_gpu_node(PrefixNodePtr& new_node, const int cache_num,
                      const int ref_cnt, const int prefix_len,
                      const NodeTimestamp ts,
                      std::unique_ptr<VirtualCache>& virtual_k_cache,
                      std::unique_ptr<VirtualCache>& virtual_v_cache,
                      std::string hash);

  void swap_to_cpu_by_nodelist(std::vector<PrefixNodePtr>& src_node_list);
  void swap_to_cpu_by_hashlist(std::vector<std::string>& hash_list);
  void swap_nodelist_to_cpu_impl(std::vector<PrefixNodePtr>& gpu_node_list,
                                 std::vector<PrefixNodePtr>& new_cpu_node_list);
  void swap_to_gpu_by_nodelist(std::vector<PrefixNodePtr>& cpu_node_list,
                               std::vector<PrefixNodePtr>& new_gpu_node_list,
                               std::unique_ptr<VirtualCache>& virtual_k_cache,
                               std::unique_ptr<VirtualCache>& virtual_v_cache);
  void swap_nodelist_to_gpu_impl(std::vector<PrefixNodePtr>& cpu_node_list,
                                 std::vector<PrefixNodePtr>& new_gpu_node_list);

  void evict_unrefered_by_num(int node_num);

  void ref_cache_span(const std::string& hash, DeviceType device_type);
  void unref_cache_span(const std::string& hash, DeviceType device_type);
  void free_cache_span(const std::string& hash, DeviceType device_type);

  void wait_for_async_tasks();

  std::string to_string(DeviceType device_type);

#if ENABLE_PREFIX_CACHE_DEBUG_API
  bool copy_node(PrefixNodePtr& src_node, PrefixNodePtr& dst_node);
  void print_node(PrefixNodePtr& node);
  bool compare_node(PrefixNodePtr& node1, PrefixNodePtr& node2, int idx = -1);
  PrefixNodePtr swap_node_to_cpu(PrefixNodePtr& src_node);
  bool swap_node_to_cpu(PrefixNodePtr& src_node, const std::string& hash);
  bool swap_node_to_gpu(PrefixNodePtr& src_node, PrefixNodePtr& dst_node,
                        const std::string& hash);
#endif

  /***************************************
   * privte variables
   ***************************************/
 private:
  PrefixCacheCoordinator::Ptr coordinator_;
  CacheUnionPtr gpu_union_;
  CacheUnionPtr cpu_union_;
  DataType data_type_;
  span::QuantMode quant_mode_;
  const TensorMap* tensor_map_;
  std::shared_ptr<MemcpyWorkspace> cpu_workspace_;
  std::shared_ptr<AsTensor> spanptr_tensor_device_;
  cudaStream_t stream_;
  const DeviceContext* ctx_;
  std::unordered_map<std::string, float> timecost_table_;

  int token_per_span_;
  int layer_num_;
  // int min_capacity_;
  int64_t hit_cnt_ = 0;
  int64_t miss_cnt_ = 0;

  std::unique_ptr<ThreadPool> threadpool_;
  int threadpool_size_ = 32;

  float time_to_live_sec_;

  // use prefix cachen only when matched seq_len > thre
  int seq_len_thre_ = 0;

  // how many spans can be used for prefix caching
  float prefix_cache_ratio_ = 1.0;

  // if available frame percentages are less then the threshold,
  // free all unreferenced frames
  // float free_unref_thre_ = 0.1;
};
#else
class PrefixCacheCoordinator {
 public:
  using Ptr = std::nullptr_t;
};
#endif
}  // namespace allspark
