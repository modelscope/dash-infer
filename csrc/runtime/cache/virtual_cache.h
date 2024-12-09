/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    virtual_cache.h
 */

#pragma once

#if ENABLE_SPAN_ATTENTION

#include <spanattn/span_attn.h>

#include <memory>

#include "common/common.h"
#include "common/device_context.h"
#include "common/engine_runtime.h"
#include "core/kernel/cuda/cuda_common.h"
#include "core/tensor/tensor.h"
#include "span_manager.h"

namespace allspark {

class SpannedVirtualCache;

namespace cache {
class CacheArray {
 public:
  explicit CacheArray(SpannedVirtualCache* manager, int layer_id,
                      int max_num_spans);
  ~CacheArray();

  /**
   * @brief Init cache spans for context, call this immediately after
   * constructed.
   *
   * This is extracted from the constructor, because we want to differ the
   * runtime error caused by insufficient memory from other types of
   * errors.
   *
   * This function will return false if there is no enough memory to make
   * the cache for the context, which may result in interrupted request
   * later.
   * @param context_length Number of tokens in the context.
   *
   * @return true if success.
   * @return false if failed to allocate needed spans due to insufficient
   * memory.
   */
  bool InitContext(int context_length);

  /**
   * @brief Get the tensor of cache pointers, increase the sequence length if
   * necessary.
   *
   * @param increment A non-negative integer, the increment of the
   * sequence length.
   *
   * @return const reference to tensor of cache pointers.
   */
  const AsTensor& GetCachePtrVec(int increment);

  const CacheSpan::Ptr GetCacheSpan(int index) const;

  void SetSeqLength(size_t len) { seq_length_ = len; }

  size_t GetSeqLength() const { return seq_length_; }

  size_t GetSpanNum() const { return cache_.size(); }

  std::unique_ptr<AsTensor>& GetTLB() { return tlb_; }
  std::vector<CacheSpan::Ptr>& GetCacheVector() { return cache_; }

 private:
  CacheSpan::Ptr newSpan(size_t index) const;
  size_t newSpan(std::vector<CacheSpan::Ptr>& out_vec, size_t start_index,
                 size_t count) const;
  void destroySpan(CacheSpan::Ptr span) const;
  void destroySpan(std::vector<CacheSpan::Ptr>& span_vec, size_t count) const;
  bool updateSeqLength(size_t new_seq_length);

  const std::string tag_;
  const SpanCacheConfig::ConstPtr cache_config_;
  const int layer_id_;

  std::vector<CacheSpan::Ptr> cache_;
  std::unique_ptr<AsTensor> tlb_;
  CacheSpanManager::Ptr span_manager_;
  size_t seq_length_;
};

}  // namespace cache

class VirtualCache {
 public:
  using CacheArray = cache::CacheArray;
  virtual ~VirtualCache() = default;

  /**
   * @brief Init the cache for a layer, call in Reshape.
   *
   * Only support up to INT_MAX layers.
   *
   * @param layer_id
   * @param num_heads
   * @param per_head_size
   * @param context_len Number of tokens in the context.
   * @param max_num_spans Max number of spans in the cache.
   *
   * @return AsStatus::ALLSPARK_PARAM_ERROR if any param is invalid.
   * @return AsStatus::ALLSPARK_MEMORY_ERROR if insufficient memory.
   * @return AsStatus::ALLSPARK_SUCCESS if success.
   */
  virtual AsStatus InitLayer(int layer_id, int num_heads, int per_head_size,
                             int context_len, int max_num_spans) = 0;

  /**
   * @brief Get the tensor of cache pointers, increase the sequence length if
   * necessary.
   *
   * @param increment A non-negative integer, the increment of the
   * sequence length.
   *
   * @return const reference to tensor of cache pointers.
   *
   * @throw AsException ALLSPARK_PARAM_ERROR if layer_id out of range.
   * @throw AsException ALLSPARK_PARAM_ERROR if increment is negative.
   * @throw AsException ALLSPARK_CACHE_MEMORY_OUT if failed to allocate span.
   */
  virtual const AsTensor& GetCache(int layer_id, int increment) = 0;

  /**
   * @brief Return the cached sequence length of the specified layer.
   */
  virtual size_t GetSeqLength(int layer_id) const = 0;

  virtual int GetLayerNum() const = 0;
  virtual void FillCache(std::vector<CacheSpan::Ptr>& cache) = 0;
  virtual std::vector<std::unique_ptr<CacheArray>>& GetLayerCache() = 0;
};

class CacheUtils {
  CacheUtils() = delete;

 public:
  static size_t GetSpanSizeInBytes(const SpanCacheConfig& cache_config,
                                   DataType data_type, int num_heads,
                                   int per_head_size);
  static span::QuantMode toQuantMode(AsCacheMode cache_mode);
};

class SpannedVirtualCache : public VirtualCache {
 public:
  explicit SpannedVirtualCache(CacheSpanManager::Ptr span_manager,
                               SpanCacheConfig::Ptr cache_config,
                               const std::string& tag, int layer_num)
      : tag_(tag),
        cache_config_(cache_config),
        layer_num_(layer_num),
        layer_cache_(layer_num),
        span_manager_(span_manager) {}

  AsStatus InitLayer(int layer_id, int num_heads, int per_head_size,
                     int context_len, int max_num_spans) override;

  const AsTensor& GetCache(int layer_id, int increment) override;

  size_t GetSeqLength(int layer_id) const override;

  int GetLayerNum() const override { return layer_num_; }

  void FillCache(std::vector<CacheSpan::Ptr>& cache) override;

  CacheSpanManager::Ptr GetSpanManager() const { return span_manager_; }
  std::vector<std::unique_ptr<CacheArray>>& GetLayerCache() {
    return layer_cache_;
  }
  SpanCacheConfig::ConstPtr GetCacheConfig() const { return cache_config_; }
  std::string GetTag() const { return tag_; }

 private:
  const std::string tag_;
  const SpanCacheConfig::ConstPtr cache_config_;
  const int layer_num_;

  std::vector<std::unique_ptr<CacheArray>> layer_cache_;
  CacheSpanManager::Ptr span_manager_;
};

}  // namespace allspark

#endif  // ENABLE_SPAN_ATTENTION
