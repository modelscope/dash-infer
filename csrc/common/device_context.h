/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    device_context.h
 */

#pragma once

#include <memory>

#include "common/block.h"
#include "common/common.h"
#include "common/engine_runtime.h"

namespace allspark {

// base device context interface
class DeviceContext {
  DeviceContext(const DeviceContext&) = delete;
  DeviceContext(DeviceContext&&) = delete;
  DeviceContext& operator=(const DeviceContext&) = delete;
  DeviceContext& operator=(DeviceContext&&) = delete;

 public:
  DeviceContext() = default;
  virtual ~DeviceContext() {}
  virtual void Init() = 0;
  virtual DeviceType GetDeviceType() const = 0;
  virtual int GetRank() const = 0;
  virtual int GetNranks() const = 0;
  virtual void SetNumThreads(int num_threads) = 0;
  virtual void SetDeviceId(int device_id) = 0;
  virtual int GetDeviceId() = 0;
  virtual void Synchronize() const = 0;
  virtual Block::Ptr AllocBlock(int64_t nbytes) = 0;
  virtual void FreeBlock(const Block::Ptr& block) = 0;
  virtual void ResetBlockPools() = 0;
  virtual void SemPostInterProcess(){};
  virtual void SemWaitSendInterProcess(){};
  virtual bool SemWaitMsgSynInterProcess(int msg_size) { return false; };
  virtual void SetSparsityMatmulMode(bool enable_sparsity_matmul){};
  virtual bool GetSparsityMatmulMode() const { return false; };
  void SetKVcacheSize(int kvsize) { engine_kvcache_alloc_size = kvsize; }
  int GetKVcacheSize() const { return engine_kvcache_alloc_size; }
  void SetModelMaxLength(int max_length) { engine_max_length = max_length; }
  int GetModelMaxLength() const { return engine_max_length; }
  void SetModelMaxBatch(int max_batch) { engine_max_batch = max_batch; }
  int GetModelMaxBatch() const { return engine_max_batch; }
  void SetModelMaxPrefillLength(int max_length) {
    engine_max_prefill_length = max_length;
  }
  int GetModelMaxPrefillLength() const { return engine_max_prefill_length; }
  virtual void SetDtype(DataType new_dtype) { dtype = new_dtype; }
  DataType GetDtype() const { return dtype; }
  void SetDatatype(std::string str_dtype) {
    if (str_dtype == "float16") {
      dtype = DataType::FLOAT16;
    } else if (str_dtype == "float32") {
      dtype = DataType::FLOAT32;
    } else if (str_dtype == "bfloat16") {
      dtype = DataType::BFLOAT16;
    } else {
      LOG(ERROR) << "DeviceContext: unsupported data type: " << str_dtype;
      AS_THROW(AsStatus::ALLSPARK_PARAM_ERROR);
    }
  }
  void SetMatmulPrecision(int precision) { precision_ = precision; }
  int GetMatmulPrecision() const { return precision_; }
  void SetNumberHeads(int num_heads) { num_heads_ = num_heads; }
  void SetDecoderLayer(int dec_layer) { dec_layer_ = dec_layer; }
  void SetSizePerHead(int size_per_head) { size_per_head_ = size_per_head; }
  void SetNumberGroups(int num_groups) { num_groups_ = num_groups; }
  void SetIntermediateSize(int intermediate_size) {
    intermediate_size_ = intermediate_size;
  }
  void SetUseTorchSample(bool use_torch_sample) {
    use_torch_sample_ = use_torch_sample;
  }
  void SetPrefillMode(AsMHAPrefill prefill_mode) {
    prefill_mode_ = prefill_mode;
  }
  void SetEvictionStrategy(AsEvictionStrategy eviction_strategy) {
    eviction_strategy_ = eviction_strategy;
  }
  void SetSchedulingStrategy(AsSchedulingStrategy scheduling_strategy) {
    scheduling_strategy_ = scheduling_strategy;
  }
  void SetFallbackDecoderWeightOnly(bool decoder_weight_only) {
    decoder_weight_only_ = decoder_weight_only;
  }
#if ENABLE_SPAN_ATTENTION
  void SetCacheConfig(SpanCacheConfig::Ptr cache_config_) {
    cache_config = cache_config_;
  }
  SpanCacheConfig::Ptr GetCacheConfig() const {
    if (!cache_config) {
      LOG(ERROR) << "DeviceContext: cache config uninitialized";
      AS_THROW(AsStatus::ALLSPARK_RUNTIME_ERROR)
    }
    return cache_config;
  }
  AsCacheMode GetCacheMode() const {
    if (!cache_config) {
      LOG(ERROR) << "DeviceContext: cache config uninitialized";
      AS_THROW(AsStatus::ALLSPARK_RUNTIME_ERROR)
    }
    return cache_config->mode;
  }
  int GetCacheSpanSize() const {
    if (!cache_config) {
      LOG(ERROR) << "DeviceContext: cache config uninitialized";
      AS_THROW(AsStatus::ALLSPARK_RUNTIME_ERROR)
    }
    return cache_config->span_size;
  }
  int GetCacheSpanNumInit() const {
    if (!cache_config) {
      LOG(ERROR) << "DeviceContext: cache config uninitialized";
      AS_THROW(AsStatus::ALLSPARK_RUNTIME_ERROR)
    }
    return cache_config->span_num_init;
  }
  int GetCacheSpanNumGrow() const {
    if (!cache_config) {
      LOG(ERROR) << "DeviceContext: cache config uninitialized";
      AS_THROW(AsStatus::ALLSPARK_RUNTIME_ERROR)
    }
    return cache_config->span_num_grow;
  }
#endif
  int GetNumberHeads() const { return num_heads_; }
  int GetDecoderLayer() const { return dec_layer_; }
  int GetSizePerHead() const { return size_per_head_; }
  int GetNumberGroups() const { return num_groups_; }
  int GetIntermediateSize() const { return intermediate_size_; }
  int GetMaxTopLogprobs() const { return engine_max_top_logprobs; }
  bool GetUseTorchSample() const { return use_torch_sample_; }
  AsMHAPrefill GetPrefillMode() const { return prefill_mode_; }
  AsEvictionStrategy GetEvictionStrategy() const { return eviction_strategy_; }
  AsSchedulingStrategy GetSchedulingStrategy() const {
    return scheduling_strategy_;
  }
  int GetLoraMaxNum() const { return lora_max_num_; }
  void SetLoraMaxNum(int lora_max_num) { lora_max_num_ = lora_max_num; }
  int GetLoraMaxRank() const { return lora_max_rank_; }
  void SetLoraMaxRank(int lora_max_rank) { lora_max_rank_ = lora_max_rank; }
  bool GetLoraEnabled() const { return lora_enabled_; }
  void SetLoraEnabled(bool enabled) { lora_enabled_ = enabled; }
  // decoder fallback weight only which is used in A8W8
  bool GetFallbackDecoderWeightOnly() const { return decoder_weight_only_; }
  void CopyFromOther(const DeviceContext* other_ctx) {
#if ENABLE_SPAN_ATTENTION
    if (GetDeviceType() == DeviceType::CUDA) {
      SetCacheConfig(other_ctx->GetCacheConfig());
    }
#endif
    SetNumberHeads(other_ctx->GetNumberHeads());
    SetDecoderLayer(other_ctx->GetDecoderLayer());
    SetSizePerHead(other_ctx->GetSizePerHead());
    SetNumberGroups(other_ctx->GetNumberGroups());
    SetIntermediateSize(other_ctx->GetIntermediateSize());
    SetUseTorchSample(other_ctx->GetUseTorchSample());
    SetKVcacheSize(other_ctx->GetKVcacheSize());
    SetModelMaxLength(other_ctx->GetModelMaxLength());
    SetModelMaxBatch(other_ctx->GetModelMaxBatch());
    SetModelMaxPrefillLength(other_ctx->GetModelMaxPrefillLength());
    SetDtype(other_ctx->GetDtype());
    SetMatmulPrecision(other_ctx->GetMatmulPrecision());
    SetPrefillMode(other_ctx->GetPrefillMode());
    SetEvictionStrategy(other_ctx->GetEvictionStrategy());
    SetFallbackDecoderWeightOnly(other_ctx->GetFallbackDecoderWeightOnly());
    SetSparsityMatmulMode(other_ctx->GetSparsityMatmulMode());
    SetLoraMaxNum(other_ctx->GetLoraMaxNum());
    SetLoraMaxRank(other_ctx->GetLoraMaxRank());
    SetLoraEnabled(other_ctx->GetLoraEnabled());
  }

 private:
  SpanCacheConfig::Ptr cache_config;

  int engine_max_length = 0;
  int engine_max_batch = 0;
  const int engine_max_top_logprobs = 10;  // const
  int engine_kvcache_alloc_size = -1;
  int engine_max_prefill_length = 1024;
  int num_heads_ = 0;
  int dec_layer_ = 0;
  int size_per_head_ = 0;
  int num_groups_ = 0;
  int intermediate_size_ = 0;
  bool use_torch_sample_ = false;
  // fallback to A16Wx in decoder phase for A8WX or AF8Wx quantized gemm if
  // decoder_weight_only_ is true.
  bool decoder_weight_only_ = false;
  int precision_ = PrecisionLevel::HIGHEST;
#ifdef ENABLE_CUDA
  AsMHAPrefill prefill_mode_ = AsMHAPrefill::AsPrefillXformer;
#else   // ENABLE_CUDA
  AsMHAPrefill prefill_mode_ = AsMHAPrefill::AsPrefillDefault;
#endif  // ENABLE_CUDA
  AsEvictionStrategy eviction_strategy_ = AsEvictionStrategy::MaxLength;
  AsSchedulingStrategy scheduling_strategy_ =
      AsSchedulingStrategy::ContextPriority;
  bool lora_enabled_ = false;
  int lora_max_num_ = 2;
  int lora_max_rank_ = 64;

 protected:
  DataType dtype = DataType::DATATYPE_UNDEFINED;
};

// use factory class to avoid device related dependency pollute user program.
class DeviceContextFactory {
 public:
  static std::shared_ptr<DeviceContext> CreateDeviceContext(
      const DeviceType device_type);
  static std::shared_ptr<DeviceContext> CreateCPUContext();
  static std::shared_ptr<DeviceContext> CreateCUDAContext();
};

}  // namespace allspark
