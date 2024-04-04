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
  void SetKVcacheSize(int kvsize) { engine_kvcache_alloc_size = kvsize; }
  int GetKVcacheSize() const { return engine_kvcache_alloc_size; }
  void SetModelMaxLength(int max_length) { engine_max_length = max_length; }
  int GetModelMaxLength() const { return engine_max_length; }
  void SetModelMaxBatch(int max_batch) { engine_max_batch = max_batch; }
  int GetModelMaxBatch() const { return engine_max_batch; }
  void SetDtype(DataType new_dtype) { dtype = new_dtype; }
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
  void SetPrefillMode(AsMHAPrefill prefill_mode) {
    prefill_mode_ = prefill_mode;
  }

  int GetNumberHeads() const { return num_heads_; }
  int GetDecoderLayer() const { return dec_layer_; }
  int GetSizePerHead() const { return size_per_head_; }
  int GetNumberGroups() const { return num_groups_; }
  int GetMaxTopLogprobs() const { return engine_max_top_logprobs; }
  AsMHAPrefill GetPrefillMode() const { return prefill_mode_; }
  void CopyFromOther(const DeviceContext* other_ctx) {
    SetNumberHeads(other_ctx->GetNumberHeads());
    SetDecoderLayer(other_ctx->GetDecoderLayer());
    SetSizePerHead(other_ctx->GetSizePerHead());
    SetNumberGroups(other_ctx->GetNumberGroups());
    SetKVcacheSize(other_ctx->GetKVcacheSize());
    SetModelMaxLength(other_ctx->GetModelMaxLength());
    SetModelMaxBatch(other_ctx->GetModelMaxBatch());
    SetDtype(other_ctx->GetDtype());
    SetMatmulPrecision(other_ctx->GetMatmulPrecision());
    SetPrefillMode(other_ctx->GetPrefillMode());
  }

 private:
  int engine_max_length = 0;
  int engine_max_batch = 0;
  const int engine_max_top_logprobs = 10;  // const
  int engine_kvcache_alloc_size = -1;

  int num_heads_ = 0;
  int dec_layer_ = 0;
  int size_per_head_ = 0;
  int num_groups_ = 0;

  int precision_ = PrecisionLevel::HIGHEST;
  AsMHAPrefill prefill_mode_ = AsMHAPrefill::AsPrefillDefault;
  DataType dtype = DataType::DATATYPE_UNDEFINED;
};

// use factory class to avoid device related dependency pollute user program.
class DeviceContextFactory {
 public:
  static std::shared_ptr<DeviceContext> CreateDeviceContext(
      const DeviceType device_type);
  static std::shared_ptr<DeviceContext> CreateCPUContext();
};

}  // namespace allspark
