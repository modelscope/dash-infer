/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    weight_manager_lora.h
 */
#pragma once

#include "weight_manager.h"

namespace allspark {

class LoraManager : public WeightManagerImpl {
 public:
  LoraManager() = delete;
  LoraManager(const int lora_max_num, const RankInfo& rank_info)
      : lora_max_num_(lora_max_num), rank_info_(rank_info) {
    weight_handler_store_.resize(lora_max_num_, nullptr);
  }

  // 为了安全，lora
  // 禁止使用swap功能。必须由外部调用者来显式地load_lora和unload_lora
  void SwapInWeight(std::shared_ptr<ModelWeightHandler>& handler,
                    RankInfo info) {
    throw AsException("ALLSPARK_INVALID_CALL");
  }
  void SwapOutWeight(std::shared_ptr<ModelWeightHandler>& handler,
                     RankInfo info) {
    throw AsException("ALLSPARK_INVALID_CALL");
  }

  static std::shared_ptr<LoraManager> Create(const int lora_max_num,
                                             const RankInfo& rank_info);
  std::shared_ptr<ModelWeightHandler>& RegisterLora(
      const AsModelConfig& lora_config);
  void UnRegisterLora(const std::string& lora_name);
  std::shared_ptr<AsTensor> GetLoraTensorByName(const std::string& lora_name,
                                                const std::string& tensor_name);
  std::string MakeLoraTensorName(const std::string& op_name_,
                                 const std::string& base_weight_name,
                                 bool is_bias);
  std::string GetBiasName(std::string lora_weight_name);
  bool HasLoraBias(const std::string& lora_name,
                   const std::string& lora_weight_name);
  std::shared_ptr<ModelWeightHandler> GetHandleByName(
      const std::string& lora_name);
  int GetNumLoras() const { return lora_name_idx_map_.size(); }
  bool IsEmpty() const { return lora_name_idx_map_.size() == 0; }
  bool IsLoraExists(const std::string& lora_name) const {
    return lora_name_idx_map_.count(lora_name) != 0;
  }

  const std::unordered_map<std::string, size_t>& GetLoraNameIdxMap() {
    return lora_name_idx_map_;
  }
  virtual AsStatus ValidateWeight(
      std::shared_ptr<ModelWeightHandler>& weight_handler,
      const ModelWeightAccessInfo& weight_info,
      const DeviceContext& device_ctx);
  void PrintLoras();

 private:
  std::unordered_map<std::string, size_t> lora_name_idx_map_;
  mutable std::shared_timed_mutex lora_lock_;
  RankInfo rank_info_;  // 与AsModel的RankInfo保持一致
  int lora_max_num_;
};

}  // namespace allspark
