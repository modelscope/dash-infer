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
  LoraManager(const RankInfo& rank_info) : rank_info_(rank_info) {}
  static std::shared_ptr<LoraManager> Create(const RankInfo& rank_info);
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
  bool IsEmpty() const { return lora_name_idx_map_.size() == 0; }
  bool IsLoraExists(const std::string& lora_name) const {
    return lora_name_idx_map_.count(lora_name) != 0;
  }

  const std::unordered_map<std::string, size_t>& GetLoraNameIdxMap() {
    return lora_name_idx_map_;
  }

  void PrintLoras();

 private:
  std::unordered_map<std::string, size_t> lora_name_idx_map_;
  mutable std::shared_timed_mutex lora_lock_;
  RankInfo rank_info_;  // 与AsModel的RankInfo保持一致
};

}  // namespace allspark
