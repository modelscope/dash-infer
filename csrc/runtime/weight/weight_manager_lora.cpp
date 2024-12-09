/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    weight_manager_lora.cpp
 */

#include "weight_manager_lora.h"

namespace allspark {

std::shared_ptr<LoraManager> LoraManager::Create(const RankInfo& rank_info) {
  return std::make_shared<LoraManager>(rank_info);
}

// currently, only support 1 base-model for each LoraManager
std::shared_ptr<ModelWeightHandler>& LoraManager::RegisterLora(
    const AsModelConfig& lora_config) {
  DLOG(INFO) << "enter LoraManager::RegisterLora " << lora_config.model_name;
  rw_write_lock lk(lora_lock_,
                   "RegisterLora");  // 每个AsModel有自己的LoraManager，
  // 无需锁
  size_t new_id = weight_handler_store_.size();
  auto fake_proto = std::make_shared<TransformerProto>();
  weight_handler_store_.emplace_back(std::make_shared<ModelWeightHandler>(
      new_id, lora_config,
      fake_proto));  // nullptr: lora只有权重，没有对应的子图
  lora_name_idx_map_[lora_config.model_name] = new_id;
  LOG(INFO) << "LoraManager::RegisterLora " << lora_config.model_name << " "
            << weight_handler_store_.back() << " done!";
  return weight_handler_store_.back();
}

void LoraManager::UnRegisterLora(const std::string& lora_name) {
  rw_write_lock lk(lora_lock_,
                   "UnRegisterLora");  // 每个AsModel有自己的LoraManager，
  // 无需锁
  DLOG(INFO) << "enter LoraManager::UnRegisterLora " << lora_name;
  auto idx = lora_name_idx_map_.at(lora_name);
  auto& lora_weight_handle = weight_handler_store_.at(idx);
  SwapOutWeight(lora_weight_handle, rank_info_);  // free cuda mem
  weight_handler_store_[idx] = nullptr;
  lora_name_idx_map_.erase(lora_name);

  // 因为weight_handler_store_设计成vector， 删掉某个lora后,
  // weight_handler_store_概念上会产生"空洞"，需要重新compact: compact
  // weight_handler_store_
  std::vector<std::shared_ptr<ModelWeightHandler>> tmp_weight_handler_store;
  for (auto& item : lora_name_idx_map_) {
    auto& lora_name = item.first;
    auto& idx = item.second;
    tmp_weight_handler_store.emplace_back(weight_handler_store_[idx]);
    idx = tmp_weight_handler_store.size() - 1;
  }
  // std::swap(weight_handler_store_, tmp_weight_handler_store);
  weight_handler_store_ = tmp_weight_handler_store;
  LOG(INFO) << "LoraManager::UnRegisterLora " << lora_name << " done!";
}

std::shared_ptr<AsTensor> LoraManager::GetLoraTensorByName(
    const std::string& lora_name, const std::string& tensor_name) {
  rw_write_lock lk(lora_lock_,
                   "GetLoraTensorByName");  // 每个AsModel有自己的LoraManager，
  auto& lora_weight_handle =
      weight_handler_store_.at(lora_name_idx_map_.at(lora_name));
  return GetWeightTensor(lora_weight_handle, rank_info_, tensor_name);
}

std::string LoraManager::MakeLoraTensorName(const std::string& op_name_,
                                            const std::string& base_weight_name,
                                            bool is_bias) {
  std::string lora_suffix = ".lora_";
  auto pos = op_name_.rfind(lora_suffix);
  assert(pos == op_name_.length() - lora_suffix.length() - 1);
  lora_suffix = op_name_.substr(pos);
  std::string weight_suffix = ".weight";
  pos = base_weight_name.rfind(weight_suffix);
  assert(pos == base_weight_name.length() - weight_suffix.length());
  std::string ret = base_weight_name;
  ret.replace(pos, 0, lora_suffix);
  if (is_bias) {
    pos = ret.rfind(weight_suffix);
    ret.replace(pos, weight_suffix.length(), ".bias");
  }
  return ret;
}

bool LoraManager::HasLoraBias(const std::string& lora_name,
                              const std::string& lora_weight_name) {
  rw_write_lock lk(lora_lock_,
                   "HasLoraBias");  // 每个AsModel有自己的LoraManager，
  DLOG(INFO) << "enter LoraManager::HasLoraBias " << lora_name << " "
             << lora_weight_name;
  auto& lora_weight_handle =
      weight_handler_store_.at(lora_name_idx_map_.at(lora_name));
  DLOG(INFO) << "gethandler:" << lora_weight_handle;
  std::string lora_bias_name = lora_weight_name;
  std::string weight_suffix = ".weight";
  std::string bias_suffix = ".bias";
  auto pos = lora_weight_name.rfind(weight_suffix);
  lora_bias_name.replace(pos, weight_suffix.length(), bias_suffix);

  bool ret = false;
  if (!handler_is_avalibile(lora_weight_handle) ||
      !weight_on_rank_is_avalibile(lora_weight_handle, rank_info_)) {
    LOG(ERROR) << "Try to get from non-exist lora " << lora_name << " : "
               << lora_weight_name << " for rank " << rank_info_
               << " handler_is_avalibile="
               << handler_is_avalibile(lora_weight_handle)
               << " weight_on_rank_is_avalibile="
               << weight_on_rank_is_avalibile(lora_weight_handle, rank_info_);
    throw AsException("weight get: lora " + lora_name + " not found!");
  }
  if (handler_is_swapout(lora_weight_handle, rank_info_)) {
    throw AsException("access swap out tensor");
  }
  auto& weight_map = get_weight_on_rank(lora_weight_handle, rank_info_);
  ret = weight_map->count(lora_bias_name) != 0;
  return ret;
}

std::string LoraManager::GetBiasName(std::string lora_weight_name) {
  auto pos = lora_weight_name.rfind(".weight");
  if (pos == std::string::npos) return lora_weight_name + ".bias";
  return lora_weight_name.replace(pos, 7, ".bias");
}

std::shared_ptr<ModelWeightHandler> LoraManager::GetHandleByName(
    const std::string& lora_name) {
  rw_write_lock lk(lora_lock_,
                   "GetHandleByName");  // 每个AsModel有自己的LoraManager，
  return weight_handler_store_.at(lora_name_idx_map_.at(lora_name));
}

void LoraManager::PrintLoras() {
  LOG(INFO) << "loraStorage size=" << weight_handler_store_.size() << " "
            << lora_name_idx_map_.size();
  for (auto& item : lora_name_idx_map_) {
    LOG(INFO)
        << item.first << " : " << item.second << ", "
        << weight_handler_store_.at(item.second)->GetModelConfig().model_name;
  }
  LOG(INFO) << "-----lora print done";
}

}  // namespace allspark