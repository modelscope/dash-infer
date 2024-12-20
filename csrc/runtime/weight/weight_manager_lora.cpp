/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    weight_manager_lora.cpp
 */

#include "weight_manager_lora.h"

#include <sys/stat.h>

namespace allspark {

std::shared_ptr<LoraManager> LoraManager::Create(const int lora_max_num,
                                                 const RankInfo& rank_info) {
  return std::make_shared<LoraManager>(lora_max_num, rank_info);
}

// currently, only support 1 base-model for each LoraManager
std::shared_ptr<ModelWeightHandler>& LoraManager::RegisterLora(
    const AsModelConfig& lora_config) {
  rw_write_lock lk(lora_lock_,
                   "RegisterLora");  // 每个AsModel有自己的LoraManager，
  // 无需锁
  // 寻找空闲slot
  size_t i = 0;
  for (i = 0; i < lora_max_num_; i++) {
    if (weight_handler_store_[i] == nullptr) break;
  }
  size_t new_id = i;
  AS_ENFORCE(new_id < lora_max_num_);
  auto fake_proto = std::make_shared<TransformerProto>();
  auto weight_handle_ptr = std::make_shared<ModelWeightHandler>(
      new_id, lora_config,
      fake_proto);  // nullptr: lora只有权重，没有对应的子图
  weight_handler_store_[new_id] = weight_handle_ptr;  // 复用 handle slot
  lora_name_idx_map_[lora_config.model_name] = new_id;
  DLOG(INFO) << "LoraManager::RegisterLora " << lora_config.model_name
             << ", handle ptr=" << weight_handler_store_[new_id].get()
             << " done!";
  return weight_handler_store_[new_id];
}

void LoraManager::UnRegisterLora(const std::string& lora_name) {
  rw_write_lock lk(lora_lock_,
                   "UnRegisterLora");  // 每个AsModel有自己的LoraManager，
  // 无需锁
  auto idx = lora_name_idx_map_.at(lora_name);
  auto& lora_weight_handle = weight_handler_store_.at(idx);
  AS_ENFORCE(handler_is_avalibile(lora_weight_handle));
  LOG(INFO) << "UnRegisterLora " << lora_name
            << " handle id=" << lora_weight_handle->GetId();
  weight_storage_[lora_weight_handle].erase(rank_info_);  // 释放cuda mem
  // 让该lora handle彻底消失
  weight_storage_.erase(lora_weight_handle);
  weight_handler_store_[idx] = nullptr;
  lora_name_idx_map_.erase(lora_name);
  LOG(INFO) << "LoraManager::UnRegisterLora " << lora_name << " done!";
}

std::shared_ptr<AsTensor> LoraManager::GetLoraTensorByName(
    const std::string& lora_name, const std::string& tensor_name) {
  rw_write_lock lk(lora_lock_,
                   "GetLoraTensorByName");  // 每个AsModel有自己的LoraManager，
  AS_ENFORCE(lora_name_idx_map_.count(lora_name) > 0);
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
  AS_ENFORCE(lora_name_idx_map_.count(lora_name) > 0);
  auto& lora_weight_handle =
      weight_handler_store_.at(lora_name_idx_map_.at(lora_name));
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
  AS_ENFORCE(lora_name_idx_map_.count(lora_name) > 0);
  return weight_handler_store_.at(lora_name_idx_map_.at(lora_name));
}

AsStatus LoraManager::ValidateWeight(
    std::shared_ptr<ModelWeightHandler>& weight_handler,
    const ModelWeightAccessInfo& weight_info, const DeviceContext& device_ctx) {
  auto inference_dtype = device_ctx.GetDtype();
  auto lora_dtype = weight_info.info.dtype;
  if (lora_dtype != inference_dtype) {
    LOG(ERROR) << "lora " << weight_handler->GetModelConfig().model_name
               << " dtype mismatch!"
               << " dtype for inference is " << DataType_Name(inference_dtype)
               << " lora's dtype is " << DataType_Name(lora_dtype);
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
  if (weight_info.name.find(".lora_B") != std::string::npos &&
      weight_info.info.shape[0] >
          weight_handler->GetModelConfig().lora_max_rank)
    return AsStatus::ALLSPARK_LORA_RANK_EXCEED_LIMIT_ERROR;
  return AsStatus::ALLSPARK_SUCCESS;
}

void LoraManager::PrintLoras() {
  LOG(INFO) << "loraStorage size=" << weight_handler_store_.size() << " "
            << lora_name_idx_map_.size();
  LOG(INFO) << "loras in map:";
  for (auto& item : lora_name_idx_map_) {
    LOG(INFO)
        << item.first << " : " << item.second << ", "
        << weight_handler_store_.at(item.second)->GetModelConfig().model_name
        << ", addr=" << &(weight_handler_store_.at(item.second))
        << ", ptr=" << weight_handler_store_.at(item.second).get();
  }
  LOG(INFO) << "loras in vector:";
  for (int i = 0; i < weight_handler_store_.size(); i++) {
    if (weight_handler_store_.at(i) == nullptr) continue;
    LOG(INFO) << i << " : "
              << weight_handler_store_.at(i)->GetModelConfig().model_name;
  }

  LOG(INFO) << "-----lora print done";
}

}  // namespace allspark
