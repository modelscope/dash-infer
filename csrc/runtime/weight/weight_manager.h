/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    weight_manager.h
 */
#pragma once

// #include <core/model/model.h>
#include <device_context.h>
#include <interface/allspark.h>
#include <utility/mutex_wrapper.h>

#include <mutex>
#include <string>

#include "weight_loader.h"
#include "weight_saver.h"

namespace allspark {

class RankInfo;
class TransformerProto;

class ModelWeightHandler {
 public:
  typedef size_t handler_id_t;

  ModelWeightHandler(size_t id, AsModelConfig model_config,
                     std::shared_ptr<TransformerProto>& ptr)
      : id_(id), model_config_(std::move(model_config)), model_ir_weak_(ptr){};
  size_t GetId() const { return id_; }

  AsModelConfig& GetModelConfig() { return model_config_; }

  inline bool operator==(const ModelWeightHandler& other) const {
    return this->id_ == other.id_ && model_config_ == other.model_config_;
  }

 private:
  handler_id_t id_;
  AsModelConfig model_config_;

  // don't use shared ptr here, because this handler will spread with their
  // caller.
  std::weak_ptr<TransformerProto> model_ir_weak_;
};

struct ModelWeightAccessInfo {
  std::string name;
  TensorInfo info;
  uint64_t weight_offset;
  size_t size_bytes;
};

/**
 * handle a weight file load task, include load form map or load from saved
 * memory buffer. replace all the allsparkz functions.
 *
 * save the memory footprint and paralel split the tensor.
 */
class WeightManager {
 public:
  static std::shared_ptr<WeightManager> Create();

  // register the model information
  // check the model file exists
  // raise exception if there is some error.
  std::shared_ptr<ModelWeightHandler> RegisterModel(
      AsModelConfig& confg, std::shared_ptr<TransformerProto> model_ir);

  // load the model by different worker, only load the worker's shard, will be
  // called different thread.
  //
  // this function may call by different thread concurrently.
  // in this function it must make sure read file in single thread and can
  // split the weight in different thread. read file in sequence will save
  // disk io, but in this function it should split the weight concurrently.
  AsStatus LoadWeightForModel(
      const DeviceContext& target_device_ctx,
      std::shared_ptr<ModelWeightHandler> weight_handler, RankInfo& rank_info);

  /**
   * Get the weight tensor, if will return the weight for this rank, it maybe
   * swap in tensor if it was swapped out. Becuase it return the pointer of
   * tensor pointer, if the whole model was being swapped out, the return
   * tensor pointer will be invalid (will be manually free tensor's storage
   * block to make sure gpu memory is going to be free.
   *
   * @param handler weight handler.
   * @param rank_info current rank
   * @param name  weight name
   *
   * @return shared ptr of tensor.
   */
  std::shared_ptr<AsTensor> GetWeightTensor(
      std::shared_ptr<ModelWeightHandler>& handler, RankInfo& rank_info,
      const std::string& name);

  /**
   * save weight in a string buffer, only support non tensor loaded by one
   * rank
   *
   * this function only is a legancy support in c++ interface
   *
   * @param handler weight handler.
   * @param out_allsparkz out string buffer.
   */
  void SaveWeights(std::shared_ptr<ModelWeightHandler> handler,
                   std::string* out_allsparkz);

  void CheckModelConsistency(
      std::shared_ptr<ModelWeightHandler> weight_handler);

 protected:
  WeightManager();
};

class WeightManagerImpl : public WeightManager {
 public:
  // register the model information
  // check the model file exists
  // raise exception if there is some error.
  std::shared_ptr<ModelWeightHandler> RegisterModel(
      AsModelConfig& confg, std::shared_ptr<TransformerProto> model_ir);

  // check the weight and param consistency
  void CheckModelConsistency(std::shared_ptr<ModelWeightHandler> handler);

  // load the model by different worker.
  // this function may call by different thread concurrently.
  // in this function it must make sure read file in single thread and can
  // split the weight in different thread. read file in sequence will save
  // disk io, but in this function it should split the weight concurrently.
  AsStatus LoadWeightForModel(
      const DeviceContext& target_device_ctx,
      std::shared_ptr<ModelWeightHandler> weight_handler, RankInfo& rank_info);

  std::vector<ModelWeightAccessInfo> GetAccessOrderOfWeightFile(
      std::shared_ptr<ModelWeightHandler> mhandle);

  std::shared_ptr<AsTensor> GetWeightTensor(
      std::shared_ptr<ModelWeightHandler>& handler, RankInfo& rank_info,
      const std::string& name);

  void SaveWeights(std::shared_ptr<ModelWeightHandler> handler,
                   std::string* out_allsparkz);

  bool SeekToNextTensor(FILE* fp, TensorInfo& info);

 protected:
  typedef unique_lock_wrapper<std::shared_timed_mutex> rw_write_lock;
  typedef shared_lock_wrapper<std::shared_timed_mutex> rw_read_lock;

  typedef std::map<RankInfo, std::shared_ptr<TensorMap>> weights_of_rank_t;

  typedef std::map<std::shared_ptr<ModelWeightHandler>, weights_of_rank_t>
      weight_storage_t;

  bool handler_is_avalibile(std::shared_ptr<ModelWeightHandler>& handler) {
    return weight_storage_.count(handler) > 0;
  }

  bool weight_on_rank_is_avalibile(std::shared_ptr<ModelWeightHandler>& handler,
                                   RankInfo& rank_info) {
    return (weight_storage_.count(handler) > 0) &&
           (weight_storage_[handler].count(rank_info) > 0);
  }

  auto& get_weight_on_rank(std::shared_ptr<ModelWeightHandler>& handler,
                           const RankInfo& rank_info) {
    return weight_storage_[handler][rank_info];
  }

  void DuplicateTensorsToDeviceType(weights_of_rank_t& weight_on_handler,
                                    DeviceType type,
                                    const RankInfo* rank_info_p);

  mutable std::shared_timed_mutex lock_;

  std::vector<std::shared_ptr<ModelWeightHandler>> weight_handler_store_;

  std::map<ModelWeightHandler::handler_id_t, std::shared_ptr<TransformerProto>>
      proto_store_;

  weight_storage_t weight_storage_;
};

inline bool operator<(const ModelWeightHandler& lhs,
                      const ModelWeightHandler& rhs) {
  return lhs.GetId() < rhs.GetId();
}

inline WeightManagerImpl* GetImpl(WeightManager* ptr) {
  return (WeightManagerImpl*)ptr;
}
inline const WeightManagerImpl* GetImpl(const WeightManager* ptr) {
  return (const WeightManagerImpl*)ptr;
}
}  // namespace allspark
