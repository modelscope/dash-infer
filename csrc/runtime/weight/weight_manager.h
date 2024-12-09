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
// class ModelWeightHandler;

enum class SwapStatus {
  SwapInit = 0,  // initial state.
  SwapOut = 1,
  SwapIn = 2,
};

class WeightSwapConfig {
 public:
  bool enable = false;
};

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

enum class WeightEvent {
  WeightOnLoad,
  WeightOnSwapOut,
  WeightOnSwapIn,
  WeightOnFree,
};

using WeightEventCallback = std::function<void(
    std::shared_ptr<ModelWeightHandler> weight_handler, WeightEvent event)>;

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
  virtual std::shared_ptr<ModelWeightHandler> RegisterModel(
      AsModelConfig& confg, std::shared_ptr<TransformerProto> model_ir);
  virtual int GetNumModels() = 0;

  // load the model by different worker, only load the worker's shard, will be
  // called different thread.
  //
  // this function may call by different thread concurrently.
  // in this function it must make sure read file in single thread and can
  // split the weight in different thread. read file in sequence will save
  // disk io, but in this function it should split the weight concurrently.
  virtual AsStatus LoadWeightForModel(
      const DeviceContext& target_device_ctx,
      std::shared_ptr<ModelWeightHandler>& weight_handler, RankInfo& rank_info);

  // weight buffer function:
  // 1. clear already swap out weight buffer (swap out)
  // 2. fill in tensor saved buffer( swap in  ), opt, this should be save
  // splitted weights.
  // 3. give weight map buffer to give op init. (model.cpp: 112) , (don't want
  // to expose the weight map pointer, how it?)
  // 4. save weight into an output string (this part can be move into
  // manager.)

  /**
   * swap out device memory, save those tensor for that rank to cpu, first
   * time call this function will allocate cpu tensor to save the device
   * memory.
   * This function must called after models's load and split for multiple
   * devices.
   *
   * @param handler weight handler
   */
  virtual void SwapOutWeight(std::shared_ptr<ModelWeightHandler>& handler,
                             RankInfo info);

  /**
   * swap in device memory, allocate device tensor and copy cpu tensor.
   * because each rank have their
   *
   * @param handler weight handler.
   */
  virtual void SwapInWeight(std::shared_ptr<ModelWeightHandler>& handler,
                            RankInfo info);

  /**
   * free cpu memory resource for swap function
   * the cpu memory allocate in first swap out will be free after this
   * function.
   *
   * @param handler weight handler
   */
  virtual void FreeSwapResource(std::shared_ptr<ModelWeightHandler>& handler);

  virtual SwapStatus GetSwapStatus(
      std::shared_ptr<ModelWeightHandler>& handler);

  virtual void SetSwapConfig(std::shared_ptr<ModelWeightHandler>& handler,
                             WeightSwapConfig swap_config);

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
  virtual std::shared_ptr<AsTensor> GetWeightTensor(
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
  virtual void SaveWeights(std::shared_ptr<ModelWeightHandler> handler,
                           std::string* out_allsparkz);

  /**
   * free the weight memory belong to this model.
   */
  virtual void FreeWeight(std::shared_ptr<ModelWeightHandler> handler);

  virtual void CheckModelConsistency(
      std::shared_ptr<ModelWeightHandler> weight_handler);

  virtual void RegisterWeightEventListener(WeightEventCallback callback);

 protected:
  WeightManager();
};

// tmp moved here, for compiling of LoraManager
class WeightManagerImpl : public WeightManager {
 public:
  ~WeightManagerImpl();
  // register the model information
  // check the model file exists
  // raise exception if there is some error.
  std::shared_ptr<ModelWeightHandler> RegisterModel(
      AsModelConfig& confg,
      std::shared_ptr<TransformerProto> model_ir) override;
  int GetNumModels();

  // check the weight and param consistency
  void CheckModelConsistency(
      std::shared_ptr<ModelWeightHandler> handler) override;

  // load the model by different worker.
  // this function may call by different thread concurrently.
  // in this function it must make sure read file in single thread and can
  // split the weight in different thread. read file in sequence will save
  // disk io, but in this function it should split the weight concurrently.
  AsStatus LoadWeightForModel(
      const DeviceContext& target_device_ctx,
      std::shared_ptr<ModelWeightHandler>& weight_handler,
      RankInfo& rank_info) override;

  std::shared_ptr<AsTensor> GetWeightTensor(
      std::shared_ptr<ModelWeightHandler>& handler, RankInfo& rank_info,
      const std::string& name) override;

  void SaveWeights(std::shared_ptr<ModelWeightHandler> handler,
                   std::string* out_allsparkz) override;

  bool SeekToNextTensor(FILE* fp, TensorInfo& info);

  void SwapOutWeight(std::shared_ptr<ModelWeightHandler>& handler,
                     RankInfo info) override;

  void SwapInWeight(std::shared_ptr<ModelWeightHandler>& handler,
                    RankInfo info) override;

  void FreeSwapResource(std::shared_ptr<ModelWeightHandler>& handler) override;

  SwapStatus GetSwapStatus(
      std::shared_ptr<ModelWeightHandler>& handler) override;

  void SetSwapConfig(std::shared_ptr<ModelWeightHandler>& handler,
                     WeightSwapConfig swap_config) override;

  bool IsSwapEnable(std::shared_ptr<ModelWeightHandler>& handler);

  void FreeWeight(std::shared_ptr<ModelWeightHandler> handler) override;

  void RegisterWeightEventListener(WeightEventCallback callback) override;

 protected:
  //    typedef std::unique_lock<std::shared_timed_mutex> rw_write_lock;
  //    typedef std::shared_lock<std::shared_timed_mutex> rw_read_lock;

  typedef unique_lock_wrapper<std::shared_timed_mutex> rw_write_lock;
  typedef shared_lock_wrapper<std::shared_timed_mutex> rw_read_lock;

  typedef std::map<RankInfo, std::shared_ptr<TensorMap>> weights_of_rank_t;

  typedef std::map<std::shared_ptr<ModelWeightHandler>, weights_of_rank_t>
      weight_storage_t;

  typedef std::map<RankInfo, SwapStatus> swap_status_of_weight_t;
  typedef std::map<std::shared_ptr<ModelWeightHandler>, swap_status_of_weight_t>
      swap_status_t;

  typedef std::map<std::shared_ptr<ModelWeightHandler>, WeightSwapConfig>
      swap_config_t;

  bool handler_is_avalibile(std::shared_ptr<ModelWeightHandler>& handler) {
    return weight_storage_.count(handler) > 0;
  }

  bool handler_swap_in_is_avalibile(
      std::shared_ptr<ModelWeightHandler>& handler) {
    return swap_weight_storage_.count(handler) > 0;
  }

  bool handler_is_swapout(std::shared_ptr<ModelWeightHandler>& handler,
                          RankInfo rank_info) {
    try {
      return swap_status_.at(handler).at(rank_info) == SwapStatus::SwapOut;
    } catch (std::out_of_range& e) {
      return false;
    }
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

  weight_storage_t swap_weight_storage_;

  std::vector<WeightEventCallback> weight_event_callback;

  // weight swap status.
  swap_status_t swap_status_;

  swap_config_t swap_config_;
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
