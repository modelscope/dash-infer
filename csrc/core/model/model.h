/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    model.h
 */

#pragma once
#include <common/common.h>
#include <common/generate_context.h>
#include <common/request.h>
#include <common/thread_pool.h>
#include <core/operator/operator.h>
#include <core/tensor/tensor.h>
#include <runtime/weight/weight_manager.h>
#include <utility/model_profiler.h>

#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace allspark {

/*
class ModelWeightHandler;
class WeightManager;
*/

using GraphOpMap =
    std::unordered_map<std::string, std::vector<std::unique_ptr<AsOperator>>>;
class AsModel {
 public:
  explicit AsModel(const std::string& model_type = "");
  virtual ~AsModel() = default;
  virtual AsStatus Init(const TransformerProto& model_proto,
                        const DeviceContext& ctx);
  // new api
  virtual AsStatus EnqueueRequest(const DLTensorMap& inputs, TensorMap* outputs,
                                  GenerateConfig& gen_cfg);
  virtual AsStatus GenerateContinue();
  virtual AsStatus GenerateContinueDecoder();
  virtual AsStatus GenerateContinueContext();
  virtual AsStatus AllocDecoderMemory();
  virtual AsStatus Warmup(int64_t bytes_available, int64_t bytes_per_req);
  virtual int64_t GetAvailableMemoryBytes();

  virtual AsStatus StartRequest(std::shared_ptr<Request> request);
  virtual AsStatus StopRequest(std::string request_id);
  virtual AsStatus ReleaseRequest(std::string request_id);
  virtual Request* GetRequestById(std::string request_id);
  AsStatus SaveWeights(std::string* out_allsparkz);
  AsStatus UnloadModelFromDeviceMemory();
  AsStatus ReloadModelToDeviceMemory();

  void UpdateAsEngineStat(AsEngineStat* as_stat);

  void PrintWeights();
  void SetRank(int rank, int nranks) {
    rank_ = rank;
    nranks_ = nranks;
  }
  RankInfo GetRankInfo() { return RankInfo(rank_, nranks_); }
  int GetRankId() { return rank_; }
  void GetInformation(std::string* model_info);
  AsTensor GetOutputTensor(std::string tensor_name);

  TensorMap& GetWeightsBuffer() { return weights_buffer_; }
  AsStatus ErrorProcess(AsStatus status) {
    gen_ctx_ = std::make_unique<GenerateContext>();
    return status;
  }
  void SetEmbedding(const std::vector<DLTensorListMap>& extra_embedding);

  void ResetProfiler();
  std::string GetOpProfilingInfo();

  void SetWeightHandler(std::shared_ptr<WeightManager> weight_manager,
                        std::shared_ptr<ModelWeightHandler> weight_handler) {
    weight_manager_ = weight_manager;
    weight_handler_ = weight_handler;
  }

  // get current running request count
  int GetUnFinishedRequest() { return current_unfinished_request_.load(); }

 protected:
  AsStatus runDecoderContext();
  AsStatus buildGenContext(GenerateContext* gen_ctx,
                           const std::shared_ptr<Request>& request) const;

  std::string model_type_;
  GraphOpMap graph_ops_;
  TensorMap weights_buffer_;  // 权重的副本，保存于CPU mem中，用于显存的换入换出
  TensorMap tensors_;  // 管理所有计算参与的tensor的空间，可用来debug中间结果
  std::vector<TensorListMap>
      embedding_;  // 管理rich_text的输入embedding，每次要清空
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::vector<AsOperator*> topo_ops_;
  const DeviceContext* ctx_;
  std::unique_ptr<GenerateContext> gen_ctx_;
  std::unique_ptr<RuntimeContext> runtime_ctx_;

  std::atomic<int> current_unfinished_request_;
  std::mutex gen_ctx_lock_;
  std::queue<std::shared_ptr<Request>> pending_request_queue_;
  std::unordered_map<std::string, std::shared_ptr<Request>> all_request_map_;
  int rank_ = 0;
  int nranks_ = 1;

  // for async, avoid of re-entrance
  std::mutex decoder_lock_;

 private:
  std::shared_ptr<ModelWeightHandler> weight_handler_;
  std::shared_ptr<WeightManager> weight_manager_;
  std::unique_ptr<ModelProfiler> model_profiler_;
  bool is_rebuild_ = false;
};

using ModelConstructor = std::function<std::unique_ptr<AsModel>()>;
using ModelMap = std::unordered_map<std::string, ModelConstructor>;

/*!
 * @brief Model factory class
 */
class ModelFactory {
 public:
  static ModelFactory& getInstance();
  ModelConstructor GetModel(const std::string& model_type_str);
  void Register(const std::string& model_type_str,
                ModelConstructor model_constructor);

 private:
  ModelFactory() = default;
  ModelFactory(const ModelFactory&) = delete;
  ModelFactory(ModelFactory&&) = delete;
  ModelMap model_set_;
};

/*!
 * @brief Model reflector class
 */
class ModelRegisterHelper {
 public:
  ModelRegisterHelper(const std::string& model_type_str,
                      ModelConstructor model_constructor) {
    ModelFactory::getInstance().Register(model_type_str.c_str(),
                                         model_constructor);
  }
};

#define REGISTER_MODEL(key, typed_class)            \
  static ModelRegisterHelper typed_class##Register( \
      key, []() -> std::unique_ptr<AsModel> {       \
        return std::make_unique<typed_class>(key);  \
      });

}  // namespace allspark
