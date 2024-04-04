/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    engine_worker.h
 */

#ifndef ENGINE_WORKER_H
#define ENGINE_WORKER_H

#include <common/device_context.h>
#include <core/model/model.h>
#include <weight/weight_manager.h>

namespace allspark {

class Worker {
 public:
  Worker(int rank, int nranks, int device_id)
      : rank_(rank), nranks_(nranks), device_id_(device_id) {}

  virtual AsStatus InitCCL(int rank, int nranks) = 0;
  virtual void SetWorkerDeviceId(int device_id) = 0;

  virtual AsStatus SetNumThreads(int nums) {
    return AsStatus::ALLSPARK_SUCCESS;
  }
  AsStatus BuildModel(const TransformerProto& model_proto,
                      std::shared_ptr<WeightManager> weight_manager,
                      std::shared_ptr<ModelWeightHandler> model_handler,
                      const DeviceContext* main_ctx);

  AsStatus RebuildModelFromBuffer(
      const std::unique_ptr<TransformerProto>& model_ir);

  AsStatus EnqueueRequest(const DLTensorMap& inputs, TensorMap* outputs,
                          GenerateConfig& gen_cfg);

  int GetUnFinishedRequest();
  Request* GetRequestById(std::string request_id);
  AsStatus StopRequest(std::string request_id);
  AsStatus ReleaseRequest(std::string request_id);

  AsStatus RunTextGenerationContinue();
  AsStatus RunTextGenerationContext();
  AsStatus AllocDecoderMemory();
  AsStatus Warmup(int64_t bytes_available, int64_t bytes_per_req);
  int64_t GetAvailableMemoryBytes();

  void UpdateAsEngineStat(AsEngineStat* as_stat);
  DeviceContext* GetDeviceContext() { return device_ctx_.get(); }

  void ResetProfiler();
  std::string GetOpProfilingInfo();

  AsStatus UnloadModelFromDeviceMemory();

  int GetRank() { return rank_; }

  int GetRankNums() { return nranks_; }

  AsStatus GetInformation(std::string* model_info);

  void SetWeightManager(std::shared_ptr<WeightManager>& manager) {
    weight_manager_ = manager;
  }

  std::unique_ptr<AsModel>& GetModel() { return model_; }

 protected:
  int rank_;
  int nranks_;
  int device_id_;
  std::unique_ptr<DeviceContext> device_ctx_;
  std::unique_ptr<AsModel> model_;
  std::shared_ptr<WeightManager> weight_manager_;

 private:
  Worker() = delete;
};

class CpuWorker : public Worker {
 public:
  CpuWorker(int rank, int nranks, int device_id);
  void SetWorkerDeviceId(int device_id) override{};
  AsStatus InitCCL(int rank, int nranks) override;
  AsStatus SetNumThreads(int nums) override;
};

};  // namespace allspark

#endif
