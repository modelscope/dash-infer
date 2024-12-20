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
  virtual ~Worker() {}

  virtual void Init() = 0;

  virtual AsStatus InitCCL(int rank, int nranks) = 0;
  virtual void SetWorkerDeviceId(int device_id) = 0;

  virtual AsStatus SetNumThreads(int nums) {
    return AsStatus::ALLSPARK_SUCCESS;
  }
  AsStatus BuildModel(
      const TransformerProto& model_proto,
      std::shared_ptr<WeightManager> weight_manager,
      std::shared_ptr<ModelWeightHandler> model_handler,
      const DeviceContext* main_ctx,
      PrefixCacheCoordinator::Ptr prefix_cache_coordinator = nullptr);

  AsStatus RebuildModelFromBuffer(
      const std::unique_ptr<TransformerProto>& model_ir);

  AsStatus StartRequestImpl(std::shared_ptr<RequestHandle> request_handle,
                            std::string uuid, TensorMap* outputs,
                            const GenerateConfig& gen_cfg);

  int GetUnFinishedRequest();
  Request* GetRequestById(std::string request_id);
  AsStatus StopRequest(std::string request_id);
  AsStatus ReleaseRequest(std::string request_id);

  AsStatus RunTextGenerationContinue();
  AsStatus RunTextGenerationContext(bool is_new_context);
  AsStatus AllocDecoderMemory();
  AsStatus Warmup(int64_t bytes_available, int64_t bytes_runtime);
  int64_t GetAvailableMemoryBytes();
  int64_t GetOccupiedMemoryBytes();
  int64_t GetTotalMemoryBytes();

#if ENABLE_SPAN_ATTENTION
  int64_t GetFreeFrame();
#endif
  void UpdateAsEngineStat(AsEngineStat* as_stat);
  DeviceContext* GetDeviceContext() { return device_ctx_.get(); }

  void ResetProfiler() { model_->ResetProfiler(); }

#if ENABLE_SPAN_ATTENTION
  void ResetPrefixCache() { model_->ResetPrefixCache(); }
  void SetPrefixCacheSeqlenThre(int thre) {
    model_->SetPrefixCacheSeqlenThre(thre);
  }
#endif

  std::string GetOpProfilingInfo();

  AsStatus UnloadModelFromDeviceMemory();

  int GetRank() { return rank_; }

  int GetRankNums() { return nranks_; }

  AsStatus GetInformation(std::string* model_info);

  void SetWeightManager(std::shared_ptr<WeightManager>& manager) {
    weight_manager_ = manager;
  }

  AsStatus LoadLoraByName(const std::string lora_name);
  AsStatus UnloadLoraByName(const std::string lora_name);
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

#ifdef ENABLE_CUDA
class CudaWorker : public Worker {
 public:
  CudaWorker(int rank, int nranks, const ncclUniqueId& id, int device_id);
  virtual ~CudaWorker();
  void SetWorkerDeviceId(int device_id) override;
  void Init() override;
  AsStatus InitCCL(int rank, int nranks) override;

 private:
  const ncclUniqueId nccl_id_;
  thread_local static int last_device_id_of_this_thread_;
};
#endif
class CpuWorker : public Worker {
 public:
  CpuWorker(int rank, int nranks, int device_id);
  void Init() override;
  void SetWorkerDeviceId(int device_id) override{};
  AsStatus InitCCL(int rank, int nranks) override;
  AsStatus SetNumThreads(int nums) override;
};

};  // namespace allspark

#endif
