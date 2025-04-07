/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    as_engine.h
 */

#include "engine_worker.h"
#include "interface/allspark_check.h"
#include "thread_pool_with_id.h"
#include "thread_utils.h"
#include "weight/weight_loader.h"
#include "weight/weight_manager.h"
#ifdef ENABLE_CUDA
#include <cuda/cuda_context.h>
#include <cuda/gpu_profiler.h>
#endif
#include <common/allocator.h>
#include <common/device_context.h>
#include <common/env_config.h>
#include <core/model/model.h>
#include <cpu/cpu_context.h>
#include <cpu/cpu_info.h>
#include <fcntl.h>
#include <git_version.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <mutex_wrapper.h>
#include <unistd.h>
#include <utility/allspark_logging.h>
#include <utility/allsparkz_util.h>
#include <utility/check.h>
#include <utility/file_util.h>
#include <utility/mem_registry.h>
#include <utility/timer.h>
#include <utility/uuid.h>

#include <common/as_param_check.hpp>
#include <exception>
#include <filesystem>
#include <fstream>
#include <future>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "engine_runtime.h"
#include "extra_embedding.hpp"
#include "runtime/weight/weight_manager_lora.h"

#ifdef ENABLE_CUDA
#include <nccl.h>
#endif
#include <core/tensor/tensor.h>

#ifdef ENABLE_MULTINUMA
#include <mpi.h>
#endif

#ifdef ENABLE_JSON_MODE
#include <utility/format_enforcer.h>
#endif

namespace fs = std::filesystem;
namespace allspark {
class AsEngineImpl final {
 public:
  AsEngineImpl();
  ~AsEngineImpl();

  // Model Inference API
  // build model from structure config.
  AsStatus BuildModelFromConfigStruct(AsModelConfig& model_config);

  // unload model from device memory, aka, swap out.
  AsStatus UnloadModelFromDeviceMemory(const char* model_name);

  // build model from saved memory buffer, for the swap function.
  AsStatus ReloadModelFromDeviceMemory(const char* model_name);

  AsStatus LoadLoraByName(const char* model_name, const char* lora_name);
  AsStatus UnloadLoraByName(const char* model_name, const char* lora_name);
  std::vector<std::string> LoadFakeLoras(const char* model_name);
  void UnloadFakeLoras(const char* model_name,
                       const std::vector<std::string>& fake_lora_names);

  AsStatus GetModelInformation(const char* model_name, std::string* model_info);

  AsFileInfo GetFileInformation(const char* as_model_path,
                                const char* as_param_path);

  AsStatus StartModel(const char* model_name);
  AsStatus TunePrefixCache(const char* model_name);
  AsStatus StopModel(const char* model_name);
  AsStatus ReleaseModel(const char* model_name);

  AsStatus StartRequest(const char* model_name,
                        std::shared_ptr<AsEngine::RequestContent> request_info,
                        RequestHandle** request_handle,
                        AsEngine::ResultQueue** queue,
                        const std::string customized_uuid = "");

  AsStatus StopRequest(const char* model_name, RequestHandle* request_handle);
  AsStatus ReleaseRequest(const char* model_name,
                          RequestHandle* request_handle);
  // sync request handle, nullptr for all request on this model.
  AsStatus SyncRequest(const char* model_name, RequestHandle* request_handle);

#if ENABLE_SPAN_ATTENTION
  int64_t GetFreeFrame(const char* model_name);
#endif

  AsEngineStat GetAsEngineStat(const char* model_name);

  std::string GetVersionFull();  // eg: 1.0.0(git_sha1-build_time)

  std::string GetOpProfilingInfo(const char* model_name);

  int GetRankId();
  int GetRankNums();

 private:
  // build model from saved allspark model files
  // many interface goes to here, but this interface only by self use.
  AsStatus BuildModel(const char* model_name, const std::string& model_proto,
                      std::shared_ptr<ModelWeightHandler> weight_handler,
                      const std::map<std::string, int>& model_limits = {});
  AsStatus WarmupModelInternal_(const char* model_name,
                                int64_t min_bytes_available,
                                std::vector<std::string>& fake_lora_names);
  AsStatus WarmupModel(const char* model_name);

  AsStatus SetNumThreads(int num_threads);
  AsStatus SetDeviceIds(const std::vector<int>& device_ids);
  AsStatus CreateDeviceContext(const std::string& compute_unit);
  void DestroyDeviceContext();
  AsStatus SetMatmulPrecision(const std::string& precision);
#if ENABLE_SPAN_ATTENTION
  AsStatus setSpanCacheConfig(AsCacheMode mode, int span_size,
                              int span_num_init, int span_num_grow);
#endif

  AsStatus RunTextGenerationContinue(const char* model_name);
  AsStatus RunTextGenerationContext(const char* model_name);

  AsStatus StartRequestImpl(const char* model_name,
                            std::shared_ptr<RequestHandle> request_handle,
                            GenerateConfig& gen_cfg);

  AsStatus StopRequestByRequestID(const char* model_name,
                                  std::string request_id,
                                  bool is_prefill_worker);
  AsStatus ReleaseRequestByRequestID(const char* model_name,
                                     const std::string& request_id,
                                     bool is_prefill_worker);

  template <typename T>
  int64_t GetInputBatch(const T& inputs);

  void UpdateAsEngineStat();

  // the main loop of a model name
  //
  void ModelRunningThread(std::string model_name,
                          std::shared_ptr<ModelControlState> model_state);

  void PrefillThread(std::string model_name,
                     std::shared_ptr<ModelControlState> model_state);

  void DecodeThread(std::string model_name,
                    std::shared_ptr<ModelControlState> model_state);

  void UpdateResult(std::string model_name,
                    std::shared_ptr<ModelControlState> model_state,
                    bool synchronizing = false,
                    std::shared_ptr<std::unordered_set<std::string>>
                        sync_pending_set_ptr = nullptr);

#if ENABLE_SPAN_ATTENTION
  void UpdateMinFreeFrame();
#endif

  // verify input params
  AsStatus InputParamsVerify(
      const char* model_name,
      std::shared_ptr<AsEngine::RequestContent>& request_info);

  AsStatus RichInputVerify(
      TensorListMap& extra_embedding,
      std::shared_ptr<AsEngine::RequestContent>& request_info);

  void ExpandRankThreadPool();

  std::string ChooseVictimRequest(const std::vector<std::string>& request_ids,
                                  const std::vector<int>& request_lens, int n);
  AsStatus RunEngineContext(std::string model_name);

  AsStatus RunPrefillWorker(std::string model_name);

  AsStatus RunDecodeWorker(std::string model_name,
                           std::shared_ptr<ModelControlState> model_state);

  bool is_device_id_set_ = false;
  // Both multi-gpu and multi-numa share the same variable;
  bool is_multi_nodes_;

  int nranks_ = 1;
  std::unique_ptr<AsEngineStat> as_stat_;
  std::vector<std::unique_ptr<Worker>> workers_;
  std::vector<std::unique_ptr<Worker>> workers_decode_;
  std::unique_ptr<DeviceContext> device_ctx_;
  std::unordered_map<std::string, std::unique_ptr<AsModel>> models_;
  std::unordered_map<std::string, std::unique_ptr<TransformerProto>> model_irs_;
  static std::unordered_map<std::string, int> precision_map_;

  // model control state is shared between the main thread and the target
  // model loop thread
  std::unordered_map<std::string, std::shared_ptr<ModelControlState>>
      model_state_map_;

  // TODO: engine needs support two or model can running within one engine.
  std::mutex engine_lock_;      // for async decoder lock
  std::mutex lora_lock_;        // mutex for lora WeightManager
  std::mutex lora_usage_lock_;  // mutex for loras_in_use_
  fs::path fake_lora_temp_dir_;
  int engine_max_length_ = 0;
  int engine_max_batch_ = 0;
  int engine_max_prefill_length_ = 0;
  const int engine_max_top_logprobs_ = 10;  // const
  int engine_gpu_swap_threshold_ = -1;
  // using thread pool instead of vector<std::thread> in text generation
  std::unique_ptr<ThreadPoolWithID> threadpool_;
  std::unique_ptr<ThreadPoolWithID> threadpool_decode_;
  int threadpool_size_{1};
  bool use_adaptive_cache_{false};
  std::mt19937 random_engine;

  std::shared_ptr<WeightManager> weight_manager_;
  std::unordered_map<std::string, std::multiset<std::string>> loras_in_use_;
  std::atomic<int> lora_use_count_;
  std::atomic<int64_t> min_free_frame_count_;

  PrefixCacheCoordinator::Ptr prefix_cache_coordinator_;
};
}  // namespace allspark
