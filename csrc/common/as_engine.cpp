/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    as_engine.cpp
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

using google::protobuf::Message;
using google::protobuf::io::FileInputStream;
const float SAMPLING_EPS = 1e-5;
const int warmup_input = 5;
/**
 * @brief Define this macro to enforce enable top-k and force K to be within (0,
 * 100]. Otherwise K is within [0, +inf).
 */
// #define CONFIG_ENFORCE_TOPK

// Check if in graceful stopping model mode. Deny any agressive request during
// stopping.
#define CHECK_MODEL_STOPPING(model_state)              \
  do {                                                 \
    if (model_state->model_stopping.load()) {          \
      LOG(INFO) << "model is stopping, access denied"; \
      return AsStatus::ALLSPARK_REQUEST_DENIED;        \
    }                                                  \
  } while (0)

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
  AsStatus RunTextGenerationContext(const char* model_name,
                                    bool is_new_context);

  AsStatus StartRequestImpl(const char* model_name,
                            std::shared_ptr<RequestHandle> request_handle,
                            DLTensorMap* outputs, GenerateConfig& gen_cfg);

  AsStatus StopRequestByRequestID(const char* model_name,
                                  std::string request_id);
  AsStatus ReleaseRequestByRequestID(const char* model_name,
                                     const std::string& request_id);

  template <typename T>
  int64_t GetInputBatch(const T& inputs);

  void UpdateAsEngineStat();

  // the main loop of a model name
  //
  void ModelRunningThread(std::string model_name,
                          std::shared_ptr<ModelControlState> model_state);
  void UpdateResult(std::string model_name,
                    std::shared_ptr<ModelControlState> model_state,
                    bool& synchronizing,
                    std::unordered_set<std::string>& sync_pending_set);
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

  bool is_device_id_set_ = false;
  // Both multi-gpu and multi-numa share the same variable;
  bool is_multi_nodes_;

  int nranks_ = 1;
  std::unique_ptr<AsEngineStat> as_stat_;
  std::vector<std::unique_ptr<Worker>> workers_;
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
  int threadpool_size_{1};
  bool use_adaptive_cache_{false};
  std::mt19937 random_engine;

  std::shared_ptr<WeightManager> weight_manager_;
  std::unordered_map<std::string, std::multiset<std::string>> loras_in_use_;
  std::atomic<int> lora_use_count_;

  PrefixCacheCoordinator::Ptr prefix_cache_coordinator_;
};

ModelControlState::ModelControlState(const std::string& name)
    : model_name(name), msg_queue(1000) {
  request_handle_map.reserve(1000);
  result_queue_map.reserve(1000);
  msg_queue_size.store(0);
}

static bool ReadProtoFromTextFile(const char* filename, Message* proto) {
  int fd = open(filename, O_RDONLY);
  CHECK_NE(fd, -1) << "File not found: " << filename;
  auto* input = new FileInputStream(fd);
  bool success = google::protobuf::TextFormat::Parse(input, proto);
  delete input;
  close(fd);
  return success;
}

static DeviceType GetDeviceTypeFromString(const std::string& device_type) {
  std::unordered_map<std::string, DeviceType> device_map(
      {{"CPU", DeviceType::CPU}, {"CUDA", DeviceType::CUDA}});
  if (device_map.find(device_type) == device_map.end()) {
    // LOG(ERROR) << "Invalid device_type:" << device_type << std::endl;
    return DeviceType::DEVICETYPE_UNDEFINED;
  }
  return device_map[device_type];
}

static std::pair<DeviceType, std::vector<int>> ParseDeviceType(
    const std::string& compute_unit) {
  size_t pos = compute_unit.find(':');
  if (pos == std::string::npos) {
    LOG(ERROR) << "Not Support ComputeUnit: " << compute_unit;
    throw std::invalid_argument("not support compute unit");
  }

  DeviceType device_type = GetDeviceTypeFromString(compute_unit.substr(0, pos));

  std::vector<int> device_ids;

  std::string remain = compute_unit.substr(pos + 1);
  std::stringstream ss(remain);
  std::string item;
  while (std::getline(ss, item, ',')) {
    device_ids.push_back(std::stoi(item));
  }
  return std::make_pair(device_type, device_ids);
}

AsEngineImpl::AsEngineImpl()
    : device_ctx_(std::make_unique<CPUContext>()),
      is_multi_nodes_(false),
      threadpool_size_(1),
      lora_use_count_(0) {
  util::as_init_log();
  // set threadpool_size_ to 1 for default to avoid additional overhead,
  // such as thread switching and lock contention in CPU streaming mode.
  threadpool_ = std::make_unique<ThreadPoolWithID>(threadpool_size_);

  device_ctx_->Init();

  weight_manager_ = WeightManager::Create();

  weight_manager_->RegisterWeightEventListener(
      [&](const std::shared_ptr<ModelWeightHandler>& handler,
          WeightEvent event) {
        if (event == WeightEvent::WeightOnLoad) {
          std::unique_lock<std::mutex> locker(engine_lock_);
          auto model_name = handler->GetModelConfig().model_name;
          if (model_state_map_.count(model_name) > 0) {
            model_state_map_[model_name]->weight_handler_ = handler;
          }
        }
      });

  std::random_device rand_dev;
  random_engine.seed(rand_dev());
  LOG(INFO) << "AllSpark Init with Version: " << GetVersionFull();
}

AsEngineImpl::~AsEngineImpl() {
  // we should wait for the running thread stop otherwise it will cause
  // exception.
  LOG(INFO) << "~AsEngine called";

  std::vector<std::string> pending_stop_model;

  {
    std::lock_guard<std::mutex> guard(engine_lock_);
    LOG(INFO) << "model_state_map_ size:" << model_state_map_.size();
    for (auto& model_state : model_state_map_) {
      if (!model_state.second->model_stopped) {
        LOG(INFO) << "Stopping model " << model_state.first;
        pending_stop_model.push_back(model_state.first);
      }
    }
  }

  for (auto& name : pending_stop_model) {
    StopModel(name.c_str());
    ReleaseModel(name.c_str());
  }

  // LOG(INFO) << "~Engine clear BFC Allocator ";
  //  free weight manager before destroy bfc.
  bool do_destroy_bfc = weight_manager_->GetNumModels() > 0;
  workers_.clear();
  models_.clear();
  weight_manager_.reset();
  if (do_destroy_bfc) {
    DestroyBFCAllocator();
  }
  LOG(INFO) << "~AsEngineImpl finished.";
}

std::string AsEngineImpl::GetVersionFull() {
  char buf[256];
  snprintf(buf, 256, "%s.%s.%s/(GitSha1:%s)", ALLSPARK_VERSION_MAJOR,
           ALLSPARK_VERSION_MINOR, ALLSPARK_VERSION_PATCH, kGitHash);
  return std::string(buf);
}

AsStatus AsEngineImpl::SetNumThreads(int num_threads) {
  DLOG(INFO) << "AsEngineImpl::SetNumThreads()" << std::endl;
  AsStatus ret = AsStatus::ALLSPARK_SUCCESS;
  device_ctx_->SetNumThreads(num_threads);

  std::future<AsStatus> result[nranks_];
  for (int i = 0; i < workers_.size(); ++i) {
    result[i] = threadpool_->enqueue(i, [this, i, &num_threads]() {
      return workers_[i]->SetNumThreads(num_threads);
    });
  }
  for (int i = 0; i < workers_.size(); ++i) {
    ret = result[i].get();
    AS_CHECK_STATUS(ret);
  }
  return AsStatus::ALLSPARK_SUCCESS;
}

std::unordered_map<std::string, int> AsEngineImpl::precision_map_({
    {"highest", PrecisionLevel::HIGHEST},
    {"high", PrecisionLevel::HIGH},
    {"medium", PrecisionLevel::MEDIUM_BF16},
    {"medium_bf16", PrecisionLevel::MEDIUM_BF16},
    {"medium_fp16", PrecisionLevel::MEDIUM_FP16},
});

AsStatus AsEngineImpl::SetMatmulPrecision(const std::string& precision) {
  DLOG(INFO) << "AsEngineImpl::SetMatmulPrecision()" << std::endl;
  if (precision_map_.find(precision) == precision_map_.end()) {
    LOG(ERROR) << "Invalid precision_type:" << precision << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }

  device_ctx_->SetMatmulPrecision(precision_map_[precision]);
  // TODO: possibly duplicated setting
  for (int i = 0; i < nranks_; ++i) {
    workers_[i]->GetDeviceContext()->SetMatmulPrecision(
        precision_map_[precision]);
  }
  return AsStatus::ALLSPARK_SUCCESS;
}

#if ENABLE_SPAN_ATTENTION
AsStatus AsEngineImpl::setSpanCacheConfig(AsCacheMode mode, int span_size,
                                          int span_num_init,
                                          int span_num_grow) {
  SpanCacheConfig::Ptr cache_config =
      SpanCacheConfig::Create(mode, span_size, span_num_init, span_num_grow);
  if (!cache_config) {
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }

  device_ctx_->SetCacheConfig(cache_config);
  return AsStatus::ALLSPARK_SUCCESS;
}
#endif

AsStatus AsEngineImpl::SetDeviceIds(const std::vector<int>& device_ids) {
  DLOG(INFO) << "AsEngineImpl::SetDeviceIds()" << std::endl;
  if (is_device_id_set_) {
    LOG(WARNING) << "WARNING: device_ids already set, ignored!" << std::endl;
    return AsStatus::ALLSPARK_SUCCESS;
  }
  if (device_ctx_ == nullptr) {
    LOG(WARNING) << "device type should be set first" << std::endl;
    return AsStatus::ALLSPARK_INVALID_CALL_ERROR;
  }

  DeviceType backend = device_ctx_->GetDeviceType();
  if (backend == DeviceType::CUDA) {  // tmp only support CUDA
    InitBFCAllocator(backend, device_ids);
  }

  nranks_ = device_ids.size();

  LOG(INFO) << "SetDeviceIds: DeviceIDs.size() " << device_ids.size();
  for (int i = 0; i < device_ids.size(); i++) {
    DLOG(INFO) << device_ids[i];
  }
  // 所有device inferer都走worker线程
  workers_.resize(nranks_);
#ifdef ENABLE_CUDA
  ncclUniqueId id;
  if (backend == DeviceType::CUDA) {
    ncclGetUniqueId(&id);
  }
#endif
  std::vector<std::thread> vthreads(nranks_);
  LOG(INFO) << "Start create " << nranks_ << " Device: " << backend
            << " workers.";
  // the only reason use multiple-thread here is init nccl requires multiple
  // process/thread be callled in current way.
  for (int i = 0; i < nranks_; ++i) {
    vthreads[i] = std::thread([&, i]() {
      switch (backend) {
#ifdef ENABLE_CUDA
        case DeviceType::CUDA: {
          workers_[i] =
              std::make_unique<CudaWorker>(i, nranks_, id, device_ids[i]);
          break;
        }
#endif
        case DeviceType::CPU: {
          workers_[i] = std::make_unique<CpuWorker>(i, nranks_, device_ids[i]);
          break;
        }
        default:
          LOG(ERROR) << "Unsupported device type: " << int(backend);
          break;
      }
      // cuda require multiple nccl client init in parallel, otherwise
      // will wait for other device.
      workers_[i]->Init();
      workers_[i]->InitCCL(i, nranks_);
      workers_[i]->SetWeightManager(weight_manager_);
    });
  }
  for (int i = 0; i < nranks_; ++i) {
    vthreads[i].join();
  }

  is_device_id_set_ = true;
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus AsEngineImpl::CreateDeviceContext(const std::string& compute_unit) {
  DLOG(INFO) << "AsEngineImpl::CreateDeviceContext()" << compute_unit
             << std::endl;
  DeviceType device_type = DeviceType::CUDA;
  std::vector<int> device_ids;

  try {
    std::tie(device_type, device_ids) = ParseDeviceType(compute_unit);
  } catch (std::invalid_argument& e) {
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }

  switch (device_type) {
    case DeviceType::CPU: {
      device_ctx_ = std::make_unique<CPUContext>();
      // device id is required by GPU like device,
      // cpu threads controler by numa control like cmd.
      AS_CHECK_STATUS(this->SetDeviceIds({0}));
      break;
    }
#ifdef ENABLE_CUDA
    case DeviceType::CUDA: {
      device_ctx_ = std::make_unique<CUDAContext>();
      AS_CHECK_STATUS(this->SetDeviceIds(device_ids));
      break;
    }
#endif
    default: {
      LOG(ERROR) << "Not Support ComputeUnit: " << compute_unit;
      return AsStatus::ALLSPARK_PARAM_ERROR;
    }
  }
  return AsStatus::ALLSPARK_SUCCESS;
}

void AsEngineImpl::DestroyDeviceContext() {
  is_device_id_set_ = false;
  DestroyBFCAllocator();
}

static void CheckAndOverridePrefillMode(AsModelConfig& model_config) {
  try {
    DeviceType device_type = DeviceType::CUDA;
    std::vector<int> device_ids;
    std::tie(device_type, device_ids) =
        ParseDeviceType(model_config.compute_unit);

    if (device_type == DeviceType::CPU) {
      if (CPUInfo::SupportAVX512F()) {
        LOG(INFO) << "Detect avx512f supported, switch Prefill mode to flash";
        model_config.prefill_mode = AsMHAPrefill::AsPrefillFlashV2;
      } else if (model_config.prefill_mode != AsMHAPrefill::AsPrefillDefault) {
        LOG(INFO) << "Warn: CPU only support Prefill model default";
        model_config.prefill_mode = AsMHAPrefill::AsPrefillDefault;
      }
    } else if (device_type == DeviceType::CUDA) {
#ifdef ENABLE_CUDA
      int sm_version = allspark::CUDAContext::GetStreamProcessorVersion(
          device_ids.size() > 0 ? device_ids[0] : 0);
      LOG(INFO) << "Auto Prefill selection, CUDA Detected, SM: " << std::hex
                << sm_version;
      if (sm_version >= CUDASMDef::SM_Ampere && CUDA_VERSION >= 11080) {
        model_config.prefill_mode = AsMHAPrefill::AsPrefillFlashV2;
        LOG(INFO) << "Prefill Auto Select: Ampler GPU detected, choose "
                  << " flashv2 as prefill flash.";
      } else if (sm_version >= CUDASMDef::SM_Volta) {
        model_config.prefill_mode = AsMHAPrefill::AsPrefillXformer;
        LOG(INFO) << "Prefill Auto Select: Volta GPU detected,"
                  << " choose Xformer as prefill flash.";
      } else {
        LOG(INFO)
            << "Prefill Auto Select: unknown SM detected, keep prefill as "
               "user setted..";
      }
#endif
    }

  } catch (std::invalid_argument& e) {
    LOG(INFO) << "Prefill Auto Select got exception, ignore this auto set. "
              << e.what();
  }
}

AsStatus AsEngineImpl::BuildModelFromConfigStruct(AsModelConfig& model_config) {
  EnvVarConfig env_config;

  // if running on GPU device, ignore the prefill select option,
  // override by auto select by device.
  // if running on cpu, check selection, only default is supported.
  CheckAndOverridePrefillMode(model_config);

  DLOG(INFO) << "AsEngineImpl::BuildModelFromConfigStruct()" << std::endl;
  LOG(INFO) << "Build model use following config:\n"
            << model_config.ToString() << std::endl;

  LOG(INFO) << "Memory Info:  BFC_ALLOCATOR: "
            << env_config.GetString("BFC_ALLOCATOR", "default:ON")
            << " BFC_MEM_RATIO: "
            << env_config.GetString("BFC_MEM_RATIO", "default:0.975");

  std::string model_path = model_config.model_path;
  LOG(INFO) << "Load model from : " << model_path << std::endl;

  if (model_path.empty() || !util::IsExists(model_path)) {
    LOG(ERROR) << "No such file or directory : " << model_path << std::endl;
    return AsStatus::ALLSPARK_IO_ERROR;
  }

  AS_CHECK_STATUS(this->CreateDeviceContext(model_config.compute_unit));
#if ENABLE_SPAN_ATTENTION
  if (device_ctx_->GetDeviceType() == DeviceType::CUDA) {
    AS_CHECK_STATUS(this->setSpanCacheConfig(
        model_config.cache_mode, model_config.cache_span_size,
        model_config.cache_span_num_init, model_config.cache_span_num_grow));
  }
#endif
  device_ctx_->SetPrefillMode(model_config.prefill_mode);
  device_ctx_->SetEvictionStrategy(model_config.eviction_strategy);
  device_ctx_->SetSparsityMatmulMode(model_config.enable_sparsity_matmul);
  device_ctx_->SetSchedulingStrategy(model_config.scheduling_strategy);
  if (model_config.num_threads != 0) {
    AS_CHECK_STATUS(this->SetNumThreads(model_config.num_threads));
  }
  AS_CHECK_STATUS(this->SetMatmulPrecision(model_config.matmul_precision));

  engine_max_length_ = model_config.engine_max_length;
  engine_max_batch_ = model_config.engine_max_batch;
  engine_max_prefill_length_ = model_config.engine_max_prefill_length;
  if (engine_max_length_ <= 2) {
    LOG(ERROR) << "Illegal egnine_max_length = " << engine_max_length_;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
  if (engine_max_batch_ <= 0) {
    LOG(ERROR) << "Illegal egnine_max_batch = " << engine_max_batch_;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
  if (engine_max_prefill_length_ < 0 ||
      engine_max_prefill_length_ > engine_max_length_) {
    LOG(ERROR) << "Illegal engine_max_prefill_length = "
               << engine_max_prefill_length_;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
#if ENABLE_SPAN_ATTENTION
  if (device_ctx_->GetDeviceType() == DeviceType::CUDA) {
    if (engine_max_prefill_length_ % device_ctx_->GetCacheSpanSize() != 0) {
      LOG(ERROR) << "Illegal engine_max_prefill_length = "
                 << engine_max_prefill_length_
                 << "need be a multiple of span_size";
      return AsStatus::ALLSPARK_PARAM_ERROR;
    }
  }
#endif
  if (engine_max_prefill_length_ > 0 && !model_config.enable_prefix_cache) {
    LOG(ERROR) << "Chunk Prefill only support in enable_prefix_cache=true";
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
  if (engine_max_prefill_length_ == 0) {  // don't use prefill chunk
    LOG(INFO) << "engine_max_prefill_length_ = 0 ,disabled chunk_prefill";
    engine_max_prefill_length_ = engine_max_length_;
  }
  std::shared_ptr<TransformerProto> model_ir =
      std::make_shared<TransformerProto>();

  if (model_config.text_graph) {
    Message* message = model_ir.get();
    if (!ReadProtoFromTextFile(model_path.c_str(), message)) {
      LOG(ERROR) << "Invalid text model format. model_path:" << model_path
                 << std::endl;
      return AsStatus::ALLSPARK_IO_ERROR;
    }
  } else {
    std::ifstream in(model_path);
    if (!model_ir->ParseFromIstream(&in)) {
      LOG(ERROR) << "Invalid binary model format. model_path:" << model_path
                 << std::endl;
      return AsStatus::ALLSPARK_IO_ERROR;
    }
  }

  std::shared_ptr<ModelWeightHandler> model_weight_handler;
  std::string weights, weights_path;
  if (model_config.weights_path.empty()) {
    LOG(ERROR) << "weights path not set , please check input param: "
                  "AsModelConfig: weights_path";
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
  DLOG(INFO) << "Load weights from : " << model_config.weights_path
             << std::endl;

#if 0
    std::ofstream output_file("graph.txt");

    std::string text;
    if (!google::protobuf::TextFormat::PrintToString(*model_ir, &text)) {
        std::cerr << "Failed to convert person to text format." << std::endl;
        // return -1;
    }

    // 输出字符串格式的 protobuf message
    output_file << text;
#endif

  try {
    model_weight_handler =
        weight_manager_->RegisterModel(model_config, model_ir);
    weight_manager_->CheckModelConsistency(model_weight_handler);
  } catch (AsModelException& e) {
    LOG(ERROR) << "Failed to register model file. " << strerror(errno);
    return AsStatus::ALLSPARK_IO_ERROR;
  }

  std::string model_proto;
  model_ir->SerializeToString(&model_proto);
  std::string& model_name = model_config.model_name;
  if (model_name.empty()) {
    LOG(ERROR)
        << "model name mot set, please check input param: AsModelConfig: "
           "model_name";
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }

  // drop the thread threshold function, use all or none policy.
  model_config.swap_threshold =
      std::min((int64_t)0, model_config.swap_threshold);
  util::SetSwapThreshold(model_config.swap_threshold);

  WeightSwapConfig swap_config;
  swap_config.enable = model_config.swap_threshold >= 0;

  weight_manager_->SetSwapConfig(model_weight_handler, swap_config);
  //--------- build model -------------//
  AS_CHECK_STATUS(
      this->BuildModel(model_name.c_str(), model_proto, model_weight_handler));
  // TODO: store model weight handler in model state, so the weight can be
  // freeed later.

  LOG(INFO) << "Build model success...";

  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus AsEngineImpl::BuildModel(
    const char* model_name, const std::string& model_proto,
    std::shared_ptr<ModelWeightHandler> weight_handler,
    const std::map<std::string, int>& model_limits) {
  DLOG(INFO) << "AsEngineImpl::BuildModel()" << std::endl;
  AsModelConfig model_config = weight_handler->GetModelConfig();
  std::unique_ptr<TransformerProto> model_ir =
      std::make_unique<TransformerProto>();
  model_ir->ParseFromString(model_proto);
  // LOG(INFO)<<model_ir->model_conf().num_heads()<<"
  // "<<model_ir->model_conf().size_per_head()<<"
  // "<<model_ir->model_conf().dec_layer();
  auto& graph = model_ir->graphs();
  device_ctx_->SetLoraEnabled(false);
  for (auto& g_name : model_ir->graph_names()) {  // search for LoRA op
    for (auto& op_proto : graph.at(g_name).ops()) {
      if (op_proto.op_type() == "GemmLoraCapsule") {
        device_ctx_->SetLoraEnabled(true);
        break;
      }
    }
  }
  if (device_ctx_->GetLoraEnabled()) {
    LOG(INFO) << "lora enabled";
  }

  device_ctx_->SetNumberHeads(model_ir->model_conf().num_heads());
  device_ctx_->SetNumberGroups(model_ir->model_conf().multi_query_group_num());
  device_ctx_->SetSizePerHead(model_ir->model_conf().size_per_head());
  device_ctx_->SetIntermediateSize(model_ir->model_conf().intermediate_size());
  device_ctx_->SetDecoderLayer(model_ir->model_conf().dec_layer());
  device_ctx_->SetDtype(model_ir->model_conf().dtype());
  device_ctx_->SetLoraMaxNum(model_config.lora_max_num);
  device_ctx_->SetLoraMaxRank(model_config.lora_max_rank);
  for (auto& item : model_limits) {
    if (item.second < 0) {
      LOG(ERROR) << "invalid engine limit param, should >= 0" << std::endl;
      return AsStatus::ALLSPARK_PARAM_ERROR;
    }
    if (item.first == "engine_max_length") engine_max_length_ = item.second;
    if (item.first == "engine_max_batch") engine_max_batch_ = item.second;
    if (item.first == "swap_threshold") util::SetSwapThreshold(item.second);
  }
  const char* var = std::getenv("ALLSPARK_KVCACHE_ALLOC_SIZE");
  if (var == nullptr) {
    device_ctx_->SetKVcacheSize(engine_max_length_);
  } else {
    int kvsize = std::atoi(var);
    if (kvsize > engine_max_length_) {
      LOG(ERROR) << "invalid ALLSPARK_KVCACHE_ALLOC_SIZE = " << kvsize
                 << ", should <= engine_max_length" << std::endl;
      return AsStatus::ALLSPARK_PARAM_ERROR;
    }
    if (kvsize == -1) {
      DLOG(INFO) << "ALLSPARK_KVCACHE_ALLOC_SIZE = -1 ,use engine_max_length"
                 << std::endl;
      device_ctx_->SetKVcacheSize(engine_max_length_);
    } else {
      device_ctx_->SetKVcacheSize(kvsize);
    }
  }
  const char* var2 = std::getenv("ALLSPARK_USE_TORCH_SAMPLE");
  if (var2 == nullptr) {
    device_ctx_->SetUseTorchSample(false);
  } else {
    int flag = std::atoi(var2);
    if (flag == 0) {
      DLOG(INFO) << "ALLSPARK_USE_TORCH_SAMPLE = 0 ,use normol sample"
                 << std::endl;
      device_ctx_->SetUseTorchSample(false);
    } else {
      DLOG(INFO) << "ALLSPARK_USE_TORCH_SAMPLE = 1 ,use torch sample"
                 << std::endl;
      device_ctx_->SetUseTorchSample(true);
    }
  }
  device_ctx_->SetModelMaxLength(engine_max_length_);
  device_ctx_->SetModelMaxBatch(engine_max_batch_);
  device_ctx_->SetModelMaxPrefillLength(engine_max_prefill_length_);
#if ENABLE_SPAN_ATTENTION
  if (device_ctx_->GetDeviceType() == DeviceType::CUDA) {
    // adaptive cache span growth
    if (device_ctx_->GetCacheSpanSize() != 0 &&
        device_ctx_->GetCacheSpanNumInit() == 0 &&
        device_ctx_->GetCacheSpanNumGrow() == 0) {
      LOG(INFO) << "BuildModel: using adaptive cache span settings";
      constexpr int kv_cache_count = 2;
      int warmup_single_batch_spans = (device_ctx_->GetModelMaxLength() +
                                       device_ctx_->GetCacheSpanSize() - 1) /
                                          device_ctx_->GetCacheSpanSize() +
                                      1;
      //  假设warmup的input_len为warmup_input
      //  由于engine_max_prefill_length的限制，每轮只能进行
      //  engine_max_prefill_length / warmup_input
      //  个请求，总共需要engine_max_batch个
      //  另外出于边界message不一定同步考虑，为了让最大同时running_batch_size=engine_max_batch，所以max_length额外+5最为保险，且不影响性能
      //  所以max_length不能少于warmup_input +
      //  engine_max_batch/(engine_max_prefill_length / warmup_input) + 5
      //  如果scheduling_strategy =
      //  allspark.AsSchedulingStrategy.Balance，将会使用过去版本的warmup策略，即warmup长度额外加上engine_max_batch
      int multi_batch_tokens =
          warmup_input +
          (device_ctx_->GetModelMaxBatch() /
               (device_ctx_->GetModelMaxPrefillLength() / warmup_input) +
           5);
      if (device_ctx_->GetSchedulingStrategy() !=
          AsSchedulingStrategy::ContextPriority) {
        multi_batch_tokens += device_ctx_->GetModelMaxBatch();
      }

      int warmup_multi_batch_spans =
          (((multi_batch_tokens + 32) + device_ctx_->GetCacheSpanSize() - 1) /
               device_ctx_->GetCacheSpanSize() +
           1) *
          device_ctx_->GetModelMaxBatch();
      int num_spans_per_seq =
          std::max(warmup_single_batch_spans, warmup_multi_batch_spans);
      int num_spans = kv_cache_count * device_ctx_->GetDecoderLayer() *
                      (num_spans_per_seq + 1);
      // reset cache config
      AS_CHECK_STATUS(this->setSpanCacheConfig(device_ctx_->GetCacheMode(),
                                               device_ctx_->GetCacheSpanSize(),
                                               num_spans, 0));
      use_adaptive_cache_ = true;
    }
  }
#endif

  LOG(INFO) << "Start BuildModel";

  ExpandRankThreadPool();
  // device_ctx_->SetDtype(model_ir->model_conf().dtype());
  std::vector<std::thread> vthreads(nranks_);
  std::vector<std::promise<AsStatus>> promise_vec(nranks_);
#if ENABLE_SPAN_ATTENTION
  if (device_ctx_->GetDeviceType() == DeviceType::CUDA) {
    AsModelConfig model_config = weight_handler->GetModelConfig();
    if (model_config.enable_prefix_cache) {
      prefix_cache_coordinator_ =
          std::make_shared<PrefixCacheCoordinator>(nranks_);
    }
  }
#endif

  for (int i = 0; i < nranks_; ++i) {
    // load models & weights
    vthreads[i] = std::thread([&, i]() {
      setThreadName(i, "ModelBuildThread");
      try {
        LOG(INFO) << "Start Build model for rank: " << i;
        AsStatus ret = AsStatus::ALLSPARK_SUCCESS;

        ret = workers_[i]->BuildModel(*model_ir, weight_manager_,
                                      weight_handler, device_ctx_.get(),
                                      prefix_cache_coordinator_);

        LOG(INFO) << "Finish Build model for rank, Set to Future: " << i << " Ret: " << (int)ret;
        promise_vec[i].set_value(ret);

      } catch (std::exception& e) {
        LOG(ERROR) << " build model: " << i
                   << " rank, build failed with exception: " << e.what();
        promise_vec[i].set_exception(std::current_exception());
      }
    });
  }

  // check the status of each task.
  AsStatus build_status = AsStatus::ALLSPARK_SUCCESS;
  for (int i = 0; i < nranks_; ++i) {
    try {
      auto status = promise_vec[i].get_future().get();
      if (status != AsStatus::ALLSPARK_SUCCESS) {
        LOG(INFO) << "Build Model task " << i << " Get From Future: " << (int)status;
        build_status = status;
      }
    } catch (std::exception& e) {
      LOG(ERROR) << "Build model failed with exception: " << e.what()
                 << " rank " << i;
      throw e;
    }
  }

  for (int i = 0; i < nranks_; ++i) {
    vthreads[i].join();
  }

  model_irs_[model_name] = std::move(model_ir);

  LOG(INFO) << "BuildModel Return  " << (int)build_status;

  return build_status;
}

AsStatus AsEngineImpl::UnloadModelFromDeviceMemory(const char* model_name) {
  DLOG(INFO) << "[" << model_name << "] "
             << "AsEngineImpl::UnloadModelFromDeviceMemory()" << std::endl;

  const char* use_bfc = std::getenv("BFC_ALLOCATOR");
  bool is_bfc_enabled = true;
  if (use_bfc) {
    is_bfc_enabled = std::string(use_bfc) != std::string("OFF");
  }

  const char* bfc_allow_growth = std::getenv("BFC_ALLOW_GROWTH");
  bool is_bfc_allow_growth =
      (!bfc_allow_growth || std::string(bfc_allow_growth) != "OFF");
  if (is_bfc_enabled && !is_bfc_allow_growth) {
    LOG(ERROR)
        << "[" << model_name << "] "
        << "Cannot unload device memory when using BFC allocator while "
           "BFC_ALLOW_GROWTH is OFF! You should export BFC_ALLOW_GROWTH=ON"
        << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }

  AsStatus ret = AsStatus::ALLSPARK_SUCCESS;
  std::future<AsStatus> result[nranks_];
  for (int i = 0; i < nranks_; ++i) {
    result[i] = threadpool_->enqueue(
        i, [this, i]() { return workers_[i]->UnloadModelFromDeviceMemory(); });
  }
  for (int i = 0; i < nranks_; ++i) {
    ret = result[i].get();
    AS_CHECK_STATUS(ret);
  }

  util::RegFreeMem();  // 释放注册表中残存的mem
  SweepBFCAllocator();  // 真正释放已free的内存，如果使用了BFCAllocator的话
  DLOG(INFO) << "[" << model_name << "] "
             << "AsEngineImpl::UnloadModelFromDeviceMemory() END" << std::endl;
  return AsStatus::ALLSPARK_SUCCESS;
}

AsFileInfo AsEngineImpl::GetFileInformation(const char* as_model_path,
                                            const char* as_param_path) {
  AsFileInfo ret_info;
  std::shared_ptr<TransformerProto> model_ir =
      std::make_shared<TransformerProto>();

  std::ifstream in(as_model_path);
  if (!model_ir->ParseFromIstream(&in)) {
    LOG(ERROR) << "Invalid binary model format. model_path:" << as_model_path
               << std::endl;
    throw std::invalid_argument("invalid path");
  }
  const BuildMetaProto& build_meta = model_ir->build_meta();
  AsParamGuard guard;
  std::string version_str;

  bool succ = guard.get_version(build_meta, version_str);

  if (!succ) {
    LOG(ERROR) << "Error on get graph version info";
    throw std::invalid_argument("no version info");
  }

  char buf[256];
  snprintf(buf, 256, "%s.%s.%s", ALLSPARK_VERSION_MAJOR, ALLSPARK_VERSION_MINOR,
           ALLSPARK_VERSION_PATCH);

  ret_info.create_version_param = version_str;
  ret_info.create_version_graph = version_str;
  ret_info.current_version_engine = buf;

  return ret_info;
}

AsStatus AsEngineImpl::TunePrefixCache(const char* model_name) {
#if ENABLE_SPAN_ATTENTION
  if (device_ctx_->GetDeviceType() == DeviceType::CUDA) {
    int token_per_span = device_ctx_->GetCacheSpanSize();
    int engine_max_length = engine_max_length_;
    int request_id = engine_max_batch_ + 1;
    int nranks = nranks_;

    for (int i = 0; i < nranks; ++i) {
      workers_[i]->ResetPrefixCache();
    }

    for (int cache_len = token_per_span;
         (cache_len + 1) < (engine_max_length - 2);
         cache_len *= 2, request_id++) {
      // prepare warmup request
      int seq_len = cache_len + 1;
      AsTensor in0("input_ids", DeviceType::CPU, DataType::INT64,
                   DataMode::DENSE, Shape({1, seq_len}));
      TensorUtils::Memset(in0, cache_len);
      const DLTensorMap warmup_inputs = {{"input_ids", in0.ToDLPack(nullptr)}};

      std::unique_ptr<GenerateConfig> warmup_cfg =
          std::make_unique<GenerateConfig>();
      warmup_cfg->max_length = seq_len + 1;
      warmup_cfg->top_k = 0;
      warmup_cfg->top_p = 0.5;

      std::shared_ptr<AsEngine::RequestContent> warmup_req =
          std::make_shared<AsEngine::RequestContent>();
      warmup_req->config = *(warmup_cfg);
      warmup_req->infer_type = AsEngine::RequestInferType::Generate;
      warmup_req->inputs = std::make_shared<DLTensorMap>(warmup_inputs);
      warmup_req->mm_type = AsEngine::RequestMMType::TextInput;

      float duration_in_ms[2] = {0};
      for (int j = 0; j < 2; j++) {
        warmup_req->config.uuid = "warmup_request_" +
                                  std::to_string(request_id) + "_" +
                                  std::to_string(j);
        RequestHandle* warmup_handle{nullptr};
        AsEngine::ResultQueue* warmup_queue{nullptr};

        auto start_time_point = std::chrono::steady_clock::now();
        AS_CHECK_STATUS(this->StartRequest(model_name, warmup_req,
                                           &warmup_handle, &warmup_queue));
        AS_CHECK_STATUS(this->SyncRequest(model_name, warmup_handle));
        AS_CHECK_STATUS(this->ReleaseRequest(model_name, warmup_handle));
        auto end_time_point = std::chrono::steady_clock::now();

        auto duration = end_time_point - start_time_point;
        duration_in_ms[j] =
            std::chrono::duration_cast<std::chrono::nanoseconds>(duration)
                .count() /
            1000000.0;
      }

      LOG(INFO) << __FUNCTION__ << ": "
                << "cache_len: " << cache_len << ", "
                << "duration_in_ms[0]: " << duration_in_ms[0] << " ms, "
                << "duration_in_ms[1]: " << duration_in_ms[1] << " ms";

      if (duration_in_ms[0] > duration_in_ms[1]) {
        for (int i = 0; i < nranks; ++i) {
          workers_[i]->SetPrefixCacheSeqlenThre(cache_len);
        }
        break;
      }
    }
  }
#endif
  return AsStatus::ALLSPARK_SUCCESS;
}

std::vector<std::string> AsEngineImpl::LoadFakeLoras(const char* model_name) {
  std::vector<std::string> ret;
  if (!device_ctx_->GetLoraEnabled()) {
    return ret;
  }

  char temp_dir[] = "/tmp/allspark_fake_lora-XXXXXX";
  AS_ENFORCE(mkdtemp(temp_dir) != nullptr);
  fake_lora_temp_dir_ = temp_dir;
  LOG(INFO) << "Successfully created fake lora temp dir: "
            << fake_lora_temp_dir_;

  auto num_heads = device_ctx_->GetNumberHeads();
  auto size_per_head = device_ctx_->GetSizePerHead();
  auto hidden_size = num_heads * size_per_head;
  auto num_groups = device_ctx_->GetNumberGroups();
  auto kv_size = num_groups * size_per_head;
  auto num_hidden_layers = device_ctx_->GetDecoderLayer();
  auto intermediate_size = device_ctx_->GetIntermediateSize();
  auto lora_max_num = device_ctx_->GetLoraMaxNum();
  auto lora_max_rank = device_ctx_->GetLoraMaxRank();
  auto dtype = device_ctx_->GetDtype();

  // create fake loras
  const std::string lora_base_name = "fake-lora-";
  fs::path aslora_path = fake_lora_temp_dir_ / (lora_base_name + "0.aslora");
  TensorMap tensor_map;
  char dtype_ch = dtype == DataType::FLOAT16 ? 'f' : 'b';
  std::map<std::string, TensorAttribute> attr_map{
      {"decoder.layer.__LAYER__.attention.self.lora_A.weight",
       {0,
        SplitMode::NOSPLIT,
        {hidden_size, 3 * lora_max_rank},
        {},
        dtype_ch,
        2,
        0}},
      {"decoder.layer.__LAYER__.attention.self.lora_B.weight",
       {0,
        SplitMode::GROUP_VSPLIT,
        {lora_max_rank, hidden_size + 2 * kv_size},
        {hidden_size, kv_size, kv_size},
        dtype_ch,
        2,
        0}},
      {"decoder.layer.__LAYER__.attention.output.dense.lora_A.weight",
       {0,
        SplitMode::HSPLIT,
        {hidden_size, lora_max_rank},
        {},
        dtype_ch,
        2,
        0}},
      {"decoder.layer.__LAYER__.attention.output.dense.lora_B.weight",
       {0,
        SplitMode::NOSPLIT,
        {lora_max_rank, hidden_size},
        {},
        dtype_ch,
        2,
        0}},
      {"decoder.layer.__LAYER__.ffn.intermediate.dense.lora_A.weight",
       {0,
        SplitMode::NOSPLIT,
        {hidden_size, lora_max_rank},
        {},
        dtype_ch,
        2,
        0}},
      {"decoder.layer.__LAYER__.ffn.intermediate.dense.lora_B.weight",
       {0,
        SplitMode::VSPLIT,
        {lora_max_rank, intermediate_size},
        {},
        dtype_ch,
        2,
        0}},
      {"decoder.layer.__LAYER__.ffn.linear.dense.lora_A.weight",
       {0,
        SplitMode::NOSPLIT,
        {hidden_size, lora_max_rank},
        {},
        dtype_ch,
        2,
        0}},
      {"decoder.layer.__LAYER__.ffn.linear.dense.lora_B.weight",
       {0,
        SplitMode::VSPLIT,
        {lora_max_rank, intermediate_size},
        {},
        dtype_ch,
        2,
        0}},
      {"decoder.layer.__LAYER__.ffn.output.dense.lora_A.weight",
       {0,
        SplitMode::HSPLIT,
        {intermediate_size, lora_max_rank},
        {},
        dtype_ch,
        2,
        0}},
      {"decoder.layer.__LAYER__.ffn.output.dense.lora_B.weight",
       {0,
        SplitMode::NOSPLIT,
        {lora_max_rank, hidden_size},
        {},
        dtype_ch,
        2,
        0}},
  };
  std::ofstream fout(aslora_path.string(), std::ios::out);  // 清空文件
  for (int layer = 0; layer < num_hidden_layers; layer++) {
    for (auto& [tensor_pattern, tensor_info] : attr_map) {
      std::string t_name = tensor_pattern;
      t_name.replace(t_name.find("__LAYER__"), strlen("__LAYER__"),
                     std::to_string(layer));
      auto nbytes = 1;
      for (auto size : tensor_info.shape) {
        nbytes *= size;
      }
      nbytes *= tensor_info.word_size;
      std::vector<char> bin_data(nbytes, 0);
      util::save_allsparky_tofile(aslora_path.string(), t_name, bin_data.data(),
                                  nbytes, tensor_info);
    }
  }
  util::set_global_header(aslora_path.string());  // 结束

  // load fake loras
  ret.emplace_back(lora_base_name + "0");
  AS_ENFORCE(AS_STATUS_OK(LoadLoraByName(model_name, aslora_path.c_str())));
  for (int i = 1; i < lora_max_num; i++) {
    auto lora_name = lora_base_name + std::to_string(i);
    auto symlink_path = fake_lora_temp_dir_ / (lora_name + ".aslora");
    symlink(aslora_path.c_str(), symlink_path.c_str());
    ret.emplace_back(lora_name);
    AS_ENFORCE(AS_STATUS_OK(LoadLoraByName(model_name, symlink_path.c_str())));
  }
  return ret;
}

void AsEngineImpl::UnloadFakeLoras(
    const char* model_name, const std::vector<std::string>& fake_lora_names) {
  if (!device_ctx_->GetLoraEnabled()) {
    return;
  }
  for (auto& lora_name : fake_lora_names) {
    AS_ENFORCE(AS_STATUS_OK(UnloadLoraByName(model_name, lora_name.c_str())));
  }
  try {
    fs::remove_all(fake_lora_temp_dir_);
  } catch (const std::exception& e) {
    LOG(ERROR) << "Failed to remove fake lora temp dir: " << fake_lora_temp_dir_
               << ", reason: " << e.what();
  } catch (...) {
    LOG(ERROR) << "Failed to remove fake lora temp dir: " << fake_lora_temp_dir_
               << ", unknown exception!";
  }
}

/*
 * send fake request to warm up engine, allocate necessary resources, like gpu
 * tensor, etc. send one max_length request, and N (max_engine_batch_size) times
 * request to make sure the coverage.
 */
AsStatus AsEngineImpl::WarmupModelInternal_(
    const char* model_name, int64_t min_bytes_available,
    std::vector<std::string>& fake_lora_names) {
  //* step 1: record memory usage before warmup
  std::vector<int64_t> bytes_limit(nranks_);
  std::vector<int64_t> bytes_before_warmup(nranks_);
  for (int i = 0; i < nranks_; ++i) {
    bytes_limit[i] = workers_[i]->GetTotalMemoryBytes();
    bytes_before_warmup[i] = workers_[i]->GetOccupiedMemoryBytes();
  }

  //* step 2: run one fake request of max_length
  AsTensor in0("input_ids", DeviceType::CPU, DataType::INT64, DataMode::DENSE,
               Shape({1, engine_max_length_ - 2}));
  TensorUtils::Memset(in0, 0);
  const DLTensorMap warmup_inputs = {{"input_ids", in0.ToDLPack(nullptr)}};

  std::string warmup_id = "warmup_request_0";
  std::unique_ptr<GenerateConfig> warmup_cfg =
      std::make_unique<GenerateConfig>();
  warmup_cfg->max_length = engine_max_length_;
  warmup_cfg->top_k = 0;
  warmup_cfg->top_p = 0.5;
  if (device_ctx_->GetLoraEnabled()) {
    warmup_cfg->lora_name = fake_lora_names[0];
  }

  std::shared_ptr<AsEngine::RequestContent> warmup_req =
      std::make_shared<AsEngine::RequestContent>();
  warmup_req->config = *(warmup_cfg);
  warmup_req->infer_type = AsEngine::RequestInferType::Generate;
  warmup_req->inputs = std::make_shared<DLTensorMap>(warmup_inputs);
  warmup_req->mm_type = AsEngine::RequestMMType::TextInput;

  RequestHandle* warmup_handle{nullptr};
  AsEngine::ResultQueue* warmup_queue{nullptr};
  AS_CHECK_STATUS(this->StartRequest(model_name, warmup_req, &warmup_handle,
                                     &warmup_queue, warmup_id));

  LOG(INFO) << "Warmup: start sync...";
  AS_CHECK_STATUS(this->SyncRequest(model_name, warmup_handle));
  if (warmup_queue->GenerateStatus() !=
      AsEngine::GenerateRequestStatus::GenerateFinished) {
    LOG(ERROR)
        << "AsEngineImpl::WarmupModelInternal_: warmup failed! Please check "
           "engine_max_length & engine_max_batch";
    return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }
  AS_CHECK_STATUS(this->ReleaseRequest(model_name, warmup_handle));

  //* step 3: run max_batch fake requests, input=5, output=engine_max_batch+10
  std::vector<RequestHandle*> warmup_handle_list;
  std::vector<AsEngine::ResultQueue*> warmup_queue_list;
  for (int i = 0; i < engine_max_batch_; i++) {
    AsTensor in0("input_ids", DeviceType::CPU, DataType::INT64, DataMode::DENSE,
                 Shape({1, warmup_input}));
    TensorUtils::Memset(in0, 0);

    const DLTensorMap warmup_inputs = {{"input_ids", in0.ToDLPack(nullptr)}};

    std::unique_ptr<GenerateConfig> warmup_cfg =
        std::make_unique<GenerateConfig>();

    /*
     * 假设warmup的input_len为warmup_input
     *
     * 由于engine_max_prefill_length的限制，每轮只能进行
     * engine_max_prefill_length / warmup_input
     * 个请求，总共需要engine_max_batch个
     *
     * 另外出于边界message不一定同步考虑，为了让最大同时running_batch_size=engine_max_batch，
     * 所以max_length额外+5最为保险，且不影响性能，所以max_length不能少于
     * warmup_input+engine_max_batch/(engine_max_prefill_length/warmup_input)+5
     *
     * 如果scheduling_strategy=allspark.AsSchedulingStrategy.Balance，
     * 将会使用过去版本的warmup策略，即warmup长度额外加上engine_max_batch
     */
    warmup_cfg->max_length =
        warmup_input +
        (device_ctx_->GetModelMaxBatch() /
             (device_ctx_->GetModelMaxPrefillLength() / warmup_input) +
         5);

    if (device_ctx_->GetSchedulingStrategy() !=
        AsSchedulingStrategy::ContextPriority) {
      warmup_cfg->max_length += engine_max_batch_;
    }
    std::string warmup_id = "warmup_request_" + std::to_string(i + 1);
    warmup_cfg->top_k = 0;
    warmup_cfg->top_p = 0.5;
    warmup_cfg->early_stopping = false;
    if (device_ctx_->GetLoraEnabled()) {
      warmup_cfg->lora_name = fake_lora_names[i % fake_lora_names.size()];
    }

    std::shared_ptr<AsEngine::RequestContent> warmup_req =
        std::make_shared<AsEngine::RequestContent>();
    warmup_req->config = *(warmup_cfg);
    warmup_req->infer_type = AsEngine::RequestInferType::Generate;
    warmup_req->inputs = std::make_shared<DLTensorMap>(warmup_inputs);
    warmup_req->mm_type = AsEngine::RequestMMType::TextInput;

    RequestHandle* warmup_handle{nullptr};
    AsEngine::ResultQueue* warmup_queue{nullptr};
    AS_CHECK_STATUS(this->StartRequest(model_name, warmup_req, &warmup_handle,
                                       &warmup_queue, warmup_id));
    warmup_handle_list.push_back(warmup_handle);
    warmup_queue_list.push_back(warmup_queue);
  }

  AS_CHECK_STATUS(this->SyncRequest(model_name, nullptr));
  for (int i = 0; i < engine_max_batch_; i++) {
    RequestHandle* warmup_handle = warmup_handle_list[i];
    AsEngine::ResultQueue* warmup_queue = warmup_queue_list[i];
    if (warmup_queue->GenerateStatus() !=
        AsEngine::GenerateRequestStatus::GenerateFinished) {
      LOG(ERROR)
          << "AsEngineImpl::WarmupModelInternal_: warmup failed! Please check "
             "engine_max_length & engine_max_batch";
      return AsStatus::ALLSPARK_RUNTIME_ERROR;
    }
    AS_CHECK_STATUS(this->ReleaseRequest(model_name, warmup_handle));
  }

  //* step 4: reset footprint of warmup requests
  // TunePrefixCache(model_name);
  for (int i = 0; i < nranks_; ++i) {
    workers_[i]->ResetProfiler();
#if ENABLE_SPAN_ATTENTION
    if (device_ctx_->GetDeviceType() == DeviceType::CUDA) {
      workers_[i]->ResetPrefixCache();
    }
#endif
  }

  //* step 5: record memory usage after warmup
  std::vector<int64_t> bytes_post_warmup(nranks_);
  for (int i = 0; i < nranks_; ++i) {
    bytes_post_warmup[i] = workers_[i]->GetOccupiedMemoryBytes();
  }

  std::vector<int64_t> bytes_runtime(nranks_);
  for (int i = 0; i < nranks_; ++i) {
    bytes_runtime[i] = bytes_post_warmup[i] - bytes_before_warmup[i];
    LOG(INFO) << "warm-up: device rank " << i
              << ", memory limit (MB): " << (bytes_limit[i] >> 20)
              << ", memory used before warm-up (MB): "
              << (bytes_before_warmup[i] >> 20)
              << ", memory used after warm-up (MB): "
              << (bytes_post_warmup[i] >> 20)
              << ", runtime memory consumption during warm-up (MB): "
              << (bytes_runtime[i] >> 20);
  }

  int64_t max_bytes_runtime =
      *std::max_element(bytes_runtime.cbegin(), bytes_runtime.cend());
  LOG(INFO) << "warm-up: maximum runtime memory consumption accross all "
               "devices (MB): "
            << (max_bytes_runtime >> 20);

  if (!use_adaptive_cache_) {
    // not using adaptive cache setting, just return
    return AsStatus::ALLSPARK_SUCCESS;
  }

  //* step 6: invoke worker warmup
  if (min_bytes_available > 0 && max_bytes_runtime >= 0) {
    int64_t min_bytes_limit =
        *std::min_element(bytes_limit.cbegin(), bytes_limit.cend());
    if (min_bytes_available > min_bytes_limit) {
      LOG(INFO) << "warm-up: min_bytes_available is larger than "
                   "min_bytes_limit accross all devices, using min_bytes_limit="
                << (min_bytes_limit >> 20) << " (MB) instead";
      min_bytes_available = min_bytes_limit;
    }

    LOG(INFO) << "warm-up: workers warming up...";
    ExpandRankThreadPool();

    std::future<AsStatus> result[nranks_];
    for (int i = 0; i < nranks_; ++i) {
      result[i] = threadpool_->enqueue(
          i,
          [this, i](int64_t _bytes_available, int64_t _bytes_runtime) {
            try {
              return workers_[i]->Warmup(_bytes_available, _bytes_runtime);
            } catch (std::exception& e) {
              LOG(ERROR) << "AsEngineImpl::WarmupModelInternal_: worker " << i
                         << " warmup failed: " << e.what();
              if (std::string(e.what()) == "ALLSPARK_MEMORY_ERROR" ||
                  std::string(e.what()) == "ALLSPARK_CACHE_MEMORY_OUT") {
                return AsStatus::ALLSPARK_MEMORY_ERROR;
              } else {
                return AsStatus::ALLSPARK_RUNTIME_ERROR;
              }
            }
          },
          min_bytes_available, max_bytes_runtime);
    }

    AsStatus ret = AsStatus::ALLSPARK_SUCCESS;
    for (int i = 0; i < nranks_; ++i) {
      AsStatus worker_ret = result[i].get();
      if (worker_ret != AsStatus::ALLSPARK_SUCCESS) {
        ret = worker_ret;
      }
    }

    if (ret == AsStatus::ALLSPARK_SUCCESS) {
      LOG(INFO) << "warm-up: all workers successfully finished!";
    }

    return ret;
  } else {
    LOG(WARNING) << "warm-up: invalid memory usage, min_bytes_available="
                 << min_bytes_available
                 << ", max_bytes_runtime=" << max_bytes_runtime
                 << ", worker warm-up is skipped";

    return AsStatus::ALLSPARK_SUCCESS;
  }
}

AsStatus AsEngineImpl::WarmupModel(const char* model_name) {
  if (EnvVarConfig::GetInt("ALLSPARK_DISABLE_WARMUP", 0) == 1) {
    return AsStatus::ALLSPARK_SUCCESS;
  }

  // generate and load fake loras upto limit
  auto fake_lora_names = LoadFakeLoras(model_name);

  // collect mem stats from all workers
  int64_t min_bytes_available = std::numeric_limits<int64_t>::max();
  if (use_adaptive_cache_) {
    std::future<int64_t> result[nranks_];
    for (int i = 0; i < nranks_; ++i) {
      result[i] = threadpool_->enqueue(i, [this, i]() -> int64_t {
        try {
          return workers_[i]->GetAvailableMemoryBytes();
        } catch (std::exception& e) {
          LOG(ERROR) << "StartModel: worker " << i
                     << " collect mem stats failed: " << std::string(e.what());
          return -1;
        }
      });
    }

    AsStatus failed_ret = AsStatus::ALLSPARK_SUCCESS;
    for (int i = 0; i < nranks_; ++i) {
      int64_t res = result[i].get();
      if (res < 0) {
        failed_ret = AsStatus::ALLSPARK_RUNTIME_ERROR;
      } else {
        min_bytes_available = std::min(min_bytes_available, res);
      }
    }

    if (failed_ret != AsStatus::ALLSPARK_SUCCESS) {
      UnloadFakeLoras(model_name,
                      fake_lora_names);  // just free fake loras if failed
      return failed_ret;
    } else {
      LOG(INFO) << "StartModel: min available device memory in MB"
                   "across all devices: "
                << min_bytes_available / (1024 * 1024);
    }
  }

  auto ret =
      WarmupModelInternal_(model_name, min_bytes_available, fake_lora_names);
  UnloadFakeLoras(model_name, fake_lora_names);
  return ret;
}

AsStatus AsEngineImpl::StartModel(const char* model_name) {
  util::as_init_log();

  // start model
  DLOG(INFO) << "[" << model_name << "] "
             << "AsEngineImpl::StartModel";
  // create a new model
  {
    auto name = std::string(model_name);
    as_stat_ = std::make_unique<AsEngineStat>(name);
    model_state_map_[name] = std::make_shared<ModelControlState>(name);
    model_state_map_[name]->StartLoop(&AsEngineImpl::ModelRunningThread, this,
                                      name, model_state_map_[name]);
  }

#if ENABLE_SPAN_ATTENTION
  if (device_ctx_->GetDeviceType() == DeviceType::CUDA) {
    if (device_ctx_->GetCacheSpanSize() == 0) {
      LOG(INFO) << "StartModel: span cache is disabled, skip warm-up";
      return AsStatus::ALLSPARK_SUCCESS;
    }
  }
#endif

  // start the thread pool
  ExpandRankThreadPool();

  // warmup
  auto ret = WarmupModel(model_name);
  return ret;
}

void AsEngineImpl::ExpandRankThreadPool() {
  if (nranks_ > threadpool_size_) {
    threadpool_size_ = nranks_ * 2;
    threadpool_ = std::make_unique<ThreadPoolWithID>(threadpool_size_);
  }
}

AsStatus AsEngineImpl::StopModel(const char* model_name) {
  LOG(INFO) << "[" << model_name << "] "
            << "StopModel";
  // notify loop thread to exit
  auto reply_promise = std::make_shared<std::promise<AsStatus>>();
  // TODO: possibly thread unsafe wrt ReleaseModel
  // TODO: this check is strange
  assert(model_state_map_[model_name].get() != nullptr);
  auto& model_state = model_state_map_[model_name];

  CHECK_MODEL_STOPPING(model_state);
  {
    model_state->model_stopping = true;
    auto msg = EngineControlMessage(EngineControlMessageId::GracefulStopModel,
                                    reply_promise);

    LOG(INFO) << "AsEngineImpl:: send model stop message.";
    bool succ = model_state->msg_queue.enqueue(std::move(msg));
    if (!succ) {
      LOG(ERROR) << "push message queue failed.";
    }
  }

  LOG(INFO) << "AsEngineImpl:: wait stop model return";
  auto ret = reply_promise->get_future().get();
  LOG(INFO) << "AsEngineImpl:: stop model got return.";

  if (ret != AsStatus::ALLSPARK_SUCCESS) {
    LOG(ERROR) << "[" << model_name << "] "
               << "StopModel failed with error " << (int)ret;
    return ret;
  } else {
    LOG(INFO) << "[" << model_name << "] "
              << "waiting to join loop thread";
    model_state->StopLoop();
    LOG(INFO) << "[" << model_name << "] "
              << "loop thread joined";
    // TODO: other necessary work
    // because we already stop model, release all the resources belongs to this
    // model.
    return AsStatus::ALLSPARK_SUCCESS;
  }
}

AsStatus AsEngineImpl::ReleaseModel(const char* model_name) {
  assert(model_state_map_[model_name].get() != nullptr);
  auto& model_state = model_state_map_[model_name];
  // because we assume call release model after stop model, this check is
  // make it's impossible be called.
  if (!model_state->model_stopped) {
    LOG(INFO) << "Model release without calling model stop, "
                 "please call model stop first!!!";
  }
  LOG(INFO) << "Release Model, intent to start a new model "
            << "so release all bfc memory";
  // try to release works, works contains models.
  // clear work can release ASModel.
  workers_.clear();

  {
    std::unique_lock<std::mutex> locker(engine_lock_);
    // free the weight belongs to this model.
    // XXX: because weight handler register is in ASModel, this weight
    // handler is null...
    weight_manager_->FreeWeight(model_state->weight_handler_);
    if (model_state->model_stopped) {
      model_state_map_[model_name]->StopLoop();
    } else {
      LOG(INFO) << "Model thread not stopped, stop thread may hang.";
      model_state_map_[model_name]->StopLoop();
    }
    // erase control state can clear all context.
    model_state_map_.erase(model_name);
  }
  // bfc can reclaim all gpu memory that not in use.
  DestroyDeviceContext();
  return AsStatus::ALLSPARK_SUCCESS;

  // TODO;
}

AsStatus AsEngineImpl::ReloadModelFromDeviceMemory(const char* model_name) {
  DLOG(INFO) << "[" << model_name << "] "
             << "AsEngineImpl::ReloadModelFromDeviceMemory()" << std::endl;

  const auto& model_ir = model_irs_[model_name];
  if (model_ir == nullptr) {
    LOG(ERROR) << "[" << model_name << "] "
               << "model_ir ptr is NULL" << std::endl;
    return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }

  // because each thread already bind cuda device, so we send msg to the
  // corresponding thread to ensure that we execute operations on the correct
  // device.

  AsStatus ret;
  std::future<AsStatus> result[nranks_];
  for (int i = 0; i < nranks_; ++i) {
    result[i] = threadpool_->enqueue(i, [this, i, &model_ir]() {
      return workers_[i]->RebuildModelFromBuffer(model_ir);
    });
  }
  for (int i = 0; i < nranks_; ++i) {
    ret = result[i].get();
    AS_CHECK_STATUS(ret);
  }

  DLOG(INFO) << "[" << model_name << "] "
             << "Finish AsModel::ReloadModelFromDeviceMemory()" << std::endl;
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus AsEngineImpl::GetModelInformation(const char* model_name,
                                           std::string* model_info) {
  DLOG(INFO) << "[" << model_name << "] "
             << "AsEngineImpl::GetModelInformation()" << std::endl;
  if (workers_.size() > 0 && workers_[0]->GetRank() == 0) {
    workers_[0]->GetInformation(model_info);
  } else {
    LOG(ERROR) << "[" << model_name << "] "
               << "workers is empty" << std::endl;
    return AsStatus::ALLSPARK_INVALID_CALL_ERROR;
  }
  return AsStatus::ALLSPARK_SUCCESS;
}

std::string AsEngineImpl::GetOpProfilingInfo(const char* model_name) {
  DLOG(INFO) << "[" << model_name << "] "
             << "AsEngineImpl::GetOpProfilingInfo()" << std::endl;
  std::string profiling_info;
  if (workers_.size() > 0) {
    profiling_info = workers_[0]->GetOpProfilingInfo();
  } else {
    LOG(ERROR) << "[" << model_name << "] "
               << "workers is empty" << std::endl;
  }
  return profiling_info;
}
// return rank id 0 in cuda; return rank id for cpu
int AsEngineImpl::GetRankId() {
  DLOG(INFO) << "AsEngineImpl::GetRandId()" << std::endl;
  int rank = 0;
  if (workers_.size() > 0) {
    rank = workers_[0]->GetRank();
  } else {
    LOG(ERROR) << "workers is empty" << std::endl;
  }
  return rank;
}

int AsEngineImpl::GetRankNums() {
  DLOG(INFO) << "AsEngineImpl::GetRankNums()" << std::endl;
  int rank = 0;
  if (workers_.size() > 0) {
    rank = workers_[0]->GetRankNums();
  } else {
    LOG(ERROR) << "workers is empty" << std::endl;
  }
  return rank;
}

// -----------------------------------------------------------------
// User Side Code Begin
// Following code may called in another thread.
// ------------------------------------------------------------------

AsStatus AsEngineImpl::StartRequest(
    const char* model_name,
    std::shared_ptr<AsEngine::RequestContent> request_info,
    RequestHandle** request_handle, AsEngine::ResultQueue** queue,
    const std::string customized_uuid) {
  std::string uuid = GenNewUUID();
  if (customized_uuid.size() > 0) {
    uuid = customized_uuid;
    // XXX: customized uuid only use for warm up request.
  }
  std::lock_guard<std::mutex> lora_guard(lora_lock_);
  TracerLog trace(device_ctx_->GetDeviceType(), "StartRequest", 1);

  if (request_info) {
    LOG(INFO) << "[" << model_name << "] "
              << "StartRequest Received: " << to_string(*request_info)
              << " uuid: " << uuid;
  } else {
    auto ret = AsStatus::ALLSPARK_PARAM_ERROR;
    LOG(ERROR) << "[" << model_name << "] "
               << "StartRequest with null request_info " << (int)ret;
    return ret;
  }

  // verify input param.
  auto ret = InputParamsVerify(model_name, request_info);
  if (ret != AsStatus::ALLSPARK_SUCCESS) {
    LOG(ERROR) << "[" << model_name << "] "
               << "StartRequest failed with error " << (int)ret;
    return ret;
  }

  TensorListMap extra_embedding;
  ExtraEmbeddingUtils::ParseMmInfo(extra_embedding,
                                   request_info->config.mm_info);

  ret = RichInputVerify(extra_embedding, request_info);
  if (ret != AsStatus::ALLSPARK_SUCCESS) {
    LOG(ERROR) << "[" << model_name << "] "
               << "StartRequest failed with rich input varify error"
               << (int)ret;
    return ret;
  }
  auto lora_name = request_info->config.lora_name;
  if (!lora_name.empty()) {
    LOG(INFO) << "req lora_name=" << lora_name;
    if (!workers_[0]->GetModel()->GetLoraManager()->IsLoraExists(lora_name)) {
      LOG(ERROR) << "LoRA " << lora_name << " not found, cannot StartRequest!";
      return AsStatus::ALLSPARK_LORA_NOT_FOUND;
    }

    std::lock_guard<std::mutex> lora_guard(lora_usage_lock_);
    if (!lora_name.empty()) {
      lora_use_count_++;
      loras_in_use_[model_name].insert(lora_name);
    }
  }

  auto reply_promise = std::make_shared<std::promise<AsStatus>>();

  // replace uuid id with new.
  auto handle = std::make_shared<RequestHandle>();

  handle->request_uuid = uuid;

  // todo: do some protection of this input....
  handle->context_length =
      (*request_info->inputs)["input_ids"]->dl_tensor.shape[1];

  // copy input from user's dltensor to our as tensor.
  // to avoid manage dltensor 's reference
  // store this in request handle to hide from user

  handle->inputs_internal = TensorUtils::DeepCopyDLTensorMapToTensorMap(
      request_info->inputs, DeviceType::CPU);
  handle->mm_type_internal = request_info->mm_type;

  handle->mm_embedding_internal = extra_embedding;

#ifdef ENABLE_JSON_MODE
  if (request_info->config.response_format.count("type") &&
      request_info->config.response_format["type"] == "json_object") {
    if (util::FormatEnforcer::vocab_.empty() &&
        request_info->config.vocab.empty()) {
      // no tokenizer found in model path, disable guided decoding
      request_info->config.response_format["type"] = "";
    } else {
      std::string schema =
          request_info->config.response_format.find("json_schema") !=
                  request_info->config.response_format.end()
              ? request_info->config.response_format["json_schema"]
              : "";
      DLOG(INFO) << "Request:" << handle->request_uuid
                 << " building parser using JSON Schema:\n"
                 << schema
                 << "\nVocabType:" << (int)request_info->config.vocab_type
                 << "\n";
      handle->format_enforcer = std::make_shared<util::FormatEnforcer>(
          request_info->config.vocab, schema, request_info->config.vocab_type,
          request_info->config.eos_token_id);
    }
  }
#endif

  auto new_queue = std::make_shared<ResultQueueImpl>(uuid);

  assert(model_state_map_[model_name].get() != nullptr);
  auto& model_state = model_state_map_[model_name];
  CHECK_MODEL_STOPPING(model_state);
  {
    // do sync in cpu multi processes
#ifndef ENABLE_CUDA
    workers_[0]->GetDeviceContext()->SemPostInterProcess();
#endif
    auto msg = EngineControlMessage(EngineControlMessageId::StartRequest,
                                    reply_promise, uuid, handle, new_queue,
                                    request_info);
    model_state->msg_queue.enqueue(std::move(msg));

    // create result queue & handle
  }

#ifndef ENABLE_CUDA
  workers_[0]->GetDeviceContext()->SemWaitSendInterProcess();
#endif
  DLOG(INFO) << "[" << model_name << "] "
             << "request with uuid " << handle->request_uuid << " notified";

  // this is safe because this scope holds shared ptrs
  *request_handle = handle.get();
  *queue = new_queue.get();

  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus AsEngineImpl::StopRequest(const char* model_name,
                                   RequestHandle* request_handle) {
  std::lock_guard<std::mutex> lora_guard(lora_lock_);
  // TODO: param check
  if (!request_handle) {
    LOG(ERROR) << "[" << model_name << "] "
               << "StopRequest: request_handle cannot be nullptr";
    return AsStatus::ALLSPARK_EMPTY_REQUEST;
  }

  TracerLog t(device_ctx_->GetDeviceType(), "StopRequest", 1);
  // put request to queue.
  // wait the the promise to be filled.
  DLOG(INFO) << "[" << model_name << "] "
             << "StopRequest";

  auto reply_promise = std::make_shared<std::promise<AsStatus>>();
  std::string uuid;

  assert(model_state_map_[model_name].get() != nullptr);
  auto& model_state = model_state_map_[model_name];
  // CHECK_MODEL_STOPPING(model_state);  // should allow to StopRequest
#ifndef ENABLE_CUDA
  workers_[0]->GetDeviceContext()->SemPostInterProcess();
#endif
  {
    uuid = request_handle->request_uuid;
    auto msg = EngineControlMessage(EngineControlMessageId::StopRequest,
                                    reply_promise, uuid);
    model_state->msg_queue.enqueue(std::move(msg));
  }
#ifndef ENABLE_CUDA
  workers_[0]->GetDeviceContext()->SemWaitSendInterProcess();
#endif
  auto ret = reply_promise->get_future().get();
  if (ret == AsStatus::ALLSPARK_SUCCESS) {
    LOG(INFO) << "[" << model_name << "] "
              << "StopRequest success with uuid: " << uuid;
  } else {
    LOG(ERROR) << "[" << model_name << "] "
               << "StopRequest failed with error " << (int)ret;
  }
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus AsEngineImpl::ReleaseRequest(const char* model_name,
                                      RequestHandle* request_handle) {
  std::lock_guard<std::mutex> lora_guard(lora_lock_);
  // TODO: param check
  if (!request_handle) {
    LOG(ERROR) << "[" << model_name << "] "
               << "ReleaseRequest: request_handle cannot be nullptr";
    return AsStatus::ALLSPARK_EMPTY_REQUEST;
  }

  TracerLog t(device_ctx_->GetDeviceType(), "ReleaseRequest", 1);
  // put request to queue.
  // wait the the promise to be filled.
  LOG(INFO) << "[" << model_name << "] "
            << "ReleaseRequest received, uuid: "
            << request_handle->request_uuid;

  auto reply_promise = std::make_shared<std::promise<AsStatus>>();
  std::string uuid;

  assert(model_state_map_[model_name].get() != nullptr);
  auto& model_state = model_state_map_[model_name];
  // CHECK_MODEL_STOPPING(model_state);  // allow ReleaseRequest
#ifndef ENABLE_CUDA
  workers_[0]->GetDeviceContext()->SemPostInterProcess();
#endif
  {
    uuid = request_handle->request_uuid;
    auto msg = EngineControlMessage(EngineControlMessageId::ReleaseRequest,
                                    reply_promise, uuid);
    model_state->msg_queue.enqueue(std::move(msg));
  }

  auto ret = reply_promise->get_future().get();
  if (ret == AsStatus::ALLSPARK_SUCCESS) {
    LOG(INFO) << "[" << model_name << "] "
              << "ReleaseRequest success with uuid: " << uuid;
  } else {
    LOG(ERROR) << "[" << model_name << "] "
               << "ReleaseRequest failed with error " << (int)ret;
  }
#ifndef ENABLE_CUDA
  workers_[0]->GetDeviceContext()->SemWaitSendInterProcess();
#endif
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus AsEngineImpl::SyncRequest(const char* model_name,
                                   RequestHandle* request_handle) {
  std::lock_guard<std::mutex> lora_guard(lora_lock_);
  // TODO: param check

  LOG(INFO) << "[" << model_name << "] "
            << "SyncRequest: "
            << (request_handle == nullptr ? "all"
                                          : request_handle->request_uuid);

  auto reply_promise = std::make_shared<std::promise<AsStatus>>();
  std::string uuid;

  assert(model_state_map_[model_name].get() != nullptr);
  auto& model_state = model_state_map_[model_name];
  // CHECK_MODEL_STOPPING(model_state);  // allow SyncRequest
#ifndef ENABLE_CUDA
  workers_[0]->GetDeviceContext()->SemPostInterProcess();
#endif
  if (request_handle) {
    // sync one request
    uuid = request_handle->request_uuid;
    auto msg = EngineControlMessage(EngineControlMessageId::SyncRequest,
                                    reply_promise, uuid);
    model_state->msg_queue.enqueue(std::move(msg));
  } else {
    //    LOG(INFO) << "Sync request must have a handler.";
    //    return AsStatus::ALLSPARK_PARAM_ERROR;
    // sync all requests, msg.request_handle carries nullptr
    uuid = "<ALL>";
    auto msg = EngineControlMessage(EngineControlMessageId::SyncAllRequest,
                                    reply_promise, uuid);
    model_state->msg_queue.enqueue(std::move(msg));
  }
#ifndef ENABLE_CUDA
  workers_[0]->GetDeviceContext()->SemWaitSendInterProcess();
#endif
  auto ret = reply_promise->get_future().get();

  if (ret == AsStatus::ALLSPARK_SUCCESS) {
    DLOG(INFO) << "[" << model_name << "] "
               << "SyncRequest success with uuid: " << uuid;
  } else {
    LOG(ERROR) << "[" << model_name << "] "
               << "SyncRequest failed with error " << (int)ret;
  }
  DLOG(INFO) << "[" << model_name << "] "
             << "SyncRequest promise ref count " << reply_promise.use_count();
  return ret;
}

// -----------------------------------------------------------------
// User Side Code  End
// ------------------------------------------------------------------

AsStatus AsEngineImpl::LoadLoraByName(const char* model_name,
                                      const char* lora_name) {
  std::lock_guard<std::mutex> lora_guard(lora_lock_);
  DLOG(INFO) << "before load_lora " << lora_name
             << ", free space=" << workers_[0]->GetAvailableMemoryBytes();

  if (!lora_name || strlen(lora_name) == 0) {
    LOG(ERROR) << "[" << model_name << "] "
               << "LoadLoraByName: Invalid lora_name ";
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
  if (workers_[0]->GetModel()->GetLoraManager()->IsLoraExists(lora_name)) {
    LOG(ERROR) << "LoRA " << lora_name << " already loaded, unload it first!";
    return AsStatus::ALLSPARK_LORA_ALREADY_LOADED;
  }
  {
    std::lock_guard<std::mutex> lora_guard(lora_usage_lock_);
    if (loras_in_use_.count(model_name) &&
        loras_in_use_[model_name].count(lora_name)) {
      LOG(ERROR) << "LoRA " << lora_name << " in use, cannot load!";
      return AsStatus::ALLSPARK_LORA_IN_USE;
    }
  }

  assert(model_state_map_[model_name].get() != nullptr);
  auto& model_state = model_state_map_[model_name];
  CHECK_MODEL_STOPPING(model_state);

  AsStatus ret = AsStatus::ALLSPARK_SUCCESS;
  std::future<AsStatus> result[nranks_];
  for (int i = 0; i < nranks_; ++i) {
    // TODO: verify model_name of workers_[i]
    result[i] = threadpool_->enqueue(i, [this, i, lora_name]() {
      return workers_[i]->LoadLoraByName(lora_name);
    });
  }
  for (int i = 0; i < nranks_; ++i) {
    ret = result[i].get();
    AS_CHECK_STATUS(ret);
  }
  LOG(INFO) << "after load_lora " << lora_name
            << ", free space=" << workers_[0]->GetAvailableMemoryBytes();
  workers_[0]->GetModel()->GetLoraManager()->PrintLoras();
  return ret;
}

AsStatus AsEngineImpl::UnloadLoraByName(const char* model_name,
                                        const char* lora_name) {
  DLOG(INFO) << "[" << model_name << "] "
             << "UnloadLoraByName: " << model_name << "::" << lora_name;
  std::lock_guard<std::mutex> lora_guard(lora_lock_);
  LOG(INFO) << "before unload_lora " << lora_name
            << ", free space=" << workers_[0]->GetAvailableMemoryBytes();
  workers_[0]->GetModel()->GetLoraManager()->PrintLoras();

  if (!lora_name || strlen(lora_name) == 0) {
    LOG(ERROR) << "[" << model_name << "] "
               << "UnloadLoraByName: Invalid lora_name ";
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
  if (!workers_[0]->GetModel()->GetLoraManager()->IsLoraExists(lora_name)) {
    LOG(ERROR) << "LoRA " << lora_name << " not found, cannot unload!";
    return AsStatus::ALLSPARK_LORA_NOT_FOUND;
  }
  {
    std::lock_guard<std::mutex> lora_guard(lora_usage_lock_);
    if (loras_in_use_.count(model_name) &&
        loras_in_use_[model_name].count(lora_name)) {
      LOG(ERROR) << "LoRA " << lora_name << " in use, cannot unload!";
      return AsStatus::ALLSPARK_LORA_IN_USE;
    }
  }

  assert(model_state_map_[model_name].get() != nullptr);
  auto& model_state = model_state_map_[model_name];
  CHECK_MODEL_STOPPING(model_state);

  AsStatus ret = AsStatus::ALLSPARK_SUCCESS;
  if (nranks_ > threadpool_size_) {
    threadpool_size_ = nranks_ * 2;
    threadpool_ = std::make_unique<ThreadPoolWithID>(threadpool_size_);
  }
  std::future<AsStatus> result[nranks_];
  for (int i = 0; i < nranks_; ++i) {
    // TODO: verify model_name of workers_[i]
    result[i] = threadpool_->enqueue(i, [this, i, lora_name]() {
      return workers_[i]->UnloadLoraByName(lora_name);
    });
  }
  for (int i = 0; i < nranks_; ++i) {
    ret = result[i].get();
    AS_CHECK_STATUS(ret);
  }

  workers_[0]->GetModel()->GetLoraManager()->PrintLoras();
  LOG(INFO) << "after unload_lora " << lora_name
            << ", free space=" << workers_[0]->GetAvailableMemoryBytes();
  return ret;
}

template <>
int64_t AsEngineImpl::GetInputBatch<DLTensorMap>(const DLTensorMap& inputs) {
  if (inputs.count("input_ids") == 0) throw AsException("ALLSPARK_PARAM_ERROR");
  return inputs.at("input_ids")->dl_tensor.shape[0];
}

template <>
int64_t AsEngineImpl::GetInputBatch<
    std::map<std::string, std::vector<std::vector<int64_t>>>>(
    const std::map<std::string, std::vector<std::vector<int64_t>>>& inputs) {
  if (inputs.count("input_ids") == 0) throw AsException("ALLSPARK_PARAM_ERROR");
  return inputs.at("input_ids").size();
}

AsStatus AsEngineImpl::RunTextGenerationContinue(const char* model_name) {
  DLOG(INFO) << "[" << model_name << "] "
             << "AsEngineImpl::RunTextGenerationContinue" << std::endl;
  std::lock_guard<std::mutex> guard(engine_lock_);
  // check model registered
  if (model_irs_[model_name] == nullptr) {
    LOG(ERROR) << "[" << model_name << "] "
               << "Invalid model name : " << model_name << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
  // verify params
  if (!model_irs_[model_name]->model_conf().is_generate()) {
    LOG(ERROR) << "[" << model_name << "] "
               << "RunTextGenerationContinue() is only supported in text "
                  "generation model.Please use RunModel() API."
               << std::endl;
    return AsStatus::ALLSPARK_INVALID_CALL_ERROR;
  }
  AsStatus ret = AsStatus::ALLSPARK_SUCCESS;
  std::future<AsStatus> result[nranks_];
  for (int i = 0; i < nranks_; ++i) {
    result[i] = threadpool_->enqueue(i, [this, i]() {
      try {
        return workers_[i]->AllocDecoderMemory();
      } catch (std::exception& e) {
        if (std::string(e.what()) == "ALLSPARK_MEMORY_ERROR" ||
            std::string(e.what()) == "ALLSPARK_CACHE_MEMORY_OUT") {
          return AsStatus::ALLSPARK_CACHE_MEMORY_OUT;
        } else {
          return AsStatus::ALLSPARK_RUNTIME_ERROR;
        }
      }
    });
  }
  for (int i = 0; i < nranks_; ++i) {
    ret = result[i].get();
    if (ret != AsStatus::ALLSPARK_SUCCESS) {
      LOG(ERROR) << "AllocDecoderMemory Failed!";
      return ret;
      // 任何一个子线程reshape阶段出问题都直接返回
    }
  }
  // LOG(INFO) << "RunTextGenerationContinue AllocDecoderMemory() pass";
  for (int i = 0; i < nranks_; ++i) {
    result[i] = threadpool_->enqueue(
        i, [this, i]() { return workers_[i]->RunTextGenerationContinue(); });
  }

  // 即使失败、异常，也要让各子线程运行完毕，以保证原子性。在可恢复的情况下，确保下一次请求有干净的环境
  AsStatus failed_ret = AsStatus::ALLSPARK_SUCCESS;
  for (int i = 0; i < nranks_; ++i) {
    try {
      ret = result[i].get();
      if (not AS_STATUS_OK(ret)) failed_ret = ret;
    } catch (std::exception& e) {
      if (std::string(e.what()) == "ALLSPARK_MEMORY_ERROR") {
        LOG(ERROR) << "AsEngineImpl::RunTextGenerationContinue: "
                      "exception caught: ALLSPARK_MEMORY_ERROR";
        throw AsException(("ALLSPARK_MEMORY_ERROR"));
      } else {
        AsSaveError(e.what());
        LOG(ERROR) << "AsEngineImpl::RunTextGenerationContinue: "
                      "exception caught: "
                   << e.what() << ", saved with AsSaveError";
        failed_ret = AsStatus::ALLSPARK_RUNTIME_ERROR;
      }
    }
  }
  return failed_ret;
}

AsStatus AsEngineImpl::RunTextGenerationContext(const char* model_name,
                                                bool is_new_context) {
  DLOG(INFO) << "[" << model_name << "] "
             << "AsEngineImpl::RunTextGenerationContext" << std::endl;
  std::lock_guard<std::mutex> guard(engine_lock_);
  DLOG(INFO) << "[" << model_name << "] "
             << "AsEngineImpl::RunTextGenerationContext mutex lock passed"
             << std::endl;

  TracerLog t(device_ctx_->GetDeviceType(), "RunCtx", 2);
  // check model registered
  if (model_irs_[model_name] == nullptr) {
    LOG(ERROR) << "[" << model_name << "] "
               << "Invalid model name : " << model_name << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
  // verify params
  if (model_irs_[model_name]->model_conf().is_generate() == false) {
    LOG(ERROR) << "[" << model_name << "] "
               << "RunTextGenerationContext() is only supported in text "
                  "generation model.Please use RunModel() API."
               << std::endl;
    return AsStatus::ALLSPARK_INVALID_CALL_ERROR;
  }
  AsStatus ret = AsStatus::ALLSPARK_SUCCESS;
  std::future<AsStatus> result[nranks_];
  for (int i = 0; i < nranks_; ++i) {
    result[i] = threadpool_->enqueue(i, [this, i, is_new_context]() {
      try {
        return workers_[i]->RunTextGenerationContext(is_new_context);
      } catch (std::exception& e) {
        LOG(ERROR) << "RunTextGenerationContext Failed:"
                   << std::string(e.what());
        if (std::string(e.what()) == "ALLSPARK_MEMORY_ERROR" ||
            std::string(e.what()) == "ALLSPARK_CACHE_MEMORY_OUT") {
          return AsStatus::ALLSPARK_CACHE_MEMORY_OUT;
        } else {
          return AsStatus::ALLSPARK_RUNTIME_ERROR;
        }
      }
    });
  }

  // 即使失败、异常，也要让各子线程运行完毕，以保证原子性。在可恢复的情况下，确保下一次请求有干净的环境
  AsStatus failed_ret = AsStatus::ALLSPARK_SUCCESS;
  for (int i = 0; i < nranks_; ++i) {
    ret = result[i].get();
    if (ret != AsStatus::ALLSPARK_SUCCESS) {
      failed_ret = ret;
    }
  }
  return failed_ret;
}
#if ENABLE_SPAN_ATTENTION
int64_t AsEngineImpl::GetFreeFrame(const char* model_name) {
  DLOG(INFO) << "[" << model_name << "] "
             << "AsEngineImpl::GetFreeFrame" << std::endl;
  return workers_[0]->GetFreeFrame();
}
#endif
AsEngineStat AsEngineImpl::GetAsEngineStat(const char* model_name) {
  DLOG(INFO) << "[" << model_name << "] "
             << "AsEngineImpl::GetAsEngineStat" << std::endl;
  return *as_stat_;
}
void AsEngineImpl::UpdateAsEngineStat() {
  DLOG(INFO) << "AsEngineImpl::UpdateAsEngineStat" << std::endl;
  workers_[0]->UpdateAsEngineStat(as_stat_.get());
  int64_t bytes_pre_warmup{0};
  int64_t bytes_limit{0};
#ifdef ENABLE_CUDA
  DeviceType backend = device_ctx_->GetDeviceType();
  if (backend == DeviceType::CUDA) {
    std::shared_ptr<Allocator> allocator = GetBFCAllocatorByDeviceId(
        workers_[0]->GetDeviceContext()->GetDeviceType(),
        workers_[0]->GetDeviceContext()->GetDeviceId());
    std::shared_ptr<BFCAllocator> bfc_allocator =
        std::dynamic_pointer_cast<BFCAllocator>(allocator);
    if (bfc_allocator != nullptr) {
      auto stats = bfc_allocator->GetStats();
      bytes_pre_warmup = stats.bytes_in_use;
      bytes_limit = stats.bytes_limit;
    } else {
      LOG(WARNING) << "UpdateAsEngineStat: BFCAllocator is not enabled";
    }
  }
#endif
  as_stat_->total_device_memory_pool_size = bytes_limit;
  as_stat_->used_device_memory_pool_size = bytes_pre_warmup;
}
AsStatus AsEngineImpl::StopRequestByRequestID(const char* model_name,
                                              std::string request_id) {
  DLOG(INFO) << "AsEngineImpl::StopRequestByRequestID" << std::endl;
  std::lock_guard<std::mutex> guard(engine_lock_);
  DLOG(INFO) << "AsEngineImpl::StopRequestByRequestID mutex lock passed"
             << std::endl;

  TracerLog t(device_ctx_->GetDeviceType(), "StopRequestById", 3);
  // check model registered
  if (model_irs_[model_name] == nullptr) {
    LOG(ERROR) << "Invalid model name : " << model_name << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }

  std::future<AsStatus> result[nranks_];
  for (int i = 0; i < nranks_; ++i) {
    result[i] = threadpool_->enqueue(i, [this, i, request_id]() {
      return workers_[i]->StopRequest(request_id);
    });
  }

  AsStatus ret = AsStatus::ALLSPARK_SUCCESS;
  // 即使失败、异常，也要让各子线程运行完毕，以保证原子性。在可恢复的情况下，确保下一次请求有干净的环境
  AsStatus failed_ret = AsStatus::ALLSPARK_SUCCESS;
  for (int i = 0; i < nranks_; ++i) {
    try {
      ret = result[i].get();
      if (not AS_STATUS_OK(ret)) failed_ret = ret;
    } catch (std::exception& e) {
      AsSaveError(e.what());
      LOG(ERROR) << "AsEngineImpl::StopRequestByRequestID: "
                    "exception caught: "
                 << e.what() << ", saved with AsSaveError";
      failed_ret = AsStatus::ALLSPARK_RUNTIME_ERROR;
    }
  }

  return failed_ret;
}

AsStatus AsEngineImpl::ReleaseRequestByRequestID(
    const char* model_name, const std::string& request_id) {
  DLOG(INFO) << "AsEngineImpl::ReleaseRequestByRequestID" << std::endl;
  std::lock_guard<std::mutex> guard(engine_lock_);
  DLOG(INFO) << "AsEngineImpl::ReleaseRequestByRequestID mutex lock passed"
             << std::endl;
  // check model registered
  if (model_irs_[model_name] == nullptr) {
    LOG(ERROR) << "Invalid model name : " << model_name << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }

  std::future<AsStatus> result[nranks_];
  for (int i = 0; i < nranks_; ++i) {
    result[i] = threadpool_->enqueue(i, [this, i, request_id]() {
      return workers_[i]->ReleaseRequest(request_id);
    });
  }

  AsStatus ret = AsStatus::ALLSPARK_SUCCESS;
  // 即使失败、异常，也要让各子线程运行完毕，以保证原子性。在可恢复的情况下，确保下一次请求有干净的环境
  AsStatus failed_ret = AsStatus::ALLSPARK_SUCCESS;
  for (int i = 0; i < nranks_; ++i) {
    try {
      ret = result[i].get();
      if (!AS_STATUS_OK(ret)) failed_ret = ret;
    } catch (std::exception& e) {
      AsSaveError(e.what());
      LOG(ERROR) << "AsEngineImpl::ReleaseRequestByRequestID: "
                    "exception caught: "
                 << e.what() << ", saved with AsSaveError";
      failed_ret = AsStatus::ALLSPARK_RUNTIME_ERROR;
    }
  }
  return failed_ret;
}

/**
 * verify rich input, if not fit in input, report error here.
 */
AsStatus AsEngineImpl::RichInputVerify(
    TensorListMap& extra_embedding,
    std::shared_ptr<AsEngine::RequestContent>& request_info) {
  if (!request_info) {
    LOG(ERROR) << "Input request info is null.";
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }

  auto input_dl_tensor = (*request_info->inputs)["input_ids"]->dl_tensor;
  // assume input id tensor is on cpu
  if (input_dl_tensor.device.device_type != DLDeviceType::kDLCPU) {
    LOG(ERROR) << "Input only support CPU Tensor.";
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }

  auto seq_len = input_dl_tensor.shape[1];
  int64_t* input_ids_ptr = static_cast<int64_t*>(input_dl_tensor.data);

  AsStatus rt = ExtraEmbeddingUtils::ParseExtraEmbedding(
      extra_embedding, input_ids_ptr, seq_len);
  return rt;
}

AsStatus AsEngineImpl::InputParamsVerify(
    const char* model_name,
    std::shared_ptr<AsEngine::RequestContent>& request_info) {
  auto& gen_cfg = request_info->config;
  auto& inputs = *request_info->inputs;
  // check model registered
  if (model_irs_[model_name] == nullptr) {
    LOG(ERROR) << "Invalid model name : " << model_name << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
  // verify params
  if (gen_cfg.bad_words_ids.size() != 0) {
    LOG(WARNING) << "[" << model_name << "] "
                 << "Not Support bad_words_ids" << std::endl;
  }
  if (model_irs_[model_name]->model_conf().is_generate() == false) {
    LOG(ERROR) << "[" << model_name << "] "
               << "StartRequest() is only supported in text "
                  "generation model.Please use RunModel() API."
               << std::endl;
    return AsStatus::ALLSPARK_INVALID_CALL_ERROR;
  }
  if (engine_max_length_ != 0 && gen_cfg.max_length > engine_max_length_) {
    LOG(ERROR) << "[" << model_name << "] "
               << "genconfig.max_length (" << gen_cfg.max_length << ") "
               << "> engine_max_length_ (" << engine_max_length_ << ")"
               << std::endl;
    return AsStatus::ALLSPARK_EXCEED_LIMIT_ERROR;
  }
  long input_length = (*request_info->inputs)["input_ids"]->dl_tensor.shape[1];
  // check if too long.
  if (engine_max_length_ != 0 && input_length > engine_max_length_ - 2) {
    LOG(ERROR) << "[" << model_name << "] "
               << "Too large input_len ! Input_len > engine_max_length_ - 2"
               << std::endl;
    return AsStatus::ALLSPARK_EXCEED_LIMIT_ERROR;
  }

  if (input_length <= 0) {
    LOG(ERROR) << "[" << model_name << "] "
               << "Invalid input request, input length == 0  " << std::endl;
    return AsStatus::ALLSPARK_EXCEED_LIMIT_ERROR;
  }

  // check if too short
  int64_t input_batch = GetInputBatch(inputs);
  if (gen_cfg.top_k < 0) {
    LOG(ERROR) << "[" << model_name << "] "
               << "gen_cfg.top_k can't < 0" << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
#ifdef CONFIG_ENFORCE_TOPK
  else if (gen_cfg.top_k > 100 || gen_cfg.top_k == 0) {
    LOG(WARNING) << "[" << model_name << "] "
                 << "genconfig.top_k=" << gen_cfg.top_k
                 << ",force modified to 100" << std::endl;
    gen_cfg.top_k = 100;
  }
#endif  // #ifdef CONFIG_ENFORCE_TOPK

  if (gen_cfg.top_p < 0 || gen_cfg.top_p > 1.0) {
    LOG(ERROR) << "[" << model_name << "] "
               << "gen_cfg.top_p must in [0,1]" << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }

  if (gen_cfg.temperature < SAMPLING_EPS) {
    DLOG(INFO) << "[" << model_name << "] "
               << "gen_cfg.temperature = " << gen_cfg.temperature
               << "use greedy sampling" << std::endl;
    gen_cfg.top_k = 1;
    gen_cfg.top_p = 0;
    gen_cfg.temperature = 1.0;
  }

  if (std::abs(gen_cfg.top_p - 1.0) < 1e-6) {
    LOG(WARNING) << "[" << model_name << "] "
                 << "gen_cfg.top_p == 1.0, This might lead to performance "
                    "issues, so it is manually set to 0.99. "
                 << std::endl;
    gen_cfg.top_p = 0.99;
  }
  // user customized max batch size
  if (engine_max_batch_ != 0 && input_batch > engine_max_batch_) {
    LOG(ERROR) << "[" << model_name << "] "
               << "" << input_batch << "(should <= " << engine_max_batch_ << ")"
               << std::endl;
    return AsStatus::ALLSPARK_EXCEED_LIMIT_ERROR;
  }
  // checkout top_logprobs
  if (gen_cfg.logprobs && gen_cfg.top_logprobs > engine_max_top_logprobs_) {
    LOG(ERROR) << "[" << model_name << "] "
               << "gen_cfg.top_logprobs=" << gen_cfg.top_logprobs
               << "(should <= " << engine_max_top_logprobs_ << ")" << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
  if (gen_cfg.repetition_penalty < 0.0 || gen_cfg.repetition_penalty > 2.0) {
    LOG(ERROR) << "[" << model_name << "] "
               << "gen_cfg.repetition_penalty must in [0,2]" << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
  if (gen_cfg.presence_penalty < -2.0 || gen_cfg.presence_penalty > 2.0) {
    LOG(ERROR) << "[" << model_name << "] "
               << "gen_cfg.presence_penalty must in [-2,2]" << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
  if (gen_cfg.frequency_penalty < -2.0 || gen_cfg.frequency_penalty > 2.0) {
    LOG(ERROR) << "[" << model_name << "] "
               << "gen_cfg.frequency_penalty must in [-2,2]" << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus AsEngineImpl::StartRequestImpl(
    const char* model_name, std::shared_ptr<RequestHandle> request_handle,
    DLTensorMap* outputs, GenerateConfig& gen_cfg) {
  DLOG(INFO) << "[" << model_name << "] "
             << "AsEngineImpl::RunTextGeneration" << std::endl;
  lock_guard_wrapper<std::mutex> guard(engine_lock_);
  DLOG(INFO) << "[" << model_name << "] "
             << "AsEngineImpl::RunTextGeneration mutex lock passed"
             << std::endl;

  TensorMap out_tensors;
  // TODO: alloc generated_ids_global on CPU
  std::string out_name = "generated_ids_global";
  out_tensors.insert(
      {out_name, std::make_shared<AsTensor>(out_name, DeviceType::CPU,
                                            DataType::INT64, DataMode::DENSE,
                                            Shape{1, engine_max_length_})});
  DLOG(INFO) << "[" << model_name << "] "
             << "AsEngineImpl::MultiGPU_StartRequestImpl()" << std::endl;
  std::future<AsStatus> result[nranks_];

  std::string& uuid = request_handle->request_uuid;

  for (int i = 0; i < nranks_; ++i) {
    result[i] = threadpool_->enqueue(
        i, [this, i, request_handle, uuid, &out_tensors, gen_cfg]() {
          return this->workers_[i]->StartRequestImpl(request_handle, uuid,
                                                     &out_tensors, gen_cfg);
        });
  }

  AsStatus failed_ret = AsStatus::ALLSPARK_SUCCESS;
  for (int i = 0; i < nranks_; ++i) {
    try {
      AsStatus ret;
      ret = result[i].get();
      if (not AS_STATUS_OK(ret)) failed_ret = ret;
    } catch (std::exception& e) {
      AsSaveError(e.what());
      LOG(ERROR) << "AsEngineImpl::StartRequestImpl: "
                    "exception caught: "
                 << e.what() << ", saved with AsSaveError";
      failed_ret = AsStatus::ALLSPARK_RUNTIME_ERROR;
    }
  }
  AsClearErrors();

  return failed_ret;
}

static std::shared_ptr<AsEngine::GeneratedElements>
FetchGenerationResultAndIncreaseCounter(
    Request* request, const std::shared_ptr<RequestHandle>& handle,
    ResultQueueImpl* result_queue) {
  std::shared_ptr<AsEngine::GeneratedElements> ele =
      std::make_shared<AsEngine::GeneratedElements>();

  ele->prefix_cache_len = request->prefix_len;
  ele->prefix_len_gpu = request->prefix_len_gpu;
  ele->prefix_len_cpu = request->prefix_len - request->prefix_len_gpu;

  const TensorMap& tmap = request->outputs;

  std::vector<std::vector<std::pair<int, float>>> log_probs_list =
      request->log_probs_list;
  auto device_tensor_ptr = tmap.at("generated_ids_global");
  if (device_tensor_ptr->GetShape().Count() == 0) {
    return nullptr;
  }
  // already in CPU
  auto host_tensor_ptr = device_tensor_ptr;
  assert(host_tensor_ptr->GetShape()[0] == 1);

  // put new element into queue.
  size_t new_length = host_tensor_ptr->GetShape()[1];
  size_t old_length = handle->context_length + handle->generate_length;
  if (new_length > old_length) {
    int64_t* raw_ptr = static_cast<int64_t*>(host_tensor_ptr->GetDataPtr());
    ele->ids_from_generate.reserve(new_length - old_length);

    for (int i = 0; i < new_length - old_length; i++) {
      bool is_dump_output =
          EnvVarConfig::GetInt("ALLSPARK_DUMP_OUTPUT_TOKEN", 0);
      if (is_dump_output) {
        for (auto id : ele->ids_from_generate)
          LOG(INFO) << "[DUMP] New Token > " << id
                    << "\t UUID: " << handle->request_uuid;
      }
      ele->ids_from_generate.push_back(raw_ptr[old_length + i]);
      if (request->gen_cfg.logprobs) {
        ele->log_probs_list.push_back(
            request->log_probs_list[handle->generate_length + i]);
        ele->token_logprobs_list.push_back(
            request->token_logprobs_list[handle->generate_length + i]);
      }
    }
    handle->generate_length = new_length - handle->context_length;
    request->generated_len = handle->generate_length;

    // update request queue stat counter.

    auto convert_to_long =
        [](const std::chrono::time_point<std::chrono::steady_clock>& time) {
          auto duration = time.time_since_epoch();
          long milliseconds =
              std::chrono::duration_cast<std::chrono::milliseconds>(duration)
                  .count();
          return milliseconds;
        };
    auto convert_to_duration =
        [](const std::chrono::time_point<std::chrono::steady_clock>& start,
           const std::chrono::time_point<std::chrono::steady_clock>& end) {
          auto duration = end - start;
          long ms =
              std::chrono::duration_cast<std::chrono::milliseconds>(duration)
                  .count();
          return ms > 0 ? ms : 0;
        };

    result_queue->UpdateStatInfo(stat_key::KEY_INPUT_LEN,
                                 handle->context_length);
    result_queue->UpdateStatInfo(stat_key::KEY_INPUT_CACHE_LEN,
                                 ele->prefix_cache_len);
    result_queue->UpdateStatInfo(stat_key::KEY_OUTPUT_LEN,
                                 handle->generate_length);

    result_queue->UpdateStatInfo(stat_key::KEY_REQUEST_TIME_TS,
                                 convert_to_long(request->enqueue_ts));
    result_queue->UpdateStatInfo(stat_key::KEY_FIRST_TOKEN_TIME_TS,
                                 convert_to_long(request->context_ts));
    result_queue->UpdateStatInfo(stat_key::KEY_LAST_TOKEN_TIME_TS,
                                 convert_to_long(request->generate_ts));

    result_queue->UpdateStatInfo(
        stat_key::KEY_SCHEDULE_TIME_MS,
        convert_to_duration(request->enqueue_ts, request->start_ts));
    result_queue->UpdateStatInfo(
        stat_key::KEY_TTFT_MS,
        convert_to_duration(request->enqueue_ts, request->context_ts));
    result_queue->UpdateStatInfo(
        stat_key::KEY_TOTAL_TIME_MS,
        convert_to_duration(request->start_ts, request->generate_ts));

    return ele;
  } else {
    // there is no tokens;
    return nullptr;
  }
}

void AsEngineImpl::UpdateResult(
    std::string model_name, std::shared_ptr<ModelControlState> model_state,
    bool& synchronizing, std::unordered_set<std::string>& sync_pending_set) {
  DLOG(INFO) << "UpdateResult:";
  auto updateSyncStatus =
      [model_name, &synchronizing, &sync_pending_set](
          const std::shared_ptr<RequestHandle>& handle,
          const std::shared_ptr<ResultQueueImpl>& out_queue_impl_ptr) -> bool {
    if (!synchronizing) {
      return false;
    }

    auto it = sync_pending_set.find(handle->request_uuid);
    if (it != sync_pending_set.end()) {
      sync_pending_set.erase(it);
      return true;
    }
    return false;
  };

  DLOG(INFO) << "request map size: " << model_state->request_handle_map.size();
  for (auto& entry : model_state->request_handle_map) {
    auto& handle = entry.second;
    if (!handle) {
      LOG(INFO) << "UpdateResult: ignore empty handle.";
      continue;
    }

    DLOG(INFO) << "UpdateResult: << uuid: " << handle->request_uuid;

    if (model_state->result_queue_map.count(handle->request_uuid) == 0) {
      LOG(ERROR) << "Update request cannot find request id queue: "
                 << handle->request_uuid;
      continue;
    }

    auto& out_queue = model_state->result_queue_map[handle->request_uuid];
    auto out_queue_impl_ptr =
        std::static_pointer_cast<ResultQueueImpl>(out_queue);

    // ignore finished & newly initialized, yet update sync status
    if (out_queue_impl_ptr->GenerateStatus() ==
            AsEngine::GenerateRequestStatus::GenerateFinished ||
        out_queue_impl_ptr->GenerateStatus() ==
            AsEngine::GenerateRequestStatus::GenerateInterrupted) {
      updateSyncStatus(handle, out_queue_impl_ptr);
      continue;
    }

    Request* request = workers_[0]->GetRequestById(handle->request_uuid);
    if (request == nullptr ||
        request->status == AsEngine::GenerateRequestStatus::Init) {
      continue;
    }

    if (request->status ==
        AsEngine::GenerateRequestStatus::GenerateInterrupted) {
      out_queue_impl_ptr->SetStatus(
          AsEngine::GenerateRequestStatus::GenerateInterrupted);
      updateSyncStatus(handle, out_queue_impl_ptr);
      continue;
    }
    // TODO: nearly impossible, but what if this overflows?
    handle->continue_count++;

    // needs to call continue once, otherwise output will be full of
    // random values.
    if (handle->continue_count == 1) {
      as_stat_->total_prefill_token += request->input_len;
      out_queue_impl_ptr->SetStatus(
          AsEngine::GenerateRequestStatus::Generating);
    }

    auto new_ele = FetchGenerationResultAndIncreaseCounter(
        request, handle, static_cast<ResultQueueImpl*>(out_queue.get()));
    bool is_finish = request->finish;
    if (new_ele != nullptr) {
      as_stat_->total_generated_token += new_ele->ids_from_generate.size();
      out_queue_impl_ptr->AppendGenerateElement(new_ele);
    }

    // check if finished & update sync status
    if (out_queue_impl_ptr->GenerateStatus() ==
        AsEngine::GenerateRequestStatus::GenerateInterrupted) {
      updateSyncStatus(handle, out_queue_impl_ptr);
    } else if (is_finish) {
      out_queue_impl_ptr->SetStatus(
          AsEngine::GenerateRequestStatus::GenerateFinished);
      LOG(INFO) << "[" << model_name << "] "
                << "request finished with uuid: " << handle->request_uuid;
      updateSyncStatus(handle, out_queue_impl_ptr);
    }

    // release lora_name from loras_in_use_
    if (request->status == AsEngine::GenerateRequestStatus::GenerateFinished) {
      if (lora_use_count_.load() > 0) {
        auto lora_name = request->gen_cfg.lora_name;
        std::lock_guard<std::mutex> lora_guard(lora_usage_lock_);
        if (loras_in_use_.count(model_name) and !lora_name.empty()) {
          loras_in_use_[model_name].extract(lora_name);
          lora_use_count_--;
        }
      }
    }
  }
}

static double BytesToMegabytes(size_t bytes) {
  const size_t bytesPerMegabyte = 1024 * 1024;
  return static_cast<double>(bytes) / bytesPerMegabyte;
}

static void PrintEngineStat(AsEngineStat& stat) {
  static int64_t last_gen_token = 0;
  static int64_t last_prompt_token = 0;

  static auto lastUpdateTime = std::chrono::steady_clock::now();
  auto currentTime = std::chrono::steady_clock::now();

  std::chrono::duration<double> elapsedSeconds = currentTime - lastUpdateTime;
  lastUpdateTime = currentTime;
  double diff_gen = stat.total_generated_token - last_gen_token;
  last_gen_token = stat.total_generated_token;
  double diff_prompt = stat.total_prefill_token - last_prompt_token;
  last_prompt_token = stat.total_prefill_token;

  double avg_prompt_tput = diff_prompt / elapsedSeconds.count();
  double avg_gen_tput = diff_gen / elapsedSeconds.count();

  int old_p = std::cout.precision();

  std::cout.precision(1);
  LOG(INFO) << "| AllSparkStat | Req: Running: " << stat.running_request
            << " Pending: " << stat.pendding_request << " \t| Token: ( "
            << stat.total_token - stat.free_token << " / " << stat.total_token
            << " ) "
            << " \t| Prompt: " << avg_prompt_tput << " T/s "
            << " Gen: " << avg_gen_tput << " T/s "
            << " \t| Mem(MB): ( "
            << BytesToMegabytes(stat.used_device_memory_pool_size) << " / "
            << BytesToMegabytes(stat.total_device_memory_pool_size) << " ) "
            << " \t| Prefix Cache:"
            << " Hit: " << stat.prefix_cache_hit_rate << " "
            << " Miss: " << stat.prefix_cache_miss_rate;

  std::cout.precision(old_p);
}

// a while(1) loop get from control command queue, if no msg from control,
// running the one decoding job.
// if decoding was all idle, wait on the control comamnd queue.
void AsEngineImpl::ModelRunningThread(
    std::string model_name, std::shared_ptr<ModelControlState> model_state) {
  std::string s = "ModelRunningThread";
#if 0
  pthread_setname_np(pthread_self(),
                     s.c_str());  // set the name (pthread_self() returns the
                                  // pthread_t of the current thread)
#endif
  bool looping = true;
  int continues_error_count = 0;  // accumulated error on run continue.
  const int k_main_loop_error_threshold =
      100;  // threshold to abort the whole process if keep in error state.

  bool graceful_stop_phase = false;
  bool graceful_final_released = false;
  EngineControlMessage graceful_stop_msg{};

  bool synchronizing = false;
  std::shared_ptr<std::promise<AsStatus>> deferred_promise{nullptr};
  std::unordered_set<std::string> sync_pending_set;
  // TODO: reserved for future use
  // std::unordered_map<std::string, AsEngine::GenerateRequestStatus>
  // sync_finished_map;

  using clock = std::chrono::steady_clock;
  auto next_log_time = clock::now();

  while (looping) {
    util::Timer time_outer;
    UpdateAsEngineStat();
    // print the engine state for easier service trace.
    if (clock::now() >= next_log_time) {
      int next_sec = EnvVarConfig::GetInt("HIE_LOG_STATUS_INTERVAL", 5);
      auto stat = GetAsEngineStat("todo");
      PrintEngineStat(stat);
      next_log_time += std::chrono::seconds(next_sec);
    }

    // decoupling message decoding phase and model execution phase
    bool no_execution = false;
    // Phase 1: message decoding
    // Pick one control message, handle control message, return control
    // message promise.
    // If synchronizing, block any message until finished.

    util::Timer t_msg_handle;

#ifndef ENABLE_CUDA
    // this message counter is only for multi-numa syncing.
    long processed_msg_counter = 0;
#endif

    if (!synchronizing) {
      TracerLog t(device_ctx_->GetDeviceType(), "LoopHandleMsg", 3);
      int cur_msg_size = model_state->msg_queue.size_approx();
      // XXX: don't check message size, after change to concurrent queue, this
      // may not accurate.

      while (!synchronizing) {
        // NOTE: the front is moved, do not use it anymore

        EngineControlMessage msg;

        // When there are new messages, take them out, and if the size is
        // greater than 0, process the new messages again. If there are no new
        // messages, and sync_pending_set equals 0, and it is not in the middle
        // of stop model, then skip message processing and continue executing
        // the main loop. If there are no new messages and the above check
        // fails, then enter the wait queue.

        if (model_state->msg_queue.size_approx() > 0) {
          bool got_new_message = model_state->msg_queue.try_dequeue(msg);
          if (!got_new_message) {
            LOG(ERROR) << "queue size > 0, but not no message";
            goto skip_message_process;
          } else {
            DLOG(INFO) << "[" << model_name << "] "
                       << "ModelRunningThread: receive message: "
                       << ToString(msg.msg);
          }
        } else {
          // message queue size == 0
          // TODO: add handle of gracefully stop.
          if (sync_pending_set.size() != 0) {
            LOG(INFO) << " pending syncing request, skip message process..";
            goto skip_message_process;
          }

          // Note: get unfinished request will require one model running to
          // change, so check the current management request too.
          if (!graceful_stop_phase &&
              workers_[0]->GetUnFinishedRequest() == 0 &&
              model_state->request_handle_map.size() == 0 &&
              !model_state->model_stopping.load()) {
            LOG(INFO) << "ModelRunningThread: EventLoop is going to suspend.";

            model_state->msg_queue.wait_dequeue(msg);

            LOG(INFO) << "ModelRunningThread: EventLoop is going to  resume.";

          } else {
            goto skip_message_process;
          }
        }

        LOG(INFO) << "[" << model_name << "] "
                  << "ModelRunningThread: receive message: "
                  << ToString(msg.msg);

        // dispatch message
        switch (msg.msg) {
          case EngineControlMessageId::GracefulStopModel: {
            graceful_stop_phase = true;
            model_state->model_stopping = true;
            graceful_stop_msg = std::move(
                msg);  // save, its promise will be filled when all done
            break;
          }
          case EngineControlMessageId::StopModel: {  // deprecated
            // stop the looper thread.
            looping = false;
            no_execution = true;
            msg.promise->set_value(AsStatus::ALLSPARK_SUCCESS);
            break;
          }
          case EngineControlMessageId::StartModel: {
            // TODO: like resume ?
            break;
          }
          case EngineControlMessageId::StartRequest: {
            if (graceful_stop_phase) {
              // blocked by API, should not enter here...
              no_execution = true;
              msg.promise->set_value(AsStatus::ALLSPARK_REQUEST_DENIED);
              break;
            }
            // start the request?
            DLOG(INFO) << "[" << model_name << "] "
                       << "RunTextGeneration";
            util::Timer t1;

            LOG(INFO) << "Start request add : " << msg.request_uuid
                      << " into queue map , ptr: " << msg.result_queue.get();

            auto ret_insert1 = model_state->result_queue_map.emplace(
                msg.request_uuid, msg.result_queue);
            auto ret_insert2 = model_state->request_handle_map.emplace(
                msg.request_uuid, msg.request_handle);
            if (!ret_insert1.second && !ret_insert2.second) {
              // this means the key already in the map.
              LOG(ERROR) << "[" << model_name << "] "
                         << " Start Request: with duplicated uuid: "
                         << msg.request_uuid;
            }

            auto request_handle_sp =
                model_state->request_handle_map.at(msg.request_uuid);
            auto ret =
                this->StartRequestImpl(model_name.c_str(), request_handle_sp,
                                       nullptr, msg.request->config);
            DLOG(INFO) << "[" << model_name << "] "
                       << "RunTextGeneration finish " << t1.elapsed() << " ms";
            if (ret != AsStatus::ALLSPARK_SUCCESS) {
              LOG(ERROR) << "ModelRunningThread: Start Request return failed: "
                         << " uuid: " << msg.request_uuid << " " << (int)ret;
            }
            // start request don't wait for reply from client side code.
            // msg.promise->set_value(ret);

            // since already async s tart request, no needs to set promise back.
            break;
          }
          case EngineControlMessageId::StopRequest: {
            auto uuid = msg.request_uuid;
            DLOG(INFO) << "[" << model_name << "] "
                       << "StopRequest: " << uuid;
            auto ret = this->StopRequestByRequestID(model_name.c_str(), uuid);
            msg.promise->set_value(ret);
            break;
            // TODO;
            // make the request is end?  and call api to stop it.
          }
          case EngineControlMessageId::SyncRequest: {
            // sync one request
            LOG(INFO) << "[" << model_name << "] "
                      << "SyncRequest: [Control] sync request with uuid: "
                      << msg.request_uuid;
            sync_pending_set.insert(msg.request_uuid);
            // promise will not be fulfilled until sync
            // finished
            deferred_promise = msg.promise;
            synchronizing = true;
            break;
          }
          case EngineControlMessageId::SyncAllRequest: {
            // sync all requests
            for (const auto& entry : model_state->request_handle_map) {
              const auto& handle = entry.second;
              sync_pending_set.insert(handle->request_uuid);
            }
            LOG(INFO) << "[" << model_name << "] "
                      << "SyncRequest: [Control] sync all requests, size: "
                      << sync_pending_set.size();
            // promise will not be fulfilled until sync finished
            deferred_promise = std::move(msg.promise);
            synchronizing = true;
            break;
          }

          case EngineControlMessageId::ReleaseRequest: {
            LOG(INFO) << "[" << model_name << "]"
                      << "Release Request received: " << msg.request_uuid;
            // before release , stop request
            // auto ret1 = this->StopRequestByRequestID(model_name.c_str(),
            //                                          handle_ptr->request_uuid);

            // auto lora_name = workers_[0]
            //                      ->GetRequestById(handle_ptr->request_uuid)
            //                      ->gen_cfg.lora_name;
            // if (loras_in_use_.count(model_name) and !lora_name.empty()) {
            //   loras_in_use_[model_name].erase(lora_name);
            // }
            auto uuid = msg.request_uuid;

            auto ret_stop =
                this->StopRequestByRequestID(model_name.c_str(), uuid);

            auto ret_release =
                this->ReleaseRequestByRequestID(model_name.c_str(), uuid);

            LOG(INFO) << "Release Request: remove uuid: " << uuid
                      << " From control state";

            if (model_state->result_queue_map.count(uuid) != 0) {
              model_state->result_queue_map.erase(uuid);
              model_state->request_handle_map.erase(uuid);
            } else {
              LOG(ERROR) << "Attempt to release a non-exist request: uuid: "
                         << uuid;
            }

            // check if it's the final msg
            if (graceful_stop_phase &&
                workers_[0]->GetUnFinishedRequest() == 0 &&
                model_state->msg_queue.size_approx() == 0) {
              LOG(INFO) << "graceful_stop_phase: no unfinished request";
              graceful_final_released = true;
            }

            DLOG(INFO) << "[" << model_name << "] "
                       << "Release request: release request with uuid: "
                       << uuid;

            if (ret_stop != AsStatus::ALLSPARK_SUCCESS)
              msg.promise->set_value(ret_stop);
            else
              msg.promise->set_value(ret_release);

            break;
          }
          default: {
            LOG(WARNING) << "[" << model_name << "] "
                         << "Warning: unhandle message received: "
                         << (int)msg.msg;
          }
        }

#ifndef ENABLE_CUDA
        processed_msg_counter++;
#endif
      }
    } else {
      DLOG(INFO) << "[" << model_name << "] "
                 << "skipping message handling due to synchronization";
    }

  skip_message_process:

    util::Timer t_msg_handle_finish;

    auto msg_handle_time_ms =
        util::Timer::duration_ms(t_msg_handle, t_msg_handle_finish);

    if (msg_handle_time_ms > 1) {
      // usually message handle cost around 0.01-0.1 ms
      LOG(INFO)
          << "ModelRunningThread: message handle cost too much, time(ms): "
          << msg_handle_time_ms;
    }

#ifndef ENABLE_CUDA
    no_execution = workers_[0]->GetDeviceContext()->SemWaitMsgSynInterProcess(
        processed_msg_counter);
#endif

    // Phase 2: model execution
    if (no_execution) {
      // this means the thread is called by stop model,should back
      // to the loop while check,and stop the thread.
      continue;
    }

    // Step 2.1: check running task, run the context step
    // This will automatically and internally call AsModel::StopRequest if a
    // request is naturally finished, and thus finishes the `Request`.
    util::Timer timer_cont;
    AsStatus status = RunEngineContext(model_name);

    UpdateResult(model_name, model_state, synchronizing, sync_pending_set);

    if (status != AsStatus::ALLSPARK_SUCCESS &&
        status != AsStatus::ALLSPARK_EMPTY_REQUEST) {
      LOG(ERROR) << "RunTextGenerationContext Failed:"
                 << AsGetErrorByCode(status);
    }
    // check running task, run the decoder step.
    while (true) {
      // LOG(INFO) << "RunTextGenerationContinue Start:";
      AsStatus status = this->RunTextGenerationContinue(model_name.c_str());
      if (status == AsStatus::ALLSPARK_SUCCESS ||
          status == AsStatus::ALLSPARK_EMPTY_REQUEST) {
        continues_error_count = 0;
        break;
      }

      LOG(ERROR) << "RunTextGenerationContinue Failed:"
                 << AsGetErrorByCode(status);
      if (status == AsStatus::ALLSPARK_CACHE_MEMORY_OUT) {
        // TOOD: instead of stop request and pop to upper service
        // we can do the recompute as a new request, so
        // it's more likely higher to archive higher performance.
        // 显存不足，从目前正在进行的request中，随机选择一个停止
        std::vector<std::string> request_ids;
        std::vector<int> request_lens;
        for (auto& entry : model_state->request_handle_map) {
          auto& handle = entry.second;
          auto& out_queue = model_state->result_queue_map[handle->request_uuid];
          auto out_queue_impl_ptr =
              std::static_pointer_cast<ResultQueueImpl>(out_queue);
          // maybe ContextFinished also can stop?
          if (out_queue_impl_ptr->GenerateStatus() ==
              AsEngine::GenerateRequestStatus::Generating) {
            request_ids.push_back(handle->request_uuid);
            request_lens.push_back(handle->context_length +
                                   handle->generate_length);
          }
        }
        int running_requests = request_ids.size();
        if (running_requests == 0) {
          LOG(ERROR) << " No Generating request!";
          break;
        }

        std::string victim_request_id =
            ChooseVictimRequest(request_ids, request_lens, running_requests);
        auto ret =
            this->StopRequestByRequestID(model_name.c_str(), victim_request_id);

        DLOG(INFO)
            << "[" << model_name << "]"
            << "MainLoop: Memory is running out, choose one request to stop "
            << " ID: " << victim_request_id;

        if (ret != AsStatus::ALLSPARK_SUCCESS) {
          LOG(INFO) << "MainLoop: " << model_name << " stop request"
                    << victim_request_id << " failed " << (int)ret;
        }
        auto& out_queue = model_state->result_queue_map[victim_request_id];
        auto out_queue_impl_ptr =
            std::static_pointer_cast<ResultQueueImpl>(out_queue);
        out_queue_impl_ptr->SetStatus(
            AsEngine::GenerateRequestStatus::GenerateInterrupted);
        LOG(INFO) << "[" << model_name << "] "
                  << "Memory is running out, Request :" << victim_request_id
                  << " GenerateInterruptedInterrupted!";
      } else {
        // TODO Other EROOR
        LOG(ERROR) << "RunTextGenerationContinue Failed:"
                   << AsGetErrorByCode(status);
        // instead of abort(), we may continue running, and mark all request
        // queue into a error state.
        continues_error_count++;
        if (continues_error_count > k_main_loop_error_threshold) {
          LOG(INFO) << "[" << model_name << "] "
                    << " Already conitnues error in " << continues_error_count
                    << " Time, seems unrecovery error, going to abort()";
        }
        // TODO: instead of abort, clean all request and restart.
        abort();
        break;
      }
    }
    auto cont_ms = timer_cont.elapsed();

    // Step 2.2: get every running request and put result to the queue.
    {
      TracerLog t(device_ctx_->GetDeviceType(), "Loop:Update", 4);
      // LOG(INFO) << "[" << model_name << "] "
      //           << "now handle_size = "
      //           << model_state->request_handle_map.size();

      UpdateResult(model_name, model_state, synchronizing, sync_pending_set);
      // check if sync finished
      DLOG(INFO) << "sync: " << synchronizing
                 << " sync pending size: " << sync_pending_set.size();

      if (synchronizing && sync_pending_set.empty()) {
        // LOG(INFO) << "[" << model_name << "] "
        //           << "sync request finished";
        synchronizing = false;
        // sync_finished_map.clear();
        sync_pending_set.clear();

        // TODO: check status in sync_finished_map and set value
        // accordingly
        LOG(INFO) << "Sync Return\n";

        deferred_promise->set_value(AsStatus::ALLSPARK_SUCCESS);
        deferred_promise.reset();
        // remove the place-holding message after fulfilling the
        // deferred promise. we can't do this earlier, because this
        // thread will wait on the cond var if the msg queue size is 0.
      }
    }

    // if there is no message unfinished task, check whether can enter stop
    // state.
    if (workers_[0]->GetUnFinishedRequest() == 0 &&
        model_state->msg_queue.size_approx() == 0) {
      // check if in GracefulStopModel phase
      if (graceful_stop_phase) {
        LOG(INFO) << "Enter graceful stop phase.";
        // in graceful stop phase, no undergoing requests, and the final request
        // has been released, then we'll really stop the loop
        if (model_state->result_queue_map.size() == 0 &&
            model_state->request_handle_map.size() == 0) {
          LOG(INFO) << "Enter graceful stop phase, real start.";
          graceful_final_released = true;
        } else {
          // even though we cannot graceful stop, we needs make sure
          // api caller will get a reply, otherwise the thread cannot join
          // and stop will hang.
          LOG(INFO)
              << "Warnning: force stop enter real graceful stop: result queue: "
              << model_state->result_queue_map.size() << " request handle "
              << model_state->request_handle_map.size()
              << " call releaseRequest to release all pending request first "
                 "!!!";
          graceful_stop_msg.promise->set_value(
              AsStatus::ALLSPARK_RUNTIME_ERROR);
          model_state->model_stopped = true;
          goto loop_end;  // use goto is more clear going to end of loop.
        }
        if (graceful_final_released) {
          graceful_stop_msg.promise->set_value(AsStatus::ALLSPARK_SUCCESS);
          model_state->model_stopped = true;
          DLOG(INFO) << "All done, gracefully stopped!";
          goto loop_end;
        }
      }
      // there is no running task , put our thread into wait.
    }
    // if no control message and no running task, wait on task.
  }

loop_end:
  LOG(INFO) << "ModelRunning Thread is going to exit";
}
std::string AsEngineImpl::ChooseVictimRequest(
    const std::vector<std::string>& request_ids,
    const std::vector<int>& request_lens, int n) {
  std::string stop_request_id;
  AsEvictionStrategy eviction_strategy = device_ctx_->GetEvictionStrategy();
  switch (eviction_strategy) {
    case AsEvictionStrategy::MaxLength: {
      int x = 0;
      for (int i = 0; i < request_lens.size(); i++) {
        if (request_lens[i] > request_lens[x]) {
          x = i;
        }
      }
      stop_request_id = request_ids[x];
      LOG(INFO) << "ALLSPARK_CACHE_MEMORY_OUT, stop MaxLength request_id = "
                << request_ids[x] << " ,length = " << request_lens[x];
      break;
    }
    case AsEvictionStrategy::Random: {
      auto rand = random_engine();
      int x = rand % n;
      stop_request_id = request_ids[x];
      LOG(INFO) << "ALLSPARK_CACHE_MEMORY_OUT, stop Random request_id = "
                << request_ids[x] << " ,length = " << request_lens[x];
      break;
    }
    default: {
      LOG(ERROR) << "not support EvictionStrategy = ";
      auto rand = random_engine();
      int x = rand % n;
      LOG(INFO) << "ALLSPARK_CACHE_MEMORY_OUT, stop Random request_id = "
                << request_ids[x] << " ,length = " << request_lens[x];
      stop_request_id = request_ids[x];
      break;
    }
  }
  return stop_request_id;
}

AsStatus AsEngineImpl::RunEngineContext(std::string model_name) {
  AsSchedulingStrategy scheduling_strategy =
      device_ctx_->GetSchedulingStrategy();
  switch (scheduling_strategy) {
    case AsSchedulingStrategy::ContextPriority: {
      bool is_new_context = true;
      int run_context_count = 0;
      while (true) {
        AsStatus status =
            this->RunTextGenerationContext(model_name.c_str(), is_new_context);
        run_context_count += 1;
        // LOG(INFO) << "run_context_count = " << run_context_count <<
        // "status="<<AsGetErrorByCode(status);
        if (status != AsStatus::ALLSPARK_SUCCESS &&
            status != AsStatus::ALLSPARK_EMPTY_REQUEST &&
            status != AsStatus::ALLSPARK_CHUNK_PREFILL) {
          // context error
          return status;
        }
        if (status == AsStatus::ALLSPARK_CHUNK_PREFILL ||
            status == AsStatus::ALLSPARK_EMPTY_REQUEST) {
          // can't do more context
          break;
        }
        is_new_context = false;
        // continue context
      }
      return AsStatus::ALLSPARK_SUCCESS;
    }
    case AsSchedulingStrategy::Balance: {
      // just run one context,must be new context
      bool is_new_context = true;
      AsStatus status =
          this->RunTextGenerationContext(model_name.c_str(), is_new_context);
      return status;
    }
    default: {
      LOG(ERROR) << "not support scheduling_strategy ";
      return AsStatus::ALLSPARK_PARAM_ERROR;
    }
  }
  return AsStatus::ALLSPARK_UNKNOWN_ERROR;
}
//-------------------------
// interface
//

AsEngine::AsEngine() : as_engine_impl_(std::make_unique<AsEngineImpl>()) {}
AsEngine::~AsEngine() = default;

AsStatus AsEngine::BuildModelFromConfigStruct(AsModelConfig& model_config) {
  return as_engine_impl_->BuildModelFromConfigStruct(model_config);
}

AsStatus AsEngine::UnloadModelFromDeviceMemory(const char* model_name) {
  return as_engine_impl_->UnloadModelFromDeviceMemory(model_name);
}

AsStatus AsEngine::ReloadModelToDeviceMemory(const char* model_name) {
  return as_engine_impl_->ReloadModelFromDeviceMemory(model_name);
}

AsStatus AsEngine::GetModelInformation(const char* model_name,
                                       std::string* model_info) {
  return as_engine_impl_->GetModelInformation(model_name, model_info);
}

AsStatus AsEngine::StartModel(const char* model_name) {
  return as_engine_impl_->StartModel(model_name);
}

AsStatus AsEngine::StopModel(const char* model_name) {
  return as_engine_impl_->StopModel(model_name);
}

AsStatus AsEngine::ReleaseModel(const char* model_name) {
  return as_engine_impl_->ReleaseModel(model_name);
}

AsStatus AsEngine::StartRequest(const char* model_name,
                                std::shared_ptr<RequestContent> request_info,
                                RequestHandle** request_handle,
                                ResultQueue** queue) {
  return as_engine_impl_->StartRequest(model_name, request_info, request_handle,
                                       queue);
}

AsStatus AsEngine::StopRequest(const char* model_name,
                               RequestHandle* request_handle) {
  return as_engine_impl_->StopRequest(model_name, request_handle);
}

AsStatus AsEngine::ReleaseRequest(const char* model_name,
                                  RequestHandle* request_handle) {
  return as_engine_impl_->ReleaseRequest(model_name, request_handle);
}

AsStatus AsEngine::SyncRequest(const char* model_name,
                               RequestHandle* request_handle) {
  return as_engine_impl_->SyncRequest(model_name, request_handle);
}

AsStatus AsEngine::LoadLoraByName(const char* model_name,
                                  const char* lora_name) {
  return as_engine_impl_->LoadLoraByName(model_name, lora_name);
}

AsStatus AsEngine::UnloadLoraByName(const char* model_name,
                                    const char* lora_name) {
  return as_engine_impl_->UnloadLoraByName(model_name, lora_name);
}

AsEngineStat AsEngine::GetAsEngineStat(const char* model_name) {
  return as_engine_impl_->GetAsEngineStat(model_name);
}

AsFileInfo AsEngine::GetFileInformation(const char* as_model_path,
                                        const char* as_param_path) {
  return as_engine_impl_->GetFileInformation(as_model_path, as_param_path);
}

std::string AsEngine::GetVersionFull() {
  return as_engine_impl_->GetVersionFull();
}

std::string AsEngine::GetOpProfilingInfo(const char* model_name) {
  return as_engine_impl_->GetOpProfilingInfo(model_name);
}

int AsEngine::GetRankId() { return as_engine_impl_->GetRankId(); }

int AsEngine::GetRankNums() { return as_engine_impl_->GetRankNums(); }

bool AsEngine::IsAllSparkWorkAsService() {
  // it must return false if code runs here
  return false;
}

AsModelConfig::AsModelConfig() {
#ifdef ENABLE_CUDA
  prefill_mode = AsMHAPrefill::AsPrefillXformer;
#else   // ENABLE_CUDA
  prefill_mode = AsMHAPrefill::AsPrefillDefault;
#endif  // ENABLE_CUDA

  cache_span_size = default_span_size;
}

AsModelConfig::AsModelConfig(
    std::string in_model_name, std::string in_model_path,
    std::string in_weights_path, std::string in_compute_unit,
    int in_engine_max_length, int in_engine_max_batch,
    int in_engine_max_prefill_length, int64_t in_swap_threshold,
    bool in_text_graph, int in_num_threads, std::string in_matmul_precision,
    std::vector<std::string> lora_names, int in_cache_span_size,
    int in_cache_span_num_init, int in_cache_span_num_grow,
    bool enable_prefix_cache, int prefix_cache_ttl,
    AsMHAPrefill in_prefill_mode, AsCacheMode in_cache_mode,
    AsEvictionStrategy in_eviction_strategy,
    AsSchedulingStrategy in_scheduling_strategy, bool enable_sparsity_matmul,
    int lora_max_rank, int lora_max_num)
    : model_name(std::move(in_model_name)),
      model_path(std::move(in_model_path)),
      weights_path(std::move(in_weights_path)),
      compute_unit(std::move(in_compute_unit)),
      num_threads(in_num_threads),
      matmul_precision(in_matmul_precision),
      swap_threshold(in_swap_threshold),
      engine_max_length(in_engine_max_length),
      engine_max_batch(in_engine_max_batch),
      engine_max_prefill_length(in_engine_max_prefill_length),
      lora_names(lora_names),
      cache_span_size(in_cache_span_size),
      cache_span_num_init(in_cache_span_num_init),
      cache_span_num_grow(in_cache_span_num_grow),
      cache_mode(in_cache_mode),
      enable_prefix_cache(enable_prefix_cache),
      prefix_cache_ttl(prefix_cache_ttl),
      prefill_mode(in_prefill_mode),
      eviction_strategy(in_eviction_strategy),
      text_graph(in_text_graph),
      scheduling_strategy(in_scheduling_strategy),
      enable_sparsity_matmul(enable_sparsity_matmul),
      lora_max_rank(lora_max_rank),
      lora_max_num(lora_max_num) {
  // replace the defualt setting in header.
  if (in_prefill_mode == AsMHAPrefill::AsPrefillDefault) {
#ifdef ENABLE_CUDA
    prefill_mode = AsMHAPrefill::AsPrefillXformer;
#else   // ENABLE_CUDA
    prefill_mode = AsMHAPrefill::AsPrefillDefault;
#endif  // ENABLE_CUDA
  }
}

std::string AsModelConfig::ToString() const {
  // prefill
  std::string prefill_string = "";
  switch (prefill_mode) {
    case AsMHAPrefill::AsPrefillDefault:
      prefill_string = "AsPrefillDefault";
      break;
    case AsMHAPrefill::AsPrefillFlashV2:
      prefill_string = "AsPrefillFlashV2";
      break;
    case AsMHAPrefill::AsPrefillXformer:
      prefill_string = "AsPrefillXformer";
      break;
    default:
      prefill_string = "AsPrefillUnknown";
  }
  // cache mode
  std::string cache_mode_string = SpanCacheConfig::CacheMode2String(cache_mode);
  std::string eviction_strategy_string = "";
  switch (eviction_strategy) {
    case AsEvictionStrategy::MaxLength:
      eviction_strategy_string = "MaxLength";
      break;
    case AsEvictionStrategy::Random:
      eviction_strategy_string = "Random";
      break;
    default:
      eviction_strategy_string = "Unknown";
  }

  // lora
  std::string lora_name_string = "init_loaded_loras: [";
  for (auto name : lora_names) {
    std::string name_str = name + ",\t";
    lora_name_string += name_str;
  }
  lora_name_string += "]";

  std::string result = std::string("AsModelConfig :\n");
  result += std::string("\tmodel_name: ") + model_name + "\n";
  result += std::string("\tmodel_path: ") + model_path + "\n";
  result += std::string("\tweights_path: ") + weights_path + "\n";
  result += std::string("\tcompute_unit: ") + compute_unit + "\n";
  result += std::string("\tnum_threads: ") + std::to_string(num_threads) + "\n";
  result += std::string("\tmatmul_precision: ") + matmul_precision + "\n";
  result += std::string("\tprefill_mode: ") + prefill_string + "\n";
  result += std::string("\tcache_mode: ") + cache_mode_string + "\n";
  result +=
      std::string("\teviction_strategy: ") + eviction_strategy_string + "\n";
  result += std::string("\tengine_max_length = ") +
            std::to_string(engine_max_length) + "\n";
  result += std::string("\tengine_max_batch = ") +
            std::to_string(engine_max_batch) + "\n";
  result += std::string("\tengine_max_prefill_length = ") +
            std::to_string(engine_max_prefill_length) + "\n";
#ifdef FIXED_SPAN_SIZE
  result += std::string("\tcache_span_size (fixed) = ") +
            std::to_string((FIXED_SPAN_SIZE)) + "\n";
#else
  result += std::string("\tcache_span_size = ") +
            std::to_string(cache_span_size) + "\n";
#endif  // FIXED_SPAN_SIZE
  result += std::string("\tcache_span_num_init = ") +
            std::to_string(cache_span_num_init) + "\n";
  result += std::string("\tcache_span_num_grow = ") +
            std::to_string(cache_span_num_grow) + "\n";
  result += std::string("\tenable_prefix_cache = ") +
            std::to_string(enable_prefix_cache) + "\n";
  result += std::string("\tprefix_cache_ttl = ") +
            std::to_string(prefix_cache_ttl) + "\n";
  result += "\t" + lora_name_string + "\n";
  result += std::string("\tswap_threshold = ") +
            std::to_string(swap_threshold) + "\n";
  result += std::string("\tenable_sparsity_matmul = ") +
            std::to_string(enable_sparsity_matmul) + "\n";
  result +=
      std::string("\tlora_max_rank= ") + std::to_string(lora_max_rank) + "\n";
  result +=
      std::string("\tlora_max_num= ") + std::to_string(lora_max_num) + "\n";

  return result;
}

std::string AsEngineStat::ToString() const {
  std::string result = "";
  result += std::string("Members of ") + std::string("AsEngineStat") +
            std::string("\n");
  result += "total_span = " + std::to_string(total_span) + "\n";
  result += "used_span = " + std::to_string(used_span) + "\n";
  result += "free_span = " + std::to_string(free_span) + "\n";
  result += "span_size = " + std::to_string(span_size) + "\n";
  result += "free_token = " + std::to_string(free_token) + "\n";
  result += "pendding_request = " + std::to_string(pendding_request) + "\n";
  result += "running_request = " + std::to_string(running_request) + "\n";
  result += "total_device_memory_pool_size = " +
            std::to_string(total_device_memory_pool_size) + "\n";
  result += "used_device_memory_pool_size = " +
            std::to_string(used_device_memory_pool_size) + "\n";
  result +=
      "total_generated_token = " + std::to_string(total_generated_token) + "\n";
  result +=
      "total_prefill_token = " + std::to_string(total_prefill_token) + "\n";
  result +=
      "prefix_cache_hit_rate = " + std::to_string(prefix_cache_hit_rate) + "\n";
  result +=
      "prefix_cache_miss_rate = " + std::to_string(prefix_cache_miss_rate) +
      "\n";

  return result;
}

std::map<std::string, std::string> AsEngineStat::ToMap() const {
  std::map<std::string, std::string> engine_stat_map;
  engine_stat_map["total_span"] = std::to_string(total_span);
  engine_stat_map["used_span"] = std::to_string(used_span);
  engine_stat_map["free_span"] = std::to_string(free_span);
  engine_stat_map["free_token"] = std::to_string(free_token);
  engine_stat_map["total_token"] = std::to_string(total_token);
  engine_stat_map["span_size"] = std::to_string(span_size);
  engine_stat_map["pendding_request"] = std::to_string(pendding_request);
  engine_stat_map["running_request"] = std::to_string(running_request);
  engine_stat_map["total_device_memory_pool_size"] =
      std::to_string(total_device_memory_pool_size);
  engine_stat_map["used_device_memory_pool_size"] =
      std::to_string(used_device_memory_pool_size);
  engine_stat_map["total_generated_token"] =
      std::to_string(total_generated_token);
  engine_stat_map["total_prefill_token"] = std::to_string(total_prefill_token);
  engine_stat_map["generate_token_persec"] =
      std::to_string(generate_token_persec);
  engine_stat_map["process_token_persec"] =
      std::to_string(process_token_persec);
  engine_stat_map["token_usage_percentage"] =
      std::to_string(token_usage_percentage);
  engine_stat_map["prefix_cache_hit_token"] =
      std::to_string(prefix_cache_hit_token);
  engine_stat_map["prefix_cache_miss_token"] =
      std::to_string(prefix_cache_miss_token);
  engine_stat_map["prefix_cache_hit_rate"] =
      std::to_string(prefix_cache_hit_rate);
  engine_stat_map["prefix_cache_miss_rate"] =
      std::to_string(prefix_cache_miss_rate);
  return engine_stat_map;
}

}  // namespace allspark
