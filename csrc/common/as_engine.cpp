/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    as_engine.cpp
 */

#include <common/allocator.h>
#include <common/device_context.h>
#include <common/env_config.h>
#include <core/model/model.h>
#include <cpu/cpu_context.h>
#include <fcntl.h>
#include <git_version.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <mutex_wrapper.h>
#include <utility/allspark_logging.h>
#include <utility/check.h>
#include <utility/file_util.h>
#include <utility/uuid.h>

#include <common/as_param_check.hpp>
#include <exception>
#include <fstream>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "engine_runtime.h"
#include "engine_worker.h"
#include "interface/allspark_check.h"
#include "thread_pool.h"
#include "utility/timer.h"
#include "weight/weight_loader.h"
#include "weight/weight_manager.h"

using google::protobuf::Message;
using google::protobuf::io::FileInputStream;
const float SAMPLING_EPS = 1e-5;
/**
 * @brief Define this macro to enforce enable top-k and force K to be within (0,
 * 100]. Otherwise K is within [0, +inf).
 */
// #define CONFIG_ENFORCE_TOPK

// Check if in graceful stopping model mode. Deny any agressive request during
// stopping.
#define CHECK_MODEL_STOPPING(model_state)               \
  do {                                                  \
    if (model_state->model_stopping) {                  \
      DLOG(INFO) << "model is stopping, access denied"; \
      return AsStatus::ALLSPARK_REQUEST_DENIED;         \
    }                                                   \
  } while (0);

#define AS_OUTPUT_GEN_ID_KEY "generated_ids";

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

  AsStatus GetModelInformation(const char* model_name, std::string* model_info);

  AsFileInfo GetFileInformation(const char* as_model_path,
                                const char* as_param_path);

  AsStatus StartModel(const char* model_name, bool do_warmup = true);
  AsStatus StopModel(const char* model_name);
  AsStatus ReleaseModel(const char* model_name);

  AsStatus StartRequest(const char* model_name,
                        std::shared_ptr<AsEngine::RequestContent> request_info,
                        RequestHandle** request_handle,
                        AsEngine::ResultQueue** queue);
  AsStatus StopRequest(const char* model_name, RequestHandle* request_handle);
  AsStatus ReleaseRequest(const char* model_name,
                          RequestHandle* request_handle);
  // sync request handle, nullptr for all request on this model.
  AsStatus SyncRequest(const char* model_name, RequestHandle* request_handle);

  AsEngineStat GetAsEngineStat(const char* model_name);

  std::string GetVersionFull();  // eg: 1.0.0(git_sha1-build_time)

  std::string GetOpProfilingInfo(const char* model_name);

  int GetRankId();
  int GetRankNums();

 private:
  // build model from saved allspark model files
  // many interface goes to here, but this interface only by self use.
  AsStatus BuildModel(const char* model_name, const std::string& model_proto,
                      std::shared_ptr<ModelWeightHandler> model_handler,
                      const std::map<std::string, int>& model_limits = {});

  AsStatus SetNumThreads(int num_threads);
  AsStatus SetDeviceIds(const std::vector<int>& device_ids);
  AsStatus CreateDeviceContext(const std::string& compute_unit);
  AsStatus SetMatmulPrecision(const std::string& precision);

  AsStatus RunTextGenerationContinue(const char* model_name);
  AsStatus RunTextGenerationContext(const char* model_name);

  AsStatus StartRequestImpl(const char* model_name, const DLTensorMap& inputs,
                            DLTensorMap* outputs, GenerateConfig& gen_cfg);
  AsStatus StopRequestByRequestID(const char* model_name,
                                  std::string request_id);
  AsStatus ReleaseRequestByRequestID(const char* model_name,
                                     std::string request_id);

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
  bool is_device_id_set_ = false;

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
  std::mutex engine_lock_;  // for async decoder lock
  int engine_max_length_ = 0;
  int engine_max_batch_ = 0;
  const int engine_max_top_logprobs_ = 10;  // const
  // using thread pool instead of vector<std::thread> in text generation
  std::unique_ptr<ThreadPool> threadpool_;
  int threadpool_size_ = 1;
  bool use_adaptive_cache_ = false;

  std::shared_ptr<WeightManager> weight_manager_;
};

static bool ReadProtoFromTextFile(const char* filename,
                                  google::protobuf::Message* proto) {
  int fd = open(filename, O_RDONLY);
  CHECK_NE(fd, -1) << "File not found: " << filename;
  google::protobuf::io::FileInputStream* input =
      new google::protobuf::io::FileInputStream(fd);
  bool success = google::protobuf::TextFormat::Parse(input, proto);
  delete input;
  close(fd);
  return success;
}

static DeviceType GetDeviceTypeFromString(std::string device_type) {
  std::unordered_map<std::string, DeviceType> device_map(
      {{"CPU", DeviceType::CPU}});
  if (device_map.find(device_type) == device_map.end()) {
    // LOG(ERROR) << "Invalid device_type:" << device_type << std::endl;
    return DeviceType::DEVICETYPE_UNDEFINED;
  }
  return device_map[device_type];
}

static std::pair<DeviceType, std::vector<int>> ParseDeviceType(
    const std::string& compute_unit) {
  int pos = compute_unit.find(":");
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
      threadpool_size_(1) {
  // set threadpool_size_ to 1 for default to avoid additional overhead,
  // such as thread switching and lock contention in CPU streaming mode.
  threadpool_ = std::make_unique<ThreadPool>(threadpool_size_);
  weight_manager_ = WeightManager::Create();
  LOG(INFO) << "AllSpark Init with Version: " << GetVersionFull();
}

AsEngineImpl::~AsEngineImpl() {}

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
  if (nranks_ > threadpool_size_) {
    threadpool_size_ = nranks_ * 2;
    threadpool_ = std::make_unique<ThreadPool>(threadpool_size_);
  }
  std::future<AsStatus> result[nranks_];
  for (int i = 0; i < workers_.size(); ++i) {
    result[i] = threadpool_->enqueue([this, i, &num_threads]() {
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

  nranks_ = device_ids.size();

  LOG(INFO) << "SetDeviceIds: DeviceIDs.size() " << device_ids.size();
  for (int i = 0; i < device_ids.size(); i++) {
    DLOG(INFO) << device_ids[i];
  }
  // 所有device inferer都走worker线程
  workers_.resize(nranks_);
  std::vector<std::thread> vthreads(nranks_);
  LOG(INFO) << "Start create " << nranks_ << " Device: " << backend
            << " workers.";
  // the only reason use multiple-thread here is init nccl requires multiple
  // process/thread be callled in current way.
  for (int i = 0; i < nranks_; ++i) {
    vthreads[i] = std::thread([&, i]() {
      switch (backend) {
        case DeviceType::CPU: {
          workers_[i] = std::make_unique<CpuWorker>(i, nranks_, device_ids[i]);
          break;
        }
      }
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
  DeviceType device_type = DeviceType::CPU;
  std::vector<int> device_ids;

  try {
    std::tie(device_type, device_ids) = ParseDeviceType(compute_unit);
  } catch (std::invalid_argument& e) {
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }

  switch (device_type) {
    case DeviceType::CPU: {
      device_ctx_ = std::make_unique<CPUContext>();
      AS_CHECK_STATUS(this->SetDeviceIds({0}));
      break;
    }
    default: {
      LOG(ERROR) << "Not Support ComputeUnit: " << compute_unit;
      return AsStatus::ALLSPARK_PARAM_ERROR;
    }
  }
  return AsStatus::ALLSPARK_SUCCESS;
}

static void CheckAndOverridePrefillMode(AsModelConfig& model_config) {
  try {
    DeviceType device_type = DeviceType::CPU;
    std::vector<int> device_ids;
    std::tie(device_type, device_ids) =
        ParseDeviceType(model_config.compute_unit);

    if (device_type == DeviceType::CPU) {
      if (model_config.prefill_mode != AsMHAPrefill::AsPrefillDefault) {
        LOG(INFO) << "Warn: CPU only support Prefill model default";
      }
      model_config.prefill_mode = AsMHAPrefill::AsPrefillDefault;
    }
  } catch (std::invalid_argument& e) {
    LOG(INFO) << "Prefll Auto Select got exception, ignore this auto set. "
              << e.what();
  }
}

AsStatus AsEngineImpl::BuildModelFromConfigStruct(AsModelConfig& model_config) {
  EnvVarConfig env_config;
  CheckAndOverridePrefillMode(model_config);

  DLOG(INFO) << "AsEngineImpl::BuildModelFromConfigStruct()" << std::endl;
  LOG(INFO) << "Build model use following config:\n"
            << model_config.ToString() << std::endl;

  std::string model_path = model_config.model_path;
  LOG(INFO) << "Load model from : " << model_path << std::endl;

  if (model_path.empty() || !util::IsExists(model_path)) {
    LOG(ERROR) << "No such file or directory : " << model_path << std::endl;
    return AsStatus::ALLSPARK_IO_ERROR;
  }

  AS_CHECK_STATUS(this->CreateDeviceContext(model_config.compute_unit));
  device_ctx_->SetPrefillMode(model_config.prefill_mode);

  if (model_config.num_threads != 0) {
    AS_CHECK_STATUS(this->SetNumThreads(model_config.num_threads));
  }
  AS_CHECK_STATUS(this->SetMatmulPrecision(model_config.matmul_precision));

  engine_max_length_ = model_config.engine_max_length;
  engine_max_batch_ = model_config.engine_max_batch;
  if (engine_max_length_ <= 2) {
    LOG(ERROR) << "Illegal egnine_max_length = " << engine_max_length_;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
  if (engine_max_batch_ <= 0) {
    LOG(ERROR) << "Illegal egnine_max_batch = " << engine_max_batch_;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }

  std::shared_ptr<TransformerProto> model_ir =
      std::make_shared<TransformerProto>();

  if (model_config.text_graph) {
    google::protobuf::Message* message = model_ir.get();
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
  weights_path = model_config.weights_path;
  DLOG(INFO) << "Load weights from : " << model_config.weights_path
             << std::endl;

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

  //--------- build model -------------//
  AS_CHECK_STATUS(
      this->BuildModel(model_name.c_str(), model_proto, model_weight_handler));
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus AsEngineImpl::BuildModel(
    const char* model_name, const std::string& model_proto,
    std::shared_ptr<ModelWeightHandler> weight_handler,
    const std::map<std::string, int>& model_limits) {
  DLOG(INFO) << "AsEngineImpl::BuildModel()" << std::endl;
  std::unique_ptr<TransformerProto> model_ir =
      std::make_unique<TransformerProto>();
  model_ir->ParseFromString(model_proto);
  device_ctx_->SetNumberHeads(model_ir->model_conf().num_heads());
  device_ctx_->SetNumberGroups(model_ir->model_conf().multi_query_group_num());
  device_ctx_->SetSizePerHead(model_ir->model_conf().size_per_head());
  device_ctx_->SetDecoderLayer(model_ir->model_conf().dec_layer());
  device_ctx_->SetDtype(model_ir->model_conf().dtype());
  for (auto& item : model_limits) {
    if (item.second < 0) {
      LOG(ERROR) << "invalid engine limit param, should >= 0" << std::endl;
      return AsStatus::ALLSPARK_PARAM_ERROR;
    }
    if (item.first == "engine_max_length") engine_max_length_ = item.second;
    if (item.first == "engine_max_batch") engine_max_batch_ = item.second;
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
  device_ctx_->SetModelMaxLength(engine_max_length_);
  device_ctx_->SetModelMaxBatch(engine_max_batch_);
  std::vector<std::thread> vthreads(nranks_);
  std::vector<std::promise<AsStatus>> promise_vec(nranks_);
  for (int i = 0; i < nranks_; ++i) {
    // load models & weights
    vthreads[i] = std::thread([&, i]() {
      try {
        LOG(INFO) << "Start Build model for rank: " << i;
        auto ret = workers_[i]->BuildModel(*model_ir, weight_manager_,
                                           weight_handler, device_ctx_.get());

        LOG(INFO) << "Finish Build model for rank: " << i;
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

  return build_status;
}

AsStatus AsEngineImpl::UnloadModelFromDeviceMemory(const char* model_name) {
  DLOG(INFO) << "[" << model_name << "] "
             << "AsEngineImpl::UnloadModelFromDeviceMemory()" << std::endl;
  AsStatus ret = AsStatus::ALLSPARK_SUCCESS;
  if (nranks_ > threadpool_size_) {
    threadpool_size_ = nranks_ * 2;
    threadpool_ = std::make_unique<ThreadPool>(threadpool_size_);
  }
  std::future<AsStatus> result[nranks_];
  for (int i = 0; i < nranks_; ++i) {
    result[i] = threadpool_->enqueue(
        [this, i]() { return workers_[i]->UnloadModelFromDeviceMemory(); });
  }
  for (int i = 0; i < nranks_; ++i) {
    ret = result[i].get();
    AS_CHECK_STATUS(ret);
  }

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

AsStatus AsEngineImpl::StartModel(const char* model_name, bool do_warmup) {
  // start model
  DLOG(INFO) << "[" << model_name << "] "
             << "AsEngineImpl::StartModel";
  // create a new model
  {
    std::unique_lock<std::mutex> locker(engine_lock_);
    auto name = std::string(model_name);
    as_stat_ = std::make_unique<AsEngineStat>(name);
    model_state_map_[name] = std::make_shared<ModelControlState>(name);
    model_state_map_[name]->StartLoop(&AsEngineImpl::ModelRunningThread, this,
                                      name, model_state_map_[name]);
  }

  // collect mem stats from all workers
  int64_t min_bytes_available = std::numeric_limits<int64_t>::max();
  int64_t rank_0_bytes_available{0};
  if (use_adaptive_cache_) {
    if (nranks_ > threadpool_size_) {
      threadpool_size_ = nranks_ * 2;
      threadpool_ = std::make_unique<ThreadPool>(threadpool_size_);
    }
    std::future<int64_t> result[nranks_];
    for (int i = 0; i < nranks_; ++i) {
      result[i] = threadpool_->enqueue([this, i]() -> int64_t {
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
        if (i == 0) {
          rank_0_bytes_available = res;
        }
      }
    }

    if (failed_ret != AsStatus::ALLSPARK_SUCCESS) {
      return failed_ret;
    } else {
      LOG(INFO) << "StartModel: min available device memory in bytes "
                   "across all devices: "
                << min_bytes_available;
    }
  }
  if (!do_warmup) return AsStatus::ALLSPARK_SUCCESS;

#if 1
  // start a fake request to make sure max length can be supported.
  LOG(INFO) << "StartModel: warming up...";

  // First warmup context , Generate at least (engine_max_batch_+2) tokens to
  // warm up context & decoder
  if (engine_max_length_ <= 2) {
    LOG(ERROR) << "StartModel: engine_max_length_ <= 2, skip warm-up";
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
  AsTensor in0("input_ids", DeviceType::CPU, DataType::INT64, DataMode::DENSE,
               Shape({1, engine_max_length_ - 2}));
  TensorUtils::Memset(in0, 0);

  const DLTensorMap warmup_inputs = {{"input_ids", in0.ToDLPack(nullptr)}};

  std::unique_ptr<GenerateConfig> warmup_cfg =
      std::make_unique<GenerateConfig>();
  warmup_cfg->max_length = engine_max_length_;
  warmup_cfg->uuid = std::to_string(0);
  warmup_cfg->top_k = 0;
  warmup_cfg->top_p = 0.5;

  std::shared_ptr<AsEngine::RequestContent> warmup_req =
      std::make_shared<AsEngine::RequestContent>();
  warmup_req->config = *(warmup_cfg);
  warmup_req->infer_type = AsEngine::RequestInferType::Generate;
  warmup_req->inputs = std::make_shared<DLTensorMap>(warmup_inputs);
  warmup_req->mm_type = AsEngine::RequestMMType::TextInput;

  RequestHandle* warmup_handle{nullptr};
  AsEngine::ResultQueue* warmup_queue{nullptr};
  AS_CHECK_STATUS(this->StartRequest(model_name, warmup_req, &warmup_handle,
                                     &warmup_queue));
  AS_CHECK_STATUS(this->SyncRequest(model_name, warmup_handle));
  if (warmup_queue->GenerateStatus() !=
      AsEngine::GenerateRequestStatus::GenerateFinished) {
    LOG(ERROR) << "Warmup failed! Please checkout engine_max_length & "
                  "engine_max_batch";
    return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }
  AS_CHECK_STATUS(this->ReleaseRequest(model_name, warmup_handle));

  // Then warmup max_batch request,input = 5, output = engine_max_batch + 10
  std::vector<RequestHandle*> warmup_handle_list;
  std::vector<AsEngine::ResultQueue*> warmup_queue_list;
  for (int i = 0; i < engine_max_batch_; i++) {
    AsTensor in0("input_ids", DeviceType::CPU, DataType::INT64, DataMode::DENSE,
                 Shape({1, 5}));
    TensorUtils::Memset(in0, 0);

    const DLTensorMap warmup_inputs = {{"input_ids", in0.ToDLPack(nullptr)}};

    std::unique_ptr<GenerateConfig> warmup_cfg =
        std::make_unique<GenerateConfig>();
    warmup_cfg->max_length = engine_max_batch_ + 10;
    warmup_cfg->uuid = std::to_string(0);
    warmup_cfg->top_k = 0;
    warmup_cfg->top_p = 0.5;

    std::shared_ptr<AsEngine::RequestContent> warmup_req =
        std::make_shared<AsEngine::RequestContent>();
    warmup_req->config = *(warmup_cfg);
    warmup_req->infer_type = AsEngine::RequestInferType::Generate;
    warmup_req->inputs = std::make_shared<DLTensorMap>(warmup_inputs);
    warmup_req->mm_type = AsEngine::RequestMMType::TextInput;

    RequestHandle* warmup_handle{nullptr};
    AsEngine::ResultQueue* warmup_queue{nullptr};
    AS_CHECK_STATUS(this->StartRequest(model_name, warmup_req, &warmup_handle,
                                       &warmup_queue));
    warmup_handle_list.push_back(warmup_handle);
    warmup_queue_list.push_back(warmup_queue);
  }

  AS_CHECK_STATUS(this->SyncRequest(model_name, nullptr));
  for (int i = 0; i < engine_max_batch_; i++) {
    RequestHandle* warmup_handle = warmup_handle_list[i];
    AsEngine::ResultQueue* warmup_queue = warmup_queue_list[i];
    if (warmup_queue->GenerateStatus() !=
        AsEngine::GenerateRequestStatus::GenerateFinished) {
      LOG(ERROR) << "Warmup failed! Please checkout engine_max_length & "
                    "engine_max_batch";
      return AsStatus::ALLSPARK_RUNTIME_ERROR;
    }
    AS_CHECK_STATUS(this->ReleaseRequest(model_name, warmup_handle));
  }

  for (int i = 0; i < nranks_; ++i) {
    workers_[i]->ResetProfiler();
  }

  int64_t bytes_per_req{0};
  // not using adaptive cache setting, just return
  if (!use_adaptive_cache_) {
    return AsStatus::ALLSPARK_SUCCESS;
  }

  // worker warmup
  if (min_bytes_available > 0 && bytes_per_req > 0) {
    LOG(INFO) << "StartModel: workers warming up...";
    if (nranks_ > threadpool_size_) {
      threadpool_size_ = nranks_ * 2;
      threadpool_ = std::make_unique<ThreadPool>(threadpool_size_);
    }
    std::future<AsStatus> result[nranks_];
    for (int i = 0; i < nranks_; ++i) {
      result[i] = threadpool_->enqueue(
          [this, i](int64_t bytes_available, int64_t bytes_per_req) {
            try {
              return workers_[i]->Warmup(bytes_available, bytes_per_req);
            } catch (std::exception& e) {
              LOG(ERROR) << "StartModel: worker " << i
                         << " warmup failed: " << std::string(e.what());
              if (std::string(e.what()) == "ALLSPARK_MEMORY_ERROR" ||
                  std::string(e.what()) == "ALLSPARK_CACHE_MEMORY_OUT") {
                return AsStatus::ALLSPARK_MEMORY_ERROR;
              } else {
                return AsStatus::ALLSPARK_RUNTIME_ERROR;
              }
            }
          },
          min_bytes_available, bytes_per_req);
    }

    AsStatus failed_ret = AsStatus::ALLSPARK_SUCCESS;
    for (int i = 0; i < nranks_; ++i) {
      AsStatus ret = result[i].get();
      if (ret != AsStatus::ALLSPARK_SUCCESS) {
        failed_ret = ret;
      }
    }

    if (failed_ret == AsStatus::ALLSPARK_SUCCESS) {
      LOG(INFO) << "StartModel: warm-up finished!";
    }

    // call the run generate interface,
    // and looping call the run text generate.
    // and put the res
    return failed_ret;
  } else {
    LOG(WARNING) << "StartModel: failed to retrieve warm-up memory usage, "
                    "skip worker warm-up";
    return AsStatus::ALLSPARK_SUCCESS;
  }
#else
  return AsStatus::ALLSPARK_SUCCESS;
#endif
  // end warm up
}

AsStatus AsEngineImpl::StopModel(const char* model_name) {
  DLOG(INFO) << "[" << model_name << "] "
             << "StopModel";
  // notify loop thread to exit
  auto reply_promise = std::make_shared<std::promise<AsStatus>>();
  // TODO: possibly thread unsafe wrt ReleaseModel
  // TODO: this check is strange
  assert(model_state_map_[model_name].get() != nullptr);
  auto& model_state = model_state_map_[model_name];
  CHECK_MODEL_STOPPING(model_state);
  {
    std::unique_lock<std::mutex> lock(*(model_state->lock));
    auto msg = std::make_unique<EngineControlMessage>(
        EngineControlMessageId::GracefulStopModel, reply_promise);
    model_state->msg_queue.push(std::move(msg));
  }
  model_state->cond_var->notify_all();

  auto ret = reply_promise->get_future().get();

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
    return AsStatus::ALLSPARK_SUCCESS;
  }
}

AsStatus AsEngineImpl::ReleaseModel(const char* model_name) {
  assert(model_state_map_[model_name].get() != nullptr);
  auto& model_state = model_state_map_[model_name];
  CHECK_MODEL_STOPPING(model_state);
  return AsStatus::ALLSPARK_SUCCESS;
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

  AsStatus ret = AsStatus::ALLSPARK_SUCCESS;
  if (nranks_ > threadpool_size_) {
    threadpool_size_ = nranks_ * 2;
    threadpool_ = std::make_unique<ThreadPool>(threadpool_size_);
  }
  std::future<AsStatus> result[nranks_];
  for (int i = 0; i < nranks_; ++i) {
    result[i] = threadpool_->enqueue([this, i, &model_ir]() {
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

// return rank id for cpu
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

AsStatus AsEngineImpl::StartRequest(
    const char* model_name,
    std::shared_ptr<AsEngine::RequestContent> request_info,
    RequestHandle** request_handle, AsEngine::ResultQueue** queue) {
  DLOG(INFO) << "[" << model_name << "] "
             << "StartRequest";

  auto ret = InputParamsVerify(model_name, request_info);
  if (ret != AsStatus::ALLSPARK_SUCCESS) {
    LOG(ERROR) << "[" << model_name << "] "
               << "StartRequest failed with error " << (int)ret;
    return ret;
  }

  auto reply_promise = std::make_shared<std::promise<AsStatus>>();

  // replace uuid id with new.
  auto handle = std::make_shared<RequestHandle>();
  handle->request_uuid = GenNewUUID();
  handle->context_length =
      (*request_info->inputs)["input_ids"]->dl_tensor.shape[1];
  request_info->config.uuid = handle->request_uuid;

  auto new_queue = std::make_shared<ResultQueueImpl>();

  assert(model_state_map_[model_name].get() != nullptr);
  auto& model_state = model_state_map_[model_name];
  CHECK_MODEL_STOPPING(model_state);
  {
    // do sync in cpu multi processes
    workers_[0]->GetDeviceContext()->SemPostInterProcess();
    std::unique_lock<std::mutex> lock(*(model_state->lock));
    auto msg = std::make_unique<EngineControlMessage>(
        EngineControlMessageId::StartRequest, reply_promise, handle,
        request_info);
    model_state->msg_queue.push(std::move(msg));

    // create result queue & handle
    model_state->result_queue_map[handle->request_uuid] = new_queue;
    model_state->request_handle_map[handle->request_uuid] = handle;
  }

  model_state->cond_var->notify_one();
  workers_[0]->GetDeviceContext()->SemWaitSendInterProcess();
  DLOG(INFO) << "[" << model_name << "] "
             << "request with uuid " << handle->request_uuid << " notified";

  // this is safe because this scope holds shared ptrs
  *request_handle = handle.get();
  *queue = new_queue.get();

  return ret;
}

AsStatus AsEngineImpl::StopRequest(const char* model_name,
                                   RequestHandle* request_handle) {
  // TODO: param check
  if (!request_handle) {
    LOG(ERROR) << "[" << model_name << "] "
               << "StopRequest: request_handle cannot be nullptr";
    return AsStatus::ALLSPARK_EMPTY_REQUEST;
  }

  // put request to queue.
  // wait the the promise to be filled.
  DLOG(INFO) << "[" << model_name << "] "
             << "StopRequest";

  auto reply_promise = std::make_shared<std::promise<AsStatus>>();
  std::string uuid;

  assert(model_state_map_[model_name].get() != nullptr);
  auto& model_state = model_state_map_[model_name];
  workers_[0]->GetDeviceContext()->SemPostInterProcess();
  {
    std::unique_lock<std::mutex> lock(*(model_state->lock));
    uuid = request_handle->request_uuid;
    auto msg = std::make_unique<EngineControlMessage>(
        EngineControlMessageId::StopRequest, reply_promise,
        model_state->request_handle_map[uuid]);
    model_state->msg_queue.push(std::move(msg));
  }
  model_state->cond_var->notify_one();
  workers_[0]->GetDeviceContext()->SemWaitSendInterProcess();

  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus AsEngineImpl::ReleaseRequest(const char* model_name,
                                      RequestHandle* request_handle) {
  // TODO: param check
  if (!request_handle) {
    LOG(ERROR) << "[" << model_name << "] "
               << "ReleaseRequest: request_handle cannot be nullptr";
    return AsStatus::ALLSPARK_EMPTY_REQUEST;
  }

  // put request to queue.
  // wait the the promise to be filled.
  DLOG(INFO) << "[" << model_name << "] "
             << "ReleaseRequest";

  auto reply_promise = std::make_shared<std::promise<AsStatus>>();
  std::string uuid;

  assert(model_state_map_[model_name].get() != nullptr);
  auto& model_state = model_state_map_[model_name];
  workers_[0]->GetDeviceContext()->SemPostInterProcess();
  {
    std::unique_lock<std::mutex> lock(*(model_state->lock));
    uuid = request_handle->request_uuid;
    auto msg = std::make_unique<EngineControlMessage>(
        EngineControlMessageId::ReleaseRequest, reply_promise,
        model_state->request_handle_map[uuid]);
    model_state->msg_queue.push(std::move(msg));
  }

  model_state->cond_var->notify_one();
  workers_[0]->GetDeviceContext()->SemWaitSendInterProcess();
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus AsEngineImpl::SyncRequest(const char* model_name,
                                   RequestHandle* request_handle) {
  DLOG(INFO) << "[" << model_name << "] "
             << "SyncRequest";

  auto reply_promise = std::make_shared<std::promise<AsStatus>>();
  std::string uuid;

  assert(model_state_map_[model_name].get() != nullptr);
  auto& model_state = model_state_map_[model_name];
  workers_[0]->GetDeviceContext()->SemPostInterProcess();
  if (request_handle) {
    // sync one request
    std::unique_lock<std::mutex> lock(*(model_state->lock));
    uuid = request_handle->request_uuid;
    auto msg = std::make_unique<EngineControlMessage>(
        EngineControlMessageId::SyncRequest, reply_promise,
        model_state->request_handle_map[uuid]);
    model_state->msg_queue.push(std::move(msg));
  } else {
    // sync all requests, msg.request_handle carries nullptr
    uuid = "<ALL>";
    auto msg = std::make_unique<EngineControlMessage>(
        EngineControlMessageId::SyncRequest, reply_promise);
    std::unique_lock<std::mutex> lock(*(model_state->lock));
    model_state->msg_queue.push(std::move(msg));
  }
  model_state->cond_var->notify_one();
  workers_[0]->GetDeviceContext()->SemWaitSendInterProcess();
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
  DLOG(INFO) << "[" << model_name << "] "
             << "AsEngineImpl::RunTextGenerationContinue mutex lock passed"
             << std::endl;
  // check model registered
  if (model_irs_[model_name] == nullptr) {
    LOG(ERROR) << "[" << model_name << "] "
               << "Invalid model name : " << model_name << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
  // verify params
  if (model_irs_[model_name]->model_conf().is_generate() == false) {
    LOG(ERROR) << "[" << model_name << "] "
               << "RunTextGenerationContinue() is only supported in text "
                  "generation model.Please use RunModel() API."
               << std::endl;
    return AsStatus::ALLSPARK_INVALID_CALL_ERROR;
  }
  AsStatus ret = AsStatus::ALLSPARK_SUCCESS;
  if (nranks_ > threadpool_size_) {
    threadpool_size_ = nranks_ * 2;
    threadpool_ = std::make_unique<ThreadPool>(threadpool_size_);
  }
  std::future<AsStatus> result[nranks_];
  for (int i = 0; i < nranks_; ++i) {
    result[i] = threadpool_->enqueue([this, i]() {
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
    }
  }
  for (int i = 0; i < nranks_; ++i) {
    result[i] = threadpool_->enqueue(
        [this, i]() { return workers_[i]->RunTextGenerationContinue(); });
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

AsStatus AsEngineImpl::RunTextGenerationContext(const char* model_name) {
  DLOG(INFO) << "[" << model_name << "] "
             << "AsEngineImpl::RunTextGenerationContext" << std::endl;
  std::lock_guard<std::mutex> guard(engine_lock_);
  DLOG(INFO) << "[" << model_name << "] "
             << "AsEngineImpl::RunTextGenerationContext mutex lock passed"
             << std::endl;

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
  if (nranks_ > threadpool_size_) {
    threadpool_size_ = nranks_ * 2;
    threadpool_ = std::make_unique<ThreadPool>(threadpool_size_);
  }
  std::future<AsStatus> result[nranks_];
  for (int i = 0; i < nranks_; ++i) {
    result[i] = threadpool_->enqueue([this, i]() {
      try {
        return workers_[i]->RunTextGenerationContext();
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
  as_stat_->total_device_memory_pool_size = bytes_limit;
  as_stat_->used_device_memory_pool_size = bytes_pre_warmup;
}
AsStatus AsEngineImpl::StopRequestByRequestID(const char* model_name,
                                              std::string request_id) {
  DLOG(INFO) << "AsEngineImpl::StopRequestByRequestID" << std::endl;
  std::lock_guard<std::mutex> guard(engine_lock_);
  DLOG(INFO) << "AsEngineImpl::StopRequestByRequestID mutex lock passed"
             << std::endl;

  // check model registered
  if (model_irs_[model_name] == nullptr) {
    LOG(ERROR) << "Invalid model name : " << model_name << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
  AsStatus ret = AsStatus::ALLSPARK_SUCCESS;
  if (nranks_ > threadpool_size_) {
    threadpool_size_ = nranks_ * 2;
    threadpool_ = std::make_unique<ThreadPool>(threadpool_size_);
  }
  std::future<AsStatus> result[nranks_];
  for (int i = 0; i < nranks_; ++i) {
    result[i] = threadpool_->enqueue([this, i, request_id]() {
      return workers_[i]->StopRequest(request_id);
    });
  }

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

AsStatus AsEngineImpl::ReleaseRequestByRequestID(const char* model_name,
                                                 std::string request_id) {
  DLOG(INFO) << "AsEngineImpl::ReleaseRequestByRequestID" << std::endl;
  std::lock_guard<std::mutex> guard(engine_lock_);
  DLOG(INFO) << "AsEngineImpl::ReleaseRequestByRequestID mutex lock passed"
             << std::endl;
  // check model registered
  if (model_irs_[model_name] == nullptr) {
    LOG(ERROR) << "Invalid model name : " << model_name << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
  AsStatus ret = AsStatus::ALLSPARK_SUCCESS;
  if (nranks_ > threadpool_size_) {
    threadpool_size_ = nranks_ * 2;
    threadpool_ = std::make_unique<ThreadPool>(threadpool_size_);
  }
  std::future<AsStatus> result[nranks_];
  for (int i = 0; i < nranks_; ++i) {
    result[i] = threadpool_->enqueue([this, i, request_id]() {
      return workers_[i]->ReleaseRequest(request_id);
    });
  }

  // 即使失败、异常，也要让各子线程运行完毕，以保证原子性。在可恢复的情况下，确保下一次请求有干净的环境
  AsStatus failed_ret = AsStatus::ALLSPARK_SUCCESS;
  for (int i = 0; i < nranks_; ++i) {
    try {
      ret = result[i].get();
      if (not AS_STATUS_OK(ret)) failed_ret = ret;
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
  if (model_irs_[model_name]->model_conf().is_generate() == false) {
    LOG(ERROR) << "[" << model_name << "] "
               << "StartRequest() is only supported in text "
                  "generation model.Please use RunModel() API."
               << std::endl;
    return AsStatus::ALLSPARK_INVALID_CALL_ERROR;
  }
  if (engine_max_length_ != 0 && gen_cfg.max_length > engine_max_length_) {
    LOG(ERROR) << "[" << model_name << "] "
               << "genconfig.max_length > engine_max_length_ " << std::endl;
    return AsStatus::ALLSPARK_EXCEED_LIMIT_ERROR;
  }
  if (engine_max_length_ != 0 &&
      (*request_info->inputs)["input_ids"]->dl_tensor.shape[1] >
          engine_max_length_ - 2) {
    LOG(ERROR) << "[" << model_name << "] "
               << "Too large input_len ! Input_len > engine_max_length_ - 2"
               << std::endl;
    return AsStatus::ALLSPARK_EXCEED_LIMIT_ERROR;
  }
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
  // hard limits, to avoid invalid mem write
  if (input_batch > AsLimit::LIMIT_MAX_BATCH) {
    LOG(ERROR) << "[" << model_name << "] "
               << "Input limit exceeds, your batchsize=" << input_batch
               << "(should <= " << AsLimit::LIMIT_MAX_BATCH << ")" << std::endl;
    return AsStatus::ALLSPARK_EXCEED_LIMIT_ERROR;
  }
  // user customized max batch size
  if (engine_max_batch_ != 0 && input_batch > engine_max_batch_) {
    LOG(ERROR) << "[" << model_name << "] "
               << "Input batchsize=" << input_batch
               << "(should <= " << engine_max_batch_ << ")" << std::endl;
    return AsStatus::ALLSPARK_EXCEED_LIMIT_ERROR;
  }
  // checkout top_logprobs
  if (gen_cfg.logprobs && gen_cfg.top_logprobs > engine_max_top_logprobs_) {
    LOG(ERROR) << "[" << model_name << "] "
               << "gen_cfg.top_logprobs=" << gen_cfg.top_logprobs
               << "(should <= " << engine_max_top_logprobs_ << ")" << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus AsEngineImpl::StartRequestImpl(const char* model_name,
                                        const DLTensorMap& inputs,
                                        DLTensorMap* outputs,
                                        GenerateConfig& gen_cfg) {
  DLOG(INFO) << "[" << model_name << "] "
             << "AsEngineImpl::RunTextGeneration" << std::endl;

  lock_guard_wrapper<std::mutex> guard(engine_lock_);
  AsStatus ret = AsStatus::ALLSPARK_SUCCESS;

  // allocate output tensors.
  TensorMap out_tensors;
  std::string out_name = AS_OUTPUT_GEN_ID_KEY;
  out_tensors.insert(
      {out_name, std::make_shared<AsTensor>(out_name, DeviceType::CPU,
                                            DataType::INT64, DataMode::DENSE,
                                            Shape{1, engine_max_length_})});
  if (nranks_ > threadpool_size_) {
    threadpool_size_ = nranks_ * 2;
    threadpool_ = std::make_unique<ThreadPool>(threadpool_size_);
  }
  std::future<AsStatus> result[nranks_];
  for (int i = 0; i < nranks_; ++i) {
    result[i] =
        threadpool_->enqueue([this, i, &inputs, &out_tensors, &gen_cfg]() {
          return workers_[i]->EnqueueRequest(inputs, &out_tensors, gen_cfg);
        });
  }

  //  Even in the event of failure or exceptions, ensure that all subthreads
  //  are allowed to run to completion to maintain atomicity.
  //  In recoverable situations, ensure a clean environment for the next
  //  request.
  AsStatus failed_ret = AsStatus::ALLSPARK_SUCCESS;
  for (int i = 0; i < nranks_; ++i) {
    try {
      ret = result[i].get();
      if (not AS_STATUS_OK(ret)) failed_ret = ret;
    } catch (std::exception& e) {
      AsSaveError(e.what());
      LOG(ERROR) << "AsEngineImpl::EqueueRequest: "
                    "exception caught: "
                 << e.what() << ", saved with AsSaveError";
      failed_ret = AsStatus::ALLSPARK_RUNTIME_ERROR;
    }
  }
  AsClearErrors();

  return failed_ret;
}

static std::shared_ptr<AsEngine::GeneratedElements>
FetchGenrationResultAndIncreaseCounter(
    Request* request, const std::shared_ptr<RequestHandle>& handle,
    ResultQueueImpl* result_queue) {
  std::shared_ptr<AsEngine::GeneratedElements> ele =
      std::make_shared<AsEngine::GeneratedElements>();
  TensorMap& tmap = request->outputs;
  std::vector<std::vector<std::pair<int64_t, float>>> log_probs_list =
      request->log_probs_list;
  auto device_tensor_ptr = tmap.at("generated_ids");
  if (device_tensor_ptr->GetShape().Count() == 0) {
    return nullptr;
  }
  // already in CPU
  auto host_tensor_ptr = device_tensor_ptr;
  assert(host_tensor_ptr->GetShape()[0] == 1);
  size_t new_length = host_tensor_ptr->GetShape()[1];
  size_t old_length = handle->context_length + handle->generate_length;

  if (new_length > old_length) {
    int64_t* raw_ptr = static_cast<int64_t*>(host_tensor_ptr->GetDataPtr());
    ele->ids_from_generate.reserve(new_length - old_length);
    for (int i = 0; i < new_length - old_length; i++) {
      ele->ids_from_generate.push_back(raw_ptr[old_length + i]);
      if (request->gen_cfg.logprobs) {
        ele->log_probs_list.push_back(
            request->log_probs_list[handle->generate_length + i]);
      }
    }
    handle->generate_length = new_length - handle->context_length;
    return ele;
  } else {
    // there is no tokens;
    return nullptr;
  }
}
void AsEngineImpl::UpdateResult(
    std::string model_name, std::shared_ptr<ModelControlState> model_state,
    bool& synchronizing, std::unordered_set<std::string>& sync_pending_set) {
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
  for (auto& entry : model_state->request_handle_map) {
    auto& handle = entry.second;
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
    std::shared_ptr<AsEngine::GeneratedElements> new_ele =
        FetchGenrationResultAndIncreaseCounter(
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
  }
}

static double BytesToMegabytes(size_t bytes) {
  const size_t bytesPerMegabyte = 1024 * 1024;
  return static_cast<double>(bytes) / bytesPerMegabyte;
}

static void PrintEngineStat(AsEngineStat& stat, DeviceType dev_type) {
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
  avg_prompt_tput = std::max(0.0, avg_prompt_tput);
  avg_gen_tput = std::max(0.0, avg_gen_tput);

  int old_p = std::cout.precision();

  std::cout.precision(1);
  if (dev_type == DeviceType::CPU) {
    LOG(INFO) << "| AllsparkStat | Req: Running: " << stat.running_request
              << " Pending: " << stat.pendding_request
              << " \t Prompt: " << avg_prompt_tput << " T/s "
              << " Gen: " << avg_gen_tput << " T/s ";

  } else {
    LOG(INFO) << "| AllsparkStat | Req: Running: " << stat.running_request
              << " Pending: " << stat.pendding_request << " \t| Token: ( "
              << stat.total_token - stat.free_token << " / " << stat.total_token
              << " ) "
              << " \t| Prompt: " << avg_prompt_tput << " T/s "
              << " Gen: " << avg_gen_tput << " T/s "
              << " \t| DeviceMem(MB): ( "
              << BytesToMegabytes(stat.used_device_memory_pool_size) << " / "
              << BytesToMegabytes(stat.total_device_memory_pool_size) << " ) ";
  }

  std::cout.precision(old_p);
}

// a while(1) loop get from control command queue, if no msg from control,
// running the one decoding job.
// if decoding was all idle, wait on the control comamnd queue.
void AsEngineImpl::ModelRunningThread(
    std::string model_name, std::shared_ptr<ModelControlState> model_state) {
  std::string s = "ModelRunningThread";
  pthread_setname_np(pthread_self(),
                     s.c_str());  // set the name (pthread_self() returns the
                                  // pthread_t of the current thread)
  bool looping = true;
  long loop_cnt = 0;
  bool graceful_stop_phase = false;
  bool graceful_final_released = false;
  std::unique_ptr<EngineControlMessage> graceful_stop_msg = nullptr;

  bool synchronizing = false;
  std::shared_ptr<std::promise<AsStatus>> deferred_promise{nullptr};
  std::unordered_set<std::string> sync_pending_set;

  using clock = std::chrono::steady_clock;
  auto next_log_time = clock::now();

  while (looping) {
    util::Timer time_outer;
    loop_cnt++;
    UpdateAsEngineStat();
    // print the engine state for easier service trace.
    // for multiple numa, only print this info on node 0.
    if (clock::now() >= next_log_time && GetRankId() == 0) {
      int next_sec = EnvVarConfig::GetInt("HIE_LOG_STATUS_INTERVAL", 5);
      auto stat = GetAsEngineStat(model_name.c_str());
      PrintEngineStat(stat, device_ctx_->GetDeviceType());
      next_log_time += std::chrono::seconds(next_sec);
    }

    // decoupling message decoding phase and model execution phase
    bool no_execution = false;
    int process_msg_size = 0;
    // Phase 1: message decoding
    // Pick one control message, handle control message, return control
    // message promise.
    // If synchronizing, block any message until finished.
    if (!synchronizing) {
      std::unique_lock<std::mutex> lock(*(model_state->lock));
      int cur_msg_size = model_state->msg_queue.size();
      if (model_state->msg_queue.size() > 0) {
        // NOTE: the front is moved, do not use it anymore
        std::unique_ptr<EngineControlMessage> msg{
            std::move(model_state->msg_queue.front())};
        DLOG(INFO) << "[" << model_name << "] "
                   << "ModelRunningThread: receive message " << (int)msg->msg;

        // dispatch message
        switch (msg->msg) {
          case EngineControlMessageId::GracefulStopModel: {
            graceful_stop_phase = true;
            model_state->model_stopping = true;
            graceful_stop_msg = std::move(
                msg);  // save, its promise will be filled when all done
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
              msg->promise->set_value(AsStatus::ALLSPARK_REQUEST_DENIED);
              break;
            }
            // start the reuqest ?
            DLOG(INFO) << "[" << model_name << "] "
                       << "RunTextGeneration";
            util::Timer t1;
            auto ret = this->StartRequestImpl(model_name.c_str(),
                                              *msg->request->inputs, nullptr,
                                              msg->request->config);
            DLOG(INFO) << "[" << model_name << "] "
                       << "RunTextGeneration finish " << t1.elapsed() << " ms";
            break;
          }
          case EngineControlMessageId::StopRequest: {
            auto handle_ptr = msg->request_handle.lock();
            if (!handle_ptr) {
              LOG(ERROR) << "[" << model_name << "] "
                         << "StopRequest: request expired";
              msg->promise->set_value(AsStatus::ALLSPARK_ILLEGAL_REQUEST_ID);
            } else {
              DLOG(INFO) << "[" << model_name << "] "
                         << "StopRequest: " << handle_ptr->request_uuid;
              auto ret = this->StopRequestByRequestID(model_name.c_str(),
                                                      handle_ptr->request_uuid);
              msg->promise->set_value(ret);
            }
            break;
          }
          case EngineControlMessageId::SyncRequest: {
            if (msg->EmptyRequestHandle()) {
              // sync all requests
              for (const auto& entry : model_state->request_handle_map) {
                const auto& handle = entry.second;
                sync_pending_set.insert(handle->request_uuid);
              }
              DLOG(INFO) << "[" << model_name << "] "
                         << "SyncRequest: sync all requests, size: "
                         << sync_pending_set.size();
              // promise will not be fulfilled until sync finished
              deferred_promise = msg->promise;
              synchronizing = true;
            } else {
              // sync one request
              auto handle_ptr = msg->request_handle.lock();
              if (!handle_ptr) {
                LOG(ERROR) << "[" << model_name << "] "
                           << "SyncRequest: request expired";
                msg->promise->set_value(AsStatus::ALLSPARK_ILLEGAL_REQUEST_ID);
              } else {
                DLOG(INFO) << "[" << model_name << "] "
                           << "SyncRequest: sync request with uuid: "
                           << handle_ptr->request_uuid;
                sync_pending_set.insert(handle_ptr->request_uuid);
                // promise will not be fulfilled until sync
                // finished
                deferred_promise = msg->promise;
                synchronizing = true;
              }
            }
            break;
          }
          case EngineControlMessageId::ReleaseRequest: {
            auto handle_ptr = msg->request_handle.lock();
            if (!handle_ptr) {
              LOG(ERROR) << "[" << model_name << "] "
                         << "ReleaseRequest: request expired";
              msg->promise->set_value(AsStatus::ALLSPARK_ILLEGAL_REQUEST_ID);
            } else {
              // before release , stop request
              auto ret1 = this->StopRequestByRequestID(
                  model_name.c_str(), handle_ptr->request_uuid);
              auto ret2 = this->ReleaseRequestByRequestID(
                  model_name.c_str(), handle_ptr->request_uuid);
              if (ret1 != AsStatus::ALLSPARK_SUCCESS)
                msg->promise->set_value(ret1);
              else
                msg->promise->set_value(ret2);
              model_state->result_queue_map.erase(handle_ptr->request_uuid);
              model_state->request_handle_map.erase(handle_ptr->request_uuid);
            }
            // check if it's the final msg
            if (graceful_stop_phase &&
                workers_[0]->GetUnFinishedRequest() == 0 &&
                model_state->msg_queue.size() == 0) {
              graceful_final_released = true;
            }
            DLOG(INFO) << "[" << model_name << "] "
                       << "Release request: release request with uuid: "
                       << handle_ptr->request_uuid;
            break;
          }
          default: {
            LOG(WARNING) << "[" << model_name << "] "
                         << "Warning: unhandle message received: "
                         << (int)msg->msg;
          }
        }

        // If the promise is not deferred, safely pop the queue front.
        // NOTE: though the msg is not popped, it has been moved and
        // the resource will be released after going beyond this scope.
        if (!deferred_promise) {
          model_state->msg_queue.pop();
        }
      }
      process_msg_size = cur_msg_size - model_state->msg_queue.size();
    } else {
      DLOG(INFO) << "[" << model_name << "] "
                 << "skipping message handling due to synchronization";
    }

    no_execution = workers_[0]->GetDeviceContext()->SemWaitMsgSynInterProcess(
        process_msg_size);

    // Phase 2: model execution
    if (no_execution) {
      continue;
    }

    // Step 2.1: check running task, run the cont step
    // This will automatically and internally call AsModel::StopRequest if a
    // request is naturally finished, and thus finishes the `Request`.
    util::Timer timer_cont;
    AsStatus status = this->RunTextGenerationContext(model_name.c_str());
    if (status != AsStatus::ALLSPARK_SUCCESS &&
        status != AsStatus::ALLSPARK_EMPTY_REQUEST) {
      LOG(ERROR) << "RunTextGenerationContext Failed:"
                 << AsGetErrorByCode(status);
    }
    if (status == AsStatus::ALLSPARK_SUCCESS) {
      UpdateResult(model_name, model_state, synchronizing, sync_pending_set);
    }

    // check running task, run the cont step.
    while (true) {
      AsStatus status = this->RunTextGenerationContinue(model_name.c_str());
      if (status == AsStatus::ALLSPARK_SUCCESS ||
          status == AsStatus::ALLSPARK_EMPTY_REQUEST) {
        break;
      }

      LOG(ERROR) << "RunTextGenerationContinue Failed:"
                 << AsGetErrorByCode(status);

      // if there is no resource during generation, randomly stop one request
      // and let other request have time to start.
      if (status == AsStatus::ALLSPARK_CACHE_MEMORY_OUT) {
        std::vector<std::string> request_ids;
        for (auto& entry : model_state->request_handle_map) {
          auto& handle = entry.second;
          auto& out_queue = model_state->result_queue_map[handle->request_uuid];
          auto out_queue_impl_ptr =
              std::static_pointer_cast<ResultQueueImpl>(out_queue);
          // maybe ContextFinished also can stop?
          if (out_queue_impl_ptr->GenerateStatus() ==
              AsEngine::GenerateRequestStatus::Generating) {
            request_ids.push_back(handle->request_uuid);
          }
        }
        int n = request_ids.size();
        if (n == 0) {
          LOG(ERROR) << " No Generating reqeust!";
          break;
        }

        srand(time(nullptr));

        int x = rand() % n;
        std::string stop_request_id = request_ids[x];
        auto ret =
            this->StopRequestByRequestID(model_name.c_str(), stop_request_id);
        auto& out_queue = model_state->result_queue_map[stop_request_id];
        auto out_queue_impl_ptr =
            std::static_pointer_cast<ResultQueueImpl>(out_queue);
        out_queue_impl_ptr->SetStatus(
            AsEngine::GenerateRequestStatus::GenerateInterrupted);
        LOG(INFO) << "[" << model_name << "] "
                  << "Request :" << stop_request_id
                  << " GenerateInterruptedInterrupted!";
      } else {
        LOG(ERROR) << "RunTextGenerationContinue Failed:"
                   << AsGetErrorByCode(status);
        abort();
        break;
      }
    }
    auto cont_ms = timer_cont.elapsed();

    // Step 2.2: get every running request and put result to the queue.
    {
      std::unique_lock<std::mutex> lock(*(model_state->lock));

      UpdateResult(model_name, model_state, synchronizing, sync_pending_set);
      // check if sync finished
      if (synchronizing && sync_pending_set.empty()) {
        synchronizing = false;
        sync_pending_set.clear();

        // TODO: check status in sync_finished_map and set value
        // accordingly
        deferred_promise->set_value(AsStatus::ALLSPARK_SUCCESS);
        deferred_promise.reset();

        // remove the place-holding message after fulfilling the
        // deferred promise. we can't do this earlier, because this
        // thread will wait on the cond var if the msg queue size is 0.
        model_state->msg_queue.pop();
      }
    }

    if (workers_[0]->GetUnFinishedRequest() == 0 &&
        model_state->msg_queue.size() == 0) {
      // check if in GracefulStopModel phase
      if (graceful_stop_phase) {
        // in graceful stop phase, no undergoing requests, and the final request
        // has been released, then we'll really stop the loop
        if (model_state->result_queue_map.size() == 0 &&
            model_state->request_handle_map.size() == 0)
          graceful_final_released = true;
        if (graceful_final_released) {
          assert(graceful_stop_msg != nullptr);
          graceful_stop_msg->promise->set_value(AsStatus::ALLSPARK_SUCCESS);
          DLOG(INFO) << "All done, gracefully stopped!";
          break;
        }
      }
      // there is no running task , put our thread into wait.
      std::unique_lock<std::mutex> lock(*(model_state->lock));
      DLOG(INFO) << "[" << model_name << "] "
                 << "no request, waiting on cond "
                 << workers_[0]->GetUnFinishedRequest();
      model_state->cond_var->wait(lock, [this, model_state]() {
        return this->workers_[0]->GetUnFinishedRequest() != 0 ||
               model_state->msg_queue.size() > 0;
      });
    }
    // if no control message and no running task, wait on task.
  }
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
  prefill_mode = AsMHAPrefill::AsPrefillDefault;
}

AsModelConfig::AsModelConfig(
    std::string in_model_name, std::string in_model_path,
    std::string in_weights_path, std::string in_compute_unit,
    int in_engine_max_length, int in_engine_max_batch, bool in_text_graph,
    int in_num_threads, std::string in_matmul_precision,
    AsMHAPrefill in_prefill_mode, AsCacheMode in_cache_mode)
    : model_name(std::move(in_model_name)),
      model_path(std::move(in_model_path)),
      weights_path(std::move(in_weights_path)),
      compute_unit(std::move(in_compute_unit)),
      num_threads(in_num_threads),
      matmul_precision(in_matmul_precision),
      engine_max_length(in_engine_max_length),
      engine_max_batch(in_engine_max_batch),
      cache_mode(in_cache_mode),
      prefill_mode(in_prefill_mode),
      text_graph(in_text_graph) {
  // replace the defualt setting in header.
  if (in_prefill_mode == AsMHAPrefill::AsPrefillDefault) {
    prefill_mode = AsMHAPrefill::AsPrefillDefault;
  }
}

std::string AsModelConfig::ToString() const {
  // prefill
  std::string prefill_string = "";
  switch (prefill_mode) {
    case AsMHAPrefill::AsPrefillDefault:
      prefill_string = "AsPrefillDefault";
      break;
    default:
      prefill_string = "AsPrefillUnknown";
  }
  // cache mode
  std::string cache_mode_string = "";
  switch (cache_mode) {
    case AsCacheMode::AsCacheDefault:
      cache_mode_string = "AsCacheDefault";
      break;
    case AsCacheMode::AsCacheQuantI8:
      cache_mode_string = "AsCacheQuantI8";
      break;
    default:
      cache_mode_string = "AsCacheUnknown";
  }

  std::string result = std::string("AsModelConfig :\n");
  result += std::string("\tmodel_name: ") + model_name + "\n";
  result += std::string("\tmodel_path: ") + model_path + "\n";
  result += std::string("\tweights_path: ") + weights_path + "\n";
  result += std::string("\tcompute_unit: ") + compute_unit + "\n";
  result += std::string("\tnum_threads: ") + std::to_string(num_threads) + "\n";
  result += std::string("\tmatmul_precision: ") + matmul_precision + "\n";
  result += std::string("\tprefill_mode: ") + prefill_string + "\n";
  result += std::string("\tcache_mode: ") + cache_mode_string + "\n";
  result += std::string("\tengine_max_length = ") +
            std::to_string(engine_max_length) + "\n";
  result += std::string("\tengine_max_batch = ") +
            std::to_string(engine_max_batch) + "\n";
  return result;
}

std::string AsEngineStat::ToString() const {
  std::string result = "";
  result += std::string("Members of ") + std::string("AsEngineStat") +
            std::string("\n");
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

  return result;
}

}  // namespace allspark
