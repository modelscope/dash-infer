/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    model_stress_test.cpp
 */

#include <unistd.h>

#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <thread>

#include "input.h"

#ifdef ENABLE_AS_SERVICE
#include "allspark_client.h"
#else
#include "allspark.h"
#endif

using namespace allspark;

#define LOG(INFO) std::cout << std::endl
#define LOG(INFO) std::cout << std::endl

#ifdef ENABLE_AS_SERVICE
using ASCommonEngine = allspark::AsClientEngine;
#else
using ASCommonEngine = allspark::AsEngine;
#endif

#define CONCAt2(a, b) a##b
#define CONCAT2(a, b) CONCAt2(a, b)
#define CONCAT3(a, b, c) CONCAT2(CONCAT2(a, b), c)

class ModelSession;
std::mutex mtx;
std::condition_variable cv;
bool isFinished = false;
std::queue<std::shared_ptr<ModelSession>> messageQueue;

class Summary {
 public:
  Summary() = default;
  ~Summary() = default;
  void DoCalc(int in_size, int out_size, float context_time, float decoder_time,
              bool success) {
    std::unique_lock<std::mutex> lock(mtx_);
    total_request_ += 1;
    failure_request_ += success ? 0 : 1;
    total_input_ += success ? in_size : 0;
    total_output_ += success ? out_size : 0;
    total_context_time_ += success ? context_time : 0;
    total_decoder_time_ += success ? decoder_time : 0;
  }
  void PrintSummary(float total_time_ms) {
    LOG(INFO) << "total request: " << total_request_
              << " total_time_ms: " << total_time_ms
              << " failure request: " << failure_request_
              << " total input: " << total_input_
              << " total output: " << total_output_
              << " total context time: " << total_context_time_
              << " total decoder time: " << total_decoder_time_
              << " avg_input_len: "
              << total_input_ / float(total_request_ - failure_request_)
              << " avg_output_len: "
              << total_output_ / float(total_request_ - failure_request_)
              << " avg_context_time: "
              << total_context_time_ / float(total_request_ - failure_request_)
              << " avg_decoder_time_each_session: "
              << total_decoder_time_ / float(total_request_ - failure_request_)
              << " avg_decoder_time_each_token: "
              << total_decoder_time_ / float(total_output_)
              << " request_throughput: "
              << (total_request_ - failure_request_) / total_time_ms * 1000
              << " output_throughput: " << total_output_ / total_time_ms * 1000
              << " output_throughput_including failed request: "
              << total_output_including_failed_request_ / total_time_ms * 1000;
  }
  void AddTotolOutput(int size) {
    std::unique_lock<std::mutex> lock(mtx_);
    total_output_including_failed_request_ += size;
  }

 public:
  std::mutex mtx_;
  int total_input_ = 0;
  int total_output_ = 0;
  int total_request_ = 0;
  int failure_request_ = 0;
  float total_context_time_ = 0;
  float total_decoder_time_ = 0;
  int total_output_including_failed_request_ = 0;
};

Summary summary;

class Timer {
 public:
  void Start() { t_ = std::chrono::steady_clock::now(); }
  double Stop() {  // in ms
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::steady_clock::now() - t_)
               .count() /
           1000;
  }

 private:
  std::chrono::time_point<std::chrono::steady_clock> t_;
};

class DLTensorManager {
 public:
  DLTensorManager() : dl_managed_tensor_(nullptr) {}
  DLTensorManager(DLManagedTensor* dl_tensor) : dl_managed_tensor_(dl_tensor) {}
  ~DLTensorManager() {
    if (dl_managed_tensor_) {
      if (dl_managed_tensor_->deleter) {
        //        std::cout << "clinet destroy dl tensor" << std::endl;
        //        dl_managed_tensor_->deleter(dl_managed_tensor_);
        //        dl_managed_tensor_->deleter = nullptr;
      }
    }
  }
  void ToDlTensor(const std::vector<std::vector<int64_t>>& input) {
    if (dl_managed_tensor_ && dl_managed_tensor_->deleter) {
      dl_managed_tensor_->deleter(dl_managed_tensor_);
      dl_managed_tensor_->deleter = nullptr;
    }
    dl_managed_tensor_ = new DLManagedTensor();

    // only CPU support now
    dl_managed_tensor_->dl_tensor.device.device_id = 0;
    dl_managed_tensor_->dl_tensor.device.device_type = DLDeviceType::kDLCPU;
    dl_managed_tensor_->dl_tensor.ndim = 2;
    dl_managed_tensor_->dl_tensor.strides = nullptr;
    int64_t* shape = new int64_t[2];
    shape[0] = input.size();
    shape[1] = input[0].size();
    // copy data
    int64_t* data = new int64_t[shape[0] * shape[1]];
    for (int i = 0; i < shape[0]; i++) {
      std::memcpy(data + i * shape[1], input[i].data(),
                  sizeof(int64_t) * shape[1]);
    }
    dl_managed_tensor_->dl_tensor.data = reinterpret_cast<void*>(data);

    dl_managed_tensor_->dl_tensor.shape = shape;
    dl_managed_tensor_->dl_tensor.dtype.lanes = 1;
    dl_managed_tensor_->dl_tensor.byte_offset = 0;
    dl_managed_tensor_->dl_tensor.dtype.code = DLDataTypeCode::kDLInt;
    dl_managed_tensor_->dl_tensor.dtype.bits = 64;
    dl_managed_tensor_->deleter = [](DLManagedTensor* self) {
      if (self) {
        if (self->dl_tensor.shape) {
          delete[] self->dl_tensor.shape;
        }
        if (self->dl_tensor.strides) {
          delete[] self->dl_tensor.strides;
        }
        if (self->dl_tensor.data) {
          delete[] static_cast<int64_t*>(self->dl_tensor.data);
        }
        delete self;
      }
    };
    dl_managed_tensor_->manager_ctx = nullptr;
  }

  void ToVectorData(std::vector<std::vector<int64_t>>& output) {
    assert(dl_managed_tensor_ && dl_managed_tensor_->dl_tensor.ndim == 2);
    // set data
    for (int i = 0; i < dl_managed_tensor_->dl_tensor.shape[0]; i++) {
      std::vector<int64_t> out(dl_managed_tensor_->dl_tensor.shape[1]);
      int data_size = dl_managed_tensor_->dl_tensor.shape[1] *
                      dl_managed_tensor_->dl_tensor.dtype.bits / 8;
      char* data_ptr =
          reinterpret_cast<char*>(dl_managed_tensor_->dl_tensor.data) +
          i * data_size;
      memcpy(out.data(), data_ptr, data_size);
      output.push_back(out);
    }
  }

  DLManagedTensor* GetDlTensor() { return dl_managed_tensor_; }

 private:
  DLManagedTensor* dl_managed_tensor_;
};

class ModelSession {
 public:
  ModelSession(ASCommonEngine* as_engine, allspark::GenerateConfig* gen_cfg,
               std::string& model_name, int request_id, int output_len)
      : as_engine_(as_engine),
        model_name_(model_name),
        request_id_(request_id),
        output_len_(output_len) {
    // config
    // allspark::DLTensorMap outputs;
    req_ = std::make_shared<AsEngine::RequestContent>();
    req_->config.max_length = gen_cfg->max_length;
    // auto now = std::chrono::high_resolution_clock::now();
    // auto timestamp =
    // std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
    // req_->config.uuid = std::to_string(request_id_) + std::string("_") +
    // std::to_string(timestamp);
    req_->config.early_stopping = gen_cfg->early_stopping;
    req_->config.top_k = gen_cfg->top_k;
    req_->config.top_p = gen_cfg->top_p;
    req_->config.eos_token_id = gen_cfg->eos_token_id;
    req_->config.stop_words_ids = gen_cfg->stop_words_ids;
    // req_->config.prefill_mode = gen_cfg->prefill_mode;
    req_->infer_type = AsEngine::RequestInferType::Generate;
    req_->mm_type = AsEngine::RequestMMType::TextInput;

    dl_manager_ = std::make_shared<DLTensorManager>();

    dl_manager_mask_ = std::make_shared<DLTensorManager>();
  }
  void forward() {
    // prepare data
    // #define ARRAY_VARIABLE CONCAT2(data_, request_id_)
    // #define ARRAY_SIZE_VARIABLE CONCAT3(data_, request_id_, _size)
    const int64_t* input = data_vec[request_id_];
    int batch_size = 1;
    int seq_len = data_size_vec[request_id_];
    std::vector<int64_t> in_vec(input, input + seq_len);
    std::vector<int64_t> in_mask(batch_size * seq_len, 1);
    if (output_len_ != -1) {
      req_->config.max_length = seq_len + output_len_;
    }
    dl_manager_->ToDlTensor({in_vec});
    dl_manager_mask_->ToDlTensor({in_mask});
    const DLTensorMap inputs = {
        {"input_ids", dl_manager_->GetDlTensor()},
        {"attention_mask", dl_manager_mask_->GetDlTensor()}};

    // start request
    req_->inputs = std::make_shared<DLTensorMap>(inputs);
    Timer forward_time;
    forward_time.Start();
    status_ =
        as_engine_->StartRequest(model_name_.c_str(), req_, &handle_, &queue_);
    if (status_ != allspark::AsStatus::ALLSPARK_SUCCESS) {
      // TBD
      std::cout << "Start request error" << std::endl;
      session_success_ = false;
      return;
    }
    std::shared_ptr<allspark::AsEngine::GeneratedElements> ele = nullptr;
    while (true) {
      ele = queue_->Get();
      if (ele == nullptr) {
        if (queue_->GenerateStatus() ==
            allspark::AsEngine::GenerateRequestStatus::GenerateFinished) {
          session_success_ = true;
          decoder_time_ = forward_time.Stop();
          break;
        } else if (queue_->GenerateStatus() ==
                   allspark::AsEngine::GenerateRequestStatus::
                       GenerateInterrupted) {
          std::cout << "GenerateInterrupted... request id: " << request_id_
                    << std::endl;
          session_success_ = false;
          break;
        } else {
          std::cout << "Weird... requst id: " << request_id_ << ", status: "
                    << static_cast<int>(queue_->GenerateStatus()) << std::endl;
          session_success_ = false;
          break;
        }
      }
      in_size_ = seq_len;
      if (out_size_ == 0 && ele->ids_from_generate.size() > 0) {
        // prefill finished
        context_size_ = out_size_;
        context_time_ = forward_time.Stop();
      }
      auto new_size = ele->ids_from_generate.size();
      ;
      out_size_ += new_size;
      summary.AddTotolOutput(new_size);
      LOG(INFO) << "req_id: " << request_id_ << " seq_len: " << seq_len
                << " out size: " << out_size_
                << " context_time: " << context_time_
                << " queue_->GenerateStatus(): "
                << (int)queue_->GenerateStatus();
    }
    as_engine_->StopRequest(model_name_.c_str(), handle_);
    as_engine_->ReleaseRequest(model_name_.c_str(), handle_);
  }
  int get_in_size() { return in_size_; }
  int get_out_size() { return out_size_; }
  int get_context_size() { return context_size_; }
  int get_context_time() { return context_time_; }
  int get_decoder_time() { return decoder_time_; }
  bool is_session_success() { return session_success_; }

  int get_req_id() { return request_id_; }

 private:
  int request_id_;
  int in_size_ = 0;
  int out_size_ = 0;
  int context_size_ = 0;
  int output_len_ = -1;
  double context_time_ = 0;
  double decoder_time_ = 0;
  RequestHandle_t handle_;
  AsEngine::ResultQueue_t queue_;
  std::string model_name_;
  std::shared_ptr<AsEngine::RequestContent> req_;
  allspark::AsStatus status_;
  bool session_success_ = true;
  ASCommonEngine* as_engine_;
  std::shared_ptr<DLTensorManager> dl_manager_;
  std::shared_ptr<DLTensorManager> dl_manager_mask_;
};

void ThreadCount(int time_sec) {
  std::cout << "####ready to sleep seconds: " << time_sec << std::endl;
  std::this_thread::sleep_for(std::chrono::seconds(time_sec));
  std::cout << "####after wake up, ready to stop..." << std::endl;
  // 设置标志位为true，通知其他线程退出
  {
    std::unique_lock<std::mutex> lock(mtx);
    isFinished = true;
  }
  cv.notify_all();
}

void ThreadQueue(float freq, std::string model_name, ASCommonEngine* as_engine,
                 allspark::GenerateConfig* gen_cfg, int output_len) {
  // 向队列中发消息，直到计时完成
  int request_id = 0;
  Timer timer_queue;
  timer_queue.Start();
  // push times, for debug
  int times = 1;
  while (true) {
    // wait to push
    int ms = 1000 / freq;
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
    std::unique_lock<std::mutex> lock(mtx);
    if (isFinished) {
      summary.PrintSummary(timer_queue.Stop());
      std::cout << "ThreadQueue is going to shutdown..." << std::endl;
      break;
    }
    // data_size is the total size of dataset
    request_id = request_id % data_size;
    std::shared_ptr<ModelSession> ptr = std::make_shared<ModelSession>(
        as_engine, gen_cfg, model_name, request_id++, output_len);
    // std::shared_ptr<ModelSession> ptr = std::make_shared<ModelSession>(
    //     as_engine, gen_cfg, model_name, request_id);
    // std::shared_ptr<ModelSession> ptr = std::make_shared<ModelSession>(
    //     as_engine, gen_cfg, model_name, 353);
    if (times > 0) {
      // times--;
      messageQueue.push(ptr);
      cv.notify_one();
    }
  }
}

void ThreadLoop(int id) {
  while (true) {
    std::shared_ptr<ModelSession> ptr;
    {
      std::unique_lock<std::mutex> lock(mtx);

      // 等待消息或者计时完成
      cv.wait(lock, []() { return !messageQueue.empty() || isFinished; });

      // 如果计时完成并且队列为空，退出线程
      if (isFinished) {
        std::cout << "thread id: " << id << " going to shutdown..."
                  << std::endl;
        break;
      }

      // 取出消息
      ptr = messageQueue.front();
      messageQueue.pop();
      std::cout << "message size: " << messageQueue.size()
                << " thread id: " << id << std::endl;
    }

    // forward
    if (ptr) {
      std::cout << "start forward req id: " << ptr->get_req_id() << std::endl;
      ptr->forward();
      std::cout << "after forward req id: " << ptr->get_req_id() << std::endl;
      summary.DoCalc(ptr->get_in_size(), ptr->get_out_size(),
                     ptr->get_context_time(), ptr->get_decoder_time(),
                     ptr->is_session_success());
      std::cout << "after summary req id: " << ptr->get_req_id() << std::endl;
    }
    // check again...
    //  if (isFinished) break;
  }
}

std::string GenerateString(std::string device, int n) {
  std::ostringstream oss;
  oss << device + ":0";

  for (int i = 1; i < n; ++i) {
    oss << "," << i;
  }

  return oss.str();
}

AsMHAPrefill GetPrefillMode(const std::string& fill_name) {
  if (fill_name == "default") {
    return AsMHAPrefill::AsPrefillDefault;
  } else if (fill_name == "flashv2") {
    return AsMHAPrefill::AsPrefillFlashV2;
  } else if (fill_name == "xformer") {
    return AsMHAPrefill::AsPrefillXformer;
  } else {
    LOG(ERROR) << "fill mode error: got : " << fill_name
               << " but only support (default/flashv2/xformer) please change "
                  "key word.";
    abort();
  }
}

AsCacheMode GetCacheMode(const std::string& fill_name) {
  if (fill_name == "default") {
    return AsCacheMode::AsCacheDefault;
  } else if (fill_name == "int8") {
    return AsCacheMode::AsCacheQuantI8;
  } else if (fill_name == "uint4") {
    return AsCacheMode::AsCacheQuantU4;
  } else {
    LOG(ERROR) << "cache mode error: got : " << fill_name
               << " but only support (default/int8) please change key word.";
    abort();
  }
}

int main(int argc, char** argv) {
  int opt = 0;
  int total_time = 0;
  float freq = 0;
  int req_num = 0;
  int max_batch = 0;
  int engine_max_length = 8192;
  std::string model_path("");
  std::string model_name("");
  std::string model_type = std::string("Qwen_v20");
  AsMHAPrefill prefill_mode = AsMHAPrefill::AsPrefillDefault;
  AsCacheMode kv_cache_mode = AsCacheMode::AsCacheDefault;
  int enable_flash_attention = 1;
  int device_num = 8;
  float top_k = 0;
  float top_p = 0.8;
  int enable_warmup = 1;
  int output_len = -1;
  int early_stopping = 1;
  int num_thread = -1;
  std::string device_type = "CUDA";
  std::string matmul_precision = "highest";
  while ((opt = getopt(argc, argv,
                       "ht:f:r:b:d:l:m:M:F:P:N:k:p:w:C:a:c:o:s:n:")) != -1) {
    switch (opt) {
      case 'h':
        std::cout << "\nDESCRIPTION:\n"
                  << "------------\n"
                  << "Example application demonstrating how to load and "
                     "execute "
                     "an allspark benchmark test using the C++ API.\n"
                  << "\n\n"
                  << "REQUIRED ARGUMENTS:\n"
                  << "-------------------\n"
                  << "  -t <int> total time (seconds) for the test\n"
                  << "  -f <float> frequency to generate request, e.g. 2 means "
                     "twice per second\n"
                  << "  -r <int> thread nums to send request to allspark\n"
                  << "  -b <int> max batch size of allspark generate config\n"
                  << "  -d <String> directory to model file\n"
                  << "  -m <String> model name of the model file,"
                     "  will load ${directory}/${modelname}.asparam.\n"
                  << "\n"
                  << "OPTIONAL ARGUMENTS:\n"
                  << "-------------------\n"
                  << "  -C <0/1> run model on cpu, default 0\n"
                  << "  -c <default/int8> set span kv cache mode\n"
                  << "  -M <String> model type. e.g. [m6_7b, m6_14b, "
                     " m6_50b, m6_72b, m6_200b], only support m6_7b and "
                     " m6_200b for now, default is m6_200b \n"
                  << "  -F <default/flashv2/xformer> enable flash attention, "
                     "default: default\n"
                  << "  -N <int> total device number, default is 8\n"
                  << "  -k <float> topk,default 0\n"
                  << "  -l <int> max engine length, default 8192\n"
                  << "  -p <float> topp,default 0.8\n"
                  << "  -w <0/1> enable warmup, default 1\n"
                  << "  -o <int> set output length, used for test a fixed "
                     "output length\n"
                  << "  -s <0/1> enable early stop and stop words. default 1\n"
                  << "  -n <int> infer thread nums, only valid in cpu inferer\n"
                  << "  -a <int> set matmul precision, default 0 (0:fp32, "
                     "1:tf32, 2:bf16, 3:fp16)\n"
                  << std::endl;
        std::exit(0);
      case 't':
        total_time = atoi(optarg);
        break;
      case 'f':
        freq = atof(optarg);
        break;
      case 'r':
        req_num = atoi(optarg);
        break;
      case 'b':
        max_batch = atoi(optarg);
        break;
      case 'd':
        model_path = optarg;
        break;
      case 'm':
        model_name = optarg;
        break;
      case 'l':
        engine_max_length = atoi(optarg);
        break;
      case 'M':
        model_type = optarg;
        break;
      case 'F':
        prefill_mode = GetPrefillMode(std::string(optarg));
        break;
      case 'N':
        device_num = atoi(optarg);
        break;
      case 'k':
        top_k = atof(optarg);
        break;
      case 'p':
        top_p = atof(optarg);
        break;
      case 'w':
        enable_warmup = atoi(optarg);
        break;
      case 'C':
        device_type = "CPU";
        break;
      case 'c':
        kv_cache_mode = GetCacheMode(std::string(optarg));
        break;
      case 'a':
        if (atoi(optarg) == 1)
          matmul_precision = "high";
        else if (atoi(optarg) == 2)
          matmul_precision = "medium";
        else if (atoi(optarg) == 3)
          matmul_precision = "medium_fp16";
        else
          matmul_precision = "highest";
        break;
      case 'o':
        output_len = atoi(optarg);
        break;
      case 's':
        early_stopping = atoi(optarg);
        break;
      case 'n':
        num_thread = atoi(optarg);
        break;
      default:
        std::cout << "Invalid parameter specified. " << opt
                  << " Please run "
                     "model_stress_test with "
                     "the -h flag to see required arguments"
                  << std::endl;
        std::exit(-1);
    }
  }
  // check params
  if (total_time == 0 || freq == 0.f || req_num == 0 || model_path.empty() ||
      model_name.empty()) {
    std::cout << "Invalid parameter specified. Please run "
                 "model_stress_test with "
                 "the -h flag to see required arguments"
              << std::endl;
    std::exit(-1);
  }
  ASCommonEngine as_engine;
  LOG(INFO) << "total_time: " << total_time << " freq: " << freq
            << " req_num: " << req_num << " max_batch: " << max_batch
            << " flash attention: " << (int)prefill_mode
            << " model tpye: " << model_type << " device num: " << device_num
            << std::endl;
  std::string compute_unit = GenerateString(device_type, device_num);

  AsModelConfigBuilder builder;

  auto as_model_config =
      builder.withModelName(model_name)
          .withModelPath(model_path + "/" + model_name + ".asgraph")
          .withWeightsPath(model_path + "/" + model_name + ".asparam")
          .withComputeUnit(compute_unit)
          .withEngineMaxBatch(max_batch)
          .withEngineMaxLength(engine_max_length)
          .withPrefillMode(prefill_mode)
          .withCacheSpanNumGrow(0)
          .withCacheSpanNumInit(0)
          .withCacheSpanSize(16)
          .withCacheMode(kv_cache_mode)
          .build();

  if (device_type == "CPU") {
    as_model_config = builder.withMatmulPrecision(matmul_precision).build();
    if (num_thread > 0) {
      as_model_config = builder.withNumThreads(num_thread).build();
    }
  }
  AS_CHECK(as_engine.BuildModelFromConfigStruct(as_model_config));

  AS_CHECK(as_engine.StartModel(model_name.c_str()));

  // generate config
  // TBD: all params should support to set via cmd params
  allspark::GenerateConfig gen_config;
  gen_config.max_length = engine_max_length;
  if (early_stopping == 1) {
    gen_config.early_stopping = true;
    gen_config.eos_token_id = 151643;
    gen_config.stop_words_ids = {{151643}, {151644}, {151645}};
  } else {
    gen_config.early_stopping = false;
  }
  gen_config.top_k = top_k;
  gen_config.top_p = top_p;

  std::cout << "Start Model Done..." << std::endl;
  Timer timer;
  timer.Start();
  std::thread thread_count(ThreadCount, total_time);
  std::thread thread_queue(ThreadQueue, freq, model_name, &as_engine,
                           &gen_config, output_len);
  std::vector<std::thread> thread_loops;

  for (int i = 0; i < req_num; i++) thread_loops.emplace_back(ThreadLoop, i);

  thread_count.join();
  thread_queue.join();

  for (auto& thread : thread_loops) thread.join();

  auto total_time_ms = timer.Stop();
  std::cout << "queue left size: " << messageQueue.size()
            << " total_time_ms: " << total_time_ms << std::endl;
  while (messageQueue.size() > 0) {
    // 请求没有处理的队列
    summary.DoCalc(0, 0, 0, 0, false);
    messageQueue.pop();
  }
  AS_CHECK(as_engine.StopModel(model_name.c_str()));

  std::cout << "profile info: \n"
            << as_engine.GetOpProfilingInfo(model_name.c_str()) << std::endl;

  std::cout << "total request: " << summary.total_request_
            << " failure request: " << summary.failure_request_
            << " total input: " << summary.total_input_
            << " total output: " << summary.total_output_ << std::endl;

  std::cout << "avg_input_len: "
            << summary.total_input_ /
                   float(summary.total_request_ - summary.failure_request_)
            << " avg_output_len: "
            << summary.total_output_ /
                   float(summary.total_request_ - summary.failure_request_)
            << " request_throughput: "
            << (summary.total_request_ - summary.failure_request_) /
                   total_time_ms * 1000
            << " output_throughput: "
            << summary.total_output_ / total_time_ms * 1000 << " failed rate: "
            << summary.failure_request_ / (float)summary.total_request_
            << std::endl;
  return 0;
}
