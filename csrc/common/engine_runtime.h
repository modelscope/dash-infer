/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    engine_runtime.h
 */

#pragma once

#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>

#include "allspark.h"
#include "core/tensor/tensor.h"
#include "engine_control_message.h"
#include "utility/blockingconcurrentqueue.h"
#include "utility/concurrentqueue.h"

/// @brief if defined, user input span size will be ignored
// #define FIXED_SPAN_SIZE 128

/// @brief if defined, span managers run concurrently
#define CONFIG_CONCURRENT_SPAN

// XXX: keep this header internal, not public interface !
// put some data structure define in this header.
//
//
namespace allspark {

#ifdef ENABLE_JSON_MODE
namespace util {
class FormatEnforcer;
}
#endif

class ModelWeightHandler;

class ModelControlState final {
  std::unique_ptr<std::thread> loop_thread_;
  std::unique_ptr<std::thread> prefill_thread_;
  std::unique_ptr<std::thread> decode_thread_;

 public:
  std::string model_name;

  moodycamel::BlockingConcurrentQueue<EngineControlMessage> msg_queue;
  moodycamel::BlockingConcurrentQueue<EngineControlMessage> msg_queue_prefill;
  moodycamel::BlockingConcurrentQueue<EngineControlMessage> msg_queue_decode;
  std::atomic<int> msg_queue_size;

  std::unordered_map<std::string, std::shared_ptr<RequestHandle>>
      request_handle_map;
  std::unordered_map<std::string, std::shared_ptr<AsEngine::ResultQueue>>
      result_queue_map;
  std::mutex map_lock_;  // mutex for request_handle_map and result_queue_map

  std::atomic<bool> model_stopping =
      false;                                // after GracefulStopModel called...
  std::atomic<bool> model_stopped = false;  // after GracefulStopModel is done.
  std::shared_ptr<ModelWeightHandler> weight_handler_;

  explicit ModelControlState(const std::string& name);

  template <typename Func, typename... Args>
  void StartLoop(Func&& func, Args&&... args) {
    loop_thread_ = std::make_unique<std::thread>(std::forward<Func>(func),
                                                 std::forward<Args>(args)...);
  }

  template <typename Func, typename... Args>
  void StartPrefillLoop(Func&& func, Args&&... args) {
    prefill_thread_ = std::make_unique<std::thread>(
        std::forward<Func>(func), std::forward<Args>(args)...);
  }

  template <typename Func, typename... Args>
  void StartDecodeLoop(Func&& func, Args&&... args) {
    decode_thread_ = std::make_unique<std::thread>(std::forward<Func>(func),
                                                   std::forward<Args>(args)...);
  }

  void StopLoop();

 private:
  // 将拷贝构造函数和拷贝赋值运算符声明为私有
  ModelControlState(const ModelControlState&);
  ModelControlState& operator=(const ModelControlState&);
};

class AsTensor;

class RequestHandle {
 public:
  std::string request_uuid;
  size_t generate_length = 0;
  size_t context_length = 0;
  bool need_set_generating_stat = true;
  const std::chrono::time_point<std::chrono::steady_clock> create_ts;

  RequestHandle() : create_ts(std::chrono::steady_clock::now()) {}

  // copy input from user's dltensor to our as tensor.
  // to avoid manage dltensor 's reference

  std::shared_ptr<TensorMap> inputs_internal;
  AsEngine::RequestMMType mm_type_internal;
  TensorListMap mm_embedding_internal;  /// multiple media embedding tensor
#ifdef ENABLE_JSON_MODE
  std::shared_ptr<util::FormatEnforcer> format_enforcer;
#endif
};

namespace stat_key {
/// time stamp part
static const char* KEY_REQUEST_TIME_TS = "arrival_time";
static const char* KEY_FIRST_TOKEN_TIME_TS = "first_token_time";
static const char* KEY_LAST_TOKEN_TIME_TS = "last_token_time";

/// duration part
static const char* KEY_TTFT_MS = "time_to_first_token";
static const char* KEY_SCHEDULE_TIME_MS = "time_in_queue";
static const char* KEY_TOTAL_TIME_MS = "finished_time";

// input/output counter part.
static const char* KEY_INPUT_LEN = "context_token_length";
static const char* KEY_OUTPUT_LEN = "generated_token_length";
static const char* KEY_INPUT_CACHE_LEN = "context_cached_length";

// speed part.
static const char* KEY_CONTEXT_TPS = "context_tps";
static const char* KEY_GENERATE_TPS = "generate_tps";
};  // namespace stat_key

class ResultQueueImpl : public AsEngine::ResultQueue {
 public:
  typedef std::shared_ptr<AsEngine::GeneratedElements> GenerateElementPtr;
  ResultQueueImpl(const std::string& uuid);
  ~ResultQueueImpl();
  AsEngine::GenerateRequestStatus GenerateStatus() override;
  size_t GeneratedLength() override;

  void InitStatInfo();
  StatInfo RequestStatInfo() override;

  void UpdateStatInfo(const std::string& key, long value) {
    if (request_stat_.find(key) == request_stat_.end()) {
      DLOG(INFO) << "RequestQueue: stat key not found : " << key;
      return;
    }
    request_stat_[key].store(value);
  }

  GenerateElementPtr Get() override;
  GenerateElementPtr Get(int timeout_ms) override;
  GenerateElementPtr GetNoWait() override;

  void SetStatus(AsEngine::GenerateRequestStatus);

  void AppendGenerateElement(
      std::shared_ptr<AsEngine::GeneratedElements> new_ele);

  // private class can add function that push elements.
 private:
  GenerateElementPtr GetWithTimeout(int timeout_ms);
  unsigned long CountSpinCounts(std::chrono::milliseconds duration);
  moodycamel::ConcurrentQueue<AsEngine::GeneratedElements> store_queue_;
  std::atomic<AsEngine::GenerateRequestStatus> status_;
  std::atomic<size_t> generate_length_;

  std::string request_uuid_ = "Unknown-UUID";
  std::atomic<bool> closed_ = false;
  std::atomic<int> get_user = 0;

  std::map<std::string, std::atomic<long>> request_stat_;

  unsigned long spin_lock_max_counts_ =
      200 * 1000;  // default value for spin count

  std::mutex
      queue_lock_;  // notice: this lock should only used in request_stat_.

  typedef std::unique_lock<std::mutex> queue_read_lock_t;
  typedef std::unique_lock<std::mutex> queue_write_lock_t;
};

class SpanCacheConfig final {
  SpanCacheConfig(AsCacheMode mode_, int span_size_, int span_num_init_,
                  int span_num_grow_);

 public:
  using Ptr = std::shared_ptr<SpanCacheConfig>;
  using ConstPtr = std::shared_ptr<const SpanCacheConfig>;

  static Ptr Create(AsCacheMode mode, int span_size, int span_num_init,
                    int span_num_grow);

  static std::string CacheMode2String(AsCacheMode mode) {
    switch (mode) {
      case AsCacheMode::AsCacheDefault:
        return "AsCacheDefault";
      case AsCacheMode::AsCacheQuantI8:
        return "AsCacheQuantI8";
      case AsCacheMode::AsCacheQuantU4:
        return "AsCacheQuantU4";
      default:
        return "AsCacheUnknown";
    }
  }

  const AsCacheMode mode;
  const int span_size;
  const int span_num_init;
  const int span_num_grow;
};

inline static std::string to_string(
    const allspark::SpanCacheConfig& cache_cfg) {
  std::ostringstream os;
  os << "Span Cache Config : { ";
  os << "mode: " << SpanCacheConfig::CacheMode2String(cache_cfg.mode) << ", ";
  os << "span_size: " << cache_cfg.span_size << ", ";
  os << "span_num_init: " << cache_cfg.span_num_init << ", ";
  os << "span_num_grow: " << cache_cfg.span_num_grow;
  os << " }";
  return os.str();
}

inline static std::string to_string(const allspark::GenerateConfig& gen_cfg) {
  std::ostringstream os;
  os << "Generation Config : { ";
  os << "do_sample: " << gen_cfg.do_sample << ", ";
  os << "num_beams: " << gen_cfg.num_beams << ", ";
  os << "num_return_sequences: " << gen_cfg.num_return_sequences << ", ";
  os << "early_stopping: " << gen_cfg.early_stopping << ", ";

  os << "stop_words_ids: [";
  for (const auto& vec : gen_cfg.stop_words_ids) {
    os << "[";
    for (const auto& id : vec) {
      os << id << ",";
    }
    os << "], ";
  }
  os << "], ";

  os << "eos_token_id: " << gen_cfg.eos_token_id << ", ";
  os << "seed: " << gen_cfg.seed << ", ";

  os << "bad_words_ids: [";
  for (const auto& vec : gen_cfg.bad_words_ids) {
    os << "[";
    for (const auto& id : vec) {
      os << id << ",";
    }
    os << "], ";
  }
  os << "], ";

  os << "temperature: " << gen_cfg.temperature << ", ";
  os << "top_k: " << gen_cfg.top_k << ", ";
  os << "top_p: " << gen_cfg.top_p << ", ";
  os << "repetition_penalty: " << gen_cfg.repetition_penalty << ", ";
  os << "length_penalty: " << gen_cfg.length_penalty << ", ";
  os << "presence_penalty: " << gen_cfg.presence_penalty << ", ";
  os << "frequency_penalty: " << gen_cfg.frequency_penalty << ", ";
  os << "suppress_repetition_in_generation: "
     << gen_cfg.suppress_repetition_in_generation << ", ";
  os << "no_repeat_ngram_size: " << gen_cfg.no_repeat_ngram_size << ", ";
  os << "logprobs: " << gen_cfg.logprobs << ", ";
  os << "top_logprobs: " << gen_cfg.top_logprobs << ", ";
  os << "min_length: " << gen_cfg.min_length << ", ";
  os << "max_length: " << gen_cfg.max_length << ", ";
  os << "lora_name: " << gen_cfg.lora_name << ", ";
  os << "enable_tensors_from_model_inference: "
     << gen_cfg.enable_tensors_from_model_inference << ", ";
  // For mm_info we assume some way to print it is provided
  if (gen_cfg.mm_info != nullptr) {
    os << "mm_info: " << gen_cfg.mm_info << ", ";
  } else {
    os << "mm_info: null, ";
  }

  os << "input_len: " << gen_cfg.input_len << ", ";
  os << " }";

  return os.str();
}

inline static std::string to_string(
    const allspark::AsEngine::RequestContent& content) {
  using namespace allspark;
  std::ostringstream os;
  os << " token input length: ";
  if (content.inputs) {
    long input_length = (*content.inputs)["input_ids"]->dl_tensor.shape[1];
    os << input_length << ", ";
  } else {
    os << " 0 (nullptr)"
       << ", ";
  }
  if (content.mm_type == AsEngine::RequestMMType::MultiMediaTypeRichText) {
    os << " MMType: Rich "
       << ", ";
  } else if (content.mm_type == AsEngine::RequestMMType::TextInput) {
    os << " MMType: Text "
       << ", ";
  }
  os << to_string(content.config);

  return os.str();
}

};  // namespace allspark
