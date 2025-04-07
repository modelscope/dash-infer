/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    allspark.h
 */

#pragma once

#include <hash_fun.h>

#include <cmath>
#include <functional>
#include <map>
#include <memory>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "allspark_check.h"  // NOLINT
#include "dlpack.h"          // NOLINT

namespace allspark {

class AsEngineImpl;
class MultiMediaInfoImpl;
class RequestHandle;
typedef RequestHandle* RequestHandle_t;

// AllSpark Inference Engine Interface
using DLTensorMap = std::map<std::string, DLManagedTensor*>;
using DLTensorListMap = std::map<std::string, std::vector<DLManagedTensor*>>;
class ITensor {
 public:
  virtual DLManagedTensor* ToDLPack() const = 0;

 protected:
  ~ITensor() {}
};
enum class AsMHAPrefill {
  AsPrefillDefault = 0,
  AsPrefillFlashV2 = 10,
  AsPrefillXformer = 11,
};

enum class VocabType : int {
  /* Tokenization algorithms */
  VOCAB_TYPE_WPM = 0,  // Word Piece
  VOCAB_TYPE_SPM = 1,  // Sentence Piece
  VOCAB_TYPE_UGM = 2,  // Unigram
  VOCAB_TYPE_BPE = 3,  // Byte Pair Encoding
};

class MultiMediaInfo {
 public:
  MultiMediaInfo();
  ~MultiMediaInfo();
  AsStatus add_multimedia_content(std::string key,
                                  std::vector<std::vector<float>> data_list,
                                  std::vector<std::vector<int64_t>> shape_list);
  AsStatus add_multimedia_content(
      std::string key, std::vector<DLManagedTensor*>& dl_tensor_py_list);
  AsStatus set_multimedia_type(int multimedia_type_new);
  std::map<std::string,
           std::vector<std::pair<std::vector<float>, std::vector<int64_t>>>>&
  get_tensor_list_map();
  DLTensorListMap get_multimedia_map();

 private:
  std::unique_ptr<MultiMediaInfoImpl> multi_media_info_impl_;
};

enum class AsCacheMode {
  AsCacheDefault = 0,
  AsCacheQuantI8 = 1,
  AsCacheQuantU4 = 2,
};

enum class AsEvictionStrategy {
  MaxLength = 0,
  Random = 1,
};
enum class AsSchedulingStrategy {
  ContextPriority = 0,
  Balance = 1,
};
/**
 * Converted asparam, asgraph file info class.
 */

class AsFileInfo {
 public:
  std::string create_version_graph;    /// allspark version of graph file
  std::string create_version_param;    /// allspark version of param file
  std::string current_version_engine;  // allspark engine version.
};

struct GenerateConfig {
  // Global options
  bool do_sample = true;  ///< Flag to enable/disable sampling in generation,
                          ///< current this option must be on.
  int num_beams = 1;  ///< Number of beams to use in beam search, beam search
                      ///< cannot be support for current version
  int num_return_sequences =
      1;  ///< Beam Search Related options: Number of sequences to return, not
          ///< work for current version
  // Generation stop control options
  bool early_stopping =
      true;  ///< Stop generation when the EOS token is encountered.
  std::vector<std::vector<int64_t>>
      stop_words_ids;      ///< IDs of words that signal the end of generation.
  int eos_token_id = 102;  ///< ID of the EOS (end of sequence) token, specifiy
                           ///< it according your model
  // Generation content control options.
  unsigned long long seed = 0;  ///< Seed for random number generation.
  std::vector<std::vector<int>>
      bad_words_ids;  ///< IDs of words to avoid during generation, will supress
                      ///< thoese ids's generation.
  float temperature = 1.0;         ///< Sampling temperature.
  int top_k = 50;                  ///< Top-K sampling hyperparameter.
  float top_p = 1.0;               ///< Top-P sampling hyperparameter.
  float repetition_penalty = 1.0;  ///< Penalty for word repetition.
  float length_penalty =
      1.0;  ///< Penalty for the length of the sequence.Only use in beamsearch
  float presence_penalty = 0;   ///< Penalty for the presence of words.
  float frequency_penalty = 0;  ///< Penalty for the frequency of words.
  bool suppress_repetition_in_generation =
      false;  ///< Suppress repetition of words during generation,
              /// if this option is on,
              /// engine will use presence_penalty rather than
              /// repetition_penalty
  int no_repeat_ngram_size = 0;  ///< Size of n-gram that must not repeat.
  bool logprobs =
      false;  /// Whether to return log probabilities of the output tokens or
              /// not. If true, returns the log probabilities of each output
              /// token returned in the content of message. This option is
              /// currently not available on the gpt-4-vision-preview model.
  int top_logprobs =
      0;  /// An integer between 0 and 5 specifying the number of most likely
          /// tokens to return at each token position, each with an associated
          /// log probability. logprobs must be set to true if this parameter is
          /// used.
  // Generation Length Control options
  int min_length =
      0;  ///< Minimum total length toal length, 0 means disable this filter
  int max_length = 20;  ///< Maximum total length,  max_length =  input(prefill)
                        ///< + output(generation)
  // other options
  std::string lora_name = "";
  MultiMediaInfo* mm_info = nullptr;
  // for guided decoding
  std::map<std::string, std::string>
      response_format;  /// specify output format, currently only JSON supported
  std::map<std::string, int> vocab;  /// model tokenizer vocab
  VocabType vocab_type;              /// model tokenizer type
  // deprecated options
  std::string uuid =
      "Default-UUID(deprecated)";  ///< deprecated option, will filled by engine
                                   ///< internally.
  std::string user_request_id =
      "Default-User-UUID";  /// This uuid should  only use for logging.

  int input_len = 0;  ///< deprecated option, will filled by engine internally.
  bool enable_tensors_from_model_inference = false;
};

class AsModelConfig {
 public:
  static constexpr int default_engine_max_length = 2048;
  static constexpr int default_engine_max_batch = 32;
  static constexpr int default_engine_max_prefill_length = 0;
  static constexpr int default_swap_threshold = -1;
  static constexpr int default_num_thread = 0;
  static constexpr const char* default_matmul_precision = "highest";
  static constexpr const char* default_compute_unit = "CUDA:0";
  static constexpr int default_span_size = 128;
  static constexpr AsMHAPrefill default_prefill_mode =
      AsMHAPrefill::AsPrefillDefault;
  static constexpr AsCacheMode default_kv_cache_mode =
      AsCacheMode::AsCacheDefault;
  static constexpr AsEvictionStrategy default_eviction_strategy =
      AsEvictionStrategy::MaxLength;
  static constexpr AsSchedulingStrategy default_scheduling_strategy =
      AsSchedulingStrategy::ContextPriority;
  static constexpr int default_lora_max_rank = 64;
  static constexpr int default_lora_max_num = 5;

  AsModelConfig();
  AsModelConfig(
      std::string in_model_name, std::string in_model_path,
      std::string in_weights_path, std::string in_compute_unit,
      int in_engine_max_length = default_engine_max_length,
      int in_engine_max_batch = default_engine_max_batch,
      int in_engine_max_prefill_length = default_engine_max_prefill_length,
      int64_t in_swap_threshold = default_swap_threshold,
      bool in_text_graph = false, int in_num_threads = default_num_thread,
      std::string in_matmul_precision = default_matmul_precision,
      std::vector<std::string> lora_names = std::vector<std::string>(),
      int in_cache_span_size = default_span_size,
      int in_cache_span_num_init = 0, int in_cache_span_num_grow = 0,
      bool enable_prefix_cache = true, int prefix_cache_ttl = 300,
      AsMHAPrefill in_prefill_mode = default_prefill_mode,
      AsCacheMode in_cache_mode = default_kv_cache_mode,
      AsEvictionStrategy in_eviction_strategy = default_eviction_strategy,
      AsSchedulingStrategy in_scheduling_strategy = default_scheduling_strategy,
      bool enable_sparsity_matmul = false,
      int lora_max_rank = default_lora_max_rank,
      int lora_max_num = default_lora_max_num);

  bool operator==(const AsModelConfig& other) const {
    return model_name == other.model_name && model_path == other.model_path &&
           weights_path == other.weights_path &&
           compute_unit == other.weights_path &&
           engine_max_length == other.engine_max_length &&
           engine_max_batch == other.engine_max_batch &&
           engine_max_prefill_length == other.engine_max_prefill_length &&
           num_threads == other.num_threads &&
           matmul_precision == other.matmul_precision &&
           swap_threshold == other.swap_threshold &&
           text_graph == other.text_graph && is_lora_cfg == other.is_lora_cfg &&
           cache_span_size == other.cache_span_size &&
           cache_span_num_init == other.cache_span_num_init &&
           cache_span_num_grow == other.cache_span_num_grow &&
           cache_mode == other.cache_mode &&
           enable_prefix_cache == other.enable_prefix_cache &&
           prefix_cache_ttl == other.prefix_cache_ttl &&
           prefill_mode == other.prefill_mode &&
           eviction_strategy == other.eviction_strategy &&
           enable_sparsity_matmul == other.enable_sparsity_matmul &&
           lora_max_rank == other.lora_max_rank &&
           lora_max_num == other.lora_max_num;
  }

  std::string ToString() const;  // detail see as_engine.cpp

 public:
  std::string model_name;
  std::string model_path;
  std::string weights_path;
  std::string compute_unit = default_compute_unit;
  std::string matmul_precision = default_matmul_precision;
  int num_threads = default_num_thread;
  bool is_lora_cfg = false;  // 表示该配置的类型，是for基模的，还是for LoRA的

  int64_t swap_threshold =
      default_swap_threshold;  // -1: 不swap  0: swap全部tensors
                               // 其他正整数：仅swap超过该阈值的tensors
  int engine_max_length = default_engine_max_length;
  int engine_max_batch = default_engine_max_batch;
  int engine_max_prefill_length = default_engine_max_prefill_length;
  int cache_span_size = default_span_size;
  int cache_span_num_init = 0;  // deprecated
  int cache_span_num_grow = 0;  // deprecated
  bool enable_prefix_cache = true;
  int prefix_cache_ttl = 300;
  AsCacheMode cache_mode = default_kv_cache_mode;
  AsMHAPrefill prefill_mode = default_prefill_mode;
  AsEvictionStrategy eviction_strategy = default_eviction_strategy;
  AsSchedulingStrategy scheduling_strategy = default_scheduling_strategy;
  bool text_graph = false;
  bool enable_sparsity_matmul = false;
  std::vector<std::string> lora_names;  // deprecated, not used any longer
  int lora_max_rank = default_lora_max_rank;
  int lora_max_num = default_lora_max_num;
};

/**
 * Engine level statistic information
 *
 */
class AsEngineStat {
 public:
  AsEngineStat() {}
  AsEngineStat(std::string in_model_name)
      : model_name(std::move(in_model_name)) {}

  // bool operator==(const AsEngineStat& other) const {
  //     return model_name == other.model_name &&
  // }
  std::string ToString() const;
  std::map<std::string, std::string> ToMap() const;

 public:
  std::string model_name;
  int64_t total_span = 0;
  int64_t used_span = 0;
  int64_t free_span = 0;
  int64_t total_token = 0;
  int64_t free_token = 0;
  int64_t span_size = 0;
  int pendding_request = 0;
  int running_request = 0;
  int64_t total_device_memory_pool_size = 0;
  int64_t used_device_memory_pool_size = 0;

  int64_t total_generated_token = 0;
  int64_t total_prefill_token = 0;
  float generate_token_persec = 0;
  float process_token_persec = 0;
  float token_usage_percentage = 0;
  int64_t prefix_cache_hit_token = 0;
  int64_t prefix_cache_miss_token = 0;
  float prefix_cache_hit_rate = 0;
  float prefix_cache_miss_rate = 0;
  // size_t TotalDeviceMemoryPoolSize;
  // size_t UsedDeviceMemoryPoolSize;
};

struct TensorAttribute {
  int sparse_type = 0;
  int split_mode = 0;
  std::vector<int> shape;
  std::vector<int> group_list;
  char dtype;
  int word_size;
  int nnz = 0;
};
/** The Generate Output Callback for Async request
 *
 * This callback keeps updating the output of a generation result of a async
 * request
 *
 * @param string Request UUID: the generate uuid to identify which request's
 * callback.
 * @param DLTensorMap *output: output tensor
 * @param bool isFinished:  the result in this batched request is finished or
 * not.
 */
typedef std::function<void(const char*, DLTensorMap*, bool)> GenerateCallback;

class AsEngine final {
 public:
  // -------------------------- V2.0 API -----------------------
  enum class RequestInferType {
    Generate,        // GPT style generative models.
    ModelInference,  // Infer the whole model, for the non-generative style
                     // model like bert.
  };

  enum class RequestMMType {
    TextInput,               // default is all text input.
    MultiMediaTypeRichText,  // behavior will be like RunRichTextGeneration.
  };

  class RequestContent {
   public:
    RequestInferType infer_type;
    RequestMMType mm_type;
    std::shared_ptr<DLTensorMap>
        inputs;  /// input tensors, format: {input_name_1: tensor,
                 /// input_name_2: tensor}
    GenerateConfig config;
  };

  /**
   * The GeneratedElements class represents generated elements and associated
   * probability information. It consists of four main data members:
   * 1. ids_from_generate - Stores generated IDs, which only keep new results
   * and exclude historical ones.
   * 2. log_probs_list - Stores a probability list for each token, along with
   * the top_logprobs tokens and their probabilities when generated.
   * 3. token_logprobs_list - Stores the probability value for each selected
   * token.
   * 4. tensors_from_model_inference - Contains tensors from model inference.
   */
  class GeneratedElements {
   public:
    /**
     * Stores generated IDs used by GPT-style generative models, excluding
     * historical results.
     */
    std::vector<int64_t> ids_from_generate;

    /**
     * A probability list for each token, including the top_logprobs tokens and
     * their probabilities when generated. Dimension: [num_token][top_logprobs],
     * where each token has a pair [token_id, prob].
     */
    std::vector<std::vector<std::pair<int, float>>> log_probs_list;

    /**
     * Stores the probability value for each selected token.
     */
    std::vector<float> token_logprobs_list;

    /**
     * Tensor outputs from model inference.
     */
    std::vector<std::unordered_map<std::string, std::shared_ptr<ITensor>>>
        tensors_from_model_inference;

    /**
     * Cached prefix token length.
     */
    int prefix_cache_len;  // total cache len (gpu + cpu)
    int prefix_len_gpu;    // gpu cache len
    int prefix_len_cpu;    // cpu cache len

    /**
      add new single element to the end.
    */
    void AppendNewSingleElementToEnd(GeneratedElements& rhs) {
      int size = rhs.ids_from_generate.size();
      for (int i = 0; i < size; i++) {
        ids_from_generate.push_back(rhs.ids_from_generate[i]);
        if (rhs.log_probs_list.size() > 0)
          log_probs_list.push_back(rhs.log_probs_list[i]);
        if (rhs.token_logprobs_list.size() > 0)
          token_logprobs_list.push_back(rhs.token_logprobs_list[i]);
        if (rhs.tensors_from_model_inference.size() > 0)
          tensors_from_model_inference.push_back(
              rhs.tensors_from_model_inference[i]);
      }
      prefix_cache_len = rhs.prefix_cache_len;
      prefix_len_gpu = rhs.prefix_len_gpu;
      prefix_len_cpu = rhs.prefix_len_cpu;
    }
  };

  enum class GenerateRequestStatus {
    Init,                 /// Init status.
    ContextFinished,      /// Context computation finished.
    Generating,           /// Start generating.
    GenerateFinished,     /// Generation finished, EOS token was generated.
    GenerateInterrupted,  /// The Generation was interrupted, often means
                          /// there is no enough memory, and this request
                          /// was unfortunedly stopped.
    InternalError,        /// Internal error status.

  };

  /**
   * @class ResultQueue
   *
   * The ResultQueue class is designed to generate status, and retrieve results
   * in a queue. It provides four main virtual methods:
   * 1. A method for generating status;
   * 2. A method to get the length of generated results;
   * 3. A method to fetch a result from the queue; and
   * 4. A method to fetch a result without waiting.
   */
  class ResultQueue {
   public:
    /**
     * A Key-Value statistic info structure, for float like info, convert into
     * long type.
     */
    typedef std::map<std::string, long> StatInfo;

    virtual ~ResultQueue() {}

    /**
     * Generates the status of a request.
     *
     * @return GenerateRequestStatus Returns the status of the generation
     * request.
     */

    virtual GenerateRequestStatus GenerateStatus() {
      throw std::runtime_error("Function not implemented.");
    }

    /**
     * Get Request statisitic info.
     *
     * @return key-value dict of all the statistic of this request.
     */
    virtual StatInfo RequestStatInfo() {
      throw std::runtime_error("Function not implemented.");
    }

    /**
     * Retrieves the length of generated results.
     *
     * @return size_t Returns the length of the generated results.
     */
    virtual size_t GeneratedLength() {
      throw std::runtime_error("Function not implemented.");
    }

    /**
     * Fetches a result from the queue, will be block until new token generated
     *
     * @return std::shared_ptr<GeneratedElements> Returns a smart pointer to the
     * generated elements.
     */
    virtual std::shared_ptr<GeneratedElements> Get() {
      throw std::runtime_error("Function not implemented.");
    }

    /**
     * Fetches a result from the queue, will be block until new token generated
     *
     * @prarm timeout_ms wait with timeout in milliseconds.
     *
     * @return std::shared_ptr<GeneratedElements> Returns a smart pointer to the
     * generated elements.
     */
    virtual std::shared_ptr<GeneratedElements> Get(int timeout_ms) {
      throw std::runtime_error("Function not implemented.");
    }

    /**
     * Retrieves a result from the queue without waiting.
     *
     * @return std::shared_ptr<GeneratedElements> Returns a smart pointer to the
     * generated elements.
     */
    virtual std::shared_ptr<GeneratedElements> GetNoWait() {
      throw std::runtime_error("Function not implemented.");
    }
  };

  /**
   * @typedef ResultQueue_t
   *
   * ResultQueue_t is an alias for a pointer to a ResultQueue object, providing
   * a clearer syntax for pointer usage.
   */
  typedef ResultQueue* ResultQueue_t;

  AsEngine();

  ~AsEngine();

  /**
   * Builds a model from the provided configuration structure.
   *
   * @param model_config A reference to the model configuration struct
   * containing all necessary settings for building the model.
   * @return Returns an AsStatus value indicating the success or failure of the
   * model building operation. If successful, returns AS_SUCCESS; otherwise,
   * returns an error code indicating the specific issue.
   */
  AsStatus BuildModelFromConfigStruct(AsModelConfig& model_config);

  /**
   * rebuild the model, the inverse API of #UnloadModelFromDeviceMemory
   *
   * @param model_name model_name
   *  */
  AsStatus ReloadModelToDeviceMemory(const char* model_name);

  /**
   * unload model from device memory (usually GPU) to CPU
   *
   * @param model_name  model name.
   */
  AsStatus UnloadModelFromDeviceMemory(const char* model_name);

  /**
   * Get loaded model's model type, input tensor , output tensor info
   *
   * @param model_name  model name key.
   * @param model_info  return model key.
   *
   * @return status code
   */
  AsStatus GetModelInformation(const char* model_name, std::string* model_info);

  /**
   * Get a model's info by passing model file paths
   * this api can fetch convert version information from user
   * */
  AsFileInfo GetFileInformation(const char* as_model_path,
                                const char* as_param_path);

  /**
   * Start running this model, model start to spinning.
   */
  AsStatus StartModel(const char* model_name);

  /**
   * Stop the model running
   */
  AsStatus StopModel(const char* model_name);

  /**
   * Release model's resource, the resource(include output and input) will
   * completely released.
   */
  AsStatus ReleaseModel(const char* model_name);

  /**
   * Start a new request in async
   *
   * request will start process since this function return,
   * the output token will filled in output queue, and it can
   * be controlled by request_handler which returned by this function.
   */
  AsStatus StartRequest(const char* model_name,
                        std::shared_ptr<RequestContent> request_info,
                        RequestHandle_t* request_handle, ResultQueue_t* queue);

  /**
   * Stop the request
   *
   * Stop generated this request, can be used in "cancel" case, and will release
   * the compute resource (not include output)
   */
  AsStatus StopRequest(const char* model_name, RequestHandle_t request_handle);

  /**
   * Release the request
   *
   * Release the request resource, the handler will become invalid. */
  AsStatus ReleaseRequest(const char* model_name,
                          RequestHandle_t request_handle);

  /** Sync the request
   *
   * Since start request is async api, this sync request will
   * wait until all generate finished, this api can be used to simulate sync
   * api.
   */
  AsStatus SyncRequest(const char* model_name, RequestHandle_t request_handle);

  /**
   * Retrieves the statistic information for specified model
   * @param model_name the model name already install into engine.
   * @return return the statistic information structure
   */
  AsEngineStat GetAsEngineStat(const char* model_name);

  AsStatus LoadLoraByName(const char* model_name, const char* lora_name);

  AsStatus UnloadLoraByName(const char* model_name, const char* lora_name);

  /**
   * Get SDK version string
   * version string will include version, git sha1, and
   * build time, eg:
   * 0.1.4/(GitSha1:beaca93)/(Build:20230403154806)
   */
  std::string GetVersionFull();

  /**
   * Get op profiling info
   * profiling string includes op name, min_time, max_time, count, sum,
   * percentage
   */
  std::string GetOpProfilingInfo(const char* model_name);

  /**
   * Get rank id (0~rank_num-1)
   * Since openmpi is used to manage CPU inferer task,
   * which may launch multiply process to do the inferer,
   * GetRankId is used to indicate the manager process
   * and get the output in manager process.
   * @note 0 is the manager process, we get output only
   * if GetRandId return 0; GetRankId always return 0 in
   * GPU inferer.
   */
  int GetRankId();

  /**
   * Get Rank nums
   */
  int GetRankNums();

  /**
   * check if allspark work as servie.
   * Normally in CPU mode, it uses mpi to make AllSpark run a larger model
   * fast, in this case, AllSpark serves as MPI daemon service.
   */
  bool IsAllSparkWorkAsService();

 private:
  std::unique_ptr<AsEngineImpl> as_engine_impl_;
};

// keep this builder in header.
class AsModelConfigBuilder {
 private:
  AsModelConfig config;

 public:
  AsModelConfigBuilder& withModelName(const std::string& name) {
    config.model_name = name;
    return *this;
  }

  AsModelConfigBuilder& withModelPath(const std::string& path) {
    config.model_path = path;
    return *this;
  }

  AsModelConfigBuilder& withWeightsPath(const std::string& path) {
    config.weights_path = path;
    return *this;
  }

  AsModelConfigBuilder& withComputeUnit(const std::string& unit) {
    config.compute_unit = unit;
    return *this;
  }

  AsModelConfigBuilder& withEngineMaxLength(int length) {
    config.engine_max_length = length;
    return *this;
  }

  AsModelConfigBuilder& withEngineMaxBatch(int batch) {
    config.engine_max_batch = batch;
    return *this;
  }

  AsModelConfigBuilder& withSwapThreshold(int64_t threshold) {
    config.swap_threshold = threshold;
    return *this;
  }

  AsModelConfigBuilder& withTextGraph(bool text_graph) {
    config.text_graph = text_graph;
    return *this;
  }

  AsModelConfigBuilder& withNumThreads(int threads) {
    config.num_threads = threads;
    return *this;
  }

  AsModelConfigBuilder& withMatmulPrecision(const std::string& precision) {
    config.matmul_precision = precision;
    return *this;
  }

  AsModelConfigBuilder& withLoraNames(const std::vector<std::string>& names) {
    config.lora_names = names;
    return *this;
  }

  AsModelConfigBuilder& withCacheSpanSize(int size) {
    config.cache_span_size = size;
    return *this;
  }

  AsModelConfigBuilder& withCacheSpanNumInit(int num_init) {
    config.cache_span_num_init = num_init;
    return *this;
  }

  AsModelConfigBuilder& withCacheSpanNumGrow(int num_grow) {
    config.cache_span_num_grow = num_grow;
    return *this;
  }

  AsModelConfigBuilder& withEnablePrefixCache(bool mode) {
    config.enable_prefix_cache = mode;
    return *this;
  }

  AsModelConfigBuilder& withPrefixCacheTTL(int ttl) {
    config.prefix_cache_ttl = ttl;
    return *this;
  }

  AsModelConfigBuilder& withPrefillMode(AsMHAPrefill mode) {
    config.prefill_mode = mode;
    return *this;
  }

  AsModelConfigBuilder& withCacheMode(AsCacheMode mode) {
    config.cache_mode = mode;
    return *this;
  }

  AsModelConfig build() { return config; }
};

inline std::string ToString(
    const allspark::AsEngine::GenerateRequestStatus& status) {
  using namespace allspark;
  switch (status) {
    case AsEngine::GenerateRequestStatus::Init:
      return "Init";
      break;
    case AsEngine::GenerateRequestStatus::ContextFinished:
      return "ContextFnished";
      break;
    case AsEngine::GenerateRequestStatus::Generating:
      return "Generating";
      break;
    case AsEngine::GenerateRequestStatus::GenerateFinished:
      return "GenerateFinished";
      break;
    case AsEngine::GenerateRequestStatus::GenerateInterrupted:
      return "GenerateInterrupted";
      break;
    default:
      return "Unknown";
  }
}

}  // namespace allspark

inline static std::ostream& operator<<(
    std::ostream& os, const allspark::AsEngine::GenerateRequestStatus& status) {
  using namespace allspark;
  switch (status) {
    case AsEngine::GenerateRequestStatus::Init:
      os << std::string("Init");
      break;
    case AsEngine::GenerateRequestStatus::ContextFinished:
      os << std::string("ContextFnished");
      break;
    case AsEngine::GenerateRequestStatus::Generating:
      os << std::string("Generating");
      break;
    case AsEngine::GenerateRequestStatus::GenerateFinished:
      os << std::string("GenerateFinished");
      break;
    case AsEngine::GenerateRequestStatus::GenerateInterrupted:
      os << std::string("GenerateInterrupted");
      break;
    case AsEngine::GenerateRequestStatus::InternalError:
      os << std::string("InternalError");
      break;
    default:
      os << std::string("UnknownStatus");
  }
  return os;
}
