/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    generate_op.h
 */

#pragma once
#include <core/operator/operator.h>
namespace allspark {
class GenerateOp : public AsOperator {
 public:
  explicit GenerateOp(const std::string& op_type = "")
      : AsOperator(op_type), batch_size_(1) {}
  ~GenerateOp();
  AsStatus Init(const OperatorProto& op_proto, const DeviceContext& ctx,
                const TensorMap& weights_map, TensorMap* tensor_map);
  AsStatus Reshape(RuntimeContext* runtime_ctx) override;
  AsStatus Forward(RuntimeContext* runtime_ctx) override;
  AsStatus RunContext(RuntimeContext* runtime_ctx);
  AsStatus RunDecoder(RuntimeContext* runtime_ctx);
  AsStatus RunOneBatch(std::shared_ptr<GenerateContext> gen_ctx,
                       int current_batch);
  AsStatus RunSample(RuntimeContext* runtime_ctx);

 private:
  BatchGencfg batch_gencfg_;
  int rank_id_ = 0;
  int nrank_ = 1;
  int generate_method_ = 0;  // 0:sample 1:beamsearch
  DataType dtype_ = DATATYPE_UNDEFINED;

  int batch_size_;
  int seq_len_ = 0;
  int vocab_size_ = 0;

  // basic params
  int beam_size_ = 1;
  float repetition_penalty_ = 1.0f;
  float length_penalty_ = 1.0f;
  int no_repeat_ngram_size_ = 0;
  int eos_token_id_ = -1;
  int min_length_ = 10;
  bool need_logprobs_ = false;
  /* deprecated basic params
  int k_=1;
  */
  int default_k_ = 1;
  // now we use batch params
  /*
  int* beam_size_arr_ = nullptr;
  float* repetition_penalty_arr_ = nullptr;
  float* length_penalty_arr_ = nullptr;
  int* no_repeat_ngram_size_arr_ = nullptr;
  int* eos_token_id_arr_ = nullptr;
  int* min_length_arr_ = nullptr;
  */
  int max_k_ = -1;
  std::unique_ptr<AsTensor> logprobs_;
  std::unique_ptr<AsTensor> token_logprobs_;
  std::unique_ptr<AsTensor> topk_list_;
  std::unique_ptr<AsTensor> topp_list_;
  std::unique_ptr<AsTensor> temperature_list_;
  std::unique_ptr<AsTensor> device_prop_;  // for cudaDeviceProp
  std::unique_ptr<AsTensor> sample_states_;
  std::shared_ptr<AsTensor> dec_ids_;
  std::shared_ptr<AsTensor> max_dec_ids_;
  int64_t* dec_ids_host_{nullptr};
  std::shared_ptr<AsTensor> gen_ids_ptr_;

  std::unique_ptr<AsTensor> repetition_penalty_list;
  std::unique_ptr<AsTensor> presence_penalty_list;
  std::unique_ptr<AsTensor> frequency_penalty_list;
  std::unique_ptr<AsTensor> no_repeat_ngram_size_list;
  std::unique_ptr<AsTensor> min_length_list;
  std::unique_ptr<AsTensor> eos_token_id_list;
  std::unique_ptr<AsTensor> cur_len_list;
  std::unique_ptr<AsTensor> input_len_list;
  std::unique_ptr<AsTensor> suppress_repetition_in_generation_list;
  void* topk_value_ptr_;
  void* topk_indice_ptr_;
  void* topp_value_ptr_;
  // std::unique_ptr<AsTensor> tmp_data_;
  // std::unique_ptr<AsTensor> tmp_max_dec_data_;
  // std::unique_ptr<AsTensor> beam_score_;
  // std::unique_ptr<AsTensor> sample_states_;   // [batch, "size of
  // sample_state"], reserved for later refactor of Batch-GenerateOp early stop
  // std::unique_ptr<AsTensor> hyps_beam_score_;  // [batch, beam_size]
  // std::unique_ptr<AsTensor> hyps_beam_idx_;  // [batch, beam_size]
  // std::unique_ptr<AsTensor> hyps_cur_id_;    // [batch]
  // std::unique_ptr<AsTensor> tmp_eos_count_;  // [batch]
  // std::unique_ptr<AsTensor> eos_count_;      // [batch]
  // std::unique_ptr<AsTensor> update_hyps_;    // [batch]
  // std::unique_ptr<AsTensor> seed_list_;
  /**
   * @brief CUDA core sample operator.
   *
   * @param[in] dtype Data type of input logits.
   * @param[out] out_tokens Output token.
   * @param[out] topk_value Final probabilities for sampling (after top-k and
   * top-p).
   * @param[out] topp_value Sorted top-k probabilities (temp variable for
   * top-p); note: this should be regarded as a part of the workspace.
   * @param[out] topk_indice Final indices for sampling (after top-k and top-p).
   * @param[in] in_logits Input logits.
   * @param[in] sample_state Random states.
   * @param[in] batch_size Batch size.
   * @param[in] max_k K value of top-k.
   * @param[in] length Input length.
   * @param[out] k_arr Lengths of each task after top-k and top-p.
   * @param[in] p_arr Cut-off probability values of top-p.
   * @param[in] ctx Device context pointer.
   * @param[in] runtime_ctx Runtime context pointer.
   * @param[in] ws_ptr Workspace pointer.
   * @param[in] ws_bytes Workspace size in bytes.
   * @param[in] device_prop: cudaDeviceProp info
   * @return AsStatus
   */
#ifdef ENABLE_JSON_MODE
  AsStatus FormatModelOutput(std::shared_ptr<GenerateContext> gen_ctx,
                             char* in_ptr, int current_batch, bool is_context);
#endif
  void build_batch_gencfg(RuntimeContext* runtime_ctx,
                          BatchGencfg& batch_gencfg, const DeviceContext* ctx);
  AsStatus (*kernel_launcher)(DataType dtype, int64_t* out_tokens,
                              void* topk_value, void* topp_value,
                              int* topk_indice, void* in_logits,
                              void** sample_states, int batch_size, int max_k,
                              int length, int* k_arr, float* p_arr,
                              float* temperature_arr, const DeviceContext* ctx,
                              RuntimeContext* runtime_ctx, void* ws_ptr,
                              size_t ws_bytes, void* device_prop) = nullptr;

  void (*beam_init_launcher)(DataType dtype, void* beam_score,
                             void* hyps_beam_score, int64_t* hyps_beam_idx,
                             int* eos_count, int batch_size, int beam_size,
                             const DeviceContext* ctx) = nullptr;

  void (*sample_init_launcher)(void* sample_state, unsigned long long seed,
                               int batch_size,
                               const DeviceContext* ctx) = nullptr;
  AsStatus (*logprobs_launcher)(DataType dtype, void* in_logits,
                                int64_t* out_tokens, void* token_logprobs,
                                void* logprobs, void* topk_value,
                                int* topk_indice, int batch_size, int length,
                                RuntimeContext* runtime_ctx, void* ws_ptr,
                                size_t ws_bytes,
                                const DeviceContext* ctx) = nullptr;

  AsStatus (*fill_max_dec_ids_launcher)(RuntimeContext* runtime_ctx,
                                        std::shared_ptr<AsTensor>& max_dec_ids,
                                        const DeviceContext* ctx) = nullptr;

  AsStatus (*process_logits_launcher)(DataType dtype, int64_t* in_tokens,
                                      void* in_logits, int batch_size,
                                      int vocab_size, const DeviceContext* ctx,
                                      RuntimeContext* runtime_ctx,
                                      BatchGencfg batch_gencfg, void* ws_ptr,
                                      size_t ws_bytes) = nullptr;

  AsStatus (*copy_matrix)(DataType dtype, void* in_ptr, void* new_ptr, int M,
                          int N, int lda, int ldb,
                          const DeviceContext* ctx) = nullptr;
};
}  // namespace allspark
