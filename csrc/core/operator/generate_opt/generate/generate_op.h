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
  AsStatus Init(const OperatorProto& op_proto, const DeviceContext& ctx,
                const TensorMap& weights_map, TensorMap* tensor_map);
  AsStatus Reshape(RuntimeContext* runtime_ctx) override;
  AsStatus Forward(RuntimeContext* runtime_ctx) override;
  AsStatus RunContext(RuntimeContext* runtime_ctx);
  AsStatus RunDecoder(RuntimeContext* runtime_ctx);
  AsStatus RunOneBatch(GenerateContext* gen_ctx, int current_batch);
  AsStatus RunSample(RuntimeContext* runtime_ctx);

 private:
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
  int default_k_ = 1;
  int max_k_ = -1;
  std::unique_ptr<AsTensor> logprobs_;
  std::unique_ptr<AsTensor> last_data_;
  std::unique_ptr<AsTensor> topk_value_;
  std::unique_ptr<AsTensor> topp_value_;
  std::unique_ptr<AsTensor> topk_indice_;
  std::unique_ptr<AsTensor> topk_list_;
  std::unique_ptr<AsTensor> topp_list_;
  std::unique_ptr<AsTensor> temperature_list_;
  std::unique_ptr<AsTensor> bad_words_ids_;  // assert max[1024, 1024]

  /**
   * @brief sample operator.
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
   * @return AsStatus
   */
  AsStatus (*kernel_launcher)(DataType dtype, int64_t* out_tokens,
                              void* topk_value, void* topp_value,
                              int64_t* topk_indice, void* in_logits,
                              void* sample_states, int batch_size, int max_k,
                              int length, int* k_arr, float* p_arr,
                              float* temperature_arr, const DeviceContext* ctx,
                              RuntimeContext* runtime_ctx, void* ws_ptr,
                              size_t ws_bytes) = nullptr;

  void (*beam_init_launcher)(DataType dtype, void* beam_score,
                             void* hyps_beam_score, int64_t* hyps_beam_idx,
                             int* eos_count, int batch_size, int beam_size,
                             const DeviceContext* ctx) = nullptr;

  void (*sample_init_launcher)(void* sample_state, unsigned long long seed,
                               int batch_size,
                               const DeviceContext* ctx) = nullptr;
  AsStatus (*logprobs_launcher)(DataType dtype, void* in_logits, void* logprobs,
                                void* topk_value, int64_t* topk_indice,
                                int batch_size, int length,
                                RuntimeContext* runtime_ctx, void* ws_ptr,
                                size_t ws_bytes,
                                const DeviceContext* ctx) = nullptr;
  AsStatus (*process_logits_launcher)(DataType dtype, int64_t* in_tokens,
                                      void* in_logits, int batch_size,
                                      int length, const DeviceContext* ctx,
                                      GenerateContext* gen_ctx,
                                      std::unique_ptr<AsTensor>& bad_words_ids,
                                      void* ws_ptr, size_t ws_bytes) = nullptr;

  AsStatus (*copy_matrix)(DataType dtype, void* in_ptr, void* new_ptr, int M,
                          int N, int lda, int ldb,
                          const DeviceContext* ctx) = nullptr;
};
}  // namespace allspark
