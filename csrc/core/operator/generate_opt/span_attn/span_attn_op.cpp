/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    span_attn_op.cpp
 */

#if ENABLE_SPAN_ATTENTION
#include "span_attn_op.h"

#include <cmath>

#include "core/kernel/kernel.h"
#include "utility/datatype_dispatcher.h"

namespace allspark {

namespace {
/// @brief Fallback CPU prefill kernel
/// @deprecated This CPU kernel needs refactor
void cpu_dec_single_mha(DataType dtype, void* out, void* score,
                        const void* query, const void* key, const void* value,
                        const float* mask, const void* position_embedding,
                        void* k_cache, void* v_cache, void** q_array,
                        void** k_array, void** v_array, void** score_array,
                        void** out_array, int batch_size, int beam_size,
                        int seq_len, int step, int cache_max_len,
                        int hidden_size, int num_heads, int size_per_head,
                        int gemm_batch, float alpha, bool xlogn_enable,
                        int xlogn_len, const DeviceContext* ctx) {
  DLOG(INFO) << "cpu_dec_single_mha" << std::endl;
  auto functor = [&]<typename T>() {
    cpu::UpdateKVLauncher((T*)k_cache, (T*)v_cache, (const T*)key,
                          (const T*)value, batch_size, step - 1, cache_max_len,
                          hidden_size, seq_len, 3 * hidden_size);
    if (seq_len != 1) {
      step = seq_len;
    }
    int q_stride = hidden_size * 3;
    int kv_stride = hidden_size;
    int out_stride = hidden_size;
    int score_stride = step * num_heads;
    cpu::GetBatchArrayLauncher((T*)query, (T*)k_cache, (T*)v_cache, (T*)score,
                               (T*)out, (T**)q_array, (T**)k_array,
                               (T**)v_array, (T**)score_array, (T**)out_array,
                               batch_size, 1, num_heads, size_per_head, step,
                               q_stride * seq_len, kv_stride * cache_max_len,
                               score_stride * seq_len, out_stride * seq_len);
    cpu::BatchGemmWraper<T>(score_array, q_array, k_array, seq_len, step,
                            size_per_head, false, true, alpha, 0.0f, q_stride,
                            kv_stride, score_stride, gemm_batch);
    if (position_embedding) {
      cpu::SimpleAdd((T*)score, (T*)score, (T*)position_embedding,
                     batch_size * num_heads * step * seq_len);
    }
    cpu::BatchSoftmax<T>((T*)score, mask, batch_size, beam_size, num_heads,
                         seq_len, step);
    cpu::BatchGemmWraper<T>(out_array, score_array, v_array, seq_len,
                            size_per_head, step, false, false, 1.0f, 0.0f,
                            score_stride, kv_stride, out_stride, gemm_batch);
  };
  DispatchCPU(dtype, functor);
}

}  // anonymous namespace

AsStatus SpanAttnOp::setAttributes(const OperatorProto& op_proto) {
  auto& attr_map = op_proto.attr();

  if (attr_map.find("multigpu") != attr_map.end()) {
    multi_nodes_ = *(bool*)(attr_map.at("multigpu").c_str());
  } else {
    // multi node by default
    multi_nodes_ = true;
  }

  if (attr_map.find("alpha") != attr_map.end()) {
    alpha_ = *(float*)(attr_map.at("alpha").c_str());
  }

  /// NOTE: causal mask is always true
  causal_mask_ = true;

  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus SpanAttnOp::setWorkspace(const RuntimeContext* runtime_ctx) {
  return AsStatus::ALLSPARK_SUCCESS;
}

/* ---------- private helpers ---------- */
AsStatus SpanAttnOp::decoderAppendCache(const RuntimeContext* runtime_ctx) {
  const AsTensor* in_tensor = tensor_map_->at(in_names_[0]).get();

  const int max_seq_len = ctx_->GetModelMaxLength();
  const int span_len = ctx_->GetCacheSpanSize();
  const int max_num_spans = (max_seq_len + span_len - 1) / span_len;

  std::vector<int> old_seq_lens(batch_size_);
  for (int batch = 0; batch < batch_size_; batch++) {
    std::shared_ptr<GenerateContext> gen_ctx = runtime_ctx->GetGenCtx(batch);
    old_seq_lens[batch] = gen_ctx->step;
#ifdef ENABLE_SPAN_DEBUG
    DLOG(INFO) << "SpanAttnOp::decoderAppendCache: [" << batch
               << "] old seq len=" << old_seq_lens[batch];
#endif

    const AsTensor& k_cache_ptrs =
        gen_ctx->virtual_k_cache->GetCache(layer_num_, 0);
    const AsTensor& v_cache_ptrs =
        gen_ctx->virtual_v_cache->GetCache(layer_num_, 0);
    TensorUtils::DeepCopyVectorPart(*k_span_array_tensor_host_,
                                    batch * max_num_spans, k_cache_ptrs, 0,
                                    k_cache_ptrs.GetShape()[0]);
    TensorUtils::DeepCopyVectorPart(*v_span_array_tensor_host_,
                                    batch * max_num_spans, v_cache_ptrs, 0,
                                    v_cache_ptrs.GetShape()[0]);
  }

  TensorUtils::DeepCopyFromStdVector(*decoder_seq_len_tensor_host_, 0,
                                     old_seq_lens);

  decoderAppendCacheLauncher();

  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus SpanAttnOp::runContext(RuntimeContext* runtime_ctx) {
  if (batch_size_ != 1) {
    LOG(ERROR) << "SpanAttnOp only support multibatch in decoder pharse, "
                  "not context pharse.";
    return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }
  std::shared_ptr<GenerateContext> gen_ctx = runtime_ctx->GetContextGenCtx();
#ifdef ENABLE_SPAN_DEBUG
  DLOG(INFO) << "SpanAttnOp::runContext [" << gen_ctx->request->request_id
             << "][layer " << layer_num_ << "]";
#endif

  void* k_cache_buf = tensor_map_->at("context_k_workspace")->GetDataPtr();
  void* v_cache_buf = tensor_map_->at("context_v_workspace")->GetDataPtr();

  // prepare cache span ptrs
  const AsTensor& k_cache_ptrs =
      gen_ctx->virtual_k_cache->GetCache(layer_num_, 0);
  const AsTensor& v_cache_ptrs =
      gen_ctx->virtual_v_cache->GetCache(layer_num_, 0);

  // valid kv cache size: num_tokens * (hidden_size / head_q) * head_k
  copyPrefixSpanToCtxMemLauncher(k_cache_ptrs, v_cache_ptrs, k_cache_buf,
                                 v_cache_buf);

  contextAttnLauncher(k_cache_buf, v_cache_buf, gen_ctx->num_beams);
#ifdef ENABLE_SPAN_DEBUG
  // slice & copy cache to spans
  DLOG(INFO) << "runContext [layer " << layer_num_ << "]: copy cache";
#endif

  contextCopySpanLauncher(k_cache_ptrs, v_cache_ptrs, k_cache_buf, v_cache_buf);
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus SpanAttnOp::runDecoder(RuntimeContext* runtime_ctx) {
#ifdef ENABLE_SPAN_DEBUG
  DLOG(INFO) << "SpanAttnOp::runDecoder: batch size=" << batch_size_;
#endif
  decoderAppendCache(runtime_ctx);
  decoderAttnLauncher(runtime_ctx);
  return AsStatus::ALLSPARK_SUCCESS;
}

/* ---------- interfaces ---------- */
AsStatus SpanAttnOp::Init(const OperatorProto& op_proto,
                          const DeviceContext& ctx,
                          const TensorMap& weights_map, TensorMap* tensor_map) {
  AS_CHECK_STATUS(AsOperator::Init(op_proto, ctx, weights_map, tensor_map));

  // sanity check
  if (ctx_->GetCacheSpanSize() <= 0) {
    LOG(ERROR) << "SpanAttnOp: span size must be a positive integer";
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }

  layer_num_ = get_layer_num(this->op_name_);
  if (layer_num_ < 0) {
    LOG(ERROR) << "SpanAttnOp: cannot get layer_num from op name";
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }

  // type inference
  dtype_ = tensor_map_->at(in_names_[0])->GetDataType();
  tensor_map_->at(out_names_[0])->SetDataType(dtype_);

  // load model config, order MATTERS!
  AS_CHECK_STATUS(setAttributes(op_proto));

  int num_heads = ctx_->GetNumberHeads();
  int num_groups = ctx_->GetNumberGroups();
  int size_per_head = ctx_->GetSizePerHead();
  if (num_heads == num_groups || num_groups == 0) {
    attn_head_ = std::make_unique<HeadMHA>(num_heads, num_groups, size_per_head,
                                           multi_nodes_, ctx_->GetNranks());
  } else {
    attn_head_ = std::make_unique<HeadGQA>(num_heads, num_groups, size_per_head,
                                           multi_nodes_, ctx_->GetNranks());
  }

  if (alpha_ < 0) {
    alpha_ = 1.0f / std::sqrt(attn_head_->SizePerHead() * 1.0f);
  }

  decoder_q_tensor_ = std::make_unique<AsTensor>(
      "decoder_q_tensor", ctx.GetDeviceType(), dtype_, DataMode::DENSE,
      Shape{1, 1, attn_head_->NumHeads(), attn_head_->SizePerHead()});

  decoder_seq_len_tensor_host_ =
      std::make_unique<AsTensor>("decoder_seq_len_tensor_host", DeviceType::CPU,
                                 DataType::INT32, DataMode::DENSE, Shape{1});

  // cache ptr arrays
  k_span_array_tensor_host_ = std::make_unique<AsTensor>(
      "k_span_array_tensor_host", DeviceType::CPU, DataType::POINTER,
      DataMode::DENSE, Shape{1 * 1});
  v_span_array_tensor_host_ = std::make_unique<AsTensor>(
      "k_span_array_tensor_host", DeviceType::CPU, DataType::POINTER,
      DataMode::DENSE, Shape{1 * 1});

  AS_CHECK_STATUS(deviceInit());

  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus SpanAttnOp::Forward(RuntimeContext* runtime_ctx) {
  AsStatus status;
  if (runtime_ctx->is_context) {
    status = runContext(runtime_ctx);
  } else {
    status = runDecoder(runtime_ctx);
  }
  return status;
}

AsStatus SpanAttnOp::Reshape(RuntimeContext* runtime_ctx) {
  const int max_batch = ctx_->GetModelMaxBatch();
  const int max_seq_len = ctx_->GetModelMaxLength();
  const int span_len = ctx_->GetCacheSpanSize();
  const int max_num_spans = (max_seq_len + span_len - 1) / span_len;

  const AsTensor* x = tensor_map_->at(in_names_[0]).get();
  const Shape& x_shape = x->GetShape();
  batch_size_ = x_shape[0];
  seq_len_ = x_shape[1];
  int qkv_stride = x_shape[2];

  AS_CHECK_STATUS(attn_head_->UpdateShape(qkv_stride));

#ifdef ENABLE_SPAN_DEBUG
  DLOG(INFO) << "SpanAttnOp::Reshape: batch_size = " << batch_size_
             << ", seq_len = " << seq_len_
             << ", qkv_stride = " << attn_head_->QKVStride()
             << ", hidden_size = " << attn_head_->HiddenSize();
#endif

  // shape sanity check
  if (attn_head_->HiddenSize() % attn_head_->NumHeads() != 0) {
    LOG(ERROR) << "SpanAttnOp::Reshape: invalid shape, hidden_size: "
               << attn_head_->HiddenSize()
               << ", num_heads: " << attn_head_->NumHeads();
    return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }

  if (attn_head_->SizePerHead() !=
      attn_head_->HiddenSize() / attn_head_->NumHeads()) {
    LOG(ERROR) << "SpanAttnOp::Reshape: invalid shape, hidden_size: "
               << attn_head_->HiddenSize()
               << ", num_heads: " << attn_head_->NumHeads()
               << ", size_per_head: " << attn_head_->SizePerHead();
    return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }

  if (attn_head_->QKVStride() !=
      attn_head_->HiddenSize() + attn_head_->KVStride() * 2) {
    LOG(ERROR) << "SpanAttnOp::Reshape: invalid shape, hidden_size: "
               << attn_head_->HiddenSize()
               << ", kv_stride: " << attn_head_->KVStride()
               << ", qkv_stride: " << attn_head_->QKVStride();
    return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }

  // set tensor shapes
  Shape y_shape(x_shape);
  y_shape[2] = attn_head_->HiddenSize();
  tensor_map_->at(out_names_[0])->SetShape(std::move(y_shape));

  if (runtime_ctx->is_context) {
    k_span_array_tensor_host_->SetShape(Shape({1 * max_num_spans}));
    v_span_array_tensor_host_->SetShape(Shape({1 * max_num_spans}));
  } else {
    decoder_q_tensor_->SetShape(
        Shape{max_batch, 1, attn_head_->NumHeads(), attn_head_->SizePerHead()});

    decoder_seq_len_tensor_host_->SetShape(Shape{max_batch});

    k_span_array_tensor_host_->SetShape(Shape({max_batch * max_num_spans}));
    v_span_array_tensor_host_->SetShape(Shape({max_batch * max_num_spans}));
  }

  AS_CHECK_STATUS(deviceReshape(runtime_ctx));

  // workspace
  AS_CHECK_STATUS(setWorkspace(runtime_ctx));

  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus SpanAttnOp::Alloc(RuntimeContext* runtime_ctx) {
  const int max_seq_len = ctx_->GetModelMaxLength();
  const int span_len = ctx_->GetCacheSpanSize();
  const int max_num_spans = (max_seq_len + span_len - 1) / span_len;

  // noncontiguous cache
  if (runtime_ctx->is_context) {
    std::shared_ptr<GenerateContext> gen_ctx = runtime_ctx->GetContextGenCtx();

    const int cache_increment = seq_len_;
    const int old_seq_len = gen_ctx->step;

    bool san_check =
        (old_seq_len == gen_ctx->virtual_k_cache->GetSeqLength(layer_num_)) &&
        (old_seq_len == gen_ctx->virtual_v_cache->GetSeqLength(layer_num_));
    if (!san_check) {
      LOG(ERROR) << "SpanAttnOp::Reshape: [" << gen_ctx->request->request_id
                 << "][layer " << layer_num_ << "] gen_ctx step ("
                 << old_seq_len << ") and cached seq len ("
                 << gen_ctx->virtual_k_cache->GetSeqLength(layer_num_) << ", "
                 << gen_ctx->virtual_v_cache->GetSeqLength(layer_num_)
                 << ") mismatch";
      return AsStatus::ALLSPARK_RUNTIME_ERROR;
    }

    (void)gen_ctx->virtual_k_cache->GetCache(layer_num_, cache_increment);
    (void)gen_ctx->virtual_v_cache->GetCache(layer_num_, cache_increment);
  } else {
    //* NOTE: do NOT run this concurrently, it decreases performance
    // for each request in batch, claim its cache
    for (int batch = 0; batch < batch_size_; batch++) {
      std::shared_ptr<GenerateContext> gen_ctx = runtime_ctx->GetGenCtx(batch);
      const int cache_increment = seq_len_;
      const int old_seq_len = gen_ctx->step;

      bool san_check =
          (old_seq_len == gen_ctx->virtual_k_cache->GetSeqLength(layer_num_)) &&
          (old_seq_len == gen_ctx->virtual_v_cache->GetSeqLength(layer_num_));
      if (!san_check) {
        LOG(ERROR) << "SpanAttnOp::Reshape: [" << gen_ctx->request->request_id
                   << "][layer " << layer_num_ << "] gen_ctx step ("
                   << old_seq_len << ") and cached seq len ("
                   << gen_ctx->virtual_k_cache->GetSeqLength(layer_num_) << ", "
                   << gen_ctx->virtual_v_cache->GetSeqLength(layer_num_)
                   << ") mismatch";
        return AsStatus::ALLSPARK_RUNTIME_ERROR;
      }

      (void)gen_ctx->virtual_k_cache->GetCache(layer_num_, cache_increment);
      (void)gen_ctx->virtual_v_cache->GetCache(layer_num_, cache_increment);
    }
  }
  return AsStatus::ALLSPARK_SUCCESS;
}

}  // namespace allspark
#endif
