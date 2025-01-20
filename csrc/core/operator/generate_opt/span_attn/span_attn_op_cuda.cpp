/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    span_attn_op_cuda.cpp
 */

#ifdef ENABLE_CUDA
#if ENABLE_SPAN_ATTENTION

#include "span_attn_op_cuda.h"

#include <cuda_runtime.h>

#include "cuda/cuda_context.h"
#include "utility/check_cuda.h"
#include "utility/datatype_dispatcher.h"

#define CHECK_SPAN_ATTN(expr)                                                  \
  do {                                                                         \
    auto __sa_status = (expr);                                                 \
    if (__sa_status != span::SaStatus::SUCCESS) {                              \
      LOG(ERROR) << "SpanAttention " << span::GetErrorName(__sa_status)        \
                 << " (" << int(__sa_status)                                   \
                 << ") at SpanAttnOpCUDA::" << __FUNCTION__ << ":" << __LINE__ \
                 << ": " << span::GetErrorString(__sa_status) << " in " #expr; \
      AS_THROW(AsStatus::ALLSPARK_RUNTIME_ERROR);                              \
    }                                                                          \
  } while (0)

namespace allspark {
namespace {

cudaDataType_t to_cuda_type(DataType dtype_) {
  switch (dtype_) {
    case DataType::FLOAT16:
      return cudaDataType_t::CUDA_R_16F;
    case DataType::BFLOAT16:
      return cudaDataType_t::CUDA_R_16BF;
    default:
      return cudaDataType_t::CUDA_R_32F;
  }
}

template <typename T>
span::DataType to_span_data_type() {
  return span::DataType::FP32;
}

#ifdef ENABLE_FP16
template <>
span::DataType to_span_data_type<half>() {
  return span::DataType::FP16;
}
#endif

#ifdef ENABLE_BF16
template <>
span::DataType to_span_data_type<hie::bfloat16>() {
  return span::DataType::BF16;
}
#endif

}  // anonymous namespace

void SpanAttnOpCUDA::contextAttnLauncher(void* k_cache_buf, void* v_cache_buf,
                                         int beam_size) {
  constexpr int current_batch = 0;
  AsTensor* in_tensor = tensor_map_->at(in_names_[0]).get();
  AsTensor* out_tensor = tensor_map_->at(out_names_[0]).get();
  AsTensor* wss_tensor = tensor_map_->at("workspace").get();

  std::pair<bool, AsMHAPrefill> prefill_mode_pair = GetPrefillMode();
  if (!prefill_mode_pair.first) {
    LOG(ERROR) << "SpanAttnOpCUDA get prefill mode error. " << std::endl;
    AS_THROW(AsStatus::ALLSPARK_RUNTIME_ERROR);
    return;
  }

  // flashv2 logic
  if (prefill_mode_pair.second == AsMHAPrefill::AsPrefillFlashV2) {
#ifdef FLASH_ATTN_V2
    char* cptr = (char*)in_tensor->GetDataPtr();
    char* qptr = cptr;
    char* kptr = qptr + attn_head_->HiddenSize() * SizeofType(dtype_);
    char* vptr = kptr + attn_head_->KVStride() * SizeofType(dtype_);
    char* optr = (char*)out_tensor->GetDataPtr();
    char* wptr = (char*)wss_tensor->GetDataPtr();

    if (gen_ctx_->prefix_len == 0) {
      cuda::flashv2_set_runtime_param(flash_v2_params_, qptr, kptr, vptr, optr,
                                      wptr, alpha_);
      cuda::flashv2_dispatch(
          flash_v2_params_, static_cast<const CUDAContext*>(ctx_)->GetStream());

      void* kbuf = (char*)cptr + attn_head_->HiddenSize() * SizeofType(dtype_);
      void* vbuf = (char*)kbuf + attn_head_->KVStride() * SizeofType(dtype_);
      DispatchCUDA(dtype_, [&]<typename T>() {
        cuda::UpdateKVLauncher(
            (T*)k_cache_buf, (T*)v_cache_buf, (const T*)kbuf, (const T*)vbuf, 1,
            0, seq_len_, attn_head_->KVStride(), seq_len_,
            attn_head_->QKVStride(),
            static_cast<const CUDAContext*>(ctx_)->GetStream());
      });
    } else {
      DispatchCUDA(dtype_, [&]<typename T>() {
        cuda::UpdateKVLauncher(
            (T*)k_cache_buf, (T*)v_cache_buf, (const T*)kptr, (const T*)vptr, 1,
            gen_ctx_->prefix_len, seq_len_, attn_head_->KVStride(), seq_len_,
            attn_head_->QKVStride(),
            static_cast<const CUDAContext*>(ctx_)->GetStream());
      });

      cuda::flashv2_set_runtime_param(flash_v2_params_, qptr, k_cache_buf,
                                      v_cache_buf, optr, wptr, alpha_);
      cuda::flashv2_dispatch(
          flash_v2_params_, static_cast<const CUDAContext*>(ctx_)->GetStream());
    }
#else
    LOG(ERROR) << "Flash-Attention is not compiled";
    AS_THROW(AsStatus::ALLSPARK_RUNTIME_ERROR);
#endif  // FLASH_ATTN_V2
  } else if (prefill_mode_pair.second == AsMHAPrefill::AsPrefillXformer) {
    char* cptr = (char*)in_tensor->GetDataPtr();
    char* qptr = cptr;
    char* kptr = qptr + attn_head_->HiddenSize() * SizeofType(dtype_);
    char* vptr = kptr + attn_head_->KVStride() * SizeofType(dtype_);
    char* optr = (char*)out_tensor->GetDataPtr();
    char* wptr = (char*)wss_tensor->GetDataPtr();
    if (gen_ctx_->prefix_len == 0) {
      allspark::cuda::xformer_prefill_attention(
          xformer_params_, qptr, kptr, vptr, optr, wptr, alpha_,
          static_cast<const CUDAContext*>(ctx_)->GetStream());

      DispatchCUDA(dtype_, [&]<typename T>() {
        cuda::UpdateKVLauncher(
            (T*)k_cache_buf, (T*)v_cache_buf, (const T*)kptr, (const T*)vptr, 1,
            0, seq_len_, attn_head_->KVStride(), seq_len_,
            attn_head_->QKVStride(),
            static_cast<const CUDAContext*>(ctx_)->GetStream());
      });
    } else {
      DispatchCUDA(dtype_, [&]<typename T>() {
        cuda::UpdateKVLauncher(
            (T*)k_cache_buf, (T*)v_cache_buf, (const T*)kptr, (const T*)vptr, 1,
            gen_ctx_->prefix_len, seq_len_, attn_head_->KVStride(), seq_len_,
            attn_head_->QKVStride(),
            static_cast<const CUDAContext*>(ctx_)->GetStream());
      });

      allspark::cuda::xformer_prefill_attention(
          xformer_params_, qptr, k_cache_buf, v_cache_buf, optr, wptr, alpha_,
          static_cast<const CUDAContext*>(ctx_)->GetStream());
    }
  } else {
    // fallback MHA kernel
    LOG(ERROR) << "SpanAttnOpCUDA::contextAttnLauncher: "
                  "trivial_prefill_attention is "
                  "deprecated, consider using newer GPUs";
    AS_THROW(AsStatus::ALLSPARK_RUNTIME_ERROR);
#if 0
    if (gen_ctx_->prefix_len != 0) {
      LOG(ERROR) << "SpanAttnOpCUDA::contextAttnLauncher: does not support "
                    "prefix cache";
      AS_THROW(AsStatus::ALLSPARK_RUNTIME_ERROR);
    }
    void* concat_ptr = (char*)in_tensor->GetDataPtr();
    void* output_ptr = (char*)out_tensor->GetDataPtr();
    float* mask_ptr =
        tensor_map_->at(in_names_[1])->GetShape().Count() == 0
            ? nullptr
            : (float*)(tensor_map_->at(in_names_[1])->GetDataPtr());
    void* workspace_ptr = wss_tensor->GetDataPtr();
    // position-embedding no longer used in future code
    // void* position_embedding =
    //     pos_embedding ? tensor_map_->at(in_names_[2])->GetDataPtr()
    //                   : nullptr;

    // update kv-cache
    void* kbuf =
        (char*)concat_ptr + attn_head_->HiddenSize() * SizeofType(dtype_);
    void* vbuf =
        (char*)kbuf +
        attn_head_->KVStride() *
            SizeofType(dtype_);  // see span_mha_op.h, size_per_head * num_groups
    DispatchCUDA(dtype_, [&]<typename T>() {
      cuda::UpdateKVLauncher(
          (T*)k_cache_buf, (T*)v_cache_buf, (const T*)kbuf, (const T*)vbuf, 1,
          0, seq_len_, attn_head_->KVStride(), seq_len_,
          attn_head_->QKVStride(),
          static_cast<const CUDAContext*>(ctx_)->GetStream());
    });

    // attention
    allspark::cuda::trivial_prefill_attention(
        trivial_params_,
        static_cast<const CUDAContext*>(ctx_)->GetCublasHandle(),
        static_cast<const CUDAContext*>(ctx_)->GetStream(), concat_ptr,
        mask_ptr, output_ptr, k_cache_buf, v_cache_buf, workspace_ptr,
        beam_size, alpha_);
#endif
  }
  return;
}

void SpanAttnOpCUDA::copyPrefixSpanToCtxMemLauncher(
    const AsTensor& k_cache_ptr_tensor, const AsTensor& v_cache_ptr_tensor,
    const void* k_contiguous_cache, const void* v_contiguous_cache) {
  if (gen_ctx_->prefix_len == 0) return;

  // clear previous errors
  auto err = cudaGetLastError();
  if (err != cudaSuccess) {
    LOG(WARNING) << "gpu_copy_span_to_continuous_mem: previous error cleared: "
                 << cudaGetErrorString(err);
  }

  // prepare device tensor
  TensorUtils::DeepCopyVectorPartAsync(*k_span_array_tensor_device_, 0,
                                       k_cache_ptr_tensor, 0,
                                       k_cache_ptr_tensor.GetShape()[0], ctx_);
  TensorUtils::DeepCopyVectorPartAsync(*v_span_array_tensor_device_, 0,
                                       v_cache_ptr_tensor, 0,
                                       v_cache_ptr_tensor.GetShape()[0], ctx_);
  const CUDAContext* gpu_ctx = dynamic_cast<const CUDAContext*>(ctx_);
  auto qMode = CacheUtils::toQuantMode(ctx_->GetCacheMode());
  cudaStream_t cu_stream = gpu_ctx->GetStream();

  DispatchCUDA(dtype_, [&]<typename T>() {
    cuda::PrefixCacheCopyLauncher<T>(
        static_cast<const void**>(k_span_array_tensor_device_->GetDataPtr()),
        reinterpret_cast<T*>(const_cast<void*>(k_contiguous_cache)),
        attn_head_->NumGroups(), attn_head_->SizePerHead(),
        ctx_->GetCacheSpanSize(), gen_ctx_->prefix_len, qMode, cu_stream);
    cuda::PrefixCacheCopyLauncher<T>(
        static_cast<const void**>(v_span_array_tensor_device_->GetDataPtr()),
        reinterpret_cast<T*>(const_cast<void*>(v_contiguous_cache)),
        attn_head_->NumGroups(), attn_head_->SizePerHead(),
        ctx_->GetCacheSpanSize(), gen_ctx_->prefix_len, qMode, cu_stream);
  });
}

void SpanAttnOpCUDA::contextCopySpanLauncher(const AsTensor& k_cache_ptr_tensor,
                                             const AsTensor& v_cache_ptr_tensor,
                                             const void* k_contiguous_cache,
                                             const void* v_contiguous_cache) {
  // clear previous errors
  auto err = cudaGetLastError();
  if (err != cudaSuccess) {
    LOG(WARNING) << "gpu_context_copy_span: previous error cleared: "
                 << cudaGetErrorString(err);
  }

  // prepare device tensor
  TensorUtils::DeepCopyVectorPartAsync(*k_span_array_tensor_device_, 0,
                                       k_cache_ptr_tensor, 0,
                                       k_cache_ptr_tensor.GetShape()[0], ctx_);
  TensorUtils::DeepCopyVectorPartAsync(*v_span_array_tensor_device_, 0,
                                       v_cache_ptr_tensor, 0,
                                       v_cache_ptr_tensor.GetShape()[0], ctx_);

  const CUDAContext* gpu_ctx = dynamic_cast<const CUDAContext*>(ctx_);
  auto qMode = CacheUtils::toQuantMode(ctx_->GetCacheMode());
  cudaStream_t cu_stream = gpu_ctx->GetStream();

  int64_t span_offset = gen_ctx_->prefix_len / ctx_->GetCacheSpanSize();
  int64_t cont_offset = gen_ctx_->prefix_len * attn_head_->KVStride();

  DispatchCUDA(dtype_, [&]<typename T>() {
    cuda::ContextSpanCopyLauncher<T>(
        static_cast<void**>(k_span_array_tensor_device_->GetDataPtr()) +
            span_offset,
        static_cast<const T*>(k_contiguous_cache) + cont_offset,
        attn_head_->NumGroups(), attn_head_->SizePerHead(),
        ctx_->GetCacheSpanSize(), seq_len_, qMode, cu_stream);
    cuda::ContextSpanCopyLauncher<T>(
        static_cast<void**>(v_span_array_tensor_device_->GetDataPtr()) +
            span_offset,
        static_cast<const T*>(v_contiguous_cache) + cont_offset,
        attn_head_->NumGroups(), attn_head_->SizePerHead(),
        ctx_->GetCacheSpanSize(), seq_len_, qMode, cu_stream);
  });
  return;
}

void SpanAttnOpCUDA::decoderAppendCacheLauncher() {
  // clear previous errors
  auto err = cudaGetLastError();
  if (err != cudaSuccess) {
    LOG(WARNING) << "gpu_decoder_append_cache: previous error cleared: "
                 << cudaGetErrorString(err);
  }

#ifdef ENABLE_SPAN_DEBUG
  DLOG(INFO)
      << "gpu_decoder_append_cache: call cuda::DecoderCacheAppendLauncher";
#endif

  const int max_seq_len = ctx_->GetModelMaxLength();
  const int span_len = ctx_->GetCacheSpanSize();
  const int max_num_spans = (max_seq_len + span_len - 1) / span_len;

  const AsTensor* in_tensor = tensor_map_->at(in_names_[0]).get();

  // prepare device tensors
  TensorUtils::DeepCopyWholeAsync(*k_span_array_tensor_device_,
                                  *k_span_array_tensor_host_, ctx_);
  TensorUtils::DeepCopyWholeAsync(*v_span_array_tensor_device_,
                                  *v_span_array_tensor_host_, ctx_);
  TensorUtils::DeepCopyWholeAsync(*decoder_seq_len_tensor_device_,
                                  *decoder_seq_len_tensor_host_, ctx_);

  const CUDAContext* gpu_ctx = dynamic_cast<const CUDAContext*>(ctx_);
  auto qMode = CacheUtils::toQuantMode(ctx_->GetCacheMode());
  cudaStream_t cu_stream = gpu_ctx->GetStream();
  DispatchCUDA(dtype_, [&]<typename T>() {
    cuda::DecoderCacheAppendLauncher<T>(
        static_cast<void**>(k_span_array_tensor_device_->GetDataPtr()),
        static_cast<void**>(v_span_array_tensor_device_->GetDataPtr()),
        static_cast<T*>(decoder_q_tensor_->GetDataPtr()),
        static_cast<const T*>(in_tensor->GetDataPtr()),
        static_cast<const uint32_t*>(
            decoder_seq_len_tensor_device_->GetDataPtr()),
        batch_size_, attn_head_->NumHeads(), attn_head_->NumGroups(),
        attn_head_->SizePerHead(), ctx_->GetCacheSpanSize(), max_num_spans,
        qMode, cu_stream);
  });
  return;
}

void SpanAttnOpCUDA::decoderAttnLauncher(const RuntimeContext* runtime_ctx) {
  // clear previous errors
  auto err = cudaGetLastError();
  if (err != cudaSuccess) {
    LOG(WARNING)
        << "SpanAttnOpCUDA::decoderAttnLauncher: previous error cleared: "
        << cudaGetErrorString(err);
  }

  AsTensor* out_tensor = tensor_map_->at(out_names_[0]).get();

  const CUDAContext* gpu_ctx = dynamic_cast<const CUDAContext*>(ctx_);
  cudaStream_t cu_stream = gpu_ctx->GetStream();
  int device_id = gpu_ctx->GetDeviceId();

  const AsCacheMode mode = ctx_->GetCacheMode();
  const int max_seq_len = ctx_->GetModelMaxLength();
  const int span_len = ctx_->GetCacheSpanSize();
  const int max_num_spans = (max_seq_len + span_len - 1) / span_len;

  // sanity check
  if (seq_len_ != 1) {
    LOG(ERROR) << "SpanAttnOpCUDA::decoderAttnLauncher: only support "
                  "seq_len_ == 1 for now";
    AS_THROW(AsStatus::ALLSPARK_RUNTIME_ERROR);
  }

  std::vector<int> new_seq_lens(batch_size_);
  for (int batch = 0; batch < batch_size_; ++batch) {
    std::shared_ptr<GenerateContext> gen_ctx = runtime_ctx->GetGenCtx(batch);
    new_seq_lens[batch] = gen_ctx->step + seq_len_;
  }

  DispatchCUDA(dtype_, [&]<typename T>() {
    span::DataType dtype = to_span_data_type<T>();
    span::SpanAttnHandle_t handle{nullptr};
    CHECK_SPAN_ATTN(span::CreateHandle(
        &handle, dtype, CacheUtils::toQuantMode(mode), batch_size_,
        attn_head_->NumHeads(), attn_head_->NumGroups(),
        attn_head_->SizePerHead(), span_len, max_num_spans, new_seq_lens.data(),
        dprop_));
    if (handle == nullptr) {
      LOG(ERROR)
          << "SpanAttnOpCUDA::decoderAttnLauncher: span::CreateHandle failed";
      AS_THROW(AsStatus::ALLSPARK_RUNTIME_ERROR);
    }
    CHECK_SPAN_ATTN(
        span::Run(static_cast<T*>(out_tensor->GetDataPtr()),
                  static_cast<const T*>(decoder_q_tensor_->GetDataPtr()),
                  static_cast<const void* const*>(
                      k_span_array_tensor_device_->GetDataPtr()),
                  static_cast<const void* const*>(
                      v_span_array_tensor_device_->GetDataPtr()),
                  tensor_map_->at("workspace")->GetDataPtr(),
                  tensor_map_->at("workspace")->GetSizeInByte(),
                  host_workspace_->GetDataPtr(),
                  host_workspace_->GetSizeInByte(), alpha_, handle, cu_stream));
    CHECK_SPAN_ATTN(span::DestroyHandle(handle));
  });
  return;
}

AsStatus SpanAttnOpCUDA::setDecoderWorkspaceSize() {
  const int max_batch = ctx_->GetModelMaxBatch();
  // reserve 1 token for newly generated one
  const int max_seq_len = ctx_->GetModelMaxLength() + 1;
  const int span_len = ctx_->GetCacheSpanSize();
  const int max_num_spans = (max_seq_len + span_len - 1) / span_len;
  const AsCacheMode mode = ctx_->GetCacheMode();
  const CUDAContext* gpu_ctx = dynamic_cast<const CUDAContext*>(ctx_);

  // workspace for span attention kernel
  // use max batch and length to claim a maximum workspace size
  size_t device_ws_size{0};
  size_t host_ws_size{0};

  // set to max
  std::vector<int> max_seq_lens(max_batch, max_seq_len);

  DispatchCUDA(dtype_, [&]<typename T>() {
    span::DataType dtype = to_span_data_type<T>();
    span::SpanAttnHandle_t handle{nullptr};
    CHECK_SPAN_ATTN(span::CreateHandle(
        &handle, dtype, CacheUtils::toQuantMode(mode), max_batch,
        attn_head_->NumHeads(), attn_head_->NumGroups(),
        attn_head_->SizePerHead(), span_len, max_num_spans, max_seq_lens.data(),
        dprop_));
    if (handle == nullptr) {
      LOG(ERROR) << "SpanAttnOpCUDA::Reshape: span::CreateHandle failed";
      AS_THROW(AsStatus::ALLSPARK_RUNTIME_ERROR);
    }
    CHECK_SPAN_ATTN(span::GetDeviceWorkspaceSize(&device_ws_size, handle));
    CHECK_SPAN_ATTN(span::GetHostWorkspaceSize(&host_ws_size, handle));
    CHECK_SPAN_ATTN(span::DestroyHandle(handle));
  });

  if (device_ws_size > size_t(std::numeric_limits<int64_t>::max())) {
    LOG(ERROR) << "SpanAttnOpCUDA::Reshape: device workspace size too large";
    return AsStatus::ALLSPARK_EXCEED_LIMIT_ERROR;
  }

  if (host_ws_size > size_t(std::numeric_limits<int64_t>::max())) {
    LOG(ERROR) << "SpanAttnOpCUDA::Reshape: host workspace size too large";
    return AsStatus::ALLSPARK_EXCEED_LIMIT_ERROR;
  }

  AS_CHECK_STATUS(
      tensor_map_->at("workspace")
          ->SetShape(Shape({static_cast<int64_t>(device_ws_size)})));
  AS_CHECK_STATUS(
      host_workspace_->SetShape(Shape({static_cast<int64_t>(host_ws_size)})));
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus SpanAttnOpCUDA::setWorkspace(const RuntimeContext* runtime_ctx) {
  AS_CHECK_STATUS(SpanAttnOp::setWorkspace(runtime_ctx));

  constexpr int single_batch = 1;

  if (!runtime_ctx->is_context) {
    // AS_CHECK_STATUS(setDecoderWorkspaceSize());
  } else {
    std::pair<bool, AsMHAPrefill> prefill_mode_pair = GetPrefillMode();
    if (!prefill_mode_pair.first) {
      LOG(ERROR) << "SpanAttnOpCUDA get prefill mode error. " << std::endl;
      return AsStatus::ALLSPARK_RUNTIME_ERROR;
    }

    // flashv2 switch
    if (prefill_mode_pair.second == AsMHAPrefill::AsPrefillFlashV2) {
#ifdef FLASH_ATTN_V2
      cuda::flashv2_clear_param(flash_v2_params_);

      if (gen_ctx_->prefix_len == 0) {
        // direct calculate flash-v2 using concat-qkv input.
        cuda::flashv2_set_static_param(
            flash_v2_params_, dprop_, to_cuda_type(dtype_), single_batch,
            seq_len_, seq_len_, attn_head_->NumHeads(), attn_head_->NumGroups(),
            attn_head_->SizePerHead(), cuda::FlashQKVFormat::INTERLEAVED,
            causal_mask_);
      } else {
        cuda::flashv2_set_static_param(
            flash_v2_params_, dprop_, to_cuda_type(dtype_), single_batch,
            seq_len_, gen_ctx_->prefix_len + seq_len_, attn_head_->NumHeads(),
            attn_head_->NumGroups(), attn_head_->SizePerHead(),
            cuda::FlashQKVFormat::MIX, causal_mask_);
      }
      size_t flash_workspace_size = cuda::flashv2_wss(flash_v2_params_);
      AS_CHECK_STATUS(tensor_map_->at("workspace")
                          ->SetShape(Shape({int64_t(flash_workspace_size)})));
#else
      LOG(ERROR) << "Flash-Attention is not compiled" << std::endl;
      return AsStatus::ALLSPARK_RUNTIME_ERROR;
#endif  // FLASH_ATTN_V2
    } else if (prefill_mode_pair.second ==
               allspark::AsMHAPrefill::AsPrefillXformer) {
      xformer_params_.causal = causal_mask_;
      xformer_params_.batch = single_batch;
      xformer_params_.nhead_kv = attn_head_->NumGroups();
      xformer_params_.nhead = attn_head_->NumHeads();
      xformer_params_.phead = attn_head_->SizePerHead();
      xformer_params_.seqlen_q = seq_len_;
      xformer_params_.seqlen_k = gen_ctx_->prefix_len + seq_len_;
      xformer_params_.sm_version = dprop_.major << 8 | dprop_.minor;
      xformer_params_.dtype = dtype_;
      if (gen_ctx_->prefix_len == 0) {
        xformer_params_.qkv_format = cuda::XformerQKVFormat::INTERLEAVED;
      } else {
        xformer_params_.qkv_format = cuda::XformerQKVFormat::MIX;
      }
      size_t xformer_workspace_size =
          allspark::cuda::xformer_prefill_attention_workspace_inbytes(
              xformer_params_);
      tensor_map_->at("workspace")
          ->SetShape(Shape({int64_t(xformer_workspace_size)}));
    } else {
      // fallback MHA kernel
      LOG(ERROR) << "SpanAttnOpCUDA::setWorkspace: trivial_prefill_attention "
                    "is deprecated, consider using newer GPUs";
      AS_THROW(AsStatus::ALLSPARK_RUNTIME_ERROR);
#if 0
      if (gen_ctx_->prefix_len != 0) {
        LOG(ERROR)
            << "SpanAttnOpCUDA::setWorkspace: does not support prefix cache";
        return AsStatus::ALLSPARK_RUNTIME_ERROR;
      }
      // not using flash attention
      // only set fallback MHA workspace when context
      trivial_params_.dtype = dtype_;
      trivial_params_.maxlen = ctx_->GetModelMaxLength();
      trivial_params_.batch = 1;
      trivial_params_.nhead = attn_head_->NumHeads();
      trivial_params_.phead = attn_head_->SizePerHead();
      trivial_params_.seqlen = seq_len_;
      size_t trivial_workspace_size = trivial_params_.workspace_inbytes();
      AS_CHECK_STATUS(tensor_map_->at("workspace")
                          ->SetShape(Shape({int64_t(trivial_workspace_size)})));

      const CUDAContext* gpu_ctx = static_cast<const CUDAContext*>(ctx_);
      cublasHandle_t cublas_handle = gpu_ctx->GetCublasHandle();
      AS_CHECK_CUBLAS(cublasSetWorkspace(
          cublas_handle, tensor_map_->at("cublas_workspace")->GetDataPtr(),
          tensor_map_->at("cublas_workspace")->GetSizeInByte()));
#endif
    }
  }
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus SpanAttnOpCUDA::deviceInit() {
  int device_id;
  AS_CHECK_CUDA(cudaGetDevice(&device_id));
  AS_CHECK_CUDA(cudaGetDeviceProperties(&dprop_, device_id));

  DeviceType backend = ctx_->GetDeviceType();

  decoder_seq_len_tensor_device_ =
      std::make_unique<AsTensor>("decoder_seq_len_tensor_device", backend,
                                 DataType::INT32, DataMode::DENSE, Shape{1});

  k_span_array_tensor_device_ = std::make_unique<AsTensor>(
      "k_span_array_tensor_device", backend, DataType::POINTER, DataMode::DENSE,
      Shape{1 * 1});
  v_span_array_tensor_device_ = std::make_unique<AsTensor>(
      "k_span_array_tensor_device", backend, DataType::POINTER, DataMode::DENSE,
      Shape{1 * 1});

  host_workspace_ =
      std::make_unique<AsTensor>("host_workspace", DeviceType::CPU,
                                 DataType::INT8, DataMode::DENSE, Shape{1});
  AS_CHECK_STATUS(setDecoderWorkspaceSize());
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus SpanAttnOpCUDA::deviceReshape(const RuntimeContext* runtime_ctx) {
  const int max_batch = ctx_->GetModelMaxBatch();
  const int max_seq_len = ctx_->GetModelMaxLength();
  const int span_len = ctx_->GetCacheSpanSize();
  const int max_num_spans = (max_seq_len + span_len - 1) / span_len;

  if (runtime_ctx->is_context) {
    AS_CHECK_STATUS(
        k_span_array_tensor_device_->SetShape(Shape({1 * max_num_spans})));
    AS_CHECK_STATUS(
        v_span_array_tensor_device_->SetShape(Shape({1 * max_num_spans})));
  } else {
    AS_CHECK_STATUS(decoder_seq_len_tensor_device_->SetShape(Shape{max_batch}));

    AS_CHECK_STATUS(k_span_array_tensor_device_->SetShape(
        Shape({max_batch * max_num_spans})));
    AS_CHECK_STATUS(v_span_array_tensor_device_->SetShape(
        Shape({max_batch * max_num_spans})));
  }
  return AsStatus::ALLSPARK_SUCCESS;
}

REGISTER_OP(DecOptMHA, CUDA, SpanAttnOpCUDA)
REGISTER_OP(DecOptMQA, CUDA, SpanAttnOpCUDA)

}  // namespace allspark

#undef CHECK_SPAN_ATTN

#endif
#endif
