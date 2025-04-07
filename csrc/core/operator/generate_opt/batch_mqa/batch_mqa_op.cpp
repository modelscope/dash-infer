/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    batch_mqa_op.cpp
 */
#include "batch_mqa_op.h"  // NOLINT

#include <core/kernel/kernel.h>
#include <cpu/cpu_context.h>
#include <cpu/cpu_info.h>
#include <utility/datatype_dispatcher.h>

#include <cmath>
#ifdef ENABLE_CUDA
#include <cuda/cuda_context.h>
#endif
#include <cpu/cpu_context.h>
namespace allspark {

#ifdef ENABLE_CUDA
void gpu_dec_single_mqa(DataType dtype, void* out, void* score,
                        const void* query, const void* key, const void* value,
                        const float* mask, const void* position_embedding,
                        void* k_cache, void* v_cache, void** q_array,
                        void** k_array, void** v_array, void** score_array,
                        void** out_array, int batch_size, int beam_size,
                        int seq_len, int step, int cache_max_len,
                        int hidden_size, int num_heads, int size_per_head,
                        int group_num, int gemm_batch, float alpha,
                        void* other_workspace, size_t other_workspace_size,
                        const DeviceContext* ctx) {
  // LOG(INFO) << "gpu_dec_single_mqa" << std::endl;
  // printf("gpu_dec_single_mqa(batch=%d, beam=%d, seqlen=%d, step=%d,
  // cache=%d,",
  //         batch_size, beam_size, seq_len, step, cache_max_len);
  // printf("hidden=%d, numheads=%d, sizeperhead=%d, gemmbatch=%d)\n",
  //         hidden_size, num_heads, size_per_head, gemm_batch);
  const CUDAContext* gpu_ctx = static_cast<const CUDAContext*>(ctx);
  cublasHandle_t cublas_handle = gpu_ctx->GetCublasHandle();
  if (gpu_ctx->GetMatmulPrecision() == 1 && dtype == FLOAT32) {
    cublasSetMathMode(cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH);
  }
  cudaStream_t cu_stream = gpu_ctx->GetStream();
  auto functor = [&]<typename T>() {
    int kv_stride = size_per_head * group_num;
    int q_stride = hidden_size + 2 * kv_stride;
    cuda::UpdateKVLauncher((T*)k_cache, (T*)v_cache, (const T*)key,
                           (const T*)value, batch_size, step - 1, cache_max_len,
                           kv_stride, seq_len, q_stride, cu_stream);
    if (seq_len != 1) {
      step = seq_len;
    }
    int out_stride = hidden_size;
    int score_stride = step * num_heads;
    cuda::MultiQueryGetBatchArrayLauncher(
        (T*)query, (T*)k_cache, (T*)v_cache, (T*)score, (T*)out, (T**)q_array,
        (T**)k_array, (T**)v_array, (T**)score_array, (T**)out_array,
        batch_size, 1, num_heads, size_per_head, group_num, step,
        q_stride * seq_len, kv_stride * cache_max_len, score_stride * seq_len,
        out_stride * seq_len, cu_stream);

    // batch gemm 1
    cuda::BatchGemmWraper<T>(score_array, q_array, k_array, seq_len, step,
                             size_per_head, false, true, alpha, 0.0f, q_stride,
                             kv_stride, score_stride, gemm_batch,
                             cublas_handle);
    if (position_embedding) {
      cuda::BinaryKernelLauncher((T*)score, (T*)score, (T*)position_embedding,
                                 batch_size * num_heads * step * seq_len, 1,
                                 cu_stream);
    }
    // if (seq_len != 1) {
    // cuda::SoftmaxKernelLauncher((T*)score, mask, batch_size, beam_size,
    //                             num_heads, seq_len, step, cu_stream);
    // } else {
    cuda::StridedSoftmaxLauncher((T*)score, (T*)score, nullptr, nullptr,
                                 other_workspace, other_workspace_size,
                                 batch_size * seq_len * num_heads, step,
                                 cu_stream);
    // }
    // batch gemm 2
    // cuda::BatchedGEMV<T, T>(step, size_per_head, gemm_batch,
    // (T**)score_array,
    //                         (T**)v_array, kv_stride, (T**)out_array,
    //                         other_workspace, other_workspace_size,
    //                         cu_stream);
    cuda::BatchGemmWraper<T>(out_array, score_array, v_array, seq_len,
                             size_per_head, step, false, false, 1.0f, 0.0f,
                             score_stride, kv_stride, out_stride, gemm_batch,
                             cublas_handle);
  };
  DispatchCUDA(dtype, functor);
}
#endif

#if (defined(__x86_64__) || defined(_M_X64)) && defined(ENABLE_AVX512)
void cpu_ctx_single_famqa(DataType dtype, void* out, const void* query,
                          const void* key, const void* value, const float* mask,
                          const void* position_embedding, void* k_cache,
                          void* v_cache, int batch_size, int beam_size,
                          int seq_len, int step, int cache_max_len,
                          int hidden_size, int num_heads, int size_per_head,
                          int group_num, void* workspace, int src_blk,
                          int tgt_blk, float alpha, const DeviceContext* ctx) {
  DLOG(INFO) << "cpu_ctx_single_famqa" << std::endl;
  if (position_embedding) {
    DLOG(WARNING) << "cpu_ctx_single_famqa : can't do position embedding";
  }

  auto functor = [&]<typename T>() {
    int o_stride = hidden_size;

    int q_stride = hidden_size + size_per_head * group_num * 2;
    // use k, v directly, not from KV Cache
    int kv_stride = hidden_size + size_per_head * group_num * 2;

    int total_token_size = 0;
    int input_seq_lens[batch_size], past_seq_lens[batch_size];
    for (int i = 0; i < batch_size; ++i) {
      input_seq_lens[i] = seq_len;
      past_seq_lens[i] = 0;  // since we only run in context phase
      total_token_size += input_seq_lens[i];
    }

    cpu::SelfScaledDpAttention(
        (T*)out, (const T*)query, (const T*)key, (const T*)value, num_heads,
        group_num, size_per_head, o_stride, q_stride, kv_stride, batch_size,
        input_seq_lens, past_seq_lens, workspace, src_blk, tgt_blk, mask, alpha,
        cpu::get_max_threads());

    // copy current key/value to k_cache/v_cache
    cpu::UpdateKVLauncher((T*)k_cache, (T*)v_cache, (const T*)key,
                          (const T*)value, batch_size, step - 1, cache_max_len,
                          size_per_head * group_num, seq_len, q_stride);
  };

  DispatchCPU(dtype, functor);
}
#endif

void cpu_dec_single_mqa(DataType dtype, void* out, void* score,
                        const void* query, const void* key, const void* value,
                        const float* mask, const void* position_embedding,
                        void* k_cache, void* v_cache, void** q_array,
                        void** k_array, void** v_array, void** score_array,
                        void** out_array, int batch_size, int beam_size,
                        int seq_len, int step, int cache_max_len,
                        int hidden_size, int num_heads, int size_per_head,
                        int group_num, int gemm_batch, float alpha,
                        void* other_workspace, size_t other_workspace_size,
                        const DeviceContext* ctx) {
  // LOG(INFO) << "cpu_dec_single_mqa" << std::endl;
  auto functor = [&]<typename T>() {
    int kv_stride = size_per_head * group_num;
    int q_stride = hidden_size + 2 * kv_stride;
    cpu::UpdateKVLauncher((T*)k_cache, (T*)v_cache, (const T*)key,
                          (const T*)value, batch_size, step - 1, cache_max_len,
                          kv_stride, seq_len, q_stride);
    if (seq_len != 1) {
      step = seq_len;
    }
    int out_stride = hidden_size;
    int score_stride = step * num_heads;
    cpu::MultiQueryGetBatchArrayLauncher(
        (T*)query, (T*)k_cache, (T*)v_cache, (T*)score, (T*)out, (T**)q_array,
        (T**)k_array, (T**)v_array, (T**)score_array, (T**)out_array,
        batch_size, 1, num_heads, size_per_head, group_num, step,
        q_stride * seq_len, kv_stride * cache_max_len, score_stride * seq_len,
        out_stride * seq_len);
    cpu::BatchGemmWraper<T>(score_array, q_array, k_array, seq_len, step,
                            size_per_head, false, true, alpha, 0.0f, q_stride,
                            kv_stride, score_stride, gemm_batch);
    if (position_embedding) {
      cpu::SimpleAdd((T*)score, (T*)score, (T*)position_embedding,
                     batch_size * num_heads * step);
    }
    cpu::BatchSoftmax<T>((T*)score, mask, batch_size, beam_size, num_heads,
                         seq_len, step);
    cpu::BatchGemmWraper<T>(out_array, score_array, v_array, seq_len,
                            size_per_head, step, false, false, 1.0f, 0.0f,
                            score_stride, kv_stride, out_stride, gemm_batch);
  };
  DispatchCPU(dtype, functor);
}

AsStatus BatchMQAOp::Init(const OperatorProto& op_proto,
                          const DeviceContext& ctx,
                          const TensorMap& weights_map, TensorMap* tensor_map) {
  AS_CHECK_STATUS(AsOperator::Init(op_proto, ctx, weights_map, tensor_map));

#ifdef ENABLE_CUDA
  if (ctx_->GetDeviceType() == DeviceType::CUDA) {
    int device_id;
    cudaGetDevice(&device_id);
    cudaGetDeviceProperties(&dprop_, device_id);
  }
#endif

  // layer num
  layer_num_ = get_layer_num(this->op_name_);
  if (layer_num_ < 0) {
    LOG(ERROR) << "BatchMQAOp : can't find layer_num_" << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }

  // type inference
  dtype_ = tensor_map_->at(in_names_[0])->GetDataType();
  tensor_map_->at(out_names_[0])->SetDataType(dtype_);

  // attr
  auto& attr_map = op_proto.attr();
  if (attr_map.find("multigpu") != attr_map.end()) {
    multi_nodes_ = *(bool*)(attr_map.at("multigpu").c_str());
  } else {
    // 默认打开多卡
    multi_nodes_ = true;
  }
  if (attr_map.find("num_heads") == attr_map.end()) {
    LOG(ERROR) << "BatchMQAOp : can't find num_heads attribute." << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
  num_heads_ = *(int*)(attr_map.at("num_heads").c_str());
  size_per_head_ = ctx.GetSizePerHead();
  if (attr_map.find("multi_query_group_num") != attr_map.end()) {
    group_num_ = *(int*)(attr_map.at("multi_query_group_num").c_str());
  } else {
    group_num_ = num_heads_;
  }
  // if (attr_map.find("position_embedding") != attr_map.end()) {
  //     pos_embedding_ = true;
  // }
  // if (attr_map.find("alpha") != attr_map.end()) {
  //     alpha_ = *(float*)(attr_map.at("alpha").c_str());
  // }

  DeviceType backend = ctx.GetDeviceType();
  switch (backend) {
#ifdef ENABLE_CUDA
    case DeviceType::CUDA: {
      kernel_launcher = gpu_dec_single_mqa;
      if (multi_nodes_) {
        const CUDAContext* gpu_ctx = static_cast<const CUDAContext*>(ctx_);
        num_heads_ /= gpu_ctx->GetNranks();
        if (group_num_ != 1) {
          group_num_ /= gpu_ctx->GetNranks();
        }
      }
      break;
    }
#endif
    case DeviceType::CPU: {
#if (defined(__x86_64__) || defined(_M_X64)) && defined(ENABLE_AVX512)
      ctx_kernel_launcher = cpu_ctx_single_famqa;
#endif
      kernel_launcher = cpu_dec_single_mqa;
      if (multi_nodes_) {
        const CPUContext* cpu_ctx = static_cast<const CPUContext*>(ctx_);
        num_heads_ /= cpu_ctx->GetNranks();
        if (group_num_ != 1) {
          group_num_ /= cpu_ctx->GetNranks();
        }
      }
      break;
    }
    default:
      LOG(ERROR) << op_type_ << " Operator does not support "
                 << DeviceType_Name(backend) << " device type" << std::endl;
      return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }
  kv_stride_ = size_per_head_ * group_num_;
  hidden_size_ = size_per_head_ * num_heads_;
  return AsStatus::ALLSPARK_SUCCESS;
}

#ifdef ENABLE_CUDA
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

AsStatus BatchMQAOp::setWorkspace(const RuntimeContext* runtime_ctx) {
  DeviceType backend = ctx_->GetDeviceType();
  if (backend != DeviceType::CUDA) {
    return AsStatus::ALLSPARK_SUCCESS;
  }
  constexpr int single_batch = 1;

  if (runtime_ctx->is_context) {
    std::pair<bool, AsMHAPrefill> prefill_mode_pair = GetPrefillMode();
    if (!prefill_mode_pair.first) {
      LOG(ERROR) << "SpanMhaOpCUDA get prefill mode error. " << std::endl;
      return AsStatus::ALLSPARK_RUNTIME_ERROR;
    }

    // flashv2 switch
    if (prefill_mode_pair.second == AsMHAPrefill::AsPrefillFlashV2) {
#ifdef FLASH_ATTN_V2
      cuda::flashv2_clear_param(flash_v2_params_);
      // direct calculate flash-v2 using concat-qkv input.
      cuda::flashv2_set_static_param(
          flash_v2_params_, dprop_, to_cuda_type(dtype_), single_batch,
          seq_len_, seq_len_, num_heads_, group_num_, size_per_head_,
          cuda::FlashQKVFormat::INTERLEAVED, causal_mask_);
      size_t flash_workspace_size = cuda::flashv2_wss(flash_v2_params_);
      AS_CHECK_STATUS(tensor_map_->at("workspace")
                          ->SetShape(Shape({int64_t(flash_workspace_size)})));
#else
      LOG(ERROR) << "Flash-Attention is not compiled";
      AS_THROW(AsStatus::ALLSPARK_RUNTIME_ERROR);
#endif  // FLASH_ATTN_V2
    } else if (prefill_mode_pair.second ==
               allspark::AsMHAPrefill::AsPrefillXformer) {
      xformer_params_.causal = causal_mask_;
      xformer_params_.batch = single_batch;
      xformer_params_.nhead = num_heads_;
      xformer_params_.phead = size_per_head_;
      xformer_params_.seqlen_q = seq_len_;
      xformer_params_.seqlen_k = seq_len_;
      xformer_params_.sm_version = dprop_.major << 8 | dprop_.minor;
      xformer_params_.dtype = dtype_;
      xformer_params_.nhead_kv = group_num_;
      xformer_params_.qkv_format = cuda::XformerQKVFormat::INTERLEAVED;
      size_t xformer_workspace_size =
          allspark::cuda::xformer_prefill_attention_workspace_inbytes(
              xformer_params_);
      tensor_map_->at("workspace")
          ->SetShape(Shape({int64_t(xformer_workspace_size)}));
    } else {
      // not using flash attention
      // only set fallback MHA workspace when context
      trivial_params_.dtype = dtype_;
      trivial_params_.maxlen = ctx_->GetModelMaxLength();
      trivial_params_.batch = 1;
      trivial_params_.nhead = num_heads_;
      trivial_params_.phead = size_per_head_;
      trivial_params_.seqlen = seq_len_;
      size_t trivial_workspace_size = trivial_params_.workspace_inbytes();
      AS_CHECK_STATUS(tensor_map_->at("workspace")
                          ->SetShape(Shape({int64_t(trivial_workspace_size)})));

      const CUDAContext* gpu_ctx = static_cast<const CUDAContext*>(ctx_);
      cublasHandle_t cublas_handle = gpu_ctx->GetCublasHandle();
      AS_CHECK_CUBLAS(cublasSetWorkspace(
          cublas_handle, tensor_map_->at("cublas_workspace")->GetDataPtr(),
          tensor_map_->at("cublas_workspace")->GetSizeInByte()));
    }
  }
  return AsStatus::ALLSPARK_SUCCESS;
}
#endif
AsStatus BatchMQAOp::Reshape(RuntimeContext* runtime_ctx) {
  const AsTensor* x = tensor_map_->at(in_names_[0]).get();
  const Shape& x_shape = x->GetShape();
  Shape y_shape(x_shape);
  int single_batch = 1;
  batch_size_ = y_shape[0];
  seq_len_ = y_shape[1];
  qkv_stride_ = y_shape[2];
  if (qkv_stride_ != hidden_size_ + 2 * kv_stride_) {
    LOG(ERROR) << "Invalid qkv_stride_ in BatchMQAOp"
               << "qkv_strde = " << qkv_stride_
               << ",hidden_size = " << hidden_size_
               << ", kv_stride = " << kv_stride_ << std::endl;
    return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }
  // set variable
  gemm_batch_ = single_batch * num_heads_;
  if (alpha_ < 0) {
    alpha_ = 1.0f / std::sqrt(size_per_head_ * 1.0f);
  }
  score_size_ =
      round32((int64_t)single_batch * ctx_->GetModelMaxLength() * num_heads_ *
              (ctx_->GetModelMaxLength()) * SizeofType(dtype_));
  int min_blk = (int)std::pow(2, int(std::log2(seq_len_ / 2)));
  src_blk_ = std::min(256, min_blk);
  tgt_blk_ = std::min(512, seq_len_);
  int64_t ws_size =
      score_size_ + (int64_t)sizeof(void*) * round32(gemm_batch_) * 5;

  other_workspace_size_ = 0;

#ifdef ENABLE_CUDA
  if (ctx_->GetDeviceType() == DeviceType::CUDA) {
    size_t softmax_workspace = 0;
    cuda::StridedSoftmaxGetWorkspaceSize<float>(
        &softmax_workspace, ctx_->GetModelMaxLength() * num_heads_,
        ctx_->GetModelMaxLength());

    other_workspace_size_ = std::max(other_workspace_size_, softmax_workspace);
    size_t gemv_workspace = cuda::GetBatchedGEMVWorkspaceSize<float, float>(
        ctx_->GetModelMaxLength(), size_per_head_, gemm_batch_);
    other_workspace_size_ = std::max(other_workspace_size_, gemv_workspace);
    AS_CHECK_STATUS(setWorkspace(runtime_ctx));
  }
#endif

#if (defined(__x86_64__) || defined(_M_X64)) && defined(ENABLE_AVX512)
  // if we are in runContext phase, it is possible to reduce memory alloc since
  // flash attention will be adopted.
  if (runtime_ctx->is_context && UseFlashAttn()) {
    // each omp thread will hold intermediate data
    //   - Q_i shape: [src_blk * size_per_head]
    //   - S_ij = Q_iK^T_j shape: [src_blk * tgt_blk]
    //   - l_i, preSum shape: [src_blk, 1]
    //   - l_i_new, sum shape: [src_blk, 1]
    //   - m_i, preMax shape: [src_blk, 1]
    //   - m_i_new, max shape: [src_blk, 1]
    //   - O_i shape: [src_blk * head_dim]
    int64_t ws_size_per_omp_thread =
        (4 + tgt_blk_ + 2 * size_per_head_) * src_blk_;
    ws_size =
        cpu::get_max_threads() * SizeofType(dtype_) * ws_size_per_omp_thread;
    // each omp thread will hold its own offset pointer to above data
    ws_size += cpu::get_max_threads() * 7 * sizeof(void*);
  }
#endif
  tensor_map_->at("workspace")
      ->SetShape(Shape({ws_size + (int64_t)other_workspace_size_}));
  y_shape[2] = hidden_size_;
  tensor_map_->at(out_names_[0])->SetShape(std::move(y_shape));

  if (runtime_ctx->is_context) {
    // 初始化cachememory，每次多申请kv_size个token长度的cache，默认kv_size就是engine_max_length
    int64_t per_size = ctx_->GetKVcacheSize() * kv_stride_ * SizeofType(dtype_);
    runtime_ctx->GetContextGenCtx()->k_cache_list.push_back(
        std::make_unique<CacheMemory>(ctx_->GetDeviceType(), per_size));
    runtime_ctx->GetContextGenCtx()->v_cache_list.push_back(
        std::make_unique<CacheMemory>(ctx_->GetDeviceType(), per_size));
  }
  return AsStatus::ALLSPARK_SUCCESS;
}

#if (defined(__x86_64__) || defined(_M_X64)) && defined(ENABLE_AVX512)
AsStatus BatchMQAOp::RunFlash(GenerateContext* gen_ctx) {
  AsTensor* in_tensor = tensor_map_->at(in_names_[0]).get();
  AsTensor* out_tensor = tensor_map_->at(out_names_[0]).get();

  // alloc kv_cache
  int64_t kv_size =
      (int64_t)(gen_ctx->step + seq_len_) * kv_stride_ * SizeofType(dtype_);
  gen_ctx->k_cache_list[layer_num_]->Alloc(kv_size);
  gen_ctx->v_cache_list[layer_num_]->Alloc(kv_size);
  void* k_cache_buf = gen_ctx->k_cache_list[layer_num_]->GetData();
  void* v_cache_buf = gen_ctx->v_cache_list[layer_num_]->GetData();

  // prepare input buf
  // qkv
  void* out_ptr = (char*)out_tensor->GetDataPtr();
  void* q_buf = (char*)in_tensor->GetDataPtr();
  void* k_buf = (char*)q_buf + hidden_size_ * SizeofType(dtype_);
  void* v_buf = (char*)k_buf + kv_stride_ * SizeofType(dtype_);
  // mask
  float* mask_buf = gen_ctx->step == 0
                        ? (float*)(tensor_map_->at(in_names_[1])->GetDataPtr())
                        : nullptr;
  if (tensor_map_->at(in_names_[1])->GetShape().Count() == 0) {
    mask_buf = nullptr;
  }
  // position
  void* position_embedding =
      pos_embedding_ ? tensor_map_->at(in_names_[2])->GetDataPtr() : nullptr;

  // workspace
  void* workspace = (char*)(tensor_map_->at("workspace")->GetDataPtr());

  ctx_kernel_launcher(
      dtype_, out_ptr, q_buf, k_buf, v_buf, mask_buf, position_embedding,
      k_cache_buf, v_cache_buf, 1, gen_ctx->num_beams, seq_len_,
      (gen_ctx->step + 1), ctx_->GetModelMaxLength(), hidden_size_, num_heads_,
      size_per_head_, group_num_, workspace, src_blk_, tgt_blk_, alpha_, ctx_);

  return AsStatus::ALLSPARK_SUCCESS;
}
#endif

AsStatus BatchMQAOp::RunOneBatch(GenerateContext* gen_ctx, int current_batch) {
  // LOG(INFO) << "BatchMQAOp::RunOneBatch, step: " << gen_ctx->step
  //           << ", seq_len: " << seq_len_ << std::endl;

  AsTensor* in_tensor = tensor_map_->at(in_names_[0]).get();
  AsTensor* out_tensor = tensor_map_->at(out_names_[0]).get();

  // alloc kv_cache
  int64_t kv_size =
      (int64_t)(gen_ctx->step + seq_len_) * kv_stride_ * SizeofType(dtype_);
  gen_ctx->k_cache_list[layer_num_]->Alloc(kv_size);
  gen_ctx->v_cache_list[layer_num_]->Alloc(kv_size);
  void* k_cache_buf = gen_ctx->k_cache_list[layer_num_]->GetData();
  void* v_cache_buf = gen_ctx->v_cache_list[layer_num_]->GetData();

  // prepare input buf
  // qkv
  int offset = hidden_size_ * SizeofType(dtype_);
  void* out_ptr =
      (char*)out_tensor->GetDataPtr() + current_batch * seq_len_ * offset;
  void* q_buf = (char*)in_tensor->GetDataPtr() +
                current_batch * seq_len_ * (qkv_stride_ * SizeofType(dtype_));
  void* k_buf = (char*)q_buf + hidden_size_ * SizeofType(dtype_);
  void* v_buf = (char*)k_buf + kv_stride_ * SizeofType(dtype_);
  // mask
  float* mask_buf = gen_ctx->step == 0
                        ? (float*)(tensor_map_->at(in_names_[1])->GetDataPtr())
                        : nullptr;
  if (tensor_map_->at(in_names_[1])->GetShape().Count() == 0) {
    mask_buf = nullptr;
  }
  // position
  void* position_embedding =
      pos_embedding_ ? tensor_map_->at(in_names_[2])->GetDataPtr() : nullptr;

  // workspace
  char* score_buf = (char*)(tensor_map_->at("workspace")->GetDataPtr());
  void** q_array = (void**)(score_buf + score_size_);
  void** k_array = q_array + round32(gemm_batch_);
  void** v_array = k_array + round32(gemm_batch_);
  void** score_array = v_array + round32(gemm_batch_);
  void** out_array = score_array + round32(gemm_batch_);
  void* other_workspace = out_array + round32(gemm_batch_);

  kernel_launcher(dtype_, out_ptr, score_buf, q_buf, k_buf, v_buf, mask_buf,
                  position_embedding, k_cache_buf, v_cache_buf, q_array,
                  k_array, v_array, score_array, out_array, 1,
                  gen_ctx->num_beams, seq_len_, (gen_ctx->step + 1),
                  ctx_->GetModelMaxLength(), hidden_size_, num_heads_,
                  size_per_head_, group_num_, gemm_batch_, alpha_,
                  other_workspace, other_workspace_size_, ctx_);

  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus BatchMQAOp::RunContext(RuntimeContext* runtime_ctx) {
  if (batch_size_ != 1) {
    LOG(ERROR) << "BatchMQAOp only support multibatch in decoder pharse, "
                  "not context pharse."
               << std::endl;
    return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }
  GenerateContext* gen_ctx = runtime_ctx->GetContextGenCtx();
  DeviceType backend = ctx_->GetDeviceType();
  switch (backend) {
#ifdef ENABLE_CUDA
    case DeviceType::CUDA: {
      AsTensor* in_tensor = tensor_map_->at(in_names_[0]).get();
      AsTensor* out_tensor = tensor_map_->at(out_names_[0]).get();
      AsTensor* wss_tensor = tensor_map_->at("workspace").get();
      int64_t kv_size =
          (int64_t)(gen_ctx->step + seq_len_) * kv_stride_ * SizeofType(dtype_);
      gen_ctx->k_cache_list[layer_num_]->Alloc(kv_size);
      gen_ctx->v_cache_list[layer_num_]->Alloc(kv_size);
      void* k_cache_buf = gen_ctx->k_cache_list[layer_num_]->GetData();
      void* v_cache_buf = gen_ctx->v_cache_list[layer_num_]->GetData();
      char* cptr = (char*)in_tensor->GetDataPtr();
      char* qptr = cptr;
      char* kptr = qptr + hidden_size_ * SizeofType(dtype_);
      char* vptr = kptr + kv_stride_ * SizeofType(dtype_);
      constexpr int current_batch = 0;

      std::pair<bool, AsMHAPrefill> prefill_mode_pair = GetPrefillMode();
      if (!prefill_mode_pair.first) {
        LOG(ERROR) << "SpanMhaOpCUDA get prefill mode error. " << std::endl;
        AS_THROW(AsStatus::ALLSPARK_RUNTIME_ERROR);
        // return;
      }
      // flashv2 logic
      if (prefill_mode_pair.second == AsMHAPrefill::AsPrefillFlashV2) {
#ifdef FLASH_ATTN_V2
        char* cptr = (char*)in_tensor->GetDataPtr();
        char* qptr = cptr;
        char* kptr = qptr + hidden_size_ * SizeofType(dtype_);
        char* vptr = kptr + kv_stride_ * SizeofType(dtype_);
        char* optr = (char*)out_tensor->GetDataPtr();
        char* wptr = (char*)wss_tensor->GetDataPtr();
        cuda::flashv2_set_runtime_param(flash_v2_params_, qptr, kptr, vptr,
                                        optr, wptr, alpha_);
        cuda::flashv2_dispatch(
            flash_v2_params_,
            static_cast<const CUDAContext*>(ctx_)->GetStream());

        void* kbuf = (char*)cptr + hidden_size_ * SizeofType(dtype_);
        void* vbuf = (char*)kbuf + kv_stride_ * SizeofType(dtype_);
        DispatchCUDA(dtype_, [&]<typename T>() {
          cuda::UpdateKVLauncher(
              (T*)k_cache_buf, (T*)v_cache_buf, (const T*)kbuf, (const T*)vbuf,
              1, 0, seq_len_, kv_stride_, seq_len_, qkv_stride_,
              static_cast<const CUDAContext*>(ctx_)->GetStream());
        });
#else
        LOG(ERROR) << "Flash-Attention is not compiled" << std::endl;
        return AsStatus::ALLSPARK_RUNTIME_ERROR;
#endif  // FLASH_ATTN_V2
      } else if (prefill_mode_pair.second == AsMHAPrefill::AsPrefillXformer) {
        char* cptr = (char*)in_tensor->GetDataPtr();
        char* qptr = cptr;
        char* kptr = qptr + hidden_size_ * SizeofType(dtype_);
        char* vptr = kptr + kv_stride_ * SizeofType(dtype_);
        char* optr = (char*)out_tensor->GetDataPtr();
        char* wptr = (char*)wss_tensor->GetDataPtr();
        allspark::cuda::xformer_prefill_attention(
            xformer_params_, qptr, kptr, vptr, optr, wptr, alpha_,
            static_cast<const CUDAContext*>(ctx_)->GetStream());
        void* kbuf = (char*)cptr + hidden_size_ * SizeofType(dtype_);
        void* vbuf = (char*)kbuf + kv_stride_ * SizeofType(dtype_);
        DispatchCUDA(dtype_, [&]<typename T>() {
          cuda::UpdateKVLauncher(
              (T*)k_cache_buf, (T*)v_cache_buf, (const T*)kbuf, (const T*)vbuf,
              1, 0, seq_len_, kv_stride_, seq_len_, qkv_stride_,
              static_cast<const CUDAContext*>(ctx_)->GetStream());
        });
      } else {
        RunOneBatch(gen_ctx, 0);
      }
      break;
    }
      DLOG(INFO) << "BatchMQAOp::RunContext [" << gen_ctx->request->request_id
                 << "][layer " << layer_num_
                 << "], PrefillMode = " << int(ctx_->GetPrefillMode());
#endif
    case DeviceType::CPU: {
#if (defined(__x86_64__) || defined(_M_X64)) && defined(ENABLE_AVX512)
      if (UseFlashAttn()) {
        RunFlash(gen_ctx);
      } else {
#endif
        RunOneBatch(gen_ctx, 0);
#if (defined(__x86_64__) || defined(_M_X64)) && defined(ENABLE_AVX512)
      }
#endif
      break;
    }
    default:
      LOG(ERROR) << op_type_ << " Operator does not support "
                 << DeviceType_Name(backend) << " device type" << std::endl;
      return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus BatchMQAOp::RunDecoder(RuntimeContext* runtime_ctx) {
  for (int batch = 0; batch < batch_size_; batch++) {
    GenerateContext* gen_ctx = runtime_ctx->GetGenCtx(batch);
    RunOneBatch(gen_ctx, batch);
  }
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus BatchMQAOp::Forward(RuntimeContext* runtime_ctx) {
  AsStatus status;
  if (runtime_ctx->is_context) {
    status = RunContext(runtime_ctx);
  } else {
    status = RunDecoder(runtime_ctx);
  }
  return status;
}

AsStatus BatchMQAOp::ResetCache() { return AsStatus::ALLSPARK_SUCCESS; }

#if defined(ENABLE_SPAN_ATTENTION) && (ENABLE_SPAN_ATTENTION == OFF)
REGISTER_OP(DecOptMQA, CUDA, BatchMQAOp)
#endif
REGISTER_OP(DecOptMQA, CPU, BatchMQAOp)
}  // namespace allspark
