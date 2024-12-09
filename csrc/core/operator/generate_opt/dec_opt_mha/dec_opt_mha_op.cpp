/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    dec_opt_mha_op.cpp
 */

#include "dec_opt_mha_op.h"  // NOLINT

#include <core/kernel/kernel.h>
#include <utility/datatype_dispatcher.h>

#include <cmath>
#ifdef ENABLE_CUDA
#include <cuda/cuda_context.h>
#endif
#include <cpu/cpu_context.h>
namespace allspark {

#ifdef ENABLE_CUDA
void gpu_dec_opt_mha(DataType dtype, void* out, void* score, const void* query,
                     const void* key, const void* value, const float* mask,
                     const void* position_embedding, void* k_cache,
                     void* v_cache, void** q_array, void** k_array,
                     void** v_array, void** score_array, void** out_array,
                     int batch_size, int beam_size, int seq_len, int step,
                     int cache_max_len, int hidden_size, int num_heads,
                     int size_per_head, int gemm_batch, float alpha,
                     int input_len, bool xlogn_enable, int xlogn_len,
                     const DeviceContext* ctx) {
#ifdef CONFIG_DEBUG_OP
  DLOG(INFO) << "gpu_dec_opt_mha" << std::endl;
#endif
  // printf("gpu_dec_opt_mha(batch=%d, beam=%d, seqlen=%d, step=%d, cache=%d,",
  //         batch_size, beam_size, seq_len, step, cache_max_len);
  // printf("hidden=%d, numheads=%d, sizeperhead=%d, gemmbatch=%d)\n",
  //         hidden_size, num_heads, size_per_head, gemm_batch);
  const CUDAContext* gpu_ctx = static_cast<const CUDAContext*>(ctx);
  cublasHandle_t cublas_handle = gpu_ctx->GetCublasHandle();
  if (gpu_ctx->GetMatmulPrecision() == PrecisionLevel::HIGH &&
      dtype == FLOAT32) {
    cublasSetMathMode(cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH);
  }
  cudaStream_t cu_stream = gpu_ctx->GetStream();
  auto functor = [&]<typename T>() {
    auto is_prefix = [&]() -> bool { return seq_len != 1; };
    cuda::UpdateKVLauncher((T*)k_cache, (T*)v_cache, (const T*)key,
                           (const T*)value, batch_size, step - 1, cache_max_len,
                           hidden_size, seq_len, 3 * hidden_size, cu_stream);
    if (is_prefix()) step = seq_len;
    int q_stride = hidden_size * 3;
    int kv_stride = hidden_size;
    int out_stride = hidden_size;
    int score_stride = step * num_heads;
    cuda::GetBatchArrayLauncher(
        (T*)query, (T*)k_cache, (T*)v_cache, (T*)score, (T*)out, (T**)q_array,
        (T**)k_array, (T**)v_array, (T**)score_array, (T**)out_array,
        batch_size, 1, num_heads, size_per_head, step, q_stride * seq_len,
        kv_stride * cache_max_len, score_stride * seq_len, out_stride * seq_len,
        cu_stream);

    // batch gemm 1
    float qxk_scale = alpha;
    if (!is_prefix() && xlogn_enable && step > xlogn_len) {
      // logn logic, if decoder, query require log_xlogn(xseql) * alpha scale if
      // longer than model basic length (xlogn). for encoder / prefix, this
      // logices are implement in softmax kernel.
      qxk_scale *= logf(step) / logf(xlogn_len);
    }
    cuda::BatchGemmWraper<T>(score_array, q_array, k_array, seq_len, step,
                             size_per_head, false, true, qxk_scale, 0.0f,
                             q_stride, kv_stride, score_stride, gemm_batch,
                             cublas_handle);
    if (position_embedding) {
      cuda::BinaryKernelLauncher((T*)score, (T*)score, (T*)position_embedding,
                                 batch_size * num_heads * step * seq_len, 1,
                                 cu_stream);
    }
    if (is_prefix() && xlogn_enable) {
      // logn prefix
      if (beam_size != 1) {
        LOG(ERROR) << "Logn attention not support beam search. "
                   << "disgard input beam param = " << beam_size << "."
                   << std::endl;
      }
      cuda::LognSoftmaxKernelLauncher((T*)score, mask, batch_size, num_heads,
                                      seq_len, step, xlogn_len, cu_stream);
    } else if (is_prefix() && !xlogn_enable) {
      // normal prefix
      cuda::SoftmaxKernelLauncher((T*)score, mask, batch_size, beam_size,
                                  num_heads, seq_len, step, cu_stream);
    } else {
      // decoder softmax
      cuda::DecoderSoftmaxKernelLauncher((T*)score, mask, batch_size, beam_size,
                                         num_heads, seq_len, step, input_len,
                                         cu_stream);
    }

    // batch gemm 2
    cuda::BatchGemmWraper<T>(out_array, score_array, v_array, seq_len,
                             size_per_head, step, false, false, 1.0f, 0.0f,
                             score_stride, kv_stride, out_stride, gemm_batch,
                             cublas_handle);
  };
  DispatchCUDA(dtype, functor);
}

void gpu_reorder_kv_cache(DataType dtype, void* k_cache, void* v_cache,
                          void* old_k_cache, void* old_v_cache, int* beam_idx,
                          int batch_size, int beam_size, int inner_dim,
                          const DeviceContext* ctx) {
  const CUDAContext* gpu_ctx = static_cast<const CUDAContext*>(ctx);
  auto functor = [&]<typename T>() {
    cuda::ReorderKVCacheLauncher((T*)k_cache, (T*)v_cache, (T*)old_k_cache,
                                 (T*)old_v_cache, beam_idx, batch_size,
                                 beam_size, inner_dim, gpu_ctx->GetStream());
  };
  DispatchCUDA(dtype, functor);
}
#endif

void cpu_dec_opt_mha(DataType dtype, void* out, void* score, const void* query,
                     const void* key, const void* value, const float* mask,
                     const void* position_embedding, void* k_cache,
                     void* v_cache, void** q_array, void** k_array,
                     void** v_array, void** score_array, void** out_array,
                     int batch_size, int beam_size, int seq_len, int step,
                     int cache_max_len, int hidden_size, int num_heads,
                     int size_per_head, int gemm_batch, float alpha,
                     int input_len, bool xlogn_enable,
                     int xlogn_len, /* TODO(cpu-logn): not implement yet. */
                     const DeviceContext* ctx) {
  DLOG(INFO) << "cpu_dec_opt_mha" << std::endl;
  if (xlogn_enable) {
    LOG(ERROR) << "Logn attention still not support on CPU. " << std::endl;
  }
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
    if (seq_len != 1 || mask == nullptr) {
      cpu::BatchSoftmax<T>((T*)score, mask, batch_size, beam_size, num_heads,
                           seq_len, step);
    } else {
      cpu::BatchDecoderSoftmax<T>((T*)score, mask, batch_size, beam_size,
                                  num_heads, seq_len, step, input_len);
    }
    cpu::BatchGemmWraper<T>(out_array, score_array, v_array, seq_len,
                            size_per_head, step, false, false, 1.0f, 0.0f,
                            score_stride, kv_stride, out_stride, gemm_batch);
  };
  DispatchCPU(dtype, functor);
}
void cpu_reorder_kv_cache(DataType dtype, void* k_cache, void* v_cache,
                          void* old_k_cache, void* old_v_cache, int* beam_idx,
                          int batch_size, int beam_size, int inner_dim,
                          const DeviceContext* ctx) {
  auto functor = [&]<typename T>() {
    cpu::ReorderKVCacheLauncher((T*)k_cache, (T*)v_cache, (T*)old_k_cache,
                                (T*)old_v_cache, beam_idx, batch_size,
                                beam_size, inner_dim);
  };
  DispatchCPU(dtype, functor);
}
AsStatus DecOptMHAOp::Init(const OperatorProto& op_proto,
                           const DeviceContext& ctx,
                           const TensorMap& weights_map,
                           TensorMap* tensor_map) {
  AS_CHECK_STATUS(AsOperator::Init(op_proto, ctx, weights_map, tensor_map));
  // type inference
  dtype_ = tensor_map_->at(in_names_[0])->GetDataType();
  tensor_map_->at(out_names_[0])->SetDataType(dtype_);
  // attr
  auto& attr_map = op_proto.attr();
  if (attr_map.find("num_heads") == attr_map.end()) {
    LOG(ERROR) << "DecOptMHAOp : can't find num_heads attribute." << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
  num_heads_ = *(int*)(attr_map.at("num_heads").c_str());
  if (attr_map.find("multigpu") != attr_map.end()) {
    multi_nodes_ = *(bool*)(attr_map.at("multigpu").c_str());
  } else {
    multi_nodes_ = true;
  }
  if (attr_map.find("position_embedding") != attr_map.end()) {
    pos_embedding_ = true;
  }
  if (attr_map.find("alpha") != attr_map.end()) {
    alpha_ = *(float*)(attr_map.at("alpha").c_str());
  }
  AS_CHECK_STATUS(logn_from_attributes_(op_proto));
  DeviceType backend = ctx.GetDeviceType();
  switch (backend) {
#ifdef ENABLE_CUDA
    case DeviceType::CUDA: {
      kernel_launcher = gpu_dec_opt_mha;
      reorder_kv_cache_launcher = gpu_reorder_kv_cache;
      if (multi_nodes_) {
        const CUDAContext* gpu_ctx = static_cast<const CUDAContext*>(ctx_);
        num_heads_ /= gpu_ctx->GetNranks();
      }
      break;
    }
#endif
    case DeviceType::CPU: {
      kernel_launcher = cpu_dec_opt_mha;
      reorder_kv_cache_launcher = cpu_reorder_kv_cache;
      if (multi_nodes_) {
        const CPUContext* cpu_ctx = static_cast<const CPUContext*>(ctx_);
        num_heads_ /= cpu_ctx->GetNranks();
      }
      break;
    }
    default:
      LOG(ERROR) << op_type_ << " Operator does not support "
                 << DeviceType_Name(backend) << " device type" << std::endl;
      return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }
  k_cache_ = std::make_unique<AsTensor>("k_cache", backend, dtype_,
                                        DataMode::DENSE, Shape{1});
  v_cache_ = std::make_unique<AsTensor>("v_cache", backend, dtype_,
                                        DataMode::DENSE, Shape{1});
  tmp_k_cache_ = std::make_unique<AsTensor>("tmp_k_cache", backend, dtype_,
                                            DataMode::DENSE, Shape{1});
  tmp_v_cache_ = std::make_unique<AsTensor>("tmp_v_cache", backend, dtype_,
                                            DataMode::DENSE, Shape{1});
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus DecOptMHAOp::Reshape() {
  const AsTensor* x = tensor_map_->at(in_names_[0]).get();
  const Shape& x_shape = x->GetShape();
  Shape y_shape(x_shape);
  y_shape[2] /= 3;
  batch_size_ = y_shape[0];
  seq_len_ = y_shape[1];
  hidden_size_ = y_shape[2];
  // set variable
  if (hidden_size_ % num_heads_) {
    LOG(ERROR) << "Invalid attribute in DecOptMHAOp. hidden_size : "
               << hidden_size_ << ", num_heads : " << num_heads_ << std::endl;
    return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }
  size_per_head_ = hidden_size_ / num_heads_;
  gemm_batch_ = batch_size_ * num_heads_;
  // cublasGemmBatchedEx接口需要地址对齐
  if (dtype_ == DataType::FLOAT16 || dtype_ == DataType::BFLOAT16) {
    if ((seq_len_ % 8 == 0 && size_per_head_ % 16 != 0) ||
        (seq_len_ % 2 == 0 && size_per_head_ % 4 != 0)) {
      LOG(ERROR)
          << "Invalid seq_len_ &  size_per_head_ in DecOptMHAOp. seq_len_ = "
          << seq_len_ << ", size_per_head_ : " << size_per_head_ << std::endl;
      return AsStatus::ALLSPARK_RUNTIME_ERROR;
    }
  }
  if (alpha_ < 0) {
    alpha_ = 1.0f / std::sqrt(size_per_head_ * 1.0f);
  }
  score_size_ = round32((int64_t)batch_size_ * seq_len_ * num_heads_ *
                        (gen_ctx_->max_length) * SizeofType(dtype_));
  int64_t ws_size =
      score_size_ + (int64_t)sizeof(void*) * round32(gemm_batch_) * 5;
  tensor_map_->at("workspace")->SetShape(Shape({ws_size}));
  tensor_map_->at(out_names_[0])->SetShape(std::move(y_shape));

  //  reshape只会在变大的时候重新malloc，所以预先申请最大size，这样qkv
  //  cache在后面运行时就不会变化了
  int64_t max_kv_size =
      (int64_t)std::max(gen_ctx_->max_length, ctx_->GetModelMaxLength()) *
      std::max(gen_ctx_->batch_size, ctx_->GetModelMaxBatch()) * hidden_size_ *
      gen_ctx_->gen_cfg.num_beams;
  k_cache_->SetShape(Shape{max_kv_size});
  v_cache_->SetShape(Shape{max_kv_size});
  k_cache_buf_ = k_cache_->GetDataPtr();
  v_cache_buf_ = v_cache_->GetDataPtr();
  if (gen_ctx_->generate_method == 1) {
    tmp_k_cache_->SetShape(Shape{max_kv_size});
    tmp_v_cache_->SetShape(Shape{max_kv_size});
    old_k_cache_buf_ = tmp_k_cache_->GetDataPtr();
    old_v_cache_buf_ = tmp_v_cache_->GetDataPtr();
  }
#ifdef ENABLE_CUDA
  if (ctx_->GetDeviceType() == DeviceType::CUDA) {
    const CUDAContext* gpu_ctx = static_cast<const CUDAContext*>(ctx_);
    cublasHandle_t cublas_handle = gpu_ctx->GetCublasHandle();
    cublasSetWorkspace(cublas_handle,
                       tensor_map_->at("cublas_workspace")->GetDataPtr(),
                       tensor_map_->at("cublas_workspace")->GetSizeInByte());
  }
#endif
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus DecOptMHAOp::Forward() {
  AsTensor* in_tensor = tensor_map_->at(in_names_[0]).get();
  int offset = hidden_size_ * SizeofType(dtype_);
  void* q_buf = in_tensor->GetDataPtr();
  void* k_buf = (char*)q_buf + offset;
  void* v_buf = (char*)k_buf + offset;

  // float* mask_buf =
  //     gen_ctx_->step == 0
  //         ? (float*)(tensor_map_->at(in_names_[1])->GetDataPtr())
  //         : nullptr;
  float* mask_buf = (float*)(tensor_map_->at(in_names_[1])->GetDataPtr());
  if (tensor_map_->at(in_names_[1])->GetShape().Count() == 0) {
    mask_buf = nullptr;
  }
  void* position_embedding =
      pos_embedding_ ? tensor_map_->at(in_names_[2])->GetDataPtr() : nullptr;
  char* score_buf = (char*)(tensor_map_->at("workspace")->GetDataPtr());
  void** q_array = (void**)(score_buf + score_size_);
  void** k_array = q_array + round32(gemm_batch_);
  void** v_array = k_array + round32(gemm_batch_);
  void** score_array = v_array + round32(gemm_batch_);
  void** out_array = score_array + round32(gemm_batch_);

  if (seq_len_ != 1) {
    input_len_ = seq_len_;
  }
  if (gen_ctx_->input_len != 0) {
    input_len_ = gen_ctx_->input_len;
  }
  if (gen_ctx_->current_batch != 0) {
    k_cache_buf_ = ((char*)k_cache_->GetDataPtr()) +
                   gen_ctx_->current_batch * gen_ctx_->max_length *
                       hidden_size_ * SizeofType(dtype_);
    v_cache_buf_ = ((char*)v_cache_->GetDataPtr()) +
                   gen_ctx_->current_batch * gen_ctx_->max_length *
                       hidden_size_ * SizeofType(dtype_);
  }
  // beam search kv update
  int* beam_idx = nullptr;
  if ((gen_ctx_->generate_method == 1) && gen_ctx_->step != 0) {
    if (first_beam_) {  // first generate,need copy [batch,1,max_len,hidden]
                        // to [batch,beam,max_len,hidden]
      beam_idx = nullptr;
      first_beam_ = false;
    } else {
      beam_idx = (int*)(tensor_map_->at(in_names_[in_names_.size() - 1])
                            ->GetDataPtr());
    }
    std::swap(k_cache_buf_, old_k_cache_buf_);
    std::swap(v_cache_buf_, old_v_cache_buf_);
    reorder_kv_cache_launcher(
        dtype_, k_cache_buf_, v_cache_buf_, old_k_cache_buf_, old_v_cache_buf_,
        beam_idx, batch_size_ / gen_ctx_->num_beams, gen_ctx_->num_beams,
        gen_ctx_->max_length * hidden_size_, ctx_);
  }

  if (gen_ctx_->step == 0) {
    DeviceType backend = ctx_->GetDeviceType();
    size_t kvcache_size = gen_ctx_->batch_size * gen_ctx_->max_length *
                          hidden_size_ * gen_ctx_->gen_cfg.num_beams *
                          SizeofType(dtype_);

    if (DeviceType::CPU == backend) {
      memset(k_cache_buf_, 0, kvcache_size);
      memset(v_cache_buf_, 0, kvcache_size);
    }
#ifdef ENABLE_CUDA
    else if (DeviceType::CUDA == backend) {
      cudaStream_t cu_stream =
          static_cast<const CUDAContext*>(ctx_)->GetStream();
      cudaMemsetAsync(k_cache_buf_, 0, kvcache_size, cu_stream);
      cudaMemsetAsync(v_cache_buf_, 0, kvcache_size, cu_stream);
    }
#endif  // ENABLE_CUDA
    else {
      LOG(ERROR) << op_type_ << " Operator does not support "
                 << DeviceType_Name(backend) << " device type" << std::endl;
      return AsStatus::ALLSPARK_RUNTIME_ERROR;
    }
  }

  kernel_launcher(dtype_, tensor_map_->at(out_names_[0])->GetDataPtr(),
                  score_buf, q_buf, k_buf, v_buf, mask_buf, position_embedding,
                  k_cache_buf_, v_cache_buf_, q_array, k_array, v_array,
                  score_array, out_array, batch_size_, gen_ctx_->num_beams,
                  seq_len_, (gen_ctx_->step + 1), gen_ctx_->max_length,
                  hidden_size_, num_heads_, size_per_head_, gemm_batch_, alpha_,
                  input_len_, enable_logn_, xlogn_, ctx_);
  if (gen_ctx_->generate_method == 1 && gen_ctx_->only_decoder == true &&
      gen_ctx_->step == 0) {
    first_beam_ = true;
  }
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus DecOptMHAOp::ResetCache() {
  k_cache_ = std::make_unique<AsTensor>("k_cache", k_cache_->GetDeviceType(),
                                        k_cache_->GetDataType(),
                                        k_cache_->GetDataMode(), Shape{1});
  v_cache_ = std::make_unique<AsTensor>("v_cache", k_cache_->GetDeviceType(),
                                        k_cache_->GetDataType(),
                                        k_cache_->GetDataMode(), Shape{1});
  tmp_k_cache_ = std::make_unique<AsTensor>(
      "tmp_k_cache", k_cache_->GetDeviceType(), k_cache_->GetDataType(),
      k_cache_->GetDataMode(), Shape{1});
  tmp_v_cache_ = std::make_unique<AsTensor>(
      "tmp_v_cache", k_cache_->GetDeviceType(), k_cache_->GetDataType(),
      k_cache_->GetDataMode(), Shape{1});
  return AsStatus::ALLSPARK_SUCCESS;
}

// REGISTER_OP(DecOptMHA, CUDA, DecOptMHAOp)
// REGISTER_OP(DecOptMHA, CPU,  DecOptMHAOp)
}  // namespace allspark
