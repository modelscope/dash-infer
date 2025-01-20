/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    dec_opt_mha_i8cache_op.cpp
 */

#include <core/kernel/kernel.h>
#include <utility/datatype_dispatcher.h>

#include <cmath>
#ifdef ENABLE_CUDA
#include <cuda/cuda_context.h>
#endif
#include <cpu/cpu_context.h>

#include "dec_opt_mha_i8cache_op.h"  // NOLINT

namespace allspark {

#ifdef ENABLE_CUDA
int64_t gpu_dec_opt_mha_cache_context_wss(DataType dtype, int32_t batch,
                                          int32_t first, int32_t xseql,
                                          int32_t cache, int32_t nhead,
                                          int32_t phead) {
  // xseql is context sequence length.
  int64_t score_size = AsOperator::round32((int64_t)batch * nhead * xseql *
                                           xseql * SizeofType(dtype));
  int64_t array_size =
      AsOperator::round32((int64_t)batch * nhead) * 5 * sizeof(void*);
  return score_size + array_size;
}

template <typename QT>
void gpu_dec_opt_mha_cache_context(
    DataType dtype, const void* qkv, const float* mask,
    const float* position_embedding, QT* kc, float* kz, float* ks, QT* vc,
    float* vz, float* vs, char* workspace, void* context, int32_t batch,
    int32_t first, int32_t xseql, int32_t cache, int32_t nhead, int32_t phead,
    int32_t xlogn, float alpha, bool enable_logn, QuantType quant_type,
    const DeviceContext* ctx) {
  // first time, init kv-cache call this function.
  DLOG(INFO) << "gpu_dec_opt_mha_cache_context" << std::endl;
  // printf("gpu_dec_opt_mha_cache_context(batch=%d, first=%d, xseql=%d,
  // cache=%d, nhead=%d, phead=%d, xlogn=%d)\n",
  //         batch, first, xseql, cache, nhead, phead, xlogn);
  const CUDAContext* gpu_ctx = static_cast<const CUDAContext*>(ctx);
  cublasHandle_t cublas_handle = gpu_ctx->GetCublasHandle();
  cudaStream_t stream = gpu_ctx->GetStream();
  if (gpu_ctx->GetMatmulPrecision() == PrecisionLevel::HIGH &&
      dtype == FLOAT32) {
    cublasSetMathMode(cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH);
  }

  // qkv      [batch, xseql, 3, nhead, phead]
  // score    [batch, nhead,    xseql, xseql]
  // out      [batch, xseql,    nhead, phead]
  auto functor = [&]<typename T>() {
    T* score = (T*)workspace;
    char** q_arr = (char**)((char*)workspace +
                            AsOperator::round32((int64_t)batch * nhead * xseql *
                                                xseql * SizeofType(dtype)));
    char** k_arr = q_arr + AsOperator::round32(batch * nhead);
    char** v_arr = k_arr + AsOperator::round32(batch * nhead);
    char** s_arr = v_arr + AsOperator::round32(batch * nhead);
    char** o_arr = s_arr + AsOperator::round32(batch * nhead);

    // prepare.
    cuda::GetBatchArrayLauncher(
        (T*)qkv, (T*)qkv + 1 * nhead * phead, (T*)qkv + 2 * nhead * phead,
        (T*)score, (T*)context, (T**)q_arr, (T**)k_arr, (T**)v_arr, (T**)s_arr,
        (T**)o_arr, batch, 1, nhead, phead, xseql,
        xseql * 3 * nhead * phead,  // q-bidx-stride
        xseql * 3 * nhead * phead,  // kv-bidx-stride
        nhead * xseql * xseql,      // score-bidx-stride
        xseql * nhead * phead,      // out-bidx-stride
        stream);

    // batch gemm 1
    // [batch, xseql, 3, nhead, phead] x [batch, xseql, 3, nhead, phead] ->
    // [batch, xseql, nhead, xseql]
    //             m                k                n                k m n
    cuda::BatchGemmWraper<T>(
        (void**)s_arr, (void**)q_arr, (void**)k_arr, xseql, xseql, phead, false,
        true, alpha, 0.0f,  // m, n, k, ta, tb, alpha, beta,
        3 * nhead * phead, 3 * nhead * phead, nhead * xseql,
        batch * nhead,  // lda, ldb, ldc, batch
        cublas_handle);

    if (position_embedding) {
      cuda::BinaryKernelLauncher((T*)score, (T*)score, (T*)position_embedding,
                                 batch * nhead * xseql * xseql, 1, stream);
    }

    // score softmax [batch, nhead, xseql, xseql]
    if (enable_logn) {
      cuda::LognSoftmaxKernelLauncher((T*)score, mask, batch, nhead, xseql,
                                      xseql, xlogn, stream);
    } else {
      cuda::SoftmaxKernelLauncher((T*)score, mask, batch, 1, nhead, xseql,
                                  xseql, stream);
    }

    // batch gemm 2
    // [batch, xseql, nhead, xseql] x [batch, xseql, 3, nhead, phead] ->
    // [batch, xseql, nhead, phead]
    //            m              k                k                n m n
    cuda::BatchGemmWraper<T>((void**)o_arr, (void**)s_arr, (void**)v_arr, xseql,
                             phead, xseql, false, false, 1.0f,
                             0.0f,  // m, n, k, ta, tb, alpha, beta
                             nhead * xseql, 3 * nhead * phead, nhead * phead,
                             batch * nhead,  // lda, ldb, ldc, batch
                             cublas_handle);

    // update kv-cache.
    cuda::mha_quant_cache::load_and_quant_to_kv_cache_context(
        stream, (T*)qkv, kc, kz, ks, vc, vz, vs, batch, nhead, phead, cache,
        xseql, quant_type);
  };
  DispatchCUDA(dtype, functor);
}

int64_t gpu_dec_opt_mha_cache_decoder_wss(DataType dtype, int32_t batch,
                                          int32_t first, int32_t xseql,
                                          int32_t cache, int32_t nhead,
                                          int32_t phead) {
  // score [batch, nhead, xseql], max xseql = cache.
  int64_t score_max_size =
      AsOperator::round32((int64_t)batch * nhead * cache * SizeofType(dtype));
  return score_max_size;
}

template <typename QT>
void gpu_dec_opt_mha_cache_decoder(
    DataType dtype, const void* qkv, const float* mask,
    const float* position_embedding, QT* kc, float* kz, float* ks, QT* vc,
    float* vz, float* vs, char* workspace, void* context, int32_t batch,
    int32_t first, int32_t xseql, int32_t cache, int32_t nhead, int32_t phead,
    float alpha, QuantType quant_type, const DeviceContext* ctx) {
  // decoder phase call this function.
  DLOG(INFO) << "gpu_dec_opt_mha_cache_decoder" << std::endl;
  // printf("gpu_dec_opt_mha_cache_decoder(batch=%d, first=%d, xseql=%d,
  // cache=%d, nhead=%d, phead=%d)\n",
  //         batch, first, xseql, cache, nhead, phead);
  const CUDAContext* gpu_ctx = static_cast<const CUDAContext*>(ctx);
  cudaStream_t stream = gpu_ctx->GetStream();

  auto functor = [&]<typename T>() {
    T* score = (T*)workspace;
    /*  workspace usage:
     *  |<--qxk-->|
     */
    // stride load kv from qkv to cache
    // qkv              [batch, 3, nhead, phead]
    // data_cache       [batch, nhead, cache, phead]
    // param_cache      [batch, nhead, cache]
    cuda::mha_quant_cache::load_and_quant_to_kv_cache_decoder(
        stream, (T*)qkv, kc, kz, ks, vc, vz, vs, batch, nhead, phead, cache,
        xseql - 1, quant_type);

    // gemm-nt
    // qkv              [batch,     3, nhead, phead]
    // k data_cache     [batch, nhead, cache, phead] x
    // k param_cache    [batch, nhead, cache]        +
    // pos_embedding    [batch,     1, first, first]
    // qxk              [batch, nhead,     1, xseql]
    cuda::mha_quant_cache::score_gemv_w8_position_embedding(
        stream, (T*)qkv, mask, position_embedding, kc, kz, ks, score, alpha,
        batch, nhead, phead, cache, xseql, first, quant_type);

    // softmax
    // qxk              [batch, nhead, xseql]
    cuda::mha_quant_cache::inplace_softmax(stream, score, batch, nhead, xseql);

    // gemm nn
    // qxk              [batch, nhead,     1, xseql]
    // v data_cache     [batch, nhead, cache, phead]
    // v param_cache    [batch, nhead, cache]
    // output           [batch, nhead,     1, phead]
    cuda::mha_quant_cache::context_gemv_w8(stream, score, vc, vz, vs,
                                           (T*)context, batch, nhead, phead,
                                           cache, xseql, quant_type);
  };
  DispatchCUDA(dtype, functor);
}
#endif  // ENABLE_CUDA

AsStatus DecOptMHAI8CacheOp::Init(const OperatorProto& op_proto,
                                  const DeviceContext& ctx,
                                  const TensorMap& weights_map,
                                  TensorMap* tensor_map) {
  AS_CHECK_STATUS(AsOperator::Init(op_proto, ctx, weights_map, tensor_map));
  layer_num_ = get_layer_num(this->op_name_);
  if (layer_num_ < 0) {
    LOG(ERROR) << "DecOptMHAI8CacheOp : can't find layer_num_" << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
  if (ctx_->GetDeviceType() != DeviceType::CUDA) {
    LOG(ERROR) << op_type_ << " Operator does not support "
               << DeviceType_Name(ctx_->GetDeviceType()) << " device type"
               << std::endl;
    return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }

  // type inference
  dtype_ = tensor_map_->at(in_names_[0])->GetDataType();
  tensor_map_->at(out_names_[0])->SetDataType(dtype_);

  // attr
  auto& attr_map = op_proto.attr();
  if (attr_map.find("multigpu") != attr_map.end())
    multigpu_ = *(bool*)(attr_map.at("multigpu").c_str());
  if (attr_map.find("position_embedding") != attr_map.end())
    pos_embedding_exist_ = true;
  AS_CHECK_STATUS(nhead_from_attributes_(op_proto));       // nhead_
  AS_CHECK_STATUS(quant_type_from_attributes_(op_proto));  // quant_type_
  AS_CHECK_STATUS(alpha_from_attributes_(op_proto));  // alpha_, alpha_exist_
  AS_CHECK_STATUS(
      logn_from_attributes_(op_proto));  // logn_enable_, logn_embedding_length_

#ifdef ENABLE_CUDA
  if (multigpu_) nhead_ /= static_cast<const CUDAContext*>(ctx_)->GetNranks();
#endif  // ENABLE_CUDA

  DataType qtype =
      (quant_type_ == QuantType::INT8) ? DataType::INT8 : DataType::UINT8;
  kc_ = std::make_unique<AsTensor>("k_cache", ctx_->GetDeviceType(), qtype,
                                   DataMode::DENSE, Shape{1});
  vc_ = std::make_unique<AsTensor>("v_cache", ctx_->GetDeviceType(), qtype,
                                   DataMode::DENSE, Shape{1});
  kz_ =
      std::make_unique<AsTensor>("k_zero", ctx_->GetDeviceType(),
                                 DataType::FLOAT32, DataMode::DENSE, Shape{1});
  vz_ =
      std::make_unique<AsTensor>("v_zero", ctx_->GetDeviceType(),
                                 DataType::FLOAT32, DataMode::DENSE, Shape{1});
  ks_ =
      std::make_unique<AsTensor>("k_scale", ctx_->GetDeviceType(),
                                 DataType::FLOAT32, DataMode::DENSE, Shape{1});
  vs_ =
      std::make_unique<AsTensor>("v_scale", ctx_->GetDeviceType(),
                                 DataType::FLOAT32, DataMode::DENSE, Shape{1});
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus DecOptMHAI8CacheOp::Reshape(RuntimeContext* runtime_ctx) {
  // notice: this reshape will be called twice.
  if (ctx_->GetDeviceType() != DeviceType::CUDA) {
    LOG(ERROR) << op_type_ << " Operator does not support "
               << DeviceType_Name(ctx_->GetDeviceType()) << " device type"
               << std::endl;
    return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }

  // if (gen_ctx_->num_beams != 1) {
  //     LOG(ERROR) << op_type_ << " Operator does not support beams." <<
  //     std::endl; return AsStatus::ALLSPARK_RUNTIME_ERROR;
  // }

  const AsTensor* qkvtensor = tensor_map_->at(in_names_[0]).get();
  const Shape& qkvshape = qkvtensor->GetShape();
  bool is_decoder = !runtime_ctx->is_context;
  if (is_decoder) {
    // is decoder
    batch_ = qkvshape[0];
    phead_ = qkvshape[2] / 3 / nhead_;
    cache_ = ctx_->GetModelMaxLength();
    xseql_ = runtime_ctx->GetContextGenCtx()->step + 1;
  } else {
    // is context, self-attention.
    batch_ = qkvshape[0];
    phead_ = qkvshape[2] / 3 / nhead_;
    cache_ = ctx_->GetModelMaxLength();
    xseql_ = qkvshape[1];  // for loop-context, xseql is current batch, not
                           // max(xseql)
    first_ = qkvshape[1];
  }

  if (qkvshape[2] % (3 * nhead_)) {
    LOG(ERROR) << "Invalid hidden size in DecOptMHAI8CacheOp, hidden size = "
               << qkvshape[2] << ", current head num = " << nhead_
               << ", current head size = " << phead_ << std::endl;
    return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }
  if (alpha_exist_ == false) alpha_ = 1.0f / std::sqrt(phead_ * 1.0f);

  // workspace and out
  int64_t ws_size = 0;
  int64_t out_size = 0;

#ifdef ENABLE_CUDA
  if (is_decoder) {
    ws_size = gpu_dec_opt_mha_cache_decoder_wss(dtype_, batch_, first_, xseql_,
                                                cache_, nhead_, phead_);
    out_size =
        round32((int64_t)batch_ * 1 * nhead_ * phead_ * SizeofType(dtype_));
    tensor_map_->at("workspace")->SetShape(Shape({ws_size}));
    tensor_map_->at(out_names_[0])
        ->SetShape(Shape({batch_, 1, nhead_ * phead_}));
  } else {
    ws_size = gpu_dec_opt_mha_cache_context_wss(dtype_, batch_, first_,
                                                ctx_->GetModelMaxLength(),
                                                cache_, nhead_, phead_);
    out_size = round32((int64_t)batch_ * xseql_ * nhead_ * phead_ *
                       SizeofType(dtype_));
    tensor_map_->at("workspace")->SetShape(Shape({ws_size}));
    tensor_map_->at(out_names_[0])
        ->SetShape(Shape({batch_, xseql_, nhead_ * phead_}));
  }
#endif  // ENABLE_CUDA

  // kv-cache
  int32_t model_batch = 1;
  int64_t k_or_v_cache_size = 0;
  if (quant_type_ == QuantType::INT8) {
    k_or_v_cache_size =
        AsOperator::round32((int64_t)model_batch * nhead_ * cache_ * phead_ *
                            sizeof(int8_t));  // as i8
  } else if (quant_type_ == QuantType::UINT4) {
    k_or_v_cache_size =
        AsOperator::round32((int64_t)model_batch * nhead_ * cache_ * phead_ *
                            sizeof(uint8_t) / 2);  // 2xUINT4 pack into UINT8
  }
  int64_t k_or_v_param_size = AsOperator::round32(
      (int64_t)model_batch * nhead_ * cache_ * sizeof(float));  // zero or scale
  if (runtime_ctx->is_context) {
    // 初始化cachememory，每次多申请kv_size个token长度的cache,默认kv_size就是engine_max_length
    runtime_ctx->GetContextGenCtx()->k_cache_list.push_back(
        std::make_unique<CacheMemory>(ctx_->GetDeviceType(), k_or_v_cache_size,
                                      k_or_v_param_size));
    runtime_ctx->GetContextGenCtx()->v_cache_list.push_back(
        std::make_unique<CacheMemory>(ctx_->GetDeviceType(), k_or_v_cache_size,
                                      k_or_v_param_size));
    // kc_->SetShape(Shape({k_or_v_cache_size}));
  }

#ifdef ENABLE_CUDA
  const CUDAContext* gpu_ctx = static_cast<const CUDAContext*>(ctx_);
  cublasHandle_t cublas_handle = gpu_ctx->GetCublasHandle();
  cublasSetWorkspace(cublas_handle,
                     tensor_map_->at("cublas_workspace")->GetDataPtr(),
                     tensor_map_->at("cublas_workspace")->GetSizeInByte());
#endif
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus DecOptMHAI8CacheOp::RunContext(RuntimeContext* runtime_ctx) {
  if (batch_ != 1) {
    LOG(ERROR) << "BatchMHAOp only support multibatch in decoder pharse, "
                  "not context pharse."
               << std::endl;
    return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }

#ifdef ENABLE_CUDA
  std::shared_ptr<GenerateContext> gen_ctx = runtime_ctx->GetContextGenCtx();
  void* qkv = tensor_map_->at(in_names_[0])->GetDataPtr();
  void* mask =
      mask_exist_ ? tensor_map_->at(in_names_[1])->GetDataPtr() : nullptr;
  void* pos_embed = nullptr;
  void* workspace = tensor_map_->at("workspace")->GetDataPtr();
  void* context = tensor_map_->at(out_names_[0])->GetDataPtr();

  const AsTensor* qkvtensor = tensor_map_->at(in_names_[0]).get();
  const Shape& qkvshape = qkvtensor->GetShape();
  int batch_cache_offset = 0;
  int batch_param_offset = 0;

  first_ = gen_ctx->input_len;
  kc_ptr_ = gen_ctx->k_cache_list[layer_num_]->GetData();
  kz_ptr_ = (float*)gen_ctx->k_cache_list[layer_num_]->GetZero();
  ks_ptr_ = (float*)gen_ctx->k_cache_list[layer_num_]->GetScale();
  vc_ptr_ = gen_ctx->v_cache_list[layer_num_]->GetData();
  vz_ptr_ = (float*)gen_ctx->v_cache_list[layer_num_]->GetZero();
  vs_ptr_ = (float*)gen_ctx->v_cache_list[layer_num_]->GetScale();
  if (quant_type_ == QuantType::INT8) {
    gpu_dec_opt_mha_cache_context<int8_t>(
        dtype_, qkv, (float*)mask, (float*)pos_embed,
        (int8_t*)kc_ptr_ + batch_cache_offset, kz_ptr_ + batch_param_offset,
        ks_ptr_ + batch_param_offset, (int8_t*)vc_ptr_ + batch_cache_offset,
        vz_ptr_ + batch_param_offset, vs_ptr_ + batch_param_offset,
        (char*)workspace, context, 1, first_, xseql_, cache_, nhead_, phead_,
        xlogn_, alpha_, enable_logn_, quant_type_, ctx_);
  } else if (quant_type_ == QuantType::UINT4) {
    gpu_dec_opt_mha_cache_context<uint8_t>(
        dtype_, qkv, (float*)mask, (float*)pos_embed,
        (uint8_t*)kc_ptr_ + batch_cache_offset / 2,
        kz_ptr_ + batch_param_offset, ks_ptr_ + batch_param_offset,
        (uint8_t*)vc_ptr_ + batch_cache_offset / 2,
        vz_ptr_ + batch_param_offset, vs_ptr_ + batch_param_offset,
        (char*)workspace, context, 1, first_, xseql_, cache_, nhead_, phead_,
        xlogn_, alpha_, enable_logn_, quant_type_, ctx_);
  }
  return AsStatus::ALLSPARK_SUCCESS;
#endif
  LOG(ERROR) << "no valid impl found for dec-opt-mha i8-cache";
  return AsStatus::ALLSPARK_RUNTIME_ERROR;
}
AsStatus DecOptMHAI8CacheOp::RunDecoder(RuntimeContext* runtime_ctx) {
  // printf("[i8mha] forward-enable, current batch = %d, phead = %d, nhead =
  // %d, cache = %d, xseql = %d, first = %d, ctx-in = %d, ctx-batch = %d,
  // ctx-step = %d\n",
  //     batch_, phead_, nhead_, cache_, xseql_, first_, gen_ctx_->input_len,
  //     gen_ctx_->current_batch, gen_ctx_->step);
#ifdef ENABLE_CUDA
  for (int batch = 0; batch < batch_; batch++) {
    int current_batch = batch;
    std::shared_ptr<GenerateContext> gen_ctx = runtime_ctx->GetGenCtx(batch);
    void* qkv = tensor_map_->at(in_names_[0])->GetDataPtr();
    void* mask = nullptr;
    // TODO(ZHANG YUFEI): add this back for all situation. currently we
    // remove this cause loop context not given any possible solution to
    // describe this shape. void* pos_embed = pos_embedding_exist_ ?
    // tensor_map_->at(in_names_[2])->GetDataPtr() : nullptr;
    void* pos_embed = nullptr;
    void* workspace = tensor_map_->at("workspace")->GetDataPtr();
    void* context = tensor_map_->at(out_names_[0])->GetDataPtr();

    const AsTensor* qkvtensor = tensor_map_->at(in_names_[0]).get();
    const Shape& qkvshape = qkvtensor->GetShape();
    xseql_ = gen_ctx->step + 1;
    first_ = gen_ctx->input_len;
    kc_ptr_ = gen_ctx->k_cache_list[layer_num_]->GetData();
    kz_ptr_ = (float*)gen_ctx->k_cache_list[layer_num_]->GetZero();
    ks_ptr_ = (float*)gen_ctx->k_cache_list[layer_num_]->GetScale();
    vc_ptr_ = gen_ctx->v_cache_list[layer_num_]->GetData();
    vz_ptr_ = (float*)gen_ctx->v_cache_list[layer_num_]->GetZero();
    vs_ptr_ = (float*)gen_ctx->v_cache_list[layer_num_]->GetScale();
    // decoder logn logic.
    float scale = enable_logn_ && xseql_ > xlogn_
                      ? alpha_ * logf(xseql_) / logf(xlogn_)
                      : alpha_;
    int offset = nhead_ * phead_ * SizeofType(dtype_);
    context = (char*)context + current_batch * 1 * offset;
    qkv = (char*)qkv + current_batch * 1 * offset * 3;
    if (quant_type_ == QuantType::INT8) {
      gpu_dec_opt_mha_cache_decoder<int8_t>(
          dtype_, qkv, (float*)mask, (float*)pos_embed, (int8_t*)kc_ptr_,
          kz_ptr_, ks_ptr_, (int8_t*)vc_ptr_, vz_ptr_, vs_ptr_,
          (char*)workspace, context, 1, first_, xseql_, cache_, nhead_, phead_,
          scale, quant_type_, ctx_);
    } else if (quant_type_ == QuantType::UINT4) {
      gpu_dec_opt_mha_cache_decoder<uint8_t>(
          dtype_, qkv, (float*)mask, (float*)pos_embed, (uint8_t*)kc_ptr_,
          kz_ptr_, ks_ptr_, (uint8_t*)vc_ptr_, vz_ptr_, vs_ptr_,
          (char*)workspace, context, 1, first_, xseql_, cache_, nhead_, phead_,
          scale, quant_type_, ctx_);
    }
  }
  return AsStatus::ALLSPARK_SUCCESS;
#endif
  LOG(ERROR) << "no valid impl found for dec-opt-mha i8-cache";
  return AsStatus::ALLSPARK_RUNTIME_ERROR;
}
AsStatus DecOptMHAI8CacheOp::Forward(RuntimeContext* runtime_ctx) {
  AsStatus status;
  if (runtime_ctx->is_context) {
    status = RunContext(runtime_ctx);
  } else {
    status = RunDecoder(runtime_ctx);
  }
  return status;
}

REGISTER_OP(DecOptMHAI8Cache, CUDA, DecOptMHAI8CacheOp)
// REGISTER_OP(DecOptMHAI8Cache, CPU, DecOptMHAI8CacheOp)
}  // namespace allspark
