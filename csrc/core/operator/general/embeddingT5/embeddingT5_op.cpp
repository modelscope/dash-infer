/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    embeddingT5_op.cpp
 */

#include "embeddingT5_op.h"  // NOLINT

#include <core/kernel/kernel.h>
#include <utility/datatype_dispatcher.h>

#include <utility>
#ifdef ENABLE_CUDA
#include <cuda/cuda_context.h>
#endif
#include <cpu/cpu_context.h>

#include <string>
namespace allspark {

#ifdef ENABLE_CUDA
AsStatus gpu_embedding(DataType dtype, void* out, void* in_ids,
                       const void* embedding_table, int batch_size, int seq_len,
                       int hidden_size, int vocab_size,
                       const DeviceContext* ctx) {
  const CUDAContext* gpu_ctx = static_cast<const CUDAContext*>(ctx);
  auto functor = [&]<typename T>() {
    T* typed_out = static_cast<T*>(out);
    const int64_t* typed_in_ids = static_cast<const int64_t*>(in_ids);
    const T* typed_embedding_table = static_cast<const T*>(embedding_table);
    cuda::EmbeddingT5KernelLauncher<false, T>(
        typed_out, typed_in_ids, typed_embedding_table, batch_size, seq_len,
        hidden_size, vocab_size, gpu_ctx->GetStream());
  };
  DispatchCUDA(dtype, functor);
  return AsStatus::ALLSPARK_SUCCESS;
}
#endif

AsStatus cpu_embedding(DataType dtype, void* out, void* in_ids,
                       const void* embedding_table, int batch_size, int seq_len,
                       int hidden_size, int vocab_size,
                       const DeviceContext* ctx) {
  DLOG(INFO) << "cpu_embedding" << std::endl;
  auto functor = [&]<typename T>() {
    T* typed_out = static_cast<T*>(out);
    const int64_t* typed_in_ids = static_cast<const int64_t*>(in_ids);
    const T* typed_embedding_table = static_cast<const T*>(embedding_table);
    cpu::EmbeddingT5KernelLauncher(typed_out, typed_in_ids,
                                   typed_embedding_table, batch_size, seq_len,
                                   hidden_size, vocab_size, false);
  };
  DispatchCPU(dtype, functor);
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus EmbeddingT5Op::Init(const OperatorProto& op_proto,
                             const DeviceContext& ctx,
                             const TensorMap& weights_map,
                             TensorMap* tensor_map) {
  AS_CHECK_STATUS(AsOperator::Init(op_proto, ctx, weights_map, tensor_map));
  // check weight
  if (weights_.size() != 1) {
    LOG(ERROR) << "EmbeddingT5Op has 1 weights [word_embedding_table], "
                  "[pos_embedding_table], [token_embedding_table](optional)"
               << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
  hidden_size_ = weights_[0]->GetShape()[1];
  for (int i = 1; i < weights_.size(); ++i) {
    if (weights_[i]->GetShape()[1] != hidden_size_) {
      LOG(ERROR) << "EmbeddingT5Op : Invalid weight shape." << std::endl;
      return AsStatus::ALLSPARK_PARAM_ERROR;
    }
  }
  vocab_size_ = weights_[0]->GetShape()[0];
  // type inference
  DataType dtype = weights_[0]->GetDataType();
  tensor_map_->at(out_names_[0])->SetDataType(dtype);

  // kernel choose
  DeviceType backend = ctx.GetDeviceType();

  in_ids_ = std::make_unique<AsTensor>("in_ids", backend, DataType::INT64,
                                       DataMode::DENSE,
                                       Shape{ctx_->GetModelMaxBatch()});
  switch (backend) {
#ifdef ENABLE_CUDA
    case DeviceType::CUDA: {
      kernel_launcher = gpu_embedding;
      break;
    }
#endif
    case DeviceType::CPU: {
      kernel_launcher = cpu_embedding;
      break;
    }
    default:
      LOG(ERROR) << "Embedding Operator does not support "
                 << DeviceType_Name(backend) << " device type" << std::endl;
      return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus EmbeddingT5Op::Reshape(RuntimeContext* runtime_ctx) {
  if (runtime_ctx->is_context) {
    batch_size_ = 1;
    seq_len_ = runtime_ctx->GetContextGenCtx()
                   ->request->interim.at("new_input_ids")
                   ->GetShape()[1];
  } else {
    batch_size_ = runtime_ctx->GetGenCtxListSize();
    seq_len_ = 1;
  }

  Shape out_shape = Shape({batch_size_, seq_len_, hidden_size_});
  AS_CHECK_STATUS(
      tensor_map_->at(out_names_[0])->SetShape(std::move(out_shape)));
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus EmbeddingT5Op::Forward(RuntimeContext* runtime_ctx) {
  void* in_ids = nullptr;

  DeviceType backend = ctx_->GetDeviceType();
  switch (backend) {
#ifdef ENABLE_CUDA
    case DeviceType::CUDA: {
      if (runtime_ctx->is_context) {
        GenerateContext* gen_ctx = runtime_ctx->GetContextGenCtx();
        in_ids =
            gen_ctx->request->interim.at("new_input_ids_gpu")->GetDataPtr();
      } else {
        std::vector<int64_t> new_tokens(batch_size_);
        for (int i = 0; i < batch_size_; i++) {
          GenerateContext* gen_ctx = runtime_ctx->GetGenCtx(i);
          std::shared_ptr<AsTensor> generated_ids_tensor =
              gen_ctx->request->interim.at("generated_ids");
          new_tokens[i] =
              *(static_cast<int64_t*>(generated_ids_tensor->GetDataPtr()) +
                generated_ids_tensor->GetShape()[1] - 1);
        }
        cudaStream_t stream =
            static_cast<const CUDAContext*>(ctx_)->GetStream();
        AS_CHECK_CUDA(cudaMemcpyAsync(in_ids_->GetDataPtr(), new_tokens.data(),
                                      batch_size_ * sizeof(int64_t),
                                      cudaMemcpyHostToDevice, stream));

        in_ids = in_ids_->GetDataPtr();
      }
      break;
    }
#endif
    case DeviceType::CPU: {
      if (runtime_ctx->is_context) {
        GenerateContext* gen_ctx = runtime_ctx->GetContextGenCtx();
        in_ids = gen_ctx->request->interim.at("new_input_ids")->GetDataPtr();
      } else {
        int64_t* ptr = static_cast<int64_t*>(in_ids_->GetDataPtr());
        for (int i = 0; i < batch_size_; i++) {
          GenerateContext* gen_ctx = runtime_ctx->GetGenCtx(i);
          std::shared_ptr<AsTensor> generated_ids_tensor =
              gen_ctx->request->interim.at("generated_ids");
          ptr[i] = *(static_cast<int64_t*>(generated_ids_tensor->GetDataPtr()) +
                     generated_ids_tensor->GetShape()[1] - 1);
        }
        in_ids = in_ids_->GetDataPtr();
      }
      break;
    }
    default:
      LOG(ERROR) << op_type_ << " Operator does not support "
                 << DeviceType_Name(backend) << " device type" << std::endl;
      return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }

  void* out = tensor_map_->at(out_names_[0])->GetDataPtr();
  kernel_launcher(weights_[0]->GetDataType(), out, in_ids,
                  weights_[0]->GetDataPtr(), batch_size_, seq_len_,
                  hidden_size_, vocab_size_, ctx_);
  // PrintInformation();
  return AsStatus::ALLSPARK_SUCCESS;
}

REGISTER_OP(EmbeddingT5, CUDA, EmbeddingT5Op)
REGISTER_OP(EmbeddingT5, CPU, EmbeddingT5Op)
}  // namespace allspark
