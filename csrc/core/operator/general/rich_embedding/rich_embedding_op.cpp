/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    rich_embedding_op.cpp
 */

#include "rich_embedding_op.h"  // NOLINT

#include <core/kernel/kernel.h>
#include <utility/check_cuda.h>
#include <utility/datatype_dispatcher.h>

#include <utility>

#include "common/extra_embedding.hpp"
#ifdef ENABLE_CUDA
#include <cuda/cuda_context.h>
#endif
#include <cpu/cpu_context.h>

#include <string>
namespace allspark {

AsStatus RichEmbeddingOp::Init(const OperatorProto& op_proto,
                               const DeviceContext& ctx,
                               const TensorMap& weights_map,
                               TensorMap* tensor_map) {
  AS_CHECK_STATUS(AsOperator::Init(op_proto, ctx, weights_map, tensor_map));
  int32_t flags = static_cast<int32_t>(AsTensorFlags::empty_flag);
  flags |= static_cast<int32_t>(AsTensorFlags::cuda_pinned_mem);
#ifdef ENABLE_CUDA
  if (ctx_->GetDeviceType() == DeviceType::CUDA) {
    embedding_device_ = std::make_unique<AsTensor>(
        "embedding_device_", DeviceType::CUDA, DataType::FLOAT32,
        DataMode::DENSE, Shape{0});
  }
#endif
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus RichEmbeddingOp::Reshape(RuntimeContext* runtime_ctx) {
  Shape in_shape = tensor_map_->at(in_names_[1])->GetShape();
  batch_size_ = in_shape[0];
  hidden_size_ = in_shape[2];
#ifdef ENABLE_CUDA
  if (ctx_->GetDeviceType() == DeviceType::CUDA) {
    embedding_device_->SetShape(
        Shape{ctx_->GetModelMaxLength() * hidden_size_});
  }
#endif
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus RichEmbeddingOp::Forward(RuntimeContext* runtime_ctx) {
  if (!runtime_ctx->is_context) {
    return AsStatus::ALLSPARK_SUCCESS;
  }
  std::shared_ptr<GenerateContext> gen_ctx = runtime_ctx->GetContextGenCtx();
  TensorListMap extra_embedding = gen_ctx->request->extra_embedding;
  if (extra_embedding.empty()) {
    return AsStatus::ALLSPARK_SUCCESS;
  }
  if (batch_size_ != 1) {
    LOG(ERROR) << "RichEmbeddingOp only support single batch in context pharse."
               << std::endl;
    return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }

  void* out = tensor_map_->at(out_names_[0])->GetDataPtr();
  DataType dtype = tensor_map_->at(in_names_[1])->GetDataType();
  int word_size = SizeofType(dtype);
  int64_t* input_ids_host_ptr =
      (int64_t*)gen_ctx->request->inputs.at("input_ids")->GetDataPtr();
  DeviceType backend = ctx_->GetDeviceType();
  ctx_->Synchronize();

  int seq_len = gen_ctx->request->inputs.at("input_ids")->GetShape()[1];
  auto reinfo_vec = std::make_shared<ExtraEmbeddingUtils::REInfoList>();
  AS_CHECK_STATUS(ExtraEmbeddingUtils::ParseExtraEmbedding(
      extra_embedding, input_ids_host_ptr, seq_len, reinfo_vec));

  AS_CHECK_STATUS(
      ExtraEmbeddingUtils::UpdateREInfo(reinfo_vec, gen_ctx->prefix_len));
  for (const auto& reinfo : *reinfo_vec) {
    if (reinfo.start_pos < 0) {
      continue;
    }

    switch (backend) {
#ifdef ENABLE_CUDA
      case DeviceType::CUDA: {
        // ctx_->Synchronize();
        // assume input type is float32.
        AS_CHECK_CUDA(cudaMemcpyAsync(
            (char*)embedding_device_->GetDataPtr(),
            (char*)reinfo.embedding->GetDataPtr(),
            reinfo.place_holder_cnt * hidden_size_ * sizeof(float),
            cudaMemcpyHostToDevice,
            static_cast<const CUDAContext*>(ctx_)->GetStream()));
        auto functor = [&]<typename T>() {
          cuda::ReplaceEmbedding<T>(
              ((T*)out) + reinfo.start_pos * hidden_size_,
              ((float*)embedding_device_->GetDataPtr()) +
                  reinfo.offset * hidden_size_,
              (reinfo.place_holder_cnt - reinfo.offset) * hidden_size_,
              static_cast<const CUDAContext*>(ctx_)->GetStream());
        };
        DispatchCUDA(dtype, functor);
        break;
      }
#endif
      case DeviceType::CPU: {
        if (dtype != DataType::FLOAT32) {
          LOG(ERROR) << "CPU only support FLOAT32";
          return AsStatus::ALLSPARK_RUNTIME_ERROR;
        }
        memcpy((char*)out + reinfo.start_pos * hidden_size_ * word_size,
               (char*)reinfo.embedding->GetDataPtr() +
                   reinfo.offset * hidden_size_,
               (reinfo.place_holder_cnt - reinfo.offset) * hidden_size_ *
                   word_size);
        break;
      }
    }
  }
  return AsStatus::ALLSPARK_SUCCESS;
}

REGISTER_OP(RichEmbedding, CUDA, RichEmbeddingOp)
REGISTER_OP(RichEmbedding, CPU, RichEmbeddingOp)
}  // namespace allspark
