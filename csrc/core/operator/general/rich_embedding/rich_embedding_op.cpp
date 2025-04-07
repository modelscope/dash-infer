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
  DeviceType backend = ctx.GetDeviceType();
  reply_part_ = std::make_unique<AsTensor>(
      "reply_part_", backend, ctx_->GetDtype(), DataMode::DENSE, Shape{0});
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus RichEmbeddingOp::Reshape(RuntimeContext* runtime_ctx) {
  Shape in_shape = tensor_map_->at(in_names_[1])->GetShape();
  batch_size_ = in_shape[0];
  seq_len_ = in_shape[1];
  hidden_size_ = in_shape[2];
#ifdef ENABLE_CUDA
  if (ctx_->GetDeviceType() == DeviceType::CUDA) {
    embedding_device_->SetShape(
        Shape{ctx_->GetModelMaxLength() * hidden_size_});
  }
#endif
  reply_part_->SetShape(Shape{ctx_->GetModelMaxLength(), hidden_size_});
  reply_part_->SetShape(Shape{seq_len_, hidden_size_});
  return AsStatus::ALLSPARK_SUCCESS;
}
void RichEmbeddingOp::UpdateInputsEmbedding(RuntimeContext* runtime_ctx,
                                            AsTensor* out_tensor) {
  if (rank_info_.rank_id == 0) {
    std::string tensor_name = "inputs_embedding";
    if (runtime_ctx->is_context) {
      GenerateContext* gen_ctx = runtime_ctx->GetContextGenCtx();
      if (gen_ctx->gen_cfg.enable_tensors_from_model_inference) {
        std::shared_ptr<AsTensor> output_tensor = std::make_shared<AsTensor>(
            tensor_name, DeviceType::CPU, out_tensor->GetDataType(),
            DataMode::DENSE, Shape({seq_len_, hidden_size_}));
        int data_size = SizeofType(output_tensor->GetDataType());
        CopyData(output_tensor->GetDataPtr(), output_tensor->GetDeviceType(),
                 out_tensor->GetDataPtr(), out_tensor->GetDeviceType(),
                 seq_len_ * hidden_size_ * data_size, ctx_);
        gen_ctx->request->tensors_from_model_inference_list[tensor_name]
            .push_back(output_tensor);
      }
    } else {
      int batch_size = runtime_ctx->GetGenCtxListSize();
      for (int i = 0; i < batch_size; i++) {
        GenerateContext* gen_ctx = runtime_ctx->GetGenCtx(i);
        if (gen_ctx->gen_cfg.enable_tensors_from_model_inference) {
          std::shared_ptr<AsTensor> output_tensor = std::make_shared<AsTensor>(
              tensor_name, DeviceType::CPU, out_tensor->GetDataType(),
              DataMode::DENSE, Shape({seq_len_, hidden_size_}));
          int data_size = SizeofType(output_tensor->GetDataType());
          CopyData(output_tensor->GetDataPtr(), output_tensor->GetDeviceType(),
                   out_tensor->GetDataPtr() + i * hidden_size_ * data_size,
                   out_tensor->GetDeviceType(), hidden_size_ * data_size, ctx_);
          gen_ctx->request->tensors_from_model_inference_list[tensor_name]
              .push_back(output_tensor);
        }
      }
    }
  }
}
AsStatus RichEmbeddingOp::RunContext(RuntimeContext* runtime_ctx) {
  GenerateContext* gen_ctx = runtime_ctx->GetContextGenCtx();
  TensorListMap extra_embedding = gen_ctx->request->extra_embedding;
  if (extra_embedding.empty()) {
    return AsStatus::ALLSPARK_SUCCESS;
  }
  if (batch_size_ != 1) {
    LOG(ERROR) << "RichEmbeddingOp only support single batch in context pharse."
               << std::endl;
    return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }

  DeviceType backend = ctx_->GetDeviceType();
  void* out = tensor_map_->at(out_names_[0])->GetDataPtr();
  DataType dtype = tensor_map_->at(in_names_[1])->GetDataType();
  int word_size = SizeofType(dtype);
  int64_t* input_ids_host_ptr =
      (int64_t*)gen_ctx->request->inputs.at("input_ids")->GetDataPtr();
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
  // reply_part
  if (extra_embedding.count("reply_part") == 0) {
    return AsStatus::ALLSPARK_SUCCESS;
  }
  AsTensor* reply_part_host = extra_embedding["reply_part"][0].get();
  int reply_size = reply_part_host->GetShape()[0];
  int reply_bias = 0;
  int data_size = SizeofType(reply_part_host->GetDataType());
  if (backend == DeviceType::CUDA) {
#ifdef ENABLE_CUDA
    AS_CHECK_CUDA(cudaMemcpyAsync(
        (char*)reply_part_->GetDataPtr(),
        (char*)reply_part_host->GetDataPtr() +
            reply_bias * hidden_size_ * data_size,
        seq_len_ * hidden_size_ * data_size, cudaMemcpyHostToDevice,
        static_cast<const CUDAContext*>(ctx_)->GetStream()));
    // AS_CHECK_CUDA(
    //     cudaMemsetAsync((char*)tensor_map_->at(out_names_[0])->GetDataPtr(),
    //     0,
    //                     seq_len_ * hidden_size_ * data_size,
    //                     static_cast<const CUDAContext*>(ctx_)->GetStream()));
    const CUDAContext* gpu_ctx = static_cast<const CUDAContext*>(ctx_);
    auto functor = [&]<typename T>() {
      T* typed_out =
          static_cast<T*>(tensor_map_->at(out_names_[0])->GetDataPtr());
      const T* typed_in1 =
          static_cast<const T*>(tensor_map_->at(out_names_[0])->GetDataPtr());
      const T* typed_in2 = static_cast<const T*>(reply_part_->GetDataPtr());
      int64_t count = seq_len_ * hidden_size_;
      cuda::BinaryKernelLauncher(typed_out, typed_in1, typed_in2, count,
                                 BinaryType::ADD, gpu_ctx->GetStream());
    };
    DispatchCUDA(tensor_map_->at(out_names_[0])->GetDataType(), functor);
#endif
  } else if (backend == DeviceType::CPU) {
    // TODO
    // memcpy(reply_part_->GetDataPtr()->GetDataPtr(),
    //        reply_part_host->GetDataPtr() +
    //            data_size * reply_bias seq_len_ * data_size);
  }
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus RichEmbeddingOp::RunDecoder(RuntimeContext* runtime_ctx) {
  DeviceType backend = ctx_->GetDeviceType();
  int batch_size = runtime_ctx->GetGenCtxListSize();
  if (seq_len_ != 1) {
    LOG(ERROR)
        << "RichEmbeddingOp only support single seq_len_ in decoder pharse."
        << std::endl;
    return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }
  for (int i = 0; i < batch_size; i++) {
    GenerateContext* gen_ctx = runtime_ctx->GetGenCtx(i);
    TensorListMap extra_embedding = gen_ctx->request->extra_embedding;
    if (extra_embedding.count("reply_part") == 0) {
      continue;
    }
    AsTensor* reply_part_host = extra_embedding["reply_part"][0].get();
    int reply_size = reply_part_host->GetShape()[0];
    int data_size = SizeofType(reply_part_host->GetDataType());
    if (backend == DeviceType::CUDA) {
#ifdef ENABLE_CUDA
      if (gen_ctx->step >= reply_size) {
        // out thinker range, always use last hidden
        AS_CHECK_CUDA(cudaMemcpyAsync(
            (char*)reply_part_->GetDataPtr(),
            (char*)reply_part_host->GetDataPtr() +
                (reply_size - 1) * hidden_size_ * data_size,
            seq_len_ * hidden_size_ * data_size, cudaMemcpyHostToDevice,
            static_cast<const CUDAContext*>(ctx_)->GetStream()));
      } else {
        AS_CHECK_CUDA(cudaMemcpyAsync(
            (char*)reply_part_->GetDataPtr(),
            (char*)reply_part_host->GetDataPtr() +
                gen_ctx->step * hidden_size_ * data_size,
            seq_len_ * hidden_size_ * data_size, cudaMemcpyHostToDevice,
            static_cast<const CUDAContext*>(ctx_)->GetStream()));
      }
      const CUDAContext* gpu_ctx = static_cast<const CUDAContext*>(ctx_);
      auto functor = [&]<typename T>() {
        T* typed_out =
            static_cast<T*>(tensor_map_->at(out_names_[0])->GetDataPtr() +
                            i * seq_len_ * hidden_size_ * data_size);
        const T* typed_in1 =
            static_cast<const T*>(tensor_map_->at(out_names_[0])->GetDataPtr() +
                                  i * seq_len_ * hidden_size_ * data_size);
        const T* typed_in2 = static_cast<const T*>(reply_part_->GetDataPtr());
        int64_t count = seq_len_ * hidden_size_;
        cuda::BinaryKernelLauncher(typed_out, typed_in1, typed_in2, count,
                                   BinaryType::ADD, gpu_ctx->GetStream());
      };
      DispatchCUDA(tensor_map_->at(out_names_[0])->GetDataType(), functor);
#endif
    } else if (backend == DeviceType::CPU) {
      // TODO
      // memcpy(reply_part_->GetDataPtr()->GetDataPtr(),
      //        reply_part_host->GetDataPtr() +
      //            data_size * reply_bias seq_len_ * data_size);
    }
  }
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus RichEmbeddingOp::Forward(RuntimeContext* runtime_ctx) {
  AsStatus status;
  if (runtime_ctx->is_context) {
    status = RunContext(runtime_ctx);
  } else {
    status = RunDecoder(runtime_ctx);
  }
  UpdateInputsEmbedding(runtime_ctx, tensor_map_->at(out_names_[0]).get());
  return status;
}

REGISTER_OP(RichEmbedding, CUDA, RichEmbeddingOp)
REGISTER_OP(RichEmbedding, CPU, RichEmbeddingOp)
}  // namespace allspark
