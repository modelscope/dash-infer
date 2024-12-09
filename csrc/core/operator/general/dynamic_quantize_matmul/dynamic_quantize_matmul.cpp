/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    dynamic_quantize_matmul.cpp
 */

#ifdef ENABLE_CUDA
#include "dynamic_quantize_matmul.h"  // NOLINT

#include <core/kernel/kernel.h>
#include <utility/datatype_dispatcher.h>
#ifdef ENABLE_CUDA
#include <cuda/cuda_context.h>
#endif
#include <cpu/cpu_context.h>
using dnnl::memory;
namespace allspark {
AsStatus DynamicQuantizeMatmulOp::Init(const OperatorProto& op_proto,
                                       const DeviceContext& ctx,
                                       const TensorMap& weights_map,
                                       TensorMap* tensor_map) {
  AS_CHECK_STATUS(AsOperator::Init(op_proto, ctx, weights_map, tensor_map));
  // type inference
  dtype_ = tensor_map_->at(in_names_[0])->GetDataType();
  tensor_map_->at(out_names_[0])->SetDataType(dtype_);
  // check weight
  if (weights_.size() != 5 && weights_.size() != 4) {
    LOG(ERROR) << "DynamicQuantizeMatmulOp has 4~5 weights: [weight], "
                  "(optional) [bias]."
               << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
  // attr
  auto& attr_map = op_proto.attr();
  if (attr_map.find("transB") != attr_map.end()) {
    transB_ = *(bool*)(attr_map.at("transB").c_str());
  }
  if (attr_map.find("is_pooler") != attr_map.end()) {
    is_pooler_ = *(bool*)(attr_map.at("is_pooler").c_str());
  }
  if (attr_map.find("activation") != attr_map.end()) {
    activation_ = *(UnaryType*)(attr_map.at("activation").c_str());
  }
  if (attr_map.find("alpha") != attr_map.end()) {
    alpha_ = *(float*)(attr_map.at("alpha").c_str());
  }
  if (weights_.size() > 4) {
    with_bias_ = true;
  }
  // set k_, n_, batch
  const Shape& w_shape = weights_[0]->GetShape();
  int ndims_w = w_shape.Size();
  if (ndims_w < 2) {
    LOG(ERROR) << "DynamicQuantizeMatmulOp : Invalid weight shape."
               << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
  k_ = transB_ ? w_shape[ndims_w - 1] : w_shape[ndims_w - 2];
  n_ = transB_ ? w_shape[ndims_w - 2] : w_shape[ndims_w - 1];
  batch_ = w_shape.Count(0, ndims_w - 2);
  lda_ = k_;
  ldb_ = transB_ ? k_ : n_;
  ldc_ = n_;
  DeviceType backend = ctx.GetDeviceType();
  elemwiseA = std::make_unique<AsTensor>("elemwiseA", backend, DataType::INT32,
                                         DataMode::DENSE, Shape{0});
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus DynamicQuantizeMatmulOp::Reshape() {
  const Shape& x_shape = tensor_map_->at(in_names_[0])->GetShape();
  int x_ndims = x_shape.Size();
  Shape y_shape;
  if (is_pooler_) {
    if (x_shape.Size() != 3) {
      return AsStatus::ALLSPARK_RUNTIME_ERROR;
    }
    m_ = x_shape[0];
    lda_ = k_ * x_shape[1];
    y_shape = Shape({m_, n_});
  } else {
    m_ = x_shape.Count(0, x_ndims - 1);
    if (batch_ != 1) {
      y_shape.Append(batch_);
    }
    for (int i = 0; i < x_ndims - 1; ++i) {
      y_shape.Append(x_shape[i]);
    }
    y_shape.Append(n_);
  }
  tensor_map_->at(out_names_[0])->SetShape(std::move(y_shape));
  elemwiseA->SetShape(std::move(y_shape));
  elemwiseA_ndims_ = 2;
  elemwiseB_ndims_ = 2;
  elemwiseA_dims_[0] = m_;
  elemwiseA_dims_[1] = n_;
  elemwiseB_dims_[0] = m_;
  elemwiseB_dims_[1] = n_;
  output_dims_[0] = m_;
  output_dims_[1] = n_;
  if (with_bias_) {
    for (int i = 0; i < weights_[4]->GetShape().Size(); ++i) {
      bias_dims_[i] = weights_[4]->GetShape()[i];
    }
  }
  Shape lhs_shape = tensor_map_->at(in_names_[0])->GetShape();
  lhs_cnt_ = lhs_shape.Count();
  lhs_reduce_cnt_ = lhs_cnt_;
  lhs_reduce_cnt_ /= (per_channel_ & 0x2) ? (k_) : (k_ * m_);
  Shape rhs_shape = weights_[0]->GetShape();
  rhs_cnt_ = rhs_shape.Count();
  rhs_reduce_cnt_ = rhs_cnt_;
  rhs_reduce_cnt_ /= (per_channel_ & 0x1) ? (k_) : (k_ * n_);
  lhs_reduce_dims_[0] = per_channel_ & 0x2 ? m_ : 1;
  lhs_reduce_dims_[1] = 1;
  rhs_reduce_dims_[0] = 1;
  rhs_reduce_dims_[1] = per_channel_ & 0x1 ? n_ : 1;
  dim_t ws_size = 0;
  // Lhs isn`t weight and datatype is float.
  // That need workspace to store qdata, scale, zero and redsum.
  if (tensor_map_->at(in_names_[0])->GetDataType() != DataType::INT8) {
    ws_size += aligne_size(lhs_cnt_) * sizeof(int8_t) +
               aligne_size(lhs_reduce_cnt_) * sizeof(float) +
               aligne_size(lhs_reduce_cnt_) * sizeof(int8_t) +
               aligne_size(lhs_reduce_cnt_) * sizeof(int);
  }
  // Rhs isn`t weight and datatype is float.
  // That need workspace to store qdata, scale, zero and redsum.
  // if (inputs[4]->GetDataType() != DataType::INT8) {
  //     ws_size += aligne_size(rhs_cnt_) * sizeof(int8_t) +
  //                aligne_size(rhs_reduce_cnt_) * sizeof(float) +
  //                aligne_size(rhs_reduce_cnt_) * sizeof(int8_t) +
  //                aligne_size(rhs_reduce_cnt_) * sizeof(int);
  // }

  // Gemm-int32 result workspace
  ws_size += aligne_size(y_shape.Count()) * sizeof(int);
  tensor_map_->at("workspace")->SetShape(Shape({ws_size}));

#ifdef ENABLE_CUDA
  const CUDAContext* gpu_ctx = static_cast<const CUDAContext*>(ctx_);
  cublasHandle_t cublas_handle = gpu_ctx->GetCublasHandle();
  cublasSetWorkspace(cublas_handle,
                     tensor_map_->at("cublas_workspace")->GetDataPtr(),
                     tensor_map_->at("cublas_workspace")->GetSizeInByte());
#endif
  return AsStatus::ALLSPARK_SUCCESS;
}
template <typename FType, typename QType>
AsStatus DynamicQuantizeMatmulOp::Preprocess() {
  DLOG(INFO) << "DynamicQuantizeMatmulOp::Preprocess()" << std::endl;
  AsTensor* in_tensor = tensor_map_->at(in_names_[0]).get();
  DataType dtype = in_tensor->GetDataType();
  switch (ctx_->GetDeviceType()) {
    case DeviceType::CUDA: {
      const CUDAContext* gpu_ctx = static_cast<const CUDAContext*>(ctx_);
      cudaStream_t cu_stream = gpu_ctx->GetStream();
      if (dtype != DataType::INT8) {
        const FType* lhs_fdata = static_cast<FType*>(in_tensor->GetDataPtr());
        // workspace store qdata scale zero and redsum
        lhs_qdata_ = (int8_t*)tensor_map_->at("workspace")->GetDataPtr();
        lhs_scale_ =
            (float*)((char*)lhs_qdata_ + aligne_size(lhs_cnt_) * sizeof(QType));
        lhs_zero_ = (int8_t*)((char*)lhs_scale_ +
                              aligne_size(lhs_reduce_cnt_) * sizeof(float));
        lhs_redsum_ = (int32_t*)((char*)lhs_zero_ +
                                 aligne_size(lhs_reduce_cnt_) * sizeof(QType));
        // Quantize Lhs
        bool do_per_channel = (per_channel_ & 0x2) ? true : false;
        Shape lhs_shape = in_tensor->GetShape();
        const int lhs_ndim = lhs_shape.Size();
        int inner_len = lhs_shape[lhs_ndim - 1];
        inner_len *= do_per_channel ? true : lhs_shape[lhs_ndim - 2];
        int outer_len = lhs_shape.Count() / inner_len;
        cuda::QuantizePerChannelImp<FType, QType>(
            lhs_fdata, lhs_qdata_, lhs_scale_, lhs_zero_, lhs_redsum_,
            inner_len, outer_len, cu_stream);
      } else {
        lhs_qdata_ =
            static_cast<int8_t*>(tensor_map_->at(in_names_[0])->GetDataPtr());
        lhs_scale_ =
            static_cast<float*>(tensor_map_->at(in_names_[1])->GetDataPtr());
        lhs_zero_ =
            static_cast<int8_t*>(tensor_map_->at(in_names_[2])->GetDataPtr());
        lhs_redsum_ =
            static_cast<int*>(tensor_map_->at(in_names_[3])->GetDataPtr());
      }
      if (weights_[0]->GetDataType() != DataType::INT8) {
        // TODO: quantize Rhs Kernel
      } else {
        rhs_qdata_ = static_cast<int8_t*>(weights_[0]->GetDataPtr());
        rhs_scale_ = static_cast<float*>(weights_[1]->GetDataPtr());
        rhs_zero_ = static_cast<int8_t*>(weights_[2]->GetDataPtr());
        rhs_redsum_ = static_cast<int*>(weights_[3]->GetDataPtr());
      }
      break;
    }

    case DeviceType::CPU: {
      break;
    }
  }
  return AsStatus::ALLSPARK_SUCCESS;
}

template <typename FType, typename QType>
AsStatus DynamicQuantizeMatmulOp::Postprocess() {
  DLOG(INFO) << "DynamicQuantizeMatmulOp::Postprocess()" << std::endl;
  int bias_ndims = 0;
  FType* bias_ptr = nullptr;
  if (with_bias_ == true) {
    bias_ndims = weights_[4]->GetShape().Size();
    bias_ptr = static_cast<FType*>(weights_[4]->GetDataPtr());
  }
  FType* elemwiseB_ptr = nullptr;
  if (with_elemwiseB_ == true) {
    // elemwiseB_ndims_ = inputs[9]->dim().num_axes();
    // elemwiseB_ptr = static_cast<FType*>(inputs[9]->data());
  }
  const int lhs_ndims = 2;
  const int rhs_ndims = 2;
  const int out_ndims = 2;

  switch (ctx_->GetDeviceType()) {
    case DeviceType::CUDA: {
      const CUDAContext* gpu_ctx = static_cast<const CUDAContext*>(ctx_);
      cudaStream_t cu_stream = gpu_ctx->GetStream();
      if (activation_ == UnaryType::UNARYTYPE_UNDEFINED) {
        hie::NONE<FType> active;
        cuda::postProcessImp(
            lhs_ndims, lhs_reduce_dims_, lhs_scale_, lhs_zero_, lhs_redsum_,
            rhs_ndims, rhs_reduce_dims_, rhs_scale_, rhs_zero_, rhs_redsum_,
            bias_ndims, bias_dims_, bias_ptr, elemwiseA_ndims_, elemwiseA_dims_,
            elemwiseA_, elemwiseB_ndims_, elemwiseB_dims_, elemwiseB_ptr,
            out_ndims, output_dims_,
            static_cast<FType*>(tensor_map_->at(out_names_[0])->GetDataPtr()),
            static_cast<FType>(alpha_), static_cast<FType>(beta_), k_,
            cu_stream, active);
      } else if (activation_ == UnaryType::GELU_ERF) {
        hie::GELU<FType> active;
        cuda::postProcessImp(
            lhs_ndims, lhs_reduce_dims_, lhs_scale_, lhs_zero_, lhs_redsum_,
            rhs_ndims, rhs_reduce_dims_, rhs_scale_, rhs_zero_, rhs_redsum_,
            bias_ndims, bias_dims_, bias_ptr, elemwiseA_ndims_, elemwiseA_dims_,
            elemwiseA_, elemwiseB_ndims_, elemwiseB_dims_, elemwiseB_ptr,
            out_ndims, output_dims_,
            static_cast<FType*>(tensor_map_->at(out_names_[0])->GetDataPtr()),
            static_cast<FType>(alpha_), static_cast<FType>(beta_), k_,
            cu_stream, active);
      } else if (activation_ == UnaryType::GELU_TANH) {
        hie::GELU_TANH<FType> active;
        cuda::postProcessImp(
            lhs_ndims, lhs_reduce_dims_, lhs_scale_, lhs_zero_, lhs_redsum_,
            rhs_ndims, rhs_reduce_dims_, rhs_scale_, rhs_zero_, rhs_redsum_,
            bias_ndims, bias_dims_, bias_ptr, elemwiseA_ndims_, elemwiseA_dims_,
            elemwiseA_, elemwiseB_ndims_, elemwiseB_dims_, elemwiseB_ptr,
            out_ndims, output_dims_,
            static_cast<FType*>(tensor_map_->at(out_names_[0])->GetDataPtr()),
            static_cast<FType>(alpha_), static_cast<FType>(beta_), k_,
            cu_stream, active);
      }

      break;
    }
    case DeviceType::CPU: {
      break;
    }
  }
  return AsStatus::ALLSPARK_SUCCESS;
}
bool inline useTensorCore(const int batch, const int m, const int n,
                          const int k) {
  // useTC = gpuinfo.support_imma();
  // when m < 512, Tensor-core doesn't improve very much.
  if (m < 512) return false;
  // use tensor-core limit
  if (n % 16 != 0 || k % 16 != 0) return false;
  // 目前不支持多batch
  if (batch != 1) return false;
  return true;
}
AsStatus DynamicQuantizeMatmulOp::Forward() {
  switch (dtype_) {
    case DataType::FLOAT32:
      Preprocess<float, int8_t>();
      break;
    case DataType::FLOAT16:
      Preprocess<half, int8_t>();
      break;
    default:
      break;
  }

  switch (ctx_->GetDeviceType()) {
#ifdef ENABLE_CUDA
    case DeviceType::CUDA: {
      const CUDAContext* gpu_ctx = static_cast<const CUDAContext*>(ctx_);
      cublasHandle_t cublas_handle = gpu_ctx->GetCublasHandle();
      cudaStream_t cu_stream = gpu_ctx->GetStream();
      elemwiseA_ = (int32_t*)elemwiseA->GetDataPtr();
      const int8_t* typed_l = static_cast<const int8_t*>(lhs_qdata_);
      const int8_t* typed_r = static_cast<const int8_t*>(rhs_qdata_);
      int32_t* typed_out = static_cast<int32_t*>(elemwiseA_);
      if (useTensorCore(batch_, m_, n_, k_) == true && transA_ == 0 &&
          transB_ == 0) {
        cuda::GemmHIEInt8(typed_out, typed_l, typed_r, m_, n_, k_, cu_stream);
      } else {
        cuda::GemmInt8(typed_out, typed_l, typed_r, m_, n_, k_, transA_,
                       transB_, lda_, ldb_, ldc_, 1, 0, cublas_handle,
                       cu_stream);
      }
      break;
    }
#endif
    case DeviceType::CPU: {
      break;
    }
    default: {
      LOG(ERROR) << "DynamicQuantizeMatmulOp Operator does not support "
                 << DeviceType_Name(ctx_->GetDeviceType()) << " device type"
                 << std::endl;
      return AsStatus::ALLSPARK_RUNTIME_ERROR;
    }
  }
  switch (dtype_) {
    case DataType::FLOAT32:
      Postprocess<float, int8_t>();
      break;
    case DataType::FLOAT16:
      Postprocess<half, int8_t>();
      break;
    default:
      break;
  }
  return AsStatus::ALLSPARK_SUCCESS;
}

REGISTER_OP(DynamicQuantizeMatmul, CUDA, DynamicQuantizeMatmulOp)
REGISTER_OP(DynamicQuantizeMatmul, CPU, DynamicQuantizeMatmulOp)
}  // namespace allspark
#endif
