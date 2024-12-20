/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    gemm_op_gpu.cpp
 */

#ifdef ENABLE_CUDA
#include "gemm_op_gpu.h"

#include <core/kernel/cuda/gemm_lowp/gemm_a16w8_kernel.h>
#include <core/kernel/kernel.h>
#include <cuda/cuda_context.h>
#include <utility/datatype_dispatcher.h>

namespace allspark {
AsStatus hgemm_32x128x16_simt_Aldg1_kernel_launcher(
    const half* in, const half* weight, const half* bias, half* out, uint32_t m,
    uint32_t n, uint32_t k, const int activation, const int sm_count,
    void* ws_ptr, cudaStream_t stream) {
  switch (activation) {
    case 0:  // None
      cuda::hgemm_32x128x16_simt_Aldg1<hie::activation::Identity>(
          in, weight, bias, out, m, n, k, sm_count, ws_ptr, stream);
      break;
    // case 1:// Tanh
    //     break;
    case 2:  // Gelu
      cuda::hgemm_32x128x16_simt_Aldg1<hie::activation::Gelu>(
          in, weight, bias, out, m, n, k, sm_count, ws_ptr, stream);
      break;
    case 3:  // GeluTanh
      cuda::hgemm_32x128x16_simt_Aldg1<hie::activation::GeluTanh>(
          in, weight, bias, out, m, n, k, sm_count, ws_ptr, stream);
      break;
    case 4:  // Relu
      cuda::hgemm_32x128x16_simt_Aldg1<hie::activation::Relu>(
          in, weight, bias, out, m, n, k, sm_count, ws_ptr, stream);
      break;
    default:
      LOG(ERROR) << "No Kernel for acvtivtion : " << activation << std::endl;
      break;
  }
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus dense_gemm_rawptr(DataType dtype, void* out, const void* in,
                           const void* bias, const void* weight, int m, int n,
                           int k, int lda, int ldb, int ldc, bool transA,
                           bool transB, int batch, float alpha,
                           const void* binary_in, UnaryType activation,
                           const DeviceContext* ctx) {
  const CUDAContext* gpu_ctx = static_cast<const CUDAContext*>(ctx);
  cublasHandle_t cublas_handle = gpu_ctx->GetCublasHandle();
  cudaStream_t cu_stream = gpu_ctx->GetStream();
  if (binary_in != nullptr && bias != nullptr) {
    LOG(ERROR) << "binary_in and bias cannot be used at the same time";
    AS_THROW(AsStatus::ALLSPARK_PARAM_ERROR);
  }
  if (gpu_ctx->GetMatmulPrecision() == PrecisionLevel::HIGH &&
      dtype == FLOAT32) {
    cublasSetMathMode(cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH);
  }
  auto functor = [&]<typename T>() {
    T* typed_out = static_cast<T*>(out);
    const T* typed_input = static_cast<const T*>(in);
    const T* typed_bias = static_cast<const T*>(bias);
    const T* typed_weight = static_cast<const T*>(weight);
    const T* typed_binary_in = static_cast<const T*>(binary_in);
    if (batch == 1) {
      cuda::GemmWraper<T>(typed_out, typed_input, typed_weight, typed_bias, m,
                          n, k, transA, transB, lda, ldb, ldc, alpha, 0.0f,
                          typed_binary_in, cublas_handle, cu_stream);
    } else {
      cuda::StridedBatchGemmWraper<T>(
          typed_out, typed_input, typed_weight, typed_bias, m, n, k, false,
          transB, lda, ldb, ldc, alpha, 0.0f, batch, typed_binary_in,
          cublas_handle, cu_stream);
    }
    if (activation != UNARYTYPE_UNDEFINED) {
      cuda::UnaryKernelLauncher(typed_out, typed_out, (int64_t)m * n * batch,
                                activation, cu_stream);
    }
  };
  DispatchCUDA(dtype, functor);
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus dense_gemm(DataType dtype, void* out, const void* in, const void* bias,
                    const AsTensor* weight, int m, int n, int k, int lda,
                    int ldb, int ldc, bool transA, bool transB, int batch,
                    float alpha, const void* binary_in, UnaryType activation,
                    const DeviceContext* ctx) {
  const CUDAContext* gpu_ctx = static_cast<const CUDAContext*>(ctx);
  cublasHandle_t cublas_handle = gpu_ctx->GetCublasHandle();
  cudaStream_t cu_stream = gpu_ctx->GetStream();
  if (binary_in != nullptr && bias != nullptr) {
    LOG(ERROR) << "binary_in and bias cannot be used at the same time";
    AS_THROW(AsStatus::ALLSPARK_PARAM_ERROR);
  }
  if (gpu_ctx->GetMatmulPrecision() == PrecisionLevel::HIGH &&
      dtype == FLOAT32) {
    cublasSetMathMode(cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH);
  }
  auto functor = [&]<typename T>() {
    T* typed_out = static_cast<T*>(out);
    const T* typed_input = static_cast<const T*>(in);
    const T* typed_bias = static_cast<const T*>(bias);
    const T* typed_weight = static_cast<const T*>(weight->GetDataPtr());
    const T* typed_binary_in = static_cast<const T*>(binary_in);
    if (batch == 1) {
      cuda::GemmWraper<T>(typed_out, typed_input, typed_weight, typed_bias, m,
                          n, k, transA, transB, lda, ldb, ldc, alpha, 0.0f,
                          typed_binary_in, cublas_handle, cu_stream);
    } else {
      cuda::StridedBatchGemmWraper<T>(
          typed_out, typed_input, typed_weight, typed_bias, m, n, k, false,
          transB, lda, ldb, ldc, alpha, 0.0f, batch, typed_binary_in,
          cublas_handle, cu_stream);
    }
    if (activation != UNARYTYPE_UNDEFINED) {
      cuda::UnaryKernelLauncher(typed_out, typed_out, (int64_t)m * n * batch,
                                activation, cu_stream);
    }
  };
  DispatchCUDA(dtype, functor);
  return AsStatus::ALLSPARK_SUCCESS;
}
#ifdef ENABLE_SPARSE
AsStatus patternbase_gemm(DataType dtype, void* out, const void* in,
                          const void* bias, const AsTensor* weight, int m,
                          int n, int k, int lda, int ldb, int ldc, bool transA,
                          bool transB, int batch, float alpha,
                          const void* binary_in, UnaryType activation,
                          const DeviceContext* ctx) {
  (void)binary_in;
  const CUDAContext* gpu_ctx = static_cast<const CUDAContext*>(ctx);
  cublasHandle_t cublas_handle = gpu_ctx->GetCublasHandle();
  cudaStream_t cu_stream = gpu_ctx->GetStream();
  auto functor = [&]<typename T>() {
    T* typed_out = static_cast<T*>(out);
    const T* typed_input = static_cast<const T*>(in);
    const T* typed_bias = static_cast<const T*>(bias);
    const CSCData* weight_data = static_cast<const CSCData*>(weight->GetData());
    const int* weight_col_offset =
        static_cast<const int*>(weight_data->GetColOffsets());
    const int* weight_row_indices =
        static_cast<const int*>(weight_data->GetRowIndices());
    const T* weight_val = static_cast<const T*>(weight_data->GetRawData());
    cuda::CuSparseGemmCSC(typed_out, typed_input, weight_col_offset,
                          weight_row_indices, weight_val, weight_data->GetNNZ(),
                          typed_bias, m, n, k, transA, transB, lda, ldb, ldc,
                          alpha, 0.0f, batch, cu_stream);
    if (activation != UNARYTYPE_UNDEFINED) {
      cuda::UnaryKernelLauncher(typed_out, typed_out, m * n * batch, activation,
                                cu_stream);
    }
  };
  DispatchCUDA(dtype, functor);
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus pattern0_gemm(DataType dtype, void* out, const void* in,
                       const void* bias, const AsTensor* weight, int m, int n,
                       int k, int lda, int ldb, int ldc, bool transA,
                       bool transB, int batch, float alpha,
                       const void* binary_in, UnaryType activation,
                       const DeviceContext* ctx) {
  (void)binary_in;
  const CUDAContext* gpu_ctx = static_cast<const CUDAContext*>(ctx);
  cudaStream_t cu_stream = gpu_ctx->GetStream();
  auto functor = [&]<typename T>() {
    T* typed_out = static_cast<T*>(out);
    const T* typed_input = static_cast<const T*>(in);
    const T* typed_bias = static_cast<const T*>(bias);
    const CSCData* weight_data = static_cast<const CSCData*>(weight->GetData());
    const int* weight_col_offset =
        static_cast<const int*>(weight_data->GetColOffsets());
    const int* weight_row_indices =
        static_cast<const int*>(weight_data->GetRowIndices());
    const T* weight_val = static_cast<const T*>(weight_data->GetRawData());
    cuda::broadcast_kernel_launcher(typed_out, typed_bias, m * n, n, batch,
                                    cu_stream);
    cuda::spmm_pattern0(typed_input, weight_val, weight_col_offset,
                        weight_row_indices, typed_out, m, n, k,
                        weight_data->GetNNZ(), cu_stream);
    if (activation != UNARYTYPE_UNDEFINED) {
      cuda::UnaryKernelLauncher(typed_out, typed_out, m * n * batch, activation,
                                cu_stream);
    }
  };
  DispatchCUDA(dtype, functor);
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus pattern1_gemm(DataType dtype, void* out, const void* in,
                       const void* bias, const AsTensor* weight, int m, int n,
                       int k, int lda, int ldb, int ldc, bool transA,
                       bool transB, int batch, float alpha,
                       const void* binary_in, UnaryType activation,
                       const DeviceContext* ctx) {
  (void)binary_in;
  const CUDAContext* gpu_ctx = static_cast<const CUDAContext*>(ctx);
  cudaStream_t cu_stream = gpu_ctx->GetStream();
  auto functor = [&]<typename T>() {
    T* typed_out = static_cast<T*>(out);
    const T* typed_input = static_cast<const T*>(in);
    const T* typed_bias = static_cast<const T*>(bias);
    const ELLData* weight_data = static_cast<const ELLData*>(weight->GetData());
    const unsigned short* weight_row_indices =
        static_cast<const unsigned short*>(weight_data->GetRowIndices());
    const T* weight_val = static_cast<const T*>(weight_data->GetRawData());
    cuda::broadcast_kernel_launcher(typed_out, typed_bias, m * n, n, batch,
                                    cu_stream);
    cuda::spmm_pattern1(typed_input, weight_val, weight_row_indices, typed_out,
                        m, n, k, weight_data->GetNNZ(), cu_stream);
    if (activation != UNARYTYPE_UNDEFINED) {
      cuda::UnaryKernelLauncher(typed_out, typed_out, m * n * batch, activation,
                                cu_stream);
    }
  };
  DispatchCUDA(dtype, functor);
  return AsStatus::ALLSPARK_SUCCESS;
}
#endif

AsStatus GemmOpGPU::Init(const OperatorProto& op_proto,
                         const DeviceContext& ctx, const TensorMap& weights_map,
                         TensorMap* tensor_map) {
  LOG(ERROR) << "GemmOpGPU only support InitV2()" << std::endl;
  return AsStatus::ALLSPARK_INVALID_CALL_ERROR;
}

AsStatus GemmOpGPU::InitV2(const OperatorProto& op_proto,
                           const DeviceContext& ctx,
                           const TensorMap& weights_map,
                           TensorMap& weights_buffer, TensorMap* tensor_map) {
  AS_CHECK_STATUS(GemmOpBase::InitV2(op_proto, ctx, weights_map, weights_buffer,
                                     tensor_map));

  // Padding case 1: k % 8 != 0 and n % 8 == 0
  if (n_ % 8 == 0 && k_ % 8 != 0 && !transB_ && batch_ == 1 &&
      weights_[0]->GetDataType() == DataType::FLOAT16) {
    get_weight_padded_k_align8<half>(weights_buffer);
    is_kpad_ = true;
    // Reset lda_
    lda_ = k_;
  }
  // Padding case 2: k % 8 == 0 and n % 8 != 0
  if (n_ % 8 != 0 && k_ % 8 == 0 && !transB_ && batch_ == 1 &&
      weights_[0]->GetDataType() == DataType::FLOAT16) {
    get_weight_padded_n_align8<half>(weights_buffer);
    is_npad_ = true;
    // Reset ldb_, ldc_
    ldb_ = n_;
    ldc_ = n_;
  }
  switch (weights_[0]->GetDataMode()) {
    case allspark::DataMode::DENSE:
      kernel_launcher = dense_gemm;
      break;
#ifdef ENABLE_SPARSE
    case allspark::DataMode::CSC: {
      kernel_launcher = pattern0_gemm;
      break;
    }
    case allspark::DataMode::ELL:
      kernel_launcher = pattern1_gemm;
      break;
#endif
  }

  // Get Device SM Count
  /*
  cudaDeviceProp device_prop;
  int device_id;
  cudaGetDevice(&device_id);
  cudaGetDeviceProperties(&device_prop, device_id);
  sm_count_ = device_prop.multiProcessorCount;
  */
  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus GemmOpGPU::Reshape() {
  int yn = is_npad_ ? n_padded_before_ : n_;
  AS_CHECK_STATUS(GemmOpBase::Reshape(yn));

  dim_t ws_size = 0;
  ws_size += is_kpad_ ? size_t(m_ * k_) * sizeof(half) : size_t(0);
  ws_size += is_npad_ ? size_t(m_ * n_) * sizeof(half) : size_t(0);
  if (ws_size > 0) {
    tensor_map_->at("workspace")->SetShape(Shape{ws_size});
  }
  /*
      if (m_ < 32 && weights_[0]->GetDataType() == DataType::FLOAT16 && batch_
     == 1) {

          uint32_t grid_x = (m_ + 31) / 32;
          uint32_t grid_y = (n_ + 127) / 128;
          uint32_t grid_z;

          const float SPLIT_THRESHOLD = 5;
          uint32_t n_slice = 2;
          for (n_slice = 1; n_slice < k_ / 128; ++n_slice) {
              uint32_t n_block = grid_x * grid_y * n_slice;
              if (n_block >= sm_count_ * SPLIT_THRESHOLD && (
                  n_block % sm_count_ == 0 ||
                  n_block % sm_count_ >= sm_count_ / 2)) {
                  break;
              }
          }

          uint32_t k_slice = (k_ / n_slice) % 16 == 0 ?
                          k_ / n_slice : k_ / n_slice / 16 * 16 + 16;

          grid_z = (k_ + k_slice - 1) / k_slice;
          ws_size += grid_z * m_ * n_ * sizeof(half);
          tensor_map_->at("workspace")->SetShape(Shape{ws_size});
      }
  */
  const CUDAContext* gpu_ctx = static_cast<const CUDAContext*>(ctx_);
  cublasHandle_t cublas_handle = gpu_ctx->GetCublasHandle();
  cublasSetWorkspace(cublas_handle,
                     tensor_map_->at("cublas_workspace")->GetDataPtr(),
                     tensor_map_->at("cublas_workspace")->GetSizeInByte());
  return AsStatus::ALLSPARK_SUCCESS;
}
template <typename FType, typename QType>
AsStatus GemmOpGPU::DeQuantize() {
  DLOG(INFO) << "GemmOpGPU::DeQuantize()" << std::endl;

  const CUDAContext* gpu_ctx = static_cast<const CUDAContext*>(ctx_);
  cudaStream_t cu_stream = gpu_ctx->GetStream();
  FType* lhs_fdata =
      static_cast<FType*>(tensor_map_->at("workspace")->GetDataPtr());
  // workspace store qdata scale zero and redsum
  const int8_t* lhs_qdata_ =
      static_cast<const int8_t*>(weights_[0]->GetDataPtr());
  const float* lhs_scale_ =
      static_cast<const float*>(weights_[1]->GetDataPtr());
  // const int8_t* lhs_zero_ =
  //     static_cast<const int8_t*>(weights_[2]->GetDataPtr());
  const float* lhs_zero_ = static_cast<const float*>(weights_[2]->GetDataPtr());
  const int* lhs_redsum_ = static_cast<const int*>(weights_[3]->GetDataPtr());
  // DeQuantize Lhs
  Shape lhs_shape = weights_[0]->GetShape();
  const int lhs_ndim = lhs_shape.Size();
  int inner_len = lhs_shape[lhs_ndim - 1];
  int outer_len = lhs_shape.Count() / inner_len;
  cuda::DeQuantizePerChannelImp<FType, QType>(lhs_fdata, lhs_qdata_, lhs_scale_,
                                              lhs_zero_, lhs_redsum_, inner_len,
                                              outer_len, cu_stream);

  return AsStatus::ALLSPARK_SUCCESS;
}
AsStatus GemmOpGPU::Forward() {
  AsTensor* in_tensor = tensor_map_->at(in_names_[0]).get();
  void* in = in_tensor->GetDataPtr();
  void* out = tensor_map_->at(out_names_[0])->GetDataPtr();
  const AsTensor* weight = static_cast<const AsTensor*>(weights_[0]);
  void* bias = (weights_.size() == 2) ? weights_[1]->GetDataPtr() : nullptr;
  void* bin_in = (in_names_.size() >= 2)
                     ? tensor_map_->at(in_names_[1])->GetDataPtr()
                     : nullptr;
  bin_in = (binary_type_ == BinaryType::ADD) ? bin_in : nullptr;
  if (weights_[0]->GetDataType() == DataType::INT8) {
    switch (dtype_) {
      case DataType::FLOAT32: {
        DeQuantize<float, int8_t>();
        weight = tensor_map_->at("workspace").get();
        bias = (weights_.size() == 5) ? weights_[4]->GetDataPtr() : nullptr;
        break;
      }
      case DataType::FLOAT16: {
        DeQuantize<half, int8_t>();
        weight = tensor_map_->at("workspace").get();
        bias = (weights_.size() == 5) ? weights_[4]->GetDataPtr() : nullptr;
        break;
      }
      default:
        break;
    }
  }

  const CUDAContext* gpu_ctx = static_cast<const CUDAContext*>(ctx_);
  cudaStream_t cu_stream = gpu_ctx->GetStream();
  void* ws_ptr = tensor_map_->at("workspace")->GetDataPtr();
  if (is_kpad_) {
    void* in_padded_ptr = ws_ptr;
    const Shape& x_shape = tensor_map_->at(in_names_[0])->GetShape();
    int x_ndims = x_shape.Size();
    cuda::get_input_padded_k_align(static_cast<half*>(in),
                                   static_cast<half*>(in_padded_ptr), m_,
                                   x_shape[x_ndims - 1], k_, cu_stream);
    in = in_padded_ptr;
    ws_ptr = static_cast<half*>(ws_ptr) + m_ * k_;
  }
  if (is_npad_) {
    out = static_cast<half*>(ws_ptr);
    ws_ptr = static_cast<half*>(ws_ptr) + m_ * n_;
  }
  if (is_split_k_) {
    in = (char*)in + k_ * rank_id_ * SizeofType(dtype_);
  }
  // TODO: Close. This kernel may be have a precision problem.
  /*

  if (m_ < 32 && weights_[0]->GetDataType() == DataType::FLOAT16 && batch_ == 1)
  { hgemm_32x128x16_simt_Aldg1_kernel_launcher( static_cast<const half*>(in),
          static_cast<const half*>(weight->GetDataPtr()),
          static_cast<const half*>(bias),
          static_cast<half*>(out),
          uint32_t(m_),
          uint32_t(n_),
          uint32_t(k_),
          activation_,
          sm_count_,
          ws_ptr,
          cu_stream);
  } else {
  */
  kernel_launcher(in_tensor->GetDataType(), out, in, bias, weight, m_, n_, k_,
                  lda_, ldb_, ldc_, false, transB_, batch_, alpha_, bin_in,
                  activation_, ctx_);
  /*
  }
  */

  // if padding n to multiples of 8 before, need to remove padding here
  if (is_npad_) {
    cuda::remove_padded_n_align(
        static_cast<half*>(out),
        static_cast<half*>(tensor_map_->at(out_names_[0])->GetDataPtr()), m_,
        n_padded_before_, n_, cu_stream);
  }
  return AsStatus::ALLSPARK_SUCCESS;
}

template <typename FType>
void GemmOpGPU::get_weight_padded_k_align8(TensorMap& weights_buffer) {
  int64_t k_padded = (k_ + 7) / 8 * 8;

  AsTensor old_weight_cpu = AsTensor(*weights_[0], DeviceType::CPU);
  AsTensor padded_weight_cpu =
      AsTensor(weights_[0]->GetName() + "padded_cpu", DeviceType::CPU,
               weights_[0]->GetDataType(), weights_[0]->GetDataMode(),
               Shape({k_padded, n_}));
  // padding weight
  FType* padded_weight_ptr =
      static_cast<FType*>(padded_weight_cpu.GetDataPtr());
  FType* old_weight_ptr = static_cast<FType*>(old_weight_cpu.GetDataPtr());

  for (int i = 0; i < k_padded; ++i) {
    for (int j = 0; j < n_; ++j) {
      if (i < k_) {
        padded_weight_ptr[i * n_ + j] = old_weight_ptr[i * n_ + j];
      } else {
        padded_weight_ptr[i * n_ + j] = FType(0.f);
      }
    }
  }

  const_cast<AsTensor*>(weights_[0])->SetShape(Shape({k_padded, n_}));
  TensorUtils::DeepCopyWholeAsync(*const_cast<AsTensor*>(weights_[0]),
                                  padded_weight_cpu, ctx_);
  util::SyncWeightsBuffer(weights_buffer, weights_[0]->GetName(),
                          std::make_shared<AsTensor>(padded_weight_cpu));
  // Reset k_
  k_ = k_padded;
}

template <typename FType>
void GemmOpGPU::get_weight_padded_n_align8(TensorMap& weights_buffer) {
  int64_t n_padded = (n_ + 7) / 8 * 8;

  AsTensor old_weight_cpu = AsTensor(*weights_[0], DeviceType::CPU);
  AsTensor padded_weight_cpu =
      AsTensor(weights_[0]->GetName() + "padded_cpu", DeviceType::CPU,
               weights_[0]->GetDataType(), weights_[0]->GetDataMode(),
               Shape({k_, n_padded}));
  // padding weight
  FType* padded_weight_ptr =
      static_cast<FType*>(padded_weight_cpu.GetDataPtr());
  const FType* old_weight_ptr =
      static_cast<FType*>(old_weight_cpu.GetDataPtr());

  for (int i = 0; i < k_; ++i) {
    for (int j = 0; j < n_padded; ++j) {
      if (j < n_) {
        padded_weight_ptr[i * n_padded + j] = old_weight_ptr[i * n_ + j];
      } else {
        padded_weight_ptr[i * n_padded + j] = FType(0.f);
      }
    }
  }

  const_cast<AsTensor*>(weights_[0])->SetShape(Shape({k_, n_padded}));

  TensorUtils::DeepCopyWholeAsync(*const_cast<AsTensor*>(weights_[0]),
                                  padded_weight_cpu, ctx_);
  util::SyncWeightsBuffer(weights_buffer, weights_[0]->GetName(),
                          std::make_shared<AsTensor>(padded_weight_cpu));

  // Pad Bias
  if (weights_.size() == 2) {
    AsTensor old_bias_cpu = AsTensor(*weights_[1], DeviceType::CPU);
    AsTensor padded_bias_cpu =
        AsTensor(weights_[1]->GetName() + "padded_cpu", DeviceType::CPU,
                 weights_[1]->GetDataType(), weights_[1]->GetDataMode(),
                 Shape({n_padded}));
    FType* padded_bias_ptr = static_cast<FType*>(padded_bias_cpu.GetDataPtr());
    const FType* old_bias_ptr = static_cast<FType*>(old_bias_cpu.GetDataPtr());
    for (int i = 0; i < n_padded; ++i) {
      padded_bias_ptr[i] = i < n_ ? old_bias_ptr[i] : FType(0.f);
    }
    AsTensor* mutable_bias = const_cast<AsTensor*>(weights_[1]);
    mutable_bias->SetShape(Shape({n_padded}));
    TensorUtils::DeepCopyWholeAsync(*mutable_bias, padded_bias_cpu, ctx_);
    util::SyncWeightsBuffer(weights_buffer, weights_[1]->GetName(),
                            std::make_shared<AsTensor>(padded_bias_cpu));
  }

  // Reset n_, n_padded_before_
  n_padded_before_ = n_;
  n_ = n_padded;
}

REGISTER_OP(Gemm, CUDA, GemmOpGPU)
}  // namespace allspark
#endif
