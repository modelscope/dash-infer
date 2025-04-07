/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    gemm_a8w8_gpu.cpp
 */

#ifdef ENABLE_CUDA
#include "gemm_a8w8_gpu.h"

#include <core/kernel/kernel.h>
#include <cpu/cpu_context.h>
#include <cuda/cuda_context.h>
#include <utility/datatype_dispatcher.h>

#ifdef ENABLE_BF16
#include <common/hie_bfloat16.hpp>
#endif

#define A8W8_N_PAD_ALIGN 16
#define A8W8_K_PAD_ALIGN 16
namespace allspark {
AsStatus GemmA8W8GPU::Init(const OperatorProto& op_proto,
                           const DeviceContext& ctx,
                           const TensorMap& weights_map,
                           TensorMap* tensor_map) {
  LOG(ERROR) << "GemmA8W8GPU only support InitV2()" << std::endl;
  return AsStatus::ALLSPARK_INVALID_CALL_ERROR;
}

AsStatus GemmA8W8GPU::InitV2(const OperatorProto& op_proto,
                             const DeviceContext& ctx,
                             const TensorMap& weights_map,
                             TensorMap& weights_buffer, TensorMap* tensor_map,
                             RuntimeContext* runtime_ctx) {
  AS_CHECK_STATUS(GemmA16W8Base::InitV2(
      op_proto, ctx, weights_map, weights_buffer, tensor_map, runtime_ctx));
  set_padding_flag(runtime_ctx);

  // Get Device SM Count
  cudaDeviceProp device_prop;
  int device_id;
  cudaGetDevice(&device_id);
  cudaGetDeviceProperties(&device_prop, device_id);
  sm_count_ = device_prop.multiProcessorCount;
  sm_version_ = (device_prop.major << 8 | device_prop.minor);
  // fallback to a16w8 for decoder
  auto non_const_ctx = const_cast<DeviceContext*>(ctx_);
  non_const_ctx->SetFallbackDecoderWeightOnly(true);
  a16w8_op_ = std::make_unique<allspark::GemmA16W8GPU>();
  a16w8_op_->CallInit(op_proto, ctx, weight_manager_, weight_handler_,
                      lora_manager_, rank_info_, tensor_map, profiler_,
                      runtime_ctx);
  GetWeightPaddedDispatch(tensor_map_->at(in_names_[0])->GetDataType(), qtype_,
                          weights_buffer);
  return AsStatus::ALLSPARK_SUCCESS;
}

void GemmA8W8GPU::GetWeightPaddedDispatch(const DataType ftype,
                                          const DataType qtype,
                                          TensorMap& weights_buffer) {
  // TODO: uint8
  switch (ftype) {
    case DataType::FLOAT16: {
      if (qtype == DataType::INT8) {
        // Padding case 1: k % A8W8_K_PAD_ALIGN != 0 and n % A8W8_N_PAD_ALIGN
        // == 0
        if (n_ % A8W8_N_PAD_ALIGN == 0 && k_ % A8W8_K_PAD_ALIGN != 0 &&
            !transB_) {
          get_weight_padded_k_align<half, int8_t>(weights_buffer);
          is_kpad_ = true;
        }
        // Padding case 2: k % A8W8_K_PAD_ALIGN == 0 and n % A8W8_N_PAD_ALIGN
        // != 0
        if (n_ % A8W8_N_PAD_ALIGN != 0 && k_ % A8W8_K_PAD_ALIGN == 0 &&
            !transB_) {
          get_weight_padded_n_align<half, int8_t>(weights_buffer);
          is_npad_ = true;
        }
        // Special case: if sm8x and group_size_ = -1
        // reorder B as N32K16 order for use with Ampere_A16W8_GEMM_PERC_16816
        // kernel
        if (sm_version_ >= 0x0800 && group_size_ == -1) {
          B_I8_Reorder_Hmma16816_N32_K16_Gpu<half, int8_t>(weights_buffer);
        }
      }
      break;
    }
    case DataType::BFLOAT16: {
      if (qtype == DataType::INT8) {
        // Padding case 1: k % A8W8_K_PAD_ALIGN != 0 and n % A8W8_N_PAD_ALIGN
        // == 0
        if (n_ % A8W8_N_PAD_ALIGN == 0 && k_ % A8W8_K_PAD_ALIGN != 0 &&
            !transB_) {
          get_weight_padded_k_align<hie::bfloat16, int8_t>(weights_buffer);
          is_kpad_ = true;
        }
        // Padding case 2: k % A8W8_N_PAD_ALIGN == 0 and n % A8W8_N_PAD_ALIGN
        // != 0
        if (n_ % A8W8_N_PAD_ALIGN != 0 && k_ % A8W8_K_PAD_ALIGN == 0 &&
            !transB_) {
          get_weight_padded_n_align<hie::bfloat16, int8_t>(weights_buffer);
          is_npad_ = true;
        }
        // Special case: if sm8x and group_size_ = -1
        // reorder B as N32K16 order for use with Ampere_A16W8_GEMM_PERC_16816
        // kernel
        if (sm_version_ >= 0x0800 && group_size_ == -1) {
          B_I8_Reorder_Hmma16816_N32_K16_Gpu<hie::bfloat16, int8_t>(
              weights_buffer);
        }
      }
      break;
    }
    default:
      LOG(ERROR) << "GemmA8W8GPU DataType Error\n";
      break;
  }
}

template <typename FT, typename QT>
void GemmA8W8GPU::DispatchKernel() {
  AsTensor* in_tensor = tensor_map_->at(in_names_[0]).get();
  void* in = in_tensor->GetDataPtr();
  void* rhs_qdata_ptr = weights_[0]->GetDataPtr();
  void* rhs_scales_ptr = weights_[1]->GetDataPtr();
  void* rhs_zeros_ptr = weights_[2]->GetDataPtr();
  void* ws_ptr = tensor_map_->at("workspace")->GetDataPtr();

  void* bias = (weights_.size() == 4) ? weights_[3]->GetDataPtr() : nullptr;
  void* out = tensor_map_->at(out_names_[0])->GetDataPtr();

  const CUDAContext* gpu_ctx = static_cast<const CUDAContext*>(ctx_);
  cublasHandle_t cublas_handle = gpu_ctx->GetCublasHandle();
  cudaStream_t cu_stream = gpu_ctx->GetStream();

  if (is_kpad_) {
    void* in_padded_ptr = ws_ptr;
    const Shape& x_shape = tensor_map_->at(in_names_[0])->GetShape();
    int x_ndims = x_shape.Size();
    cuda::get_input_padded_k_align(static_cast<FT*>(in),
                                   static_cast<FT*>(in_padded_ptr), m_,
                                   x_shape[x_ndims - 1], k_, cu_stream);
    in = in_padded_ptr;
    ws_ptr = static_cast<FT*>(ws_ptr) + aligned_size(m_ * k_);
  }

  if (is_npad_) {
    out = ws_ptr;
    ws_ptr = static_cast<FT*>(ws_ptr) + aligned_size(m_ * n_);
  }

  // A per channel quantization from FType to int8_t
  int8_t* a_qdata = reinterpret_cast<int8_t*>(ws_ptr);
  float* a_scale = reinterpret_cast<float*>(a_qdata + aligned_size(m_ * k_) *
                                                          sizeof(int8_t));
  float* a_red_max = reinterpret_cast<float*>((char*)a_scale +
                                              aligned_size(m_) * sizeof(float));
  uint32_t* a_red_count = reinterpret_cast<uint32_t*>(
      (char*)a_red_max + aligned_size(m_) * sizeof(float));
  int32_t* a_red_sum = reinterpret_cast<int32_t*>(
      (char*)a_red_count + aligned_size(m_) * sizeof(uint32_t));
  if (m_ >= 65536) {
    LOG(ERROR) << "sequence length >= 65536 in not support in A8W8 gemm";
    AS_THROW(AsStatus::ALLSPARK_PARAM_ERROR);
  }
  cuda::per_channel_symm_dynamic_quantization<FT>(
      (FT*)in, a_qdata, a_scale, a_red_max, a_red_count, a_red_sum, m_, k_,
      sm_count_, cu_stream);
  // restore weight from N32K16 special format to NK format
  int n_32align = (n_ + 32 - 1) / 32 * 32;
  int8_t* b_qdata_result = reinterpret_cast<int8_t*>(
      (char*)a_red_sum + aligned_size(m_) * sizeof(int32_t));
  FT* b_scale_result = reinterpret_cast<FT*>(
      (char*)b_qdata_result + aligned_size(n_ * k_) * sizeof(int8_t));
  FT* b_zero_result = reinterpret_cast<FT*>((char*)b_scale_result +
                                            aligned_size(n_) * sizeof(FT));
  cuda::restore_n32k16_weight_to_nk<FT>(
      (const int8_t*)rhs_qdata_ptr, (FT*)rhs_scales_ptr, (FT*)rhs_zeros_ptr,
      b_qdata_result, b_scale_result, b_zero_result, n_32align, n_, k_,
      cu_stream);
  // A8W8 Gemm
  int32_t* imd_result = reinterpret_cast<int32_t*>(
      (char*)b_zero_result + aligned_size(n_) * sizeof(FT));
  int lda = k_;
  int ldb = k_;
  int ldc = n_;
  cuda::GemmInt8(imd_result, a_qdata, (QT*)b_qdata_result, m_, n_, k_, false,
                 true, lda, ldb, ldc, 1, 0, cublas_handle, cu_stream);
  // C dequantization
  cuda::A_perc_symm_B_perc_asymm_dequantization<FT>(
      imd_result, a_scale, a_red_sum, b_scale_result, b_zero_result, (FT*)bias,
      (FT*)out, m_, n_, cu_stream);
  // LOG(INFO) << "GemmA8W8GPU::DispatchKernelA8() is_kpad_: " << is_kpad_ << "
  // is_npad_: " << is_npad_
  //           << " m_: " << m_ << " n_: " << n_ << " k_: " << k_
  //           << " n_32align: " << n_32align << " ws_ptr: " << ws_ptr
  //           << " bias: " << bias << " weights_.size(): " << weights_.size()
  //           << " weight0 name: " << weights_[0]->GetName();
  // if padding n to multiples of A8W8_N_PAD_ALIGN before, need to remove
  // padding here
  if (activation_ != UNARYTYPE_UNDEFINED) {
    cuda::UnaryKernelLauncher(static_cast<FT*>(out), static_cast<FT*>(out),
                              m_ * n_ * batch_, activation_, cu_stream);
  }
  if (is_npad_) {
    cuda::remove_padded_n_align(
        static_cast<FT*>(out),
        static_cast<FT*>(tensor_map_->at(out_names_[0])->GetDataPtr()), m_,
        n_padded_before_, n_, cu_stream);
  }
}
AsStatus GemmA8W8GPU::Reshape(RuntimeContext* runtime_ctx) {
  AsStatus status;
  if (runtime_ctx->is_context) {
    status = Reshape();
  } else {
    status = a16w8_op_->Reshape();
  }
  return status;
}
AsStatus GemmA8W8GPU::Reshape() {
  int yn = is_npad_ ? n_padded_before_ : n_;
  AS_CHECK_STATUS(GemmA16W8Base::Reshape(yn));

  const Shape& w_shape = weights_[0]->GetShape();
  dim_t ws_size = 0;
  // input A (fp16/bf16) k padding
  ws_size += is_kpad_ ? aligned_size(m_ * k_) * sizeof(uint16_t) : size_t(0);
  // output C (fp16/bf16) n padding
  ws_size += is_npad_ ? aligned_size(m_ * n_) * sizeof(uint16_t) : size_t(0);
  // A perchannel qdata
  ws_size += aligned_size(m_ * k_) * sizeof(int8_t);
  // A scale
  ws_size += aligned_size(m_) * sizeof(float);
  // A red_max
  ws_size += aligned_size(m_) * sizeof(float);
  // A red_count
  ws_size += aligned_size(m_) * sizeof(uint32_t);
  // A red_sum
  ws_size += aligned_size(m_) * sizeof(int32_t);
  // input perchannel reordered B
  // B reorder ws
  int n_32align = (n_ + 32 - 1) / 32 * 32;
  // B qdata
  ws_size += aligned_size(n_ * k_) * sizeof(int8_t);
  // B scale
  ws_size += aligned_size(n_) * sizeof(uint16_t);
  // B zero
  ws_size += aligned_size(n_) * sizeof(uint16_t);
  // output tmp C
  ws_size += aligned_size(m_ * n_) * sizeof(int32_t);
  tensor_map_->at("workspace")->SetShape(Shape({ws_size}));
#ifdef ENABLE_CUDA
  const CUDAContext* gpu_ctx = static_cast<const CUDAContext*>(ctx_);
  cublasHandle_t cublas_handle = gpu_ctx->GetCublasHandle();

  AS_CHECK_CUBLAS(cublasSetWorkspace(
      cublas_handle, tensor_map_->at("cublas_workspace")->GetDataPtr(),
      tensor_map_->at("cublas_workspace")->GetSizeInByte()));
#endif
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus GemmA8W8GPU::Forward(RuntimeContext* runtime_ctx) {
  AsStatus status;
  if (runtime_ctx->is_context) {
    status = Forward();
  } else {
    status = a16w8_op_->Forward();
  }
  return status;
}

AsStatus GemmA8W8GPU::Forward() {
  switch (ftype_) {
#ifdef ENABLE_FP16
    case DataType::FLOAT16: {
      DispatchKernel<half, int8_t>();
      break;
    }
#endif
#ifdef ENABLE_BF16
    case DataType::BFLOAT16: {
      DispatchKernel<hie::bfloat16, int8_t>();
      break;
    }
#endif
    default:
      LOG(ERROR) << "GemmA8W8GPU DataType Error\n";
      break;
  }
  return AsStatus::ALLSPARK_SUCCESS;
}

//------------------
//------------------
template <typename FType, typename QType>
void GemmA8W8GPU::B_I8_Reorder_Hmma16816_N32_K16_Gpu(
    TensorMap& weights_buffer) {
  // Rearrange B，B scale and B zero_point to facilitate Ampere Tensor Core load
  // data reorder B from (K, N) to (N / 4, K * 4)
  if (do_padding_) {
    const CUDAContext* gpu_ctx = static_cast<const CUDAContext*>(ctx_);
    cudaStream_t cu_stream = gpu_ctx->GetStream();

    int64_t n_32align = (n_ + 32 - 1) / 32 * 32;
    void* rhs_qdata_ptr = weights_[0]->GetDataPtr();
    void* rhs_scales_ptr = weights_[1]->GetDataPtr();
    void* rhs_zeros_ptr = weights_[2]->GetDataPtr();

    // FIXME: Temporarily use cudaMallocAsync & cudaFreeAsync to bypass
    // Allocator bug, fixme later
    size_t reorder_size =
        n_32align * k_ * sizeof(QType) + n_32align * 2 * sizeof(FType);
    void* workspace;
    cudaMallocAsync(&workspace, reorder_size, cu_stream);

    int8_t* reordered_weight_ptr = reinterpret_cast<int8_t*>(workspace);
    FType* reordered_scales_ptr =
        reinterpret_cast<FType*>(reordered_weight_ptr + n_32align * k_);
    FType* reordered_zeros_ptr = reordered_scales_ptr + n_32align;

    cuda::rearrange_kn_weight_as_n32k16_order_ldg16<FType>(
        reinterpret_cast<const int8_t*>(rhs_qdata_ptr),
        reinterpret_cast<const FType*>(rhs_scales_ptr),
        reinterpret_cast<const FType*>(rhs_zeros_ptr),
        reinterpret_cast<int8_t*>(reordered_weight_ptr),
        reinterpret_cast<FType*>(reordered_scales_ptr),
        reinterpret_cast<FType*>(reordered_zeros_ptr), k_, n_, n_32align,
        cu_stream);
    gpu_ctx->Synchronize();
    auto* mutable_weight = const_cast<AsTensor*>(weights_[0]);
    mutable_weight->SetShape(Shape({n_32align, k_}));
    mutable_weight->CopyDataFrom(reordered_weight_ptr,
                                 n_32align * k_ * sizeof(QType),
                                 weights_[0]->GetDeviceType(), gpu_ctx);
    AsTensor* mutable_scales = const_cast<AsTensor*>(weights_[1]);
    mutable_scales->SetShape(Shape({n_32align}));
    mutable_scales->CopyDataFrom(reordered_scales_ptr,
                                 n_32align * sizeof(FType),
                                 weights_[1]->GetDeviceType(), gpu_ctx);
    AsTensor* mutable_zeros = const_cast<AsTensor*>(weights_[2]);
    mutable_zeros->SetShape(Shape({n_32align}));
    mutable_zeros->CopyDataFrom(reordered_zeros_ptr, n_32align * sizeof(FType),
                                weights_[2]->GetDeviceType(), gpu_ctx);
    cudaFreeAsync(workspace, cu_stream);
    gpu_ctx->Synchronize();
  }
}

template <typename FType, typename QType>
void GemmA8W8GPU::get_weight_padded_k_align(TensorMap& weights_buffer) {
  // group_size_ == -1 mean PerChannel
  const int group_cnt =
      (group_size_ == -1) ? 1 : (k_ + group_size_ - 1) / group_size_;
  int64_t k_padded =
      (k_ + A8W8_K_PAD_ALIGN - 1) / A8W8_K_PAD_ALIGN * A8W8_K_PAD_ALIGN;

  if (do_padding_) {
    AsTensor old_weight_cpu = AsTensor(*weights_[0], DeviceType::CPU);
    AsTensor rhs_zeros_cpu = AsTensor(*weights_[2], DeviceType::CPU);
    AsTensor padded_weight_cpu =
        AsTensor(weights_[0]->GetName() + "padded_cpu", DeviceType::CPU,
                 weights_[0]->GetDataType(), weights_[0]->GetDataMode(),
                 Shape({k_padded, n_}));

    // padding weight
    QType* padded_weight_ptr =
        static_cast<QType*>(padded_weight_cpu.GetDataPtr());
    QType* old_weight_ptr = static_cast<QType*>(old_weight_cpu.GetDataPtr());
    const FType* rhs_zeros_ptr =
        static_cast<FType*>(rhs_zeros_cpu.GetDataPtr());

    for (int i = 0; i < k_padded; ++i) {
      for (int j = 0; j < n_; ++j) {
        if (i < k_) {
          padded_weight_ptr[i * n_ + j] = old_weight_ptr[i * n_ + j];
        } else {
          FType zero_val = rhs_zeros_ptr[(group_cnt - 1) * n_ + j];
          padded_weight_ptr[i * n_ + j] =
              static_cast<QType>(roundf(float(zero_val)));
        }
      }
    }

    auto* mutable_weight = const_cast<AsTensor*>(weights_[0]);
    mutable_weight->SetShape(Shape({k_padded, n_}));
    TensorUtils::DeepCopyWholeAsync(*mutable_weight, padded_weight_cpu, ctx_);
    util::SyncWeightsBuffer(weights_buffer, weights_[0]->GetName(),
                            std::make_shared<AsTensor>(padded_weight_cpu));
  }

  // Reset k_, lda_
  k_ = k_padded;
  lda_ = k_;
}

template <typename FType, typename QType>
void GemmA8W8GPU::get_weight_padded_n_align(TensorMap& weights_buffer) {
  // group_size_ == -1 mean PerChannel
  const int group_cnt =
      (group_size_ == -1) ? 1 : (k_ + group_size_ - 1) / group_size_;
  int64_t n_padded =
      (n_ + A8W8_N_PAD_ALIGN - 1) / A8W8_N_PAD_ALIGN * A8W8_N_PAD_ALIGN;

  if (do_padding_) {
    AsTensor old_weight_cpu = AsTensor(*weights_[0], DeviceType::CPU);
    AsTensor old_scales_cpu = AsTensor(*weights_[1], DeviceType::CPU);
    AsTensor old_zeros_cpu = AsTensor(*weights_[2], DeviceType::CPU);
    AsTensor padded_weight_cpu =
        AsTensor(weights_[0]->GetName() + "padded_cpu", DeviceType::CPU,
                 weights_[0]->GetDataType(), weights_[0]->GetDataMode(),
                 Shape({k_, n_padded}));
    AsTensor padded_scales_cpu =
        AsTensor(weights_[1]->GetName() + "padded_cpu", DeviceType::CPU,
                 weights_[1]->GetDataType(), weights_[1]->GetDataMode(),
                 Shape({group_cnt, n_padded}));
    AsTensor padded_zeros_cpu =
        AsTensor(weights_[2]->GetName() + "padded_cpu", DeviceType::CPU,
                 weights_[2]->GetDataType(), weights_[2]->GetDataMode(),
                 Shape({group_cnt, n_padded}));

    // padding weight, scales, zeros
    QType* padded_weight_ptr =
        static_cast<QType*>(padded_weight_cpu.GetDataPtr());
    const QType* old_weight_ptr =
        static_cast<QType*>(old_weight_cpu.GetDataPtr());
    FType* padded_scales_ptr =
        static_cast<FType*>(padded_scales_cpu.GetDataPtr());
    const FType* old_scales_ptr =
        static_cast<FType*>(old_scales_cpu.GetDataPtr());
    FType* padded_zeros_ptr =
        static_cast<FType*>(padded_zeros_cpu.GetDataPtr());
    const FType* old_zeros_ptr =
        static_cast<FType*>(old_zeros_cpu.GetDataPtr());

    for (int i = 0; i < k_; ++i) {
      for (int j = 0; j < n_padded; ++j) {
        if (j < n_) {
          padded_weight_ptr[i * n_padded + j] = old_weight_ptr[i * n_ + j];
        } else {
          padded_weight_ptr[i * n_padded + j] = QType(0);
        }
      }
    }

    for (int i = 0; i < group_cnt; ++i) {
      for (int j = 0; j < n_padded; ++j) {
        if (j < n_) {
          padded_scales_ptr[i * n_padded + j] = old_scales_ptr[i * n_ + j];
          padded_zeros_ptr[i * n_padded + j] = old_zeros_ptr[i * n_ + j];
        } else {
          padded_scales_ptr[i * n_padded + j] = FType(0.f);
          padded_zeros_ptr[i * n_padded + j] = FType(0.f);
        }
      }
    }

    AsTensor* mutable_weight = const_cast<AsTensor*>(weights_[0]);
    mutable_weight->SetShape(Shape({k_, n_padded}));
    TensorUtils::DeepCopyWholeAsync(*mutable_weight, padded_weight_cpu, ctx_);

    util::SyncWeightsBuffer(weights_buffer, weights_[0]->GetName(),
                            std::make_shared<AsTensor>(padded_weight_cpu));

    AsTensor* mutable_scales = const_cast<AsTensor*>(weights_[1]);
    mutable_scales->SetShape(Shape({group_cnt, n_padded}));
    TensorUtils::DeepCopyWholeAsync(*mutable_scales, padded_scales_cpu, ctx_);
    util::SyncWeightsBuffer(weights_buffer, weights_[1]->GetName(),
                            std::make_shared<AsTensor>(padded_scales_cpu));

    AsTensor* mutable_zeros = const_cast<AsTensor*>(weights_[2]);
    mutable_zeros->SetShape(Shape({group_cnt, n_padded}));
    TensorUtils::DeepCopyWholeAsync(*mutable_zeros, padded_zeros_cpu, ctx_);
    util::SyncWeightsBuffer(weights_buffer, weights_[2]->GetName(),
                            std::make_shared<AsTensor>(padded_zeros_cpu));

    // Pad Bias
    if (weights_.size() == 4) {
      AsTensor old_bias_cpu = AsTensor(*weights_[3], DeviceType::CPU);
      AsTensor padded_bias_cpu =
          AsTensor(weights_[3]->GetName() + "padded_cpu", DeviceType::CPU,
                   weights_[3]->GetDataType(), weights_[3]->GetDataMode(),
                   Shape({n_padded}));
      FType* padded_bias_ptr =
          static_cast<FType*>(padded_bias_cpu.GetDataPtr());
      const FType* old_bias_ptr =
          static_cast<FType*>(old_bias_cpu.GetDataPtr());
      for (int i = 0; i < n_padded; ++i) {
        padded_bias_ptr[i] = i < n_ ? old_bias_ptr[i] : FType(0.f);
      }
      AsTensor* mutable_bias = const_cast<AsTensor*>(weights_[3]);
      mutable_bias->SetShape(Shape({n_padded}));
      TensorUtils::DeepCopyWholeAsync(*mutable_bias, padded_bias_cpu, ctx_);
      util::SyncWeightsBuffer(weights_buffer, weights_[3]->GetName(),
                              std::make_shared<AsTensor>(padded_bias_cpu));
    }
  }

  // Reset n_, n_padded_before_, ldb_, ldc_
  n_padded_before_ = n_;
  n_ = n_padded;
  if (!transB_) {
    ldb_ = n_;
  }
  ldc_ = n_;
}

template <typename FType, typename QType>
void GemmA8W8GPU::trans_TN(TensorMap& weights_buffer) {
  AsTensor old_weight_cpu = AsTensor(*weights_[0], DeviceType::CPU);
  AsTensor reordered_weight_cpu = AsTensor(
      weights_[0]->GetName() + "reordered_cpu", DeviceType::CPU,
      weights_[0]->GetDataType(), weights_[0]->GetDataMode(), Shape({n_, k_}));

  QType* old_weight_ptr = static_cast<QType*>(old_weight_cpu.GetDataPtr());
  QType* reordered_weight_ptr =
      static_cast<QType*>(reordered_weight_cpu.GetDataPtr());

  for (int i = 0; i < k_; ++i) {
    for (int j = 0; j < n_; ++j) {
      int src_offset = i * n_ + j;
      int dst_offset = j * k_ + i;
      reordered_weight_ptr[dst_offset] = old_weight_ptr[src_offset];
    }
  }
  auto* mutable_weight = const_cast<AsTensor*>(weights_[0]);
  mutable_weight->SetShape(Shape({n_, k_}));
  TensorUtils::DeepCopyWholeAsync(*mutable_weight, reordered_weight_cpu, ctx_);
  util::SyncWeightsBuffer(weights_buffer, weights_[0]->GetName(),
                          std::make_shared<AsTensor>(reordered_weight_cpu));
}

REGISTER_OP(GemmA8W8, CUDA, GemmA8W8GPU)
}  // namespace allspark

#endif
