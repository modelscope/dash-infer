/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    gemm_a16w4_gpu.cpp
 */

#ifdef ENABLE_CUDA

#include "gemm_a16w4_gpu.h"

#include <core/kernel/cuda/gemm_lowp/gemm_a16w4_kernel.h>
#include <core/kernel/cuda/gemm_lowp/gemm_a16w8_kernel.h>
#include <core/kernel/kernel.h>
#include <cpu/cpu_context.h>
#include <cuda/cuda_context.h>
#include <utility/datatype_dispatcher.h>

#ifdef ENABLE_BF16
#include <common/hie_bfloat16.hpp>
#endif

// N_PAD_ALIGN and K_PAD_ALIGN is at least multiples of 8
#define N_PAD_ALIGN 8
#define K_PAD_ALIGN 8
namespace allspark {

const cuda::GemmA16W4Launcher::KernelType GetA16W4KernelType(
    const DataType ftype, const DataType qtype, const uint32_t M,
    const uint32_t N, const uint32_t K, const int GroupSize,
    const int sm_version) {
  switch (ftype) {
    case DataType::FLOAT16: {
      if (qtype == DataType::UINT8) {
        return cuda::GemmA16W4Launcher::SelectKernel<half, uint8_t>()(
            M, N, K, GroupSize, sm_version);
      }
      break;
    }
    case DataType::BFLOAT16: {
      if (qtype == DataType::UINT8) {
        return cuda::GemmA16W4Launcher::SelectKernel<hie::bfloat16, uint8_t>()(
            M, N, K, GroupSize, sm_version);
      }
      break;
    }
    default:
      LOG(ERROR) << "GemmA16W4GPU DataType Error\n";
      break;
  }
  return cuda::GemmA16W4Launcher::KernelType::UNDEFINE;
}

void GemmA16W4GPU::GetWeightPaddedDispatch(const DataType ftype,
                                           const DataType qtype,
                                           TensorMap& weights_buffer) {
  switch (ftype) {
    case DataType::FLOAT16: {
      if (qtype == DataType::UINT8) {
        // Padding case 1: k % N_PAD_ALIGN != 0 and n % N_PAD_ALIGN == 0
        if (n_ % N_PAD_ALIGN == 0 && k_ % K_PAD_ALIGN != 0 && !transB_) {
          get_weight_padded_k_align<half, uint8_t>(weights_buffer);
          is_kpad_ = true;
        }
        // Padding case 2: k % N_PAD_ALIGN == 0 and n % N_PAD_ALIGN != 0
        if (n_ % N_PAD_ALIGN != 0 && k_ % K_PAD_ALIGN == 0 && !transB_) {
          get_weight_padded_n_align<half, uint8_t>(weights_buffer);
          is_npad_ = true;
        }
      }
      break;
    }
    case DataType::BFLOAT16: {
      if (qtype == DataType::UINT8) {
        // Padding case 1: k % N_PAD_ALIGN != 0 and n % N_PAD_ALIGN == 0
        if (n_ % N_PAD_ALIGN == 0 && k_ % K_PAD_ALIGN != 0 && !transB_) {
          get_weight_padded_k_align<hie::bfloat16, uint8_t>(weights_buffer);
          is_kpad_ = true;
        }
        // Padding case 2: k % N_PAD_ALIGN == 0 and n % N_PAD_ALIGN != 0
        if (n_ % N_PAD_ALIGN != 0 && k_ % K_PAD_ALIGN == 0 && !transB_) {
          get_weight_padded_n_align<hie::bfloat16, uint8_t>(weights_buffer);
          is_npad_ = true;
        }
      }
      break;
    }
    default:
      LOG(ERROR) << "GemmA16W8GPU DataType Error\n";
      break;
  }
}

AsStatus GemmA16W4GPU::InitV2(const OperatorProto& op_proto,
                              const DeviceContext& ctx,
                              const TensorMap& weights_map,
                              TensorMap& weights_buffer, TensorMap* tensor_map,
                              RuntimeContext* runtime_ctx) {
  AS_CHECK_STATUS(GemmA16W4Base::InitV2(
      op_proto, ctx, weights_map, weights_buffer, tensor_map, runtime_ctx));
  set_padding_flag(runtime_ctx);

  GetWeightPaddedDispatch(tensor_map_->at(in_names_[0])->GetDataType(), qtype_,
                          weights_buffer);

  // Get Device SM Count
  cudaDeviceProp device_prop;
  int device_id;
  cudaGetDevice(&device_id);
  cudaGetDeviceProperties(&device_prop, device_id);
  sm_count_ = device_prop.multiProcessorCount;
  sm_version_ = (device_prop.major << 8 | device_prop.minor);
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus GemmA16W4GPU::Reshape() {
  const Shape& x_shape = tensor_map_->at(in_names_[0])->GetShape();
  int x_ndims = x_shape.Size();
  m_ = x_shape.Count(0, x_ndims - 1);

  int yn = is_npad_ ? n_padded_before_ : n_;
  Shape y_shape;
  for (int i = 0; i < x_ndims - 1; ++i) {
    y_shape.Append(x_shape[i]);
  }
  y_shape.Append(yn);

  ftype_ = tensor_map_->at(in_names_[0])->GetDataType();
  tensor_map_->at(out_names_[0])->SetDataType(ftype_);
  tensor_map_->at(out_names_[0])->SetShape(std::move(y_shape));

  // Workspace
  dim_t ws_size = 0;
  ws_size += is_kpad_ ? aligned_size(m_ * k_) * sizeof(uint16_t) : size_t(0);
  ws_size += is_npad_ ? aligned_size(m_ * n_) * sizeof(uint16_t) : size_t(0);

  // TODO:
  const cuda::GemmA16W4Launcher::KernelType ktype =
      GetA16W4KernelType(ftype_, qtype_, m_, n_, k_, group_size_, sm_version_);
  ws_size += cuda::GemmA16W4Launcher::GetWorkSpaceSize(
      ktype, m_, n_, k_, group_size_, sm_count_, sm_version_, splitk_params_);
  tensor_map_->at("workspace")->SetShape(Shape({ws_size}));
  const CUDAContext* gpu_ctx = static_cast<const CUDAContext*>(ctx_);
  cublasHandle_t cublas_handle = gpu_ctx->GetCublasHandle();
  cublasSetWorkspace(cublas_handle,
                     tensor_map_->at("cublas_workspace")->GetDataPtr(),
                     tensor_map_->at("cublas_workspace")->GetSizeInByte());
  return AsStatus::ALLSPARK_SUCCESS;
}

template <typename FT, typename QT>
void GemmA16W4GPU::DispatchKernel() {
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

  const cuda::GemmA16W4Launcher::KernelType ktype =
      GetA16W4KernelType(ftype_, qtype_, m_, n_, k_, group_size_, sm_version_);
  if (ktype != cuda::GemmA16W4Launcher::KernelType::UNDEFINE) {
    // Fuse Kernel
    cuda::GemmA16W4Launcher::Run<FT, QT>(
        ktype, static_cast<const FT*>(in),
        static_cast<const QT*>(rhs_qdata_ptr),
        static_cast<const FT*>(rhs_scales_ptr),
        static_cast<const FT*>(rhs_zeros_ptr), static_cast<const FT*>(bias),
        static_cast<FT*>(out), uint32_t(m_), uint32_t(n_), uint32_t(k_),
        group_size_, static_cast<FT*>(ws_ptr), activation_, sm_count_,
        sm_version_, splitk_params_, cu_stream);
  } else {
    // DQ + Cublas
    FT* rhs_fdata_ptr = static_cast<FT*>(ws_ptr);
    cuda::dequantize_rhs_a16w4<FT, QT>(
        static_cast<const QT*>(rhs_qdata_ptr),
        static_cast<const FT*>(rhs_scales_ptr),
        static_cast<const FT*>(rhs_zeros_ptr), static_cast<FT*>(rhs_fdata_ptr),
        uint32_t(n_), uint32_t(n_pack_), uint32_t(k_), uint32_t(group_size_),
        cu_stream);
    cuda::GemmWraper(static_cast<FT*>(out), static_cast<FT*>(in),
                     static_cast<FT*>(rhs_fdata_ptr), static_cast<FT*>(bias),
                     m_, n_, k_, false, transB_, lda_, ldb_, ldc_, 1.0f, 0.0f,
                     static_cast<FT*>(nullptr), cublas_handle, cu_stream);
    if (activation_ != UNARYTYPE_UNDEFINED) {
      cuda::UnaryKernelLauncher(static_cast<FT*>(out), static_cast<FT*>(out),
                                m_ * n_ * batch_, activation_, cu_stream);
    }
  }

  // if padding n to multiples of N_PAD_ALIGN before, need to remove padding
  // here
  if (is_npad_) {
    cuda::remove_padded_n_align(
        static_cast<FT*>(out),
        static_cast<FT*>(tensor_map_->at(out_names_[0])->GetDataPtr()), m_,
        n_padded_before_, n_, cu_stream);
  }
}

AsStatus GemmA16W4GPU::Forward() {
  switch (ftype_) {
#ifdef ENABLE_FP16
    case DataType::FLOAT16: {
      DispatchKernel<half, uint8_t>();
      break;
    }
#endif
#ifdef ENABLE_BF16
    case DataType::BFLOAT16: {
      DispatchKernel<hie::bfloat16, uint8_t>();
      break;
    }
#endif
    default:
      LOG(ERROR) << "GemmA16W4GPU DataType Error\n";
      break;
  }

  // const CUDAContext* gpu_ctx = static_cast<const CUDAContext*>(ctx_);
  // cublasHandle_t cublas_handle = gpu_ctx->GetCublasHandle();
  // cudaStream_t cu_stream = gpu_ctx->GetStream();
  // cudaStreamSynchronize(cu_stream);

  // AsTensor* in_tensor = tensor_map_->at(in_names_[0]).get();
  // std::cout << "\nShape: m: " << m_ << " n: " << n_ << " k: " << k_ << "
  // n_pack: " << n_pack_ << "\n"
  //     << "[I] " << in_tensor->ToString() << "\n"
  //     << "[W] " << weights_[0]->ToString() << "\n"
  //     << "[S] " << weights_[1]->ToString() << "\n"
  //     << "[Z] " << weights_[2]->ToString() << "\n";

  return AsStatus::ALLSPARK_SUCCESS;
}

template <typename FType, typename QType>
void GemmA16W4GPU::get_weight_padded_k_align(TensorMap& weights_buffer) {
  // group_size_ == -1 mean PerChannel
  const int group_cnt =
      (group_size_ == -1) ? 1 : (k_ + group_size_ - 1) / group_size_;
  int64_t k_padded = (k_ + K_PAD_ALIGN - 1) / K_PAD_ALIGN * K_PAD_ALIGN;

  if (do_padding_) {
    AsTensor old_weight_cpu = AsTensor(*weights_[0], DeviceType::CPU);
    AsTensor rhs_zeros_cpu = AsTensor(*weights_[2], DeviceType::CPU);
    AsTensor padded_weight_cpu =
        AsTensor(weights_[0]->GetName() + "padded_cpu", DeviceType::CPU,
                 weights_[0]->GetDataType(), weights_[0]->GetDataMode(),
                 Shape({k_padded, n_pack_}));

    // padding weight
    QType* padded_weight_ptr =
        static_cast<QType*>(padded_weight_cpu.GetDataPtr());
    QType* old_weight_ptr = static_cast<QType*>(old_weight_cpu.GetDataPtr());
    const FType* rhs_zeros_ptr =
        static_cast<FType*>(rhs_zeros_cpu.GetDataPtr());

    for (int i = 0; i < k_padded; ++i) {
      for (int j = 0; j < n_pack_; ++j) {
        if (i < k_) {
          padded_weight_ptr[i * n_pack_ + j] = old_weight_ptr[i * n_pack_ + j];
        } else {
          FType zero_fval_0 = rhs_zeros_ptr[(group_cnt - 1) * n_ + j * 2];
          FType zero_fval_1 =
              (j * 2 + 1) < n_ ? rhs_zeros_ptr[(group_cnt - 1) * n_ + j * 2 + 1]
                               : FType(0.f);
          QType zero_qval_0 = static_cast<QType>(roundf(float(zero_fval_0)));
          QType zero_qval_1 = static_cast<QType>(roundf(float(zero_fval_1)));
          padded_weight_ptr[i * n_pack_ + j] =
              (zero_qval_0 & 0xf) | (zero_qval_1 << 4);
        }
      }
    }

    auto* mutable_weight = const_cast<AsTensor*>(weights_[0]);
    mutable_weight->SetShape(Shape({k_padded, n_pack_}));
    TensorUtils::DeepCopyWholeAsync(*mutable_weight, padded_weight_cpu, ctx_);
    util::SyncWeightsBuffer(weights_buffer, weights_[0]->GetName(),
                            std::make_shared<AsTensor>(padded_weight_cpu));
  }
  // Reset k_, lda_
  k_ = k_padded;
  lda_ = k_;
}

template <typename FType, typename QType>
void GemmA16W4GPU::get_weight_padded_n_align(TensorMap& weights_buffer) {
  // group_size_ == -1 mean PerChannel
  const int group_cnt =
      (group_size_ == -1) ? 1 : (k_ + group_size_ - 1) / group_size_;
  int64_t n_padded = (n_ + N_PAD_ALIGN - 1) / N_PAD_ALIGN * N_PAD_ALIGN;

  if (do_padding_) {
    AsTensor old_weight_cpu = AsTensor(*weights_[0], DeviceType::CPU);
    AsTensor old_scales_cpu = AsTensor(*weights_[1], DeviceType::CPU);
    AsTensor old_zeros_cpu = AsTensor(*weights_[2], DeviceType::CPU);
    AsTensor padded_weight_cpu =
        AsTensor(weights_[0]->GetName() + "padded_cpu", DeviceType::CPU,
                 weights_[0]->GetDataType(), weights_[0]->GetDataMode(),
                 Shape({k_, n_padded / 2}));
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
      for (int j = 0; j < n_padded / 2; ++j) {
        if (j < n_pack_) {
          padded_weight_ptr[i * (n_padded / 2) + j] =
              old_weight_ptr[i * n_pack_ + j];
        } else {
          padded_weight_ptr[i * (n_padded / 2) + j] = QType(0);
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
    mutable_weight->SetShape(Shape({k_, n_padded / 2}));
    TensorUtils::DeepCopyWholeAsync(*mutable_weight, padded_weight_cpu, ctx_);

    util::SyncWeightsBuffer(weights_buffer, weights_[0]->GetName(),
                            std::make_shared<AsTensor>(padded_weight_cpu));

    AsTensor* mutable_group_cnt = const_cast<AsTensor*>(weights_[1]);
    mutable_group_cnt->SetShape(Shape({group_cnt, n_padded}));
    TensorUtils::DeepCopyWholeAsync(*mutable_group_cnt, padded_scales_cpu,
                                    ctx_);
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

  // Reset n_, n_padded_before_, n_pack_, ldb_, ldc_
  n_padded_before_ = n_;
  n_ = n_padded;
  n_pack_ = n_ / 2;
  if (!transB_) {
    ldb_ = n_;
  }
  ldc_ = n_;
}

REGISTER_OP(GemmA16W4, CUDA, GemmA16W4GPU)

}  // namespace allspark

#endif
