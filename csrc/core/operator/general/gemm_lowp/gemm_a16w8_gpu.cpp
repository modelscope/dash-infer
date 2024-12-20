/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    gemm_a16w8_gpu.cpp
 */

#ifdef ENABLE_CUDA
#include "gemm_a16w8_gpu.h"

#include <core/kernel/kernel.h>
#include <cpu/cpu_context.h>
#include <cuda/cuda_context.h>
#include <utility/datatype_dispatcher.h>

#include "gemm_a8w8_gpu.h"

#ifdef ENABLE_BF16
#include <common/hie_bfloat16.hpp>
#endif
#define A16W8_N_PAD_ALIGN 16
#define A16W8_K_PAD_ALIGN 16
namespace allspark {
AsStatus GemmA16W8GPU::Init(const OperatorProto& op_proto,
                            const DeviceContext& ctx,
                            const TensorMap& weights_map,
                            TensorMap* tensor_map) {
  LOG(ERROR) << "GemmA16W8GPU only support InitV2()" << std::endl;
  return AsStatus::ALLSPARK_INVALID_CALL_ERROR;
}

AsStatus GemmA16W8GPU::InitV2(const OperatorProto& op_proto,
                              const DeviceContext& ctx,
                              const TensorMap& weights_map,
                              TensorMap& weights_buffer,
                              TensorMap* tensor_map) {
  AS_CHECK_STATUS(GemmA16W8Base::InitV2(op_proto, ctx, weights_map,
                                        weights_buffer, tensor_map));

  // Get Device SM Count
  cudaDeviceProp device_prop;
  int device_id;
  cudaGetDevice(&device_id);
  cudaGetDeviceProperties(&device_prop, device_id);
  sm_count_ = device_prop.multiProcessorCount;
  sm_version_ = (device_prop.major << 8 | device_prop.minor);
  // it is not necessary to pad weight since it is already padded in a8w8
  if (!ctx_->GetFallbackDecoderWeightOnly()) {
    GetWeightPaddedDispatch(tensor_map_->at(in_names_[0])->GetDataType(),
                            qtype_, weights_buffer);
  }
  return AsStatus::ALLSPARK_SUCCESS;
}

const cuda::GemmA16W8Launcher::KernelType GetA16W8KernelType(
    const DataType ftype, const DataType qtype, const uint32_t M,
    const uint32_t N, const uint32_t K, const int GroupSize, const int sm_count,
    const int sm_version, SplitKParams& splitk_params) {
  // TODO: uint8
  switch (ftype) {
    case DataType::FLOAT16: {
      if (qtype == DataType::INT8) {
        return cuda::GemmA16W8Launcher::SelectKernel<half, int8_t>()(
            M, N, K, GroupSize, sm_count, sm_version, splitk_params);
      }
      break;
    }
    case DataType::BFLOAT16: {
      if (qtype == DataType::INT8) {
        return cuda::GemmA16W8Launcher::SelectKernel<hie::bfloat16, int8_t>()(
            M, N, K, GroupSize, sm_count, sm_version, splitk_params);
      }
      break;
    }
    default:
      LOG(ERROR) << "GemmA16W8GPU DataType Error\n";
      break;
  }
  return cuda::GemmA16W8Launcher::KernelType::UNDEFINE;
}

void GemmA16W8GPU::GetWeightPaddedDispatch(const DataType ftype,
                                           const DataType qtype,
                                           TensorMap& weights_buffer) {
  // TODO: uint8
  switch (ftype) {
    case DataType::FLOAT16: {
      if (qtype == DataType::INT8) {
        // Padding case 1: k % A16W8_K_PAD_ALIGN != 0 and n % A16W8_N_PAD_ALIGN
        // == 0
        if (n_ % A16W8_N_PAD_ALIGN == 0 && k_ % A16W8_K_PAD_ALIGN != 0 &&
            !transB_) {
          get_weight_padded_k_align<half, int8_t>(weights_buffer);
          is_kpad_ = true;
        }
        // Padding case 2: k % A16W8_K_PAD_ALIGN == 0 and n % A16W8_N_PAD_ALIGN
        // != 0
        if (n_ % A16W8_N_PAD_ALIGN != 0 && k_ % A16W8_K_PAD_ALIGN == 0 &&
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
        // Padding case 1: k % A16W8_K_PAD_ALIGN != 0 and n % A16W8_N_PAD_ALIGN
        // == 0
        if (n_ % A16W8_N_PAD_ALIGN == 0 && k_ % A16W8_K_PAD_ALIGN != 0 &&
            !transB_) {
          get_weight_padded_k_align<hie::bfloat16, int8_t>(weights_buffer);
          is_kpad_ = true;
        }
        // Padding case 2: k % A16W8_N_PAD_ALIGN == 0 and n % A16W8_N_PAD_ALIGN
        // != 0
        if (n_ % A16W8_N_PAD_ALIGN != 0 && k_ % A16W8_K_PAD_ALIGN == 0 &&
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
      LOG(ERROR) << "GemmA16W8GPU DataType Error\n";
      break;
  }
}

AsStatus GemmA16W8GPU::Reshape() {
  int yn = is_npad_ ? n_padded_before_ : n_;
  AS_CHECK_STATUS(GemmA16W8Base::Reshape(yn));

  const Shape& w_shape = weights_[0]->GetShape();
  dim_t ws_size = 0;
  ws_size += is_kpad_ ? aligned_size(m_ * k_) * sizeof(uint16_t) : size_t(0);
  ws_size += is_npad_ ? aligned_size(m_ * n_) * sizeof(uint16_t) : size_t(0);

  ktype = GetA16W8KernelType(ftype_, qtype_, m_, n_, k_, group_size_, sm_count_,
                             sm_version_, splitk_params_);
  if (ktype != cuda::GemmA16W8Launcher::KernelType::UNDEFINE) {
    ws_size += cuda::GemmA16W8Launcher::GetWorkSpaceSize(
        ktype, m_, n_, k_, group_size_, sm_count_, sm_version_, splitk_params_);
  } else {
    ws_size += w_shape.Count() * sizeof(uint16_t);
  }
  tensor_map_->at("workspace")->SetShape(Shape({ws_size}));
  const CUDAContext* gpu_ctx = static_cast<const CUDAContext*>(ctx_);
  cublasHandle_t cublas_handle = gpu_ctx->GetCublasHandle();
  cublasSetWorkspace(cublas_handle,
                     tensor_map_->at("cublas_workspace")->GetDataPtr(),
                     tensor_map_->at("cublas_workspace")->GetSizeInByte());
  return AsStatus::ALLSPARK_SUCCESS;
}

template <typename FT, typename QT>
void GemmA16W8GPU::DispatchKernel() {
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

  if (ktype != cuda::GemmA16W8Launcher::KernelType::UNDEFINE) {
    cuda::GemmA16W8Launcher::Run<FT, QT>(
        ktype, static_cast<const FT*>(in),
        static_cast<const QT*>(rhs_qdata_ptr),
        static_cast<const FT*>(rhs_scales_ptr),
        static_cast<const FT*>(rhs_zeros_ptr), static_cast<const FT*>(bias),
        static_cast<FT*>(out), uint32_t(m_), uint32_t(n_), uint32_t(k_),
        group_size_, static_cast<FT*>(ws_ptr), activation_, sm_count_,
        sm_version_, splitk_params_, alpha_, cu_stream);
  } else {
    FT* rhs_fdata_ptr = static_cast<FT*>(ws_ptr);
    // if use dq+cublas, need to restore weight from N32K16 order to original
    // order
    if (sm_version_ >= 0x0800 && group_size_ == -1) {
      const int n_32align = (n_ + 32 - 1) / 32 * 32;
      cuda::restore_N32_K16_dequantize_rhs_a16w8<FT, QT>(
          static_cast<const QT*>(rhs_qdata_ptr),
          static_cast<const FT*>(rhs_scales_ptr),
          static_cast<const FT*>(rhs_zeros_ptr), rhs_fdata_ptr, n_32align, n_,
          k_, group_size_, cu_stream);
    } else {
      cuda::dequantize_rhs_a16w8<FT, QT>(static_cast<const QT*>(rhs_qdata_ptr),
                                         static_cast<const FT*>(rhs_scales_ptr),
                                         static_cast<const FT*>(rhs_zeros_ptr),
                                         rhs_fdata_ptr, uint32_t(n_),
                                         uint32_t(k_), group_size_, cu_stream);
    }

    cuda::GemmWraper(static_cast<FT*>(out), static_cast<FT*>(in), rhs_fdata_ptr,
                     static_cast<FT*>(bias), m_, n_, k_, false, transB_, lda_,
                     ldb_, ldc_, alpha_, 0.0f, static_cast<FT*>(nullptr),
                     cublas_handle, cu_stream);
    if (activation_ != UNARYTYPE_UNDEFINED) {
      cuda::UnaryKernelLauncher(static_cast<FT*>(out), static_cast<FT*>(out),
                                m_ * n_ * batch_, activation_, cu_stream);
    }
  }

  // if padding n to multiples of A16W8_N_PAD_ALIGN before, need to remove
  // padding here
  if (is_npad_) {
    cuda::remove_padded_n_align(
        static_cast<FT*>(out),
        static_cast<FT*>(tensor_map_->at(out_names_[0])->GetDataPtr()), m_,
        n_padded_before_, n_, cu_stream);
  }
  return;
}

AsStatus GemmA16W8GPU::Forward() {
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
      LOG(ERROR) << "GemmA16W8GPU DataType Error\n";
      break;
  }

  // cudaStreamSynchronize(cu_stream);
  // if (weights_.size() == 4) {
  //     std::cout << "\nShape: " << m_ << " " << n_ << " " << k_ << "\n"
  //         << "Act: " << activation_ << "\n"
  //         << "GemmA16W8GPU Kernel : " <<
  //         cuda::GemmA16W8Launcher::SelectKernel(m_, n_, k_,
  //         group_size_, sm_version_) << "\n"
  //         << "[I] " << in_tensor->ToString() << "\n"
  //         << "[W] " << weights_[0]->ToString() << "\n"
  //         << "[S] " << weights_[1]->ToString() << "\n"
  //         << "[Z] " << weights_[2]->ToString() << "\n"
  //         << "[B] " << weights_[3]->ToString() << "\n"
  //         << "[O] " << tensor_map_->at(out_names_[0])->ToString() <<
  //         "\n";
  // }
  // else {
  //     std::cout << "\nShape: " << m_ << " " << n_ << " " << k_ << "\n"
  //         << "Act: " << activation_ << "\n"
  //         << "GemmA16W8GPU Kernel : " <<
  //         cuda::GemmA16W8Launcher::SelectKernel(m_, n_, k_,
  //         group_size_, sm_version_) << "\n"
  //         << "[I] " << in_tensor->ToString() << "\n"
  //         << "[W] " << weights_[0]->ToString() << "\n"
  //         << "[S] " << weights_[1]->ToString() << "\n"
  //         << "[Z] " << weights_[2]->ToString() << "\n"
  //         << "[O] " << tensor_map_->at(out_names_[0])->ToString() <<
  //         "\n";
  // }

  return AsStatus::ALLSPARK_SUCCESS;
}

//------------------
//------------------
template <typename FType, typename QType>
void GemmA16W8GPU::B_I8_Reorder_Hmma16816_N32_K16_Cpu(
    TensorMap& weights_buffer) {
  int64_t real_n = n_;
  int64_t real_k = k_;
  if (n_ % 8 != 0) {
    int64_t n_padded = (n_ + 8 - 1) / 8 * 8;
    n_padded_before_ = n_;
    n_ = n_padded;
    is_npad_ = true;
    if (!transB_) {
      ldb_ = n_;
    }
    ldc_ = n_;
  }
  if (k_ % 16 != 0) {
    int64_t k_padded = (k_ + 16 - 1) / 16 * 16;
    k_ = k_padded;
    is_kpad_ = true;
    lda_ = k_;
  }
  // Rearrange B to facilitate Ampere Tensor Core load data
  // implicit requirement of B is N % 32 == 0， K % 16 == 0
  // reorder B from (K, N) to (N / 4, K * 4)
  int64_t n_32align = (n_ + 32 - 1) / 32 * 32;
  AsTensor old_weight_cpu = AsTensor(*weights_[0], DeviceType::CPU);
  AsTensor reordered_weight_cpu =
      AsTensor(weights_[0]->GetName() + "reordered_cpu", DeviceType::CPU,
               weights_[0]->GetDataType(), weights_[0]->GetDataMode(),
               Shape({n_32align, k_}));

  QType* old_weight_ptr = static_cast<QType*>(old_weight_cpu.GetDataPtr());
  QType* reordered_weight_ptr =
      static_cast<QType*>(reordered_weight_cpu.GetDataPtr());

  for (int i = 0; i < n_32align; ++i) {
    for (int j = 0; j < k_; ++j) {
      int dst_row = (i / 32) * 8 + (i % 8);
      int dst_col = (j / 16) * 16 * 4 + ((j % 8) / 2) * 16 +
                    ((j % 16) / 8) * 2 + ((i % 32) / 8) * 4 + j % 2;
      int dst_offset = dst_row * k_ * 4 + dst_col;
      int src_offset = i + j * real_n;
      if (i < real_n && j < real_k) {
        reordered_weight_ptr[dst_offset] = old_weight_ptr[src_offset];
      } else {
        reordered_weight_ptr[dst_offset] = QType(0);
      }
    }
  }
  auto* mutable_weight = const_cast<AsTensor*>(weights_[0]);
  mutable_weight->SetShape(Shape({n_32align, k_}));
  TensorUtils::DeepCopyWholeAsync(*mutable_weight, reordered_weight_cpu, ctx_);
  util::SyncWeightsBuffer(weights_buffer, weights_[0]->GetName(),
                          std::make_shared<AsTensor>(reordered_weight_cpu));
  // Accordingly, rearrange B scale and B zero_point
  // Now only support Per Channel
  AsTensor old_scales_cpu = AsTensor(*weights_[1], DeviceType::CPU);
  AsTensor old_zeros_cpu = AsTensor(*weights_[2], DeviceType::CPU);

  AsTensor reordered_scales_cpu =
      AsTensor(weights_[1]->GetName() + "reordered_cpu", DeviceType::CPU,
               weights_[1]->GetDataType(), weights_[1]->GetDataMode(),
               Shape({n_32align}));
  AsTensor reordered_zeros_cpu =
      AsTensor(weights_[2]->GetName() + "reordered_cpu", DeviceType::CPU,
               weights_[2]->GetDataType(), weights_[2]->GetDataMode(),
               Shape({n_32align}));

  FType* reordered_scales_ptr =
      static_cast<FType*>(reordered_scales_cpu.GetDataPtr());
  const FType* old_scales_ptr =
      static_cast<FType*>(old_scales_cpu.GetDataPtr());
  FType* reordered_zeros_ptr =
      static_cast<FType*>(reordered_zeros_cpu.GetDataPtr());
  const FType* old_zeros_ptr = static_cast<FType*>(old_zeros_cpu.GetDataPtr());

  for (int i = 0; i < n_32align; ++i) {
    int dst_offset = (i / 32) * 32 + (i % 32) / 8 + ((i % 32) % 8) * 4;
    if (i < real_n) {
      reordered_scales_ptr[dst_offset] = old_scales_ptr[i];
      reordered_zeros_ptr[dst_offset] = old_zeros_ptr[i];
    } else {
      reordered_scales_ptr[dst_offset] = FType(0.f);
      reordered_zeros_ptr[dst_offset] = FType(0.f);
    }
  }

  AsTensor* mutable_scales = const_cast<AsTensor*>(weights_[1]);
  mutable_scales->SetShape(Shape({n_32align}));
  TensorUtils::DeepCopyWholeAsync(*mutable_scales, reordered_scales_cpu, ctx_);
  util::SyncWeightsBuffer(weights_buffer, weights_[1]->GetName(),
                          std::make_shared<AsTensor>(reordered_scales_cpu));

  AsTensor* mutable_zeros = const_cast<AsTensor*>(weights_[2]);
  mutable_zeros->SetShape(Shape({n_32align}));
  TensorUtils::DeepCopyWholeAsync(*mutable_zeros, reordered_zeros_cpu, ctx_);
  util::SyncWeightsBuffer(weights_buffer, weights_[2]->GetName(),
                          std::make_shared<AsTensor>(reordered_zeros_cpu));
  // if is_npad_, padding bias
  if (weights_.size() == 4 && is_npad_) {
    AsTensor old_bias_cpu = AsTensor(*weights_[3], DeviceType::CPU);
    AsTensor padded_bias_cpu = AsTensor(
        weights_[3]->GetName() + "padded_cpu", DeviceType::CPU,
        weights_[3]->GetDataType(), weights_[3]->GetDataMode(), Shape({n_}));
    FType* padded_bias_ptr = static_cast<FType*>(padded_bias_cpu.GetDataPtr());
    const FType* old_bias_ptr = static_cast<FType*>(old_bias_cpu.GetDataPtr());

    for (int i = 0; i < n_; ++i) {
      padded_bias_ptr[i] = i < real_n ? old_bias_ptr[i] : FType(0.f);
    }
    AsTensor* mutable_bias = const_cast<AsTensor*>(weights_[3]);
    mutable_bias->SetShape(Shape({n_}));
    TensorUtils::DeepCopyWholeAsync(*mutable_bias, padded_bias_cpu, ctx_);
    util::SyncWeightsBuffer(weights_buffer, weights_[3]->GetName(),
                            std::make_shared<AsTensor>(padded_bias_cpu));
  }
}

template <typename FType, typename QType>
void GemmA16W8GPU::B_I8_Reorder_Hmma16816_N32_K16_Gpu(
    TensorMap& weights_buffer) {
  // Rearrange B，B scale and B zero_point to facilitate Ampere Tensor Core load
  // data reorder B from (K, N) to (N / 4, K * 4)
  const CUDAContext* gpu_ctx = static_cast<const CUDAContext*>(ctx_);
  cudaStream_t cu_stream = gpu_ctx->GetStream();

  int64_t n_32align = (n_ + 32 - 1) / 32 * 32;
  void* rhs_qdata_ptr = weights_[0]->GetDataPtr();
  void* rhs_scales_ptr = weights_[1]->GetDataPtr();
  void* rhs_zeros_ptr = weights_[2]->GetDataPtr();

  // FIXME: Temporarily use cudaMallocAsync & cudaFreeAsync to bypass Allocator
  // bug, fixme later
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
  mutable_scales->CopyDataFrom(reordered_scales_ptr, n_32align * sizeof(FType),
                               weights_[1]->GetDeviceType(), gpu_ctx);
  AsTensor* mutable_zeros = const_cast<AsTensor*>(weights_[2]);
  mutable_zeros->SetShape(Shape({n_32align}));
  mutable_zeros->CopyDataFrom(reordered_zeros_ptr, n_32align * sizeof(FType),
                              weights_[2]->GetDeviceType(), gpu_ctx);
  cudaFreeAsync(workspace, cu_stream);
  gpu_ctx->Synchronize();
}

template <typename FType, typename QType>
void GemmA16W8GPU::get_weight_padded_k_align(TensorMap& weights_buffer) {
  // group_size_ == -1 mean PerChannel
  const int group_cnt =
      (group_size_ == -1) ? 1 : (k_ + group_size_ - 1) / group_size_;
  int64_t k_padded =
      (k_ + A16W8_K_PAD_ALIGN - 1) / A16W8_K_PAD_ALIGN * A16W8_K_PAD_ALIGN;

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
  const FType* rhs_zeros_ptr = static_cast<FType*>(rhs_zeros_cpu.GetDataPtr());

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
  // Reset k_, lda_
  k_ = k_padded;
  lda_ = k_;
}

template <typename FType, typename QType>
void GemmA16W8GPU::get_weight_padded_n_align(TensorMap& weights_buffer) {
  // group_size_ == -1 mean PerChannel
  const int group_cnt =
      (group_size_ == -1) ? 1 : (k_ + group_size_ - 1) / group_size_;
  int64_t n_padded =
      (n_ + A16W8_N_PAD_ALIGN - 1) / A16W8_N_PAD_ALIGN * A16W8_N_PAD_ALIGN;

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
  FType* padded_zeros_ptr = static_cast<FType*>(padded_zeros_cpu.GetDataPtr());
  const FType* old_zeros_ptr = static_cast<FType*>(old_zeros_cpu.GetDataPtr());

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
    FType* padded_bias_ptr = static_cast<FType*>(padded_bias_cpu.GetDataPtr());
    const FType* old_bias_ptr = static_cast<FType*>(old_bias_cpu.GetDataPtr());
    for (int i = 0; i < n_padded; ++i) {
      padded_bias_ptr[i] = i < n_ ? old_bias_ptr[i] : FType(0.f);
    }
    AsTensor* mutable_bias = const_cast<AsTensor*>(weights_[3]);
    mutable_bias->SetShape(Shape({n_padded}));
    TensorUtils::DeepCopyWholeAsync(*mutable_bias, padded_bias_cpu, ctx_);
    util::SyncWeightsBuffer(weights_buffer, weights_[3]->GetName(),
                            std::make_shared<AsTensor>(padded_bias_cpu));
  }

  // Reset n_, n_padded_before_, ldb_, ldc_
  n_padded_before_ = n_;
  n_ = n_padded;
  if (!transB_) {
    ldb_ = n_;
  }
  ldc_ = n_;
}

REGISTER_OP(GemmA16W8, CUDA, GemmA16W8GPU)
}  // namespace allspark

#endif
