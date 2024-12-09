/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    gemm_sparse_a8w8_gpu.cpp
 */

#ifdef ENABLE_CUSPARSELT
#include "gemm_sparse_a8w8_gpu.h"

#include <core/kernel/kernel.h>
#include <cpu/cpu_context.h>
#include <cuda/cuda_context.h>
#include <utility/datatype_dispatcher.h>

#include "gemm_a8w8_gpu.h"

#ifdef ENABLE_BF16
#include <common/hie_bfloat16.hpp>
#endif

#define A8W8_N_PAD_ALIGN 16
#define A8W8_K_PAD_ALIGN 16
namespace allspark {

AsStatus GemmSparseA8W8GPU::Init(const OperatorProto& op_proto,
                                 const DeviceContext& ctx,
                                 const TensorMap& weights_map,
                                 TensorMap* tensor_map) {
  LOG(ERROR) << "GemmSparseA8W8GPU only support InitV2()" << std::endl;
  return AsStatus::ALLSPARK_INVALID_CALL_ERROR;
}

AsStatus GemmSparseA8W8GPU::InitV2(const OperatorProto& op_proto,
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
  trans_TN<half, int8_t>(weights_buffer);

  GetWeightPaddedDispatch(tensor_map_->at(in_names_[0])->GetDataType(), qtype_,
                          weights_buffer);

  bool enable_sparse_opt = ctx.GetSparsityMatmulMode();
  if (enable_sparse_opt) {
    cusparseLtMatDescriptor_t matA, matB, matC;
    cusparseLtMatmulDescriptor_t matmul;
    cusparseLtMatmulAlgSelection_t alg_sel;
    cusparseLtMatmulPlan_t plan;
    // Check if weight is 2:4 sparse format
    bool check_sparse = IsSparseWeight(matA, matB, matC, matmul, alg_sel, plan);
    if (check_sparse) {
      DLOG(INFO) << "Found Sparse Matrix, compressing weight "
                 << weights_[0]->GetName();
      is_sparse = true;
      CompressWeightAndSearch(matA, matB, matC, matmul, alg_sel, plan);
    }

    // destroy plan and desc, memory deallocation
    AS_CHECK_CUSPARSE(cusparseLtMatDescriptorDestroy(&matA));
    AS_CHECK_CUSPARSE(cusparseLtMatDescriptorDestroy(&matB));
    AS_CHECK_CUSPARSE(cusparseLtMatDescriptorDestroy(&matC));
    AS_CHECK_CUSPARSE(cusparseLtMatmulPlanDestroy(&plan));
  }
  return AsStatus::ALLSPARK_SUCCESS;
}

void GemmSparseA8W8GPU::InitCuSparseDescAlgPlan(
    cusparseLtHandle_t& cslt_handle, cusparseLtMatDescriptor_t& matA,
    cusparseLtMatDescriptor_t& matB, cusparseLtMatDescriptor_t& matC,
    cusparseLtMatmulDescriptor_t& matmul,
    cusparseLtMatmulAlgSelection_t& alg_sel, cusparseLtMatmulPlan_t& plan,
    const int M, const int N, const int K) {
  cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
  cusparseOperation_t opB = CUSPARSE_OPERATION_TRANSPOSE;

  const int alignment = 32;
  // matrix descriptor initialization
  AS_CHECK_CUSPARSE(cusparseLtStructuredDescriptorInit(
      &cslt_handle, &matA, N, K, K,  // N, K must be a multiple of 32
      alignment, CUDA_R_8I, CUSPARSE_ORDER_ROW,
      CUSPARSELT_SPARSITY_50_PERCENT));
  AS_CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(
      &cslt_handle, &matB, M, K, K,  // M, K must be a multiple of 16
      alignment, CUDA_R_8I, CUSPARSE_ORDER_ROW));
  AS_CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(
      &cslt_handle, &matC, N, M, N, alignment, CUDA_R_32I, CUSPARSE_ORDER_COL));

  // matmul, algorithm selection, and plan initialization
  AS_CHECK_CUSPARSE(cusparseLtMatmulDescriptorInit(
      &cslt_handle, &matmul, opA, opB, &matA, &matB, &matC, &matC,
      CUSPARSE_COMPUTE_32I));

  AS_CHECK_CUSPARSE(cusparseLtMatmulAlgSelectionInit(
      &cslt_handle, &alg_sel, &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT));

  AS_CHECK_CUSPARSE(
      cusparseLtMatmulPlanInit(&cslt_handle, &plan, &matmul, &alg_sel));
}

/*
check if the weight format conforms to the 2:4 sparse pattern
*/
bool GemmSparseA8W8GPU::IsSparseWeight(cusparseLtMatDescriptor_t& matA,
                                       cusparseLtMatDescriptor_t& matB,
                                       cusparseLtMatDescriptor_t& matC,
                                       cusparseLtMatmulDescriptor_t& matmul,
                                       cusparseLtMatmulAlgSelection_t& alg_sel,
                                       cusparseLtMatmulPlan_t& plan) {
  const CUDAContext* gpu_ctx = static_cast<const CUDAContext*>(ctx_);
  cudaStream_t stream = gpu_ctx->GetStream();
  int8_t* dA = static_cast<int8_t*>(weights_[0]->GetDataPtr());

  const int M_large = 4096;
  cusparseLtHandle_t cslt_handle = gpu_ctx->GetCuSparseHandle();
  InitCuSparseDescAlgPlan(cslt_handle, matA, matB, matC, matmul, alg_sel, plan,
                          M_large, aligned_size(n_), aligned_size(k_));
  AS_CHECK_CUDA(cudaMalloc((void**)&d_valid, sizeof(int)));
  AS_CHECK_CUSPARSE(
      cusparseLtSpMMAPruneCheck(&cslt_handle, &matmul, dA, d_valid, stream));
  int is_valid;
  AS_CHECK_CUDA(cudaMemcpyAsync(&is_valid, d_valid, sizeof(int),
                                cudaMemcpyDeviceToHost, stream));
  AS_CHECK_CUDA(cudaStreamSynchronize(stream));
  return is_valid == 0 ? true : false;
}

void GemmSparseA8W8GPU::CompressWeightAndSearch(
    cusparseLtMatDescriptor_t& matA, cusparseLtMatDescriptor_t& matB,
    cusparseLtMatDescriptor_t& matC, cusparseLtMatmulDescriptor_t& matmul,
    cusparseLtMatmulAlgSelection_t& alg_sel, cusparseLtMatmulPlan_t& plan) {
  const int M_large = 4096;
  int8_t* dA = reinterpret_cast<int8_t*>(weights_[0]->GetDataPtr());

  const CUDAContext* gpu_ctx = static_cast<const CUDAContext*>(ctx_);
  cudaStream_t stream = gpu_ctx->GetStream();
  cusparseLtHandle_t cslt_handle = gpu_ctx->GetCuSparseHandle();

  InitCuSparseDescAlgPlan(cslt_handle, matA, matB, matC, matmul, alg_sel, plan,
                          M_large, aligned_size(n_), aligned_size(k_));
  size_t compressed_size, compressed_buffer_size;
  AS_CHECK_CUSPARSE(cusparseLtSpMMACompressedSize(
      &cslt_handle, &plan, &compressed_size, &compressed_buffer_size));

  AsTensor compressed_weight =
      AsTensor("compressed_weight", DeviceType::CUDA, DataType::INT8,
               DataMode::DENSE, Shape({compressed_size}));
  AsTensor qweight_compressedBuffer =
      AsTensor("qweight_compressedBuffer", DeviceType::CUDA, DataType::INT8,
               DataMode::DENSE, Shape({compressed_buffer_size}));

  AS_CHECK_CUSPARSE(cusparseLtSpMMACompress(
      &cslt_handle, &plan, dA,
      static_cast<int8_t*>(compressed_weight.GetDataPtr()),
      static_cast<int8_t*>(qweight_compressedBuffer.GetDataPtr()), stream));

  AsTensor dC = AsTensor("dC", DeviceType::CUDA, DataType::INT32,
                         DataMode::DENSE, Shape({M_large * aligned_size(n_)}));

  float alpha = 1.0f;
  float beta = 1.0f;
  int num_streams = 1;
  AsTensor dummy_act =
      AsTensor("dummy_act", DeviceType::CUDA, DataType::INT8, DataMode::DENSE,
               Shape({M_large, aligned_size(k_)}));
  const int8_t* dB = reinterpret_cast<const int8_t*>(dummy_act.GetDataPtr());
  int8_t* dA_compressed =
      reinterpret_cast<int8_t*>(compressed_weight.GetDataPtr());
  AS_CHECK_CUSPARSE(cusparseLtMatmulSearch(
      &cslt_handle, &plan, &alpha, dA_compressed, dB, &beta,
      static_cast<int32_t*>(dC.GetDataPtr()),
      static_cast<int32_t*>(dC.GetDataPtr()), nullptr, &stream, num_streams));
  int config_id, split_k, split_k_mode, split_k_buffers;
  AS_CHECK_CUSPARSE(cusparseLtMatmulAlgGetAttribute(
      &cslt_handle, &alg_sel, CUSPARSELT_MATMUL_ALG_CONFIG_ID, &config_id,
      sizeof(config_id)));
  AS_CHECK_CUSPARSE(cusparseLtMatmulAlgGetAttribute(&cslt_handle, &alg_sel,
                                                    CUSPARSELT_MATMUL_SPLIT_K,
                                                    &split_k, sizeof(split_k)));
  AS_CHECK_CUSPARSE(cusparseLtMatmulAlgGetAttribute(
      &cslt_handle, &alg_sel, CUSPARSELT_MATMUL_SPLIT_K_MODE, &split_k_mode,
      sizeof(split_k_mode)));
  AS_CHECK_CUSPARSE(cusparseLtMatmulAlgGetAttribute(
      &cslt_handle, &alg_sel, CUSPARSELT_MATMUL_SPLIT_K_BUFFERS,
      &split_k_buffers, sizeof(split_k_buffers)));
  best_cfg = {config_id, split_k, split_k_mode, split_k_buffers};

  AS_CHECK_CUSPARSE(
      cusparseLtMatmulPlanInit(&cslt_handle, &plan, &matmul, &alg_sel));
  AS_CHECK_CUSPARSE(
      cusparseLtMatmulGetWorkspace(&cslt_handle, &plan, &cslt_ws_size));

  weights_[0]->Free();
  weights_[0]->SetDataType(DataType::INT8);
  weights_[0]->SetShape(Shape{compressed_size / sizeof(int8_t)});
  AS_CHECK_CUDA(
      cudaMemcpyAsync(weights_[0]->GetDataPtr(),
                      static_cast<void*>(compressed_weight.GetDataPtr()),
                      compressed_size, cudaMemcpyDefault, stream));
  AS_CHECK_CUDA(cudaStreamSynchronize(stream));
}

void GemmSparseA8W8GPU::GetWeightPaddedDispatch(const DataType ftype,
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
          throw AsException("ALLSPARK_NOT_IMPLEMENT_ERROR");
        }
        // Padding case 2: k % A8W8_K_PAD_ALIGN == 0 and n % A8W8_N_PAD_ALIGN
        // != 0
        if (n_ % A8W8_N_PAD_ALIGN != 0 && k_ % A8W8_K_PAD_ALIGN == 0 &&
            !transB_) {
          throw AsException("ALLSPARK_NOT_IMPLEMENT_ERROR");
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
          throw AsException("ALLSPARK_NOT_IMPLEMENT_ERROR");
        }
        // Padding case 2: k % A8W8_N_PAD_ALIGN == 0 and n % A8W8_N_PAD_ALIGN
        // != 0
        if (n_ % A8W8_N_PAD_ALIGN != 0 && k_ % A8W8_K_PAD_ALIGN == 0 &&
            !transB_) {
          throw AsException("ALLSPARK_NOT_IMPLEMENT_ERROR");
        }
      }
      break;
    }
    default:
      LOG(ERROR) << "GemmSparseA8W8GPU DataType Error\n";
      break;
  }
}

GemmSparseA8W8GPU::~GemmSparseA8W8GPU() { cudaFree(d_valid); }

template <typename FT, typename QT>
void GemmSparseA8W8GPU::DispatchKernel() {
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

  if (!UseSparseOpt()) {
    cuda::allspark_perc_qgemm_a8w8_gpu<FT>(
        in, nullptr, reinterpret_cast<int8_t*>(rhs_qdata_ptr), rhs_scales_ptr,
        rhs_zeros_ptr, bias, out, ws_ptr, (int)m_, (int)n_, (int)k_, sm_count_,
        is_prompt, cu_stream, cublas_handle);
  } else {
    cusparseLtMatDescriptor_t matA, matB, matC;
    cusparseLtMatmulDescriptor_t matmul;
    cusparseLtMatmulAlgSelection_t alg_sel;
    cusparseLtMatmulPlan_t plan;
    cusparseLtHandle_t cslt_handle = gpu_ctx->GetCuSparseHandle();

    InitCuSparseDescAlgPlan(cslt_handle, matA, matB, matC, matmul, alg_sel,
                            plan, m_padded_after_, n_, k_);
    AS_CHECK_CUSPARSE(cusparseLtMatmulAlgSetAttribute(
        &cslt_handle, &alg_sel, CUSPARSELT_MATMUL_ALG_CONFIG_ID,
        &best_cfg.algo_config_id, sizeof(best_cfg.algo_config_id)));
    AS_CHECK_CUSPARSE(cusparseLtMatmulAlgSetAttribute(
        &cslt_handle, &alg_sel, CUSPARSELT_MATMUL_SPLIT_K, &best_cfg.split_k,
        sizeof(best_cfg.split_k)));
    AS_CHECK_CUSPARSE(cusparseLtMatmulAlgSetAttribute(
        &cslt_handle, &alg_sel, CUSPARSELT_MATMUL_SPLIT_K_MODE,
        &best_cfg.split_k_mode, sizeof(best_cfg.split_k_mode)));
    AS_CHECK_CUSPARSE(cusparseLtMatmulAlgSetAttribute(
        &cslt_handle, &alg_sel, CUSPARSELT_MATMUL_SPLIT_K_BUFFERS,
        &best_cfg.split_k_buffers, sizeof(best_cfg.split_k_buffers)));
    AS_CHECK_CUSPARSE(
        cusparseLtMatmulPlanInit(&cslt_handle, &plan, &matmul, &alg_sel));

    cuda::allspark_perc_qgemm_sparse_a8w8_gpu<FT>(
        in, nullptr, reinterpret_cast<int8_t*>(rhs_qdata_ptr), rhs_scales_ptr,
        rhs_zeros_ptr, bias, out, ws_ptr, aligned_size(m_), aligned_size(n_),
        aligned_size(k_), sm_count_, is_prompt, cu_stream, cublas_handle,
        cslt_handle, plan);
  }

  if (activation_ != UNARYTYPE_UNDEFINED) {
    cuda::UnaryKernelLauncher(static_cast<FT*>(out), static_cast<FT*>(out),
                              m_ * n_ * batch_, activation_, cu_stream);
  }
}

AsStatus GemmSparseA8W8GPU::Reshape(RuntimeContext* runtime_ctx) {
  AsStatus status;
  is_prompt = runtime_ctx->is_context;
  status = Reshape();
  return status;
}

AsStatus GemmSparseA8W8GPU::Reshape() {
  AS_CHECK_STATUS(GemmA16W8Base::Reshape(n_));

  m_padded_after_ = aligned_size(m_);
  int64_t ws_size = 0;
  if (UseSparseOpt()) {
    ws_size = perc_qgemm_a8w8_gpu_workspace_size(
        aligned_size(m_), aligned_size(n_), aligned_size(k_));
  } else {
    ws_size = perc_qgemm_a8w8_gpu_workspace_size(m_, aligned_size(n_),
                                                 aligned_size(k_));
  }

#ifdef ENABLE_CUDA
  const CUDAContext* gpu_ctx = static_cast<const CUDAContext*>(ctx_);

  cublasHandle_t cublas_handle = gpu_ctx->GetCublasHandle();
  AS_CHECK_CUBLAS(cublasSetWorkspace(
      cublas_handle, tensor_map_->at("cublas_workspace")->GetDataPtr(),
      tensor_map_->at("cublas_workspace")->GetSizeInByte()));

  if (UseSparseOpt() && m_padded_after_ > 4096) {
    cudaStream_t cu_stream = gpu_ctx->GetStream();
    cusparseLtHandle_t cslt_handle = gpu_ctx->GetCuSparseHandle();
    cusparseLtMatDescriptor_t matA, matB, matC;
    cusparseLtMatmulDescriptor_t matmul;
    cusparseLtMatmulAlgSelection_t alg_sel;
    cusparseLtMatmulPlan_t plan;
    InitCuSparseDescAlgPlan(cslt_handle, matA, matB, matC, matmul, alg_sel,
                            plan, m_padded_after_, n_, k_);
    AS_CHECK_CUSPARSE(
        cusparseLtMatmulPlanInit(&cslt_handle, &plan, &matmul, &alg_sel));
    AS_CHECK_CUSPARSE(
        cusparseLtMatmulGetWorkspace(&cslt_handle, &plan, &cslt_ws_size));

    AS_CHECK_CUSPARSE(cusparseLtMatDescriptorDestroy(&matA));
    AS_CHECK_CUSPARSE(cusparseLtMatDescriptorDestroy(&matB));
    AS_CHECK_CUSPARSE(cusparseLtMatDescriptorDestroy(&matC));
    AS_CHECK_CUSPARSE(cusparseLtMatmulPlanDestroy(&plan));
    LOG(INFO) << "create workspace for cusparse: " << cslt_ws_size;
  }
  ws_size += cslt_ws_size;
#endif
  tensor_map_->at("workspace")->SetShape(Shape({ws_size}));
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus GemmSparseA8W8GPU::Forward(RuntimeContext* runtime_ctx) {
  AsStatus status;
  is_prompt = runtime_ctx->is_context;
  status = Forward();
  return status;
}

AsStatus GemmSparseA8W8GPU::Forward() {
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
      LOG(ERROR) << "GemmSparseA8W8GPU DataType Error\n";
      break;
  }
  return AsStatus::ALLSPARK_SUCCESS;
}

template <typename FType, typename QType>
void GemmSparseA8W8GPU::trans_TN(TensorMap& weights_buffer) {
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

REGISTER_OP(GemmSparseA8W8, CUDA, GemmSparseA8W8GPU)
}  // namespace allspark

#endif
