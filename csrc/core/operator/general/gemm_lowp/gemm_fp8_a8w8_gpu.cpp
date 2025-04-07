/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    gemm_fp8_a8w8_gpu.cpp
 */

#ifdef ENABLE_FP8
#include "gemm_fp8_a8w8_gpu.h"

#include <core/kernel/kernel.h>
#include <cpu/cpu_context.h>
#include <cuda/cuda_context.h>
#include <utility/datatype_dispatcher.h>

#include <limits>
#include <regex>

#include "cutlass/float8.h"

#ifdef ENABLE_BF16
#include <common/hie_bfloat16.hpp>
#endif
#define FP8_N_PAD_ALIGN 16
#define FP8_K_PAD_ALIGN 16
namespace allspark {

/* Structure to store information about different run trials */
typedef struct {
  cublasLtMatmulAlgo_t algo;
  float time;
  size_t workspaceSize;  // actual memory workspace needed
  float wavesCount;
} customMatmulPerf_t;

typedef struct {
  dim_t M_{};
  dim_t N_{};
  dim_t K_{};
} GemmShapeKey;

using GemmKeyUnorderedMap =
    std::unordered_map<std::string,
                       std::unordered_map<dim_t, customMatmulPerf_t>>;

class GemmPerfCache {
 public:
  static GemmPerfCache& GetInstance() {
    static GemmPerfCache inst;
    return inst;
  }

  GemmKeyUnorderedMap& GetMap() { return gemm_perf_map_; }

 private:
  // map for different (gemm shape - optimal algorithm)
  GemmKeyUnorderedMap gemm_perf_map_;
};

static std::vector<dim_t> MSearchList = {16,  32,   64,   128, 256,
                                         512, 1024, 4096, 8192};
// Under a given weight shape = (N, K) and a series of pre-set M size,
// executing fp8 gemm algorithm auto tuning by querying cublasLt heuristics for
// best algorithms, iterate over the results and pick the algorithm that have
// the best performance for the given problem
void GemmFP8A8W8GPU::TuneUseHeuristic() {
  const CUDAContext* gpu_ctx = static_cast<const CUDAContext*>(ctx_);
  cublasLtHandle_t cublaslt_handle = gpu_ctx->GetCublasLtHandle();
  cudaStream_t cu_stream = gpu_ctx->GetStream();

  auto& g_perfMap = GemmPerfCache::GetInstance().GetMap();
  if (g_perfMap.count(weight_name_pattern_) != 0) {
    return;
  }
  // set bias
  void* bias = (weights_.size() == 4) ? weights_[3]->GetDataPtr() : nullptr;
  if (bias) {
    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
    AS_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        operationDesc_, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue,
        sizeof(epilogue)));
    AS_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        operationDesc_, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias,
        sizeof(bias)));
  }

  // create preference handle; ; here we could use extra attributes to disable
  // tensor ops or to make sure algo selected will work with badly aligned A, B,
  // C; here for simplicity we just assume A,B,C are always well aligned (e.g.
  // directly come from cudaMalloc)
  AS_CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&preference_));
  AS_CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
      preference_, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceLimit_,
      sizeof(workspaceLimit_)));

  const int requestedAlgoCount = 10;
  int returnedResults = 0;
  cublasLtMatmulHeuristicResult_t heuristicResult[requestedAlgoCount] = {};
  void* workspace;
  AS_CHECK_CUDA(cudaMallocAsync(&workspace, workspaceLimit_, cu_stream));

  std::unordered_map<dim_t, customMatmulPerf_t> current_weight_layer{};
  for (dim_t M_idx = 0; M_idx < MSearchList.size(); M_idx++) {
    dim_t this_M = MSearchList[M_idx];
    AS_CHECK_CUBLAS(
        cublasLtMatrixLayoutInit(Bdesc_, fp8_cuda_type_, k_, this_M, lda_));
    AS_CHECK_CUBLAS(
        cublasLtMatrixLayoutInit(Cdesc_, in_cuda_type_, n_, this_M, ldc_));
    Ddesc_ = Cdesc_;

    // we just need the best available heuristic to try and run matmul. There is
    // no guarantee this will work, e.g. if A is badly aligned, you can request
    // more (e.g. 32) algos and try to run them one by one until something works
    AS_CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(
        cublaslt_handle, operationDesc_, Adesc_, Bdesc_, Cdesc_, Ddesc_,
        preference_, requestedAlgoCount, heuristicResult, &returnedResults));
    if (returnedResults == 0) {
      AS_CHECK_CUBLAS(CUBLAS_STATUS_NOT_SUPPORTED);
    }

    const void* A = weights_[0]->GetDataPtr();
    void* B = nullptr;
    void* D = nullptr;
    AS_CHECK_CUDA(cudaMallocAsync(&B, k_ * this_M, cu_stream));
    AS_CHECK_CUDA(cudaMemsetAsync(B, 0, k_ * this_M, cu_stream));
    AS_CHECK_CUDA(cudaMallocAsync(&D, n_ * this_M * sizeof(half), cu_stream));

    cudaEvent_t startEvent, stopEvent;
    int bestAlgoIdx = 0;
    float time = 0;
    float bestAlgoTime = std::numeric_limits<float>::max();
    bool find_algo = false;

    AS_CHECK_CUDA(cudaEventCreate(&startEvent));
    AS_CHECK_CUDA(cudaEventCreate(&stopEvent));
    constexpr int repeatAlgoCheck = 5;
    std::vector<float> algoTimes(repeatAlgoCheck);
    for (int algoIdx = 0; algoIdx < returnedResults; algoIdx++) {
      AS_CHECK_CUDA(cudaEventRecord(startEvent, cu_stream));
      cublasStatus_t algoRunStatus = CUBLAS_STATUS_SUCCESS;
      for (int checkIdx = 0; checkIdx < repeatAlgoCheck; checkIdx++) {
        cublasStatus_t oneRunStatus = cublasLtMatmul(
            cublaslt_handle, operationDesc_, &alpha_, A, Adesc_, B, Bdesc_,
            &beta_, D, Cdesc_, D, Ddesc_, &heuristicResult[algoIdx].algo,
            workspace, workspaceLimit_, cu_stream);
        AS_CHECK_CUDA(cudaEventRecord(stopEvent, cu_stream));
        AS_CHECK_CUDA(cudaEventSynchronize(stopEvent));
        AS_CHECK_CUDA(cudaEventElapsedTime(&time, startEvent, stopEvent));
        if (oneRunStatus != CUBLAS_STATUS_SUCCESS) {
          algoRunStatus = oneRunStatus;
          break;
        }
        algoTimes[checkIdx] = time;
      }
      if (algoRunStatus == CUBLAS_STATUS_SUCCESS) {
        find_algo = true;
        time = median(algoTimes);
        if (time < bestAlgoTime) {
          bestAlgoTime = time;
          bestAlgoIdx = algoIdx;
        }
      }
    }
    if (!find_algo) {
      AS_CHECK_CUBLAS(CUBLAS_STATUS_NOT_SUPPORTED);
    }
    customMatmulPerf_t current_perf_config = {
        heuristicResult[bestAlgoIdx].algo, bestAlgoTime,
        heuristicResult[bestAlgoIdx].workspaceSize,
        heuristicResult[bestAlgoIdx].wavesCount};
    current_weight_layer.insert({this_M, current_perf_config});

    // free cuda memory of B and D
    AS_CHECK_CUDA(cudaFreeAsync(B, cu_stream));
    AS_CHECK_CUDA(cudaFreeAsync(D, cu_stream));
  }
  g_perfMap.insert({weight_name_pattern_, current_weight_layer});
  AS_CHECK_CUDA(cudaFreeAsync(workspace, cu_stream));
}

AsStatus GemmFP8A8W8GPU::Init(const OperatorProto& op_proto,
                              const DeviceContext& ctx,
                              const TensorMap& weights_map,
                              TensorMap* tensor_map) {
  LOG(ERROR) << "GemmFP8A8W8GPU only support InitV2()" << std::endl;
  return AsStatus::ALLSPARK_INVALID_CALL_ERROR;
}

AsStatus GemmFP8A8W8GPU::InitV2(const OperatorProto& op_proto,
                                const DeviceContext& ctx,
                                const TensorMap& weights_map,
                                TensorMap& weights_buffer,
                                TensorMap* tensor_map,
                                RuntimeContext* runtime_ctx) {
  AS_CHECK_STATUS(GemmFP8Base::InitV2(op_proto, ctx, weights_map,
                                      weights_buffer, tensor_map, runtime_ctx));
  set_padding_flag(runtime_ctx);

  // if necessary, add the implementation of N and K padding later
  if (k_ % FP8_K_PAD_ALIGN != 0 || n_ % FP8_N_PAD_ALIGN) {
    LOG(ERROR) << "GemmFP8A8W8GPU : now only supports N is multiples of"
               << FP8_N_PAD_ALIGN << ", K is multiples of " << FP8_N_PAD_ALIGN
               << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }

  const Shape& param_shape = weights_[1]->GetShape();
  if (param_shape.Count(0) != 1) {
    LOG(ERROR)
        << "GemmFP8A8W8GPU : now only supports per tensor fp8 quantization."
        << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
  const DataType param_type = weights_[1]->GetDataType();
  if (param_type != DataType::FLOAT32) {
    LOG(ERROR) << "GemmFP8A8W8GPU : weight scale must be fp32 tensor"
               << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }

  if (alpha_ != 1.0f || beta_ != 0.0f) {
    LOG(ERROR)
        << "GemmFP8A8W8GPU : now only supports alpha = 1.0f and beta = 0.0f."
        << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }

  if (wtype_ == DataType::FLOAT16) {
    weight_scaled_fp8_quant<half>(weights_buffer);
    wtype_ = DataType::FLOAT8E4M3;
  }
  if (wtype_ == DataType::BFLOAT16) {
    weight_scaled_fp8_quant<hie::bfloat16>(weights_buffer);
    wtype_ = DataType::FLOAT8E4M3;
  }

  if (wtype_ == DataType::FLOAT8E4M3) {
    trans_TN<cutlass::float_e4m3_t>(weights_buffer);
    ldb_ = k_;
  } else if (wtype_ == DataType::FLOAT8E5M2) {
    trans_TN<cutlass::float_e5m2_t>(weights_buffer);
    ldb_ = k_;
  }
  // Get Device SM Count
  cudaDeviceProp device_prop;
  int device_id;
  cudaGetDevice(&device_id);
  cudaGetDeviceProperties(&device_prop, device_id);
  sm_count_ = device_prop.multiProcessorCount;
  sm_version_ = (device_prop.major << 8 | device_prop.minor);
  if (sm_version_ < 0x0809) {
    LOG(ERROR) << "The current device does not support float8 type, requires "
                  "sm89 and above."
               << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
  // according to https://docs.nvidia.com/cuda/cublas/#cublassetstream
  // cublasSetWorkspace supports user-defined workspace for cublas,
  // suggest workspace size 32 MiB for Hopper Architecture, 4 MiB for others.
  workspaceLimit_ = 8 * 1024 * 1024;
  if (device_prop.major >= 9) {
    workspaceLimit_ = 64 * 1024 * 1024;
  }

  AS_CHECK_CUBLAS(
      cublasLtMatmulDescCreate(&operationDesc_, computeType_, scaleType_));

  cublasOperation_t transa = CUBLAS_OP_T;
  cublasOperation_t transb = CUBLAS_OP_N;
  AS_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
      operationDesc_, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
  AS_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
      operationDesc_, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));
  // Default cublasLtOrder_t value is CUBLASLT_ORDER_COL. do matmul(B,A) instead
  // of matmul(A,B)
  fp8_cuda_type_ =
      wtype_ == DataType::FLOAT8E4M3 ? CUDA_R_8F_E4M3 : CUDA_R_8F_E5M2;
  in_cuda_type_ = atype_ == DataType::FLOAT16 ? CUDA_R_16F : CUDA_R_16BF;
  AS_CHECK_CUBLAS(
      cublasLtMatrixLayoutCreate(&Adesc_, fp8_cuda_type_, k_, n_, ldb_));
  AS_CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Bdesc_, fp8_cuda_type_, 1, 1, 1));
  AS_CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Cdesc_, in_cuda_type_, 1, 1, 1));

  std::regex dot_number_regex("\\.\\d+\\.");
  weight_name_pattern_ =
      std::regex_replace(weights_[0]->GetName(), dot_number_regex, ".");
  TuneUseHeuristic();

  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus GemmFP8A8W8GPU::Reshape() {
  const Shape& x_shape = tensor_map_->at(in_names_[0])->GetShape();
  int x_ndims = x_shape.Size();
  Shape y_shape;

  m_ = x_shape.Count(0, x_ndims - 1);

  for (int i = 0; i < x_ndims - 1; ++i) {
    y_shape.Append(x_shape[i]);
  }
  y_shape.Append(n_);

  tensor_map_->at(out_names_[0])->SetShape(std::move(y_shape));
  dim_t ws_size = 0;
  // A pertensor qdata
  ws_size += aligned_size(m_ * k_) * 1;
  // A scale
  ws_size += aligned_size(1) * sizeof(float);

  tensor_map_->at("workspace")->SetShape(Shape({ws_size}));

  const CUDAContext* gpu_ctx = static_cast<const CUDAContext*>(ctx_);
  cublasHandle_t cublas_handle = gpu_ctx->GetCublasHandle();
  cublasSetWorkspace(cublas_handle,
                     tensor_map_->at("cublas_workspace")->GetDataPtr(),
                     workspaceLimit_);
  AS_CHECK_CUBLAS(
      cublasLtMatrixLayoutInit(Bdesc_, fp8_cuda_type_, k_, m_, lda_));
  AS_CHECK_CUBLAS(
      cublasLtMatrixLayoutInit(Cdesc_, in_cuda_type_, n_, m_, ldc_));
  Ddesc_ = Cdesc_;
  return AsStatus::ALLSPARK_SUCCESS;
}

template <typename AType, typename WType>
AsStatus GemmFP8A8W8GPU::DispatchKernel() {
  AsTensor* in_tensor = tensor_map_->at(in_names_[0]).get();
  const AType* in_data = static_cast<const AType*>(in_tensor->GetDataPtr());
  void* weight_f8data = weights_[0]->GetDataPtr();
  const float* weight_scale_ptr =
      static_cast<const float*>(weights_[1]->GetDataPtr());

  // Currently FP8 GEMM only use symmetric quantizaiton, but zero is reserved
  void* weight_zeros_ptr = weights_[2]->GetDataPtr();
  void* ws_ptr = tensor_map_->at("workspace")->GetDataPtr();
  void* cslt_ws_ptr = tensor_map_->at("cublas_workspace")->GetDataPtr();
  void* bias = (weights_.size() == 4) ? weights_[3]->GetDataPtr() : nullptr;
  void* out = tensor_map_->at(out_names_[0])->GetDataPtr();

  const CUDAContext* gpu_ctx = static_cast<const CUDAContext*>(ctx_);
  cublasLtHandle_t cublaslt_handle = gpu_ctx->GetCublasLtHandle();
  cudaStream_t cu_stream = gpu_ctx->GetStream();

  // Quatize A from float16/bfloat16 to float8
  WType* a_f8data = static_cast<WType*>(ws_ptr);
  float* a_scale_ptr =
      reinterpret_cast<float*>(a_f8data + aligned_size(m_ * k_));
  cuda::per_tensor_symm_quantization<AType, WType>(in_data, a_scale_ptr,
                                                   a_f8data, m_, k_, cu_stream);

  // call cublasLtMatmul do matmul(B, A)
  AS_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
      operationDesc_, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &weight_scale_ptr,
      sizeof(weight_scale_ptr)));
  AS_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
      operationDesc_, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &a_scale_ptr,
      sizeof(a_scale_ptr)));
  if (bias) {
    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
    AS_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        operationDesc_, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue,
        sizeof(epilogue)));
    AS_CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
        operationDesc_, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias,
        sizeof(bias)));
  }

  int m_search_idx = 0;
  for (int i = 0; i < MSearchList.size(); ++i) {
    if (i == MSearchList.size() - 1 || m_ <= MSearchList[i]) {
      m_search_idx = i;
      break;
    }
  }

  auto& g_perfMap = GemmPerfCache::GetInstance().GetMap();
  if (g_perfMap.count(weight_name_pattern_) == 0) {
    LOG(ERROR) << "GemmFP8A8W8GPU: Can not find cublasLt algo configuration "
                  "corresponding to current weight layer"
               << std::endl;
    return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }
  cublasLtMatmulAlgo_t algo =
      g_perfMap[weight_name_pattern_].at(MSearchList[m_search_idx]).algo;
  AS_CHECK_CUBLAS(cublasLtMatmul(cublaslt_handle, operationDesc_, &alpha_,
                                 weight_f8data, Adesc_, a_f8data, Bdesc_,
                                 &beta_, out, Cdesc_, out, Ddesc_, &algo,
                                 cslt_ws_ptr, workspaceLimit_, cu_stream));
  if (activation_ != UNARYTYPE_UNDEFINED) {
    cuda::UnaryKernelLauncher(static_cast<AType*>(out),
                              static_cast<AType*>(out), m_ * n_ * batch_,
                              activation_, cu_stream);
  }
  // PrintInformation();
  return AsStatus::ALLSPARK_SUCCESS;
}

AsStatus GemmFP8A8W8GPU::Forward() {
  switch (atype_) {
#ifdef ENABLE_FP16
    case DataType::FLOAT16: {
      if (wtype_ == DataType::FLOAT8E4M3) {
        return DispatchKernel<half, cutlass::float_e4m3_t>();
      } else if (wtype_ == DataType::FLOAT8E5M2) {
        return DispatchKernel<half, cutlass::float_e5m2_t>();
      } else {
        LOG(ERROR) << "GemmFP8A8W8GPU DataType Error\n";
        return AsStatus::ALLSPARK_RUNTIME_ERROR;
      }
    }
#endif
#ifdef ENABLE_BF16
    case DataType::BFLOAT16: {
      if (wtype_ == DataType::FLOAT8E4M3) {
        return DispatchKernel<hie::bfloat16, cutlass::float_e4m3_t>();
      } else if (wtype_ == DataType::FLOAT8E5M2) {
        return DispatchKernel<hie::bfloat16, cutlass::float_e5m2_t>();
      } else {
        LOG(ERROR) << "GemmFP8A8W8GPU DataType Error\n";
        return AsStatus::ALLSPARK_RUNTIME_ERROR;
      }
    }
#endif
    default:
      LOG(ERROR) << "GemmFP8A8W8GPU DataType Error\n";
      break;
  }
  return AsStatus::ALLSPARK_SUCCESS;
}

template <typename WType>
void GemmFP8A8W8GPU::trans_TN(TensorMap& weights_buffer) {
  if (do_padding_) {
    AsTensor old_weight_cpu =
        AsTensor(weights_[0]->GetName() + "old_cpu", DeviceType::CPU,
                 weights_[0]->GetDataType(), weights_[0]->GetDataMode(),
                 weights_[0]->GetShape());
    old_weight_cpu.CopyDataFrom(weights_[0]->GetDataPtr(), k_ * n_,
                                weights_[0]->GetDeviceType(), ctx_);

    AsTensor reordered_weight_cpu =
        AsTensor(weights_[0]->GetName() + "trans_cpu", DeviceType::CPU,
                 weights_[0]->GetDataType(), weights_[0]->GetDataMode(),
                 Shape({n_, k_}));

    WType* old_weight_ptr = static_cast<WType*>(old_weight_cpu.GetDataPtr());
    WType* reordered_weight_ptr =
        static_cast<WType*>(reordered_weight_cpu.GetDataPtr());

    const CUDAContext* gpu_ctx = static_cast<const CUDAContext*>(ctx_);
    gpu_ctx->Synchronize();

    for (int i = 0; i < k_; ++i) {
      for (int j = 0; j < n_; ++j) {
        int src_offset = i * n_ + j;
        int dst_offset = j * k_ + i;
        reordered_weight_ptr[dst_offset] = old_weight_ptr[src_offset];
      }
    }
    auto* mutable_weight = const_cast<AsTensor*>(weights_[0]);
    mutable_weight->SetShape(Shape({n_, k_}));
    TensorUtils::DeepCopyWholeAsync(*mutable_weight, reordered_weight_cpu,
                                    ctx_);
    util::SyncWeightsBuffer(weights_buffer, weights_[0]->GetName(),
                            std::make_shared<AsTensor>(reordered_weight_cpu));
  }
}

template <typename WType>
void GemmFP8A8W8GPU::weight_scaled_fp8_quant(TensorMap& weights_buffer) {
  if (do_padding_) {
    const CUDAContext* gpu_ctx = static_cast<const CUDAContext*>(ctx_);
    cudaStream_t cu_stream = gpu_ctx->GetStream();

    // Quatize weight from float16/bfloat16 to float8
    WType* weight_in = static_cast<WType*>(weights_[0]->GetDataPtr());
    float* weight_scale = static_cast<float*>(weights_[1]->GetDataPtr());
    cutlass::float_e4m3_t* weight_fp8;

    AS_CHECK_CUDA(cudaMallocAsync(&weight_fp8, k_ * n_, cu_stream));
    cuda::per_tensor_symm_quantization<WType, cutlass::float_e4m3_t>(
        weight_in, weight_scale, weight_fp8, k_, n_, cu_stream);
    auto* mutable_weight = const_cast<AsTensor*>(weights_[0]);
    mutable_weight->SetDataType(DataType::FLOAT8E4M3);
    mutable_weight->SetShape(Shape({k_, n_}));
    mutable_weight->CopyDataFrom(weight_fp8, k_ * n_,
                                 weights_[0]->GetDeviceType(), gpu_ctx);
    AS_CHECK_CUDA(cudaFreeAsync(weight_fp8, cu_stream));
    gpu_ctx->Synchronize();
  }
}

REGISTER_OP(GemmFP8A8W8, CUDA, GemmFP8A8W8GPU)
}  // namespace allspark

#endif
