/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    cuda_context.cpp
 */

#ifdef ENABLE_CUDA
#include "cuda/cuda_context.h"

#include <check_cuda.h>

#ifdef ALLSPARK_DEBUG_MODE
#define ENABLE_NSYS_PROFILE 1
#else
// change to 1 enable nsys profile control.
#define ENABLE_NSYS_PROFILE 0
#endif

#if ENABLE_NSYS_PROFILE
#include <cuda_profiler_api.h>
#endif

namespace allspark {

thread_local int CUDAContext::last_device_id_of_this_thread_ = -1;

CUDAContext::~CUDAContext() {
  try {
    if (hiednn_handle_) {
      AS_CHECK_HIEDNN(hiednnDestroyCudaHandle(hiednn_handle_));
    }
    if (cublas_handle_) {
      AS_CHECK_CUBLAS(cublasDestroy(cublas_handle_));
    }
    if (stream_) {
      AS_CHECK_CUDA(cudaStreamDestroy(stream_));
    }
    if (comm_) {
      AS_CHECK_NCCL(ncclCommDestroy(comm_));
    }
#ifdef ENABLE_CUSPARSELT
    if (cslt_handle_initialized_) {
      AS_CHECK_CUSPARSE(cusparseLtDestroy(&cslt_handle_));
    }
#endif
  } catch (...) {
    // avoid the gcc warnning.
    LOG(ERROR) << "Exception in destroy cuda context.";
  }

  // reset the thread local value, it's a static value.
  last_device_id_of_this_thread_ = -1;
}

void CUDAContext::SetDeviceId(int device_id) {
  DLOG(INFO) << "CUDAContext::SetDeviceId()" << device_id << std::endl;

  // if have old handler, destory the device id and handler.
  if (hiednn_handle_) {
    AS_CHECK_CUDA(cudaSetDevice(device_id_));
    AS_CHECK_HIEDNN(hiednnDestroyCudaHandle(hiednn_handle_));
  }
  if (cublas_handle_) {
    AS_CHECK_CUDA(cudaSetDevice(device_id_));
    AS_CHECK_CUBLAS(cublasDestroy(cublas_handle_));
  }
  if (stream_) {
    AS_CHECK_CUDA(cudaSetDevice(device_id_));
    AS_CHECK_CUDA(cudaStreamDestroy(stream_));
  }
#ifdef ENABLE_CUSPARSELT
  if (cslt_handle_initialized_) {
    AS_CHECK_CUSPARSE(cusparseLtDestroy(&cslt_handle_));
  }
#endif

  // setup new device id related resource.
  device_id_ = device_id;
  AS_CHECK_CUDA(cudaSetDevice(device_id_));
  AS_CHECK_CUDA(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));

  AS_CHECK_CUBLAS(cublasCreate(&cublas_handle_));
  AS_CHECK_CUBLAS(cublasSetStream(cublas_handle_, stream_));

  AS_CHECK_HIEDNN(hiednnCreateCudaHandle(&hiednn_handle_));
  AS_CHECK_HIEDNN(hiednnSetCudaStream(hiednn_handle_, stream_));

#ifdef ENABLE_CUSPARSELT
  int sm_version = GetStreamProcessorVersion(device_id);
  if (sm_version >= allspark::CUDASMDef::SM_80 &&
      sm_version < allspark::CUDASMDef::SM_90) {
    AS_CHECK_CUSPARSE(cusparseLtInit(&cslt_handle_));
    cslt_handle_initialized_ = true;
  }
#endif
  last_device_id_of_this_thread_ = device_id_;
}

void CUDAContext::SetSparsityMatmulMode(bool enable_sparsity_matmul) {
  if (cslt_handle_initialized_) {
    enable_sparsity_matmul_ = enable_sparsity_matmul;
  }
}

int CUDAContext::GetStreamProcessorVersion(int device_id) {
  cudaDeviceProp device_prop;
  cudaGetDeviceProperties(&device_prop, device_id);
  int sm_version = (device_prop.major << 8 | device_prop.minor);
  LOG(INFO) << "Get SM version return " << std::hex << "0x" << sm_version
            << " Device Id: " << device_id;
  return sm_version;
}

int CUDAContext::GetStreamProcessorVersion() {
  return GetStreamProcessorVersion(device_id_);
}

void CUDAContext::Synchronize() const {
  AS_CHECK_CUDA(cudaStreamSynchronize(stream_));
}

#if ENABLE_NSYS_PROFILE
void CUDAContext::NsysProfilerStart() const { cudaProfilerStart(); }
void CUDAContext::NsysProfilerStop() const { cudaProfilerStop(); }
#else
void CUDAContext::NsysProfilerStart() const {}
void CUDAContext::NsysProfilerStop() const {}
#endif

void CUDAContext::SetDtype(DataType new_dtype) {
  // Check dtype is supported status

  // for cuda support:
  // ref:
  // https://docs.nvidia.com/deeplearning/tensorrt/support-matrix/index.html#hardware-precision-matrix
  // check bf16 support status.
  if (new_dtype == DataType::BFLOAT16) {
    int sm_version = this->GetStreamProcessorVersion();
    // sm_version will get value like: 806 (in hex)
    if (sm_version < 0x800) {
      LOG(ERROR)
          << "Current device: " << device_id_ << " sm: " << std::hex
          << sm_version
          << " not support bfloat16 datatype, please set float16 as data type.";

      throw std::invalid_argument(
          "bfloat16 is not supported for current device.");
    }
  }

  // TODO: check fb8 support status
  dtype = new_dtype;
}

void CUDAContext::InitNCCL(int rank, const ncclUniqueId& id, int nRanks) {
  DLOG(INFO) << "CUDAContext::InitNCCL()"
             << ", rank_id: " << rank;
  if (comm_) {
    AS_CHECK_NCCL(ncclCommDestroy(comm_));
  }
  const char* err_str = nullptr;
  auto err = cudaGetLastError();
  if (err != cudaSuccess) {
    LOG(INFO) << "cuda error in init nccl: " << cudaGetErrorString(err);
  }

  AS_CHECK_NCCL(ncclCommInitRank(&comm_, nRanks, id, rank));
  nranks_ = nRanks;
  rank_ = rank;
}
ncclComm_t CUDAContext::GetNCCLComm() const { return comm_; }
int CUDAContext::GetRank() const { return rank_; }
int CUDAContext::GetNranks() const { return nranks_; }

}  // namespace allspark
#endif
