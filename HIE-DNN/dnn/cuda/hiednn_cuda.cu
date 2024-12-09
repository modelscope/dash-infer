/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    hiednn_cuda.cu
 */

#include <hiednn.h>
#include <hiednn_cuda.h>
#include <cuda/cuda_utils.hpp>
#include <cuda/cuda_handle.hpp>

hiednnStatus_t
hiednnCreateCudaHandle(HiednnCudaHandle **handle) {
    *handle = new HiednnCudaHandle();
    HiednnCudaHandle &handleRef = **handle;

    // cuda runtime initialization
    CHECK_CUDA_RETURN(cudaFree(0));

    // device property
    int deviceId;
    CHECK_CUDA_RETURN(cudaGetDevice(&deviceId));
    CHECK_CUDA_RETURN(cudaGetDeviceProperties(&handleRef.deviceProp, deviceId));
    size_t totalGlobalMem = handleRef.deviceProp.totalGlobalMem;

    // default stream
    handleRef.stream = 0;

    // allocate cuda memory workspace
    const size_t giga = (1LU << 30);
    const size_t mega = (1LU << 20);

    if (totalGlobalMem >= 31 * giga) {
        // 512MB workspace for GPUs with at least 32GB memory
        handleRef.deviceWsSize = 512 * mega;
    } else if (totalGlobalMem >= 23 * giga) {
        // 256MB workspace for GPUs with at least 24GB memory
        handleRef.deviceWsSize = 256 * mega;
    } else if (totalGlobalMem >= 7 * giga) {
        // 128MB workspace for GPUs with at least 8GB memory
        handleRef.deviceWsSize = 128 * mega;
    } else {
        // 64MB workspace for GPUs with less tan 8GB memory
        handleRef.deviceWsSize = 64 * mega;
    }

    CHECK_CUDA_RETURN(cudaMalloc(
        &handleRef.deviceWs, handleRef.deviceWsSize));
    CHECK_CUDA_RETURN(cudaEventCreateWithFlags(
        &handleRef.wsMutex, cudaEventDisableTiming));

    return HIEDNN_STATUS_SUCCESS;
}

hiednnStatus_t
hiednnDestroyCudaHandle(HiednnCudaHandle *handle) {
    if (handle != nullptr) {
        // release cuda memory workspace
        auto err = cudaFree(handle->deviceWs);
        if (err != cudaSuccess && err != cudaErrorCudartUnloading) {
            return HIEDNN_STATUS_RUNTIME_ERROR;
        }

        err = cudaEventDestroy(handle->wsMutex);
        if (err != cudaSuccess && err != cudaErrorCudartUnloading) {
            return HIEDNN_STATUS_RUNTIME_ERROR;
        }

        delete handle;
        return HIEDNN_STATUS_SUCCESS;
    } else {
        return HIEDNN_STATUS_INVALID_PARAMETER;
    }
}

hiednnStatus_t
hiednnSetCudaStream(HiednnCudaHandle *handle, cudaStream_t stream) {
    if (handle != nullptr) {
        handle->stream = stream;
        return HIEDNN_STATUS_SUCCESS;
    } else {
        return HIEDNN_STATUS_INVALID_PARAMETER;
    }
}


