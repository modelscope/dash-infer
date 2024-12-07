/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    cuda_handle.hpp
 */

#ifndef DNN_INCLUDE_CUDA_CUDA_HANDLE_HPP_
#define DNN_INCLUDE_CUDA_CUDA_HANDLE_HPP_

#include <cuda_runtime.h>

#include <hiednn.h>
#include <cuda/cuda_utils.hpp>

struct HiednnCudaHandle {
    cudaDeviceProp deviceProp;
    cudaStream_t stream;

    size_t deviceWsSize;
    void *deviceWs;
    cudaEvent_t wsMutex;

    // DeviceWsLock must be called after DeviceWsUnlock
    template <typename T>
    hiednnStatus_t DeviceWsLock(T **ws, size_t size) const {
        if (size > deviceWsSize) {
            *ws = nullptr;
            return HIEDNN_STATUS_INTERNAL_ERROR;
        }
        *ws = static_cast<T *>(deviceWs);
        CHECK_CUDA_RETURN(cudaStreamWaitEvent(stream, wsMutex, 0));
        return HIEDNN_STATUS_SUCCESS;
    }

    hiednnStatus_t DeviceWsUnlock() const {
        CHECK_CUDA_RETURN(cudaEventRecord(wsMutex, stream));
        return HIEDNN_STATUS_SUCCESS;
    }
};

namespace hiednn {

namespace cuda {

/**
 * DeviceWsGuard is an easy-to-use device workspace lock,
 * it's a wrapper of HiednnCudaHandle::DeviceWsLock and
 * HiednnCudaHandle::DeviceWsUnlock.
 */
struct DeviceWsGuard {
    const HiednnCudaHandle *handle;
    bool lock = false;

    explicit DeviceWsGuard(const HiednnCudaHandle &cudaHandle)
        : handle(&cudaHandle) {}

    template <typename T>
    hiednnStatus_t GetWorkspace(T **ws, size_t size) {
        auto stat = handle->DeviceWsLock(ws, size);
        lock = (stat == HIEDNN_STATUS_SUCCESS);
        return stat;
    }

    ~DeviceWsGuard() {
        if (lock) {
            handle->DeviceWsUnlock();
        }
    }
};

}  // namespace cuda

}  // namespace hiednn

#endif  // DNN_INCLUDE_CUDA_CUDA_HANDLE_HPP_


