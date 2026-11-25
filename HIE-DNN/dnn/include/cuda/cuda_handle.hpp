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
    hiednnStatus_t DeviceWsLock(T **ws, size_t size, bool capture) const {
        if (size > deviceWsSize) {
            *ws = nullptr;
            return HIEDNN_STATUS_INTERNAL_ERROR;
        }
        *ws = static_cast<T *>(deviceWs);
        unsigned int flags = 0;
#if CUDART_VERSION >= 12000
        if (capture) flags = cudaEventWaitExternal;
#endif
        CHECK_CUDA_RETURN(cudaStreamWaitEvent(stream, wsMutex, flags));
        return HIEDNN_STATUS_SUCCESS;
    }

    hiednnStatus_t DeviceWsUnlock(bool capture) const {
#if CUDART_VERSION >= 12000
        if (capture) {
            CHECK_CUDA_RETURN(cudaEventRecordWithFlags(
                wsMutex, stream, cudaEventRecordExternal));
            return HIEDNN_STATUS_SUCCESS;
        }
#endif
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
    bool captureAware;
    bool capture = false;
    bool lock = false;

    explicit DeviceWsGuard(const HiednnCudaHandle &cudaHandle,
                           bool captureAware = false)
        : handle(&cudaHandle), captureAware(captureAware) {}

    template <typename T>
    hiednnStatus_t GetWorkspace(T **ws, size_t size) {
#if CUDART_VERSION >= 12000
        if (captureAware) {
            // This event serializes shared workspace between graph and eager
            // calls. Preserve it as an event node during capture.
            cudaStreamCaptureStatus status;
            CHECK_CUDA_RETURN(cudaStreamIsCapturing(handle->stream, &status));
            capture = status == cudaStreamCaptureStatusActive;
        }
#endif
        auto stat = handle->DeviceWsLock(ws, size, capture);
        lock = (stat == HIEDNN_STATUS_SUCCESS);
        return stat;
    }

    ~DeviceWsGuard() {
        if (lock) {
            handle->DeviceWsUnlock(capture);
        }
    }
};

}  // namespace cuda

}  // namespace hiednn

#endif  // DNN_INCLUDE_CUDA_CUDA_HANDLE_HPP_


