/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    set_tensor_value.cu
 */

#include <hiednn.h>
#include <hiednn_cuda.h>

#include <cstddef>
#include <cstdint>
#include <algorithm>

#include <utils.hpp>
#include <tensor_desc.hpp>
#include <datatype_dispatch.hpp>
#include <packed_memory_access.hpp>

#include <cuda/cuda_handle.hpp>
#include <cuda/cuda_utils.hpp>
#include <cuda/intrinsic/global_memory.hpp>

namespace hiednn {

namespace cuda {

namespace {

template <int BLOCK, int UNROLL, typename T>
__global__ void SetTensorValueConstKernel(
        T *y, T value, PackedEltwiseConfig packConfig) {
    int64_t tid = static_cast<int64_t>(blockIdx.x) * BLOCK + threadIdx.x;

    if (tid < packConfig.nPack) {
        using VT_OUT = VT<T, UNROLL>;
        VT_OUT stReg;
        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            stReg.data[i] = value;
        }
        Stg(stReg, reinterpret_cast<VT_OUT *>(y) + tid);
    } else if (UNROLL > 1 && tid < packConfig.nThread) {
        int64_t idx = tid + packConfig.unpackedOffset;
        y[idx] = value;
    }
}

template <int BLOCK, int UNROLL, typename T>
__global__ void SetTensorValueRangeKernel(
        T *y, T start, T delta, PackedEltwiseConfig packConfig) {
    int64_t tid = static_cast<int64_t>(blockIdx.x) * BLOCK + threadIdx.x;

    if (tid < packConfig.nPack) {
        using V_T = VT<T, UNROLL>;
        V_T stReg;
        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            stReg.data[i] = start + (tid * UNROLL + i) * delta;
        }
        Stg(stReg, reinterpret_cast<V_T *>(y) + tid);
    } else if (UNROLL > 1 && tid < packConfig.nThread) {
        int64_t idx = tid + packConfig.unpackedOffset;
        y[idx] = start + idx * delta;
    }
}

template <int BLOCK, typename T>
__global__ void SetTensorValueDiagonalKernel(
        T *y, T value, int64_t firstRow, int64_t lastRow,
        int64_t rShift, int64_t stride0) {
    int64_t rowIdx = static_cast<int64_t>(blockIdx.x) * BLOCK + threadIdx.x +
                     firstRow;
    if (rowIdx < lastRow) {
        y[rowIdx * stride0 + (rowIdx + rShift)] = value;
    }
}

template <typename T>
hiednnStatus_t
SetTensorValueConst(T *y, size_t n, T value, cudaStream_t stream) {
    const int64_t BLOCK = 128;
    int packSize = GetPackSize(y);

    hiednnStatus_t ret = HIEDNN_STATUS_SUCCESS;
    switch (packSize) {
        case 8: {
            const int UNROLL = ValidPack<T, 8>();
            PackedEltwiseConfig packConfig(n, UNROLL, BLOCK);
            SetTensorValueConstKernel<BLOCK, UNROLL>
                <<<packConfig.nBlock, BLOCK, 0, stream>>>(
                y, value, packConfig);
            break;
        }
        case 4: {
            const int UNROLL = ValidPack<T, 4>();
            PackedEltwiseConfig packConfig(n, UNROLL, BLOCK);
            SetTensorValueConstKernel<BLOCK, UNROLL>
                <<<packConfig.nBlock, BLOCK, 0, stream>>>(
                y, value, packConfig);
            break;
        }
        case 2: {
            const int UNROLL = ValidPack<T, 2>();
            PackedEltwiseConfig packConfig(n, UNROLL, BLOCK);
            SetTensorValueConstKernel<BLOCK, UNROLL>
                <<<packConfig.nBlock, BLOCK, 0, stream>>>(
                y, value, packConfig);
            break;
        }
        case 1: {
            PackedEltwiseConfig packConfig(n, 1, BLOCK);
            SetTensorValueConstKernel<BLOCK, 1>
                <<<packConfig.nBlock, BLOCK, 0, stream>>>(
                y, value, packConfig);
            break;
        }
        default:
            ret = HIEDNN_STATUS_INTERNAL_ERROR;
            break;
    }

    return ret;
}

template <typename T>
hiednnStatus_t
SetTensorValueRange(T *y, size_t n, T start, T delta, cudaStream_t stream) {
    const int BLOCK = 128;
    int packSize = GetPackSize(y);

    hiednnStatus_t ret = HIEDNN_STATUS_SUCCESS;
    switch (packSize) {
        case 8: {
            const int UNROLL = ValidPack<T, 8>();
            PackedEltwiseConfig packConfig(n, UNROLL, BLOCK);
            SetTensorValueRangeKernel<BLOCK, UNROLL>
                <<<packConfig.nBlock, BLOCK, 0, stream>>>(
                y, start, delta, packConfig);
            break;
        }
        case 4: {
            const int UNROLL = ValidPack<T, 4>();
            PackedEltwiseConfig packConfig(n, UNROLL, BLOCK);
            SetTensorValueRangeKernel<BLOCK, UNROLL>
                <<<packConfig.nBlock, BLOCK, 0, stream>>>(
                y, start, delta, packConfig);
            break;
        }
        case 2: {
            const int UNROLL = ValidPack<T, 2>();
            PackedEltwiseConfig packConfig(n, UNROLL, BLOCK);
            SetTensorValueRangeKernel<BLOCK, UNROLL>
                <<<packConfig.nBlock, BLOCK, 0, stream>>>(
                y, start, delta, packConfig);
            break;
        }
        case 1: {
            PackedEltwiseConfig packConfig(n, 1, BLOCK);
            SetTensorValueRangeKernel<BLOCK, 1>
                <<<packConfig.nBlock, BLOCK, 0, stream>>>(
                y, start, delta, packConfig);
            break;
        }
        default:
            ret = HIEDNN_STATUS_INTERNAL_ERROR;
            break;
    }

    return ret;
}

template <typename T>
hiednnStatus_t
SetTensorValueDiagonal(int rShift,
                       T value,
                       const HiednnTensorDesc &yDesc,
                       T *y,
                       cudaStream_t stream) {
    CHECK_CUDA_RETURN(cudaMemsetAsync(
        y, 0, yDesc.size * sizeof(T), stream));

    int64_t m = yDesc.dims[0];
    int64_t n = yDesc.dims[1];
    if (rShift < 0) {
        m += rShift;  // m -= abs(rShift)
    } else {
        n -= rShift;
    }

    int64_t nnz = std::min(m, n);
    if (nnz <= 0) {
        return HIEDNN_STATUS_SUCCESS;
    }

    int64_t firstRow = rShift < 0 ? -rShift : 0;
    int64_t lastRow = firstRow + nnz;

    const int64_t BLOCK = 128;
    int nBlock = UIntDivRU(nnz, BLOCK);

    SetTensorValueDiagonalKernel<BLOCK>
        <<<nBlock, BLOCK, 0, stream>>>(
        y, value, firstRow, lastRow, rShift, yDesc.dims[1]);

    return HIEDNN_STATUS_SUCCESS;
}

template <typename T>
struct SetTensorValueConstImpl {
    hiednnStatus_t operator()(const void *valuePtr,
                              const HiednnTensorDesc &yDesc,
                              void *y,
                              cudaStream_t stream) {
        T *y_ptr = static_cast<T *>(y);
        T value = *static_cast<const T *>(valuePtr);

        return SetTensorValueConst<T>(y_ptr, yDesc.size, value, stream);
    }
};

template <typename T>
struct SetTensorValueRangeImpl {
    hiednnStatus_t operator()(const void *pStart,
                              const void *pDelta,
                              const HiednnTensorDesc &yDesc,
                              void *y,
                              cudaStream_t stream) {
        if (yDesc.nDims != 1) {
            return HIEDNN_STATUS_INVALID_PARAMETER;
        }

        T start = *static_cast<const T *>(pStart);
        T delta = *static_cast<const T *>(pDelta);
        T *y_ptr = static_cast<T *>(y);

        if (yDesc.nDims != 1) {
            return HIEDNN_STATUS_INVALID_PARAMETER;
        }

        return SetTensorValueRange<T>(y_ptr, yDesc.size, start, delta, stream);
    }
};

template <typename T>
struct SetTensorValueDiagonalImpl {
    hiednnStatus_t operator()(int rShift,
                              const void *valuePtr,
                              const HiednnTensorDesc &yDesc,
                              void *y,
                              cudaStream_t stream) {
        if (yDesc.nDims != 2 ||
            std::abs(rShift) >= yDesc.dims[0]) {
            return HIEDNN_STATUS_INVALID_PARAMETER;
        }

        T *y_ptr = static_cast<T *>(y);
        T value = *static_cast<const T *>(valuePtr);

        return SetTensorValueDiagonal(rShift, value, yDesc, y_ptr, stream);
    }
};

}  // anonymous namespace

}  // namespace cuda

}  // namespace hiednn

hiednnStatus_t
hiednnCudaSetTensorValue(HiednnCudaHandle *cudaHandle,
                         hiednnSetTensorValueMode_t mode,
                         const void *p0,
                         const void *p1,
                         hiednnTensorDesc_t yDesc,
                         void *y) {
    if (!hiednn::CheckNullptr(cudaHandle, yDesc) ||
        !hiednn::CheckTensorPtr(*yDesc, y)) {
        return HIEDNN_STATUS_INVALID_PARAMETER;
    }

    if (!hiednn::CheckNormalFormat(*yDesc)) {
        return HIEDNN_STATUS_INVALID_PARAMETER;
    }

    if (yDesc->size == 0) {
        return HIEDNN_STATUS_SUCCESS;
    }

    hiednnStatus_t ret = HIEDNN_STATUS_SUCCESS;
    switch (mode) {
        case HIEDNN_SETTENSOR_CONST:
            ret = hiednn::DispatchItemSize<
                  hiednn::cuda::SetTensorValueConstImpl>(
                      yDesc->dataType, p0, *yDesc, y, cudaHandle->stream);
            break;
        case HIEDNN_SETTENSOR_RANGE:
            ret = hiednn::DispatchAll<
                  hiednn::cuda::SetTensorValueRangeImpl>(
                      yDesc->dataType, p0, p1, *yDesc, y,
                      cudaHandle->stream);
            break;
        case HIEDNN_SETTENSOR_DIAGONAL:
            ret = hiednn::DispatchItemSize<
                  hiednn::cuda::SetTensorValueDiagonalImpl>(
                      yDesc->dataType, *static_cast<const int *>(p0), p1,
                      *yDesc, y, cudaHandle->stream);
            break;
        default:
            ret = HIEDNN_STATUS_INVALID_OPTYPE;
            break;
    }

    return ret;
}



