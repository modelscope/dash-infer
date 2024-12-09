/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    prefix_scan.cu
 */

#include <hiednn.h>
#include <hiednn_cuda.h>

#include <cstddef>
#include <cstdint>
#include <algorithm>

#include <utils.hpp>
#include <tensor_desc.hpp>
#include <datatype_dispatch.hpp>

#include <cuda/cuda_handle.hpp>
#include <cuda/cuda_utils.hpp>

#include "scan_d0.cuh"
#include "unified_scan_d1.cuh"
#include "tiled_scan_d1.cuh"

namespace hiednn {

namespace cuda {

namespace {

// scan contiguous data
template <typename T>
hiednnStatus_t PrefixSumD1(const HiednnCudaHandle &handle,
                           const T *x,
                           T *y,
                           int64_t m,
                           int64_t n,
                           int exclusive,
                           int reverse) {
    using ComputeT = typename PrefixComputeT<T>::type;

    hiednnStatus_t ret;
    if (TiledScanD1(handle, m, n)) {
        // split the scanned dimension to multi tiles
        if (exclusive == 0 && reverse == 0) {
            ret = tiled_scan_d1::TiledPrefixSum<false, false, ComputeT>(
                      handle, x, y, m, n);
        } else if (exclusive == 0 && reverse == 1) {
            ret = tiled_scan_d1::TiledPrefixSum<false, true, ComputeT>(
                      handle, x, y, m, n);
        } else if (exclusive == 1 && reverse == 0) {
            ret = tiled_scan_d1::TiledPrefixSum<true, false, ComputeT>(
                      handle, x, y, m, n);
        } else {
            // exclusive == 1 && reverse == 1
            ret = tiled_scan_d1::TiledPrefixSum<true, true, ComputeT>(
                      handle, x, y, m, n);
        }
    } else {
        if (exclusive == 0 && reverse == 0) {
            ret = unified_scan_d1::UnifiedPrefixSum<false, false, ComputeT>(
                      handle, x, y, m, n);
        } else if (exclusive == 0 && reverse == 1) {
            ret = unified_scan_d1::UnifiedPrefixSum<false, true, ComputeT>(
                      handle, x, y, m, n);
        } else if (exclusive == 1 && reverse == 0) {
            ret = unified_scan_d1::UnifiedPrefixSum<true, false, ComputeT>(
                      handle, x, y, m, n);
        } else {
            // exclusive == 1 && reverse == 1
            ret = unified_scan_d1::UnifiedPrefixSum<true, true, ComputeT>(
                      handle, x, y, m, n);
        }
    }

    return ret;
}

// scan non-contiguous data
template <typename T>
hiednnStatus_t PrefixSumD0(const HiednnCudaHandle &handle,
                           const T *x,
                           T *y,
                           int64_t m,
                           int64_t n,
                           int64_t batch,
                           int exclusive,
                           int reverse) {
    using ComputeT = typename PrefixComputeT<T>::type;

    hiednnStatus_t ret;
    if (exclusive == 0 && reverse == 0) {
        ret = scan_d0::PrefixSum<false, false, ComputeT>(
                  handle, x, y, m, n, batch);
    } else if (exclusive == 0 && reverse == 1) {
        ret = scan_d0::PrefixSum<false, true, ComputeT>(
                  handle, x, y, m, n, batch);
    } else if (exclusive == 1 && reverse == 0) {
        ret = scan_d0::PrefixSum<true, false, ComputeT>(
                  handle, x, y, m, n, batch);
    } else {
        // exclusive == 1 && reverse == 1
        ret = scan_d0::PrefixSum<true, true, ComputeT>(
                  handle, x, y, m, n, batch);
    }

    return ret;
}

void PrefixSumSortDimByStride(int nDims,
                              const int64_t *dims,
                              const int64_t *strides,
                              int axis,
                              int64_t *normalDims,
                              int64_t *normalStrides,
                              int *normalAxis) {
    struct StrideSwap {
        int64_t stride;
        int axis;
        static bool CmpGT(const StrideSwap &a, const StrideSwap &b) {
            return a.stride > b.stride;
        }
    };

    StrideSwap strideDescending[TENSOR_DIM_MAX];

    for (int i = 0; i < nDims; ++i) {
        strideDescending[i] = { strides[i], i };
    }

    std::sort(strideDescending, strideDescending + nDims, StrideSwap::CmpGT);

    for (int i = 0; i < nDims; ++i) {
        int originalAxis = strideDescending[i].axis;
        if (originalAxis == axis) {
            *normalAxis = i;
        }
        normalDims[i] = dims[originalAxis];
        normalStrides[i] = strides[originalAxis];
    }
}

void PrefixSumShrinkDimStride(const int64_t *normalDims,
                              const int64_t *normalStrides,
                              size_t tensorSize,
                              int normalAxis,
                              int64_t (&shrinkedDims)[3],
                              int64_t (&shrinkedStrides)[3],
                              int *shrinkedNDims,
                              int *shrinkedAxis) {
    *shrinkedNDims = 1;
    *shrinkedAxis = 0;

    // outer dims
    if (normalStrides[normalAxis] * normalDims[normalAxis] < tensorSize) {
        ++(*shrinkedNDims);
        ++(*shrinkedAxis);
        shrinkedStrides[0] = normalStrides[normalAxis] * normalDims[normalAxis];
        shrinkedDims[0] = tensorSize / shrinkedStrides[0];
    }

    // scanned dim
    shrinkedDims[*shrinkedAxis] = normalDims[normalAxis];
    shrinkedStrides[*shrinkedAxis] = normalStrides[normalAxis];

    // inner dims
    if (normalStrides[normalAxis] != 1) {
        ++(*shrinkedNDims);
        shrinkedStrides[*shrinkedAxis + 1] = 1;
        shrinkedDims[*shrinkedAxis + 1] = shrinkedStrides[*shrinkedAxis];
    }
}

// prefix scan for contiguous input and output tensor
template <typename T>
hiednnStatus_t PrefixSum(const HiednnCudaHandle &handle,
                         const HiednnTensorDesc &dataDesc,
                         const void *x,
                         int axis,
                         int exclusive,
                         int reverse,
                         void *y) {
    // --------------------------------------------------------
    // if scanning length is 1, just copy x to y or set y to all zero.
    // --------------------------------------------------------
    if (dataDesc.dims[axis] == 1) {
        if (exclusive == 1) {
            // exclusive mode, tensor y is all zero.
            CHECK_CUDA_RETURN(cudaMemsetAsync(
                y, 0, dataDesc.size * sizeof(T), handle.stream));
        } else {
            CHECK_CUDA_RETURN(cudaMemcpyAsync(
                y, x, dataDesc.size * sizeof(T), cudaMemcpyDefault,
                handle.stream));
        }
        return HIEDNN_STATUS_SUCCESS;
    }

    // --------------------------------------------------------
    // STEP1: sort dims by strides
    // --------------------------------------------------------
    int normalAxis = -1;
    int64_t normalDims[TENSOR_DIM_MAX];
    int64_t normalStrides[TENSOR_DIM_MAX];

    PrefixSumSortDimByStride(
        dataDesc.nDims, dataDesc.dims, dataDesc.strides, axis,
        normalDims, normalStrides, &normalAxis);

    if (normalAxis == -1) {
        return HIEDNN_STATUS_INTERNAL_ERROR;
    }

    // --------------------------------------------------------
    // STEP2: shrink dims and strides (nD -> 1D/2D/3D)
    // --------------------------------------------------------
    int shrinkedAxis;
    int shrinkedNDims;
    int64_t shrinkedDims[3];
    int64_t shrinkedStrides[3];

    PrefixSumShrinkDimStride(
        normalDims, normalStrides, dataDesc.size, normalAxis,
        shrinkedDims, shrinkedStrides, &shrinkedNDims, &shrinkedAxis);

    // --------------------------------------------------------
    // STEP3: prefix sum
    // --------------------------------------------------------
    if (shrinkedAxis == shrinkedNDims - 1) {
        // scan on contiguous data
        int64_t m = shrinkedAxis == 0 ? 1 : shrinkedDims[0];
        int64_t n = shrinkedDims[shrinkedAxis];

        return PrefixSumD1(handle, static_cast<const T *>(x),
                           static_cast<T *>(y), m, n, exclusive, reverse);
    } else {
        // scan on non-contiguous data
        int64_t m = shrinkedDims[shrinkedAxis];
        int64_t n = shrinkedDims[shrinkedAxis + 1];
        int64_t batch = shrinkedAxis == 0 ? 1 : shrinkedDims[0];

        return PrefixSumD0(handle, static_cast<const T *>(x),
                           static_cast<T *>(y), m, n, batch,
                           exclusive, reverse);
    }
}

template <typename T>
struct PrefixSumImpl {
    hiednnStatus_t operator()(const HiednnCudaHandle &handle,
                              const HiednnTensorDesc &dataDesc,
                              const void *x,
                              int axis,
                              int exclusive,
                              int reverse,
                              void *y) {
        // for tensor x with only 1 element, just copy x to y
        // or set y to zero (exclusive mode).
        if (dataDesc.size <= 1) {
            if (exclusive == 0) {
                CHECK_CUDA_RETURN(cudaMemcpyAsync(
                    y, x, dataDesc.size * sizeof(T), cudaMemcpyDefault,
                    handle.stream));
            } else {
                CHECK_CUDA_RETURN(cudaMemsetAsync(
                    y, 0, dataDesc.size * sizeof(T), handle.stream));
            }

            return HIEDNN_STATUS_SUCCESS;
        }

        return PrefixSum<T>(handle, dataDesc, x, axis, exclusive, reverse, y);
    }
};

}  // anonymous namespace

}  // namespace cuda

}  // namespace hiednn

hiednnStatus_t
hiednnCudaPrefixSum(HiednnCudaHandle *cudaHandle,
                    HiednnTensorDesc *xDesc,
                    const void *x,
                    int axis,
                    int exclusive,
                    int reverse,
                    HiednnTensorDesc *yDesc,
                    void *y) {
    if (!hiednn::CheckNullptr(cudaHandle, xDesc, yDesc) ||
        !hiednn::CheckTensorPtr(*xDesc, x, *yDesc, y)) {
        return HIEDNN_STATUS_INVALID_PARAMETER;
    }

    if (!hiednn::CheckNormalFormat(*xDesc, *yDesc)) {
        return HIEDNN_STATUS_INVALID_PARAMETER;
    }

    if (*xDesc != *yDesc ||
        axis < 0 || axis >= xDesc->nDims ||
        exclusive != 0 && exclusive != 1 ||
        reverse != 0 && reverse != 1) {
        return HIEDNN_STATUS_INVALID_PARAMETER;
    }

    if (yDesc->size == 0) {
        return HIEDNN_STATUS_SUCCESS;
    }

    return hiednn::DispatchAll<hiednn::cuda::PrefixSumImpl>(
               xDesc->dataType, *cudaHandle, *xDesc, x, axis,
               exclusive, reverse, y);
}


