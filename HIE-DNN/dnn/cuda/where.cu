/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    where.cu
 */

#include <hiednn.h>
#include <hiednn_cuda.h>

#include <cstdint>

#include <utils.hpp>
#include <tensor_desc.hpp>
#include <datatype_dispatch.hpp>
#include <integer_divmod.hpp>

#include <cuda/cuda_handle.hpp>
#include <cuda/cuda_utils.hpp>
#include <cuda/intrinsic/global_memory.hpp>

namespace hiednn {

namespace cuda {

namespace {

template <int BLOCK, int UNROLL, typename T>
__global__ void WhereKernel(
        const T *x, const T *y, const char *condition, T *z, uint32_t zSize) {
    uint32_t baseOffset = blockIdx.x * BLOCK * UNROLL + threadIdx.x;
    const T *xPtr = x + baseOffset;
    const T *yPtr = y + baseOffset;
    const char *condPtr = condition + baseOffset;
    T *zPtr = z + baseOffset;

    char cond[UNROLL];
    T zReg[UNROLL];

    uint32_t ioCount = zSize > baseOffset ?
                       UIntDivRU<uint32_t>(zSize - baseOffset, BLOCK) : 0;
    if (ioCount >= UNROLL) {
        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            cond[i] = condPtr[i * BLOCK];
        }

        // separated memory load for tensor x and y
        // to reduce integer instructions
        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            if (cond[i]) {
                zReg[i] = xPtr[i * BLOCK];
            }
        }
        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            if (!cond[i]) {
                zReg[i] = yPtr[i * BLOCK];
            }
        }

        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            zPtr[i * BLOCK] = zReg[i];
        }
    } else {
        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            if (i >= ioCount) break;

            cond[i] = condPtr[i * BLOCK];
            zReg[i] = cond[i] ? xPtr[i * BLOCK] : yPtr[i * BLOCK];
            zPtr[i * BLOCK] = zReg[i];
        }
    }
}

template <int BLOCK, int UNROLL, int NDIMS>
__device__ __forceinline__
void GetZIdx(uint32_t (&zIdx)[UNROLL][NDIMS],
             const uint32_t &zOffset0,
             const Array<U32DivMod, NDIMS> &zDivMod) {
    uint32_t divTmp[UNROLL];
    #pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
        divTmp[i] = zOffset0 + i * BLOCK;
    }

    #pragma unroll
    for (int i = 0; i < NDIMS; ++i) {
        #pragma unroll
        for (int j = 0; j < UNROLL; ++j) {
            auto dm = zDivMod[i].DivMod(divTmp[j]);
            zIdx[j][i] = dm.mod;
            divTmp[j] = dm.div;
        }
    }
}

template <int NDIMS, bool EXPAND>
__device__ __forceinline__
uint32_t GetLdgOffset(
        const uint32_t (&idx)[NDIMS],
        const Array<uint32_t, NDIMS> &stride,
        const uint32_t &linearOffset) {
    if (EXPAND) {
        uint32_t ret = 0;
        #pragma unroll
        for (int i = 0; i < NDIMS; ++i) {
            ret += idx[i] * stride[i];
        }
        return ret;
    } else {
        return linearOffset;
    }
}

template <int BLOCK,
          int UNROLL,
          int NDIMS,
          bool X_EXPAND,
          bool Y_EXPAND,
          bool COND_EXPAND,
          typename T>
__global__ void BroadcastWhereKernel(
        const T *__restrict__ x,
        const T *__restrict__ y,
        const char *__restrict__ condition,
        T *z,
        uint32_t zSize,
        Array<U32DivMod, NDIMS> zDivMod,
        Array<uint32_t, NDIMS> xStride,
        Array<uint32_t, NDIMS> yStride,
        Array<uint32_t, NDIMS> condStride) {
    uint32_t zOffset0 = blockIdx.x * BLOCK * UNROLL + threadIdx.x;

    uint32_t zIdx[UNROLL][NDIMS];
    GetZIdx<BLOCK, UNROLL, NDIMS>(zIdx, zOffset0, zDivMod);

    uint32_t condOffset[UNROLL];
    #pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
        condOffset[i] = GetLdgOffset<NDIMS, COND_EXPAND>(
            zIdx[i], condStride, zOffset0 + i * BLOCK);
    }

    uint32_t ioCount = zSize > zOffset0 ?
                       UIntDivRU<uint32_t>(zSize - zOffset0, BLOCK) : 0;
    bool fullUnroll = ioCount >= UNROLL;

    char cond[UNROLL];
    // load condition
    if (fullUnroll) {
        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            cond[i] = condition[condOffset[i]];
        }
    } else {
        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            if (i < ioCount) {
                cond[i] = condition[condOffset[i]];
            }
        }
    }

    // get ldg offset
    const T *ldgPtr[UNROLL];
    #pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
        if (cond[i]) {
            ldgPtr[i] = x + GetLdgOffset<NDIMS, X_EXPAND>(
                zIdx[i], xStride, zOffset0 + i * BLOCK);
        } else {
            ldgPtr[i] = y + GetLdgOffset<NDIMS, Y_EXPAND>(
                zIdx[i], yStride, zOffset0 + i * BLOCK);
        }
    }

    T data[UNROLL];
    // data load and store
    if (fullUnroll) {
        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            Ldg<NC>(&data[i], ldgPtr[i]);
        }
        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            z[zOffset0 + i * BLOCK] = data[i];
        }
    } else {
        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            if (i < ioCount) {
                Ldg<NC>(&data[i], ldgPtr[i]);
            }
        }
        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            if (i < ioCount) {
                z[zOffset0 + i * BLOCK] = data[i];
            }
        }
    }
}

template <typename T>
struct WhereImpl {
    hiednnStatus_t Where(
            const T *x,
            const T *y,
            const char *condition,
            T *z,
            size_t zSize,
            cudaStream_t stream) {
        const int UNROLLED_BYTE = 16;
        const int UNROLL = UNROLLED_BYTE / sizeof(T) <= 8 ?
                           UNROLLED_BYTE / sizeof(T) : 8;
        const int BLOCK = 128;

        uint32_t grid = UIntDivRU<uint32_t>(zSize, UNROLL * BLOCK);
        WhereKernel<BLOCK, UNROLL, T><<<grid, BLOCK, 0, stream>>>(
            x, y, condition, z, zSize);

        CHECK_CUDA_RETURN(cudaGetLastError());
        return HIEDNN_STATUS_SUCCESS;
    }

    template <int NDIMS, bool X_EXPAND, bool Y_EXPAND, bool COND_EXPAND>
    hiednnStatus_t LaunchBroadcastWhereKernel(
            const T *x,
            const T *y,
            const char *condition,
            T *z,
            uint32_t zSize,
            const Array<uint32_t, NDIMS> &xStrides,
            const Array<uint32_t, NDIMS> &yStrides,
            const Array<uint32_t, NDIMS> &condStrides,
            const Array<U32DivMod, NDIMS> &zDivMod,
            cudaStream_t stream) {
        const int UNROLLED_BYTE = 16;
        const int UNROLL = UNROLLED_BYTE / sizeof(T) <= 8 ?
                           UNROLLED_BYTE / sizeof(T) : 8;
        const int BLOCK = 128;

        uint32_t grid = UIntDivRU<uint32_t>(zSize, UNROLL * BLOCK);
        BroadcastWhereKernel
            <BLOCK, UNROLL, NDIMS, X_EXPAND, Y_EXPAND, COND_EXPAND, T>
            <<<grid, BLOCK, 0, stream>>>(
            x, y, condition, z, zSize, zDivMod,
            xStrides, yStrides, condStrides);

        CHECK_CUDA_RETURN(cudaGetLastError());
        return HIEDNN_STATUS_SUCCESS;
    }

    template <int NDIMS>
    hiednnStatus_t BroadcastWhereDispatchExpand(
            const HiednnTensorDesc &xDesc,
            const T *x,
            const HiednnTensorDesc &yDesc,
            const T *y,
            const HiednnTensorDesc &conditionDesc,
            const char *condition,
            const HiednnTensorDesc &zDesc,
            T *z,
            cudaStream_t stream) {
        Array<uint32_t, NDIMS> xStrides;
        Array<uint32_t, NDIMS> yStrides;
        Array<uint32_t, NDIMS> condStrides;
        Array<U32DivMod, NDIMS> zDivMod;

        const auto &xNDims = xDesc.nDims;
        const auto &yNDims = yDesc.nDims;
        const auto &condNDims = conditionDesc.nDims;

        for (int i = 0; i < zDesc.nDims; ++i) {
            xStrides[i] = i < xNDims && xDesc.dims[xNDims - i - 1] > 1 ?
                          xDesc.strides[xNDims - i - 1] : 0;
            yStrides[i] = i < yNDims && yDesc.dims[yNDims - i - 1] > 1 ?
                          yDesc.strides[yNDims - i - 1] : 0;
            condStrides[i] = i < condNDims &&
                             conditionDesc.dims[condNDims - i - 1] > 1 ?
                             conditionDesc.strides[condNDims - i - 1] : 0;
            zDivMod[i] = U32DivMod(zDesc.dims[zDesc.nDims - i - 1]);
        }
        for (int i = zDesc.nDims; i < NDIMS; ++i) {
            xStrides[i] = 0;
            yStrides[i] = 0;
            condStrides[i] = 0;
            zDivMod[i] = U32DivMod(1);
        }

        bool xExpand = zDesc.size > xDesc.size;
        bool yExpand = zDesc.size > yDesc.size;
        bool condExpand = zDesc.size > conditionDesc.size;

        if (xExpand && yExpand && condExpand) {
            return LaunchBroadcastWhereKernel<NDIMS, true, true, true>(
                x, y, condition, z, zDesc.size, xStrides,
                yStrides, condStrides, zDivMod, stream);
        } else if (xExpand && yExpand && !condExpand) {
            return LaunchBroadcastWhereKernel<NDIMS, true, true, false>(
                x, y, condition, z, zDesc.size, xStrides,
                yStrides, condStrides, zDivMod, stream);
        } else if (xExpand && !yExpand && condExpand) {
            return LaunchBroadcastWhereKernel<NDIMS, true, false, true>(
                x, y, condition, z, zDesc.size, xStrides,
                yStrides, condStrides, zDivMod, stream);
        } else if (xExpand && !yExpand && !condExpand) {
            return LaunchBroadcastWhereKernel<NDIMS, true, false, false>(
                x, y, condition, z, zDesc.size, xStrides,
                yStrides, condStrides, zDivMod, stream);
        } else if (!xExpand && yExpand && condExpand) {
            return LaunchBroadcastWhereKernel<NDIMS, false, true, true>(
                x, y, condition, z, zDesc.size, xStrides,
                yStrides, condStrides, zDivMod, stream);
        } else if (!xExpand && yExpand && !condExpand) {
            return LaunchBroadcastWhereKernel<NDIMS, false, true, false>(
                x, y, condition, z, zDesc.size, xStrides,
                yStrides, condStrides, zDivMod, stream);
        } else if (!xExpand && !yExpand && condExpand) {
            return LaunchBroadcastWhereKernel<NDIMS, false, false, true>(
                x, y, condition, z, zDesc.size, xStrides,
                yStrides, condStrides, zDivMod, stream);
        } else {
            return HIEDNN_STATUS_INTERNAL_ERROR;
        }
    }

    hiednnStatus_t BroadcastWhere(
            const HiednnTensorDesc &xDesc,
            const T *x,
            const HiednnTensorDesc &yDesc,
            const T *y,
            const HiednnTensorDesc &conditionDesc,
            const char *condition,
            const HiednnTensorDesc &zDesc,
            T *z,
            cudaStream_t stream) {
        if (zDesc.nDims <= 2) {
            return BroadcastWhereDispatchExpand<2>(
                xDesc, x, yDesc, y, conditionDesc, condition, zDesc, z, stream);
        } else if (zDesc.nDims <= 4) {
            return BroadcastWhereDispatchExpand<4>(
                xDesc, x, yDesc, y, conditionDesc, condition, zDesc, z, stream);
        } else if (zDesc.nDims <= 6) {
            return BroadcastWhereDispatchExpand<6>(
                xDesc, x, yDesc, y, conditionDesc, condition, zDesc, z, stream);
        } else {
            return BroadcastWhereDispatchExpand<TENSOR_DIM_MAX>(
                xDesc, x, yDesc, y, conditionDesc, condition, zDesc, z, stream);
        }
    }

    hiednnStatus_t operator()(
            const HiednnCudaHandle &handle,
            const HiednnTensorDesc &xDesc,
            const void *x,
            const HiednnTensorDesc &yDesc,
            const void *y,
            const HiednnTensorDesc &conditionDesc,
            const char *condition,
            const HiednnTensorDesc &zDesc,
            void *z) {
        const T *xPtr = static_cast<const T *>(x);
        const T *yPtr = static_cast<const T *>(y);
        T *zPtr = static_cast<T *>(z);
        cudaStream_t stream = handle.stream;

        // limited by uint32 offset in non-broadcast kernel and
        // fast integer division in broadcast kernel
        if (zDesc.size > UINT32_MAX) {
            return HIEDNN_STATUS_TENSOR_OVERSIZE;
        }

        if (xDesc.size == zDesc.size &&
            yDesc.size == zDesc.size &&
            conditionDesc.size == zDesc.size) {
            // without broadcast
            return Where(xPtr, yPtr, condition, zPtr, zDesc.size, stream);
        } else {
            // broadcast
            return BroadcastWhere(xDesc, xPtr, yDesc, yPtr, conditionDesc,
                                  condition, zDesc, zPtr, stream);
        }
    }
};

}  // anonymous namespace

}  // namespace cuda

}  // namespace hiednn

hiednnStatus_t
hiednnCudaWhere(HiednnCudaHandle *cudaHandle,
                HiednnTensorDesc *xDesc,
                const void *x,
                HiednnTensorDesc *yDesc,
                const void *y,
                HiednnTensorDesc *conditionDesc,
                const char *condition,
                HiednnTensorDesc *zDesc,
                void *z) {
    if (!hiednn::CheckNullptr(cudaHandle, xDesc, yDesc, conditionDesc, zDesc) ||
        !hiednn::CheckTensorPtr(*xDesc, x, *yDesc, y, *conditionDesc, condition,
                                *zDesc, z)) {
        return HIEDNN_STATUS_INVALID_PARAMETER;
    }

    if (!hiednn::CheckNormalFormat(*xDesc, *yDesc, *conditionDesc, *zDesc)) {
        return HIEDNN_STATUS_INVALID_PARAMETER;
    }

    if (zDesc->dataType != xDesc->dataType ||
        zDesc->dataType != yDesc->dataType ||
        conditionDesc->dataType != HIEDNN_DATATYPE_BOOL) {
        return HIEDNN_STATUS_INVALID_DATATYPE;
    }

    if (!xDesc->UniBroadcastableTo(*zDesc) ||
        !yDesc->UniBroadcastableTo(*zDesc) ||
        !conditionDesc->UniBroadcastableTo(*zDesc)) {
        return HIEDNN_STATUS_INVALID_PARAMETER;
    }

    if (zDesc->size == 0) {
        return HIEDNN_STATUS_SUCCESS;
    }

    return hiednn::DispatchItemSize<hiednn::cuda::WhereImpl>(
               xDesc->dataType, *cudaHandle, *xDesc, x, *yDesc, y,
               *conditionDesc, condition, *zDesc, z);
}


