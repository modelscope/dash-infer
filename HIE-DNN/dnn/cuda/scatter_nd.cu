/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    scatter_nd.cu
 */

#include <hiednn.h>
#include <hiednn_cuda.h>

#include <cstddef>
#include <cstdint>
#include <algorithm>

#include <tensor_desc.hpp>
#include <utils.hpp>
#include <datatype_dispatch.hpp>
#include <integer_divmod.hpp>
#include <packed_memory_access.hpp>
#include <cuda/cuda_handle.hpp>
#include <cuda/cuda_utils.hpp>
#include <cuda/intrinsic/global_memory.hpp>

namespace hiednn {

namespace cuda {

namespace {

template <int BLOCK, int PACK, typename IndicesT, typename T>
__global__ void ScatterNDNormalKernel(
        const IndicesT *indices,
        const T *updates,
        T *y,
        uint32_t indexTupleSize,    // indicesDesc.dim[-1]
        uint32_t updatesIdxBound,   // updatesDesc.size / PACK - 1
        U32DivMod innerDivMod,      // U32DivMod(innerSize / PACK)
        Array<uint32_t, TENSOR_DIM_MAX> yStrides) {
    // cast indices to uint32_t
    __shared__ uint32_t indicesSmem[BLOCK * TENSOR_DIM_MAX];

    uint32_t tid0 = blockIdx.x * BLOCK;
    uint32_t tid1 = blockIdx.x * BLOCK + BLOCK - 1 <= updatesIdxBound ?
                    blockIdx.x * BLOCK + BLOCK - 1 : updatesIdxBound;
    auto dm0 = innerDivMod.DivMod(tid0);
    auto dm1 = innerDivMod.DivMod(tid1);
    uint32_t indexOffset0 = dm0.div * indexTupleSize;
    uint32_t indexOffset1 = dm1.div * indexTupleSize;
    uint32_t indexSize = indexOffset1 - indexOffset0 + indexTupleSize;

    uint32_t indicesReg[TENSOR_DIM_MAX];

    #pragma unroll
    for (int i = 0; i < TENSOR_DIM_MAX; ++i) {
        uint32_t indexLdgOffset = threadIdx.x + i * BLOCK;
        indicesReg[i] =
            indexLdgOffset < indexSize ?
            static_cast<uint32_t>(indices[indexOffset0 + indexLdgOffset]) : 0;
    }

    #pragma unroll
    for (int i = 0; i < TENSOR_DIM_MAX; ++i) {
        indicesSmem[threadIdx.x + i * BLOCK] = indicesReg[i];
    }

    __syncthreads();

    uint32_t tid = blockIdx.x * BLOCK + threadIdx.x;

    if (tid > updatesIdxBound) {
        return;
    }

    auto dm = innerDivMod.DivMod(tid);
    uint32_t indexOffset = dm.div * indexTupleSize - indexOffset0;
    uint32_t stgOffset = dm.mod * PACK;

    #pragma unroll
    for (int i = 0; i < TENSOR_DIM_MAX; ++i) {
        stgOffset += indicesSmem[indexOffset + i] * yStrides[i];
    }

    using V_T = VT<T, PACK>;

    V_T reg;
    Ldg(&reg, reinterpret_cast<const V_T *>(updates + tid * PACK));
    Stg(reg, reinterpret_cast<V_T *>(y + stgOffset));
}

template <typename IndicesT, typename T>
hiednnStatus_t
LaunchScatterNDNormal(const HiednnTensorDesc &dataDesc,
                      const HiednnTensorDesc &indicesDesc,
                      const IndicesT *indices,
                      const T *updates,
                      T *y,
                      const uint32_t &outerSize,
                      const uint32_t &innerSize,
                      cudaStream_t stream) {
    if (dataDesc.size > UINT32_MAX || innerSize > INT32_MAX) {
        return HIEDNN_STATUS_TENSOR_OVERSIZE;
    }

    uint32_t updatesSize = outerSize * innerSize;
    uint32_t indexTupleSize = indicesDesc.dims[indicesDesc.nDims - 1];

    // additional @yStrides must be 0
    Array<uint32_t, TENSOR_DIM_MAX> yStrides(dataDesc.strides,
                                             indexTupleSize,
                                             0l);

    uint32_t packSize = std::min(GetPackSize(updates), GetPackSize(y));
    uint32_t innerPack = PackConfig::MAX_PACKED_SIZE;

    while (innerSize % innerPack != 0) {
        innerPack /= 2;
    }

    packSize = std::min(packSize, innerPack);

    const uint32_t BLOCK = 128;

    hiednnStatus_t ret = HIEDNN_STATUS_SUCCESS;
    switch (packSize) {
        case 8:
            ScatterNDNormalKernel<BLOCK, ValidPack<T, 8>()>
                <<<UIntDivRU(updatesSize, BLOCK * 8), BLOCK, 0, stream>>> (
                indices, updates, y, indexTupleSize, updatesSize / 8 - 1,
                U32DivMod(innerSize / 8), yStrides);
            break;
        case 4:
            ScatterNDNormalKernel<BLOCK, ValidPack<T, 4>()>
                <<<UIntDivRU(updatesSize, BLOCK * 4), BLOCK, 0, stream>>> (
                indices, updates, y, indexTupleSize, updatesSize / 4 - 1,
                U32DivMod(innerSize / 4), yStrides);
            break;
        case 2:
            ScatterNDNormalKernel<BLOCK, ValidPack<T, 2>()>
                <<<UIntDivRU(updatesSize, BLOCK * 2), BLOCK, 0, stream>>> (
                indices, updates, y, indexTupleSize, updatesSize / 2 - 1,
                U32DivMod(innerSize / 2), yStrides);
            break;
        case 1:
            ScatterNDNormalKernel<BLOCK, 1>
                <<<UIntDivRU(updatesSize, BLOCK), BLOCK, 0, stream>>> (
                indices, updates, y, indexTupleSize, updatesSize - 1,
                U32DivMod(innerSize), yStrides);
            break;
        default:
            ret = HIEDNN_STATUS_INTERNAL_ERROR;
    }

    return ret;
}

template <typename IndicesT>
struct ScatterNDNormal {
    template <typename T>
    struct Impl {
        hiednnStatus_t operator()(const HiednnTensorDesc &dataDesc,
                                  const void *x,
                                  const HiednnTensorDesc &indicesDesc,
                                  const void *indices,
                                  const HiednnTensorDesc &updatesDesc,
                                  const void *updates,
                                  void *y,
                                  cudaStream_t stream) {
            if (x != y) {
                CHECK_CUDA_RETURN(cudaMemcpyAsync(
                    y, x, dataDesc.size * sizeof(T),
                    cudaMemcpyDefault, stream));
            }

            // if updates size == 0, no kernel launch needed, just return
            if (updatesDesc.size == 0UL) {
                return HIEDNN_STATUS_SUCCESS;
            }

            const int &rankI = indicesDesc.nDims;
            int64_t innerSize;
            int64_t outerSize = 1;

            for (int i = 0; i < rankI - 1; ++i) {
                outerSize *= updatesDesc.dims[i];
            }

            innerSize = static_cast<int64_t>(updatesDesc.size) / outerSize;

            return LaunchScatterNDNormal<IndicesT, T>(
                       dataDesc,
                       indicesDesc,
                       static_cast<const IndicesT *>(indices),
                       static_cast<const T *>(updates),
                       static_cast<T *>(y),
                       outerSize,
                       innerSize,
                       stream);
        }
    };
};

}  // anonymous namespace

}  // namespace cuda

}  // namespace hiednn

hiednnStatus_t
hiednnCudaScatterND(HiednnCudaHandle *cudaHandle,
                    HiednnTensorDesc *xDesc,
                    const void *x,
                    HiednnTensorDesc *indicesDesc,
                    const void *indices,
                    HiednnTensorDesc *updatesDesc,
                    const void *updates,
                    HiednnTensorDesc *yDesc,
                    void *y) {
    if (!hiednn::CheckNullptr(cudaHandle, xDesc, indicesDesc,
                              updatesDesc, yDesc) ||
        !hiednn::CheckTensorPtr(*xDesc, x, *indicesDesc, indices,
                                *updatesDesc, updates, *yDesc, y)) {
        return HIEDNN_STATUS_INVALID_PARAMETER;
    }

    if (!hiednn::CheckNormalFormat(
            *xDesc, *indicesDesc, *updatesDesc, *yDesc)) {
        return HIEDNN_STATUS_INVALID_PARAMETER;
    }

    if (*xDesc != *yDesc) {
        return HIEDNN_STATUS_INVALID_PARAMETER;
    }

    const int &rankD = xDesc->nDims;
    const int &rankI = indicesDesc->nDims;
    const int &rankU = updatesDesc->nDims;
    const auto &rankIdx = indicesDesc->dims[rankI - 1];

    if (rankIdx > rankD || rankU != rankD + rankI - rankIdx - 1) {
        return HIEDNN_STATUS_INVALID_PARAMETER;
    }

    for (int i = 0; i < rankI - 1; ++i) {
        if (updatesDesc->dims[i] != indicesDesc->dims[i]) {
            return HIEDNN_STATUS_INVALID_PARAMETER;
        }
    }

    for (int i = rankI - 1, j = 0; i < rankU; ++i, ++j) {
        if (updatesDesc->dims[i] != xDesc->dims[j + rankIdx]) {
            return HIEDNN_STATUS_INVALID_PARAMETER;
        }
    }

    if (yDesc->size == 0) {
        return HIEDNN_STATUS_SUCCESS;
    }

    switch (indicesDesc->dataType) {
        case HIEDNN_DATATYPE_INT32:
            return hiednn::DispatchItemSize<
                   hiednn::cuda::ScatterNDNormal<int32_t>::Impl>(
                   xDesc->dataType, *xDesc, x, *indicesDesc, indices,
                   *updatesDesc, updates, y, cudaHandle->stream);
        case HIEDNN_DATATYPE_UINT32:
            return hiednn::DispatchItemSize<
                   hiednn::cuda::ScatterNDNormal<uint32_t>::Impl>(
                   xDesc->dataType, *xDesc, x, *indicesDesc, indices,
                   *updatesDesc, updates, y, cudaHandle->stream);
        case HIEDNN_DATATYPE_INT64:
            return hiednn::DispatchItemSize<
                   hiednn::cuda::ScatterNDNormal<int64_t>::Impl>(
                   xDesc->dataType, *xDesc, x, *indicesDesc, indices,
                   *updatesDesc, updates, y, cudaHandle->stream);
        default:
            return HIEDNN_STATUS_INVALID_DATATYPE;
    }
}


