/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    pad.cu
 */

#include <hiednn.h>
#include <hiednn_cuda.h>

#include <utils.hpp>
#include <tensor_desc.hpp>
#include <datatype_dispatch.hpp>
#include <integer_divmod.hpp>

#include <cuda/cuda_handle.hpp>
#include <cuda/cuda_utils.hpp>

#include "pad_functor.cuh"

namespace hiednn {

namespace cuda {

namespace pad {

namespace {

template <int BLOCK,
          int IMAGE_UNROLL,
          int NDIMS,
          typename OffsetT>
__device__ __forceinline__
void GetOutputIndex(
        const Array<U32DivMod, NDIMS> &yDimDivMod,
        const OffsetT &tid,
        OffsetT (&batchIdx)[IMAGE_UNROLL],
        OffsetT (&idx)[IMAGE_UNROLL][NDIMS]) {
    #pragma unroll
    for (int i = 0; i < IMAGE_UNROLL; ++i) {
        batchIdx[i] = tid + i * BLOCK;
    }

    #pragma unroll
    for (int i = 0; i < NDIMS; ++i) {
        #pragma unroll
        for (int j = 0; j < IMAGE_UNROLL; ++j) {
            auto dm = yDimDivMod[i].DivMod(batchIdx[j]);
            idx[j][i] = dm.mod;
            batchIdx[j] = dm.div;
        }
    }
}

template <int BLOCK,
          int NDIMS,
          int IMAGE_UNROLL,
          int BATCH_UNROLL,
          typename PadFunc,
          typename OffsetT,
          typename T>
__global__ void CudaPadKernel(
        const T *x,
        T *y,
        T param,
        Array<U32DivMod, NDIMS> yDimDivMod,
        Array<OffsetT, NDIMS> lPadBound,
        Array<OffsetT, NDIMS> rPadBound,
        Array<OffsetT, NDIMS> xIdxMax,
        Array<OffsetT, NDIMS> xStride,
        OffsetT batch,
        OffsetT batchStep,
        OffsetT xBatchStride,
        OffsetT batchLdgStride,
        OffsetT batchStgStride) {
    OffsetT tid = static_cast<OffsetT>(blockIdx.x) * BLOCK * IMAGE_UNROLL +
                  static_cast<OffsetT>(threadIdx.x);

    // index of each image unrolling point
    OffsetT batchIdx[IMAGE_UNROLL];
    OffsetT idx[IMAGE_UNROLL][NDIMS];
    GetOutputIndex<BLOCK, IMAGE_UNROLL, NDIMS>(yDimDivMod, tid, batchIdx, idx);

    PadFunc padFunc;

    // get the index of input points
    padFunc.GetInputPtr(x, batchIdx, idx, lPadBound, rPadBound,
                        xIdxMax, xStride, xBatchStride);

    T dataReg[BATCH_UNROLL][IMAGE_UNROLL];

    if (batchIdx[IMAGE_UNROLL - 1] + (BATCH_UNROLL - 1) * batchStep < batch) {
        // load input points
        #pragma unroll
        for (int batchIt = 0; batchIt < BATCH_UNROLL; ++batchIt) {
            #pragma unroll
            for (int imgIt = 0; imgIt < IMAGE_UNROLL; ++imgIt) {
                dataReg[batchIt][imgIt] = padFunc.Load(
                    batchIt, imgIt, batchLdgStride, param);
            }
        }

        // store output points
        #pragma unroll
        for (int batchIt = 0; batchIt < BATCH_UNROLL; ++batchIt) {
            #pragma unroll
            for (int imgIt = 0; imgIt < IMAGE_UNROLL; ++imgIt) {
                y[tid + imgIt * BLOCK + batchIt * batchStgStride] =
                    dataReg[batchIt][imgIt];
            }
        }
    } else {
        // load input points
        #pragma unroll
        for (int imgIt = 0; imgIt < IMAGE_UNROLL; ++imgIt) {
            if (batchIdx[imgIt] + (BATCH_UNROLL - 1) * batchStep < batch) {
                #pragma unroll
                for (int batchIt = 0; batchIt < BATCH_UNROLL; ++batchIt) {
                    dataReg[batchIt][imgIt] = padFunc.Load(
                        batchIt, imgIt, batchLdgStride, param);
                }
            } else {
                #pragma unroll
                for (int batchIt = 0; batchIt < BATCH_UNROLL; ++batchIt) {
                    if (batchIdx[imgIt] + batchIt * batchStep < batch) {
                        dataReg[batchIt][imgIt] = padFunc.Load(
                            batchIt, imgIt, batchLdgStride, param);
                    }
                }
            }
        }

        // store output points
        #pragma unroll
        for (int imgIt = 0; imgIt < IMAGE_UNROLL; ++imgIt) {
            if (batchIdx[imgIt] + (BATCH_UNROLL - 1) * batchStep < batch) {
                #pragma unroll
                for (int batchIt = 0; batchIt < BATCH_UNROLL; ++batchIt) {
                    y[tid + imgIt * BLOCK + batchIt * batchStgStride] =
                        dataReg[batchIt][imgIt];
                }
            } else {
                #pragma unroll
                for (int batchIt = 0; batchIt < BATCH_UNROLL; ++batchIt) {
                    if (batchIdx[imgIt] + batchIt * batchStep < batch) {
                        y[tid + imgIt * BLOCK + batchIt * batchStgStride] =
                            dataReg[batchIt][imgIt];
                    }
                }
            }
        }
    }
}

template <template<int NDIMS,
                   int IMAGE_UNROLL,
                   typename OffsetT,
                   typename T> class PadFunc,
          int NDIMS,
          typename T>
hiednnStatus_t LaunchPadKernel(
        size_t ySize,
        int64_t batch,
        int nDims,
        const int64_t *xDims,
        const int64_t *xStrides,
        const int64_t *yDims,
        const int64_t *yStrides,
        const int64_t *pads,
        const T *x,
        const T &param,
        T *y,
        cudaStream_t stream) {
    static_assert(sizeof(T) <= 16 && (sizeof(T) & (sizeof(T) - 1)) == 0,
                  "LaunchPadKernel: invalid typename T");
    const int BLOCK = 256;
    const int BATCH_UNROLL = 16 / sizeof(T);
    const int IMAGE_UNROLL = 64 / sizeof(T) < 16 ?
                             64 / sizeof(T) / BATCH_UNROLL : 16 / BATCH_UNROLL;

    // integer fast division only support uint32_t
    if (ySize > UINT32_MAX || U32DivMod::OutOfBound(yDims, nDims)) {
        return HIEDNN_STATUS_TENSOR_OVERSIZE;
    }

    using OffsetT = uint32_t;

    Array<U32DivMod, NDIMS> yDimDivMod;
    Array<OffsetT, NDIMS> lPadBound;
    Array<OffsetT, NDIMS> rPadBound;
    Array<OffsetT, NDIMS> xIdxMax;
    Array<OffsetT, NDIMS> xStride;

    for (int i = 0; i < nDims; ++i) {
        yDimDivMod[NDIMS - 1 - i] = U32DivMod(yDims[i]);
        lPadBound[NDIMS - 1 - i] = pads[i];
        rPadBound[NDIMS - 1 - i] = xDims[i] + pads[i];
        xIdxMax[NDIMS - 1 - i] = xDims[i] - 1;
        xStride[NDIMS - 1 - i] = xStrides[i];
    }

    for (int i = nDims; i < NDIMS; ++i) {
        yDimDivMod[NDIMS - 1 - i] = U32DivMod(1);
        lPadBound[NDIMS - 1 - i] = 0;
        rPadBound[NDIMS - 1 - i] = 0;
        xIdxMax[NDIMS - 1 - i] = 0;
        xStride[NDIMS - 1 - i] = xStrides[nDims - 1];
    }

    OffsetT batchStep = UIntDivRU<int64_t>(batch, BATCH_UNROLL);
    OffsetT xBatchStride = xStrides[0] * xDims[0];
    OffsetT batchLdgStride = xStrides[0] * xDims[0] * batchStep;
    OffsetT batchStgStride = yStrides[0] * yDims[0] * batchStep;

    OffsetT grid = UIntDivRU<int64_t>(yStrides[0] * yDims[0] * batchStep,
                                      BLOCK * IMAGE_UNROLL);

    CudaPadKernel<BLOCK, NDIMS, IMAGE_UNROLL, BATCH_UNROLL,
                  PadFunc<NDIMS, IMAGE_UNROLL, OffsetT, T>, OffsetT, T>
                  <<<grid, BLOCK, 0, stream>>>(
        x, y, param, yDimDivMod, lPadBound, rPadBound, xIdxMax, xStride, batch,
        batchStep, xBatchStride, batchLdgStride, batchStgStride);

    return HIEDNN_STATUS_SUCCESS;
}

template <template<int NDIMS,
                   int IMAGE_UNROLL,
                   typename OffsetT,
                   typename T> class PadFunc,
          typename T>
hiednnStatus_t CudaPad(
        size_t ySize,
        int64_t batch,
        int nDims,
        const int64_t *xDims,
        const int64_t *xStrides,
        const int64_t *yDims,
        const int64_t *yStrides,
        const int64_t *pads,
        const T *x,
        const T *paramPtr,
        T *y,
        cudaStream_t stream) {
    T param = paramPtr != nullptr ? *paramPtr : 0;

    if (nDims <= 1) {
        return LaunchPadKernel<PadFunc, 1, T>(
            ySize, batch, nDims, xDims, xStrides, yDims, yStrides,
            pads, x, param, y, stream);
    } else if (nDims <= 2) {
        return LaunchPadKernel<PadFunc, 2, T>(
            ySize, batch, nDims, xDims, xStrides, yDims, yStrides,
            pads, x, param, y, stream);
    } else if (nDims <= 3) {
        return LaunchPadKernel<PadFunc, 3, T>(
            ySize, batch, nDims, xDims, xStrides, yDims, yStrides,
            pads, x, param, y, stream);
    } else if (nDims <= 4) {
        return LaunchPadKernel<PadFunc, 4, T>(
            ySize, batch, nDims, xDims, xStrides, yDims, yStrides,
            pads, x, param, y, stream);
    } else if (nDims <= 6) {
        return LaunchPadKernel<PadFunc, 6, T>(
            ySize, batch, nDims, xDims, xStrides, yDims, yStrides,
            pads, x, param, y, stream);
    } else if (nDims <= 8) {
        return LaunchPadKernel<PadFunc, 8, T>(
            ySize, batch, nDims, xDims, xStrides, yDims, yStrides,
            pads, x, param, y, stream);
    } else {
        return HIEDNN_STATUS_INVALID_PARAMETER;
    }
}

void PrunePadParameter(
        const int64_t *xDims,
        const int64_t *pads,
        int nDims,
        int64_t *batchRet,
        int64_t *padsRet,
        int *prunedNDimsRet) {
    int64_t batch = 1;
    int prunedNDims = 0;

    for (int i = 0; i < nDims; ++i) {
        if (pads[i] != 0 || pads[i + nDims] != 0) {
            break;
        }
        batch *= xDims[i];
        ++prunedNDims;
    }

    int nPads = nDims - prunedNDims;
    for (int i = 0; i < nPads; ++i) {
        padsRet[i] = pads[prunedNDims + i];
        padsRet[i + nPads] = pads[prunedNDims + i + nDims];
    }

    *batchRet = batch;
    *prunedNDimsRet = prunedNDims;
}

template <typename T>
struct PadImpl {
    hiednnStatus_t operator()(
            hiednnPadMode_t mode,
            const HiednnTensorDesc &xDesc,
            const void *x,
            const int64_t *pads,
            const void *param,
            const HiednnTensorDesc &yDesc,
            void *y,
            cudaStream_t stream) {
        // Parameter Preprocess
        int64_t batch;
        int64_t prunedPads[TENSOR_DIM_MAX * 2];
        int prunedNDims;
        PrunePadParameter(xDesc.dims, pads, xDesc.nDims,
                          &batch, prunedPads, &prunedNDims);

        const int64_t *xDims = xDesc.dims + prunedNDims;
        const int64_t *xStride = xDesc.strides + prunedNDims;
        const int64_t *yDims = yDesc.dims + prunedNDims;
        const int64_t *yStride = yDesc.strides + prunedNDims;
        int nDims = xDesc.nDims - prunedNDims;
        size_t ySize = yDesc.size;

        const T *xPtr = static_cast<const T *>(x);
        const T *paramPtr = static_cast<const T *>(param);
        T *yPtr = static_cast<T *>(y);

        // Pad
        if (nDims == 0) {
            // no padding, just copy x to y
            CHECK_CUDA_RETURN(cudaMemcpyAsync(
                yPtr, xPtr, xDesc.size * sizeof(T), cudaMemcpyDefault, stream));
            return HIEDNN_STATUS_SUCCESS;
        }

        hiednnStatus_t ret = HIEDNN_STATUS_SUCCESS;
        switch (mode) {
            case HIEDNN_PAD_CONST:
                ret = CudaPad<PadConst, T>(
                    ySize, batch, nDims, xDims, xStride, yDims, yStride,
                    prunedPads, xPtr, paramPtr, yPtr, stream);
                break;
            case HIEDNN_PAD_EDGE:
                ret = CudaPad<PadEdge, T>(
                    ySize, batch, nDims, xDims, xStride, yDims, yStride,
                    prunedPads, xPtr, nullptr, yPtr, stream);
                break;
            case HIEDNN_PAD_REFLECT:
                ret = CudaPad<PadReflect, T>(
                    ySize, batch, nDims, xDims, xStride, yDims, yStride,
                    prunedPads, xPtr, nullptr, yPtr, stream);
                break;
            default:
                ret = HIEDNN_STATUS_INVALID_PARAMETER;
                break;
        }

        return ret;
    }
};

bool CheckParameter(const HiednnTensorDesc &xDesc,
                    const HiednnTensorDesc &yDesc,
                    hiednnPadMode_t mode,
                    const int64_t *pads,
                    int nPads) {
    if (xDesc.nDims != yDesc.nDims) {
        return false;
    }

    const int64_t *xDims = xDesc.dims;
    const int64_t *yDims = yDesc.dims;
    int nDims = xDesc.nDims;
    const int64_t *lPads = pads;
    const int64_t *rPads = pads + nDims;

    if (nPads != nDims * 2) {
        return false;
    }

    // check pads
    for (int i = 0; i < nPads; ++i) {
        if (pads[i] < 0) {
            return false;
        }
    }

    if (mode == HIEDNN_PAD_REFLECT) {
        for (int i = 0; i < nDims; ++i) {
            if (pads[i] > xDims[i] - 1 || pads[i + nDims] > xDims[i] - 1) {
                return false;
            }
        }
    }

    // check tensor shape
    for (int i = 0; i < nDims; ++i) {
        if (yDims[i] != xDims[i] + lPads[i] + rPads[i]) {
            return false;
        }
    }

    return true;
}

}  // anonymous namespace

}  // namespace pad

}  // namespace cuda

}  // namespace hiednn

hiednnStatus_t
hiednnCudaPad(HiednnCudaHandle *cudaHandle,
              hiednnPadMode_t mode,
              HiednnTensorDesc *xDesc,
              const void *x,
              const int64_t *pads,
              int nPads,
              const void *param,
              HiednnTensorDesc *yDesc,
              void *y) {
    if (!hiednn::CheckNullptr(cudaHandle, xDesc, pads, yDesc) ||
        !hiednn::CheckTensorPtr(*xDesc, x, *yDesc, y)) {
        return HIEDNN_STATUS_INVALID_PARAMETER;
    }

    if (!hiednn::CheckNormalFormat(*xDesc, *yDesc)) {
        return HIEDNN_STATUS_INVALID_PARAMETER;
    }

    if (!hiednn::cuda::pad::CheckParameter(*xDesc, *yDesc, mode, pads, nPads)) {
        return HIEDNN_STATUS_INVALID_PARAMETER;
    }

    if (yDesc->size == 0) {
        return HIEDNN_STATUS_SUCCESS;
    }

    return hiednn::DispatchItemSize<hiednn::cuda::pad::PadImpl>(
        xDesc->dataType, mode, *xDesc, x, pads, param, *yDesc, y,
        cudaHandle->stream);
}


