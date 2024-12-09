/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    cast.cu
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

template <int BLOCK, int UNROLL, typename ST, typename DT>
__global__ void CastKernel(const ST *x, DT *y, PackedEltwiseConfig packConfig) {
    int64_t tid = static_cast<int64_t>(blockIdx.x) * BLOCK + threadIdx.x;

    if (tid < packConfig.nPack) {
        using V_ST = VT<ST, UNROLL>;
        using V_DT = VT<DT, UNROLL>;
        V_ST ldReg;
        V_DT stReg;

        Ldg(&ldReg, reinterpret_cast<const V_ST *>(x) + tid);

        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            stReg.data[i] = static_cast<DT>(ldReg.data[i]);
        }
        Stg(stReg, reinterpret_cast<V_DT *>(y) + tid);
    } else if (UNROLL > 1 && tid < packConfig.nThread) {
        int64_t idx = tid + packConfig.unpackedOffset;
        y[idx] = static_cast<DT>(x[idx]);
    }
}

template <typename ST>
struct CastImpl {
    template <typename DT>
    struct CastDispatchDst {
        hiednnStatus_t operator()(const void *x,
                                  void *y,
                                  size_t n,
                                  cudaStream_t stream) {
            const int64_t BLOCK = 128;

            const ST *xPtr = static_cast<const ST *>(x);
            DT *yPtr = static_cast<DT *>(y);
            int packSize = std::min(GetPackSize(xPtr),
                                    GetPackSize(yPtr));

            hiednnStatus_t ret = HIEDNN_STATUS_SUCCESS;
            switch (packSize) {
                case 8: {
                    const int UNROLL = ValidPack<ST, DT, 8>();
                    PackedEltwiseConfig packConfig(n, UNROLL, BLOCK);
                    CastKernel<BLOCK, UNROLL>
                        <<<packConfig.nBlock, BLOCK, 0, stream>>>(
                        xPtr, yPtr, packConfig);
                    break;
                }
                case 4: {
                    const int UNROLL = ValidPack<ST, DT, 4>();
                    PackedEltwiseConfig packConfig(n, UNROLL, BLOCK);
                    CastKernel<BLOCK, UNROLL>
                        <<<packConfig.nBlock, BLOCK, 0, stream>>>(
                        xPtr, yPtr, packConfig);
                    break;
                }
                case 2: {
                    const int UNROLL = ValidPack<ST, DT, 2>();
                    PackedEltwiseConfig packConfig(n, UNROLL, BLOCK);
                    CastKernel<BLOCK, UNROLL>
                        <<<packConfig.nBlock, BLOCK, 0, stream>>>(
                        xPtr, yPtr, packConfig);
                    break;
                }
                case 1: {
                    PackedEltwiseConfig packConfig(n, 1, BLOCK);
                    CastKernel<BLOCK, 1>
                        <<<packConfig.nBlock, BLOCK, 0, stream>>>(
                        xPtr, yPtr, packConfig);
                    break;
                }
                default:
                    ret = HIEDNN_STATUS_INTERNAL_ERROR;
                    break;
            }

            return ret;
        }
    };

    hiednnStatus_t operator()(hiednnDataType_t dstType,
                              const void *x,
                              void *y,
                              size_t n,
                              cudaStream_t stream) {
        return DispatchAll<CastDispatchDst>(dstType, x, y, n, stream);
    }
};

}  // anonymous namespace

}  // namespace cuda

}  // namespace hiednn

hiednnStatus_t
hiednnCudaCast(HiednnCudaHandle *cudaHandle,
               HiednnTensorDesc *xDesc,
               const void *x,
               HiednnTensorDesc *yDesc,
               void *y) {
    if (!hiednn::CheckNullptr(cudaHandle, xDesc, yDesc) ||
        !hiednn::CheckTensorPtr(*xDesc, x, *yDesc, y)) {
        return HIEDNN_STATUS_INVALID_PARAMETER;
    }

    if (!xDesc->SameDimStride(*yDesc) || !xDesc->contiguous) {
        return HIEDNN_STATUS_INVALID_PARAMETER;
    }

    if (yDesc->size == 0) {
        return HIEDNN_STATUS_SUCCESS;
    }

    return hiednn::DispatchAll<hiednn::cuda::CastImpl>(xDesc->dataType,
                                                       yDesc->dataType,
                                                       x, y, xDesc->size,
                                                       cudaHandle->stream);
}


