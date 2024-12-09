/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    binary_elementwise.cu
 */

#include <hiednn.h>
#include <hiednn_cuda.h>

#include <cstdint>
#include <algorithm>

#include <utils.hpp>
#include <tensor_desc.hpp>
#include <datatype_dispatch.hpp>
#include <scalar_functor.hpp>
#include <integer_divmod.hpp>
#include <packed_memory_access.hpp>

#include <cuda/cuda_handle.hpp>
#include <cuda/cuda_utils.hpp>
#include <cuda/intrinsic/global_memory.hpp>

namespace hiednn {

namespace cuda {

namespace {

template <int BLOCK, int UNROLL, bool BETA, typename T, typename Op>
__global__ void BinaryElementwiseKernel(
        const T *x, const T *y, T *z, T alpha, T beta, Op op,
        PackedEltwiseConfig packConfig) {
    int64_t tid = static_cast<int64_t>(blockIdx.x) * BLOCK + threadIdx.x;

    if (tid < packConfig.nPack) {
        using V_T = VT<T, UNROLL>;
        V_T xReg, yReg, zReg;

        Ldg(&xReg, reinterpret_cast<const V_T *>(x) + tid);
        Ldg(&yReg, reinterpret_cast<const V_T *>(y) + tid);

        if (BETA) {
            Ldg(&zReg, reinterpret_cast<const V_T *>(z) + tid);
            #pragma unroll
            for (int i = 0; i < UNROLL; ++i) {
                zReg.data[i] = alpha * op(xReg.data[i], yReg.data[i]) +
                               beta * zReg.data[i];
            }
        } else {
            #pragma unroll
            for (int i = 0; i < UNROLL; ++i) {
                zReg.data[i] = alpha * op(xReg.data[i], yReg.data[i]);
            }
        }

        Stg(zReg, reinterpret_cast<V_T *>(z) + tid);
    } else if (UNROLL > 1 && tid < packConfig.nThread) {
        int64_t idx = tid + packConfig.unpackedOffset;
        z[idx] = BETA ?
                 alpha * op(x[idx], y[idx]) + beta * z[idx] :
                 alpha * op(x[idx], y[idx]);
    }
}

template <int BLOCK, int UNROLL, typename T, typename Op>
__global__ void BinaryElementwiseLogicalKernel(
        const T *x, const T *y, char *z, Op op,
        PackedEltwiseConfig packConfig) {
    int64_t tid = static_cast<int64_t>(blockIdx.x) * BLOCK + threadIdx.x;

    if (tid < packConfig.nPack) {
        using V_IN = VT<T, UNROLL>;
        using V_OUT = VT<char, UNROLL>;
        V_IN xReg, yReg;
        V_OUT zReg;

        Ldg(&xReg, reinterpret_cast<const V_IN *>(x) + tid);
        Ldg(&yReg, reinterpret_cast<const V_IN *>(y) + tid);

        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            zReg.data[i] = op(xReg.data[i], yReg.data[i]);
        }

        Stg(zReg, reinterpret_cast<V_OUT *>(z) + tid);
    } else if (UNROLL > 1 && tid < packConfig.nThread) {
        int64_t idx = tid + packConfig.unpackedOffset;
        z[idx] = op(x[idx], y[idx]);
    }
}

template <uint32_t BLOCK, uint32_t UNROLL>
__device__ __forceinline__
void BinaryBroadcastOffset(
        const Array<U32DivMod, TENSOR_DIM_MAX> &zDivMod,
        const Array<uint32_t, TENSOR_DIM_MAX> &xStride,
        const Array<uint32_t, TENSOR_DIM_MAX> &yStride,
        const int &ndim,
        const uint32_t &zOffsetStart,
        uint32_t (&xOffset)[UNROLL],
        uint32_t (&yOffset)[UNROLL]) {
    uint32_t divTmp[UNROLL];

    #pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
        xOffset[i] = 0;
        yOffset[i] = 0;
        divTmp[i] = zOffsetStart + i * BLOCK;
    }

    #pragma unroll
    for (int i = 0; i < TENSOR_DIM_MAX; ++i) {
        if (i >= ndim) break;

        #pragma unroll
        for (int j = 0; j < UNROLL; ++j) {
            auto dm = zDivMod[i].DivMod(divTmp[j]);
            uint32_t idx = dm.mod;
            divTmp[j] = dm.div;
            xOffset[j] += idx * xStride[i];
            yOffset[j] += idx * yStride[i];
        }
    }
}

template <uint32_t BLOCK, uint32_t UNROLL, bool BETA, typename T, typename Op>
__global__ void BinaryElementwiseBroadcastKernel(
        const T *__restrict__ x, const T *__restrict__ y, T *z,
        uint32_t n, int ndim, T alpha, T beta,
        Array<U32DivMod, TENSOR_DIM_MAX> zDivMod,
        Array<uint32_t, TENSOR_DIM_MAX> xStride,
        Array<uint32_t, TENSOR_DIM_MAX> yStride, Op op) {
    uint32_t xOffset[UNROLL];
    uint32_t yOffset[UNROLL];
    uint32_t zOffset = blockIdx.x * BLOCK * UNROLL + threadIdx.x;

    BinaryBroadcastOffset<BLOCK, UNROLL>(
        zDivMod, xStride, yStride, ndim, zOffset, xOffset, yOffset);

    T xReg[UNROLL];
    T yReg[UNROLL];
    T zReg[UNROLL];

    uint32_t zCount = n > zOffset ? UIntDivRU<uint32_t>(n - zOffset, BLOCK) : 0;

    if (zCount >= UNROLL) {
        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            xReg[i] = x[xOffset[i]];
            yReg[i] = y[yOffset[i]];
            if (BETA) {
                zReg[i] = z[zOffset + i * BLOCK];
            }
        }

        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            zReg[i] = BETA ?
                      alpha * op(xReg[i], yReg[i]) + beta * zReg[i] :
                      alpha * op(xReg[i], yReg[i]);
        }

        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            z[zOffset + i * BLOCK] = zReg[i];
        }
    } else {
        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            if (i < zCount) {
                xReg[i] = x[xOffset[i]];
                yReg[i] = y[yOffset[i]];
                if (BETA) {
                    zReg[i] = z[zOffset + i * BLOCK];
                }
            }
        }

        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            zReg[i] = BETA ?
                      alpha * op(xReg[i], yReg[i]) + beta * zReg[i] :
                      alpha * op(xReg[i], yReg[i]);
        }

        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            if (i < zCount) {
                z[zOffset + i * BLOCK] = zReg[i];
            }
        }
    }
}

template <uint32_t BLOCK, uint32_t UNROLL, typename T, typename Op>
__global__ void BinaryElementwiseBroadcastLogicalKernel(
        const T *__restrict__ x, const T *__restrict__ y, char *z,
        size_t n, int ndim,
        Array<U32DivMod, TENSOR_DIM_MAX> zDivMod,
        Array<uint32_t, TENSOR_DIM_MAX> xStride,
        Array<uint32_t, TENSOR_DIM_MAX> yStride, Op op) {
    uint32_t xOffset[UNROLL];
    uint32_t yOffset[UNROLL];
    uint32_t zOffset = blockIdx.x * BLOCK * UNROLL + threadIdx.x;

    BinaryBroadcastOffset<BLOCK, UNROLL>(
        zDivMod, xStride, yStride, ndim, zOffset, xOffset, yOffset);

    T xReg[UNROLL];
    T yReg[UNROLL];
    char zReg[UNROLL];

    uint32_t zCount = n > zOffset ? UIntDivRU<uint32_t>(n - zOffset, BLOCK) : 0;

    if (zCount >= UNROLL) {
        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            xReg[i] = x[xOffset[i]];
            yReg[i] = y[yOffset[i]];
        }

        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            zReg[i] = op(xReg[i], yReg[i]);
        }

        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            z[zOffset + i * BLOCK] = zReg[i];
        }
    } else {
        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            if (i < zCount) {
                xReg[i] = x[xOffset[i]];
                yReg[i] = y[yOffset[i]];
            }
        }

        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            zReg[i] = op(xReg[i], yReg[i]);
        }

        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            if (i < zCount) {
                z[zOffset + i * BLOCK] = zReg[i];
            }
        }
    }
}

template <typename T, typename ScalarOp>
hiednnStatus_t
LaunchBinaryElementwiseKernel(
        const T *x, const T *y, T *z, size_t n, T alpha, T beta,
        ScalarOp scalarOp, cudaStream_t stream) {
    const int64_t BLOCK = 128;
    int packSize = std::min(std::min(GetPackSize<T>(x), GetPackSize<T>(y)),
                            GetPackSize<T>(z));

    hiednnStatus_t ret = HIEDNN_STATUS_SUCCESS;
    if (beta == 0) {
        switch (packSize) {
            case 8: {
                const int UNROLL = ValidPack<T, 8>();
                PackedEltwiseConfig packConfig(n, UNROLL, BLOCK);
                BinaryElementwiseKernel<BLOCK, UNROLL, false>
                    <<<packConfig.nBlock, BLOCK, 0, stream>>>(
                    x, y, z, alpha, beta, scalarOp, packConfig);
                break;
            }
            case 4: {
                const int UNROLL = ValidPack<T, 4>();
                PackedEltwiseConfig packConfig(n, UNROLL, BLOCK);
                BinaryElementwiseKernel<BLOCK, UNROLL, false>
                    <<<packConfig.nBlock, BLOCK, 0, stream>>>(
                    x, y, z, alpha, beta, scalarOp, packConfig);
                break;
            }
            case 2: {
                const int UNROLL = ValidPack<T, 2>();
                PackedEltwiseConfig packConfig(n, UNROLL, BLOCK);
                BinaryElementwiseKernel<BLOCK, UNROLL, false>
                    <<<packConfig.nBlock, BLOCK, 0, stream>>>(
                    x, y, z, alpha, beta, scalarOp, packConfig);
                break;
            }
            case 1: {
                PackedEltwiseConfig packConfig(n, 1, BLOCK);
                BinaryElementwiseKernel<BLOCK, 1, false>
                    <<<packConfig.nBlock, BLOCK, 0, stream>>>(
                    x, y, z, alpha, beta, scalarOp, packConfig);
                break;
            }
            default:
                ret = HIEDNN_STATUS_INTERNAL_ERROR;
                break;
        }
    } else {
        switch (packSize) {
            case 8: {
                const int UNROLL = ValidPack<T, 8>();
                PackedEltwiseConfig packConfig(n, UNROLL, BLOCK);
                BinaryElementwiseKernel<BLOCK, UNROLL, true>
                    <<<packConfig.nBlock, BLOCK, 0, stream>>>(
                    x, y, z, alpha, beta, scalarOp, packConfig);
                break;
            }
            case 4: {
                const int UNROLL = ValidPack<T, 4>();
                PackedEltwiseConfig packConfig(n, UNROLL, BLOCK);
                BinaryElementwiseKernel<BLOCK, UNROLL, true>
                    <<<packConfig.nBlock, BLOCK, 0, stream>>>(
                    x, y, z, alpha, beta, scalarOp, packConfig);
                break;
            }
            case 2: {
                const int UNROLL = ValidPack<T, 2>();
                PackedEltwiseConfig packConfig(n, UNROLL, BLOCK);
                BinaryElementwiseKernel<BLOCK, UNROLL, true>
                    <<<packConfig.nBlock, BLOCK, 0, stream>>>(
                    x, y, z, alpha, beta, scalarOp, packConfig);
                break;
            }
            case 1: {
                PackedEltwiseConfig packConfig(n, 1, BLOCK);
                BinaryElementwiseKernel<BLOCK, 1, true>
                    <<<packConfig.nBlock, BLOCK, 0, stream>>>(
                    x, y, z, alpha, beta, scalarOp, packConfig);
                break;
            }
            default:
                ret = HIEDNN_STATUS_INTERNAL_ERROR;
                break;
        }
    }

    return ret;
}

template <typename T, typename ScalarOp>
hiednnStatus_t
LaunchBinaryElementwiseLogicalKernel(
        const T *x, const T *y, char *z, size_t n,
        ScalarOp scalarOp, cudaStream_t stream) {
    const int64_t BLOCK = 128;
    int packSize = std::min(std::min(GetPackSize<T>(x), GetPackSize<T>(y)),
                            GetPackSize<char>(z));

    hiednnStatus_t ret = HIEDNN_STATUS_SUCCESS;
    switch (packSize) {
        case 8: {
            const int UNROLL = ValidPack<T, 8>();
            PackedEltwiseConfig packConfig(n, UNROLL, BLOCK);
            BinaryElementwiseLogicalKernel<BLOCK, UNROLL>
                <<<packConfig.nBlock, BLOCK, 0, stream>>>(
                x, y, z, scalarOp, packConfig);
            break;
        }
        case 4: {
            const int UNROLL = ValidPack<T, 4>();
            PackedEltwiseConfig packConfig(n, UNROLL, BLOCK);
            BinaryElementwiseLogicalKernel<BLOCK, UNROLL>
                <<<packConfig.nBlock, BLOCK, 0, stream>>>(
                x, y, z, scalarOp, packConfig);
            break;
        }
        case 2: {
            const int UNROLL = ValidPack<T, 2>();
            PackedEltwiseConfig packConfig(n, UNROLL, BLOCK);
            BinaryElementwiseLogicalKernel<BLOCK, UNROLL>
                <<<packConfig.nBlock, BLOCK, 0, stream>>>(
                x, y, z, scalarOp, packConfig);
            break;
        }
        case 1: {
            PackedEltwiseConfig packConfig(n, 1, BLOCK);
            BinaryElementwiseLogicalKernel<BLOCK, 1>
                <<<packConfig.nBlock, BLOCK, 0, stream>>>(
                x, y, z, scalarOp, packConfig);
            break;
        }
        default:
            ret = HIEDNN_STATUS_INTERNAL_ERROR;
            break;
    }

    return ret;
}

template <typename T, typename ScalarOp>
hiednnStatus_t
LaunchBinaryElementwiseBroadcastKernel(
        const T *x, const T *y, T *z, size_t n, int ndim, T alpha, T beta,
        Array<U32DivMod, TENSOR_DIM_MAX> z_divmod,
        Array<uint32_t, TENSOR_DIM_MAX> x_stride,
        Array<uint32_t, TENSOR_DIM_MAX> y_stride,
        ScalarOp scalarOp, cudaStream_t stream) {
    const uint32_t BLOCK = 128;
    const uint32_t UNROLLED_BYTE = 16;
    const uint32_t UNROLL = UNROLLED_BYTE / sizeof(T) <= 8 ?
                            UNROLLED_BYTE / sizeof(T) : 8;

    uint32_t nBlock = UIntDivRU(static_cast<uint32_t>(n), BLOCK * UNROLL);

    if (beta == 0) {
        BinaryElementwiseBroadcastKernel<BLOCK, UNROLL, false>
            <<<nBlock, BLOCK, 0, stream>>>(
            x, y, z, n, ndim, alpha, beta,
            z_divmod, x_stride, y_stride, scalarOp);
    } else {
        BinaryElementwiseBroadcastKernel<BLOCK, UNROLL, true>
            <<<nBlock, BLOCK, 0, stream>>>(
            x, y, z, n, ndim, alpha, beta,
            z_divmod, x_stride, y_stride, scalarOp);
    }

    return HIEDNN_STATUS_SUCCESS;
}

template <typename T, typename ScalarOp>
hiednnStatus_t
LaunchBinaryElementwiseBroadcastLogicalKernel(
        const T *x, const T *y, char *z, size_t n, int ndim,
        Array<U32DivMod, TENSOR_DIM_MAX> z_divmod,
        Array<uint32_t, TENSOR_DIM_MAX> x_stride,
        Array<uint32_t, TENSOR_DIM_MAX> y_stride,
        ScalarOp scalarOp, cudaStream_t stream) {
    const uint32_t BLOCK = 128;
    const uint32_t UNROLLED_BYTE = 16;
    const uint32_t UNROLL = UNROLLED_BYTE / sizeof(T) <= 8 ?
                            UNROLLED_BYTE / sizeof(T) : 8;

    uint32_t nBlock = UIntDivRU(static_cast<uint32_t>(n), BLOCK * UNROLL);

    BinaryElementwiseBroadcastLogicalKernel<BLOCK, UNROLL>
        <<<nBlock, BLOCK, 0, stream>>>(
        x, y, z, n, ndim, z_divmod, x_stride, y_stride, scalarOp);

    return HIEDNN_STATUS_SUCCESS;
}

// impl functor for binary-map, such as z = x + y
template <typename T>
struct BinaryElementwiseImpl {
    template <typename GetOp, typename ...Arg>
    hiednnStatus_t operator()(const void *x,
                              const HiednnTensorDesc &xDesc,
                              const void *y,
                              const HiednnTensorDesc &yDesc,
                              void *z,
                              const HiednnTensorDesc &zDesc,
                              const void *alpha,
                              const void *beta,
                              cudaStream_t stream,
                              GetOp getop,
                              Arg&&... args) {
        if (zDesc.dataType != xDesc.dataType ||
            zDesc.dataType != yDesc.dataType) {
            return HIEDNN_STATUS_INVALID_DATATYPE;
        }

        const T *x_ptr = static_cast<const T *>(x);
        const T *y_ptr = static_cast<const T *>(y);
        T alpha_val = *reinterpret_cast<const T *>(alpha);
        T beta_val = *reinterpret_cast<const T *>(beta);
        T *z_ptr = static_cast<T *>(z);

        auto scalar_op = getop.template get<T>(std::forward<Arg>(args)...);

        hiednnStatus_t ret = HIEDNN_STATUS_SUCCESS;
        if (xDesc.size == yDesc.size && xDesc.size == zDesc.size) {
            // non-broadcast
            size_t n = zDesc.size;
            ret = LaunchBinaryElementwiseKernel<T>(
                x_ptr, y_ptr, z_ptr, n, alpha_val, beta_val, scalar_op, stream);
        } else {
            // broadcast

            // U32DivMod only work for integers from 0 to INT32_MAX
            if (zDesc.size <= UINT32_MAX &&
                !U32DivMod::OutOfBound(zDesc.dims, zDesc.nDims)) {
                size_t n = zDesc.size;
                int x_ndim = xDesc.nDims;
                int y_ndim = yDesc.nDims;
                int z_ndim = zDesc.nDims;
                Array<uint32_t, TENSOR_DIM_MAX> x_stride;
                Array<uint32_t, TENSOR_DIM_MAX> y_stride;
                Array<U32DivMod, TENSOR_DIM_MAX> z_divmod;
                for (int i = 0; i < z_ndim; ++i) {
                    x_stride[i] = i < x_ndim && xDesc.dims[x_ndim - i - 1] > 1 ?
                                  xDesc.strides[x_ndim - i - 1] : 0;
                    y_stride[i] = i < y_ndim && yDesc.dims[y_ndim - i - 1] > 1 ?
                                  yDesc.strides[y_ndim - i - 1] : 0;
                    z_divmod[i] = U32DivMod(zDesc.dims[z_ndim - i - 1]);
                }
                ret = LaunchBinaryElementwiseBroadcastKernel<T>(
                    x_ptr, y_ptr, z_ptr, n, z_ndim, alpha_val, beta_val,
                    z_divmod, x_stride, y_stride, scalar_op, stream);
            } else {
                ret = HIEDNN_STATUS_TENSOR_OVERSIZE;
            }
        }

        return ret;
    }
};

template <typename T>
struct BinaryElementwiseLogicalImpl {
    template <typename GetOp, typename ...Arg>
    hiednnStatus_t operator()(const void *x,
                              const HiednnTensorDesc &xDesc,
                              const void *y,
                              const HiednnTensorDesc &yDesc,
                              void *z,
                              const HiednnTensorDesc &zDesc,
                              cudaStream_t stream,
                              GetOp getop,
                              Arg&&... args) {
        if (zDesc.dataType != HIEDNN_DATATYPE_BOOL) {
            return HIEDNN_STATUS_INVALID_DATATYPE;
        }

        const T *x_ptr = static_cast<const T *>(x);
        const T *y_ptr = static_cast<const T *>(y);
        char *z_ptr = static_cast<char *>(z);

        auto scalar_op = getop.template get<T>(std::forward<Arg>(args)...);

        hiednnStatus_t ret = HIEDNN_STATUS_SUCCESS;
        if (xDesc.size == yDesc.size && xDesc.size == zDesc.size) {
            // non-broadcast
            size_t n = zDesc.size;
            ret = LaunchBinaryElementwiseLogicalKernel<T>(
                x_ptr, y_ptr, z_ptr, n, scalar_op, stream);
        } else {
            // broadcast
            // U32DivMod only work for integers from 0 to INT32_MAX
            if (zDesc.size <= UINT32_MAX &&
                !U32DivMod::OutOfBound(zDesc.dims, zDesc.nDims)) {
                size_t n = zDesc.size;
                int x_ndim = xDesc.nDims;
                int y_ndim = yDesc.nDims;
                int z_ndim = zDesc.nDims;
                Array<uint32_t, TENSOR_DIM_MAX> x_stride;
                Array<uint32_t, TENSOR_DIM_MAX> y_stride;
                Array<U32DivMod, TENSOR_DIM_MAX> z_divmod;
                for (int i = 0; i < z_ndim; ++i) {
                    x_stride[i] = i < x_ndim && xDesc.dims[x_ndim - i - 1] > 1 ?
                                  xDesc.strides[x_ndim - i - 1] : 0;
                    y_stride[i] = i < y_ndim && yDesc.dims[y_ndim - i - 1] > 1 ?
                                  yDesc.strides[y_ndim - i - 1] : 0;
                    z_divmod[i] = U32DivMod(zDesc.dims[z_ndim - i - 1]);
                }
                ret = LaunchBinaryElementwiseBroadcastLogicalKernel<T>(
                    x_ptr, y_ptr, z_ptr, n, z_ndim, z_divmod,
                    x_stride, y_stride, scalar_op, stream);
            } else {
                ret = HIEDNN_STATUS_TENSOR_OVERSIZE;
            }
        }

        return ret;
    }
};

}  // anonymous namespace

}  // namespace cuda

}  // namespace hiednn

// switch-case loops for binary elementwise OP
#define BINARY_CASE_LOOP(TAG, DATATYPE, FUNC) \
    case TAG: \
        ret = hiednn::Dispatch##DATATYPE< \
                hiednn::cuda::BinaryElementwiseImpl>( \
            xDesc->dataType, x, *xDesc, y, *yDesc, z, *zDesc, alpha, beta, \
            cudaHandle->stream, \
            hiednn::GetScalarOp<hiednn::scalar_functor::FUNC>()); \
        break;

#define BINARY_CMP_LOOP(TAG, DATATYPE, FUNC) \
    case TAG: \
        ret = hiednn::Dispatch##DATATYPE< \
                hiednn::cuda::BinaryElementwiseLogicalImpl>( \
            xDesc->dataType, x, *xDesc, y, *yDesc, z, *zDesc, \
            cudaHandle->stream, \
            hiednn::GetScalarOp<hiednn::scalar_functor::FUNC>()); \
        break;

#define BINARY_LOGICAL_LOOP(TAG, FUNC) \
    case TAG: \
        ret = hiednn::cuda::BinaryElementwiseLogicalImpl<char>()( \
            x, *xDesc, y, *yDesc, z, *zDesc, cudaHandle->stream, \
            hiednn::GetScalarOp<hiednn::scalar_functor::FUNC>()); \
        break;

hiednnStatus_t
hiednnCudaBinaryElementwiseOp(HiednnCudaHandle *cudaHandle,
                              hiednnBinaryEltwiseOp_t binaryEltwiseOp,
                              const void *alpha,
                              HiednnTensorDesc *xDesc,
                              const void *x,
                              HiednnTensorDesc *yDesc,
                              const void *y,
                              const void *extParam,
                              const void *beta,
                              HiednnTensorDesc *zDesc,
                              void *z) {
    if (!hiednn::CheckNullptr(cudaHandle, xDesc, yDesc, zDesc) ||
        !hiednn::CheckTensorPtr(*xDesc, x, *yDesc, y, *zDesc, z)) {
        return HIEDNN_STATUS_INVALID_PARAMETER;
    }

    if (!hiednn::CheckNormalFormat(*xDesc, *yDesc, *zDesc)) {
        return HIEDNN_STATUS_INVALID_PARAMETER;
    }

    if (xDesc->dataType != yDesc->dataType) {
        return HIEDNN_STATUS_INVALID_PARAMETER;
    }

    // just make sure that zDesc.size > broadcast(xDesc, yDesc).size
    if (!xDesc->UniBroadcastableTo(*zDesc) ||
        !yDesc->UniBroadcastableTo(*zDesc)) {
        return HIEDNN_STATUS_INVALID_PARAMETER;
    }

    if (zDesc->size == 0) {
        return HIEDNN_STATUS_SUCCESS;
    }

    hiednnStatus_t ret = HIEDNN_STATUS_SUCCESS;
    switch (binaryEltwiseOp) {
        // --------------------------------------
        // support all datatype
        // --------------------------------------
        BINARY_CASE_LOOP(HIEDNN_BINARY_MATH_ADD,
                         All,
                         Add)
        BINARY_CASE_LOOP(HIEDNN_BINARY_MATH_SUB,
                         All,
                         Sub)
        BINARY_CASE_LOOP(HIEDNN_BINARY_MATH_MUL,
                         All,
                         Mul)
        BINARY_CASE_LOOP(HIEDNN_BINARY_MATH_DIV,
                         All,
                         Div)
        BINARY_CASE_LOOP(HIEDNN_BINARY_MATH_MAX,
                         All,
                         Max)
        BINARY_CASE_LOOP(HIEDNN_BINARY_MATH_MIN,
                         All,
                         Min)

        case HIEDNN_BINARY_MATH_MOD: {
            const int &fmod = *static_cast<const int *>(extParam);
            // if input type is interger type, fmod can be 0 or 1
            // if input type is floating type, fmod can only be 1
            if (fmod == 0) {
                ret = hiednn::DispatchInt<
                        hiednn::cuda::BinaryElementwiseImpl>(
                    xDesc->dataType, x, *xDesc, y, *yDesc, z, *zDesc,
                    alpha, beta, cudaHandle->stream,
                    hiednn::GetScalarOp<hiednn::scalar_functor::Mod>());
            } else if (fmod == 1) {
                ret = hiednn::DispatchAll<
                        hiednn::cuda::BinaryElementwiseImpl>(
                    xDesc->dataType, x, *xDesc, y, *yDesc, z, *zDesc,
                    alpha, beta, cudaHandle->stream,
                    hiednn::GetScalarOp<hiednn::scalar_functor::FMod>());
            } else {
                ret = HIEDNN_STATUS_INVALID_PARAMETER;
            }
            break;
        }

        case HIEDNN_BINARY_MATH_PRELU: {
            if (!yDesc->UniBroadcastableTo(*xDesc)) {
                ret = HIEDNN_STATUS_INVALID_PARAMETER;
            } else {
                ret = hiednn::DispatchAll<
                    hiednn::cuda::BinaryElementwiseImpl>(xDesc->dataType,
                            x, *xDesc, y, *yDesc, z, *zDesc,
                            alpha, beta, cudaHandle->stream,
                            hiednn::GetScalarOp<
                            hiednn::scalar_functor::PRelu>());
            }
            break;
        }

        BINARY_CMP_LOOP(HIEDNN_BINARY_COMPARE_EQ,
                        All,
                        CompareEQ)
        BINARY_CMP_LOOP(HIEDNN_BINARY_COMPARE_GT,
                        All,
                        CompareGT)
        BINARY_CMP_LOOP(HIEDNN_BINARY_COMPARE_GE,
                        All,
                        CompareGE)
        BINARY_CMP_LOOP(HIEDNN_BINARY_COMPARE_LT,
                        All,
                        CompareLT)
        BINARY_CMP_LOOP(HIEDNN_BINARY_COMPARE_LE,
                        All,
                        CompareLE)

        // --------------------------------------
        // only support unsigned
        // --------------------------------------
        case HIEDNN_BINARY_MATH_BITSHIFT: {
            const int &direction = *static_cast<const int *>(extParam);
            // direction == 0 indicate shift left
            // direction == 1 indicate shift right
            if (direction == 0) {
                ret = hiednn::DispatchUnsigned<
                        hiednn::cuda::BinaryElementwiseImpl>(
                    xDesc->dataType, x, *xDesc, y, *yDesc, z, *zDesc,
                    alpha, beta, cudaHandle->stream,
                    hiednn::GetScalarOp<
                    hiednn::scalar_functor::LogicalShiftL>());
            } else if (direction == 1) {
                ret = hiednn::DispatchUnsigned<
                        hiednn::cuda::BinaryElementwiseImpl>(
                    xDesc->dataType, x, *xDesc, y, *yDesc, z, *zDesc,
                    alpha, beta, cudaHandle->stream,
                    hiednn::GetScalarOp<
                    hiednn::scalar_functor::LogicalShiftR>());
            } else {
                ret = HIEDNN_STATUS_INVALID_PARAMETER;
            }
            break;
        }

        // --------------------------------------
        // only support bool
        // --------------------------------------
        BINARY_LOGICAL_LOOP(HIEDNN_BINARY_LOGICAL_AND,
                            LogicalAnd)
        BINARY_LOGICAL_LOOP(HIEDNN_BINARY_LOGICAL_OR,
                            LogicalOr)
        BINARY_LOGICAL_LOOP(HIEDNN_BINARY_LOGICAL_XOR,
                            LogicalXor)

        // --------------------------------------
        // invalid op type
        // --------------------------------------
        default:
            ret = HIEDNN_STATUS_INVALID_OPTYPE;
            break;
    }

    return ret;
}

