/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    unary_elementwise.cu
 */

#include <hiednn.h>
#include <hiednn_cuda.h>

#include <cstddef>
#include <cstdint>
#include <algorithm>

#include <utils.hpp>
#include <tensor_desc.hpp>
#include <datatype_dispatch.hpp>
#include <scalar_functor.hpp>
#include <packed_memory_access.hpp>

#include <cuda/cuda_handle.hpp>
#include <cuda/cuda_utils.hpp>
#include <cuda/intrinsic/global_memory.hpp>

namespace hiednn {

namespace cuda {

namespace {

template <int BLOCK, int UNROLL, bool BETA, typename T, typename Op>
__global__ void UnaryElementwiseKernel(
        const T *x, T *y, T alpha, T beta, Op op,
        PackedEltwiseConfig packConfig) {
    int64_t tid = static_cast<int64_t>(blockIdx.x) * BLOCK + threadIdx.x;

    if (tid < packConfig.nPack) {
        using V_T = VT<T, UNROLL>;
        V_T xReg, yReg;

        Ldg(&xReg, reinterpret_cast<const V_T *>(x) + tid);

        if (BETA) {
            Ldg(&yReg, reinterpret_cast<V_T *>(y) + tid);
            #pragma unroll
            for (int i = 0; i < UNROLL; ++i) {
                yReg.data[i] = alpha * op(xReg.data[i]) + beta * yReg.data[i];
            }
        } else {
            #pragma unroll
            for (int i = 0; i < UNROLL; ++i) {
                yReg.data[i] = alpha * op(xReg.data[i]);
            }
        }

        Stg(yReg, reinterpret_cast<V_T *>(y) + tid);
    } else if (UNROLL > 1 && tid < packConfig.nThread) {
        int64_t idx = tid + packConfig.unpackedOffset;
        y[idx] = BETA ?
                 alpha * op(x[idx]) + beta * y[idx] :
                 alpha * op(x[idx]);
    }
}

template <int BLOCK, int UNROLL, typename T, typename Op>
__global__ void UnaryElementwiseLogicalKernel(
        const T *x, char *y, Op op, PackedEltwiseConfig packConfig) {
    int64_t tid = static_cast<int64_t>(blockIdx.x) * BLOCK + threadIdx.x;

    if (tid < packConfig.nPack) {
        using VT_IN = VT<T, UNROLL>;
        using VT_OUT = VT<char, UNROLL>;
        VT_IN xReg;
        VT_OUT yReg;

        Ldg(&xReg, reinterpret_cast<const VT_IN *>(x) + tid);

        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            yReg.data[i] = op(xReg.data[i]);
        }

        Stg(yReg, reinterpret_cast<VT_OUT *>(y) + tid);
    } else if (UNROLL > 1 && tid < packConfig.nThread) {
        int64_t idx = tid + packConfig.unpackedOffset;
        y[idx] = op(x[idx]);
    }
}

template <typename T, typename ScalarOp>
hiednnStatus_t
LaunchUnaryElementwiseKernel(
        const T *x, T *y, size_t n, T alpha, T beta,
        ScalarOp scalarOp, cudaStream_t stream) {
    const int64_t BLOCK = 128;
    int packSize = std::min(GetPackSize(x),
                            GetPackSize(y));

    hiednnStatus_t ret = HIEDNN_STATUS_SUCCESS;
    if (beta == 0) {
        switch (packSize) {
            case 8: {
                const int UNROLL = ValidPack<T, 8>();
                PackedEltwiseConfig packConfig(n, UNROLL, BLOCK);
                UnaryElementwiseKernel<BLOCK, UNROLL, false>
                    <<<packConfig.nBlock, BLOCK, 0, stream>>>(
                    x, y, alpha, beta, scalarOp, packConfig);
                break;
            }
            case 4: {
                const int UNROLL = ValidPack<T, 4>();
                PackedEltwiseConfig packConfig(n, UNROLL, BLOCK);
                UnaryElementwiseKernel<BLOCK, UNROLL, false>
                    <<<packConfig.nBlock, BLOCK, 0, stream>>>(
                    x, y, alpha, beta, scalarOp, packConfig);
                break;
            }
            case 2: {
                const int UNROLL = ValidPack<T, 2>();
                PackedEltwiseConfig packConfig(n, UNROLL, BLOCK);
                UnaryElementwiseKernel<BLOCK, UNROLL, false>
                    <<<packConfig.nBlock, BLOCK, 0, stream>>>(
                    x, y, alpha, beta, scalarOp, packConfig);
                break;
            }
            case 1: {
                PackedEltwiseConfig packConfig(n, 1, BLOCK);
                UnaryElementwiseKernel<BLOCK, 1, false>
                    <<<packConfig.nBlock, BLOCK, 0, stream>>>(
                    x, y, alpha, beta, scalarOp, packConfig);
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
                UnaryElementwiseKernel<BLOCK, UNROLL, true>
                    <<<packConfig.nBlock, BLOCK, 0, stream>>>(
                    x, y, alpha, beta, scalarOp, packConfig);
                break;
            }
            case 4: {
                const int UNROLL = ValidPack<T, 4>();
                PackedEltwiseConfig packConfig(n, UNROLL, BLOCK);
                UnaryElementwiseKernel<BLOCK, UNROLL, true>
                    <<<packConfig.nBlock, BLOCK, 0, stream>>>(
                    x, y, alpha, beta, scalarOp, packConfig);
                break;
            }
            case 2: {
                const int UNROLL = ValidPack<T, 2>();
                PackedEltwiseConfig packConfig(n, UNROLL, BLOCK);
                UnaryElementwiseKernel<BLOCK, UNROLL, true>
                    <<<packConfig.nBlock, BLOCK, 0, stream>>>(
                    x, y, alpha, beta, scalarOp, packConfig);
                break;
            }
            case 1: {
                PackedEltwiseConfig packConfig(n, 1, BLOCK);
                UnaryElementwiseKernel<BLOCK, 1, true>
                    <<<packConfig.nBlock, BLOCK, 0, stream>>>(
                    x, y, alpha, beta, scalarOp, packConfig);
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
LaunchUnaryElementwiseLogicalKernel(
        const T *x, char *y, size_t n,
        ScalarOp scalarOp, cudaStream_t stream) {
    const int64_t BLOCK = 128;
    int packSize = std::min(GetPackSize(x),
                            GetPackSize(y));

    hiednnStatus_t ret = HIEDNN_STATUS_SUCCESS;
    switch (packSize) {
        case 8: {
            const int UNROLL = ValidPack<T, 8>();
            PackedEltwiseConfig packConfig(n, UNROLL, BLOCK);
            UnaryElementwiseLogicalKernel<BLOCK, UNROLL>
                <<<packConfig.nBlock, BLOCK, 0, stream>>>(
                x, y, scalarOp, packConfig);
            break;
        }
        case 4: {
            const int UNROLL = ValidPack<T, 4>();
            PackedEltwiseConfig packConfig(n, UNROLL, BLOCK);
            UnaryElementwiseLogicalKernel<BLOCK, UNROLL>
                <<<packConfig.nBlock, BLOCK, 0, stream>>>(
                x, y, scalarOp, packConfig);
            break;
        }
        case 2: {
            const int UNROLL = ValidPack<T, 2>();
            PackedEltwiseConfig packConfig(n, UNROLL, BLOCK);
            UnaryElementwiseLogicalKernel<BLOCK, UNROLL>
                <<<packConfig.nBlock, BLOCK, 0, stream>>>(
                x, y, scalarOp, packConfig);
            break;
        }
        case 1: {
            PackedEltwiseConfig packConfig(n, 1, BLOCK);
            UnaryElementwiseLogicalKernel<BLOCK, 1>
                <<<packConfig.nBlock, BLOCK, 0, stream>>>(
                x, y, scalarOp, packConfig);
            break;
        }
        default:
            ret = HIEDNN_STATUS_INTERNAL_ERROR;
            break;
    }

    return ret;
}

// impl functor for unary-map, such as y = sqrt(x)
template <typename T>
struct UnaryElementwiseImpl {
    template <typename GetOp, typename ...Arg>
    hiednnStatus_t operator()(const void *x,
                              const HiednnTensorDesc &xDesc,
                              void *y,
                              const HiednnTensorDesc &yDesc,
                              const void *alpha,
                              const void *beta,
                              cudaStream_t stream,
                              GetOp getop,
                              Arg&&... args) {
        if (xDesc.dataType != yDesc.dataType) {
            return HIEDNN_STATUS_INVALID_PARAMETER;
        }

        const T *x_ptr = static_cast<const T *>(x);
        T *y_ptr = static_cast<T *>(y);
        T alpha_val = *static_cast<const T *>(alpha);
        T beta_val = *static_cast<const T *>(beta);

        auto scalar_op = getop.template get<T>(std::forward<Arg>(args)...);

        hiednnStatus_t ret = HIEDNN_STATUS_SUCCESS;
        if (xDesc == yDesc && xDesc.contiguous) {
            // x and y are in same memory layout and both are contiguous
            size_t n = xDesc.size;
            ret = LaunchUnaryElementwiseKernel<T>(
                x_ptr, y_ptr, n, alpha_val, beta_val, scalar_op, stream);
        } else {
            ret = HIEDNN_STATUS_INVALID_PARAMETER;
        }

        return ret;
    }
};

template <typename T>
struct UnaryElementwiseLogicalImpl {
    template <typename GetOp, typename ...Arg>
    hiednnStatus_t operator()(const void *x,
                              const HiednnTensorDesc &xDesc,
                              void *y,
                              const HiednnTensorDesc &yDesc,
                              cudaStream_t stream,
                              GetOp getop,
                              Arg&&... args) {
        if (yDesc.dataType != HIEDNN_DATATYPE_BOOL) {
            return HIEDNN_STATUS_INVALID_DATATYPE;
        }

        const T *x_ptr = static_cast<const T *>(x);
        char *y_ptr = static_cast<char *>(y);

        auto scalar_op = getop.template get<T>(std::forward<Arg>(args)...);

        hiednnStatus_t ret = HIEDNN_STATUS_SUCCESS;
        if (xDesc.SameDimStride(yDesc) && xDesc.contiguous) {
            // x and y are in same memory layout and both are contiguous
            size_t n = xDesc.size;
            ret = LaunchUnaryElementwiseLogicalKernel<T>(
                x_ptr, y_ptr, n, scalar_op, stream);
        } else {
            ret = HIEDNN_STATUS_INVALID_PARAMETER;
        }

        return ret;
    }
};

}  // anonymous namespace

}  // namespace cuda

}  // namespace hiednn

// switch-case loops for unary elementwise OP
#define UNARY_CASE_LOOP(TAG, DATATYPE, FUNC) \
    case TAG: \
        ret = hiednn::Dispatch##DATATYPE<hiednn::cuda::UnaryElementwiseImpl>( \
            xDesc->dataType, x, *xDesc, y, *yDesc, alpha, beta, \
            cudaHandle->stream, \
            hiednn::GetScalarOp<hiednn::scalar_functor::FUNC>()); \
        break;

// switch-case loops for unary elementwise OP with extParam
#define UNARY_CASE_LOOP_PARAM(TAG, DATATYPE, FUNC, ...) \
    case TAG: \
        ret = hiednn::Dispatch##DATATYPE<hiednn::cuda::UnaryElementwiseImpl>( \
            xDesc->dataType, x, *xDesc, y, *yDesc, alpha, beta, \
            cudaHandle->stream, \
            hiednn::GetScalarOp<hiednn::scalar_functor::FUNC>(), __VA_ARGS__); \
        break;

// switch-case loops for unary elementwise logical OP
#define UNARY_LOGICAL_CASE_LOOP(TAG, DATATYPE, FUNC) \
    case TAG: \
        ret = hiednn::Dispatch##DATATYPE< \
                hiednn::cuda::UnaryElementwiseLogicalImpl>( \
            xDesc->dataType, x, *xDesc, y, *yDesc, cudaHandle->stream, \
            hiednn::GetScalarOp<hiednn::scalar_functor::FUNC>()); \
        break;

hiednnStatus_t
hiednnCudaUnaryElementwiseOp(HiednnCudaHandle *cudaHandle,
                             hiednnUnaryEltwiseOp_t unaryEltwiseOp,
                             const void *alpha,
                             HiednnTensorDesc *xDesc,
                             const void *x,
                             const void *extParam1,
                             const void *extParam2,
                             const void *beta,
                             HiednnTensorDesc *yDesc,
                             void *y) {
    if (!hiednn::CheckNullptr(cudaHandle, xDesc, yDesc) ||
        !hiednn::CheckTensorPtr(*xDesc, x, *yDesc, y)) {
        return HIEDNN_STATUS_INVALID_PARAMETER;
    }

    if (!xDesc->SameDim(*yDesc)) {
        return HIEDNN_STATUS_INVALID_PARAMETER;
    }

    if (yDesc->size == 0) {
        return HIEDNN_STATUS_SUCCESS;
    }

    hiednnStatus_t ret = HIEDNN_STATUS_SUCCESS;
    switch (unaryEltwiseOp) {
        // --------------------------------------
        // only support bool, return bool
        // --------------------------------------
        case HIEDNN_UNARY_LOGICAL_NOT:
            if (xDesc->dataType != HIEDNN_DATATYPE_BOOL) {
                ret = HIEDNN_STATUS_INVALID_DATATYPE;
            } else {
                ret = hiednn::cuda::UnaryElementwiseLogicalImpl<char>()(
                          x, *xDesc, y, *yDesc, cudaHandle->stream,
                          hiednn::GetScalarOp<hiednn::scalar_functor::Not>());
            }
            break;

        // --------------------------------------
        // support all datatype
        // --------------------------------------
        UNARY_CASE_LOOP_PARAM(HIEDNN_UNARY_MATH_ADD,
                              All,
                              AddX,
                              extParam1);
        UNARY_CASE_LOOP_PARAM(HIEDNN_UNARY_MATH_MUL,
                              All,
                              MulX,
                              extParam1);
        UNARY_CASE_LOOP_PARAM(HIEDNN_UNARY_MATH_DIV,
                              All,
                              DivX,
                              extParam1);
        UNARY_CASE_LOOP_PARAM(HIEDNN_UNARY_MATH_SHRINK,
                              All,
                              Shrink,
                              extParam1, extParam2);
        UNARY_CASE_LOOP_PARAM(HIEDNN_UNARY_MATH_CLIP,
                              All,
                              Clip,
                              extParam1, extParam2);
        UNARY_CASE_LOOP(HIEDNN_UNARY_MATH_ABS,
                        Signed,
                        Abs);
        UNARY_CASE_LOOP(HIEDNN_UNARY_MATH_SIGN,
                        All,
                        Sign);

        // --------------------------------------
        // only support floating point, return bool
        // --------------------------------------
        UNARY_LOGICAL_CASE_LOOP(HIEDNN_UNARY_MATH_ISINF,
                                FP,
                                IsInf);
        UNARY_LOGICAL_CASE_LOOP(HIEDNN_UNARY_MATH_ISNAN,
                                FP,
                                IsNan);

        // --------------------------------------
        // only support floating point
        // --------------------------------------
        UNARY_CASE_LOOP(HIEDNN_UNARY_MATH_SQRT,
                        FP,
                        Sqrt);
        UNARY_CASE_LOOP(HIEDNN_UNARY_MATH_CBRT,
                        FP,
                        Cbrt);
        UNARY_CASE_LOOP(HIEDNN_UNARY_MATH_EXP,
                        FP,
                        Exp);
        UNARY_CASE_LOOP(HIEDNN_UNARY_MATH_ERF,
                        FP,
                        Erf);
        UNARY_CASE_LOOP(HIEDNN_UNARY_MATH_LOG,
                        FP,
                        Log);
        UNARY_CASE_LOOP(HIEDNN_UNARY_MATH_SIN,
                        FP,
                        Sin);
        UNARY_CASE_LOOP(HIEDNN_UNARY_MATH_COS,
                        FP,
                        Cos);
        UNARY_CASE_LOOP(HIEDNN_UNARY_MATH_TAN,
                        FP,
                        Tan);
        UNARY_CASE_LOOP(HIEDNN_UNARY_MATH_ASIN,
                        FP,
                        Asin);
        UNARY_CASE_LOOP(HIEDNN_UNARY_MATH_ACOS,
                        FP,
                        Acos);
        UNARY_CASE_LOOP(HIEDNN_UNARY_MATH_ATAN,
                        FP,
                        Atan);
        UNARY_CASE_LOOP(HIEDNN_UNARY_MATH_SINH,
                        FP,
                        Sinh);
        UNARY_CASE_LOOP(HIEDNN_UNARY_MATH_COSH,
                        FP,
                        Cosh);
        UNARY_CASE_LOOP(HIEDNN_UNARY_MATH_TANH,
                        FP,
                        Tanh);
        UNARY_CASE_LOOP(HIEDNN_UNARY_MATH_ASINH,
                        FP,
                        Asinh);
        UNARY_CASE_LOOP(HIEDNN_UNARY_MATH_ACOSH,
                        FP,
                        Acosh);
        UNARY_CASE_LOOP(HIEDNN_UNARY_MATH_ATANH,
                        FP,
                        Atanh);
        UNARY_CASE_LOOP(HIEDNN_UNARY_MATH_RECIPROCAL,
                        FP,
                        Reciprocal);
        UNARY_CASE_LOOP(HIEDNN_UNARY_MATH_CEIL,
                        FP,
                        Ceil);
        UNARY_CASE_LOOP(HIEDNN_UNARY_MATH_FLOOR,
                        FP,
                        Floor);
        UNARY_CASE_LOOP(HIEDNN_UNARY_MATH_ROUND,
                        FP,
                        Round);
        UNARY_CASE_LOOP(HIEDNN_UNARY_MATH_SIGMOID,
                        FP,
                        Sigmoid);
        UNARY_CASE_LOOP_PARAM(HIEDNN_UNARY_MATH_POW,
                              FP,
                              PowX,
                              extParam1);
        UNARY_CASE_LOOP_PARAM(HIEDNN_UNARY_MATH_LEAKYRELU,
                              FP,
                              LeakyRelu,
                              extParam1);
        UNARY_CASE_LOOP_PARAM(HIEDNN_UNARY_MATH_ELU,
                              FP,
                              Elu,
                              extParam1);
        UNARY_CASE_LOOP_PARAM(HIEDNN_UNARY_MATH_SELU,
                              FP,
                              Selu,
                              extParam1, extParam2);
        UNARY_CASE_LOOP_PARAM(HIEDNN_UNARY_MATH_CELU,
                              FP,
                              Celu,
                              extParam1);
        UNARY_CASE_LOOP_PARAM(HIEDNN_UNARY_MATH_HARDSIGMOID,
                              FP,
                              HardSigmoid,
                              extParam1, extParam2);
        UNARY_CASE_LOOP_PARAM(HIEDNN_UNARY_MATH_HARDSWISH,
                              FP,
                              HardSwish,
                              extParam1, extParam2);
        UNARY_CASE_LOOP(HIEDNN_UNARY_MATH_SOFTPLUS,
                        FP,
                        Softplus);
        UNARY_CASE_LOOP(HIEDNN_UNARY_MATH_SOFTSIGN,
                        FP,
                        Softsign);

        // --------------------------------------
        // only support signed number
        // --------------------------------------
        UNARY_CASE_LOOP(HIEDNN_UNARY_MATH_NEG,
                        Signed,
                        Neg);
        UNARY_CASE_LOOP_PARAM(HIEDNN_UNARY_MATH_THRESHOLDRELU,
                              Signed,
                              ThresholdRelu,
                              extParam1);

        // --------------------------------------
        // invalid op type
        // --------------------------------------
        default:
            ret = HIEDNN_STATUS_INVALID_OPTYPE;
            break;
    }

    return ret;
}


