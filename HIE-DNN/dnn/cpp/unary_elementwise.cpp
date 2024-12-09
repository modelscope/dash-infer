/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    unary_elementwise.cpp
 */

#include <hiednn.h>
#include <hiednn_cpp.h>

#include <cstddef>

#include <utils.hpp>
#include <tensor_desc.hpp>
#include <datatype_dispatch.hpp>
#include <scalar_functor.hpp>
#include <cpp/cpp_handle.hpp>

namespace hiednn {

namespace cpp {

namespace {

template <bool BETA, typename T, typename Op>
void UnaryElementwiseKernel(
        const T *x, T *y, size_t n, T alpha, T beta, Op op) {
    for (size_t i = 0; i < n; ++i) {
        T st_reg = alpha * op(x[i]);
        if (BETA) {
            st_reg += y[i] * beta;
        }
        y[i] = st_reg;
    }
}

template <typename T, typename Op>
void UnaryElementwiseLogicalKernel(const T *x, char *y, size_t n, Op op) {
    for (size_t i = 0; i < n; ++i) {
        y[i] = op(x[i]);
    }
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
                              GetOp getop,
                              Arg&&... args) {
        if (xDesc.dataType != yDesc.dataType) {
            return HIEDNN_STATUS_INVALID_DATATYPE;
        }

        const T *x_ptr = static_cast<const T *>(x);
        T *y_ptr = static_cast<T *>(y);
        T alpha_val = *reinterpret_cast<const T *>(alpha);
        T beta_val = *reinterpret_cast<const T *>(beta);

        auto scalar_op = getop.template get<T>(std::forward<Arg>(args)...);

        hiednnStatus_t ret = HIEDNN_STATUS_SUCCESS;
        if (xDesc == yDesc && xDesc.contiguous) {
            // x and y in same memory layout and both are contiguous
            size_t n = xDesc.size;
            if (beta_val == 0) {
                UnaryElementwiseKernel<false, T>(
                    x_ptr, y_ptr, n, alpha_val, beta_val, scalar_op);
            } else {
                UnaryElementwiseKernel<true, T>(
                    x_ptr, y_ptr, n, alpha_val, beta_val, scalar_op);
            }
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
            // x and y in same memory layout and both are contiguous
            size_t n = xDesc.size;
            UnaryElementwiseLogicalKernel<T>(x_ptr, y_ptr, n, scalar_op);
        } else {
            ret = HIEDNN_STATUS_INVALID_PARAMETER;
        }

        return ret;
    }
};

}  // anonymous namespace

}  // namespace cpp

}  // namespace hiednn

// switch-case loops for unary elementwise OP
#define UNARY_CASE_LOOP(TAG, DATATYPE, FUNC) \
    case TAG: \
        ret = hiednn::Dispatch##DATATYPE<hiednn::cpp::UnaryElementwiseImpl>( \
            xDesc->dataType, x, *xDesc, y, *yDesc, alpha, beta, \
            hiednn::GetScalarOp<hiednn::scalar_functor::FUNC>()); \
        break;

// switch-case loops for unary elementwise OP with extParam
#define UNARY_CASE_LOOP_PARAM(TAG, DATATYPE, FUNC, ...) \
    case TAG: \
        ret = hiednn::Dispatch##DATATYPE<hiednn::cpp::UnaryElementwiseImpl>( \
            xDesc->dataType, x, *xDesc, y, *yDesc, alpha, beta, \
            hiednn::GetScalarOp<hiednn::scalar_functor::FUNC>(), __VA_ARGS__); \
        break;

// switch-case loops for unary elementwise logical OP
#define UNARY_LOGICAL_CASE_LOOP(TAG, DATATYPE, FUNC) \
    case TAG: \
        ret = hiednn::Dispatch##DATATYPE< \
                hiednn::cpp::UnaryElementwiseLogicalImpl>( \
            xDesc->dataType, x, *xDesc, y, *yDesc, \
            hiednn::GetScalarOp<hiednn::scalar_functor::FUNC>()); \
        break;

hiednnStatus_t
hiednnCppUnaryElementwiseOp(HiednnCppHandle *handle,
                            hiednnUnaryEltwiseOp_t unaryEltwiseOp,
                            const void *alpha,
                            HiednnTensorDesc *xDesc,
                            const void *x,
                            const void *extParam1,
                            const void *extParam2,
                            const void *beta,
                            HiednnTensorDesc *yDesc,
                            void *y) {
    if (!hiednn::CheckNullptr(handle, xDesc, yDesc) ||
        !hiednn::CheckTensorPtr(*xDesc, x, *yDesc, y)) {
        return HIEDNN_STATUS_INVALID_PARAMETER;
    }

    if (!xDesc->SameDim(*yDesc)) {
        return HIEDNN_STATUS_INVALID_PARAMETER;
    }

    if (xDesc->size == 0) {
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
                ret = hiednn::cpp::UnaryElementwiseLogicalImpl<char>()(
                          x, *xDesc, y, *yDesc,
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


