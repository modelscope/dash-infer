/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    unary_elementwise_utest.cpp
 */
#include <hiednn.h>
#include <hiednn_cpp.h>

#include <gtest/gtest.h>
#include <cstdlib>
#include <cmath>

#include <utest_utils.hpp>

namespace {

const int VECTOR_LENGTH = 50;

template <typename T>
T Rand(unsigned int *seed, T bias) {
    T ret = bias + static_cast<T>(rand_r(seed));
    return ret;
}

template <>
float Rand<float>(unsigned int *seed, float bias) {
    return bias + static_cast<float>(rand_r(seed)) /
                  static_cast<float>(RAND_MAX);
}

}  // anonymous namespace

#define UTEST_UNARY_ELEMENTWISE(TEST_NAME, \
                                HIE_UNARY_OP, \
                                IN_TYPE, \
                                IN_HIEDNN_TYPE, \
                                OUT_TYPE, \
                                OUT_HIEDNN_TYPE, \
                                RAND_BIAS, \
                                EXTPARAM_TYPE, \
                                STD_EXPR) \
TEST(Unary_Elementwise_CPP, TEST_NAME) { \
    unsigned int seed = 0; \
    IN_TYPE x[VECTOR_LENGTH]; \
    for (int i = 0; i < VECTOR_LENGTH; ++i) { \
        x[i] = Rand<IN_TYPE>(&seed, static_cast<IN_TYPE>(RAND_BIAS)); \
    } \
    \
    OUT_TYPE y[VECTOR_LENGTH]; \
    for (int i = 0; i < VECTOR_LENGTH; ++i) { \
        y[i] = Rand<OUT_TYPE>(&seed, 0); \
    } \
    \
    OUT_TYPE y_out[VECTOR_LENGTH]; \
    memcpy(y_out, y, VECTOR_LENGTH * sizeof(OUT_TYPE)); \
    \
    hiednnCppHandle_t handle; \
    CHECK_HIEDNN(hiednnCreateCppHandle(&handle, HIEDNN_ASM_OPTIMIZE_NONE)); \
    \
    IN_TYPE alpha = 2.f; \
    OUT_TYPE beta = 1.5f; \
    EXTPARAM_TYPE extParam1 = 0.4f; \
    EXTPARAM_TYPE extParam2 = 0.7f; \
    \
    hiednnTensorDesc_t x_desc, y_desc; \
    CHECK_HIEDNN(hiednnCreateTensorDesc(&x_desc)); \
    CHECK_HIEDNN(hiednnCreateTensorDesc(&y_desc)); \
    CHECK_HIEDNN(hiednnSet4dTensorDesc( \
        x_desc, HIEDNN_DATATYPE_##IN_HIEDNN_TYPE, \
        HIEDNN_TENSORFORMAT_NORMAL, 1, 1, 1, VECTOR_LENGTH)); \
    CHECK_HIEDNN(hiednnSet4dTensorDesc( \
        y_desc, HIEDNN_DATATYPE_##OUT_HIEDNN_TYPE, \
        HIEDNN_TENSORFORMAT_NORMAL, 1, 1, 1, VECTOR_LENGTH)); \
    \
    CHECK_HIEDNN(hiednnCppUnaryElementwiseOp( \
        handle, HIE_UNARY_OP, &alpha, x_desc, x, \
        &extParam1, &extParam2, &beta, y_desc, y_out)); \
    \
    struct Ref { \
        float operator()(const IN_TYPE &x, \
                         const OUT_TYPE &y, \
                         const IN_TYPE &alpha, \
                         const OUT_TYPE &beta, \
                         const EXTPARAM_TYPE &extParam1, \
                         const EXTPARAM_TYPE &extParam2) { \
            return STD_EXPR; \
        } \
    }; \
    \
    for (int i = 0; i < VECTOR_LENGTH; ++i) { \
        OUT_TYPE std_val = Ref()(x[i], y[i], alpha, beta, \
                                 extParam1, extParam2); \
        OUT_TYPE diff = std_val - y_out[i]; \
        ASSERT_LE(std::abs(static_cast<double>(diff)), \
                  std::abs(static_cast<double>(std_val) * 1e-5)); \
    } \
    \
    CHECK_HIEDNN(hiednnDestroyCppHandle(handle)); \
    CHECK_HIEDNN(hiednnDestroyTensorDesc(x_desc)); \
    CHECK_HIEDNN(hiednnDestroyTensorDesc(y_desc)); \
}

#define UTEST_UNARY_ELEMENTWISE_F32(TEST_NAME, \
                                    HIE_UNARY_OP, \
                                    RAND_BIAS, \
                                    STD_EXPR) \
    UTEST_UNARY_ELEMENTWISE(TEST_NAME, HIE_UNARY_OP, \
                            float, FP32, float, FP32, \
                            RAND_BIAS, float, alpha * (STD_EXPR) + beta * y)

UTEST_UNARY_ELEMENTWISE_F32(Add_F32,
                            HIEDNN_UNARY_MATH_ADD,
                            0,
                            x + extParam1);
UTEST_UNARY_ELEMENTWISE_F32(Mul_F32,
                            HIEDNN_UNARY_MATH_MUL,
                            0,
                            x * extParam1);
UTEST_UNARY_ELEMENTWISE_F32(Div_F32,
                            HIEDNN_UNARY_MATH_DIV,
                            0,
                            x / extParam1);
UTEST_UNARY_ELEMENTWISE_F32(Pow_F32,
                            HIEDNN_UNARY_MATH_POW,
                            0,
                            std::pow(x, extParam1));
UTEST_UNARY_ELEMENTWISE_F32(Sqrt_F32,
                            HIEDNN_UNARY_MATH_SQRT,
                            0,
                            std::sqrt(x));
UTEST_UNARY_ELEMENTWISE_F32(Cbrt_F32,
                            HIEDNN_UNARY_MATH_CBRT,
                            0,
                            std::cbrt(x));
UTEST_UNARY_ELEMENTWISE_F32(Exp_F32,
                            HIEDNN_UNARY_MATH_EXP,
                            0,
                            std::exp(x));
UTEST_UNARY_ELEMENTWISE_F32(Erf_F32,
                            HIEDNN_UNARY_MATH_ERF,
                            0,
                            std::erf(x));
UTEST_UNARY_ELEMENTWISE_F32(Log_F32,
                            HIEDNN_UNARY_MATH_LOG,
                            0,
                            std::log(x));
UTEST_UNARY_ELEMENTWISE_F32(Sin_F32,
                            HIEDNN_UNARY_MATH_SIN,
                            0,
                            std::sin(x));
UTEST_UNARY_ELEMENTWISE_F32(Cos_F32,
                            HIEDNN_UNARY_MATH_COS,
                            0,
                            std::cos(x));
UTEST_UNARY_ELEMENTWISE_F32(Tan_F32,
                            HIEDNN_UNARY_MATH_TAN,
                            0,
                            std::tan(x));
UTEST_UNARY_ELEMENTWISE_F32(Asin_F32,
                            HIEDNN_UNARY_MATH_ASIN,
                            0,
                            std::asin(x));
UTEST_UNARY_ELEMENTWISE_F32(Acos_F32,
                            HIEDNN_UNARY_MATH_ACOS,
                            0,
                            std::acos(x));
UTEST_UNARY_ELEMENTWISE_F32(Atan_F32,
                            HIEDNN_UNARY_MATH_ATAN,
                            0,
                            std::atan(x));
UTEST_UNARY_ELEMENTWISE_F32(Sinh_F32,
                            HIEDNN_UNARY_MATH_SINH,
                            0,
                            std::sinh(x));
UTEST_UNARY_ELEMENTWISE_F32(Cosh_F32,
                            HIEDNN_UNARY_MATH_COSH,
                            0,
                            std::cosh(x));
UTEST_UNARY_ELEMENTWISE_F32(Tanh_F32,
                            HIEDNN_UNARY_MATH_TANH,
                            0,
                            std::tanh(x));
UTEST_UNARY_ELEMENTWISE_F32(Asinh_F32,
                            HIEDNN_UNARY_MATH_ASINH,
                            0,
                            std::asinh(x));
UTEST_UNARY_ELEMENTWISE_F32(Acosh_F32,
                            HIEDNN_UNARY_MATH_ACOSH,
                            1.5f,
                            std::acosh(x));
UTEST_UNARY_ELEMENTWISE_F32(Atanh_F32,
                            HIEDNN_UNARY_MATH_ATANH,
                            0,
                            std::atanh(x));
UTEST_UNARY_ELEMENTWISE_F32(Reciprocal_F32,
                            HIEDNN_UNARY_MATH_RECIPROCAL,
                            0,
                            1 / x);
UTEST_UNARY_ELEMENTWISE_F32(Ceil_F32,
                            HIEDNN_UNARY_MATH_CEIL,
                            0,
                            std::ceil(x));
UTEST_UNARY_ELEMENTWISE_F32(Floor_F32,
                            HIEDNN_UNARY_MATH_FLOOR,
                            0,
                            std::floor(x));
UTEST_UNARY_ELEMENTWISE_F32(Sigmoid_F32,
                            HIEDNN_UNARY_MATH_SIGMOID,
                            0,
                            1 / (1 + std::exp(-x)));
UTEST_UNARY_ELEMENTWISE_F32(LeakyRelu_F32,
                            HIEDNN_UNARY_MATH_LEAKYRELU,
                            0,
                            x < 0 ? extParam1 * x : x);


UTEST_UNARY_ELEMENTWISE_F32(Abs,
                            HIEDNN_UNARY_MATH_ABS,
                            0,
                            x < 0 ? -x : x);
UTEST_UNARY_ELEMENTWISE_F32(Sign,
                            HIEDNN_UNARY_MATH_SIGN,
                            0,
                            x > 0 ? 1 : x == 0 ? 0 : -1);
UTEST_UNARY_ELEMENTWISE_F32(Round,
                            HIEDNN_UNARY_MATH_ROUND,
                            0,
                            std::round(x));
UTEST_UNARY_ELEMENTWISE_F32(Shrink,
                            HIEDNN_UNARY_MATH_SHRINK,
                            0,
                            x < -extParam1 ? x + extParam2 :
                                             x > extParam1 ? x - extParam2 : 0);
UTEST_UNARY_ELEMENTWISE_F32(Clip,
                            HIEDNN_UNARY_MATH_CLIP,
                            0,
                            x < extParam1 ? extParam1 :
                                            x > extParam2 ? extParam2 : x);
UTEST_UNARY_ELEMENTWISE_F32(Elu,
                            HIEDNN_UNARY_MATH_ELU,
                            0,
                            x < 0 ? extParam1 * (std::exp(x) - 1) : x);
UTEST_UNARY_ELEMENTWISE_F32(Selu,
                            HIEDNN_UNARY_MATH_SELU,
                            0,
                            x > 0 ? extParam2 * x :
                                    extParam2 *
                                    (extParam1 * std::exp(x) - extParam1));
UTEST_UNARY_ELEMENTWISE_F32(Celu,
                            HIEDNN_UNARY_MATH_CELU,
                            0,
                            std::fmax(0, x) +
                            std::fmin(0, extParam1 *
                                         (std::exp(x / extParam1) - 1)));
UTEST_UNARY_ELEMENTWISE_F32(Softplus,
                            HIEDNN_UNARY_MATH_SOFTPLUS,
                            0,
                            std::log(std::exp(x) + 1));
UTEST_UNARY_ELEMENTWISE_F32(Softsign,
                            HIEDNN_UNARY_MATH_SOFTSIGN,
                            0,
                            x / (1 + std::fabs(x)));

UTEST_UNARY_ELEMENTWISE_F32(HardSigmoid,
                            HIEDNN_UNARY_MATH_HARDSIGMOID,
                            0,
                            std::fmax(0, std::fmin(1, extParam1 * x +
                                                      extParam2)));
UTEST_UNARY_ELEMENTWISE_F32(HardSwish,
                            HIEDNN_UNARY_MATH_HARDSWISH,
                            0,
                            x * std::fmax(0, std::fmin(1, extParam1 * x +
                                                          extParam2)));
UTEST_UNARY_ELEMENTWISE_F32(Neg,
                            HIEDNN_UNARY_MATH_NEG,
                            0,
                            -x);
UTEST_UNARY_ELEMENTWISE_F32(ThresholdRelu,
                            HIEDNN_UNARY_MATH_THRESHOLDRELU,
                            0,
                            x > extParam1 ? x : 0);
UTEST_UNARY_ELEMENTWISE(IsInf,
                        HIEDNN_UNARY_MATH_ISINF,
                        float, FP32, char, BOOL, 0, float,
                        std::isinf(x));
UTEST_UNARY_ELEMENTWISE(IsNan,
                        HIEDNN_UNARY_MATH_ISNAN,
                        float, FP32, char, BOOL, 0, float,
                        std::isnan(x));
UTEST_UNARY_ELEMENTWISE(Not,
                        HIEDNN_UNARY_LOGICAL_NOT,
                        char, BOOL, char, BOOL, 0, float,
                        !x);


