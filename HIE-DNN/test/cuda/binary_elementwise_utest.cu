/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    binary_elementwise_utest.cu
 */
#include <hiednn.h>
#include <hiednn_cuda.h>

#include <gtest/gtest.h>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <algorithm>

#include <utest_utils.hpp>

namespace {

const int VECTOR_LENGTH = 3003;
const float EXT_PARAM_F = 3.f;
const int EXT_MOD = 0;
const int EXT_FMOD = 1;
const int EXT_LEFT = 0;
const int EXT_RIGHT = 1;

struct broadcast_shape_t {
    int X_shape[4];
    int Y_shape[4];
    int Z_shape[4];
    int X_len;
    int Y_len;
    int Z_len;
};

const std::vector<broadcast_shape_t> BROADCAST_CASES = {
    {{2, 8, 6, 12}, {1, 8, 1, 12}, {2, 8, 6, 12}, 1152, 96, 1152},
    {{1, 8, 12, 1}, {1, 8, 1, 12}, {1, 8, 12, 12}, 96, 96, 1152},
    {{1, 5, 1, 1}, {3, 5, 1, 12}, {3, 5, 1, 12}, 5, 180, 180},
    {{4, 4, 1, 3}, {1, 4, 8, 3}, {4, 4, 8, 3}, 48, 96, 384},
};

template <typename T>
struct Mod {
    inline T operator() (const T &x, const T &y) {
        if (y == 0) return 0;
        T mod = x % y;
        return mod == 0 ? 0 : mod * y > 0 ? mod : mod + y;
    }
};

}  // anonymous namespace

#define UTEST_BINARY_ELEMENTWISE( \
        TEST_NAME, HIE_BINARY_OP, BIAS, \
        EXT_TYPE, EXT_VAL, ST, DT, INTYPE, OUTTYPE, STD_EXPR) \
TEST(Binary_Elementwise_CUDA, TEST_NAME) { \
    unsigned int seed = 0; \
    ST *x, *y; \
    DT *z;\
    x = static_cast<ST *>(malloc(sizeof(ST) * VECTOR_LENGTH)); \
    for (int i = 0; i < VECTOR_LENGTH; ++i) { \
        x[i] = BIAS + \
               static_cast<ST>(rand_r(&seed)) / \
               static_cast<ST>(RAND_MAX); \
    } \
    \
    y = static_cast<ST *>(malloc(sizeof(ST) * VECTOR_LENGTH)); \
    for (int i = 0; i < VECTOR_LENGTH; ++i) { \
        y[i] = static_cast<ST>(rand_r(&seed)) / \
               static_cast<ST>(RAND_MAX); \
    } \
    \
    z = static_cast<DT *>(malloc(sizeof(DT) * VECTOR_LENGTH)); \
    for (int i = 0; i < VECTOR_LENGTH; ++i) { \
        z[i] = static_cast<DT>(rand_r(&seed)) / \
               static_cast<DT>(RAND_MAX); \
    }\
    \
    ST *x_cu, *y_cu; \
    DT *z_cu; \
    cudaMalloc(&x_cu, VECTOR_LENGTH * sizeof(ST)); \
    cudaMalloc(&y_cu, VECTOR_LENGTH * sizeof(ST)); \
    cudaMalloc(&z_cu, VECTOR_LENGTH * sizeof(DT)); \
    cudaMemcpy(x_cu, x, VECTOR_LENGTH * sizeof(ST), cudaMemcpyHostToDevice); \
    cudaMemcpy(y_cu, y, VECTOR_LENGTH * sizeof(ST), cudaMemcpyHostToDevice); \
    cudaMemcpy(z_cu, z, VECTOR_LENGTH * sizeof(DT), cudaMemcpyHostToDevice); \
    \
    ST alpha = 2; \
    ST beta = 2; \
    EXT_TYPE extParam = EXT_VAL; \
    \
    hiednnCudaHandle_t handle; \
    CHECK_HIEDNN(hiednnCreateCudaHandle(&handle)); \
    \
    hiednnTensorDesc_t xDesc; \
    CHECK_HIEDNN(hiednnCreateTensorDesc(&xDesc)); \
    CHECK_HIEDNN(hiednnSet4dTensorDesc( \
        xDesc, INTYPE, HIEDNN_TENSORFORMAT_NORMAL, 1, 1, 1, VECTOR_LENGTH)); \
    \
    hiednnTensorDesc_t yDesc; \
    CHECK_HIEDNN(hiednnCreateTensorDesc(&yDesc)); \
    CHECK_HIEDNN(hiednnSet4dTensorDesc( \
        yDesc, INTYPE, HIEDNN_TENSORFORMAT_NORMAL, 1, 1, 1, VECTOR_LENGTH)); \
    \
    hiednnTensorDesc_t zDesc; \
    CHECK_HIEDNN(hiednnCreateTensorDesc(&zDesc)); \
    CHECK_HIEDNN(hiednnSet4dTensorDesc( \
        zDesc, OUTTYPE, HIEDNN_TENSORFORMAT_NORMAL, 1, 1, 1, VECTOR_LENGTH)); \
    \
    CHECK_HIEDNN(hiednnCudaBinaryElementwiseOp( \
        handle, HIE_BINARY_OP, static_cast<const void *>(&alpha), \
        xDesc, static_cast<const void *>(x_cu), \
        yDesc, static_cast<const void *>(y_cu), \
        static_cast<const void *>(extParam), \
        static_cast<const void *>(&beta), \
        zDesc, static_cast<void *>(z_cu))); \
    \
    DT *z_out; \
    z_out = static_cast<DT *>(malloc(sizeof(DT) * VECTOR_LENGTH));\
    cudaMemcpy(z_out, z_cu, VECTOR_LENGTH * sizeof(DT), \
               cudaMemcpyDeviceToHost); \
    \
    struct Ref { \
        float operator()(const ST &x, \
                         const ST &y, \
                         const DT &z, \
                         const ST &alpha, \
                         const ST &beta) { \
            return STD_EXPR; \
        } \
    }; \
    \
    for (int i = 0; i < VECTOR_LENGTH; ++i) { \
        DT std_val = Ref()(x[i], y[i], z[i], alpha, beta); \
        ASSERT_NEAR(z_out[i], std_val, std::fabs(std_val) * 1e-5); \
    } \
    \
    cudaFree(x_cu); \
    cudaFree(y_cu); \
    cudaFree(z_cu); \
    free(x); \
    free(y); \
    free(z); \
    free(z_out); \
    CHECK_HIEDNN(hiednnDestroyCudaHandle(handle)); \
    CHECK_HIEDNN(hiednnDestroyTensorDesc(xDesc)); \
    CHECK_HIEDNN(hiednnDestroyTensorDesc(yDesc)); \
    CHECK_HIEDNN(hiednnDestroyTensorDesc(zDesc)); \
}

UTEST_BINARY_ELEMENTWISE(Add_F32,
                         HIEDNN_BINARY_MATH_ADD,
                         0,
                         const float *,
                         &EXT_PARAM_F,
                         float,
                         float,
                         HIEDNN_DATATYPE_FP32,
                         HIEDNN_DATATYPE_FP32,
                         alpha * (x + y) + beta * z);
UTEST_BINARY_ELEMENTWISE(Sub_F32,
                         HIEDNN_BINARY_MATH_SUB,
                         0,
                         const float *,
                         &EXT_PARAM_F,
                         float,
                         float,
                         HIEDNN_DATATYPE_FP32,
                         HIEDNN_DATATYPE_FP32,
                         alpha * (x - y) + beta * z);
UTEST_BINARY_ELEMENTWISE(Mul_F32,
                         HIEDNN_BINARY_MATH_MUL,
                         0,
                         const float *,
                         &EXT_PARAM_F,
                         float,
                         float,
                         HIEDNN_DATATYPE_FP32,
                         HIEDNN_DATATYPE_FP32,
                         alpha * (x * y) + beta * z);
UTEST_BINARY_ELEMENTWISE(Div_F32,
                         HIEDNN_BINARY_MATH_DIV,
                         0,
                         const float *,
                         &EXT_PARAM_F,
                         float,
                         float,
                         HIEDNN_DATATYPE_FP32,
                         HIEDNN_DATATYPE_FP32,
                         alpha * (x / y) + beta * z);
UTEST_BINARY_ELEMENTWISE(Max_F32,
                         HIEDNN_BINARY_MATH_MAX,
                         0,
                         const float *,
                         &EXT_PARAM_F,
                         float,
                         float,
                         HIEDNN_DATATYPE_FP32,
                         HIEDNN_DATATYPE_FP32,
                         alpha * std::max(x, y) + beta * z);
UTEST_BINARY_ELEMENTWISE(Min_F32,
                         HIEDNN_BINARY_MATH_MIN,
                         0,
                         const float *,
                         &EXT_PARAM_F,
                         float,
                         float,
                         HIEDNN_DATATYPE_FP32,
                         HIEDNN_DATATYPE_FP32,
                         alpha * std::min(x, y) + beta * z);
UTEST_BINARY_ELEMENTWISE(Mod_S32,
                         HIEDNN_BINARY_MATH_MOD,
                         0,
                         const int *,
                         &EXT_MOD,
                         int32_t,
                         int32_t,
                         HIEDNN_DATATYPE_INT32,
                         HIEDNN_DATATYPE_INT32,
                         alpha * Mod<int32_t>()(x, y) + beta * z);
UTEST_BINARY_ELEMENTWISE(Mod_F32,
                         HIEDNN_BINARY_MATH_MOD,
                         0,
                         const int *,
                         &EXT_FMOD,
                         float,
                         float,
                         HIEDNN_DATATYPE_FP32,
                         HIEDNN_DATATYPE_FP32,
                         alpha * fmod(x, y) + beta * z);
UTEST_BINARY_ELEMENTWISE(Prelu_F32,
                         HIEDNN_BINARY_MATH_PRELU,
                         0,
                         const float *,
                         &EXT_PARAM_F,
                         float,
                         float,
                         HIEDNN_DATATYPE_FP32,
                         HIEDNN_DATATYPE_FP32,
                         alpha * (x >= 0 ? x : x * y) + beta * z);
UTEST_BINARY_ELEMENTWISE(EQUAL_S32,
                         HIEDNN_BINARY_COMPARE_EQ,
                         0,
                         const float *,
                         &EXT_PARAM_F,
                         int32_t,
                         char,
                         HIEDNN_DATATYPE_INT32,
                         HIEDNN_DATATYPE_BOOL,
                         x == y);
UTEST_BINARY_ELEMENTWISE(GREATER_S32,
                         HIEDNN_BINARY_COMPARE_GT,
                         0,
                         const float *,
                         &EXT_PARAM_F,
                         int32_t,
                         char,
                         HIEDNN_DATATYPE_INT32,
                         HIEDNN_DATATYPE_BOOL,
                         x > y);
UTEST_BINARY_ELEMENTWISE(GREATEROREQUAL_S32,
                         HIEDNN_BINARY_COMPARE_GE,
                         0,
                         const float *,
                         &EXT_PARAM_F,
                         int32_t,
                         char,
                         HIEDNN_DATATYPE_INT32,
                         HIEDNN_DATATYPE_BOOL,
                         x >= y);
UTEST_BINARY_ELEMENTWISE(LESS_S32,
                         HIEDNN_BINARY_COMPARE_LT,
                         0,
                         const float *,
                         &EXT_PARAM_F,
                         int32_t,
                         char,
                         HIEDNN_DATATYPE_INT32,
                         HIEDNN_DATATYPE_BOOL,
                         x < y);
UTEST_BINARY_ELEMENTWISE(LESSOREQUAL_S32,
                         HIEDNN_BINARY_COMPARE_LE,
                         0,
                         const float *,
                         &EXT_PARAM_F,
                         int32_t,
                         char,
                         HIEDNN_DATATYPE_INT32,
                         HIEDNN_DATATYPE_BOOL,
                         x <= y);
UTEST_BINARY_ELEMENTWISE(BITSHIFTR_U32,
                         HIEDNN_BINARY_MATH_BITSHIFT,
                         0,
                         const int *,
                         &EXT_RIGHT,
                         uint32_t,
                         uint32_t,
                         HIEDNN_DATATYPE_UINT32,
                         HIEDNN_DATATYPE_UINT32,
                         alpha * (x >> y) + beta * z);
UTEST_BINARY_ELEMENTWISE(BITSHIFTL_U32,
                         HIEDNN_BINARY_MATH_BITSHIFT,
                         0,
                         const int *,
                         &EXT_LEFT,
                         uint32_t,
                         uint32_t,
                         HIEDNN_DATATYPE_UINT32,
                         HIEDNN_DATATYPE_UINT32,
                         alpha * (x << y) + beta * z);
UTEST_BINARY_ELEMENTWISE(AND_BOOL,
                         HIEDNN_BINARY_LOGICAL_AND,
                         0,
                         const float *,
                         &EXT_PARAM_F,
                         char,
                         char,
                         HIEDNN_DATATYPE_BOOL,
                         HIEDNN_DATATYPE_BOOL,
                         x && y);
UTEST_BINARY_ELEMENTWISE(OR_BOOL,
                         HIEDNN_BINARY_LOGICAL_OR,
                         0,
                         const float *,
                         &EXT_PARAM_F,
                         char,
                         char,
                         HIEDNN_DATATYPE_BOOL,
                         HIEDNN_DATATYPE_BOOL,
                         x || y);
UTEST_BINARY_ELEMENTWISE(XOR_BOOL,
                         HIEDNN_BINARY_LOGICAL_XOR,
                         0,
                         const float *,
                         &EXT_PARAM_F,
                         char,
                         char,
                         HIEDNN_DATATYPE_BOOL,
                         HIEDNN_DATATYPE_BOOL,
                         !x != !y);

#undef UTEST_BINARY_ELEMENTWISE

#define UTEST_BINARY_ELEMENTWISE_BROADCAST( \
        TEST_NAME, UTEST_SIZE, HIE_BINARY_OP, BIAS, \
        EXT_TYPE, EXT_VAL, ST, DT, INTYPE, OUTTYPE, \
        X_TENSORFORMAT, Y_TENSORFORMAT, Z_TENSORFORMAT, STD_EXPR) \
TEST(Binary_Elementwise_Broadcast_CUDA, TEST_NAME) { \
    unsigned int seed = 0; \
    for (int i = 0; i < UTEST_SIZE; ++i) { \
        const auto &shape = BROADCAST_CASES[i]; \
        ST *x, *y; \
        DT *z;\
        x = static_cast<ST *>(malloc(sizeof(ST) * shape.X_len)); \
        for (int i = 0; i < shape.X_len; ++i) { \
            x[i] = BIAS + \
                   static_cast<ST>(rand_r(&seed)) / \
                   static_cast<ST>(RAND_MAX); \
        } \
        \
        y = static_cast<ST *>(malloc(sizeof(ST) * shape.Y_len)); \
        for (int i = 0; i < shape.Y_len; ++i) { \
            y[i] = static_cast<ST>(rand_r(&seed)) / \
                   static_cast<ST>(RAND_MAX); \
        } \
        \
        z = static_cast<DT *>(malloc(sizeof(DT) * shape.Z_len)); \
        for (int i = 0; i < shape.Z_len; ++i) { \
            z[i] = static_cast<DT>(rand_r(&seed)) / \
                   static_cast<DT>(RAND_MAX); \
        }\
        \
        ST *x_cu, *y_cu; \
        DT *z_cu; \
        cudaMalloc(&x_cu, shape.X_len * sizeof(ST)); \
        cudaMalloc(&y_cu, shape.Y_len * sizeof(ST)); \
        cudaMalloc(&z_cu, shape.Z_len * sizeof(DT)); \
        cudaMemcpy(x_cu, x, shape.X_len * sizeof(ST), cudaMemcpyHostToDevice); \
        cudaMemcpy(y_cu, y, shape.Y_len * sizeof(ST), cudaMemcpyHostToDevice); \
        cudaMemcpy(z_cu, z, shape.Z_len * sizeof(DT), cudaMemcpyHostToDevice); \
        \
        ST alpha = 2; \
        ST beta = 2; \
        EXT_TYPE extParam = EXT_VAL; \
        \
        hiednnCudaHandle_t handle; \
        CHECK_HIEDNN(hiednnCreateCudaHandle(&handle)); \
        \
        hiednnTensorDesc_t xDesc; \
        CHECK_HIEDNN(hiednnCreateTensorDesc(&xDesc)); \
        CHECK_HIEDNN(hiednnSet4dTensorDesc( \
            xDesc, INTYPE, X_TENSORFORMAT, \
            shape.X_shape[0], shape.X_shape[1], \
            shape.X_shape[2], shape.X_shape[3])); \
        \
        hiednnTensorDesc_t yDesc; \
        CHECK_HIEDNN(hiednnCreateTensorDesc(&yDesc)); \
        CHECK_HIEDNN(hiednnSet4dTensorDesc( \
            yDesc, INTYPE, Y_TENSORFORMAT, \
            shape.Y_shape[0], shape.Y_shape[1], \
            shape.Y_shape[2], shape.Y_shape[3])); \
        \
        hiednnTensorDesc_t zDesc; \
        CHECK_HIEDNN(hiednnCreateTensorDesc(&zDesc)); \
        CHECK_HIEDNN(hiednnSet4dTensorDesc( \
            zDesc, OUTTYPE, Z_TENSORFORMAT, \
            shape.Z_shape[0], shape.Z_shape[1], \
            shape.Z_shape[2], shape.Z_shape[3])); \
        \
        CHECK_HIEDNN(hiednnCudaBinaryElementwiseOp( \
            handle, HIE_BINARY_OP, static_cast<const void *>(&alpha), \
            xDesc, static_cast<const void *>(x_cu), \
            yDesc, static_cast<const void *>(y_cu), \
            static_cast<const void *>(extParam), \
            static_cast<const void *>(&beta), \
            zDesc, static_cast<void *>(z_cu))); \
        \
        DT *z_out; \
        z_out = static_cast<DT *>(malloc(sizeof(DT) * shape.Z_len));\
        cudaMemcpy(z_out, z_cu, shape.Z_len * sizeof(DT), \
                   cudaMemcpyDeviceToHost); \
        \
        struct Ref { \
            float operator()(const ST &x, \
                             const ST &y, \
                             const DT &z, \
                             const ST &alpha, \
                             const ST &beta) { \
                return STD_EXPR; \
            } \
        }; \
        \
        int xStrides[4], yStrides[4], zStrides[4]; \
        int xSize = 1, ySize = 1, zSize = 1; \
        for (int i = 3; i >= 0; --i) { \
            xStrides[i] = xSize; \
            yStrides[i] = ySize; \
            zStrides[i] = zSize; \
            xSize *= shape.X_shape[i]; \
            ySize *= shape.Y_shape[i]; \
            zSize *= shape.Z_shape[i]; \
        } \
        for (int in = 0; in < shape.Z_shape[0]; ++in) { \
            for (int ic = 0; ic < shape.Z_shape[1]; ++ic) { \
                for (int ih = 0; ih < shape.Z_shape[2]; ++ih) { \
                    for (int iw = 0; iw < shape.Z_shape[3]; ++iw) { \
                        uint32_t x_offset = \
                            std::min(in, shape.X_shape[0] - 1) * xStrides[0] \
                          + std::min(ic, shape.X_shape[1] - 1) * xStrides[1] \
                          + std::min(ih, shape.X_shape[2] - 1) * xStrides[2] \
                          + std::min(iw, shape.X_shape[3] - 1) * xStrides[3]; \
                        uint32_t y_offset = \
                            std::min(in, shape.Y_shape[0] - 1) * yStrides[0] \
                          + std::min(ic, shape.Y_shape[1] - 1) * yStrides[1] \
                          + std::min(ih, shape.Y_shape[2] - 1) * yStrides[2] \
                          + std::min(iw, shape.Y_shape[3] - 1) * yStrides[3]; \
                        uint32_t z_offset = in * zStrides[0] + \
                                            ic * zStrides[1] + \
                                            ih * zStrides[2] + \
                                            iw * zStrides[3]; \
                        DT std_val = Ref()(x[x_offset], \
                                           y[y_offset], \
                                           z[z_offset], \
                                           alpha, \
                                           beta); \
                        ASSERT_NEAR(z_out[z_offset], std_val, \
                                    std::fabs(std_val) * 1e-5); \
                    } \
                } \
            } \
        } \
        \
        cudaFree(x_cu); \
        cudaFree(y_cu); \
        cudaFree(z_cu); \
        free(x); \
        free(y); \
        free(z); \
        free(z_out); \
        CHECK_HIEDNN(hiednnDestroyCudaHandle(handle)); \
        CHECK_HIEDNN(hiednnDestroyTensorDesc(xDesc)); \
        CHECK_HIEDNN(hiednnDestroyTensorDesc(yDesc)); \
        CHECK_HIEDNN(hiednnDestroyTensorDesc(zDesc)); \
    } \
}

UTEST_BINARY_ELEMENTWISE_BROADCAST(
        Add_F32,
        4,
        HIEDNN_BINARY_MATH_ADD,
        0,
        const float *,
        &EXT_PARAM_F,
        float,
        float,
        HIEDNN_DATATYPE_FP32,
        HIEDNN_DATATYPE_FP32,
        HIEDNN_TENSORFORMAT_NORMAL,
        HIEDNN_TENSORFORMAT_NORMAL,
        HIEDNN_TENSORFORMAT_NORMAL,
        alpha * (x + y) + beta * z);
UTEST_BINARY_ELEMENTWISE_BROADCAST(
        Sub_F32,
        4,
        HIEDNN_BINARY_MATH_SUB,
        0,
        const float *,
        &EXT_PARAM_F,
        float,
        float,
        HIEDNN_DATATYPE_FP32,
        HIEDNN_DATATYPE_FP32,
        HIEDNN_TENSORFORMAT_NORMAL,
        HIEDNN_TENSORFORMAT_NORMAL,
        HIEDNN_TENSORFORMAT_NORMAL,
        alpha * (x - y) + beta * z);
UTEST_BINARY_ELEMENTWISE_BROADCAST(
        Mul_F32,
        4,
        HIEDNN_BINARY_MATH_MUL,
        0,
        const float *,
        &EXT_PARAM_F,
        float,
        float,
        HIEDNN_DATATYPE_FP32,
        HIEDNN_DATATYPE_FP32,
        HIEDNN_TENSORFORMAT_NORMAL,
        HIEDNN_TENSORFORMAT_NORMAL,
        HIEDNN_TENSORFORMAT_NORMAL,
        alpha * (x * y) + beta * z);
UTEST_BINARY_ELEMENTWISE_BROADCAST(
        Div_F32,
        4,
        HIEDNN_BINARY_MATH_DIV,
        0,
        const float *,
        &EXT_PARAM_F,
        float,
        float,
        HIEDNN_DATATYPE_FP32,
        HIEDNN_DATATYPE_FP32,
        HIEDNN_TENSORFORMAT_NORMAL,
        HIEDNN_TENSORFORMAT_NORMAL,
        HIEDNN_TENSORFORMAT_NORMAL,
        alpha * (x / y) + beta * z);
UTEST_BINARY_ELEMENTWISE_BROADCAST(
        Max_F32,
        4,
        HIEDNN_BINARY_MATH_MAX,
        0,
        const float *,
        &EXT_PARAM_F,
        float,
        float,
        HIEDNN_DATATYPE_FP32,
        HIEDNN_DATATYPE_FP32,
        HIEDNN_TENSORFORMAT_NORMAL,
        HIEDNN_TENSORFORMAT_NORMAL,
        HIEDNN_TENSORFORMAT_NORMAL,
        alpha * std::max(x, y) + beta * z);
UTEST_BINARY_ELEMENTWISE_BROADCAST(
        Min_F32,
        4,
        HIEDNN_BINARY_MATH_MIN,
        0,
        const float *,
        &EXT_PARAM_F,
        float,
        float,
        HIEDNN_DATATYPE_FP32,
        HIEDNN_DATATYPE_FP32,
        HIEDNN_TENSORFORMAT_NORMAL,
        HIEDNN_TENSORFORMAT_NORMAL,
        HIEDNN_TENSORFORMAT_NORMAL,
        alpha * std::min(x, y) + beta * z);
UTEST_BINARY_ELEMENTWISE_BROADCAST(
        Mod_S32,
        4,
        HIEDNN_BINARY_MATH_MOD,
        0,
        const int *,
        &EXT_MOD,
        int32_t,
        int32_t,
        HIEDNN_DATATYPE_INT32,
        HIEDNN_DATATYPE_INT32,
        HIEDNN_TENSORFORMAT_NORMAL,
        HIEDNN_TENSORFORMAT_NORMAL,
        HIEDNN_TENSORFORMAT_NORMAL,
        alpha * Mod<int32_t>()(x, y) + beta * z);
UTEST_BINARY_ELEMENTWISE_BROADCAST(
        Mod_F32,
        4,
        HIEDNN_BINARY_MATH_MOD,
        0,
        const int *,
        &EXT_FMOD,
        float,
        float,
        HIEDNN_DATATYPE_FP32,
        HIEDNN_DATATYPE_FP32,
        HIEDNN_TENSORFORMAT_NORMAL,
        HIEDNN_TENSORFORMAT_NORMAL,
        HIEDNN_TENSORFORMAT_NORMAL,
        alpha * fmod(x, y) + beta * z);
UTEST_BINARY_ELEMENTWISE_BROADCAST(
        Prelu_F32,
        1,
        HIEDNN_BINARY_MATH_PRELU,
        0,
        const float *,
        &EXT_PARAM_F,
        float,
        float,
        HIEDNN_DATATYPE_FP32,
        HIEDNN_DATATYPE_FP32,
        HIEDNN_TENSORFORMAT_NORMAL,
        HIEDNN_TENSORFORMAT_NORMAL,
        HIEDNN_TENSORFORMAT_NORMAL,
        alpha * (x >= 0 ? x : x * y) + beta * z);
UTEST_BINARY_ELEMENTWISE_BROADCAST(
        EQUAL_S32,
        4,
        HIEDNN_BINARY_COMPARE_EQ,
        0,
        const float *,
        &EXT_PARAM_F,
        int32_t,
        char,
        HIEDNN_DATATYPE_INT32,
        HIEDNN_DATATYPE_BOOL,
        HIEDNN_TENSORFORMAT_NORMAL,
        HIEDNN_TENSORFORMAT_NORMAL,
        HIEDNN_TENSORFORMAT_NORMAL,
        x == y);
UTEST_BINARY_ELEMENTWISE_BROADCAST(
        GREATER_S32,
        4,
        HIEDNN_BINARY_COMPARE_GT,
        0,
        const float *,
        &EXT_PARAM_F,
        int32_t,
        char,
        HIEDNN_DATATYPE_INT32,
        HIEDNN_DATATYPE_BOOL,
        HIEDNN_TENSORFORMAT_NORMAL,
        HIEDNN_TENSORFORMAT_NORMAL,
        HIEDNN_TENSORFORMAT_NORMAL,
        x > y);
UTEST_BINARY_ELEMENTWISE_BROADCAST(
        GREATEROREQUAL_S32,
        4,
        HIEDNN_BINARY_COMPARE_GE,
        0,
        const float *,
        &EXT_PARAM_F,
        int32_t,
        char,
        HIEDNN_DATATYPE_INT32,
        HIEDNN_DATATYPE_BOOL,
        HIEDNN_TENSORFORMAT_NORMAL,
        HIEDNN_TENSORFORMAT_NORMAL,
        HIEDNN_TENSORFORMAT_NORMAL,
        x >= y);
UTEST_BINARY_ELEMENTWISE_BROADCAST(
        LESS_S32,
        4,
        HIEDNN_BINARY_COMPARE_LT,
        0,
        const float *,
        &EXT_PARAM_F,
        int32_t,
        char,
        HIEDNN_DATATYPE_INT32,
        HIEDNN_DATATYPE_BOOL,
        HIEDNN_TENSORFORMAT_NORMAL,
        HIEDNN_TENSORFORMAT_NORMAL,
        HIEDNN_TENSORFORMAT_NORMAL,
        x < y);
UTEST_BINARY_ELEMENTWISE_BROADCAST(
        LESSOREQUAL_S32,
        4,
        HIEDNN_BINARY_COMPARE_LE,
        0,
        const float *,
        &EXT_PARAM_F,
        int32_t,
        char,
        HIEDNN_DATATYPE_INT32,
        HIEDNN_DATATYPE_BOOL,
        HIEDNN_TENSORFORMAT_NORMAL,
        HIEDNN_TENSORFORMAT_NORMAL,
        HIEDNN_TENSORFORMAT_NORMAL,
        x <= y);
UTEST_BINARY_ELEMENTWISE_BROADCAST(
        BITSHIFTR_U32,
        4,
        HIEDNN_BINARY_MATH_BITSHIFT,
        0,
        const int *,
        &EXT_RIGHT,
        uint32_t,
        uint32_t,
        HIEDNN_DATATYPE_UINT32,
        HIEDNN_DATATYPE_UINT32,
        HIEDNN_TENSORFORMAT_NORMAL,
        HIEDNN_TENSORFORMAT_NORMAL,
        HIEDNN_TENSORFORMAT_NORMAL,
        alpha * (x >> y) + beta * z);
UTEST_BINARY_ELEMENTWISE_BROADCAST(
        BITSHIFTL_U32,
        4,
        HIEDNN_BINARY_MATH_BITSHIFT,
        0,
        const int *,
        &EXT_LEFT,
        uint32_t,
        uint32_t,
        HIEDNN_DATATYPE_UINT32,
        HIEDNN_DATATYPE_UINT32,
        HIEDNN_TENSORFORMAT_NORMAL,
        HIEDNN_TENSORFORMAT_NORMAL,
        HIEDNN_TENSORFORMAT_NORMAL,
        alpha * (x << y) + beta * z);
UTEST_BINARY_ELEMENTWISE_BROADCAST(
        AND_BOOL,
        4,
        HIEDNN_BINARY_LOGICAL_AND,
        0,
        const float *,
        &EXT_PARAM_F,
        char,
        char,
        HIEDNN_DATATYPE_BOOL,
        HIEDNN_DATATYPE_BOOL,
        HIEDNN_TENSORFORMAT_NORMAL,
        HIEDNN_TENSORFORMAT_NORMAL,
        HIEDNN_TENSORFORMAT_NORMAL,
        x && y);
UTEST_BINARY_ELEMENTWISE_BROADCAST(
        OR_BOOL,
        4,
        HIEDNN_BINARY_LOGICAL_OR,
        0,
        const float *,
        &EXT_PARAM_F,
        char,
        char,
        HIEDNN_DATATYPE_BOOL,
        HIEDNN_DATATYPE_BOOL,
        HIEDNN_TENSORFORMAT_NORMAL,
        HIEDNN_TENSORFORMAT_NORMAL,
        HIEDNN_TENSORFORMAT_NORMAL,
        x || y);
UTEST_BINARY_ELEMENTWISE_BROADCAST(
        XOR_BOOL,
        4,
        HIEDNN_BINARY_LOGICAL_XOR,
        0,
        const float *,
        &EXT_PARAM_F,
        char,
        char,
        HIEDNN_DATATYPE_BOOL,
        HIEDNN_DATATYPE_BOOL,
        HIEDNN_TENSORFORMAT_NORMAL,
        HIEDNN_TENSORFORMAT_NORMAL,
        HIEDNN_TENSORFORMAT_NORMAL,
        !x != !y);

#undef UTEST_BINARY_ELEMENTWISE_BROADCAST

