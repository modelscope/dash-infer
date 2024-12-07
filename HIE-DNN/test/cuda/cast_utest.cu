/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    cast_utest.cu
 */
#include <hiednn.h>
#include <hiednn_cuda.h>

#include <gtest/gtest.h>
#include <cstdint>

#include <utest_utils.hpp>

namespace {

const int VECTOR_LENGTH = 3003;

template <typename T>
T GetRand(unsigned int *seed) {
    return static_cast<T>(rand_r(seed));
}

template <>
float GetRand<float>(unsigned int *seed) {
    return static_cast<float>(rand_r(seed)) /
           static_cast<float>(RAND_MAX);
}

template <>
double GetRand<double>(unsigned int *seed) {
    return static_cast<double>(rand_r(seed)) /
           static_cast<double>(RAND_MAX);
}

}  // anonymous namespace

#define UTEST_CAST(TEST_NAME, ST, HIEDNN_ST, DT, HIEDNN_DT) \
TEST(Cast_CUDA, TEST_NAME) { \
    unsigned int seed = 0; \
    ST *x = new ST[VECTOR_LENGTH]; \
    for (int i = 0; i < VECTOR_LENGTH; ++i) { \
        x[i] = GetRand<ST>(&seed); \
    } \
    \
    ST *x_cu; \
    DT *y_cu; \
    cudaMalloc(&x_cu, VECTOR_LENGTH * sizeof(ST)); \
    cudaMalloc(&y_cu, VECTOR_LENGTH * sizeof(DT)); \
    cudaMemcpy(x_cu, x, VECTOR_LENGTH * sizeof(ST), \
               cudaMemcpyHostToDevice); \
    \
    hiednnCudaHandle_t handle; \
    CHECK_HIEDNN(hiednnCreateCudaHandle(&handle)); \
    \
    hiednnTensorDesc_t x_desc; \
    hiednnTensorDesc_t y_desc; \
    CHECK_HIEDNN(hiednnCreateTensorDesc(&x_desc)); \
    CHECK_HIEDNN(hiednnCreateTensorDesc(&y_desc)); \
    CHECK_HIEDNN(hiednnSet4dTensorDesc( \
        x_desc, HIEDNN_ST, HIEDNN_TENSORFORMAT_NORMAL, \
        1, 1, 1, VECTOR_LENGTH)); \
    CHECK_HIEDNN(hiednnSet4dTensorDesc( \
        y_desc, HIEDNN_DT, HIEDNN_TENSORFORMAT_NORMAL, \
        1, 1, 1, VECTOR_LENGTH)); \
    \
    CHECK_HIEDNN(hiednnCudaCast(handle, x_desc, x_cu, y_desc, y_cu)); \
    \
    DT *y = new DT[VECTOR_LENGTH]; \
    cudaMemcpy(y, y_cu, VECTOR_LENGTH * sizeof(DT), \
               cudaMemcpyDeviceToHost); \
    \
    for (int i = 0; i < VECTOR_LENGTH; ++i) { \
        DT std_val = static_cast<DT>(x[i]); \
        ASSERT_EQ(y[i], std_val); \
    } \
    \
    cudaFree(x_cu); \
    cudaFree(y_cu); \
    delete[] x; \
    delete[] y; \
    CHECK_HIEDNN(hiednnDestroyCudaHandle(handle)); \
    CHECK_HIEDNN(hiednnDestroyTensorDesc(x_desc)); \
    CHECK_HIEDNN(hiednnDestroyTensorDesc(y_desc)); \
}

UTEST_CAST(float_double,
           float, HIEDNN_DATATYPE_FP32,
           double, HIEDNN_DATATYPE_FP64);

UTEST_CAST(double_float,
           double, HIEDNN_DATATYPE_FP64,
           float, HIEDNN_DATATYPE_FP32);

UTEST_CAST(int32_float,
           int32_t, HIEDNN_DATATYPE_INT32,
           float, HIEDNN_DATATYPE_FP32);

UTEST_CAST(float_int32,
           float, HIEDNN_DATATYPE_FP32,
           int32_t, HIEDNN_DATATYPE_INT32);

UTEST_CAST(int32_int16,
           int32_t, HIEDNN_DATATYPE_INT32,
           int16_t, HIEDNN_DATATYPE_INT16);

UTEST_CAST(int16_int32,
           int16_t, HIEDNN_DATATYPE_INT16,
           int32_t, HIEDNN_DATATYPE_INT32);

UTEST_CAST(int32_uint32,
           int32_t, HIEDNN_DATATYPE_INT32,
           uint32_t, HIEDNN_DATATYPE_UINT32);

UTEST_CAST(uint32_int32,
           uint32_t, HIEDNN_DATATYPE_UINT32,
           int32_t, HIEDNN_DATATYPE_INT32);

UTEST_CAST(int16_uint32,
           int16_t, HIEDNN_DATATYPE_INT16,
           uint32_t, HIEDNN_DATATYPE_UINT32);

UTEST_CAST(uint32_int16,
           uint32_t, HIEDNN_DATATYPE_UINT32,
           int16_t, HIEDNN_DATATYPE_INT16);

UTEST_CAST(uint16_int32,
           uint16_t, HIEDNN_DATATYPE_UINT16,
           int32_t, HIEDNN_DATATYPE_INT32);

UTEST_CAST(int32_uint16,
           int32_t, HIEDNN_DATATYPE_INT32,
           uint16_t, HIEDNN_DATATYPE_UINT16);

