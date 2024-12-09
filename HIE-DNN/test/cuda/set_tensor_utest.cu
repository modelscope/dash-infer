/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    set_tensor_utest.cu
 */
#include <hiednn.h>
#include <hiednn_cuda.h>

#include <gtest/gtest.h>
#include <cstdint>
#include <cmath>

#include <utest_utils.hpp>

namespace {
const int64_t VECTOR_LENGTH = 3003;
}  // anonymous namespace

#define UTEST_SET_TENSOR(TEST_NAME, DATATYPE, HIEDNN_DATA_TYPE, VALUE) \
TEST(SetTensorValueConst_CUDA, TEST_NAME) { \
    DATATYPE *y_cu; \
    cudaMalloc(&y_cu, VECTOR_LENGTH * sizeof(DATATYPE)); \
    \
    hiednnCudaHandle_t handle; \
    CHECK_HIEDNN(hiednnCreateCudaHandle(&handle)); \
    \
    hiednnTensorDesc_t y_desc; \
    CHECK_HIEDNN(hiednnCreateTensorDesc(&y_desc)); \
    CHECK_HIEDNN(hiednnSet4dTensorDesc( \
        y_desc, HIEDNN_DATA_TYPE, HIEDNN_TENSORFORMAT_NORMAL, \
        1, 1, 1, VECTOR_LENGTH)); \
    \
    DATATYPE value = VALUE; \
    CHECK_HIEDNN(hiednnCudaSetTensorValue( \
        handle, HIEDNN_SETTENSOR_CONST, \
        &value, nullptr, y_desc, y_cu)); \
    \
    DATATYPE *y = new DATATYPE[VECTOR_LENGTH]; \
    cudaMemcpy(y, y_cu, VECTOR_LENGTH * sizeof(DATATYPE), \
               cudaMemcpyDeviceToHost); \
    \
    for (int64_t i = 0; i < VECTOR_LENGTH; ++i) { \
        ASSERT_EQ(y[i], VALUE); \
    } \
    \
    cudaFree(y_cu); \
    delete[] y; \
    CHECK_HIEDNN(hiednnDestroyCudaHandle(handle)); \
    CHECK_HIEDNN(hiednnDestroyTensorDesc(y_desc)); \
}

UTEST_SET_TENSOR(FP32, float, HIEDNN_DATATYPE_FP32, 2.5);
UTEST_SET_TENSOR(INT32, int32_t, HIEDNN_DATATYPE_INT32, 10);
UTEST_SET_TENSOR(INT16, int16_t, HIEDNN_DATATYPE_INT16, 20);

#define UTEST_RANGE_SET_TENSOR(TEST_NAME, \
                               DATATYPE, \
                               HIEDNN_DATA_TYPE, \
                               START, \
                               DELTA) \
TEST(SetTensorValueRange_CUDA, TEST_NAME) { \
    DATATYPE *y_cu; \
    cudaMalloc(&y_cu, VECTOR_LENGTH * sizeof(DATATYPE)); \
    \
    hiednnCudaHandle_t handle; \
    CHECK_HIEDNN(hiednnCreateCudaHandle(&handle)); \
    \
    int64_t yDim[] = {VECTOR_LENGTH}; \
    hiednnTensorDesc_t y_desc; \
    CHECK_HIEDNN(hiednnCreateTensorDesc(&y_desc)); \
    CHECK_HIEDNN(hiednnSetNormalTensorDesc( \
        y_desc, HIEDNN_DATA_TYPE, 1, yDim)); \
    \
    DATATYPE start = START; \
    DATATYPE delta = DELTA; \
    CHECK_HIEDNN(hiednnCudaSetTensorValue( \
        handle, HIEDNN_SETTENSOR_RANGE, \
        &start, &delta, y_desc, y_cu)); \
    \
    DATATYPE *y = new DATATYPE[VECTOR_LENGTH]; \
    cudaMemcpy(y, y_cu, VECTOR_LENGTH * sizeof(DATATYPE), \
               cudaMemcpyDeviceToHost); \
    \
    for (int64_t i = 0; i < VECTOR_LENGTH; ++i) { \
        DATATYPE std_val = start + i * delta; \
        ASSERT_NEAR(y[i], std_val, std::abs(std_val) * 1e-5); \
    } \
    \
    cudaFree(y_cu); \
    delete[] y; \
    CHECK_HIEDNN(hiednnDestroyCudaHandle(handle)); \
    CHECK_HIEDNN(hiednnDestroyTensorDesc(y_desc)); \
}

UTEST_RANGE_SET_TENSOR(FP32, float, HIEDNN_DATATYPE_FP32, 1.5f, 0.3f)
UTEST_RANGE_SET_TENSOR(INT32, int32_t, HIEDNN_DATATYPE_INT32, 10, 3)
UTEST_RANGE_SET_TENSOR(INT64, int64_t, HIEDNN_DATATYPE_INT64, 5, -2)

#define UTEST_DIAGONAL_SET_TENSOR(TEST_NAME, \
                                  DATATYPE, \
                                  HIEDNN_DATA_TYPE, \
                                  DIAGONAL_M, \
                                  DIAGONAL_N, \
                                  RSHIFT, \
                                  VALUE) \
TEST(DiagonalSetTensorValueDiagonal_CUDA, TEST_NAME) { \
    DATATYPE *y_cu; \
    cudaMalloc(&y_cu, DIAGONAL_M * DIAGONAL_N* sizeof(DATATYPE)); \
    \
    hiednnCudaHandle_t handle; \
    CHECK_HIEDNN(hiednnCreateCudaHandle(&handle)); \
    \
    int64_t yDim[] = {DIAGONAL_M, DIAGONAL_N}; \
    hiednnTensorDesc_t y_desc; \
    CHECK_HIEDNN(hiednnCreateTensorDesc(&y_desc)); \
    CHECK_HIEDNN(hiednnSetNormalTensorDesc( \
        y_desc, HIEDNN_DATA_TYPE, 2, yDim)); \
    \
    DATATYPE value = VALUE; \
    int rShift = RSHIFT; \
    CHECK_HIEDNN(hiednnCudaSetTensorValue( \
        handle, HIEDNN_SETTENSOR_DIAGONAL, \
        &rShift, &value, y_desc, y_cu)); \
    \
    DATATYPE *y = new DATATYPE[DIAGONAL_M * DIAGONAL_N]; \
    cudaMemcpy(y, y_cu, DIAGONAL_M * DIAGONAL_N * sizeof(DATATYPE), \
               cudaMemcpyDeviceToHost); \
    \
    for (int64_t m_idx = 0; m_idx < DIAGONAL_M; ++m_idx) { \
        for (int64_t n_idx = 0; n_idx < DIAGONAL_N; ++n_idx) { \
            if (n_idx == m_idx + rShift) { \
                ASSERT_EQ(y[m_idx * DIAGONAL_N + n_idx], VALUE); \
            } else { \
                ASSERT_EQ(y[m_idx * DIAGONAL_N + n_idx], 0); \
            } \
        } \
    } \
    \
    cudaFree(y_cu); \
    delete[] y; \
    CHECK_HIEDNN(hiednnDestroyCudaHandle(handle)); \
    CHECK_HIEDNN(hiednnDestroyTensorDesc(y_desc)); \
}

UTEST_DIAGONAL_SET_TENSOR(FP32_100x100_R5,
    float, HIEDNN_DATATYPE_FP32, 100, 100, 5, 2.5);
UTEST_DIAGONAL_SET_TENSOR(INT16_100x100_R5,
    int16_t, HIEDNN_DATATYPE_INT16, 100, 100, 5, 2);
UTEST_DIAGONAL_SET_TENSOR(FP32_100x200_R10,
    float, HIEDNN_DATATYPE_FP32, 100, 200, 10, 2.5);
UTEST_DIAGONAL_SET_TENSOR(FP32_100x200_L10,
    float, HIEDNN_DATATYPE_FP32, 100, 200, -10, 2.5);
UTEST_DIAGONAL_SET_TENSOR(FP32_200x100_R10,
    float, HIEDNN_DATATYPE_FP32, 200, 100, 10, 2.5);
UTEST_DIAGONAL_SET_TENSOR(FP32_200x100_L10,
    float, HIEDNN_DATATYPE_FP32, 200, 100, -10, 2.5);


