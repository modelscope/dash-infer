/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    concat_utest.cu
 */
#include <hiednn.h>
#include <hiednn_cuda.h>

#include <gtest/gtest.h>
#include <cstdint>
#include <cstdlib>

#include <utest_utils.hpp>

namespace {

struct TestCase {
    static const int MAX_INPUT_COUNT = 128;
    static const int MAX_NDIMS = 8;

    int inputCount;
    int nDims;
    int axis;

    int64_t xDims[MAX_INPUT_COUNT][MAX_NDIMS];
    int64_t yDims[MAX_NDIMS];

    int64_t xSize[MAX_INPUT_COUNT];
    int64_t ySize;

    template <int NDIMS, typename T1, int INPUT_COUNT, typename T2>
    TestCase(const T1 (&dimsIn)[NDIMS],
             const T2 (&concatDimsIn)[INPUT_COUNT],
             int axisIn) {
        inputCount = INPUT_COUNT;
        nDims = NDIMS;
        axis = axisIn;

        int64_t yConcatDim = 0;

        for (int i = 0; i < INPUT_COUNT; ++i) {
            for (int j = 0; j < axis; ++j) {
                xDims[i][j] = dimsIn[j];
            }
            xDims[i][axis] = concatDimsIn[i];
            for (int j = axis + 1; j < NDIMS; ++j) {
                xDims[i][j] = dimsIn[j];
            }
            for (int j = NDIMS; j < MAX_NDIMS; ++j) {
                xDims[i][j] = 1;
            }

            yConcatDim += concatDimsIn[i];
        }

        for (int i = 0; i < MAX_NDIMS; ++i) {
            yDims[i] = xDims[0][i];
        }
        yDims[axis] = yConcatDim;

        SetTensorSize();
    }

    template <int NDIMS, typename T>
    TestCase(const T (&dimsIn)[NDIMS],
             int inputCountIn,
             int concatDimMin,
             int concatDimMax,
             int axisIn) {
        inputCount = inputCountIn;
        nDims = NDIMS;
        axis = axisIn;

        unsigned seed = 0;
        int64_t mod = concatDimMax - concatDimMin;
        int64_t yConcatDim = 0;

        for (int i = 0; i < inputCount; ++i) {
            for (int j = 0; j < axis; ++j) {
                xDims[i][j] = dimsIn[j];
            }
            xDims[i][axis] = static_cast<int64_t>(rand_r(&seed)) % mod +
                             concatDimMin;
            for (int j = axis + 1; j < NDIMS; ++j) {
                xDims[i][j] = dimsIn[j];
            }
            for (int j = NDIMS; j < MAX_NDIMS; ++j) {
                xDims[i][j] = 1;
            }

            yConcatDim += xDims[i][axis];
        }

        for (int i = 0; i < MAX_NDIMS; ++i) {
            yDims[i] = xDims[0][i];
        }
        yDims[axis] = yConcatDim;

        SetTensorSize();
    }

    void SetTensorSize() {
        for (int i = 0; i < inputCount; ++i) {
            xSize[i] = 1;
            for (int j = 0; j < nDims; ++j) {
                xSize[i] *= xDims[i][j];
            }
        }

        ySize = 1;
        for (int i = 0; i < nDims; ++i) {
            ySize *= yDims[i];
        }
    }
};

std::vector<TestCase> fastMapCases = {
    { {33, 43, 1, 11}, {7, 9, 5, 12, 11}, 2 },
    { {33, 43, 1, 12}, {7, 9, 5, 12, 11}, 2 },
};

std::vector<TestCase> serialSearchCases = {
    { {33, 43, 1, 65}, {9, 17, 13, 19, 11}, 2 },
    { {33, 43, 55, 1}, {9, 14, 13, 16, 11}, 3 },
};

std::vector<TestCase> hybridSearchCases = {
    { {33, 43, 54, 65}, 99, 7, 19, 2 },
};

std::vector<TestCase> smallBatchCases = {
    { {33, 43, 55, 65}, 19, 3, 11, 1 },
    { {11, 1}, {12, 16, 24, 8}, 1 }
};

template <typename T>
void ConcatRef(const T * const *x, T *y, const TestCase &testCase) {
    const auto &inputCount = testCase.inputCount;
    const auto &nDims = testCase.nDims;
    const auto &axis = testCase.axis;
    const auto &xDims = testCase.xDims;

    int batch = 1;
    for (int i = 0; i < axis; ++i) {
        batch *= xDims[0][i];
    }

    int xConcatDims[TestCase::MAX_INPUT_COUNT];
    for (int i = 0; i < inputCount; ++i) {
        xConcatDims[i] = 1;
        for (int j = axis; j < nDims; ++j) {
            xConcatDims[i] *= xDims[i][j];
        }
    }

    int yConcatDim = 0;
    for (int i = 0; i < inputCount; ++i) {
        yConcatDim += xConcatDims[i];
    }

    for (int i = 0; i < batch; ++i) {
        T *yPtr = y + i * yConcatDim;
        for (int j = 0; j < inputCount; ++j) {
            memcpy(yPtr, x[j] + i * xConcatDims[j], xConcatDims[j] * sizeof(T));
            yPtr += xConcatDims[j];
        }
    }
}

}  // anonymous namespace

#define UTEST_CONCAT(TEST_NAME, CASE_SET, CTYPE, HIEDNN_TYPE) \
TEST(Concat_CUDA, TEST_NAME) { \
    hiednnCudaHandle_t handle; \
    CHECK_HIEDNN(hiednnCreateCudaHandle(&handle)); \
    hiednnTensorDesc_t xDesc[128], yDesc; \
    for (int i = 0; i < 128; ++i) { \
        CHECK_HIEDNN(hiednnCreateTensorDesc(&xDesc[i])); \
    } \
    CHECK_HIEDNN(hiednnCreateTensorDesc(&yDesc)); \
    \
    CTYPE *hXPtrs[128]; \
    CTYPE *dXPtrs[128]; \
    \
    for (const auto &testCase : CASE_SET) { \
        for (int i = 0; i < testCase.inputCount; ++i) { \
            CHECK_CUDA(cudaMallocHost(&hXPtrs[i], \
                       testCase.xSize[i] * sizeof(CTYPE))); \
            CHECK_CUDA(cudaMalloc(&dXPtrs[i], \
                       testCase.xSize[i] * sizeof(CTYPE))); \
        }\
        \
        for (int i = 0; i < testCase.inputCount; ++i) { \
            for (int j = 0; j < testCase.xSize[i]; ++j) { \
                hXPtrs[i][j] = CTYPE(j); \
            } \
            CHECK_CUDA(cudaMemcpy(dXPtrs[i], hXPtrs[i], \
                                  testCase.xSize[i] * sizeof(CTYPE), \
                                  cudaMemcpyHostToDevice)); \
        }\
        \
        for (int i = 0; i < testCase.inputCount; ++i) { \
            CHECK_HIEDNN(hiednnSetNormalTensorDesc( \
                xDesc[i], HIEDNN_TYPE, testCase.nDims, testCase.xDims[i])); \
        }\
        \
        CHECK_HIEDNN(hiednnSetNormalTensorDesc( \
            yDesc, HIEDNN_TYPE, testCase.nDims, testCase.yDims)); \
        \
        CTYPE *hYPtr, *dYPtr, *yRef; \
        CHECK_CUDA(cudaMallocHost(&hYPtr, testCase.ySize * sizeof(CTYPE))); \
        CHECK_CUDA(cudaMalloc(&dYPtr, testCase.ySize * sizeof(CTYPE))); \
        CHECK_CUDA(cudaMallocHost(&yRef, testCase.ySize * sizeof(CTYPE))); \
        \
        CHECK_HIEDNN(hiednnCudaConcat( \
            handle, xDesc, reinterpret_cast<void **>(dXPtrs), \
            testCase.inputCount, testCase.axis, yDesc, dYPtr)); \
        CHECK_CUDA(cudaMemcpy(hYPtr, dYPtr, testCase.ySize * sizeof(CTYPE), \
                              cudaMemcpyDeviceToHost)); \
        \
        ConcatRef(hXPtrs, yRef, testCase); \
        for (int64_t i = 0; i < testCase.ySize; ++i) { \
            ASSERT_EQ(hYPtr[i], yRef[i]); \
        }\
        \
        for (int i = 0; i < testCase.inputCount; ++i) { \
            cudaFreeHost(hXPtrs[i]); \
            cudaFree(dXPtrs[i]); \
        }\
        cudaFreeHost(hYPtr); \
        cudaFree(dYPtr); \
        cudaFreeHost(yRef); \
    } \
    for (int i = 0; i < 128; ++i) { \
        CHECK_HIEDNN(hiednnDestroyTensorDesc(xDesc[i])); \
    } \
    CHECK_HIEDNN(hiednnDestroyTensorDesc(yDesc)); \
    CHECK_HIEDNN(hiednnDestroyCudaHandle(handle)); \
}

UTEST_CONCAT(I32_FastMap,
             fastMapCases, int32_t, HIEDNN_DATATYPE_INT32)
UTEST_CONCAT(I32_SerialSearch,
             serialSearchCases, int32_t, HIEDNN_DATATYPE_INT32)
UTEST_CONCAT(I32_HybridSearch,
             hybridSearchCases, int32_t, HIEDNN_DATATYPE_INT32)
UTEST_CONCAT(I32_SmallBatch,
             smallBatchCases, int32_t, HIEDNN_DATATYPE_INT32)


