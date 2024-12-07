/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    expand_utest.cpp
 */
#include <hiednn.h>
#include <hiednn_cpp.h>

#include <gtest/gtest.h>
#include <cstdint>
#include <vector>

#include <utest_utils.hpp>

namespace {

const int MAX_DIM = 4;

struct ExpandTestCase {
    int64_t xDims[MAX_DIM];
    int64_t yDims[MAX_DIM];
    int xNDim;
    int yNDim;
    size_t xSize;
    size_t ySize;
};

const std::vector<ExpandTestCase> testCases = {
    {{1, 1, 41, 28}, {1, 22, 41, 28}, 2, 3, 1148, 25256},
    {{1, 1, 41, 28}, {1, 22, 41, 28}, 2, 4, 1148, 25256},
    {{1, 1, 41, 28}, {1, 22, 41, 28}, 3, 4, 1148, 25256},
    {{4, 1, 17, 28}, {4, 3, 17, 28}, 4, 4, 1904, 5712},
};

}  // anonymous namespace

#define UTEST_EXPAND(TEST_NAME, DATATYPE, HIEDNN_DATA_TYPE) \
TEST(Expand_CPP, TEST_NAME) { \
    hiednnCppHandle_t handle; \
    CHECK_HIEDNN(hiednnCreateCppHandle( \
        &handle, HIEDNN_ASM_OPTIMIZE_NONE)); \
    hiednnTensorDesc_t xDesc, yDesc; \
    CHECK_HIEDNN(hiednnCreateTensorDesc(&xDesc)); \
    CHECK_HIEDNN(hiednnCreateTensorDesc(&yDesc)); \
    for (const auto &testCase : testCases) { \
        std::vector<DATATYPE> x(testCase.xSize); \
        std::vector<DATATYPE> y(testCase.ySize); \
        for (size_t i = 0; i < x.size(); ++i) { \
            x[i] = i; \
        } \
        \
        const int64_t *x_dims = testCase.xDims + (MAX_DIM - testCase.xNDim); \
        const int64_t *y_dims = testCase.yDims + (MAX_DIM - testCase.yNDim); \
        CHECK_HIEDNN(hiednnSetNormalTensorDesc( \
            xDesc, HIEDNN_DATA_TYPE, testCase.xNDim, x_dims)); \
        CHECK_HIEDNN(hiednnSetNormalTensorDesc( \
            yDesc, HIEDNN_DATA_TYPE, testCase.yNDim, y_dims)); \
        \
        CHECK_HIEDNN(hiednnCppExpand( \
            handle, xDesc, x.data(), yDesc, y.data())); \
        \
        std::vector<size_t> xStrides(testCase.xNDim); \
        xStrides[0] = 1; \
        for (int i = 1; i < testCase.xNDim; ++i) { \
            xStrides[i] = xStrides[i - 1] * x_dims[testCase.xNDim - i]; \
        } \
        for (size_t yOffset = 0; yOffset < y.size(); ++yOffset) { \
            size_t xOffset = 0; \
            size_t offset = yOffset; \
            for (int i = 0; i < testCase.xNDim; ++i) { \
                int64_t idx = offset % y_dims[testCase.yNDim - 1 - i]; \
                offset /= y_dims[testCase.yNDim - 1 - i]; \
                if (idx < x_dims[testCase.xNDim - 1 - i]) { \
                    xOffset += idx * xStrides[i]; \
                } \
            } \
            ASSERT_EQ(y[yOffset], x[xOffset]); \
        } \
    } \
    CHECK_HIEDNN(hiednnDestroyTensorDesc(xDesc)); \
    CHECK_HIEDNN(hiednnDestroyTensorDesc(yDesc)); \
    CHECK_HIEDNN(hiednnDestroyCppHandle(handle)); \
}

UTEST_EXPAND(F32, float, HIEDNN_DATATYPE_FP32);
UTEST_EXPAND(INT16, int16_t, HIEDNN_DATATYPE_INT16);
UTEST_EXPAND(UINT8, uint8_t, HIEDNN_DATATYPE_UINT8);


