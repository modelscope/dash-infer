/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    pad_utest.cu
 */
#include <hiednn.h>
#include <hiednn_cuda.h>

#include <gtest/gtest.h>
#include <cstdint>

#include <utest_utils.hpp>

namespace {

const int MAX_DIM = 4;

struct PadTestCase {
    int64_t xDims[MAX_DIM];
    int xNDim;
    int64_t pads[MAX_DIM * 2];
};

const std::vector<PadTestCase> testCases = {
    {{3, 4, 5, 6}, 4, {0, 0, 1, 2, 0, 0, 2, 3}},
    {{11, 13, 19, 21}, 4, {0, 1, 3, 5, 0, 3, 5, 7}},
};

size_t TensorSize(const int64_t *dims, int nDims) {
    size_t size = 1;
    for (int i = 0; i < nDims; ++i) {
        size *= dims[i];
    }
    return size;
}

void TensorStride(const int64_t *dims, int64_t *strides, int nDims) {
    int64_t s = 1;
    for (int i = 0; i < nDims; ++i) {
        strides[nDims - 1 - i] = s;
        s *= dims[nDims - 1 - i];
    }
}

template <typename T>
void PadConst(const T *x,
              T *y,
              T param,
              size_t ySize,
              int nDims,
              const int64_t *xDims,
              const int64_t *yDims,
              const int64_t *pads) {
    int64_t *rPadBound = static_cast<int64_t *>(
        malloc(nDims * sizeof(int64_t)));
    int64_t *xStrides = static_cast<int64_t *>(
        malloc(nDims * sizeof(int64_t)));

    for (int i = 0; i < nDims; ++i) {
        rPadBound[i] = xDims[i] + pads[i];
    }

    TensorStride(xDims, xStrides, nDims);

    for (size_t i = 0; i < ySize; ++i) {
        bool inBound = true;
        int64_t xOffset = 0;
        int64_t tmp = i;
        for (int j = nDims - 1; j >= 0; --j) {
            int64_t idx = tmp % yDims[j];
            tmp = tmp / yDims[j];
            xOffset += (idx - pads[j]) * xStrides[j];
            if (idx < pads[j] || idx >= rPadBound[j]) {
                inBound = false;
            }
        }

        y[i] = inBound ? x[xOffset] : param;
    }

    free(rPadBound);
}

template <typename T>
void PadEdge(const T *x,
             T *y,
             T param,
             size_t ySize,
             int nDims,
             const int64_t *xDims,
             const int64_t *yDims,
             const int64_t *pads) {
    int64_t *rPadBound = static_cast<int64_t *>(
        malloc(nDims * sizeof(int64_t)));
    int64_t *xStrides = static_cast<int64_t *>(
        malloc(nDims * sizeof(int64_t)));

    for (int i = 0; i < nDims; ++i) {
        rPadBound[i] = xDims[i] + pads[i];
    }

    TensorStride(xDims, xStrides, nDims);

    for (size_t i = 0; i < ySize; ++i) {
        int64_t xOffset = 0;
        int64_t tmp = i;
        for (int j = nDims - 1; j >= 0; --j) {
            int64_t idx = tmp % yDims[j];
            tmp = tmp / yDims[j];
            if (idx >= rPadBound[j]) {
                xOffset += (xDims[j] - 1) * xStrides[j];
            } else if (idx >= pads[j]) {
                xOffset += (idx - pads[j]) * xStrides[j];
            }
        }

        y[i] = x[xOffset];
    }

    free(rPadBound);
}

template <typename T>
void PadReflect(const T *x,
                T *y,
                T param,
                size_t ySize,
                int nDims,
                const int64_t *xDims,
                const int64_t *yDims,
                const int64_t *pads) {
    int64_t *rPadBound = static_cast<int64_t *>(
        malloc(nDims * sizeof(int64_t)));
    int64_t *xStrides = static_cast<int64_t *>(
        malloc(nDims * sizeof(int64_t)));

    for (int i = 0; i < nDims; ++i) {
        rPadBound[i] = xDims[i] + pads[i];
    }

    TensorStride(xDims, xStrides, nDims);

    for (size_t i = 0; i < ySize; ++i) {
        int64_t xOffset = 0;
        int64_t tmp = i;
        for (int j = nDims - 1; j >= 0; --j) {
            int64_t idx = tmp % yDims[j];
            tmp = tmp / yDims[j];
            if (idx >= rPadBound[j]) {
                xOffset += (2 * xDims[j] - idx + pads[j] - 2) * xStrides[j];
            } else if (idx < pads[j]) {
                xOffset += (pads[j] - idx) * xStrides[j];
            } else {
                xOffset += (idx - pads[j]) * xStrides[j];
            }
        }

        y[i] = x[xOffset];
    }

    free(rPadBound);
}

}  // anonymous namespace

#define UTEST_PAD(TEST_NAME, HIE_PAD_MODE, REF_FUNC) \
TEST(Pad_CUDA, TEST_NAME) { \
    hiednnCudaHandle_t handle; \
    CHECK_HIEDNN(hiednnCreateCudaHandle(&handle)); \
    hiednnTensorDesc_t xDesc, yDesc; \
    CHECK_HIEDNN(hiednnCreateTensorDesc(&xDesc)); \
    CHECK_HIEDNN(hiednnCreateTensorDesc(&yDesc)); \
    \
    for (const auto &testCase : testCases) { \
        const int64_t *xDims = testCase.xDims; \
        int xNDim = testCase.xNDim; \
        const int64_t *pads = testCase.pads; \
        int64_t yDims[MAX_DIM]; \
        \
        for (int i = 0; i < testCase.xNDim; ++i) { \
            yDims[i] = xDims[i] + pads[i] + pads[i + xNDim]; \
        } \
        \
        size_t xSize = TensorSize(xDims, xNDim); \
        size_t ySize = TensorSize(yDims, xNDim); \
        \
        int32_t *x, *y; \
        CHECK_CUDA(cudaMallocHost(&x, xSize * sizeof(int32_t))); \
        CHECK_CUDA(cudaMallocHost(&y, ySize * sizeof(int32_t))); \
        for (size_t i = 0; i < xSize; ++i) { \
            x[i] = i; \
        } \
        \
        int32_t *dx, *dy; \
        CHECK_CUDA(cudaMalloc(&dx, xSize * sizeof(int32_t))); \
        CHECK_CUDA(cudaMalloc(&dy, ySize * sizeof(int32_t))); \
        CHECK_CUDA(cudaMemcpy(dx, x, xSize * sizeof(int32_t), \
                                     cudaMemcpyHostToDevice)); \
        \
        CHECK_HIEDNN(hiednnSetNormalTensorDesc( \
            xDesc, HIEDNN_DATATYPE_INT32, xNDim, xDims)); \
        CHECK_HIEDNN(hiednnSetNormalTensorDesc( \
            yDesc, HIEDNN_DATATYPE_INT32, xNDim, yDims)); \
        \
        int32_t param = 3; \
        CHECK_HIEDNN(hiednnCudaPad(handle, HIE_PAD_MODE, xDesc, dx, \
                                   pads, xNDim * 2, &param, yDesc, dy)); \
        CHECK_CUDA(cudaMemcpy(y, dy, ySize * sizeof(int32_t), \
                                     cudaMemcpyDeviceToHost)); \
        \
        int32_t *yRef = static_cast<int32_t *>( \
            malloc(ySize * sizeof(int32_t))); \
        REF_FUNC(x, yRef, param, ySize, xNDim, xDims, yDims, pads); \
        \
        for (size_t i = 0; i < ySize; ++i) { \
            ASSERT_EQ(y[i], yRef[i]); \
        } \
        free(yRef); \
        cudaFree(dx); \
        cudaFree(dy); \
        cudaFreeHost(x); \
        cudaFreeHost(y); \
    } \
    CHECK_HIEDNN(hiednnDestroyTensorDesc(xDesc)); \
    CHECK_HIEDNN(hiednnDestroyTensorDesc(yDesc)); \
    CHECK_HIEDNN(hiednnDestroyCudaHandle(handle)); \
}

UTEST_PAD(Const, HIEDNN_PAD_CONST, PadConst)
UTEST_PAD(Edge, HIEDNN_PAD_EDGE, PadEdge)
UTEST_PAD(Reflect, HIEDNN_PAD_REFLECT, PadReflect)


