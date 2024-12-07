/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    reduce_utest.cu
 */
#include <hiednn.h>
#include <hiednn_cuda.h>

#include <gtest/gtest.h>
#include <cstdlib>
#include <cmath>

#include <utest_utils.hpp>

namespace {

void Rand(float *data, size_t n) {
    unsigned int seed = 0;
    for (int i = 0; i < n; ++i) {
        data[i] = static_cast<float>(rand_r(&seed)) /
                  static_cast<float>(RAND_MAX);
    }
}

void Init(float *data, size_t n) {
    for (int i = 0; i < n; ++i) {
        data[i] = static_cast<float>(i);
    }

    unsigned int seed = 0;
    for (int i = 0; i < n; ++i) {
        int idx = rand_r(&seed) % n;
        float tmp = data[idx];
        data[idx] = data[i];
        data[i] = tmp;
    }
}

struct TestCase {
    int d0, d1, d2;
};

TestCase rowCase[] = {
    {1, 111, 35},   // single block
    {1, 5, 1111},   // single block, block reduce
    {1, 5, 11111},  // multi block
};

TestCase columnCase[] = {
    {33, 55, 77},    // single block, block reduce
    {33, 11, 55},    // single block, warp reduce, block reduce
    {33, 1111, 77},  // multi block, block reduce
    {33, 1111, 55},  // multi block, warp reduce, block reduce
};

void LogSumExp(const float *x, float *y, uint32_t *indices,
               float alpha, TestCase tc, int axis) {
    // axis == 1 or 2
    if (axis == 0 || axis > 2) {
        return;
    }

    int innerLoop, outerLoop, reduceLoop;
    int innerStride, outerStride, reduceStride;

    if (axis == 1) {
        innerLoop = tc.d2;
        outerLoop = tc.d0;
        reduceLoop = tc.d1;
        innerStride = 1;
        outerStride = tc.d1 * tc.d2;
        reduceStride = tc.d2;
    } else if (axis == 2) {
        innerLoop = tc.d1;
        outerLoop = tc.d0;
        reduceLoop = tc.d2;
        innerStride = tc.d2;
        outerStride = tc.d1 * tc.d2;
        reduceStride = 1;
    }

    float *yPtr = y;

    for (int i = 0; i < outerLoop; ++i) {
        for (int j = 0; j < innerLoop; ++j) {
            const float *xPtr = x + i * outerStride + j * innerStride;
            double acc = 0;

            for (int k = 0; k < reduceLoop; ++k) {
                double v = static_cast<double>(xPtr[k * reduceStride]);
                acc += std::exp(v);
            }

            *yPtr = static_cast<float>(std::log(acc)) * alpha;
            ++yPtr;
        }
    }
}

void Max(const float *x, float *y, uint32_t *indices,
         float alpha, TestCase tc, int axis) {
    // axis == 1 or 2
    if (axis == 0 || axis > 2) {
        return;
    }

    int innerLoop, outerLoop, reduceLoop;
    int innerStride, outerStride, reduceStride;

    if (axis == 1) {
        innerLoop = tc.d2;
        outerLoop = tc.d0;
        reduceLoop = tc.d1;
        innerStride = 1;
        outerStride = tc.d1 * tc.d2;
        reduceStride = tc.d2;
    } else if (axis == 2) {
        innerLoop = tc.d1;
        outerLoop = tc.d0;
        reduceLoop = tc.d2;
        innerStride = tc.d2;
        outerStride = tc.d1 * tc.d2;
        reduceStride = 1;
    }

    float *yPtr = y;
    uint32_t *idxPtr = indices;

    for (int i = 0; i < outerLoop; ++i) {
        for (int j = 0; j < innerLoop; ++j) {
            const float *xPtr = x + i * outerStride + j * innerStride;
            float max = -INFINITY;
            uint32_t maxIdx = 0;

            for (int k = 0; k < reduceLoop; ++k) {
                float v = xPtr[k * reduceStride];
                if (v > max) {
                    max = v;
                    maxIdx = k;
                }
            }

            *yPtr = max * alpha;
            *idxPtr = maxIdx;
            ++yPtr;
            ++idxPtr;
        }
    }
}

}  // anonymous namespace

#define UTEST_ROW_REDUCE(TEST_NAME, HIE_OP, REF_OP, INDICES, INIT_FUNC) \
TEST(CudaRowReduce, TEST_NAME) { \
    hiednnCudaHandle_t handle; \
    CHECK_HIEDNN(hiednnCreateCudaHandle(&handle)); \
    \
    hiednnTensorDesc_t xDesc, yDesc; \
    CHECK_HIEDNN(hiednnCreateTensorDesc(&xDesc)); \
    CHECK_HIEDNN(hiednnCreateTensorDesc(&yDesc)); \
    \
    for (const auto &tc : rowCase) { \
        int xSize = tc.d0 * tc.d1 * tc.d2; \
        int ySize = tc.d0 * tc.d1; \
        float *hx, *hy, *hyRef; \
        CHECK_CUDA(cudaMallocHost(&hx, xSize * sizeof(float))); \
        CHECK_CUDA(cudaMallocHost(&hy, ySize * sizeof(float))); \
        CHECK_CUDA(cudaMallocHost(&hyRef, ySize * sizeof(float))); \
        uint32_t *hIndices, *hIndicesRef; \
        CHECK_CUDA(cudaMallocHost(&hIndices, ySize * sizeof(uint32_t))); \
        CHECK_CUDA(cudaMallocHost(&hIndicesRef, ySize * sizeof(uint32_t))); \
        \
        INIT_FUNC(hx, xSize); \
        \
        float *dx, *dy; \
        CHECK_CUDA(cudaMalloc(&dx, xSize * sizeof(float))); \
        CHECK_CUDA(cudaMalloc(&dy, ySize * sizeof(float))); \
        uint32_t *dIndices; \
        CHECK_CUDA(cudaMalloc(&dIndices, ySize * sizeof(uint32_t))); \
        \
        CHECK_CUDA(cudaMemcpy(dx, hx, xSize * sizeof(float), \
                              cudaMemcpyHostToDevice)); \
        \
        int64_t xDims[] = {tc.d0, tc.d1, tc.d2}; \
        int64_t yDims[] = {tc.d0, tc.d1, 1}; \
        CHECK_HIEDNN(hiednnSetNormalTensorDesc( \
            xDesc, HIEDNN_DATATYPE_FP32, 3, xDims)); \
        CHECK_HIEDNN(hiednnSetNormalTensorDesc( \
            yDesc, HIEDNN_DATATYPE_FP32, 3, yDims)); \
        \
        float alpha = 1.33f; \
        CHECK_HIEDNN(hiednnCudaReduce( \
            handle, HIE_OP, &alpha, xDesc, dx, 2, \
            yDesc, dy, HIEDNN_DATATYPE_UINT32, dIndices)); \
        \
        CHECK_CUDA(cudaMemcpy(hy, dy, ySize * sizeof(float), \
                              cudaMemcpyDeviceToHost)); \
        CHECK_CUDA(cudaMemcpy(hIndices, dIndices, ySize * sizeof(uint32_t), \
                              cudaMemcpyDeviceToHost)); \
        \
        REF_OP(hx, hyRef, hIndicesRef, alpha, tc, 2); \
        \
        for (int i = 0; i < ySize; ++i) { \
            CheckEq(hy[i], hyRef[i]); \
        } \
        \
        if (INDICES) { \
            for (int i = 0; i < ySize; ++i) { \
                CheckEq(hIndices[i], hIndicesRef[i]); \
            } \
        } \
        \
        CHECK_CUDA(cudaFree(dx)); \
        CHECK_CUDA(cudaFree(dy)); \
        CHECK_CUDA(cudaFree(dIndices)); \
        CHECK_CUDA(cudaFreeHost(hx)); \
        CHECK_CUDA(cudaFreeHost(hy)); \
        CHECK_CUDA(cudaFreeHost(hyRef)); \
        CHECK_CUDA(cudaFreeHost(hIndices)); \
        CHECK_CUDA(cudaFreeHost(hIndicesRef)); \
    } \
}

UTEST_ROW_REDUCE(LogSumExp, HIEDNN_REDUCE_LOG_SUM_EXP, LogSumExp, 0, Rand)
UTEST_ROW_REDUCE(Max, HIEDNN_REDUCE_MAX, Max, 0, Rand)
UTEST_ROW_REDUCE(MaxWithIndex, HIEDNN_REDUCE_MAX, Max, 1, Init)

#define UTEST_COLUMN_REDUCE(TEST_NAME, HIE_OP, REF_OP, INDICES, INIT_FUNC) \
TEST(CudaColumnReduce, TEST_NAME) { \
    hiednnCudaHandle_t handle; \
    CHECK_HIEDNN(hiednnCreateCudaHandle(&handle)); \
    \
    hiednnTensorDesc_t xDesc, yDesc; \
    CHECK_HIEDNN(hiednnCreateTensorDesc(&xDesc)); \
    CHECK_HIEDNN(hiednnCreateTensorDesc(&yDesc)); \
    \
    for (const auto &tc : columnCase) { \
        int xSize = tc.d0 * tc.d1 * tc.d2; \
        int ySize = tc.d0 * tc.d2; \
        float *hx, *hy, *hyRef; \
        CHECK_CUDA(cudaMallocHost(&hx, xSize * sizeof(float))); \
        CHECK_CUDA(cudaMallocHost(&hy, ySize * sizeof(float))); \
        CHECK_CUDA(cudaMallocHost(&hyRef, ySize * sizeof(float))); \
        uint32_t *hIndices, *hIndicesRef; \
        CHECK_CUDA(cudaMallocHost(&hIndices, ySize * sizeof(uint32_t))); \
        CHECK_CUDA(cudaMallocHost(&hIndicesRef, ySize * sizeof(uint32_t))); \
        \
        INIT_FUNC(hx, xSize); \
        \
        float *dx, *dy; \
        CHECK_CUDA(cudaMalloc(&dx, xSize * sizeof(float))); \
        CHECK_CUDA(cudaMalloc(&dy, ySize * sizeof(float))); \
        uint32_t *dIndices; \
        CHECK_CUDA(cudaMalloc(&dIndices, ySize * sizeof(uint32_t))); \
        \
        CHECK_CUDA(cudaMemcpy(dx, hx, xSize * sizeof(float), \
                              cudaMemcpyHostToDevice)); \
        \
        int64_t xDims[] = {tc.d0, tc.d1, tc.d2}; \
        int64_t yDims[] = {tc.d0, 1, tc.d2}; \
        CHECK_HIEDNN(hiednnSetNormalTensorDesc( \
            xDesc, HIEDNN_DATATYPE_FP32, 3, xDims)); \
        CHECK_HIEDNN(hiednnSetNormalTensorDesc( \
            yDesc, HIEDNN_DATATYPE_FP32, 3, yDims)); \
        \
        float alpha = 1.33f; \
        CHECK_HIEDNN(hiednnCudaReduce( \
            handle, HIE_OP, &alpha, xDesc, dx, 1, \
            yDesc, dy, HIEDNN_DATATYPE_UINT32, dIndices)); \
        \
        CHECK_CUDA(cudaMemcpy(hy, dy, ySize * sizeof(float), \
                              cudaMemcpyDeviceToHost)); \
        CHECK_CUDA(cudaMemcpy(hIndices, dIndices, ySize * sizeof(uint32_t), \
                              cudaMemcpyDeviceToHost)); \
        \
        REF_OP(hx, hyRef, hIndicesRef, alpha, tc, 1); \
        \
        for (int i = 0; i < ySize; ++i) { \
            CheckEq(hy[i], hyRef[i]); \
        } \
        \
        if (INDICES) { \
            for (int i = 0; i < ySize; ++i) { \
                CheckEq(hIndices[i], hIndicesRef[i]); \
            } \
        } \
        \
        CHECK_CUDA(cudaFree(dx)); \
        CHECK_CUDA(cudaFree(dy)); \
        CHECK_CUDA(cudaFree(dIndices)); \
        CHECK_CUDA(cudaFreeHost(hx)); \
        CHECK_CUDA(cudaFreeHost(hy)); \
        CHECK_CUDA(cudaFreeHost(hyRef)); \
        CHECK_CUDA(cudaFreeHost(hIndices)); \
        CHECK_CUDA(cudaFreeHost(hIndicesRef)); \
    } \
}

UTEST_COLUMN_REDUCE(LogSumExp, HIEDNN_REDUCE_LOG_SUM_EXP, LogSumExp, 0, Rand)
UTEST_COLUMN_REDUCE(Max, HIEDNN_REDUCE_MAX, Max, 0, Rand)
UTEST_COLUMN_REDUCE(MaxWithIndex, HIEDNN_REDUCE_MAX, Max, 1, Init)

TEST(CudaReduceIndex, Row) {
    hiednnCudaHandle_t handle;
    CHECK_HIEDNN(hiednnCreateCudaHandle(&handle));

    hiednnTensorDesc_t xDesc, idxDesc;
    CHECK_HIEDNN(hiednnCreateTensorDesc(&xDesc));
    CHECK_HIEDNN(hiednnCreateTensorDesc(&idxDesc));

    for (const auto &tc : rowCase) {
        int xSize = tc.d0 * tc.d1 * tc.d2;
        int ySize = tc.d0 * tc.d1;
        float *hx, *hyRef;
        CHECK_CUDA(cudaMallocHost(&hx, xSize * sizeof(float)));
        CHECK_CUDA(cudaMallocHost(&hyRef, ySize * sizeof(float)));
        uint32_t *hIndices, *hIndicesRef;
        CHECK_CUDA(cudaMallocHost(&hIndices, ySize * sizeof(uint32_t)));
        CHECK_CUDA(cudaMallocHost(&hIndicesRef, ySize * sizeof(uint32_t)));

        Init(hx, xSize);

        float *dx;
        CHECK_CUDA(cudaMalloc(&dx, xSize * sizeof(float)));
        uint32_t *dIndices;
        CHECK_CUDA(cudaMalloc(&dIndices, ySize * sizeof(uint32_t)));

        CHECK_CUDA(cudaMemcpy(dx, hx, xSize * sizeof(float),
                              cudaMemcpyHostToDevice));

        int64_t xDims[] = {tc.d0, tc.d1, tc.d2};
        int64_t yDims[] = {tc.d0, tc.d1, 1};
        CHECK_HIEDNN(hiednnSetNormalTensorDesc(
            xDesc, HIEDNN_DATATYPE_FP32, 3, xDims));
        CHECK_HIEDNN(hiednnSetNormalTensorDesc(
            idxDesc, HIEDNN_DATATYPE_UINT32, 3, yDims));

        CHECK_HIEDNN(hiednnCudaReduceIndex(
            handle, HIEDNN_REDUCE_MAX, xDesc, dx, 2, idxDesc, dIndices));

        CHECK_CUDA(cudaMemcpy(hIndices, dIndices, ySize * sizeof(uint32_t),
                              cudaMemcpyDeviceToHost));

        float alpha = 1.f;
        Max(hx, hyRef, hIndicesRef, alpha, tc, 2);

        for (int i = 0; i < ySize; ++i) {
            CheckEq(hIndices[i], hIndicesRef[i]);
        }

        CHECK_CUDA(cudaFree(dx));
        CHECK_CUDA(cudaFree(dIndices));
        CHECK_CUDA(cudaFreeHost(hx));
        CHECK_CUDA(cudaFreeHost(hyRef));
        CHECK_CUDA(cudaFreeHost(hIndices));
        CHECK_CUDA(cudaFreeHost(hIndicesRef));
    }
}

TEST(CudaReduceIndex, Column) {
    hiednnCudaHandle_t handle;
    CHECK_HIEDNN(hiednnCreateCudaHandle(&handle));

    hiednnTensorDesc_t xDesc, idxDesc;
    CHECK_HIEDNN(hiednnCreateTensorDesc(&xDesc));
    CHECK_HIEDNN(hiednnCreateTensorDesc(&idxDesc));

    for (const auto &tc : columnCase) {
        int xSize = tc.d0 * tc.d1 * tc.d2;
        int ySize = tc.d0 * tc.d2;
        float *hx, *hyRef;
        CHECK_CUDA(cudaMallocHost(&hx, xSize * sizeof(float)));
        CHECK_CUDA(cudaMallocHost(&hyRef, ySize * sizeof(float)));
        uint32_t *hIndices, *hIndicesRef;
        CHECK_CUDA(cudaMallocHost(&hIndices, ySize * sizeof(uint32_t)));
        CHECK_CUDA(cudaMallocHost(&hIndicesRef, ySize * sizeof(uint32_t)));

        Init(hx, xSize);

        float *dx;
        CHECK_CUDA(cudaMalloc(&dx, xSize * sizeof(float)));
        uint32_t *dIndices;
        CHECK_CUDA(cudaMalloc(&dIndices, ySize * sizeof(uint32_t)));

        CHECK_CUDA(cudaMemcpy(dx, hx, xSize * sizeof(float),
                              cudaMemcpyHostToDevice));

        int64_t xDims[] = {tc.d0, tc.d1, tc.d2};
        int64_t yDims[] = {tc.d0, 1, tc.d2};
        CHECK_HIEDNN(hiednnSetNormalTensorDesc(
            xDesc, HIEDNN_DATATYPE_FP32, 3, xDims));
        CHECK_HIEDNN(hiednnSetNormalTensorDesc(
            idxDesc, HIEDNN_DATATYPE_UINT32, 3, yDims));

        CHECK_HIEDNN(hiednnCudaReduceIndex(
            handle, HIEDNN_REDUCE_MAX, xDesc, dx, 1, idxDesc, dIndices));

        CHECK_CUDA(cudaMemcpy(hIndices, dIndices, ySize * sizeof(uint32_t),
                              cudaMemcpyDeviceToHost));

        float alpha = 1.f;
        Max(hx, hyRef, hIndicesRef, alpha, tc, 1);

        for (int i = 0; i < ySize; ++i) {
            CheckEq(hIndices[i], hIndicesRef[i]);
        }

        CHECK_CUDA(cudaFree(dx));
        CHECK_CUDA(cudaFree(dIndices));
        CHECK_CUDA(cudaFreeHost(hx));
        CHECK_CUDA(cudaFreeHost(hyRef));
        CHECK_CUDA(cudaFreeHost(hIndices));
        CHECK_CUDA(cudaFreeHost(hIndicesRef));
    }
}



