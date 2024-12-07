/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    where_utest.cu
 */
#include <hiednn.h>
#include <hiednn_cuda.h>

#include <gtest/gtest.h>
#include <cstdint>
#include <algorithm>

#include <utest_utils.hpp>

namespace {

const int MAX_DIM = 5;

struct WhereTestCase {
    int64_t xDims[MAX_DIM];
    int xNDim;
    int64_t yDims[MAX_DIM];
    int yNDim;
    int64_t condDims[MAX_DIM];
    int condNDim;
    int64_t zDims[MAX_DIM];
    int zNDim;
};

const std::vector<WhereTestCase> testCases = {
    {{667}, 1, {667}, 1, {667}, 1, {667}, 1},
    {{5, 11, 1}, 3, {3, 5, 1, 33}, 4, {3, 1, 11, 33}, 4, {3, 5, 11, 33}, 4},
    {{3, 5, 11, 33}, 4, {5, 11, 1}, 3, {3, 1, 11, 33}, 4, {3, 5, 11, 33}, 4},
};

int64_t TensorSize(const int64_t *dim, int nDims) {
    int64_t size = 1;
    for (int i = 0; i < nDims; ++i) {
        size *= dim[i];
    }
    return size;
}

template <typename DT, typename ST>
void TensorStride(const DT *dim, ST *stride, int nDim) {
    ST s = 1;
    for (int i = 0; i < nDim; ++i) {
        stride[i] = s;
        s *= dim[i];
    }
}

template <typename T>
void TensorInit(T *data, int64_t size, T start) {
    for (int64_t i = 0; i < size; ++i) {
        data[i] = start + i;
    }
}

void CondInit(char *data, int64_t size) {
    for (int64_t i = 0; i < size; ++i) {
        data[i] = i % 2;
    }
}

template <typename T>
void GetZRef(const T *x,
             const T *y,
             const char *cond,
             T *zRef,
             int64_t zSize,
             const int64_t *xDims,
             int xNDim,
             const int64_t *yDims,
             int yNDim,
             const int64_t *condDims,
             int condNDim,
             const int64_t *zDims,
             int zNDim) {
    // reverse dims
    int xd[MAX_DIM];
    int yd[MAX_DIM];
    int cd[MAX_DIM];
    int zd[MAX_DIM];

    for (int i = 0; i < xNDim; ++i) {
        xd[i] = xDims[xNDim - 1 - i];
    }
    for (int i = 0; i < yNDim; ++i) {
        yd[i] = yDims[yNDim - 1 - i];
    }
    for (int i = 0; i < condNDim; ++i) {
        cd[i] = condDims[condNDim - 1 - i];
    }
    for (int i = 0; i < zNDim; ++i) {
        zd[i] = zDims[zNDim - 1 - i];
    }

    // stride
    int xs[MAX_DIM];
    int ys[MAX_DIM];
    int cs[MAX_DIM];
    TensorStride(xd, xs, xNDim);
    TensorStride(yd, ys, yNDim);
    TensorStride(cd, cs, condNDim);

    for (int i = 0; i < zNDim; ++i) {
        if (i >= xNDim || xd[i] == 1) xs[i] = 0;
        if (i >= yNDim || yd[i] == 1) ys[i] = 0;
        if (i >= condNDim || cd[i] == 1) cs[i] = 0;
    }

    for (int zIt = 0; zIt < zSize; ++zIt) {
        int zIdx[MAX_DIM];
        int div = zIt;
        for (int i = 0; i < zNDim; ++i) {
            zIdx[i] = div % zd[i];
            div = div / zd[i];
        }

        int condOffset = 0;
        for (int i = 0; i < condNDim; ++i) {
            condOffset += zIdx[i] * cs[i];
        }

        if (cond[condOffset]) {
            int xOffset = 0;
            for (int i = 0; i < xNDim; ++i) {
                xOffset += zIdx[i] * xs[i];
            }
            zRef[zIt] = x[xOffset];
        } else {
            int yOffset = 0;
            for (int i = 0; i < yNDim; ++i) {
                yOffset += zIdx[i] * ys[i];
            }
            zRef[zIt] = y[yOffset];
        }
    }
}

void RunTest(int caseId) {
    const auto &testCase = testCases[caseId];
    const int64_t *xDims = testCase.xDims;
    int xNDim = testCase.xNDim;
    const int64_t *yDims = testCase.yDims;
    int yNDim = testCase.yNDim;
    const int64_t *condDims = testCase.condDims;
    int condNDim = testCase.condNDim;
    const int64_t *zDims = testCase.zDims;
    int zNDim = testCase.zNDim;

    int64_t xSize = TensorSize(xDims, xNDim);
    int64_t ySize = TensorSize(yDims, yNDim);
    int64_t condSize = TensorSize(condDims, condNDim);
    int64_t zSize = TensorSize(zDims, zNDim);

    char *hcond;
    CHECK_CUDA(cudaMallocHost(&hcond, condSize * sizeof(char)));

    int *hx, *hy, *hz, *zRef;
    CHECK_CUDA(cudaMallocHost(&hx, xSize * sizeof(int)));
    CHECK_CUDA(cudaMallocHost(&hy, ySize * sizeof(int)));
    CHECK_CUDA(cudaMallocHost(&hz, zSize * sizeof(int)));
    CHECK_CUDA(cudaMallocHost(&zRef, zSize * sizeof(int)));

    TensorInit(hx, xSize, 0);
    TensorInit(hy, ySize, static_cast<int>(xSize));
    CondInit(hcond, condSize);

    char *dcond;
    CHECK_CUDA(cudaMalloc(&dcond, condSize * sizeof(char)));

    int *dx, *dy, *dz;
    CHECK_CUDA(cudaMalloc(&dx, xSize * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&dy, ySize * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&dz, zSize * sizeof(int)));

    CHECK_CUDA(cudaMemcpy(dx, hx, xSize * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dy, hy, ySize * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dcond, hcond, condSize * sizeof(char),
                          cudaMemcpyHostToDevice));

    hiednnCudaHandle_t handle;
    CHECK_HIEDNN(hiednnCreateCudaHandle(&handle));

    hiednnTensorDesc_t xDesc, yDesc, condDesc, zDesc;
    CHECK_HIEDNN(hiednnCreateTensorDesc(&xDesc));
    CHECK_HIEDNN(hiednnCreateTensorDesc(&yDesc));
    CHECK_HIEDNN(hiednnCreateTensorDesc(&condDesc));
    CHECK_HIEDNN(hiednnCreateTensorDesc(&zDesc));

    CHECK_HIEDNN(hiednnSetNormalTensorDesc(
        xDesc, HIEDNN_DATATYPE_INT32, xNDim, xDims));
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(
        yDesc, HIEDNN_DATATYPE_INT32, yNDim, yDims));
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(
        condDesc, HIEDNN_DATATYPE_BOOL, condNDim, condDims));
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(
        zDesc, HIEDNN_DATATYPE_INT32, zNDim, zDims));

    CHECK_HIEDNN(hiednnCudaWhere(
        handle, xDesc, dx, yDesc, dy, condDesc, dcond, zDesc, dz));

    CHECK_CUDA(cudaMemcpy(hz, dz, zSize * sizeof(int), cudaMemcpyDeviceToHost));

    GetZRef(hx, hy, hcond, zRef, zSize, xDims, xNDim, yDims, yNDim,
            condDims, condNDim, zDims, zNDim);

    for (int64_t i = 0; i < zSize; ++i) {
        ASSERT_EQ(hz[i], zRef[i]);
    }

    CHECK_CUDA(cudaFree(dx));
    CHECK_CUDA(cudaFree(dy));
    CHECK_CUDA(cudaFree(dcond));
    CHECK_CUDA(cudaFree(dz));
    CHECK_CUDA(cudaFreeHost(hx));
    CHECK_CUDA(cudaFreeHost(hy));
    CHECK_CUDA(cudaFreeHost(hcond));
    CHECK_CUDA(cudaFreeHost(hz));
    CHECK_CUDA(cudaFreeHost(zRef));
}

}  // anonymous namespace

TEST(Where_CUDA, Case0) {
    RunTest(0);
}

TEST(Where_CUDA, Case1) {
    RunTest(1);
}

TEST(Where_CUDA, Case2) {
    RunTest(2);
}


