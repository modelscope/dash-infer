/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    linear_interpolation_utest.cu
 */

#include <hiednn.h>
#include <hiednn_cuda.h>

#include <gtest/gtest.h>
#include <cstdint>
#include <cmath>
#include <vector>

#include <utest_utils.hpp>

namespace {

constexpr int MAX_DIM = 4;

struct LinearTestCase {
    int64_t xDims[MAX_DIM];
    int64_t yDims[MAX_DIM];
    int nDim;
};

int64_t GetTensorSize(const int64_t *dim, int nDim) {
    int64_t size = 1;
    for (int i = 0; i < nDim; ++i) {
        size *= dim[i];
    }
    return size;
}

void GetTensorStride(const int64_t *dim, int64_t *stride, int nDim) {
    int64_t s = 1;
    for (int i = nDim - 1; i >= 0; --i) {
        stride[i] = s;
        s *= dim[i];
    }
}

template <typename T>
void TensorInit(T *data, int64_t size) {
    for (int i = 0; i < size; ++i) {
        data[i] = static_cast<T>(i);
    }
}

const std::vector<LinearTestCase> linearTestCases = {
    // 1D interpolation
    {{1, 100}, {1, 333}, 2},
    {{1, 333}, {1, 100}, 2},
    {{333, 111}, {333, 321}, 2},
    {{333, 321}, {333, 111}, 2},

    // 2D interpolation
    {{1, 100, 333}, {1, 333, 100}, 3},
    {{1, 222, 333}, {1, 123, 77}, 3},
    {{1, 123, 77}, {1, 222, 333}, 3},
    {{123, 111, 321}, {123, 321, 111}, 3},
    {{123, 333, 321}, {123, 111, 111}, 3},
    {{123, 111, 111}, {123, 333, 321}, 3},

    // 3D interpolation
    {{1, 100, 222, 333}, {1, 123, 345, 456}, 4},
    {{1, 123, 345, 456}, {1, 100, 222, 333}, 4},
    {{123, 33, 44, 55}, {123, 43, 54, 65}, 4},
    {{123, 43, 54, 65}, {123, 33, 44, 55}, 4},
};

template <typename T>
struct HalfPixel {
    T scale;

    HalfPixel() = default;

    HalfPixel(T xLength, T yLength) {
        scale = yLength / xLength;
    }

    T Map(const T &yCoord) {
        return (yCoord + 0.5) / scale - 0.5;
    }
};

template <typename T>
struct PytorchHalfPixel {
    T scaleRcp;

    PytorchHalfPixel() = default;

    PytorchHalfPixel(T xLength, T yLength) {
        scaleRcp = yLength > 1 ? xLength / yLength : 0;
    }

    T Map(const T &yCoord) {
        return (yCoord + 0.5) * scaleRcp - 0.5;
    }
};

template <typename T>
struct AlignCorners {
    T xLength;
    T yLength;

    AlignCorners() = default;

    AlignCorners(T xLength, T yLength) {
        this->xLength = xLength;
        this->yLength = yLength;
    }

    T Map(const T &yCoord) {
        return yCoord * (xLength - 1) / (yLength - 1);
    }
};

template <typename T>
struct Asymmetric {
    T scale;

    Asymmetric() = default;

    Asymmetric(T xLength, T yLength) {
        scale = yLength / xLength;
    }

    T Map(const T &yCoord) {
        return yCoord / scale;
    }
};

template <int INTERP_DIM>
float GetLinearOutput(
        const float *x,
        const int64_t *xStrides,
        const int *xCoord,
        int batchId,
        const float *xCoordOffset);

template <>
float GetLinearOutput<1>(
        const float *x,
        const int64_t *xStrides,
        const int *xCoord,
        int batchId,
        const float *xCoordOffset) {
    const float *xptr = x + batchId * xStrides[0];

    float x0 = xptr[xCoord[0]];
    float x1 = xptr[xCoord[1]];

    float ret = x0 + (x1 - x0) * xCoordOffset[0];
    return ret;
}

template <>
float GetLinearOutput<2>(
        const float *x,
        const int64_t *xStrides,
        const int *xCoord,
        int batchId,
        const float *xCoordOffset) {
    const float *xptr = x + batchId * xStrides[0];

    int xOffset00 = xCoord[0] * xStrides[1] + xCoord[1];
    int xOffset01 = xCoord[0] * xStrides[1] + xCoord[3];
    int xOffset10 = xCoord[2] * xStrides[1] + xCoord[1];
    int xOffset11 = xCoord[2] * xStrides[1] + xCoord[3];

    float x00 = xptr[xOffset00];
    float x01 = xptr[xOffset01];
    float x10 = xptr[xOffset10];
    float x11 = xptr[xOffset11];

    float x0 = x00 + (x01 - x00) * xCoordOffset[1];
    float x1 = x10 + (x11 - x10) * xCoordOffset[1];

    float ret = x0 + (x1 - x0) * xCoordOffset[0];
    return ret;
}

template <>
float GetLinearOutput<3>(
        const float *x,
        const int64_t *xStrides,
        const int *xCoord,
        int batchId,
        const float *xCoordOffset) {
    const float *xptr = x + batchId * xStrides[0];

    int xOffset000 =
        xCoord[0] * xStrides[1] + xCoord[1] * xStrides[2] + xCoord[2];
    int xOffset001 =
        xCoord[0] * xStrides[1] + xCoord[1] * xStrides[2] + xCoord[5];
    int xOffset010 =
        xCoord[0] * xStrides[1] + xCoord[4] * xStrides[2] + xCoord[2];
    int xOffset011 =
        xCoord[0] * xStrides[1] + xCoord[4] * xStrides[2] + xCoord[5];

    int xOffset100 =
        xCoord[3] * xStrides[1] + xCoord[1] * xStrides[2] + xCoord[2];
    int xOffset101 =
        xCoord[3] * xStrides[1] + xCoord[1] * xStrides[2] + xCoord[5];
    int xOffset110 =
        xCoord[3] * xStrides[1] + xCoord[4] * xStrides[2] + xCoord[2];
    int xOffset111 =
        xCoord[3] * xStrides[1] + xCoord[4] * xStrides[2] + xCoord[5];

    float x000 = xptr[xOffset000];
    float x001 = xptr[xOffset001];
    float x010 = xptr[xOffset010];
    float x011 = xptr[xOffset011];

    float x100 = xptr[xOffset100];
    float x101 = xptr[xOffset101];
    float x110 = xptr[xOffset110];
    float x111 = xptr[xOffset111];

    float x00 = x000 + (x001 - x000) * xCoordOffset[2];
    float x01 = x010 + (x011 - x010) * xCoordOffset[2];
    float x10 = x100 + (x101 - x100) * xCoordOffset[2];
    float x11 = x110 + (x111 - x110) * xCoordOffset[2];

    float x0 = x00 + (x01 - x00) * xCoordOffset[1];
    float x1 = x10 + (x11 - x10) * xCoordOffset[1];

    float ret = x0 + (x1 - x0) * xCoordOffset[0];
    return ret;
}

template <int INTERP_DIM,
          typename CoordFunc>
void CheckLinearOutput(
        const float *x, const float *y, const LinearTestCase &testCase) {
    const int64_t *xDims = testCase.xDims;
    const int64_t *yDims = testCase.yDims;
    int64_t ySize = GetTensorSize(testCase.yDims, testCase.nDim);
    int64_t xStrides[INTERP_DIM + 1];

    GetTensorStride(xDims, xStrides, INTERP_DIM + 1);

    CoordFunc coordFunc[INTERP_DIM];
    for (int i = 0; i < INTERP_DIM; ++i) {
        coordFunc[i] = CoordFunc(xDims[i + 1], yDims[i + 1]);
    }

    for (int yOffset = 0; yOffset < ySize; ++yOffset) {
        int yCoord[INTERP_DIM + 1];
        int div = yOffset;
        for (int i = INTERP_DIM; i >= 0; --i) {
            yCoord[i] = div % yDims[i];
            div /= yDims[i];
        }

        float xCoordFP[INTERP_DIM];
        for (int i = 0; i < INTERP_DIM; ++i) {
            xCoordFP[i] = coordFunc[i].Map(static_cast<float>(yCoord[i + 1]));
            if (xCoordFP[i] < 0) xCoordFP[i] = 0;
        }

        float xCoordRoundFP[INTERP_DIM * 2];
        for (int i = 0; i < INTERP_DIM; ++i) {
            xCoordRoundFP[i] = std::floor(xCoordFP[i]);
            xCoordRoundFP[i + INTERP_DIM] = std::ceil(xCoordFP[i]);
            if (xCoordRoundFP[i] >= xDims[i + 1]) {
                xCoordRoundFP[i] = xDims[i + 1] - 1;
            }
            if (xCoordRoundFP[i + INTERP_DIM] >= xDims[i + 1]) {
                xCoordRoundFP[i + INTERP_DIM] = xDims[i + 1] - 1;
            }
        }

        int xCoord[INTERP_DIM * 2];
        for (int i = 0; i < INTERP_DIM * 2; ++i) {
            xCoord[i] = static_cast<int>(xCoordRoundFP[i]);
        }

        float xCoordOffset[INTERP_DIM];
        for (int i = 0; i < INTERP_DIM; ++i) {
            xCoordOffset[i] = xCoordFP[i] - xCoordRoundFP[i];
        }

        float yStd = GetLinearOutput<INTERP_DIM>(
            x, xStrides, xCoord, yCoord[0], xCoordOffset);
        CheckEq(yStd, y[yOffset]);
    }
}

}  // anonymous namespace

#define LINEAR_TEST(TEST_NAME, HIE_COORD_MODE, COORD_FUNC) \
TEST(LinearInterpolation, TEST_NAME) { \
    hiednnCudaHandle_t handle; \
    CHECK_HIEDNN(hiednnCreateCudaHandle(&handle)); \
    \
    hiednnTensorDesc_t xDesc; \
    hiednnTensorDesc_t yDesc; \
    CHECK_HIEDNN(hiednnCreateTensorDesc(&xDesc)); \
    CHECK_HIEDNN(hiednnCreateTensorDesc(&yDesc)); \
    float scale[3]; \
    \
    for (int caseIt = 0; caseIt < linearTestCases.size(); ++caseIt) { \
        const auto &tc = linearTestCases[caseIt]; \
        \
        int64_t xSize = GetTensorSize(tc.xDims, tc.nDim); \
        int64_t ySize = GetTensorSize(tc.yDims, tc.nDim); \
        float *d_x, *d_y; \
        float *h_x, *h_y; \
        CHECK_CUDA(cudaMallocHost(&h_x, xSize * sizeof(float))); \
        CHECK_CUDA(cudaMallocHost(&h_y, ySize * sizeof(float))); \
        TensorInit(h_x, xSize); \
        \
        for (int i = 0; i < tc.nDim - 1; ++i) { \
            scale[i] = static_cast<float>(tc.yDims[i + 1]) / \
                       static_cast<float>(tc.xDims[i + 1]); \
        } \
        \
        CHECK_CUDA(cudaMalloc(&d_x, xSize * sizeof(float))); \
        CHECK_CUDA(cudaMalloc(&d_y, ySize * sizeof(float))); \
        CHECK_CUDA(cudaMemcpy(d_x, h_x, xSize * sizeof(float), \
                              cudaMemcpyHostToDevice)); \
        \
        CHECK_HIEDNN(hiednnSetNormalTensorDesc( \
            xDesc, HIEDNN_DATATYPE_FP32, tc.nDim, tc.xDims)); \
        CHECK_HIEDNN(hiednnSetNormalTensorDesc( \
            yDesc, HIEDNN_DATATYPE_FP32, tc.nDim, tc.yDims)); \
        \
        CHECK_HIEDNN(hiednnCudaLinearInterpolation( \
            handle, HIE_COORD_MODE, xDesc, d_x, \
            scale, tc.nDim - 1, yDesc, d_y)); \
        \
        CHECK_CUDA(cudaMemcpy(h_y, d_y, ySize * sizeof(float), \
                              cudaMemcpyDeviceToHost)); \
        \
        switch (tc.nDim - 1) { \
            case 1: \
                CheckLinearOutput<1, COORD_FUNC<float>>(h_x, h_y, tc); \
                break; \
            case 2: \
                CheckLinearOutput<2, COORD_FUNC<float>>(h_x, h_y, tc); \
                break; \
            case 3: \
                CheckLinearOutput<3, COORD_FUNC<float>>(h_x, h_y, tc); \
                break; \
            default: \
                break; \
        } \
        \
        CHECK_CUDA(cudaFree(d_x)); \
        CHECK_CUDA(cudaFree(d_y)); \
        CHECK_CUDA(cudaFreeHost(h_x)); \
        CHECK_CUDA(cudaFreeHost(h_y)); \
    } \
    CHECK_HIEDNN(hiednnDestroyTensorDesc(xDesc)); \
    CHECK_HIEDNN(hiednnDestroyTensorDesc(yDesc)); \
    CHECK_HIEDNN(hiednnDestroyCudaHandle(handle)); \
}

LINEAR_TEST(half_pixel, HIEDNN_INTERP_COORD_HALF_PIXEL, HalfPixel)
LINEAR_TEST(pytorch_half_pixel, HIEDNN_INTERP_COORD_PYTORCH_HALF_PIXEL,
            PytorchHalfPixel)
LINEAR_TEST(align_corners, HIEDNN_INTERP_COORD_ALIGN_CORNER, AlignCorners)
LINEAR_TEST(asymmetric, HIEDNN_INTERP_COORD_ASYMMETRIC, Asymmetric)

