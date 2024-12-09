/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    nearest_interpolation_utest.cu
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

struct NearestTestCase {
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

const std::vector<NearestTestCase> nearestTestCases = {
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
    T scaleRcp;

    HalfPixel() = default;

    HalfPixel(T xLength, T yLength, T scale) {
        scaleRcp = T(1) / scale;
    }

    T Map(const T &yCoord) {
        return (yCoord + 0.5) * scaleRcp - 0.5;
    }
};

template <typename T>
struct PytorchHalfPixel {
    T scaleRcp;

    PytorchHalfPixel() = default;

    PytorchHalfPixel(T xLength, T yLength, T scale) {
        scaleRcp = yLength > 1 ? T(1) / scale : T(0);
    }

    T Map(const T &yCoord) {
        return (yCoord + 0.5) * scaleRcp - 0.5;
    }
};

template <typename T>
struct AlignCorners {
    T alignCornerScale;

    AlignCorners() = default;

    AlignCorners(T xLength, T yLength, T scale) {
        alignCornerScale = yLength > 1 ?
                           (xLength - 1) / (yLength - 1) : 0;
    }

    T Map(const T &yCoord) {
        return yCoord * alignCornerScale;
    }
};

template <typename T>
struct Asymmetric {
    T scaleRcp;

    Asymmetric() = default;

    Asymmetric(T xLength, T yLength, T scale) {
        scaleRcp = T(1) / scale;
    }

    T Map(const T &yCoord) {
        return yCoord * scaleRcp;
    }
};

template <typename T>
struct RoundHalfDown {
    static T Round(const T &x) {
        return std::ceil(static_cast<float>(x) - 0.5);
    }
};

template <typename T>
struct RoundHalfUp {
    static T Round(const T &x) {
        return std::floor(static_cast<float>(x) + 0.5);
    }
};

template <typename T>
struct RoundFloor {
    static T Round(const T &x) {
        return std::floor(static_cast<float>(x));
    }
};

template <typename T>
struct RoundCeil {
    static T Round(const T &x) {
        return std::ceil(static_cast<float>(x));
    }
};

template <int INTERP_DIM,
          typename CoordFunc,
          typename RoundFunc>
void CheckNearestOutput(
        const int *x,
        const int *y,
        const float *scale,
        const NearestTestCase &testCase) {
    const int64_t *xDims = testCase.xDims;
    const int64_t *yDims = testCase.yDims;
    int64_t ySize = GetTensorSize(testCase.yDims, testCase.nDim);
    int64_t xStrides[INTERP_DIM + 1];

    GetTensorStride(xDims, xStrides, INTERP_DIM + 1);

    CoordFunc coordFunc[INTERP_DIM];
    for (int i = 0; i < INTERP_DIM; ++i) {
        coordFunc[i] = CoordFunc(xDims[i + 1], yDims[i + 1], scale[i]);
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
            xCoordFP[i] = RoundFunc::Round(xCoordFP[i]);
            if (xCoordFP[i] < 0) xCoordFP[i] = 0;
        }

        int xCoord[INTERP_DIM];
        for (int i = 0; i < INTERP_DIM; ++i) {
            xCoord[i] = static_cast<int>(xCoordFP[i]);
            if (xCoord[i] >= xDims[i + 1]) {
                xCoord[i] = xDims[i + 1] - 1;
            }
        }

        int xOffset = yCoord[0] * xStrides[0];
        for (int i = 0; i < INTERP_DIM; ++i) {
            xOffset += xCoord[i] * xStrides[i + 1];
        }

        int yStd = x[xOffset];
        CheckEq(yStd, y[yOffset]);
    }
}

}  // anonymous namespace

#define NEAREST_TEST(TEST_NAME, \
                     HIE_COORD_MODE, \
                     HIE_NEAREST_MODE, \
                     COORD_FUNC, \
                     NEAREST_FUNC) \
TEST(NearestInterpolation, TEST_NAME) { \
    hiednnCudaHandle_t handle; \
    CHECK_HIEDNN(hiednnCreateCudaHandle(&handle)); \
    \
    hiednnTensorDesc_t xDesc; \
    hiednnTensorDesc_t yDesc; \
    CHECK_HIEDNN(hiednnCreateTensorDesc(&xDesc)); \
    CHECK_HIEDNN(hiednnCreateTensorDesc(&yDesc)); \
    float scale[3]; \
    \
    for (int caseIt = 0; caseIt < nearestTestCases.size(); ++caseIt) { \
        const auto &tc = nearestTestCases[caseIt]; \
        \
        int64_t xSize = GetTensorSize(tc.xDims, tc.nDim); \
        int64_t ySize = GetTensorSize(tc.yDims, tc.nDim); \
        int *d_x, *d_y; \
        int *h_x, *h_y; \
        CHECK_CUDA(cudaMallocHost(&h_x, xSize * sizeof(int))); \
        CHECK_CUDA(cudaMallocHost(&h_y, ySize * sizeof(int))); \
        TensorInit(h_x, xSize); \
        \
        for (int i = 0; i < tc.nDim - 1; ++i) { \
            scale[i] = static_cast<float>(tc.yDims[i + 1]) / \
                       static_cast<float>(tc.xDims[i + 1]); \
        } \
        \
        CHECK_CUDA(cudaMalloc(&d_x, xSize * sizeof(int))); \
        CHECK_CUDA(cudaMalloc(&d_y, ySize * sizeof(int))); \
        CHECK_CUDA(cudaMemcpy(d_x, h_x, xSize * sizeof(int), \
                              cudaMemcpyHostToDevice)); \
        \
        CHECK_HIEDNN(hiednnSetNormalTensorDesc( \
            xDesc, HIEDNN_DATATYPE_FP32, tc.nDim, tc.xDims)); \
        CHECK_HIEDNN(hiednnSetNormalTensorDesc( \
            yDesc, HIEDNN_DATATYPE_FP32, tc.nDim, tc.yDims)); \
        \
        CHECK_HIEDNN(hiednnCudaNearestInterpolation( \
            handle, HIE_COORD_MODE, HIE_NEAREST_MODE, \
            xDesc, d_x, scale, tc.nDim - 1, yDesc, d_y)); \
        \
        CHECK_CUDA(cudaMemcpy(h_y, d_y, ySize * sizeof(int), \
                              cudaMemcpyDeviceToHost)); \
        \
        switch (tc.nDim - 1) { \
            case 1: \
                CheckNearestOutput<1, COORD_FUNC<float>, NEAREST_FUNC<float>>( \
                    h_x, h_y, scale, tc); \
                break; \
            case 2: \
                CheckNearestOutput<2, COORD_FUNC<float>, NEAREST_FUNC<float>>( \
                    h_x, h_y, scale, tc); \
                break; \
            case 3: \
                CheckNearestOutput<3, COORD_FUNC<float>, NEAREST_FUNC<float>>( \
                    h_x, h_y, scale, tc); \
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

// half_pixel
NEAREST_TEST(HalfPixel_HalfDown,
             HIEDNN_INTERP_COORD_HALF_PIXEL,
             HIEDNN_INTERP_NEAREST_HALF_DOWN,
             HalfPixel,
             RoundHalfDown)

NEAREST_TEST(HalfPixel_HalfUp,
             HIEDNN_INTERP_COORD_HALF_PIXEL,
             HIEDNN_INTERP_NEAREST_HALF_UP,
             HalfPixel,
             RoundHalfUp)

NEAREST_TEST(HalfPixel_Floor,
             HIEDNN_INTERP_COORD_HALF_PIXEL,
             HIEDNN_INTERP_NEAREST_FLOOR,
             HalfPixel,
             RoundFloor)

NEAREST_TEST(HalfPixel_Ceil,
             HIEDNN_INTERP_COORD_HALF_PIXEL,
             HIEDNN_INTERP_NEAREST_CEIL,
             HalfPixel,
             RoundCeil)

// pytorch half pixel
NEAREST_TEST(PytorchHalfPixel_HalfDown,
             HIEDNN_INTERP_COORD_PYTORCH_HALF_PIXEL,
             HIEDNN_INTERP_NEAREST_HALF_DOWN,
             PytorchHalfPixel,
             RoundHalfDown)

NEAREST_TEST(PytorchHalfPixel_HalfUp,
             HIEDNN_INTERP_COORD_PYTORCH_HALF_PIXEL,
             HIEDNN_INTERP_NEAREST_HALF_UP,
             PytorchHalfPixel,
             RoundHalfUp)

NEAREST_TEST(PytorchHalfPixel_Floor,
             HIEDNN_INTERP_COORD_PYTORCH_HALF_PIXEL,
             HIEDNN_INTERP_NEAREST_FLOOR,
             PytorchHalfPixel,
             RoundFloor)

NEAREST_TEST(PytorchHalfPixel_Ceil,
             HIEDNN_INTERP_COORD_PYTORCH_HALF_PIXEL,
             HIEDNN_INTERP_NEAREST_CEIL,
             PytorchHalfPixel,
             RoundCeil)

// align corners
NEAREST_TEST(AlignCorners_HalfDown,
             HIEDNN_INTERP_COORD_ALIGN_CORNER,
             HIEDNN_INTERP_NEAREST_HALF_DOWN,
             AlignCorners,
             RoundHalfDown)

NEAREST_TEST(AlignCorners_HalfUp,
             HIEDNN_INTERP_COORD_ALIGN_CORNER,
             HIEDNN_INTERP_NEAREST_HALF_UP,
             AlignCorners,
             RoundHalfUp)

NEAREST_TEST(AlignCorners_Floor,
             HIEDNN_INTERP_COORD_ALIGN_CORNER,
             HIEDNN_INTERP_NEAREST_FLOOR,
             AlignCorners,
             RoundFloor)

NEAREST_TEST(AlignCorners_Ceil,
             HIEDNN_INTERP_COORD_ALIGN_CORNER,
             HIEDNN_INTERP_NEAREST_CEIL,
             AlignCorners,
             RoundCeil)

// asymmetric
NEAREST_TEST(Asymmetric_HalfDown,
             HIEDNN_INTERP_COORD_ASYMMETRIC,
             HIEDNN_INTERP_NEAREST_HALF_DOWN,
             Asymmetric,
             RoundHalfDown)

NEAREST_TEST(Asymmetric_HalfUp,
             HIEDNN_INTERP_COORD_ASYMMETRIC,
             HIEDNN_INTERP_NEAREST_HALF_UP,
             Asymmetric,
             RoundHalfUp)

NEAREST_TEST(Asymmetric_Floor,
             HIEDNN_INTERP_COORD_ASYMMETRIC,
             HIEDNN_INTERP_NEAREST_FLOOR,
             Asymmetric,
             RoundFloor)

NEAREST_TEST(Asymmetric_Ceil,
             HIEDNN_INTERP_COORD_ASYMMETRIC,
             HIEDNN_INTERP_NEAREST_CEIL,
             Asymmetric,
             RoundCeil)

