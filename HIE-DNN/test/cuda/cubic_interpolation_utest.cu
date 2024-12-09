/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    cubic_interpolation_utest.cu
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

struct CubicTestCase {
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

const std::vector<CubicTestCase> cubicTestCases = {
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
void DataInit(T *data, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        float x = 0.001f * static_cast<float>(i);
        data[i] = static_cast<T>(x);
    }
}

struct HalfPixel {
    float scaleRcp;

    HalfPixel() = default;

    HalfPixel(float xLength, float yLength, float scale) {
        scaleRcp = 1 / scale;
    }

    float Map(const float &yCoord) {
        return (yCoord + 0.5) * scaleRcp - 0.5;
    }
};

struct PytorchHalfPixel {
    float scaleRcp;

    PytorchHalfPixel() = default;

    PytorchHalfPixel(float xLength, float yLength, float scale) {
        scaleRcp = yLength > 1 ? 1 / scale : 0;
    }

    float Map(const float &yCoord) {
        return (yCoord + 0.5) * scaleRcp - 0.5;
    }
};

struct AlignCorners {
    float alignCornerScale;

    AlignCorners() = default;

    AlignCorners(float xLength, float yLength, float scale) {
        alignCornerScale = yLength > 1 ? (xLength - 1) / (yLength - 1) : 0;
    }

    float Map(const float &yCoord) {
        return yCoord * alignCornerScale;
    }
};

struct Asymmetric {
    float scaleRcp;

    Asymmetric() = default;

    Asymmetric(float xLength, float yLength, float scale) {
        scaleRcp = 1 / scale;
    }

    float Map(const float &yCoord) {
        return yCoord * scaleRcp;
    }
};

void CubicWeightInit(
        float (&weight)[4], const float &coord, const float &coeff) {
    float offset = coord - std::floor(coord);

    /*
     * r: offset
     * c: coeff
     * weight[0] = ((c * (r + 1) - 5 * c) * (r + 1) + 8 * c) * (r + 1) - 4 * c;
     * weight[1] = ((c + 2) * r - (c + 3)) * r * r + 1;
     * r = 1 - r;
     * weight[2] = ((c + 2) * r - (c + 3)) * r * r + 1;
     * weight[3] = ((c * (r + 1) - 5 * c) * (r + 1) + 8 * c) * (r + 1) - 4 * c;
     */
    float rr = offset * offset;
    float rrr = rr * offset;
    float cr = coeff * offset;
    float crr = coeff * rr;
    float crrr = coeff * rrr;

    weight[0] = crrr - 2 * crr + cr;
    weight[1] = crrr + 2 * rrr - crr - 3 * rr + 1;
    weight[2] = -cr + 2 * crr + 3 * rr - crrr - 2 * rrr;
    weight[3] = crr - crrr;
}

void CubicWeightNormalize(float (&weight)[4], int coordStart, int size) {
    for (int i = 0; i < 4; ++i) {
        if (coordStart + i < 0 || coordStart + i > size) {
            weight[i] = 0;
        }
    }

    float weightAcc = 0;
    for (int i = 0; i < 4; ++i) {
        weightAcc += weight[i];
    }

    float weightAccRcp = 1 / weightAcc;
    for (int i = 0; i < 4; ++i) {
        weight[i] *= weightAccRcp;
    }
}

template <int INTERP_DIM, bool EXCLUDE_OUTSIDE, typename T>
struct GetCubicOutput;

template <bool EXCLUDE_OUTSIDE, typename T>
struct GetCubicOutput<1, EXCLUDE_OUTSIDE, T> {
    static float Get(const T *x,
                     const int64_t *xDims,
                     const int64_t *xStrides,
                     const float *xCoordFP,
                     int batchId,
                     const float &coeff) {
        int coordStart = static_cast<int>(std::floor(xCoordFP[0] - 1.f));

        float weight[4];
        CubicWeightInit(weight, xCoordFP[0], coeff);
        if (EXCLUDE_OUTSIDE) {
            CubicWeightNormalize(weight, coordStart, xDims[1]);
        }

        int xCoord[4];
        for (int i = 0; i < 4; ++i) {
            xCoord[i] = coordStart + i;
            if (xCoord[i] < 0) {
                xCoord[i] = 0;
            }
            if (xCoord[i] >= xDims[1]) {
                xCoord[i] = xDims[1] - 1;
            }
        }

        float ret = 0;
        for (int i = 0; i < 4; ++i) {
            ret += static_cast<float>(x[batchId * xStrides[0] + xCoord[i]]) *
                   weight[i];
        }
        return ret;
    }
};

template <bool EXCLUDE_OUTSIDE, typename T>
struct GetCubicOutput<2, EXCLUDE_OUTSIDE, T> {
    static float Get(const T *x,
                     const int64_t *xDims,
                     const int64_t *xStrides,
                     const float *xCoordFP,
                     int batchId,
                     const float &coeff) {
        int coordStart[2];
        coordStart[0] = static_cast<int>(std::floor(xCoordFP[0] - 1.f));
        coordStart[1] = static_cast<int>(std::floor(xCoordFP[1] - 1.f));

        float weight[2][4];
        CubicWeightInit(weight[0], xCoordFP[0], coeff);
        CubicWeightInit(weight[1], xCoordFP[1], coeff);
        if (EXCLUDE_OUTSIDE) {
            CubicWeightNormalize(weight[0], coordStart[0], xDims[1]);
            CubicWeightNormalize(weight[1], coordStart[1], xDims[2]);
        }

        int xCoord[2][4];
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 4; ++j) {
                xCoord[i][j] = coordStart[i] + j;
                if (xCoord[i][j] < 0) {
                    xCoord[i][j] = 0;
                }
                if (xCoord[i][j] >= xDims[i + 1]) {
                    xCoord[i][j] = xDims[i + 1] - 1;
                }
            }
        }

        float yData[4];
        for (int i = 0; i < 4; ++i) {
            const T *xPtr = x + batchId * xStrides[0] +
                            xCoord[0][i] * xStrides[1];
            yData[i] = 0;
            for (int j = 0; j < 4; ++j) {
                yData[i] += static_cast<float>(xPtr[xCoord[1][j]]) *
                            weight[1][j];
            }
        }

        float ret = 0;
        for (int i = 0; i < 4; ++i) {
            ret += yData[i] * weight[0][i];
        }
        return ret;
    }
};

template <bool EXCLUDE_OUTSIDE, typename T>
struct GetCubicOutput<3, EXCLUDE_OUTSIDE, T> {
    static float Get(const T *x,
                     const int64_t *xDims,
                     const int64_t *xStrides,
                     const float *xCoordFP,
                     int batchId,
                     const float &coeff) {
        int coordStart[3];
        coordStart[0] = static_cast<int>(std::floor(xCoordFP[0] - 1.f));
        coordStart[1] = static_cast<int>(std::floor(xCoordFP[1] - 1.f));
        coordStart[2] = static_cast<int>(std::floor(xCoordFP[2] - 1.f));

        float weight[3][4];
        CubicWeightInit(weight[0], xCoordFP[0], coeff);
        CubicWeightInit(weight[1], xCoordFP[1], coeff);
        CubicWeightInit(weight[2], xCoordFP[2], coeff);
        if (EXCLUDE_OUTSIDE) {
            CubicWeightNormalize(weight[0], coordStart[0], xDims[1]);
            CubicWeightNormalize(weight[1], coordStart[1], xDims[2]);
            CubicWeightNormalize(weight[2], coordStart[2], xDims[3]);
        }

        int xCoord[3][4];
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 4; ++j) {
                xCoord[i][j] = coordStart[i] + j;
                if (xCoord[i][j] < 0) {
                    xCoord[i][j] = 0;
                }
                if (xCoord[i][j] >= xDims[i + 1]) {
                    xCoord[i][j] = xDims[i + 1] - 1;
                }
            }
        }

        float yData0[4][4];
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                const T *xPtr = x + batchId * xStrides[0] +
                                xCoord[0][i] * xStrides[1] +
                                xCoord[1][j] * xStrides[2];
                yData0[i][j] = 0;
                for (int k = 0; k < 4; ++k) {
                    yData0[i][j] += static_cast<float>(xPtr[xCoord[2][k]]) *
                                    weight[2][k];
                }
            }
        }

        float yData1[4];
        for (int i = 0; i < 4; ++i) {
            yData1[i] = 0;
            for (int j = 0; j < 4; ++j) {
                yData1[i] += yData0[i][j] * weight[1][j];
            }
        }

        float ret = 0;
        for (int i = 0; i < 4; ++i) {
            ret += yData1[i] * weight[0][i];
        }
        return ret;
    }
};

template <int INTERP_DIM,
          bool EXCLUDE_OUTSIDE,
          typename CoordFunc,
          typename T>
void CheckCubicOutput(const T *x,
                      const T *y,
                      const float *scale,
                      const float &coeff,
                      const CubicTestCase &testCase,
                      float relDiff,
                      float absDiff) {
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
        }

        float yStd = GetCubicOutput<INTERP_DIM, EXCLUDE_OUTSIDE, T>::Get(
            x, xDims, xStrides, xCoordFP, yCoord[0], coeff);

        ASSERT_NEAR(yStd, y[yOffset], std::fabs(yStd) * relDiff + absDiff);
    }
}

}  // anonymous namespace

#define CUBIC_TEST(TEST_NAME, HIE_COORD_MODE, COORD_FUNC, EXCLUDE_OUTSIDE) \
TEST(CubicInterpolationF32, TEST_NAME) { \
    hiednnCudaHandle_t handle; \
    CHECK_HIEDNN(hiednnCreateCudaHandle(&handle)); \
    \
    hiednnTensorDesc_t xDesc; \
    hiednnTensorDesc_t yDesc; \
    CHECK_HIEDNN(hiednnCreateTensorDesc(&xDesc)); \
    CHECK_HIEDNN(hiednnCreateTensorDesc(&yDesc)); \
    float scale[3]; \
    \
    for (int caseIt = 0; caseIt < cubicTestCases.size(); ++caseIt) { \
        const auto &tc = cubicTestCases[caseIt]; \
        \
        int64_t xSize = GetTensorSize(tc.xDims, tc.nDim); \
        int64_t ySize = GetTensorSize(tc.yDims, tc.nDim); \
        float *d_x, *d_y; \
        float *h_x, *h_y; \
        CHECK_CUDA(cudaMallocHost(&h_x, xSize * sizeof(float))); \
        CHECK_CUDA(cudaMallocHost(&h_y, ySize * sizeof(float))); \
        DataInit(h_x, xSize); \
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
        float coeff = -0.75f; \
        CHECK_HIEDNN(hiednnCudaCubicInterpolation( \
            handle, HIE_COORD_MODE, &coeff, EXCLUDE_OUTSIDE, xDesc, d_x, \
            scale, tc.nDim - 1, yDesc, d_y)); \
        \
        CHECK_CUDA(cudaMemcpy(h_y, d_y, ySize * sizeof(float), \
                              cudaMemcpyDeviceToHost)); \
        \
        switch (tc.nDim - 1) { \
            case 1: \
                CheckCubicOutput<1, EXCLUDE_OUTSIDE, COORD_FUNC, float> \
                    (h_x, h_y, scale, coeff, tc, 1e-2, 1e-5); \
                break; \
            case 2: \
                CheckCubicOutput<2, EXCLUDE_OUTSIDE, COORD_FUNC, float> \
                    (h_x, h_y, scale, coeff, tc, 1e-2, 1e-5); \
                break; \
            case 3: \
                CheckCubicOutput<3, EXCLUDE_OUTSIDE, COORD_FUNC, float> \
                    (h_x, h_y, scale, coeff, tc, 1e-2, 1e-5); \
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

CUBIC_TEST(half_pixel, HIEDNN_INTERP_COORD_HALF_PIXEL, HalfPixel, false)
CUBIC_TEST(half_pixel_ex, HIEDNN_INTERP_COORD_HALF_PIXEL, HalfPixel, true)
CUBIC_TEST(pytorch_half_pixel, HIEDNN_INTERP_COORD_PYTORCH_HALF_PIXEL,
           PytorchHalfPixel, false)
CUBIC_TEST(pytorch_half_pixel_ex, HIEDNN_INTERP_COORD_PYTORCH_HALF_PIXEL,
           PytorchHalfPixel, true)
CUBIC_TEST(align_corners, HIEDNN_INTERP_COORD_ALIGN_CORNER,
           AlignCorners, false)
CUBIC_TEST(align_corners_ex, HIEDNN_INTERP_COORD_ALIGN_CORNER,
           AlignCorners, true)
CUBIC_TEST(asymmetric, HIEDNN_INTERP_COORD_ASYMMETRIC, Asymmetric, false)
CUBIC_TEST(asymmetric_ex, HIEDNN_INTERP_COORD_ASYMMETRIC, Asymmetric, true)

