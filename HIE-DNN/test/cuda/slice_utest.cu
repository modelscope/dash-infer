/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    slice_utest.cu
 */
#include <hiednn.h>
#include <hiednn_cuda.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <iostream>
#include <vector>

#include <utest_utils.hpp>

namespace {

constexpr int TENSOR_DIM_MAX = 8;

typedef struct {
    int xNDims;
    int64_t xDims[TENSOR_DIM_MAX];
    int yNDims;
    int64_t yDims[TENSOR_DIM_MAX];
    int nParams;
    int axes[TENSOR_DIM_MAX];
    int64_t starts[TENSOR_DIM_MAX];
    int64_t ends[TENSOR_DIM_MAX];
    int64_t steps[TENSOR_DIM_MAX];
} TestCase;

const std::vector<TestCase> tinyTestCases = {
    // zero case
    {3, {3, 6, 2}, 3, {3, 6, 2}, 0},
    // output size is 0
    {3, {3, 6, 2}, 3, {3, 0, 2}, 1, {1}, {3}, {1}, {1}},
    // output size is still 0
    {3, {3, 6, 2}, 3, {3, 0, 2}, 1, {1}, {5}, {5}, {1}},
    // simple case
    {3, {3, 6, 2}, 3, {3, 2, 2}, 2, {0, 1}, {0, 2}, {3, 6}, {1, 2}},
    // end out of bound
    {3, {3, 6, 2}, 3, {3, 2, 2}, 2, {0, 1}, {0, 2}, {3, 10000}, {1, 2}},
    // odd case
    {3, {2, 4, 3}, 3, {1, 4, 3}, 1, {0}, {0}, {1}, {1}},
    // ugly case, shuffle axes
    {3, {3, 6, 3}, 3, {2, 2, 2}, 3, {2, 0, 1}, {0, 1, 2}, {3, 3, 6}, {2, 1, 2}},
    // ---------------- neg ----------------
    // output size is 0
    {3, {3, 6, 2}, 3, {3, 0, 2}, 1, {1}, {-2}, {-2}, {1}},
    // output size is 0 again
    {3, {3, 6, 2}, 3, {3, 0, 2}, 1, {1}, {-2}, {4}, {-1}},
    // output size is still 0
    {3, {3, 6, 2}, 3, {3, 0, 2}, 1, {1}, {0}, {-1}, {-5}},
    // very neg case
    {3, {3, 6, 2}, 3, {3, 2, 2}, 2, {-3, -2}, {-3, 6}, {3, -4}, {1, -2}},
    // really neg case
    {3, {3, 6, 2}, 3, {3, 2, 2},
    3, {-1, -3, -2}, {-1, -3, 6}, {-1000, 3, -4}, {-1, 1, -2}},
    // start out of bound
    {3, {3, 6, 2}, 3, {3, 2, 2}, 2, {-3, -2}, {-1000, 6}, {3, -4}, {1, -2}},
};

const std::vector<TestCase> largeTestCases = {
    {4, {12, 70, 20, 5}, 4, {12, 70, 20, 5}, 0},
    {4, {12, 70, 20, 5}, 4, {12, 14, 20, 5}, 1, {1}, {13}, {67}, {4}},
    {4, {12, 70, 20, 5}, 4, {12, 14, 20, 2},
    2, {1, 3}, {13, 1}, {67, 4}, {4, 2}},
    {4, {12, 70, 20, 5}, 4, {12, 14, 4, 2},
    3, {1, 2, 3}, {13, 5, 1}, {67, 17, 4}, {4, 3, 2}},
    {4, {12, 70, 20, 5}, 4, {3, 14, 4, 2},
    4, {1, 2, 3, 0}, {13, 5, 1, 3}, {67, 17, 4, 10}, {4, 3, 2, 3}},
    // huge slice, may take up to 1.7 GB memory when using int64_t;
    // uncomment it with caution
    // {8, {8, 8, 8, 16, 16, 15, 13, 9}, 8, {3, 4, 2, 3, 3, 2, 2, 2},
    // 8, {0, 1, 2, 3, 4, 5, 6, 7}, {3, 2, 1, 9, 6, 11, 2, 4},
    // {8, 6, 8, 14, 16, 15, 7, 7}, {2, 1, 5, 2, 4, 2, 3, 2}},
};

const std::vector<TestCase> defaultAxesTestCases = {
    {3, {3, 6, 2}, 3, {3, 2, 2}, 2, {0, 1}, {0, 2}, {3, 6}, {1, 2}},
    {4, {12, 70, 20, 5}, 4, {3, 14, 4, 2},
    4, {0, 1, 2, 3}, {3, 13, 5, 1}, {10, 67, 17, 4}, {3, 4, 3, 2}},
};

const std::vector<TestCase> defaultStepsTestCases = {
    {3, {3, 6, 2}, 3, {3, 4, 2}, 2, {0, 1}, {0, 2}, {3, 6}, {1, 1}},
    {4, {12, 70, 20, 5}, 4, {7, 54, 12, 3},
    4, {0, 1, 2, 3}, {3, 13, 5, 1}, {10, 67, 17, 4}, {1, 1, 1, 1}},
    // full slice, but actually memcpy
    {3, {3, 6, 2}, 3, {3, 6, 2}, 3, {0, 1, 2},
    {-INT_MAX, -2000, 0}, {5000, 6, INT_MAX}, {1, 1, 1}},
};

const std::vector<TestCase> invalidParamTestCases = {
    // yNDim != xNDim
    {3, {3, 6, 2}, 2, {3, 2}, 2, {0, 1}, {0, 2}, {3, 6}, {1, 2}},
    // nParam < 0
    {3, {3, 6, 2}, 3, {3, 2, 2}, -1, {0, 1}, {0, 2}, {3, 6}, {1, 2}},
    // nParam too large
    {3, {3, 6, 2}, 3, {3, 2, 2}, 5, {0, 1}, {0, 2}, {3, 6}, {1, 2}},
    // zero step
    {3, {3, 6, 2}, 3, {3, 2, 2}, 2, {0, 1}, {0, 2}, {3, 6}, {0, 2}},
    // axis too neg
    {3, {3, 6, 2}, 3, {3, 2, 2}, 2, {0, -4}, {0, 2}, {3, 6}, {1, 2}},
    // axis too large
    {3, {3, 6, 2}, 3, {3, 2, 2}, 2, {0, 4}, {0, 2}, {3, 6}, {1, 2}},
    // dup axes
    {3, {3, 6, 2}, 3, {3, 2, 2}, 2, {1, 1}, {0, 2}, {3, 6}, {1, 2}},
    // dup neg axes
    {3, {3, 6, 2}, 3, {3, 2, 2}, 2, {1, -2}, {0, 2}, {3, 6}, {1, 2}},
    // dup neg axes 2
    {3, {3, 6, 2}, 3, {3, 2, 2}, 2, {-3, 0}, {0, 2}, {3, 6}, {1, 2}},
    // wrong y shape, outer
    {3, {3, 6, 2}, 3, {3, 1, 2}, 2, {0, 1}, {0, 2}, {3, 6}, {1, 2}},
    // wrong y shape, inner
    {3, {3, 6, 2}, 3, {3, 2, 3}, 2, {0, 1}, {0, 2}, {3, 6}, {1, 2}},
    // wrong y shape, with 0 size
    {3, {3, 6, 2}, 3, {3, 0, 3}, 2, {0, 1}, {0, 2}, {3, 6}, {1, 2}},
    // nParam == 0, x & y of different shapes
    {3, {3, 6, 2}, 3, {3, 2, 2}, 0},
};

template <typename T>
inline T SIntDivUp(T x, T y) {
    // round away from 0
    T ret;

    if (x == 0) {
        ret = 0;
    } else if (((x ^ y) & (T(1) << (8 * sizeof(T) - 1))) != 0) {
        ret = (x - y - (x > 0 ? 1 : -1)) / y;
    } else {
        ret = (x + y - (x > 0 ? 1 : -1)) / y;
    }

    return ret;
}

int64_t TensorSize(const int64_t *dims, int nDims) {
    int64_t size = 1;
    for (int i = 0; i < nDims; ++i) {
        size *= dims[i];
    }
    return size;
}

void TensorStride(int64_t *strides, const int64_t *dims, int nDims) {
    int64_t size = 1;
    for (int i = nDims - 1; i >= 0; --i) {
        strides[i] = size;
        size *= dims[i];
    }
}

template <typename T>
void printArray(const T* array, int len) {
    std::cout << "[";
    for (int i = 0; i < len; i++) {
        std::cout << (array[i]) << " ";
    }
    std::cout << "]";
    std::cout << std::endl;
}

class Slice_CUDA : public testing::Test {
 protected:
    void SetUp() override {
        CHECK_HIEDNN(hiednnCreateCudaHandle(&handle));
        CHECK_HIEDNN(hiednnCreateTensorDesc(&xDesc));
        CHECK_HIEDNN(hiednnCreateTensorDesc(&yDesc));
    }

    void TearDown() override {
        CHECK_HIEDNN(hiednnDestroyTensorDesc(xDesc));
        CHECK_HIEDNN(hiednnDestroyTensorDesc(yDesc));
        CHECK_HIEDNN(hiednnDestroyCudaHandle(handle));
        xDesc = nullptr;
        yDesc = nullptr;
        handle = nullptr;
    }

    template <typename T>
    void InitTestCase(hiednnDataType_t dataType, const TestCase& tc, bool REF) {
        CHECK_HIEDNN(
            hiednnSetNormalTensorDesc(xDesc, dataType, tc.xNDims, tc.xDims));
        CHECK_HIEDNN(
            hiednnSetNormalTensorDesc(yDesc, dataType, tc.yNDims, tc.yDims));

        auto xSize = TensorSize(tc.xDims, tc.xNDims);
        auto ySize = TensorSize(tc.yDims, tc.yNDims);

        CHECK_CUDA(cudaMallocHost(&hX, xSize * sizeof(T)));
        CHECK_CUDA(cudaMallocHost(&hY, ySize * sizeof(T)));
        if (REF) {
            CHECK_CUDA(cudaMallocHost(&hRef, ySize * sizeof(T)));
        }

        T *hX_p = static_cast<T*>(hX);
        for (int64_t i = 0; i < xSize; i++) {
            hX_p[i] = i;
        }
        // printArray(hX_p, xSize);

        CHECK_CUDA(cudaMalloc(&dX, xSize * sizeof(T)));
        CHECK_CUDA(cudaMalloc(&dY, ySize * sizeof(T)));
        CHECK_CUDA(
            cudaMemcpy(dX, hX, xSize * sizeof(T), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(dY, 0, ySize * sizeof(T)));
    }

    void ClearTestCase() {
        CHECK_CUDA(cudaFree(dX));
        CHECK_CUDA(cudaFree(dY));
        CHECK_CUDA(cudaFreeHost(hX));
        CHECK_CUDA(cudaFreeHost(hY));
        CHECK_CUDA(cudaFreeHost(hRef));
        hRef = nullptr;
        hX = nullptr;
        hY = nullptr;
        dX = nullptr;
        dY = nullptr;
    }

    template <typename T>
    void TestSlice(hiednnDataType_t dataType,
                   const std::vector<TestCase>& cases,
                   hiednnStatus_t wantRet = HIEDNN_STATUS_SUCCESS) {
        for (const auto &tc : cases) {
            InitTestCase<T>(dataType, tc, wantRet == HIEDNN_STATUS_SUCCESS);

            ASSERT_EQ(
                hiednnCudaSlice(handle, xDesc, dX, tc.starts, tc.ends, tc.steps,
                    tc.axes, tc.nParams, yDesc, dY),
                wantRet);
            if (wantRet == HIEDNN_STATUS_SUCCESS) {
                T *hX_p = static_cast<T*>(hX);
                T *hY_p = static_cast<T*>(hY);
                T *hRef_p = static_cast<T*>(hRef);
                // printArray(hX_p, TensorSize(tc.xDims, tc.xNDims));

                auto ySize = TensorSize(tc.yDims, tc.yNDims);
                CHECK_CUDA(cudaMemcpy(hY, dY, ySize * sizeof(T),
                    cudaMemcpyDeviceToHost));
                // printArray(hY_p, ySize);

                Slice(hX_p, hRef_p, tc);
                // printArray(hRef_p, ySize);
                CheckTensor(hY_p, hRef_p, ySize);
            }

            ClearTestCase();
        }
    }

    template <typename T>
    void TestSliceDefaultInputs(hiednnDataType_t dataType,
                                const std::vector<TestCase>& cases,
                                bool defaultAxes = false,
                                bool defaultSteps = false) {
        for (const auto &tc : cases) {
            InitTestCase<T>(dataType, tc, true);

            auto axes = defaultAxes ? nullptr : tc.axes;
            auto steps = defaultSteps ? nullptr : tc.steps;

            ASSERT_EQ(
                hiednnCudaSlice(handle, xDesc, dX, tc.starts, tc.ends, steps,
                    axes, tc.nParams, yDesc, dY),
                HIEDNN_STATUS_SUCCESS);
            T *hX_p = static_cast<T*>(hX);
            T *hY_p = static_cast<T*>(hY);
            T *hRef_p = static_cast<T*>(hRef);
            // printArray(hX_p, TensorSize(tc.xDims, tc.xNDims));

            auto ySize = TensorSize(tc.yDims, tc.yNDims);
            CHECK_CUDA(cudaMemcpy(hY, dY, ySize * sizeof(T),
                cudaMemcpyDeviceToHost));
            // printArray(hY_p, ySize);

            Slice(hX_p, hRef_p, tc);
            // printArray(hRef_p, ySize);
            CheckTensor(hY_p, hRef_p, ySize);

            ClearTestCase();
        }
    }

    template <typename T>
    void TestSliceSpecialCase(hiednnDataType_t dataType,
                              const std::vector<TestCase>& cases) {
        for (const auto &tc : cases) {
            ASSERT_EQ(tc.nParams, 0);
            InitTestCase<T>(dataType, tc, false);
            ASSERT_EQ(
                hiednnCudaSlice(handle, xDesc, dX, tc.starts, tc.ends, tc.steps,
                    tc.axes, tc.nParams, yDesc, dX),
                HIEDNN_STATUS_SUCCESS);
            T *hX_p = static_cast<T*>(hX);
            T *hY_p = static_cast<T*>(hY);
            auto ySize = TensorSize(tc.yDims, tc.yNDims);
            CHECK_CUDA(cudaMemcpy(hY, dX, ySize * sizeof(T),
                cudaMemcpyDeviceToHost));
            CheckTensor(hX_p, hY_p, ySize);
            ClearTestCase();
        }
    }

    // -------------- helpers --------------

    template <typename T>
    static T GetElement(const T *x, int nDim,
                        const int64_t (&index)[TENSOR_DIM_MAX],
                        const int64_t (&strides)[TENSOR_DIM_MAX]) {
        int64_t offset = 0;
        for (int i = 0; i < nDim; i++) {
            offset += index[i] * strides[i];
        }
        return x[offset];
    }

    template <typename T>
    static void SetElement(T *x, T val, int nDim,
                           const int64_t (&index)[TENSOR_DIM_MAX],
                           const int64_t (&strides)[TENSOR_DIM_MAX]) {
        int64_t offset = 0;
        for (int i = 0; i < nDim; i++) {
            offset += index[i] * strides[i];
        }
        x[offset] = val;
    }

    template <typename T>
    static void CheckTensor(const T *x, const T *y, size_t size) {
        for (size_t i = 0; i < size; i++) {
            ASSERT_EQ(x[i], y[i]);
        }
    }

    template <typename T>
    static void Slice(const T *x, T *y, const TestCase& tc) {
        ASSERT_EQ(tc.xNDims, tc.yNDims);
        int64_t starts[TENSOR_DIM_MAX]{0L};
        int64_t steps[TENSOR_DIM_MAX];
        int64_t ends[TENSOR_DIM_MAX];
        for (int i = 0; i < tc.xNDims; i++) {
            steps[i] = 1;
            ends[i] = tc.xDims[i];
        }

        const int &nDims = tc.xNDims;
        int maxAxis = -1;
        int count[TENSOR_DIM_MAX]{0};
        for (int i = 0; i < tc.nParams; i++) {
            int axis = tc.axes[i];
            if (axis < 0) {
                axis += nDims;
            }
            // forbid duplication
            ASSERT_GE(axis, 0);
            ASSERT_LT(axis, nDims);
            ASSERT_EQ(++count[axis], 1);
            maxAxis = axis > maxAxis ? axis : maxAxis;

            auto start = tc.starts[i];
            auto step = tc.steps[i];
            auto end = tc.ends[i];
            const auto &dim = tc.xDims[axis];
            if (start < 0) {
                start += dim;
            }
            if (end < 0) {
                end += dim;
            }
            ASSERT_NE(step, 0);
            if (step > 0) {
                start = std::max(start, 0L);
                start = std::min(start, dim);
                end = std::max(end, 0L);
                end = std::min(end, dim);
                ASSERT_GE(start, 0);
                ASSERT_LE(start, dim);
                ASSERT_GE(end, 0);
                ASSERT_LE(end, dim);
            } else {
                start = std::max(start, 0L);
                start = std::min(start, dim - 1L);
                end = std::max(end, -1L);
                end = std::min(end, dim - 1L);
                ASSERT_GE(start, 0);
                ASSERT_LE(start, dim - 1);
                ASSERT_GE(end, -1);
                ASSERT_LE(end, dim - 1);
            }

            starts[axis] = start;
            steps[axis] = step;
            ends[axis] = end;
        }

        ASSERT_LE(maxAxis, tc.xNDims);
        int64_t yDims[TENSOR_DIM_MAX];
        for (int i = 0; i < nDims; i++) {
            yDims[i] = SIntDivUp(ends[i] - starts[i], steps[i]);
            yDims[i] = std::max(0L, yDims[i]);
            ASSERT_EQ(yDims[i], tc.yDims[i]);
        }

        auto ySize = TensorSize(yDims, nDims);
        int64_t xStrides[TENSOR_DIM_MAX];
        int64_t yStrides[TENSOR_DIM_MAX];
        TensorStride(xStrides, tc.xDims, nDims);
        TensorStride(yStrides, yDims, nDims);

        for (int64_t yOffset = 0; yOffset < ySize; yOffset++) {
            int64_t xIndex[TENSOR_DIM_MAX];
            int64_t yIndex[TENSOR_DIM_MAX];
            int64_t innerOffset = yOffset;

            for (int dim = 0; dim < nDims; dim++) {
                yIndex[dim] = innerOffset / yStrides[dim];
                xIndex[dim] = starts[dim] + yIndex[dim] * steps[dim];
                innerOffset = innerOffset % yStrides[dim];
            }

            ASSERT_EQ(innerOffset, 0);
            int64_t xOffset = 0;
            for (int dim = 0; dim < nDims; dim++) {
                xOffset += xIndex[dim] * xStrides[dim];
            }
            y[yOffset] = x[xOffset];
        }
    }

    // -------------- members --------------

    hiednnCudaHandle_t handle = nullptr;
    hiednnTensorDesc_t xDesc = nullptr;
    hiednnTensorDesc_t yDesc = nullptr;

    void *hX = nullptr;
    void *hY = nullptr;
    void *hRef = nullptr;
    void *dX = nullptr;
    void *dY = nullptr;
};

}  // anonymous namespace

TEST_F(Slice_CUDA, INVALID_PARAM) {
    TestSlice<int32_t>(HIEDNN_DATATYPE_INT32, invalidParamTestCases,
        HIEDNN_STATUS_INVALID_PARAMETER);
}

TEST_F(Slice_CUDA, INT8_TINY) {
    TestSlice<int8_t>(HIEDNN_DATATYPE_INT8, tinyTestCases);
}
TEST_F(Slice_CUDA, INT16_TINY) {
    TestSlice<int16_t>(HIEDNN_DATATYPE_INT16, tinyTestCases);
}
TEST_F(Slice_CUDA, INT32_TINY) {
    TestSlice<int32_t>(HIEDNN_DATATYPE_INT32, tinyTestCases);
}
TEST_F(Slice_CUDA, INT64_TINY) {
    TestSlice<int64_t>(HIEDNN_DATATYPE_INT64, tinyTestCases);
}

TEST_F(Slice_CUDA, INT8_LARGE) {
    TestSlice<int8_t>(HIEDNN_DATATYPE_INT8, largeTestCases);
}
TEST_F(Slice_CUDA, INT16_LARGE) {
    TestSlice<int16_t>(HIEDNN_DATATYPE_INT16, largeTestCases);
}
TEST_F(Slice_CUDA, INT32_LARGE) {
    TestSlice<int32_t>(HIEDNN_DATATYPE_INT32, largeTestCases);
}
TEST_F(Slice_CUDA, INT64_LARGE) {
    TestSlice<int64_t>(HIEDNN_DATATYPE_INT64, largeTestCases);
}

TEST_F(Slice_CUDA, DEFAULT_AXES) {
    TestSliceDefaultInputs<int32_t>(HIEDNN_DATATYPE_INT32,
        defaultAxesTestCases, true);
}
TEST_F(Slice_CUDA, DEFAULT_STEPS) {
    TestSliceDefaultInputs<int32_t>(HIEDNN_DATATYPE_INT32,
        defaultStepsTestCases, false, true);
}
TEST_F(Slice_CUDA, DEFAULT_BOTH) {
    TestSliceDefaultInputs<int32_t>(HIEDNN_DATATYPE_INT32,
        defaultStepsTestCases, true, true);
}

/* ===========================================================
    Following tests involve huge memory alloc, copy & traverse,
    and they are for performance or oversized input test only. */
/*
const std::vector<TestCase> naiveHugeTestCases = {
    // huge memcpy, just memcpy
    {8, {8, 8, 8, 16, 16, 15, 13, 9}, 8, {8, 8, 8, 16, 16, 15, 13, 9}, 0},
};

const std::vector<TestCase> hugeTestCases = {
    // huge memcpy, AXES == 1, best case
    {8, {8, 8, 8, 16, 16, 15, 13, 9}, 8, {7, 8, 8, 16, 16, 15, 13, 9},
    1, {0}, {0}, {7}, {1}},
    // huge memcpy, able to pack, AXES == 4
    // uint64_t
    {8, {8, 8, 8, 16, 16, 15, 13, 9}, 8, {8, 8, 8, 16 - 1, 16, 15, 13, 9},
    4, {0, 1, 2, 3}, {0}, {8, 8, 8, 16 - 1}, {1, 1, 1, 1}},
    // uint32_t
    {8, {8, 8, 8, 16 * 4, 4, 15, 13, 9}, 8, {8, 8, 8, 16 * 4 - 1, 4, 15, 13, 9},
    4, {0, 1, 2, 3}, {0}, {8, 8, 8, 16 * 4 - 1}, {1, 1, 1, 1}},
    // uint16_t
    {8, {8, 8, 8, 16 * 8, 2, 15, 13, 9}, 8, {8, 8, 8, 16 * 8 - 1, 2, 15, 13, 9},
    4, {0, 1, 2, 3}, {0}, {8, 8, 8, 16 * 8 - 1}, {1, 1, 1, 1}},
    // uint8_t
    {8, {8, 8, 8, 16 * 16, 1, 15, 13, 9},
    8, {8, 8, 8, 16 * 16 - 1, 1, 15, 13, 9},
    4, {0, 1, 2, 3}, {0}, {8, 8, 8, 16 * 16 - 1}, {1, 1, 1, 1}},
    // huge memcpy, AXES == 8, worst case
    {8, {8, 8, 8, 16, 16, 15, 13, 9}, 8, {8, 8, 8, 16, 16, 15, 13, 9 - 1},
    8, {0, 1, 2, 3, 4, 5, 6, 7},
    {0}, {8, 8, 8, 16, 16, 15, 13, 9 - 1}, {1, 1, 1, 1, 1, 1, 1, 1}},
};

const std::vector<TestCase> megaTestCases = {
    // x index beyond INT32_MAX
    {1, {4294967295L + 100}, 1, {100},
    1, {0}, {4294967295L}, {4294967295L + 100}, {1}},
    {2, {INT32_MAX, 3}, 2, {1, 1}, 2, {0, 1}, {-2, -1}, {-1, -2}, {1, -1}},
    // x index within INT32_MAX
    {2, {INT32_MAX, 2}, 2, {1, 1}, 2, {0, 1}, {-2, -1}, {-1, -2}, {1, -1}},
    // following test is unavailable because of the max #thread limitation
    // {2, {1, 4294967295L}, 2, {1, 4294967295L}, 1, {0}, {0}, {1}, {1}},
};

const std::vector<TestCase> oversizeTestCases = {
    // y size too large
    {1, {UINT32_MAX + 2L}, 1, {UINT32_MAX + 1L},
    1, {0}, {1}, {UINT32_MAX + 10L}, {1}},
    // y stride too large
    {2, {2, UINT32_MAX / 2 + 2},
    2, {2, (UINT32_MAX) / 2 + 1}, 2, {0, 1},
    {0, 1}, {2, UINT32_MAX}, {1, 1}},
};

TEST_F(Slice_CUDA, OVERSIZE) {
    TestSlice<int8_t>(HIEDNN_DATATYPE_INT8, oversizeTestCases,
        HIEDNN_STATUS_TENSOR_OVERSIZE);
}
TEST_F(Slice_CUDA, MEGA) {
    TestSlice<int8_t>(HIEDNN_DATATYPE_INT8, megaTestCases);
}
// return directly if x == y & nParams == 0
TEST_F(Slice_CUDA, SPECIAL) {
    TestSliceSpecialCase<int8_t>(HIEDNN_DATATYPE_INT8, naiveHugeTestCases);
}

TEST_F(Slice_CUDA, INT8_HUGE_NAIVE) {
    TestSlice<int8_t>(HIEDNN_DATATYPE_INT8, naiveHugeTestCases);
}
TEST_F(Slice_CUDA, INT8_HUGE) {
    TestSlice<int8_t>(HIEDNN_DATATYPE_INT8, hugeTestCases);
}
TEST_F(Slice_CUDA, INT16_HUGE) {
    TestSlice<int16_t>(HIEDNN_DATATYPE_INT16, hugeTestCases);
}
TEST_F(Slice_CUDA, INT32_HUGE) {
    TestSlice<int32_t>(HIEDNN_DATATYPE_INT32, hugeTestCases);
}
TEST_F(Slice_CUDA, INT64_HUGE) {
    TestSlice<int64_t>(HIEDNN_DATATYPE_INT64, hugeTestCases);
}
*/
