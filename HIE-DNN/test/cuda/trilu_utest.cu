/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    trilu_utest.cu
 */
#include <hiednn.h>
#include <hiednn_cuda.h>

#include <gtest/gtest.h>

#include <iostream>
#include <vector>
#include <functional>

#include <utest_utils.hpp>

namespace {

constexpr int TENSOR_DIM_MAX = 8;

typedef struct {
    bool upper;
    int64_t k;
    int nDims;
    const int64_t dims[TENSOR_DIM_MAX];
} TestCase;

size_t TensorSize(const int64_t *dims, int nDims) {
    size_t size = 1UL;
    for (int i = 0; i < nDims; ++i) {
        size *= static_cast<size_t>(dims[i]);
    }
    return size;
}

void TensorStride(int64_t *strides, const int64_t *dims, int nDims) {
    int64_t size = 1L;
    for (int i = nDims - 1; i > 0; --i) {
        if (size < 0L) {
            throw std::runtime_error("TensorSize: stride overflow");
        }
        strides[i] = size;
        size *= dims[i];
    }
    // tail
    if (size < 0L) {
        throw std::runtime_error("TensorSize: stride overflow");
    }
    strides[0] = size;
}

void GetIndex(int64_t *index, size_t offset,
              const int64_t *strides, int nDims) {
    for (int i = 0; i < nDims - 1; i++) {
        index[i] = static_cast<int64_t>(
            offset / static_cast<size_t>(strides[i]));
        offset = offset % static_cast<size_t>(strides[i]);
    }
    // tail
    index[nDims - 1] = static_cast<int64_t>(
        offset / static_cast<size_t>(strides[nDims - 1]));
}

template <typename T>
static T GetElement(const T *x, int nDim,
                    const int64_t (&index)[TENSOR_DIM_MAX],
                    const int64_t (&strides)[TENSOR_DIM_MAX]) {
    size_t offset = 0UL;
    for (int i = 0; i < nDim; i++) {
        offset += static_cast<size_t>(index[i]) * strides[i];
    }
    return x[offset];
}

template <typename T>
static void SetElement(T *x, T val, int nDim,
                        const int64_t (&index)[TENSOR_DIM_MAX],
                        const int64_t (&strides)[TENSOR_DIM_MAX]) {
    size_t offset = 0UL;
    for (int i = 0; i < nDim; i++) {
        offset += static_cast<size_t>(index[i]) * strides[i];
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
void PrintArray(const T* array, int64_t len) {
    std::cout << "[";
    for (int64_t i = 0L; i < len; i++) {
        if (i > 0L) {
            std::cout << " ";
        }
        std::cout << array[i];
    }
    std::cout << "]";
}

template <typename T>
void PrintTensor(const T* array, const int64_t *dims, int nDim) {
    size_t position = 0UL;
    std::function<void(const int64_t*, int)> printTensorInternal;

    printTensorInternal = [&array, &position, &printTensorInternal](
            const int64_t *dims, int nDim){
        if (nDim == 1) {
            // PrintArray<T>(array + position, dims[0]);
            position += static_cast<size_t>(dims[0]);
            return;
        }

        auto dim = dims[0];
        std::cout << "[";
        for (int64_t i = 0L; i < dim; i++) {
            if (i > 0L) {
                std::cout << std::endl;
            }
            printTensorInternal(dims + 1, nDim - 1);
        }
        std::cout << "]";
        return;
    };

    printTensorInternal(dims, nDim);
    std::cout << std::endl;
}

class Trilu_CUDA : public testing::Test {
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
    void InitTestCase(hiednnDataType_t dataType,
                      const TestCase& tc,
                      bool REF) {
        const auto &nDims = tc.nDims;
        CHECK_HIEDNN(
            hiednnSetNormalTensorDesc(xDesc, dataType, nDims, tc.dims));
        CHECK_HIEDNN(
            hiednnSetNormalTensorDesc(yDesc, dataType, nDims, tc.dims));

        auto size = TensorSize(tc.dims, nDims);

        CHECK_CUDA(cudaMallocHost(&hX, size * sizeof(T)));
        CHECK_CUDA(cudaMallocHost(&hY, size * sizeof(T)));

        CHECK_CUDA(cudaMalloc(&dX, size * sizeof(T)));
        CHECK_CUDA(cudaMalloc(&dY, size * sizeof(T)));

        if (REF) {
            CHECK_CUDA(cudaMallocHost(&hRef, size * sizeof(T)));

            // init X, starting from 1
            T *hX_p = static_cast<T *>(hX);
            for (size_t i = 0UL; i < size; i++) {
                hX_p[i] = static_cast<T>(i + 1);
            }
            // PrintTensor(static_cast<T *>(hX), tc.dims, nDims);
            CHECK_CUDA(cudaMemcpy(
                dX, hX, size * sizeof(T), cudaMemcpyHostToDevice));

            // clear Y
            CHECK_CUDA(cudaMemset(dY, 0, size * sizeof(T)));
        }
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

    /* ----------------------------------------------------- */

    template <typename T>
    void TestTrilu(hiednnDataType_t dataType,
                   const std::vector<TestCase>& cases,
                   hiednnStatus_t wantRet = HIEDNN_STATUS_SUCCESS) {
        for (const auto &tc : cases) {
            InitTestCase<T>(dataType, tc, wantRet == HIEDNN_STATUS_SUCCESS);

            hiednnTriluOp_t op = tc.upper ? HIEDNN_TRILU_UPPER
                                          : HIEDNN_TRILU_LOWER;
            ASSERT_EQ(
                hiednnCudaTrilu(handle, xDesc, dX, yDesc, dY, tc.k, op),
                wantRet);
            if (wantRet == HIEDNN_STATUS_SUCCESS) {
                T *hX_p = static_cast<T *>(hX);
                T *hY_p = static_cast<T *>(hY);
                T *hRef_p = static_cast<T *>(hRef);

                auto ySize = TensorSize(tc.dims, tc.nDims);
                CHECK_CUDA(cudaMemcpy(hY, dY, ySize * sizeof(T),
                    cudaMemcpyDeviceToHost));
                // PrintTensor(hY_p, tc.dims, tc.nDims);

                Trilu(hRef_p, hX_p, tc);
                // PrintTensor(hRef_p, tc.dims, tc.nDims);
                CheckTensor(hY_p, hRef_p, ySize);
            }

            ClearTestCase();
        }
    }

    // -------------- reference --------------

    template <typename T>
    static void Trilu(T *y,
                      const T *x,
                      const TestCase& tc) {
        const auto &nDims = tc.nDims;
        ASSERT_GE(nDims, 2);

        const auto size = TensorSize(tc.dims, nDims);
        int64_t strides[TENSOR_DIM_MAX];
        TensorStride(strides, tc.dims, nDims);

        for (size_t offset = 0UL; offset < size; offset++) {
            int64_t index[TENSOR_DIM_MAX];
            GetIndex(index, offset, strides, nDims);
            auto index0 = index[nDims - 2];
            auto index1 = index[nDims - 1];
            if (tc.upper) {
                if (index0 <= index1 - tc.k) {
                    y[offset] = x[offset];
                } else {
                    y[offset] = static_cast<T>(0);
                }
            } else {
                if (index0 >= index1 - tc.k) {
                    y[offset] = x[offset];
                } else {
                    y[offset] = static_cast<T>(0);
                }
            }
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

/* ============= test cases ============= */
const std::vector<TestCase> invalidParamTestCases = {
    {false, 0, 1, {4}},
    {false, 0, 2, {4, 4}},
    {true, 0, 4, {10, 1, 3, 5}},
};

const std::vector<TestCase> tinyTestCases = {
    // tril
    {false, 0, 3, {1, 0, 5}},
    {false, 0, 3, {1, 4, 5}},
    {false, -1, 3, {1, 4, 5}},
    {false, 3, 3, {1, 4, 5}},
    {false, UINT32_MAX + 10L, 3, {1, 4, 5}},
    {false, 1, 3, {3, 1, 5}},
    {false, UINT32_MAX + 1L, 3, {3, 1, 5}},
    {false, -(UINT32_MAX + 1L), 3, {3, 1, 5}},
    // triu
    {true, 0, 3, {1, 0, 5}},
    {true, 0, 3, {1, 4, 5}},
    {true, -1, 3, {1, 4, 5}},
    {true, 3, 3, {1, 4, 5}},
    {true, UINT32_MAX + 10L, 3, {1, 4, 5}},
    {true, 1, 3, {3, 1, 5}},
    {true, UINT32_MAX + 1L, 3, {3, 1, 5}},
    {true, -(UINT32_MAX + 1L), 3, {3, 1, 5}},
};

const std::vector<TestCase> largeTestCases = {
    // tril
    {false, 0, 3, {7 * 7 * 7 * 7 * 3 * 3, 3, 3}},
    {false, -1, 3, {200, 10, 10}},
    {false, -7, 3, {200, 10, 10}},
    {false, 10, 3, {200, 10, 10}},
    {false, -36, 3, {7 * 9, 37, 3}},
    {false, 2, 3, {7 * 9, 37, 3}},
    {false, UINT32_MAX + 10L, 3, {7 * 9, 37, 3}},
    {false, -(UINT32_MAX + 10L), 3, {4, 1, 100}},
    // triu
    {true, 0, 3, {7 * 7 * 7 * 7 * 3 * 3, 3, 3}},
    {true, -1, 3, {200, 10, 10}},
    {true, -7, 3, {200, 10, 10}},
    {true, 10, 3, {200, 10, 10}},
    {true, -36, 3, {7 * 9, 37, 3}},
    {true, 2, 3, {7 * 9, 37, 3}},
    {true, UINT32_MAX + 10L, 3, {7 * 9, 37, 3}},
    {true, -(UINT32_MAX + 10L), 3, {4, 1, 100}},
};

// huge cases, for performance test only
/*
const std::vector<TestCase> hugeTestCases = {
    {false, INT32_MAX, 3, {1 << 11, 1  << 10, 1 << 10}},
    {true, -INT32_MAX, 3, {1 << 11, 1  << 10, 1 << 10}},
};

const std::vector<TestCase> boundaryTestCases = {
    {false, 0, 3, {INT32_MAX + 1L, 1, 1}},
};
*/

}  // anonymous namespace

TEST_F(Trilu_CUDA, INVALID_PARAM) {
    TestTrilu<int32_t>(HIEDNN_DATATYPE_INT32, invalidParamTestCases,
        HIEDNN_STATUS_INVALID_PARAMETER);
}

TEST_F(Trilu_CUDA, U8_TINY) {
    TestTrilu<uint8_t>(HIEDNN_DATATYPE_UINT8, tinyTestCases);
}

TEST_F(Trilu_CUDA, U16_TINY) {
    TestTrilu<uint16_t>(HIEDNN_DATATYPE_UINT16, tinyTestCases);
}

TEST_F(Trilu_CUDA, U32_TINY) {
    TestTrilu<uint32_t>(HIEDNN_DATATYPE_UINT32, tinyTestCases);
}

TEST_F(Trilu_CUDA, FP32_TINY) {
    TestTrilu<float>(HIEDNN_DATATYPE_FP32, tinyTestCases);
}

TEST_F(Trilu_CUDA, U8_LARGE) {
    TestTrilu<uint8_t>(HIEDNN_DATATYPE_UINT8, largeTestCases);
}

TEST_F(Trilu_CUDA, U16_LARGE) {
    TestTrilu<uint16_t>(HIEDNN_DATATYPE_UINT16, largeTestCases);
}

TEST_F(Trilu_CUDA, U32_LARGE) {
    TestTrilu<uint32_t>(HIEDNN_DATATYPE_UINT32, largeTestCases);
}

TEST_F(Trilu_CUDA, FP32_LARGE) {
    TestTrilu<float>(HIEDNN_DATATYPE_FP32, largeTestCases);
}

// huge cases, for performance test only
/*
TEST_F(Trilu_CUDA, U8_HUGE) {
    TestTrilu<uint8_t>(HIEDNN_DATATYPE_UINT8, hugeTestCases);
}

TEST_F(Trilu_CUDA, U16_HUGE) {
    TestTrilu<uint16_t>(HIEDNN_DATATYPE_UINT16, hugeTestCases);
}

TEST_F(Trilu_CUDA, FP32_HUGE) {
    TestTrilu<float>(HIEDNN_DATATYPE_FP32, hugeTestCases);
}

TEST_F(Trilu_CUDA, BOUNDARY) {
    TestTrilu<uint8_t>(HIEDNN_DATATYPE_UINT8, boundaryTestCases);
}
*/
