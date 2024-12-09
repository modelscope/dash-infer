/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    non_zero_utest.cu
 */

#include <hiednn.h>
#include <hiednn_cuda.h>

#include <gtest/gtest.h>

#include <iostream>
#include <vector>
#include <functional>
#include <limits>

#include <utest_utils.hpp>

namespace {

constexpr bool PRINT = false;

constexpr int TENSOR_DIM_MAX = 8;

class TestCase {
 public:
    int nDims;
    const int64_t xDims[TENSOR_DIM_MAX];
};

// ======================================================

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

template <typename T>
static T GetElement(const T *x, int nDims,
                    const int64_t (&index)[TENSOR_DIM_MAX],
                    const int64_t (&strides)[TENSOR_DIM_MAX]) {
    size_t offset = 0UL;
    for (int i = 0; i < nDims; ++i) {
        offset += static_cast<size_t>(index[i]) * strides[i];
    }
    return x[offset];
}

template <typename T>
static void SetElement(T *x, T val, int nDims,
                       const int64_t (&index)[TENSOR_DIM_MAX],
                       const int64_t (&strides)[TENSOR_DIM_MAX]) {
    size_t offset = 0UL;
    for (int i = 0; i < nDims; ++i) {
        offset += static_cast<size_t>(index[i]) * strides[i];
    }
    x[offset] = val;
}

template <typename T>
static void CheckTensor(const T *x, const T *y, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        ASSERT_EQ(x[i], y[i]);
    }
}

template <typename T>
void PrintArray(const T* array, int64_t len) {
    std::cout << "[";
    for (int64_t i = 0L; i < len; ++i) {
        if (i > 0L) {
            std::cout << " ";
        }
        std::cout << array[i];
    }
    std::cout << "]";
}

template <typename T>
void PrintTensor(const T* array, const int64_t *dims, int nDims) {
    size_t position = 0UL;
    std::function<void(const int64_t*, int)> printTensorInternal;

    printTensorInternal = [&array, &position, &printTensorInternal](
            const int64_t *dims, int nDims){
        if (nDims == 1) {
            PrintArray<T>(array + position, dims[0]);
            position += static_cast<size_t>(dims[0]);
            return;
        }

        auto dim = dims[0];
        std::cout << "[";
        for (int64_t i = 0L; i < dim; ++i) {
            if (i > 0L) {
                std::cout << std::endl;
            }
            printTensorInternal(dims + 1, nDims - 1);
        }
        std::cout << "]";
        return;
    };

    printTensorInternal(dims, nDims);
    std::cout << std::endl;
}

// ======================================================

class NonZero_CUDA : public testing::Test {
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
    }

    template <bool FAST, typename T, typename IDX_T>
    void InitTestCase(hiednnDataType_t dataType, hiednnDataType_t indexType,
                      const TestCase& tc, bool REF) {
        const auto &nDims = tc.nDims;
        auto xSize = TensorSize(tc.xDims, nDims);

        CHECK_HIEDNN(hiednnSetNormalTensorDesc(
            xDesc, dataType, nDims, tc.xDims));

        ASSERT_LE(xSize * nDims, std::numeric_limits<int64_t>::max());

        if (FAST) {
            const int64_t yDims[2]{nDims, static_cast<int64_t>(xSize)};
            CHECK_HIEDNN(hiednnSetNormalTensorDesc(yDesc, indexType, 2, yDims));
        } else {
            const int64_t yDims[1]{static_cast<int64_t>(xSize * nDims)};
            CHECK_HIEDNN(hiednnSetNormalTensorDesc(yDesc, indexType, 1, yDims));
        }

        CHECK_CUDA(cudaMalloc(&dX, xSize * sizeof(T)));
        CHECK_CUDA(cudaMalloc(&dY, xSize * nDims * sizeof(IDX_T)));
        CHECK_CUDA(cudaMalloc(&dWs, hiednnCudaNonZeroGetWorkspaceSize(xDesc)));
        CHECK_CUDA(cudaMalloc(&dCount, sizeof(size_t)));

        if (REF) {
            CHECK_CUDA(cudaMallocHost(&hX, xSize * sizeof(T)));
            CHECK_CUDA(cudaMallocHost(&hY, xSize * nDims * sizeof(IDX_T)));
            CHECK_CUDA(cudaMallocHost(&hRef, xSize * nDims * sizeof(IDX_T)));

            // gen X, ~1/3 zeros
            T *hX_ptr = static_cast<T *>(hX);
            GenRand<T>(xSize, 0, hX_ptr);
            for (size_t i = 0; i < xSize; ++i) {
                if (static_cast<int64_t>(hX_ptr[i]) % 3 == 0) {
                    hX_ptr[i] = 0;
                }
            }
            CHECK_CUDA(cudaMemcpy(
                dX, hX, xSize * sizeof(T), cudaMemcpyHostToDevice));

            // clear Y
            CHECK_CUDA(cudaMemset(dY, -1, xSize * nDims * sizeof(IDX_T)));

            // PrintTensor(static_cast<IDX_T *>(hX), tc.xDims, nDims);
        }
    }

    void ClearTestCase() {
        CHECK_CUDA(cudaFree(dX));
        CHECK_CUDA(cudaFree(dY));
        CHECK_CUDA(cudaFree(dWs));
        CHECK_CUDA(cudaFree(dCount));

        CHECK_CUDA(cudaFreeHost(hX));
        CHECK_CUDA(cudaFreeHost(hY));
        CHECK_CUDA(cudaFreeHost(hRef));

        hRef = nullptr;
        hX = nullptr;
        hY = nullptr;

        dX = nullptr;
        dY = nullptr;
        dCount = nullptr;
    }

    /* ----------------------------------------------------- */

    template <bool FAST, typename T, typename IDX_T>
    void TestNonZero(hiednnDataType_t dataType,
                     hiednnDataType_t indexType,
                     const std::vector<TestCase>& cases,
                     hiednnStatus_t wantRet = HIEDNN_STATUS_SUCCESS) {
        for (const auto &tc : cases) {
            InitTestCase<FAST, T, IDX_T>(dataType, indexType, tc,
                                         wantRet == HIEDNN_STATUS_SUCCESS);

            if (FAST) {
                ASSERT_EQ(
                    hiednnCudaFastNonZero(handle, xDesc, dX, yDesc, dY, dCount),
                    wantRet);
            } else {
                ASSERT_EQ(
                    hiednnCudaNonZero(handle, xDesc, dX, yDesc, dY, dCount, dWs,
                                      hiednnCudaNonZeroGetWorkspaceSize(xDesc)),
                    wantRet);
            }

            if (wantRet == HIEDNN_STATUS_SUCCESS) {
                T *hX_p = static_cast<T *>(hX);
                IDX_T *hY_p = static_cast<IDX_T *>(hY);
                IDX_T *hRef_p = static_cast<IDX_T *>(hRef);
                size_t hCount = 0xdeadbeaf;

                auto xSize = TensorSize(tc.xDims, tc.nDims);
                CHECK_CUDA(cudaMemcpy(hY, dY, xSize * tc.nDims * sizeof(IDX_T),
                    cudaMemcpyDeviceToHost));
                CHECK_CUDA(cudaMemcpy(&hCount, dCount, sizeof(size_t),
                    cudaMemcpyDeviceToHost));

                if (PRINT) {
                    if (FAST) {
                        const int64_t yDims[2]{tc.nDims,
                            static_cast<int64_t>(xSize)};
                        PrintTensor(hY_p, yDims, 2);
                    } else {
                        const int64_t yDims[2]{tc.nDims,
                            static_cast<int64_t>(hCount)};
                        PrintTensor(hY_p, yDims, 2);
                    }
                }

                auto countRef = NonZero<FAST>(hRef_p, hX_p, tc);

                if (PRINT) {
                    if (FAST) {
                        const int64_t yDims[2]{tc.nDims,
                            static_cast<int64_t>(xSize)};
                        PrintTensor(hRef_p, yDims, 2);
                    } else {
                        const int64_t yDims[2]{tc.nDims,
                            static_cast<int64_t>(hCount)};
                        PrintTensor(hRef_p, yDims, 2);
                    }
                }

                ASSERT_EQ(hCount, countRef);
                CheckTensor(hY_p, hRef_p, xSize * tc.nDims);
            }

            ClearTestCase();
        }
    }

    // -------------- reference --------------

    template <bool FAST, typename T, typename IDX_T>
    static size_t NonZero(IDX_T *y,
                          const T *x,
                          const TestCase& tc) {
        const auto &nDims = tc.nDims;

        const auto xSize = TensorSize(tc.xDims, nDims);

        memset(y, -1, xSize * nDims * sizeof(IDX_T));

        int64_t xStrides[TENSOR_DIM_MAX];
        TensorStride(xStrides, tc.xDims, nDims);

        size_t count = 0;
        if (FAST) {
            for (size_t i = 0; i < xSize; ++i) {
                if (x[i] != 0) {
                    size_t offset = i;
                    for (int j = 0; j < nDims; ++j) {
                        y[j * xSize + count] = offset / xStrides[j];
                        offset = offset % xStrides[j];
                    }
                    count++;
                }
            }
        } else {
            for (size_t i = 0; i < xSize; ++i) {
                if (x[i] != 0) {
                    count++;
                }
            }
            size_t write_offset = 0;
            for (size_t i = 0; i < xSize; ++i) {
                if (x[i] != 0) {
                    size_t offset = i;
                    for (int j = 0; j < nDims; j++) {
                        y[j * count + write_offset] = offset / xStrides[j];
                        offset = offset % xStrides[j];
                    }
                    write_offset++;
                }
            }
        }
        return count;
    }

    // -------------- members --------------

    hiednnCudaHandle_t handle;
    hiednnTensorDesc_t xDesc;
    hiednnTensorDesc_t yDesc;

    void *hX = nullptr;
    void *hY = nullptr;
    void *hRef = nullptr;

    void *dX = nullptr;
    void *dY = nullptr;
    void *dWs = nullptr;
    size_t *dCount = nullptr;
};

/* ============= test cases ============= */

const std::vector<TestCase> testCases = {
    {1, {3}},
    {2, {2, 2}},
    {3, {3, 4, 5}},
    {4, {3, 5, 7, 9}},
    {5, {3, 5, 7, 9, 5}},
    {6, {3, 5, 7, 9, 7, 5}},
    {7, {3, 5, 7, 9, 5, 4, 2}},
    {8, {3, 5, 7, 9, 4, 5, 4, 2}},
    {5, {16, 7, 16, 7, 16}},
    {6, {16, 7, 16, 7, 16, 7}}
};

}  // anonymous namespace

// ----------------------------

TEST_F(NonZero_CUDA, U8_U32) {
    TestNonZero<false, uint8_t, uint32_t>(
        HIEDNN_DATATYPE_UINT8, HIEDNN_DATATYPE_UINT32, testCases);
}

TEST_F(NonZero_CUDA, U8_U32_FAST) {
    TestNonZero<true, uint8_t, uint32_t>(
        HIEDNN_DATATYPE_UINT8, HIEDNN_DATATYPE_UINT32, testCases);
}

#ifdef HIEDNN_USE_FP16
TEST_F(NonZero_CUDA, FP16_U64) {
    TestNonZero<false, half_t, uint64_t>(
        HIEDNN_DATATYPE_FP16, HIEDNN_DATATYPE_UINT64, testCases);
}

TEST_F(NonZero_CUDA, FP16_U64_FAST) {
    TestNonZero<true, half_t, uint64_t>(
        HIEDNN_DATATYPE_FP16, HIEDNN_DATATYPE_UINT64, testCases);
}
#endif  // HIEDNN_USE_FP16

#ifdef HIEDNN_USE_BF16
TEST_F(NonZero_CUDA, BF16_U64) {
    TestNonZero<false, bf16_t, uint64_t>(
        HIEDNN_DATATYPE_BF16, HIEDNN_DATATYPE_UINT64, testCases);
}

TEST_F(NonZero_CUDA, BF16_U64_FAST) {
    TestNonZero<true, bf16_t, uint64_t>(
        HIEDNN_DATATYPE_BF16, HIEDNN_DATATYPE_UINT64, testCases);
}
#endif  // HIEDNN_USE_BF16

TEST_F(NonZero_CUDA, I32_U32) {
    TestNonZero<false, int32_t, uint32_t>(
        HIEDNN_DATATYPE_INT32, HIEDNN_DATATYPE_UINT32, testCases);
}

TEST_F(NonZero_CUDA, I32_U32_FAST) {
    TestNonZero<true, int32_t, uint32_t>(
        HIEDNN_DATATYPE_INT32, HIEDNN_DATATYPE_UINT32, testCases);
}

TEST_F(NonZero_CUDA, FP32_U32) {
    TestNonZero<false, float, uint32_t>(
        HIEDNN_DATATYPE_FP32, HIEDNN_DATATYPE_UINT32, testCases);
}

TEST_F(NonZero_CUDA, FP32_U64_FAST) {
    TestNonZero<true, float, uint64_t>(
        HIEDNN_DATATYPE_FP32, HIEDNN_DATATYPE_UINT64, testCases);
}

