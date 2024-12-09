/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    gather_elements_utest.cu
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

// for randomly generated indices
template<typename IDX_T>
class IndicesCase {
 public:
    const size_t len;
    const void * const ptr;
    IndicesCase() : len(0UL), ptr(nullptr) {}
};

// allowing user-specified indices input
template<>
class IndicesCase<int32_t> {
 public:
    size_t len;
    int32_t *ptr;

    IndicesCase(const IndicesCase<int32_t> &old) : len(old.len), ptr(nullptr) {
        if (old.ptr) {
            ptr = new int32_t[len];
            memcpy(ptr, old.ptr, len * sizeof(int32_t));
        }
    }
    IndicesCase(IndicesCase<int32_t> &&old) : len(old.len), ptr(nullptr) {
        if (old.ptr) {
            ptr = old.ptr;
            old.ptr = nullptr;
            old.len = 0;
        }
    }
    IndicesCase& operator=(const IndicesCase<int32_t> &) = delete;
    IndicesCase& operator=(IndicesCase<int32_t> &&) = delete;

    IndicesCase() : len(0UL), ptr(nullptr) {}
    IndicesCase(size_t len, const std::vector<int32_t> &vals) : len(len) {
        ptr = new int32_t[len];
        std::copy(vals.cbegin(), vals.cend(), ptr);
    }

    ~IndicesCase() {
        if (ptr) {
            delete[] ptr;
        }
        ptr = nullptr;
        len = 0UL;
    }
};

template<typename IDX_T>
class TestCase {
 public:
    int axis;
    int nDims;
    const int64_t xDims[TENSOR_DIM_MAX];
    const int64_t yDims[TENSOR_DIM_MAX];
    const IndicesCase<IDX_T> indices;
};

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
            PrintArray<T>(array + position, dims[0]);
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

class GatherElements_CUDA : public testing::Test {
 protected:
    void SetUp() override {
        CHECK_HIEDNN(hiednnCreateCudaHandle(&handle));
        CHECK_HIEDNN(hiednnCreateTensorDesc(&xDesc));
        CHECK_HIEDNN(hiednnCreateTensorDesc(&yDesc));
        CHECK_HIEDNN(hiednnCreateTensorDesc(&indicesDesc));
    }

    void TearDown() override {
        CHECK_HIEDNN(hiednnDestroyTensorDesc(xDesc));
        CHECK_HIEDNN(hiednnDestroyTensorDesc(yDesc));
        CHECK_HIEDNN(hiednnDestroyTensorDesc(indicesDesc));
        CHECK_HIEDNN(hiednnDestroyCudaHandle(handle));
        xDesc = nullptr;
        yDesc = nullptr;
        indicesDesc = nullptr;
        handle = nullptr;
    }

    template <typename T, typename IDX_T>
    void InitTestCase(hiednnDataType_t dataType, hiednnDataType_t indexType,
                      const TestCase<IDX_T>& tc, bool REF,
                      bool RAND_IDX = true) {
        const auto &nDims = tc.nDims;
        CHECK_HIEDNN(
            hiednnSetNormalTensorDesc(xDesc, dataType, nDims, tc.xDims));
        CHECK_HIEDNN(
            hiednnSetNormalTensorDesc(yDesc, dataType, nDims, tc.yDims));
        CHECK_HIEDNN(
            hiednnSetNormalTensorDesc(indicesDesc, indexType, nDims, tc.yDims));

        auto xSize = TensorSize(tc.xDims, nDims);
        auto ySize = TensorSize(tc.yDims, nDims);

        CHECK_CUDA(cudaMallocHost(&hX, xSize * sizeof(T)));
        CHECK_CUDA(cudaMallocHost(&hY, ySize * sizeof(T)));
        CHECK_CUDA(cudaMallocHost(&hIndices, ySize * sizeof(IDX_T)));

        CHECK_CUDA(cudaMalloc(&dX, xSize * sizeof(T)));
        CHECK_CUDA(cudaMalloc(&dY, ySize * sizeof(T)));
        CHECK_CUDA(cudaMalloc(&dIndices, ySize * sizeof(IDX_T)));

        if (REF) {
            CHECK_CUDA(cudaMallocHost(&hRef, ySize * sizeof(T)));

            // init X, starting from 1
            T *hX_p = static_cast<T *>(hX);
            for (size_t i = 0UL; i < xSize; i++) {
                hX_p[i] = static_cast<T>(i + 1);
            }
            // PrintTensor(static_cast<T *>(hX), tc.xDims, nDims);
            CHECK_CUDA(cudaMemcpy(
                dX, hX, xSize * sizeof(T), cudaMemcpyHostToDevice));

            // clear Y
            CHECK_CUDA(cudaMemset(dY, 0, ySize * sizeof(T)));

            // init indices
            InitIndices(indexType, tc, RAND_IDX);
        }
    }

    void ClearTestCase() {
        CHECK_CUDA(cudaFree(dX));
        CHECK_CUDA(cudaFree(dY));
        CHECK_CUDA(cudaFree(dIndices));
        CHECK_CUDA(cudaFreeHost(hX));
        CHECK_CUDA(cudaFreeHost(hY));
        CHECK_CUDA(cudaFreeHost(hIndices));
        CHECK_CUDA(cudaFreeHost(hRef));
        hRef = nullptr;
        hIndices = nullptr;
        hX = nullptr;
        hY = nullptr;
        dX = nullptr;
        dY = nullptr;
        dIndices = nullptr;
    }

    /* ----------------------------------------------------- */

    template <typename T, typename IDX_T>
    void TestGatherElements(hiednnDataType_t dataType,
                            hiednnDataType_t indexType,
                            const std::vector<TestCase<IDX_T> >& cases,
                            hiednnStatus_t wantRet = HIEDNN_STATUS_SUCCESS,
                            bool forceInitIndices = false,
                            bool randomIndices = true) {
        for (const auto &tc : cases) {
            InitTestCase<T, IDX_T>(dataType, indexType, tc,
                                   wantRet == HIEDNN_STATUS_SUCCESS,
                                   randomIndices);

            if (forceInitIndices) {
                InitIndices(indexType, tc);
            }

            ASSERT_EQ(
                hiednnCudaGatherElements(handle, xDesc, dX,
                                         indicesDesc, dIndices,
                                         yDesc, dY, tc.axis),
                wantRet);
            if (wantRet == HIEDNN_STATUS_SUCCESS) {
                T *hX_p = static_cast<T *>(hX);
                T *hY_p = static_cast<T *>(hY);
                IDX_T *hIndices_p = static_cast<IDX_T *>(hIndices);
                T *hRef_p = static_cast<T *>(hRef);

                auto ySize = TensorSize(tc.yDims, tc.nDims);
                CHECK_CUDA(cudaMemcpy(hY, dY, ySize * sizeof(T),
                    cudaMemcpyDeviceToHost));
                // PrintTensor(hY_p, tc.yDims, tc.nDims);

                GatherElements(hRef_p, hX_p, hIndices_p, tc);
                // PrintTensor(hRef_p, tc.yDims, tc.nDims);
                CheckTensor(hY_p, hRef_p, ySize);
            }

            ClearTestCase();
        }
    }

    // -------------- helper --------------

    template <typename IDX_T>
    void InitIndices(hiednnDataType_t indexType, const TestCase<IDX_T>& tc,
                     bool random = true) {
        const auto &nDims = tc.nDims;
        auto ySize = TensorSize(tc.yDims, nDims);

        if (tc.indices.len > 0U) {
            ASSERT_NE(tc.indices.ptr, nullptr);
            ASSERT_EQ(tc.indices.len, ySize);
            CHECK_CUDA(cudaMemcpy(
                hIndices, tc.indices.ptr, ySize * sizeof(IDX_T),
                cudaMemcpyHostToHost));
        } else if (random) {
            // randomly init legal indices: [-s, s-1]
            uint32_t seed = 0;
            ASSERT_GE(tc.axis, -nDims);
            ASSERT_LT(tc.axis, nDims);
            const auto axis = tc.axis < 0 ? tc.axis + nDims : tc.axis;
            const auto &s = tc.xDims[axis];

            IDX_T *hIndices_p = static_cast<IDX_T *>(hIndices);
            for (size_t offset = 0UL; offset < ySize; offset++) {
                hIndices_p[offset] = static_cast<int64_t>(rand_r(&seed))
                                        % (2 * s) - s;
            }
        } else {
            // generate indices so that gather elements behaves as memcpy
            ASSERT_GE(tc.axis, -nDims);
            ASSERT_LT(tc.axis, nDims);
            const auto axis = tc.axis < 0 ? tc.axis + nDims : tc.axis;
            int64_t yStrides[TENSOR_DIM_MAX];
            TensorStride(yStrides, tc.yDims, nDims);

            IDX_T *hIndices_p = static_cast<IDX_T *>(hIndices);
            for (size_t offset = 0UL; offset < ySize; offset++) {
                int64_t myIndex[TENSOR_DIM_MAX];
                GetIndex(myIndex, offset, yStrides, nDims);
                hIndices_p[offset] = static_cast<IDX_T>(myIndex[axis]);
            }
        }
        // PrintTensor(static_cast<IDX_T *>(hIndices), tc.yDims, nDims);
        CHECK_CUDA(cudaMemcpy(
            dIndices, hIndices, ySize * sizeof(IDX_T),
            cudaMemcpyHostToDevice));
    }

    // -------------- reference --------------

    template <typename T, typename IDX_T>
    static void GatherElements(T *y,
                               const T *x,
                               const IDX_T *indices,
                               const TestCase<IDX_T>& tc) {
        ASSERT_GE(tc.nDims, 1);
        ASSERT_GE(tc.axis, -tc.nDims);
        ASSERT_LT(tc.axis, tc.nDims);

        const auto &nDims = tc.nDims;
        const auto axis = tc.axis < 0 ? tc.axis + nDims : tc.axis;

        for (int i = 0; i < nDims; i++) {
            // for dims other than `axis', cannot index nonexistent indices
            if (i != axis) {
                ASSERT_LE(tc.yDims[i], tc.xDims[i]);
            }
        }

        const auto ySize = TensorSize(tc.yDims, nDims);
        int64_t xStrides[TENSOR_DIM_MAX];
        int64_t yStrides[TENSOR_DIM_MAX];
        TensorStride(xStrides, tc.xDims, nDims);
        TensorStride(yStrides, tc.yDims, nDims);

        const auto &s = tc.xDims[axis];
        for (size_t offset = 0UL; offset < ySize; offset++) {
            int64_t index[TENSOR_DIM_MAX];
            GetIndex(index, offset, yStrides, nDims);
            // range: [-s, s-1]
            ASSERT_GE(indices[offset], -s);
            ASSERT_LT(indices[offset], s);
            index[axis] = indices[offset] < 0 ?
                          (indices[offset] + s) : indices[offset];
            y[offset] = GetElement(x, nDims, index, xStrides);
        }
    }

    // -------------- members --------------

    hiednnCudaHandle_t handle = nullptr;
    hiednnTensorDesc_t xDesc = nullptr;
    hiednnTensorDesc_t yDesc = nullptr;
    hiednnTensorDesc_t indicesDesc = nullptr;

    void *hX = nullptr;
    void *hY = nullptr;
    void *hIndices = nullptr;
    void *hRef = nullptr;
    void *dX = nullptr;
    void *dY = nullptr;
    void *dIndices = nullptr;
};

/* ============= test cases ============= */

const std::vector<TestCase<int32_t>> tinyTestCases_32 = {
    {0, 1, {3}, {2}},
    {0, 2, {3, 4}, {2, 2}},
    {1, 2, {3, 4}, {2, 2}},
    {1, 2, {3, 4}, {2, 10}},
    {1, 3, {3, 4, 5}, {2, 10, 3}},
    {0, 4, {3, 5, 7, 9}, {2, 2, 1, 3}},
    {-2, 5, {3, 5, 7, 9, 5}, {2, 2, 1, 3, 3}},
    {-2, 6, {3, 5, 7, 9, 7, 5}, {2, 2, 1, 3, 3, 3}},
    {-1, 7, {3, 5, 7, 9, 5, 4, 2}, {2, 2, 1, 3, 3, 1, 2}},
    {-1, 8, {3, 5, 7, 9, 7, 5, 4, 2}, {2, 2, 1, 3, 3, 3, 1, 2}},
};

const std::vector<TestCase<int64_t>> tinyTestCases_64 = {
    {0, 1, {3}, {2}},
    {0, 2, {3, 4}, {2, 2}},
    {1, 2, {3, 4}, {2, 2}},
    {1, 2, {3, 4}, {2, 10}},
    {1, 3, {3, 4, 5}, {2, 10, 3}},
    {0, 4, {3, 5, 7, 9}, {2, 2, 1, 3}},
    {-2, 5, {3, 5, 7, 9, 5}, {2, 2, 1, 3, 3}},
    {-2, 6, {3, 5, 7, 9, 7, 5}, {2, 2, 1, 3, 3, 3}},
    {-1, 7, {3, 5, 7, 9, 5, 4, 2}, {2, 2, 1, 3, 3, 1, 2}},
    {-1, 8, {3, 5, 7, 9, 7, 5, 4, 2}, {2, 2, 1, 3, 3, 3, 1, 2}},
};

const std::vector<TestCase<int32_t>> largeTestCases_32 = {
    {0, 5, {16, 16, 16, 16, 16}, {7, 7, 7, 7, 7}},
    {-3, 6, {16, 16, 16, 16, 16, 16}, {7, 7, 7, 7, 7, 7}},
    {0, 7, {16, 16, 16, 16, 16, 16, 13}, {7, 7, 7, 7, 7, 7, 7}},
};

const std::vector<TestCase<int64_t>> largeTestCases_64 = {
    {0, 5, {16, 16, 16, 16, 16}, {7, 7, 7, 7, 7}},
    {-3, 6, {16, 16, 16, 16, 16, 16}, {7, 7, 7, 7, 7, 7}},
    {0, 7, {16, 16, 16, 16, 16, 16, 13}, {7, 7, 7, 7, 7, 7, 7}},
};

const std::vector<TestCase<int32_t>> indicesTestCases = {
    // output: [[4, 8, 3], [7, 2, 3]]
    {0, 2, {3, 3}, {2, 3}, {2 * 3, {1, 2, 0, 2, 0, 0}}},
    // output: [[1, 4], [8, 6]]
    {1, 2, {2, 4}, {2, 2}, {2 * 2, {0, 3, 3, 1}}},
    // output: [[1, 6], [5, 4]]
    {0, 2, {3, 2}, {2, 2}, {2 * 2, {0, 2, 2, 1}}},
    // output: [[1, 6], [5, 4]]
    {0, 2, {3, 2}, {2, 2}, {2 * 2, {-3, -1, -1, -2}}},
};

const std::vector<TestCase<int32_t>> invalidParamTestCases = {
    // bad axis
    {3, 2, {3, 4}, {2, 2}},
    // nonexistent index
    {0, 2, {3, 4}, {1, 5}},
};

const std::vector<TestCase<float>> invalidTypeTestCase = {
    {0, 1, {3}, {2}},
};

// huge cases, for performance test only
/*
const std::vector<TestCase<int32_t>> hugeTestCases_32 = {
    {0, 2, {8 * 8 * 16 * 16 * 16 * 16 * 11, 11}, {8 * 8 * 16 * 16 * 16 * 16 * 11, 11}},
    {0, 4, {8 * 8 * 16 * 16 * 16, 16, 11, 11}, {8 * 8 * 16 * 16 * 16, 16, 11, 11}},
    {0, 6, {8 * 8 * 16, 16, 16, 16, 11, 11}, {8 * 8 * 16, 16, 16, 16, 11, 11}},
    {0, 7, {8 * 8, 16, 16, 16, 16, 11, 11}, {8 * 8, 16, 16, 16, 16, 11, 11}},
    {0, 8, {8, 8, 16, 16, 16, 16, 11, 11}, {8, 8, 16, 16, 16, 16, 11, 11}},
};

const std::vector<TestCase<int64_t>> hugeTestCases_64 = {
    {0, 2, {8 * 8 * 16 * 16 * 16 * 16 * 11, 11}, {8 * 8 * 16 * 16 * 16 * 16 * 11, 11}},
    {0, 4, {8 * 8 * 16 * 16 * 16, 16, 11, 11}, {8 * 8 * 16 * 16 * 16, 16, 11, 11}},
    {0, 6, {8 * 8 * 16, 16, 16, 16, 11, 11}, {8 * 8 * 16, 16, 16, 16, 11, 11}},
    {0, 7, {8 * 8, 16, 16, 16, 16, 11, 11}, {8 * 8, 16, 16, 16, 16, 11, 11}},
    {0, 8, {8, 8, 16, 16, 16, 16, 11, 11}, {8, 8, 16, 16, 16, 16, 11, 11}},
};

const std::vector<TestCase<int64_t>> boundaryCases_64 = {
    {0, 1, {UINT32_MAX - 10UL}, {10}},
    {0, 1, {UINT32_MAX}, {10}},
    {0, 1, {UINT32_MAX + 10UL}, {10}},
};
*/

}  // anonymous namespace

TEST_F(GatherElements_CUDA, INVALID_PARAM) {
    TestGatherElements<float, int32_t>(
        HIEDNN_DATATYPE_FP32, HIEDNN_DATATYPE_INT32,
        invalidParamTestCases, HIEDNN_STATUS_INVALID_PARAMETER);
}

TEST_F(GatherElements_CUDA, INVALID_DATATYPE) {
    TestGatherElements<float, float>(
        HIEDNN_DATATYPE_FP32, HIEDNN_DATATYPE_FP32,
        invalidTypeTestCase, HIEDNN_STATUS_INVALID_DATATYPE);
}

// ----------------------------

TEST_F(GatherElements_CUDA, USER_INDICES) {
    TestGatherElements<int32_t, int32_t>(
        HIEDNN_DATATYPE_INT32, HIEDNN_DATATYPE_INT32, indicesTestCases);
}

// ----------------------------

TEST_F(GatherElements_CUDA, U8_I32_TINY) {
    TestGatherElements<uint8_t, int32_t>(
        HIEDNN_DATATYPE_UINT8, HIEDNN_DATATYPE_INT32, tinyTestCases_32);
}

TEST_F(GatherElements_CUDA, U16_I32_TINY) {
    TestGatherElements<uint16_t, int32_t>(
        HIEDNN_DATATYPE_UINT16, HIEDNN_DATATYPE_INT32, tinyTestCases_32);
}

TEST_F(GatherElements_CUDA, U32_I32_TINY) {
    TestGatherElements<uint32_t, int32_t>(
        HIEDNN_DATATYPE_UINT32, HIEDNN_DATATYPE_INT32, tinyTestCases_32);
}

TEST_F(GatherElements_CUDA, FP32_I64_TINY) {
    TestGatherElements<float, int64_t>(
        HIEDNN_DATATYPE_FP32, HIEDNN_DATATYPE_INT64, tinyTestCases_64);
}

TEST_F(GatherElements_CUDA, FP64_I64_TINY) {
    TestGatherElements<double, int64_t>(
        HIEDNN_DATATYPE_FP64, HIEDNN_DATATYPE_INT64, tinyTestCases_64);
}

// ----------------------------

TEST_F(GatherElements_CUDA, U8_I32_LARGE) {
    TestGatherElements<uint8_t, int32_t>(
        HIEDNN_DATATYPE_UINT8, HIEDNN_DATATYPE_INT32, largeTestCases_32);
}

TEST_F(GatherElements_CUDA, U16_I32_LARGE) {
    TestGatherElements<uint16_t, int32_t>(
        HIEDNN_DATATYPE_UINT16, HIEDNN_DATATYPE_INT32, largeTestCases_32);
}

TEST_F(GatherElements_CUDA, FP32_I32_LARGE) {
    TestGatherElements<float, int32_t>(
        HIEDNN_DATATYPE_FP32, HIEDNN_DATATYPE_INT32, largeTestCases_32);
}

TEST_F(GatherElements_CUDA, U8_I64_LARGE) {
    TestGatherElements<uint8_t, int64_t>(
        HIEDNN_DATATYPE_UINT8, HIEDNN_DATATYPE_INT64, largeTestCases_64);
}

TEST_F(GatherElements_CUDA, U16_I64_LARGE) {
    TestGatherElements<uint16_t, int64_t>(
        HIEDNN_DATATYPE_UINT16, HIEDNN_DATATYPE_INT64, largeTestCases_64);
}

TEST_F(GatherElements_CUDA, FP32_I64_LARGE) {
    TestGatherElements<float, int64_t>(
        HIEDNN_DATATYPE_FP32, HIEDNN_DATATYPE_INT64, largeTestCases_64);
}

// huge cases, for performance test only
/*
TEST_F(GatherElements_CUDA, U8_I32_HUGE) {
    TestGatherElements<uint8_t, int32_t>(
        HIEDNN_DATATYPE_UINT8, HIEDNN_DATATYPE_INT32, hugeTestCases_32,
        HIEDNN_STATUS_SUCCESS, false, false);
}

TEST_F(GatherElements_CUDA, U16_I32_HUGE) {
    TestGatherElements<uint16_t, int32_t>(
        HIEDNN_DATATYPE_UINT16, HIEDNN_DATATYPE_INT32, hugeTestCases_32,
        HIEDNN_STATUS_SUCCESS, false, false);
}

TEST_F(GatherElements_CUDA, FP32_I32_HUGE) {
    TestGatherElements<float, int32_t>(
        HIEDNN_DATATYPE_FP32, HIEDNN_DATATYPE_INT32, hugeTestCases_32,
        HIEDNN_STATUS_SUCCESS, false, false);
}

TEST_F(GatherElements_CUDA, U8_I64_HUGE) {
    TestGatherElements<uint8_t, int64_t>(
        HIEDNN_DATATYPE_UINT8, HIEDNN_DATATYPE_INT64, hugeTestCases_64,
        HIEDNN_STATUS_SUCCESS, false, false);
}

TEST_F(GatherElements_CUDA, U16_I64_HUGE) {
    TestGatherElements<uint16_t, int64_t>(
        HIEDNN_DATATYPE_UINT16, HIEDNN_DATATYPE_INT64, hugeTestCases_64,
        HIEDNN_STATUS_SUCCESS, false, false);
}

TEST_F(GatherElements_CUDA, FP32_I64_HUGE) {
    TestGatherElements<float, int64_t>(
        HIEDNN_DATATYPE_FP32, HIEDNN_DATATYPE_INT64, hugeTestCases_64,
        HIEDNN_STATUS_SUCCESS, false, false);
}

TEST_F(GatherElements_CUDA, BOUNDARY) {
    TestGatherElements<uint8_t, int64_t>(
        HIEDNN_DATATYPE_UINT8, HIEDNN_DATATYPE_INT64, boundaryCases_64);
}
*/
