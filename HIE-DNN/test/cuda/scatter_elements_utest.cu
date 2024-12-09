/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    scatter_elements_utest.cu
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
    }
};

template<typename IDX_T>
class TestCase {
 public:
    hiednnScatterElemReduce_t op;
    int axis;
    int nDims;
    const int64_t xDims[TENSOR_DIM_MAX];
    const int64_t updatesDims[TENSOR_DIM_MAX];
    const IndicesCase<IDX_T> indices;
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

void GetIndex(int64_t *index, size_t offset,
              const int64_t *strides, int nDims) {
    for (int i = 0; i < nDims - 1; ++i) {
        index[i] = static_cast<int64_t>(
            offset / static_cast<size_t>(strides[i]));
        offset = offset % static_cast<size_t>(strides[i]);
    }
    // tail
    index[nDims - 1] = static_cast<int64_t>(
        offset / static_cast<size_t>(strides[nDims - 1]));
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

class ScatterElements_CUDA : public testing::Test {
 protected:
    void SetUp() override {
        CHECK_HIEDNN(hiednnCreateCudaHandle(&handle));
        CHECK_HIEDNN(hiednnCreateTensorDesc(&xDesc));
        CHECK_HIEDNN(hiednnCreateTensorDesc(&yDesc));
        CHECK_HIEDNN(hiednnCreateTensorDesc(&indicesDesc));
        CHECK_HIEDNN(hiednnCreateTensorDesc(&updatesDesc));
    }

    void TearDown() override {
        CHECK_HIEDNN(hiednnDestroyTensorDesc(xDesc));
        CHECK_HIEDNN(hiednnDestroyTensorDesc(yDesc));
        CHECK_HIEDNN(hiednnDestroyTensorDesc(indicesDesc));
        CHECK_HIEDNN(hiednnDestroyTensorDesc(updatesDesc));
        CHECK_HIEDNN(hiednnDestroyCudaHandle(handle));
    }

    template <typename T, typename IDX_T>
    void InitTestCase(hiednnDataType_t dataType, hiednnDataType_t indexType,
                      const TestCase<IDX_T>& tc, bool REF,
                      bool RAND_IDX = true) {
        const auto &nDims = tc.nDims;
        CHECK_HIEDNN(hiednnSetNormalTensorDesc(
            xDesc, dataType, nDims, tc.xDims));
        CHECK_HIEDNN(hiednnSetNormalTensorDesc(
            yDesc, dataType, nDims, tc.xDims));
        CHECK_HIEDNN(hiednnSetNormalTensorDesc(
            indicesDesc, indexType, nDims, tc.updatesDims));
        CHECK_HIEDNN(hiednnSetNormalTensorDesc(
            updatesDesc, dataType, nDims, tc.updatesDims));

        auto xSize = TensorSize(tc.xDims, nDims);
        auto updatesSize = TensorSize(tc.updatesDims, nDims);

        CHECK_CUDA(cudaMalloc(&dX, xSize * sizeof(T)));
        CHECK_CUDA(cudaMalloc(&dY, xSize * sizeof(T)));
        CHECK_CUDA(cudaMalloc(&dIndices, updatesSize * sizeof(IDX_T)));
        CHECK_CUDA(cudaMalloc(&dUpdates, updatesSize * sizeof(T)));

        if (REF) {
            CHECK_CUDA(cudaMallocHost(&hX, xSize * sizeof(T)));
            CHECK_CUDA(cudaMallocHost(&hY, xSize * sizeof(T)));
            CHECK_CUDA(cudaMallocHost(&hIndices, updatesSize * sizeof(IDX_T)));
            CHECK_CUDA(cudaMallocHost(&hUpdates, updatesSize * sizeof(T)));
            CHECK_CUDA(cudaMallocHost(&hRef, xSize * sizeof(T)));

            // init updates, starting from 1
            T *hUpdates_p = static_cast<T *>(hUpdates);
            for (size_t i = 0UL; i < updatesSize; ++i) {
                hUpdates_p[i] = static_cast<T>(i + 1);
            }
            CHECK_CUDA(cudaMemcpy(dUpdates, hUpdates, updatesSize * sizeof(T),
                                  cudaMemcpyHostToDevice));

            // set X
            memset(hX, 0, xSize * sizeof(T));
            CHECK_CUDA(cudaMemset(dX, 0, xSize * sizeof(T)));

            // clear Y
            CHECK_CUDA(cudaMemset(dY, 0, xSize * sizeof(T)));

            // init indices
            InitIndices(indexType, tc, RAND_IDX,
                        tc.op == HIEDNN_SCATTERELEM_REDUCE_NONE);

            // PrintTensor(static_cast<IDX_T *>(hIndices),
            //             tc.updatesDims, nDims);
            // PrintTensor(static_cast<T *>(hUpdates), tc.updatesDims, nDims);
        }
    }

    void ClearTestCase() {
        CHECK_CUDA(cudaFree(dX));
        CHECK_CUDA(cudaFree(dY));
        CHECK_CUDA(cudaFree(dIndices));
        CHECK_CUDA(cudaFree(dUpdates));

        CHECK_CUDA(cudaFreeHost(hX));
        CHECK_CUDA(cudaFreeHost(hY));
        CHECK_CUDA(cudaFreeHost(hIndices));
        CHECK_CUDA(cudaFreeHost(hUpdates));
        CHECK_CUDA(cudaFreeHost(hRef));

        hRef = nullptr;
        hIndices = nullptr;
        hUpdates = nullptr;
        hX = nullptr;
        hY = nullptr;

        dX = nullptr;
        dY = nullptr;
        dIndices = nullptr;
        dUpdates = nullptr;
    }

    /* ----------------------------------------------------- */

    template <typename T, typename IDX_T>
    void TestScatterElements(hiednnDataType_t dataType,
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
                hiednnCudaScatterElements(handle, xDesc, dX,
                                          indicesDesc, dIndices,
                                          updatesDesc, dUpdates,
                                          yDesc, dY, tc.axis, tc.op),
                wantRet);
            if (wantRet == HIEDNN_STATUS_SUCCESS) {
                T *hX_p = static_cast<T *>(hX);
                T *hY_p = static_cast<T *>(hY);
                IDX_T *hIndices_p = static_cast<IDX_T *>(hIndices);
                T *hUpdates_p = static_cast<T *>(hUpdates);
                T *hRef_p = static_cast<T *>(hRef);

                auto xSize = TensorSize(tc.xDims, tc.nDims);
                CHECK_CUDA(cudaMemcpy(hY, dY, xSize * sizeof(T),
                    cudaMemcpyDeviceToHost));
                // PrintTensor(hY_p, tc.xDims, tc.nDims);

                ScatterElements(hRef_p, hX_p, hIndices_p, hUpdates_p, tc);
                // PrintTensor(hRef_p, tc.xDims, tc.nDims);
                CheckTensor(hY_p, hRef_p, xSize);
            }

            ClearTestCase();
        }
    }

    // -------------- helper --------------

    template <typename IDX_T>
    void InitIndices(hiednnDataType_t indexType, const TestCase<IDX_T>& tc,
                     bool random = true, bool noDup = false) {
        const auto &nDims = tc.nDims;
        auto updatesSize = TensorSize(tc.updatesDims, nDims);

        if (tc.indices.len > 0U) {
            ASSERT_NE(tc.indices.ptr, nullptr);
            ASSERT_EQ(tc.indices.len, updatesSize);
            CHECK_CUDA(cudaMemcpy(
                hIndices, tc.indices.ptr, updatesSize * sizeof(IDX_T),
                cudaMemcpyHostToHost));
        } else if (!noDup && random) {
            // randomly init legal indices: [-s, s-1]
            uint32_t seed = 0;
            ASSERT_GE(tc.axis, -nDims);
            ASSERT_LT(tc.axis, nDims);
            const auto axis = tc.axis < 0 ? tc.axis + nDims : tc.axis;
            const auto &s = tc.xDims[axis];

            IDX_T *hIndices_p = static_cast<IDX_T *>(hIndices);
            for (size_t offset = 0UL; offset < updatesSize; ++offset) {
                hIndices_p[offset] = static_cast<int64_t>(rand_r(&seed))
                                        % (2 * s) - s;
            }
        } else {
            // generate indices so that scatter elements behaves as memcpy
            ASSERT_GE(tc.axis, -nDims);
            ASSERT_LT(tc.axis, nDims);
            const auto axis = tc.axis < 0 ? tc.axis + nDims : tc.axis;
            int64_t updatesStrides[TENSOR_DIM_MAX];
            TensorStride(updatesStrides, tc.updatesDims, nDims);

            IDX_T *hIndices_p = static_cast<IDX_T *>(hIndices);
            for (size_t offset = 0UL; offset < updatesSize; ++offset) {
                int64_t myIndex[TENSOR_DIM_MAX];
                GetIndex(myIndex, offset, updatesStrides, nDims);
                hIndices_p[offset] = static_cast<IDX_T>(myIndex[axis]);
            }
        }
        CHECK_CUDA(cudaMemcpy(
            dIndices, hIndices, updatesSize * sizeof(IDX_T),
            cudaMemcpyHostToDevice));
    }

    // -------------- reference --------------

    template <typename T, typename IDX_T>
    static void ScatterElements(T *y,
                               const T *x,
                               const IDX_T *indices,
                               const T *updates,
                               const TestCase<IDX_T>& tc) {
        ASSERT_GE(tc.nDims, 1);
        ASSERT_GE(tc.axis, -tc.nDims);
        ASSERT_LT(tc.axis, tc.nDims);

        const auto &nDims = tc.nDims;
        const auto axis = tc.axis < 0 ? tc.axis + nDims : tc.axis;

        for (int i = 0; i < nDims; ++i) {
            // for dims other than `axis', cannot index nonexistent indices
            if (i != axis) {
                ASSERT_LE(tc.updatesDims[i], tc.xDims[i]);
            }
        }

        const auto xSize = TensorSize(tc.xDims, nDims);
        const auto updatesSize = TensorSize(tc.updatesDims, nDims);

        memcpy(y, x, xSize * sizeof(T));

        int64_t yStrides[TENSOR_DIM_MAX];
        int64_t updatesStrides[TENSOR_DIM_MAX];
        TensorStride(yStrides, tc.xDims, nDims);
        TensorStride(updatesStrides, tc.updatesDims, nDims);

        const auto &s = tc.xDims[axis];
        for (size_t offset = 0UL; offset < updatesSize; ++offset) {
            int64_t index[TENSOR_DIM_MAX];
            GetIndex(index, offset, updatesStrides, nDims);
            // range: [-s, s-1]
            ASSERT_GE(indices[offset], -s);
            ASSERT_LT(indices[offset], s);
            index[axis] = indices[offset] < 0 ?
                          (indices[offset] + s) : indices[offset];
            T res = updates[offset];
            switch (tc.op) {
            case HIEDNN_SCATTERELEM_REDUCE_NONE:
                break;
            case HIEDNN_SCATTERELEM_REDUCE_ADD:
                res = res + GetElement(y, nDims, index, yStrides);
                break;
            case HIEDNN_SCATTERELEM_REDUCE_MUL:
                res = res * GetElement(y, nDims, index, yStrides);
                break;
            case HIEDNN_SCATTERELEM_REDUCE_MAX:
                res = std::max(res, GetElement(y, nDims, index, yStrides));
                break;
            case HIEDNN_SCATTERELEM_REDUCE_MIN:
                res = std::min(res, GetElement(y, nDims, index, yStrides));
                break;
            default:
                FAIL() << "Bad reduction op: " << tc.op;
            }
            SetElement(y, res, nDims, index, yStrides);
        }
    }

    // -------------- members --------------

    hiednnCudaHandle_t handle;
    hiednnTensorDesc_t xDesc;
    hiednnTensorDesc_t yDesc;
    hiednnTensorDesc_t indicesDesc;
    hiednnTensorDesc_t updatesDesc;

    void *hX = nullptr;
    void *hY = nullptr;
    void *hIndices = nullptr;
    void *hUpdates = nullptr;
    void *hRef = nullptr;

    void *dX = nullptr;
    void *dY = nullptr;
    void *dIndices = nullptr;
    void *dUpdates = nullptr;
};

/* ============= test cases ============= */

const std::vector<TestCase<int32_t>> invalidParamTestCases = {
    // bad axis
    {HIEDNN_SCATTERELEM_REDUCE_NONE, 3, 2, {3, 4}, {2, 2}},
    // nonexistent index
    {HIEDNN_SCATTERELEM_REDUCE_NONE, 0, 2, {3, 4}, {1, 5}},
};

const std::vector<TestCase<float>> invalidTypeTestCase = {
    {HIEDNN_SCATTERELEM_REDUCE_NONE, 0, 1, {3}, {2}},
};

const std::vector<TestCase<int32_t>> indicesTestCases = {
    // output: [[-1, 5, 3], [1, -1, -1], [4, 2, 6]]
    {HIEDNN_SCATTERELEM_REDUCE_NONE, 0, 2, {3, 3}, {2, 3},
        {2 * 3, {1, 2, 0, 2, 0, 2}}},
    /*
    // output: [[-1, 4, 2], [0, -1, -1], [3, 1, 5]]
    {HIEDNN_SCATTERELEM_REDUCE_ADD, 0, 2, {3, 3}, {2, 3},
        {2 * 3, {1, 2, 0, 2, 0, 2}}},
    // output: [[-1, -5, -3], [-1, -1, -1], [-4, -2, -6]]
    {HIEDNN_SCATTERELEM_REDUCE_MUL, 0, 2, {3, 3}, {2, 3},
        {2 * 3, {1, 2, 0, 2, 0, 2}}},
    // output: [[-1, 5, 3], [1, -1, -1], [4, 2, 6]]
    {HIEDNN_SCATTERELEM_REDUCE_MAX, 0, 2, {3, 3}, {2, 3},
        {2 * 3, {1, 2, 0, 2, 0, 2}}},
    // output: [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]]
    {HIEDNN_SCATTERELEM_REDUCE_MIN, 0, 2, {3, 3}, {2, 3},
        {2 * 3, {1, 2, 0, 2, 0, 2}}},
    */
};

const std::vector<TestCase<int32_t>> tinyTestCases_32 = {
    {HIEDNN_SCATTERELEM_REDUCE_NONE, 0, 1, {3}, {2}},
    {HIEDNN_SCATTERELEM_REDUCE_NONE, 0, 2, {3, 4}, {2, 2}},
    {HIEDNN_SCATTERELEM_REDUCE_NONE, 1, 2, {3, 4}, {2, 2}},
    {HIEDNN_SCATTERELEM_REDUCE_NONE, 1, 3, {3, 4, 5}, {2, 2, 3}},
    {HIEDNN_SCATTERELEM_REDUCE_NONE, 0, 4, {3, 5, 7, 9}, {2, 2, 1, 3}},
    {HIEDNN_SCATTERELEM_REDUCE_NONE, -2, 5,
        {3, 5, 7, 9, 5}, {2, 2, 1, 3, 3}},
    {HIEDNN_SCATTERELEM_REDUCE_NONE, -2, 6,
        {3, 5, 7, 9, 7, 5}, {2, 2, 1, 3, 3, 3}},
    {HIEDNN_SCATTERELEM_REDUCE_NONE, -1, 7,
        {3, 5, 7, 9, 5, 4, 2}, {2, 2, 1, 3, 3, 1, 2}},
    {HIEDNN_SCATTERELEM_REDUCE_NONE, -1, 8,
        {3, 5, 7, 9, 7, 5, 4, 2}, {2, 2, 1, 3, 3, 3, 1, 2}},
};

const std::vector<TestCase<int64_t>> tinyTestCases_64 = {
    {HIEDNN_SCATTERELEM_REDUCE_NONE, 0, 1, {3}, {2}},
    {HIEDNN_SCATTERELEM_REDUCE_NONE, 0, 2, {3, 4}, {2, 2}},
    {HIEDNN_SCATTERELEM_REDUCE_NONE, 1, 2, {3, 4}, {2, 2}},
    {HIEDNN_SCATTERELEM_REDUCE_NONE, 1, 3, {3, 4, 5}, {2, 2, 3}},
    {HIEDNN_SCATTERELEM_REDUCE_NONE, 0, 4, {3, 5, 7, 9}, {2, 2, 1, 3}},
    {HIEDNN_SCATTERELEM_REDUCE_NONE, -2, 5,
        {3, 5, 7, 9, 5}, {2, 2, 1, 3, 3}},
    {HIEDNN_SCATTERELEM_REDUCE_NONE, -2, 6,
        {3, 5, 7, 9, 7, 5}, {2, 2, 1, 3, 3, 3}},
    {HIEDNN_SCATTERELEM_REDUCE_NONE, -1, 7,
        {3, 5, 7, 9, 5, 4, 2}, {2, 2, 1, 3, 3, 1, 2}},
    {HIEDNN_SCATTERELEM_REDUCE_NONE, -1, 8,
        {3, 5, 7, 9, 7, 5, 4, 2}, {2, 2, 1, 3, 3, 3, 1, 2}},
};

const std::vector<TestCase<int32_t>> largeTestCases_32 = {
    {HIEDNN_SCATTERELEM_REDUCE_NONE, 0, 5,
        {16, 16, 16, 16, 16}, {7, 7, 7, 7, 7}},
    {HIEDNN_SCATTERELEM_REDUCE_NONE, -3, 6,
        {16, 16, 16, 16, 16, 16}, {7, 7, 7, 7, 7, 7}},
    // {HIEDNN_SCATTERELEM_REDUCE_NONE, 0, 7,
    //     {16, 16, 16, 16, 16, 16, 13}, {7, 7, 7, 7, 7, 7, 7}},
};

const std::vector<TestCase<int64_t>> largeTestCases_64 = {
    {HIEDNN_SCATTERELEM_REDUCE_NONE, 0, 5,
        {16, 16, 16, 16, 16}, {7, 7, 7, 7, 7}},
    {HIEDNN_SCATTERELEM_REDUCE_NONE, -3, 6,
        {16, 16, 16, 16, 16, 16}, {7, 7, 7, 7, 7, 7}},
    // {HIEDNN_SCATTERELEM_REDUCE_NONE, 0, 7,
    //     {16, 16, 16, 16, 16, 16, 13}, {7, 7, 7, 7, 7, 7, 7}},
};

// huge cases, for performance test only
/*
const std::vector<TestCase<int32_t>> hugeTestCases_32 = {
    {HIEDNN_SCATTERELEM_REDUCE_NONE, 0, 2,
        {8 * 8 * 16 * 16 * 16 * 16 * 11, 11},
        {8 * 8 * 16 * 16 * 16 * 16 * 11, 11}},
    {HIEDNN_SCATTERELEM_REDUCE_NONE, 0, 4,
        {8 * 8 * 16 * 16 * 16, 16, 11, 11},
        {8 * 8 * 16 * 16 * 16, 16, 11, 11}},
    {HIEDNN_SCATTERELEM_REDUCE_NONE, 0, 6,
        {8 * 8 * 16, 16, 16, 16, 11, 11},
        {8 * 8 * 16, 16, 16, 16, 11, 11}},
    {HIEDNN_SCATTERELEM_REDUCE_NONE, 0, 7,
        {8 * 8, 16, 16, 16, 16, 11, 11},
        {8 * 8, 16, 16, 16, 16, 11, 11}},
    {HIEDNN_SCATTERELEM_REDUCE_NONE, 0, 8,
        {8, 8, 16, 16, 16, 16, 11, 11},
        {8, 8, 16, 16, 16, 16, 11, 11}},
};

const std::vector<TestCase<int64_t>> hugeTestCases_64 = {
    {HIEDNN_SCATTERELEM_REDUCE_NONE, 0, 2,
        {8 * 8 * 16 * 16 * 16 * 16 * 11, 11},
        {8 * 8 * 16 * 16 * 16 * 16 * 11, 11}},
    {HIEDNN_SCATTERELEM_REDUCE_NONE, 0, 4,
        {8 * 8 * 16 * 16 * 16, 16, 11, 11},
        {8 * 8 * 16 * 16 * 16, 16, 11, 11}},
    {HIEDNN_SCATTERELEM_REDUCE_NONE, 0, 6,
        {8 * 8 * 16, 16, 16, 16, 11, 11},
        {8 * 8 * 16, 16, 16, 16, 11, 11}},
    {HIEDNN_SCATTERELEM_REDUCE_NONE, 0, 7,
        {8 * 8, 16, 16, 16, 16, 11, 11},
        {8 * 8, 16, 16, 16, 16, 11, 11}},
    {HIEDNN_SCATTERELEM_REDUCE_NONE, 0, 8,
        {8, 8, 16, 16, 16, 16, 11, 11},
        {8, 8, 16, 16, 16, 16, 11, 11}},
};
*/

}  // anonymous namespace

TEST_F(ScatterElements_CUDA, INVALID_PARAM) {
    TestScatterElements<float, int32_t>(
        HIEDNN_DATATYPE_FP32, HIEDNN_DATATYPE_INT32,
        invalidParamTestCases, HIEDNN_STATUS_INVALID_PARAMETER);
}

TEST_F(ScatterElements_CUDA, INVALID_DATATYPE) {
    TestScatterElements<float, float>(
        HIEDNN_DATATYPE_FP32, HIEDNN_DATATYPE_FP32,
        invalidTypeTestCase, HIEDNN_STATUS_INVALID_DATATYPE);
}

// ----------------------------

TEST_F(ScatterElements_CUDA, USER_INDICES) {
    TestScatterElements<int32_t, int32_t>(
        HIEDNN_DATATYPE_INT32, HIEDNN_DATATYPE_INT32, indicesTestCases);
}

// ----------------------------

TEST_F(ScatterElements_CUDA, U8_I32_TINY) {
    TestScatterElements<uint8_t, int32_t>(
        HIEDNN_DATATYPE_UINT8, HIEDNN_DATATYPE_INT32, tinyTestCases_32);
}

TEST_F(ScatterElements_CUDA, U16_I32_TINY) {
    TestScatterElements<uint16_t, int32_t>(
        HIEDNN_DATATYPE_UINT16, HIEDNN_DATATYPE_INT32, tinyTestCases_32);
}

TEST_F(ScatterElements_CUDA, U32_I32_TINY) {
    TestScatterElements<uint32_t, int32_t>(
        HIEDNN_DATATYPE_UINT32, HIEDNN_DATATYPE_INT32, tinyTestCases_32);
}

TEST_F(ScatterElements_CUDA, FP32_I64_TINY) {
    TestScatterElements<float, int64_t>(
        HIEDNN_DATATYPE_FP32, HIEDNN_DATATYPE_INT64, tinyTestCases_64);
}

TEST_F(ScatterElements_CUDA, FP64_I64_TINY) {
    TestScatterElements<double, int64_t>(
        HIEDNN_DATATYPE_FP64, HIEDNN_DATATYPE_INT64, tinyTestCases_64);
}

// ----------------------------

TEST_F(ScatterElements_CUDA, U8_I32_LARGE) {
    TestScatterElements<uint8_t, int32_t>(
        HIEDNN_DATATYPE_UINT8, HIEDNN_DATATYPE_INT32, largeTestCases_32);
}

TEST_F(ScatterElements_CUDA, U16_I32_LARGE) {
    TestScatterElements<uint16_t, int32_t>(
        HIEDNN_DATATYPE_UINT16, HIEDNN_DATATYPE_INT32, largeTestCases_32);
}

TEST_F(ScatterElements_CUDA, FP32_I32_LARGE) {
    TestScatterElements<float, int32_t>(
        HIEDNN_DATATYPE_FP32, HIEDNN_DATATYPE_INT32, largeTestCases_32);
}

TEST_F(ScatterElements_CUDA, U8_I64_LARGE) {
    TestScatterElements<uint8_t, int64_t>(
        HIEDNN_DATATYPE_UINT8, HIEDNN_DATATYPE_INT64, largeTestCases_64);
}

TEST_F(ScatterElements_CUDA, U16_I64_LARGE) {
    TestScatterElements<uint16_t, int64_t>(
        HIEDNN_DATATYPE_UINT16, HIEDNN_DATATYPE_INT64, largeTestCases_64);
}

TEST_F(ScatterElements_CUDA, FP32_I64_LARGE) {
    TestScatterElements<float, int64_t>(
        HIEDNN_DATATYPE_FP32, HIEDNN_DATATYPE_INT64, largeTestCases_64);
}

// huge cases, for performance test only
/*
TEST_F(ScatterElements_CUDA, U8_I32_HUGE) {
    TestScatterElements<uint8_t, int32_t>(
        HIEDNN_DATATYPE_UINT8, HIEDNN_DATATYPE_INT32, hugeTestCases_32,
        HIEDNN_STATUS_SUCCESS, false, false);
}

TEST_F(ScatterElements_CUDA, U16_I32_HUGE) {
    TestScatterElements<uint16_t, int32_t>(
        HIEDNN_DATATYPE_UINT16, HIEDNN_DATATYPE_INT32, hugeTestCases_32,
        HIEDNN_STATUS_SUCCESS, false, false);
}

TEST_F(ScatterElements_CUDA, FP32_I32_HUGE) {
    TestScatterElements<float, int32_t>(
        HIEDNN_DATATYPE_FP32, HIEDNN_DATATYPE_INT32, hugeTestCases_32,
        HIEDNN_STATUS_SUCCESS, false, false);
}

TEST_F(ScatterElements_CUDA, U8_I64_HUGE) {
    TestScatterElements<uint8_t, int64_t>(
        HIEDNN_DATATYPE_UINT8, HIEDNN_DATATYPE_INT64, hugeTestCases_64,
        HIEDNN_STATUS_SUCCESS, false, false);
}

TEST_F(ScatterElements_CUDA, U16_I64_HUGE) {
    TestScatterElements<uint16_t, int64_t>(
        HIEDNN_DATATYPE_UINT16, HIEDNN_DATATYPE_INT64, hugeTestCases_64,
        HIEDNN_STATUS_SUCCESS, false, false);
}

TEST_F(ScatterElements_CUDA, FP32_I64_HUGE) {
    TestScatterElements<float, int64_t>(
        HIEDNN_DATATYPE_FP32, HIEDNN_DATATYPE_INT64, hugeTestCases_64,
        HIEDNN_STATUS_SUCCESS, false, false);
}
*/
