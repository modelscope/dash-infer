/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    scatter_nd_utest.cu
 */
#include <hiednn.h>
#include <hiednn_cuda.h>

#include <gtest/gtest.h>
#include <cstdint>
#include <cstring>

#include <utest_utils.hpp>

namespace {

struct TestCase {
    int dataNDims;
    int64_t dataDims[8];
    int indicesNDims;
    int64_t indicesDims[8];
    int updatesNDims;
    int64_t updatesDims[8];
};

const std::vector<TestCase> testCase = {
    {3, {1, 262144, 64}, 3, {1, 30000, 2}, 3, {1, 30000, 64}},
    {3, {1, 262144, 64}, 3, {1, 29987, 2}, 3, {1, 29987, 64}},
    {3, {1, 155555, 62}, 3, {1, 29987, 2}, 3, {1, 29987, 62}},
    {3, {1, 155555, 15}, 3, {1, 29987, 2}, 3, {1, 29987, 15}},
    {3, {1, 9999, 3}, 3, {1, 9999 * 3, 3}, 2, {1, 9999 * 3}},
    {3, {1, 16, 22}, 3, {1, 13, 2}, 3, {1, 13, 22}},
};

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

struct ScatterNDTest {
    template <typename IndexT>
    static void InitIndices(IndexT *indices,
                            int64_t indicesSize,
                            int tuple,
                            const int64_t *dataDims,
                            uint32_t *seed) {
        int64_t nTuples = indicesSize / tuple;
        int64_t offsetRange = TensorSize(dataDims, tuple);
        int64_t offsetStep = offsetRange / nTuples;

        int64_t offset = 0;

        for (int64_t i = 0; i < nTuples; ++i) {
            offset += (rand_r(seed) % offsetStep) + 1;
            int64_t div = offset;
            for (int j = tuple - 1; j >= 0; --j) {
                indices[j] = div % dataDims[j];
                div /= dataDims[j];
            }

            indices += tuple;
        }
    }

    template <typename DT>
    static void InitUpdates(DT *updates, int64_t size, uint32_t *seed) {
        for (int i = 0; i < size; ++i) {
            updates[i] = static_cast<DT>(rand_r(seed));
        }
    }

    template <typename DT, typename IndexT>
    static void CheckOutput(const DT *data,
                            const IndexT *indices,
                            const DT *updates,
                            const DT *output,
                            const TestCase &tc) {
        int64_t dataSize = TensorSize(tc.dataDims, tc.dataNDims);
        int64_t indicesSize = TensorSize(tc.indicesDims, tc.indicesNDims);
        int64_t updatesSize = TensorSize(tc.updatesDims, tc.updatesNDims);

        int tuple = tc.indicesDims[tc.indicesNDims - 1];
        int64_t nTuples = indicesSize / tuple;
        int64_t innerSize = updatesSize / nTuples;

        DT *stdOutput = static_cast<DT *>(malloc(dataSize * sizeof(DT)));
        memcpy(stdOutput, data, dataSize * sizeof(DT));

        int64_t dataStrides[8];
        TensorStride(dataStrides, tc.dataDims, tc.dataNDims);

        for (int64_t i = 0; i < nTuples; ++i) {
            int64_t offset = 0;
            for (int j = 0; j < tuple; ++j) {
                offset += indices[j] * dataStrides[j];
            }
            memcpy(stdOutput + offset, updates, innerSize * sizeof(DT));

            indices += tuple;
            updates += innerSize;
        }

        for (int64_t i = 0; i < dataSize; ++i) {
            ASSERT_EQ(stdOutput[i], output[i]);
        }

        free(stdOutput);
    }
};

}  // anonymous namespace

#define UTEST_SCATTERND(TEST_NAME, DT, HIEDNN_DT, INDEX_T, HIEDNN_INDICES_T) \
TEST(ScatterND_CUDA, TEST_NAME) { \
    hiednnCudaHandle_t handle; \
    CHECK_HIEDNN(hiednnCreateCudaHandle(&handle)); \
    \
    hiednnTensorDesc_t dataDesc; \
    hiednnTensorDesc_t indicesDesc; \
    hiednnTensorDesc_t updatesDesc; \
    CHECK_HIEDNN(hiednnCreateTensorDesc(&dataDesc)); \
    CHECK_HIEDNN(hiednnCreateTensorDesc(&indicesDesc)); \
    CHECK_HIEDNN(hiednnCreateTensorDesc(&updatesDesc)); \
    \
    for (const auto &tc : testCase) { \
        CHECK_HIEDNN(hiednnSetNormalTensorDesc( \
            dataDesc, HIEDNN_DT, tc.dataNDims, tc.dataDims)); \
        CHECK_HIEDNN(hiednnSetNormalTensorDesc( \
            indicesDesc, HIEDNN_INDICES_T, tc.indicesNDims, tc.indicesDims)); \
        CHECK_HIEDNN(hiednnSetNormalTensorDesc( \
            updatesDesc, HIEDNN_DT, tc.updatesNDims, tc.updatesDims)); \
        \
        int64_t dataSize = TensorSize(tc.dataDims, tc.dataNDims); \
        int64_t indicesSize = TensorSize(tc.indicesDims, tc.indicesNDims); \
        int64_t updatesSize = TensorSize(tc.updatesDims, tc.updatesNDims); \
        \
        DT *hData; \
        INDEX_T *hIndices; \
        DT *hUpdates; \
        DT *hOutput; \
        CHECK_CUDA(cudaMallocHost(&hData, dataSize * sizeof(DT))); \
        CHECK_CUDA(cudaMallocHost(&hIndices, indicesSize * sizeof(INDEX_T))); \
        CHECK_CUDA(cudaMallocHost(&hUpdates, updatesSize * sizeof(DT))); \
        CHECK_CUDA(cudaMallocHost(&hOutput, dataSize * sizeof(DT))); \
        \
        uint32_t seed = 0; \
        memset(hData, 0, dataSize * sizeof(DT)); \
        ScatterNDTest::InitIndices(hIndices, \
                                   indicesSize, \
                                   tc.indicesDims[tc.indicesNDims - 1], \
                                   tc.dataDims, \
                                   &seed); \
        ScatterNDTest::InitUpdates(hUpdates, updatesSize, &seed); \
        \
        DT *dData; \
        INDEX_T *dIndices; \
        DT *dUpdates; \
        DT *dOutput; \
        CHECK_CUDA(cudaMalloc(&dData, dataSize * sizeof(DT))); \
        CHECK_CUDA(cudaMalloc(&dIndices, indicesSize * sizeof(INDEX_T))); \
        CHECK_CUDA(cudaMalloc(&dUpdates, updatesSize * sizeof(DT))); \
        CHECK_CUDA(cudaMalloc(&dOutput, dataSize * sizeof(DT))); \
        \
        CHECK_CUDA(cudaMemcpy(dData, hData, \
            dataSize * sizeof(DT), cudaMemcpyHostToDevice)); \
        CHECK_CUDA(cudaMemcpy(dIndices, hIndices, \
            indicesSize * sizeof(INDEX_T), cudaMemcpyHostToDevice)); \
        CHECK_CUDA(cudaMemcpy(dUpdates, hUpdates, \
            updatesSize * sizeof(DT), cudaMemcpyHostToDevice)); \
        \
        CHECK_HIEDNN(hiednnCudaScatterND( \
            handle, dataDesc, dData, indicesDesc, dIndices, \
            updatesDesc, dUpdates, dataDesc, dOutput)); \
        \
        CHECK_CUDA(cudaMemcpy(hOutput, dOutput, \
            dataSize * sizeof(DT), cudaMemcpyDeviceToHost)); \
        ScatterNDTest::CheckOutput(hData, hIndices, hUpdates, hOutput, tc); \
        \
        CHECK_CUDA(cudaFree(dData)); \
        CHECK_CUDA(cudaFree(dIndices)); \
        CHECK_CUDA(cudaFree(dUpdates)); \
        CHECK_CUDA(cudaFree(dOutput)); \
        \
        CHECK_CUDA(cudaFreeHost(hData)); \
        CHECK_CUDA(cudaFreeHost(hIndices)); \
        CHECK_CUDA(cudaFreeHost(hUpdates)); \
        CHECK_CUDA(cudaFreeHost(hOutput)); \
    } \
    \
    CHECK_HIEDNN(hiednnDestroyTensorDesc(dataDesc)); \
    CHECK_HIEDNN(hiednnDestroyTensorDesc(indicesDesc)); \
    CHECK_HIEDNN(hiednnDestroyTensorDesc(updatesDesc)); \
    CHECK_HIEDNN(hiednnDestroyCudaHandle(handle)); \
}

UTEST_SCATTERND(S32_DATA_S32_INDICES,
                int32_t, HIEDNN_DATATYPE_INT32,
                int32_t, HIEDNN_DATATYPE_INT32)
UTEST_SCATTERND(S16_DATA_S32_INDICES,
                int16_t, HIEDNN_DATATYPE_INT16,
                int32_t, HIEDNN_DATATYPE_INT32)
UTEST_SCATTERND(S8_DATA_S32_INDICES,
                int8_t, HIEDNN_DATATYPE_INT8,
                int32_t, HIEDNN_DATATYPE_INT32)
UTEST_SCATTERND(S8_DATA_S64_INDICES,
                int8_t, HIEDNN_DATATYPE_INT8,
                int64_t, HIEDNN_DATATYPE_INT64)
UTEST_SCATTERND(S8_DATA_U32_INDICES,
                int8_t, HIEDNN_DATATYPE_INT8,
                uint32_t, HIEDNN_DATATYPE_UINT32)


