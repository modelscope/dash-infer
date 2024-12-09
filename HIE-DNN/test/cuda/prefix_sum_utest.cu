/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    prefix_sum_utest.cu
 */
#include <hiednn.h>
#include <hiednn_cuda.h>

#include <gtest/gtest.h>
#include <cstdint>
#include <cmath>
#include <vector>

#include <utest_utils.hpp>

namespace {

struct TestCase {
    int ndims;
    int64_t dim[5];
    int axis;
};

int64_t GetTensorSize(const TestCase &tc) {
    int64_t size = 1;
    for (int i = 0; i < tc.ndims; ++i) {
        size *= tc.dim[i];
    }
    return size;
}

template <typename T>
void TensorInit(T *data, int64_t size) {
    for (int i = 0; i < size; ++i) {
        data[i] = static_cast<T>(i);
    }
}

const std::vector<TestCase> testCaseD0 = {
    // unified 3D
    {3, {555, 7, 3}, 1},
    {3, {555, 3, 7}, 1},

    // unified 2D, serial inter-thread scan
    {3, {111, 43, 111}, 1},

    // unified 2D, parallel inter-thread scan
    {3, {223, 333, 3}, 1},

    // unified 3D, serial inter-thread scan
    {3, {111, 55, 3}, 1},

    // unified 3D, parallel inter-thread scan
    {3, {333, 111, 3}, 1},

    // tile 2D, serial inter-thread scan
    {3, {3, 555, 1111}, 1},

    // tile 2D, parallel inter-thread scan
    {3, {3, 667, 3}, 1},

    {3, {3, 3333, 999}, 1},
};

template <int EXCLUSIVE,
          int REVERSE,
          typename CompT,
          typename DataT>
void CheckScanD0(const DataT *x, const DataT *y,
                 int batch, int64_t m, int64_t n) {
    for (int64_t i = 0; i < batch; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            int64_t offset = REVERSE != 0 ?
                             i * m * n + (m - 1) * n + j :
                             i * m * n + j;
            const DataT *xp = x + offset;
            const DataT *yp = y + offset;
            CompT sum = 0;

            for (int64_t k = 0; k < m; ++k) {
                if (EXCLUSIVE != 0) {
                    CheckEq(static_cast<DataT>(sum), *yp);
                    sum += static_cast<CompT>(*xp);
                } else {
                    sum += static_cast<CompT>(*xp);
                    CheckEq(static_cast<DataT>(sum), *yp);
                }

                if (REVERSE != 0) {
                    xp -= n;
                    yp -= n;
                } else {
                    xp += n;
                    yp += n;
                }
            }
        }
    }
}

}  // anonymous namespace

#define PREFIX_SUM_D0_UTEST(NAME_DTYPE, \
                            DTYPE, \
                            HIEDNN_DataTYPE, \
                            CompT, \
                            NAME_EXCLUSIVE, \
                            INT_EXCLUSIVE, \
                            NAME_REVERSE, \
                            INT_REVERSE) \
TEST(PrefixSumD0, NAME_DTYPE##NAME_EXCLUSIVE##NAME_REVERSE) { \
    hiednnCudaHandle_t handle; \
    CHECK_HIEDNN(hiednnCreateCudaHandle(&handle)); \
    \
    hiednnTensorDesc_t tensor_desc; \
    CHECK_HIEDNN(hiednnCreateTensorDesc(&tensor_desc)); \
    \
    for (int i = 0; i < testCaseD0.size(); ++i) { \
        const auto &tc = testCaseD0[i]; \
        \
        int64_t tensor_size = GetTensorSize(tc); \
        DTYPE *d_x, *d_y; \
        DTYPE *h_x, *h_y; \
        CHECK_CUDA(cudaMallocHost(&h_x, tensor_size * sizeof(DTYPE))); \
        CHECK_CUDA(cudaMallocHost(&h_y, tensor_size * sizeof(DTYPE))); \
        TensorInit(h_x, tensor_size); \
        \
        CHECK_CUDA(cudaMalloc(&d_x, tensor_size * sizeof(DTYPE))); \
        CHECK_CUDA(cudaMalloc(&d_y, tensor_size * sizeof(DTYPE))); \
        CHECK_CUDA(cudaMemcpy(d_x, h_x, tensor_size * sizeof(DTYPE), \
                              cudaMemcpyHostToDevice)); \
        \
        CHECK_HIEDNN(hiednnSetNormalTensorDesc( \
            tensor_desc, HIEDNN_DataTYPE, tc.ndims, tc.dim)); \
        CHECK_HIEDNN(hiednnCudaPrefixSum(handle, \
                                         tensor_desc, \
                                         d_x, \
                                         tc.axis, \
                                         INT_EXCLUSIVE, \
                                         INT_REVERSE, \
                                         tensor_desc, \
                                         d_y)); \
        CHECK_CUDA(cudaMemcpy(h_y, d_y, tensor_size * sizeof(DTYPE), \
                              cudaMemcpyDeviceToHost)); \
        int64_t batch = 1; \
        int64_t m = tc.dim[tc.axis]; \
        int64_t n = 1; \
        \
        for (int i = 0; i < tc.axis; ++i) { \
            batch *= tc.dim[i]; \
        } \
        for (int i = tc.axis + 1; i < tc.ndims; ++i) { \
            n *= tc.dim[i]; \
        } \
        \
        CheckScanD0<INT_EXCLUSIVE, INT_REVERSE, CompT, DTYPE>( \
            h_x, h_y, batch, m, n); \
        \
        CHECK_CUDA(cudaFree(d_x)); \
        CHECK_CUDA(cudaFree(d_y)); \
        CHECK_CUDA(cudaFreeHost(h_x)); \
        CHECK_CUDA(cudaFreeHost(h_y)); \
    } \
    CHECK_HIEDNN(hiednnDestroyTensorDesc(tensor_desc)); \
    CHECK_HIEDNN(hiednnDestroyCudaHandle(handle)); \
}

// -----------------------------------------------
// inclusive prefix scan
// -----------------------------------------------
PREFIX_SUM_D0_UTEST(Float, float, HIEDNN_DATATYPE_FP32,
                    double,
                    Inclusive, 0,
                    Prefix, 0);

PREFIX_SUM_D0_UTEST(Int16, int16_t, HIEDNN_DATATYPE_INT16,
                    int32_t,
                    Inclusive, 0,
                    Prefix, 0);

PREFIX_SUM_D0_UTEST(Int32, int32_t, HIEDNN_DATATYPE_INT32,
                    int32_t,
                    Inclusive, 0,
                    Prefix, 0);

PREFIX_SUM_D0_UTEST(Int64, int64_t, HIEDNN_DATATYPE_INT64,
                    int64_t,
                    Inclusive, 0,
                    Prefix, 0);

// -----------------------------------------------
// exclusive prefix scan
// -----------------------------------------------
PREFIX_SUM_D0_UTEST(Float, float, HIEDNN_DATATYPE_FP32,
                    double,
                    Exclusive, 1,
                    Prefix, 0);

PREFIX_SUM_D0_UTEST(Int16, int16_t, HIEDNN_DATATYPE_INT16,
                    int32_t,
                    Exclusive, 1,
                    Prefix, 0);

PREFIX_SUM_D0_UTEST(Int32, int32_t, HIEDNN_DATATYPE_INT32,
                    int32_t,
                    Exclusive, 1,
                    Prefix, 0);

PREFIX_SUM_D0_UTEST(Int64, int64_t, HIEDNN_DATATYPE_INT64,
                    int64_t,
                    Exclusive, 1,
                    Prefix, 0);

// -----------------------------------------------
// inclusive suffix scan
// -----------------------------------------------
PREFIX_SUM_D0_UTEST(Float, float, HIEDNN_DATATYPE_FP32,
                    double,
                    Inclusive, 0,
                    Suffix, 1);

PREFIX_SUM_D0_UTEST(Int16, int16_t, HIEDNN_DATATYPE_INT16,
                    int32_t,
                    Inclusive, 0,
                    Suffix, 1);

PREFIX_SUM_D0_UTEST(Int32, int32_t, HIEDNN_DATATYPE_INT32,
                    int32_t,
                    Inclusive, 0,
                    Suffix, 1);

PREFIX_SUM_D0_UTEST(Int64, int64_t, HIEDNN_DATATYPE_INT64,
                    int64_t,
                    Inclusive, 0,
                    Suffix, 1);

// -----------------------------------------------
// exclusive suffix scan
// -----------------------------------------------
PREFIX_SUM_D0_UTEST(Float, float, HIEDNN_DATATYPE_FP32,
                    double,
                    Exclusive, 1,
                    Suffix, 1);

PREFIX_SUM_D0_UTEST(Int16, int16_t, HIEDNN_DATATYPE_INT16,
                    int32_t,
                    Exclusive, 1,
                    Suffix, 1);

PREFIX_SUM_D0_UTEST(Int32, int32_t, HIEDNN_DATATYPE_INT32,
                    int32_t,
                    Exclusive, 1,
                    Suffix, 1);

PREFIX_SUM_D0_UTEST(Int64, int64_t, HIEDNN_DATATYPE_INT64,
                    int64_t,
                    Exclusive, 1,
                    Suffix, 1);

namespace {

const std::vector<TestCase> testCaseD1 = {
    // Unified1D
    {2, {111, 33333}, 1},

    // Unified2D, inside-thread
    {4, {33, 57, 11, 7}, 3},

    // Unified2D, inter-thread
    {4, {33, 57, 11, 77}, 3},

    // Tiled1D
    {1, {23333}, 0},

    // Tiled2D
    {2, {3, 11111}, 1},
};

template <int EXCLUSIVE,
          int REVERSE,
          typename CompT,
          typename DataT>
void CheckScanD1(const DataT *x, const DataT *y, int64_t m, int64_t n) {
    for (int64_t i = 0; i < m; ++i) {
        int64_t offset = REVERSE != 0 ? i * n + n - 1 : i * n;
        const DataT *xp = x + offset;
        const DataT *yp = y + offset;

        CompT sum = 0;
        for (int64_t j = 0; j < n; ++j) {
            if (EXCLUSIVE != 0) {
                CheckEq(static_cast<DataT>(sum), *yp);
                sum += static_cast<CompT>(*xp);
            } else {
                sum += static_cast<CompT>(*xp);
                CheckEq(static_cast<DataT>(sum), *yp);
            }

            if (REVERSE != 0) {
                --xp;
                --yp;
            } else {
                ++xp;
                ++yp;
            }
        }
    }
}

}  // anonymous namespace

#define PREFIX_SUM_D1_UTEST(NAME_DTYPE, \
                            DTYPE, \
                            HIEDNN_DATA_TYPE, \
                            COMP_T, \
                            NAME_EXCLUSIVE, \
                            INT_EXCLUSIVE, \
                            NAME_REVERSE, \
                            INT_REVERSE) \
TEST(PrefixSumD1, NAME_DTYPE##NAME_EXCLUSIVE##NAME_REVERSE) { \
    hiednnCudaHandle_t handle; \
    CHECK_HIEDNN(hiednnCreateCudaHandle(&handle)); \
    \
    hiednnTensorDesc_t tensor_desc; \
    CHECK_HIEDNN(hiednnCreateTensorDesc(&tensor_desc)); \
    \
    for (int i = 0; i < testCaseD1.size(); ++i) { \
        const auto &tc = testCaseD1[i]; \
        \
        int64_t tensor_size = GetTensorSize(tc); \
        DTYPE *d_x, *d_y; \
        DTYPE *h_x, *h_y; \
        CHECK_CUDA(cudaMallocHost(&h_x, tensor_size * sizeof(DTYPE))); \
        CHECK_CUDA(cudaMallocHost(&h_y, tensor_size * sizeof(DTYPE))); \
        TensorInit(h_x, tensor_size); \
        \
        CHECK_CUDA(cudaMalloc(&d_x, tensor_size * sizeof(DTYPE))); \
        CHECK_CUDA(cudaMalloc(&d_y, tensor_size * sizeof(DTYPE))); \
        CHECK_CUDA(cudaMemcpy(d_x, h_x, tensor_size * sizeof(DTYPE), \
                              cudaMemcpyHostToDevice)); \
        \
        CHECK_HIEDNN(hiednnSetNormalTensorDesc( \
            tensor_desc, HIEDNN_DATA_TYPE, tc.ndims, tc.dim)); \
        CHECK_HIEDNN(hiednnCudaPrefixSum(handle, \
                                         tensor_desc, \
                                         d_x, \
                                         tc.axis, \
                                         INT_EXCLUSIVE, \
                                         INT_REVERSE, \
                                         tensor_desc, \
                                         d_y)); \
        CHECK_CUDA(cudaMemcpy(h_y, d_y, tensor_size * sizeof(DTYPE), \
                              cudaMemcpyDeviceToHost)); \
        CheckScanD1<INT_EXCLUSIVE, INT_REVERSE, COMP_T, DTYPE>( \
            h_x, h_y, tensor_size / tc.dim[tc.ndims - 1], \
            tc.dim[tc.ndims - 1]); \
        \
        CHECK_CUDA(cudaFree(d_x)); \
        CHECK_CUDA(cudaFree(d_y)); \
        CHECK_CUDA(cudaFreeHost(h_x)); \
        CHECK_CUDA(cudaFreeHost(h_y)); \
    } \
    CHECK_HIEDNN(hiednnDestroyTensorDesc(tensor_desc)); \
    CHECK_HIEDNN(hiednnDestroyCudaHandle(handle)); \
}

// -----------------------------------------------
// inclusive prefix scan
// -----------------------------------------------
PREFIX_SUM_D1_UTEST(Float, float, HIEDNN_DATATYPE_FP32,
                    double,
                    Inclusive, 0,
                    Prefix, 0);

PREFIX_SUM_D1_UTEST(Int16, int16_t, HIEDNN_DATATYPE_INT16,
                    int32_t,
                    Inclusive, 0,
                    Prefix, 0);

PREFIX_SUM_D1_UTEST(Int32, int32_t, HIEDNN_DATATYPE_INT32,
                    int32_t,
                    Inclusive, 0,
                    Prefix, 0);

PREFIX_SUM_D1_UTEST(Int64, int64_t, HIEDNN_DATATYPE_INT64,
                    int64_t,
                    Inclusive, 0,
                    Prefix, 0);

// -----------------------------------------------
// exclusive prefix scan
// -----------------------------------------------
PREFIX_SUM_D1_UTEST(Float, float, HIEDNN_DATATYPE_FP32,
                    double,
                    Exclusive, 1,
                    Prefix, 0);

PREFIX_SUM_D1_UTEST(Int16, int16_t, HIEDNN_DATATYPE_INT16,
                    int32_t,
                    Exclusive, 1,
                    Prefix, 0);

PREFIX_SUM_D1_UTEST(Int32, int32_t, HIEDNN_DATATYPE_INT32,
                    int32_t,
                    Exclusive, 1,
                    Prefix, 0);

PREFIX_SUM_D1_UTEST(Int64, int64_t, HIEDNN_DATATYPE_INT64,
                    int64_t,
                    Exclusive, 1,
                    Prefix, 0);

// -----------------------------------------------
// inclusive suffix scan
// -----------------------------------------------
PREFIX_SUM_D1_UTEST(Float, float, HIEDNN_DATATYPE_FP32,
                    double,
                    Inclusive, 0,
                    Suffix, 1);

PREFIX_SUM_D1_UTEST(Int16, int16_t, HIEDNN_DATATYPE_INT16,
                    int32_t,
                    Inclusive, 0,
                    Suffix, 1);

PREFIX_SUM_D1_UTEST(Int32, int32_t, HIEDNN_DATATYPE_INT32,
                    int32_t,
                    Inclusive, 0,
                    Suffix, 1);

PREFIX_SUM_D1_UTEST(Int64, int64_t, HIEDNN_DATATYPE_INT64,
                    int64_t,
                    Inclusive, 0,
                    Suffix, 1);

// -----------------------------------------------
// exclusive suffix scan
// -----------------------------------------------
PREFIX_SUM_D1_UTEST(Float, float, HIEDNN_DATATYPE_FP32,
                    double,
                    Exclusive, 1,
                    Suffix, 1);

PREFIX_SUM_D1_UTEST(Int16, int16_t, HIEDNN_DATATYPE_INT16,
                    int32_t,
                    Exclusive, 1,
                    Suffix, 1);

PREFIX_SUM_D1_UTEST(Int32, int32_t, HIEDNN_DATATYPE_INT32,
                    int32_t,
                    Exclusive, 1,
                    Suffix, 1);

PREFIX_SUM_D1_UTEST(Int64, int64_t, HIEDNN_DATATYPE_INT64,
                    int64_t,
                    Exclusive, 1,
                    Suffix, 1);


