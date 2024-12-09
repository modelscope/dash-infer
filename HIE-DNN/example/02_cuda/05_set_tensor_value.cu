/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    05_set_tensor_value.cu
 */

#include <hiednn.h>
#include <hiednn_cuda.h>

#include <cstdio>
#include <cstdlib>
#include <cstdint>

#define CHECK_CUDA(expr) { \
    if ((expr) != cudaSuccess) { \
        int line = __LINE__; \
        printf("cuda error at %d\n", line); \
        exit(1); \
    } \
}

#define CHECK_HIEDNN(expr) { \
    if ((expr) != HIEDNN_STATUS_SUCCESS) { \
        int line = __LINE__; \
        printf("hiednn error at %d\n", line); \
        exit(1); \
    } \
}

void SetTensorConst() {
    /*
     * input:
     * p0 = 10
     *
     * output:
     * tensor y: dim={4, 5, 6}, dataType=int32_t
     *
     * y[*] = 
     */
    int64_t dim[] = {4, 5, 6};
    int nDims = 3;
    int size = 4 * 5 * 6;

    int32_t p0 = 10;
    hiednnSetTensorValueMode_t mode = HIEDNN_SETTENSOR_CONST;

    hiednnDataType_t ytype = HIEDNN_DATATYPE_INT32;

    // create cuda handle
    hiednnCudaHandle_t handle;
    CHECK_HIEDNN(hiednnCreateCudaHandle(&handle));

    // default stream
    CHECK_HIEDNN(hiednnSetCudaStream(handle, 0));

    // create tensor descriptor
    hiednnTensorDesc_t yDesc;
    CHECK_HIEDNN(hiednnCreateTensorDesc(&yDesc));

    // init tensor descriptor
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(yDesc, ytype, nDims, dim));

    // allocate device memory for tensor y
    int32_t *dy;
    CHECK_CUDA(cudaMalloc(&dy, size * sizeof(int32_t)));

    printf("HIEDNN_SETTENSOR_CONST\n");
    CHECK_HIEDNN(hiednnCudaSetTensorValue(
        handle, mode, &p0, nullptr, yDesc, dy));

    int32_t *hy = static_cast<int32_t *>(malloc(size * sizeof(int32_t)));

    // copy tensor y from device to host
    CHECK_CUDA(cudaMemcpy(hy, dy, size * sizeof(int32_t),
                          cudaMemcpyDeviceToHost));

    // check output
    printf("check output tenosr... ");
    for (int i = 0; i < size; ++i) {
        if (hy[i] != p0) {
            printf("FAILED\n");
            exit(1);
        }
    }
    printf("OK\n");

    CHECK_HIEDNN(hiednnDestroyTensorDesc(yDesc));
    CHECK_HIEDNN(hiednnDestroyCudaHandle(handle));

    CHECK_CUDA(cudaFree(dy));

    free(hy);
}

void SetTensorRange() {
    /*
     * input:
     * p0 = 5, p1 = 3
     *
     * output(1D tensor):
     * tensor y: dim={120}, dataType=int32_t
     *
     * y[i] = p0 + p1 * i
     */
    int64_t dim[] = {120};
    int nDims = 1;
    int size = dim[0];

    int32_t p0 = 5;
    int32_t p1 = 3;
    hiednnSetTensorValueMode_t mode = HIEDNN_SETTENSOR_RANGE;

    hiednnDataType_t ytype = HIEDNN_DATATYPE_INT32;

    // create cuda handle
    hiednnCudaHandle_t handle;
    CHECK_HIEDNN(hiednnCreateCudaHandle(&handle));

    // default stream
    CHECK_HIEDNN(hiednnSetCudaStream(handle, 0));

    // create tensor descriptor
    hiednnTensorDesc_t yDesc;
    CHECK_HIEDNN(hiednnCreateTensorDesc(&yDesc));

    // init tensor descriptor
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(yDesc, ytype, nDims, dim));

    // allocate device memory for tensor y
    int32_t *dy;
    CHECK_CUDA(cudaMalloc(&dy, size * sizeof(int32_t)));

    printf("HIEDNN_SETTENSOR_RANGE\n");
    CHECK_HIEDNN(hiednnCudaSetTensorValue(handle, mode, &p0, &p1, yDesc, dy));

    int32_t *hy = static_cast<int32_t *>(malloc(size * sizeof(int32_t)));

    // copy tensor y from device to host
    CHECK_CUDA(cudaMemcpy(hy, dy, size * sizeof(int32_t),
                          cudaMemcpyDeviceToHost));

    // check output
    printf("check output tenosr... ");
    for (int i = 0; i < size; ++i) {
        if (hy[i] != p0 + p1 * i) {
            printf("FAILED\n");
            exit(1);
        }
    }
    printf("OK\n");

    CHECK_HIEDNN(hiednnDestroyTensorDesc(yDesc));
    CHECK_HIEDNN(hiednnDestroyCudaHandle(handle));

    CHECK_CUDA(cudaFree(dy));

    free(hy);
}

void SetTensorDiagonal1() {
    /*
     * input:
     * p0 = 1, p1 = 5
     *
     * output(2D tensor):
     * tensor y: dim={5, 4}, dataType=int32_t
     *
     * y[*] = {
     *     0, 5, 0, 0,
     *     0, 0, 5, 0,
     *     0, 0, 0, 5,
     *     0, 0, 0, 0,
     *     0, 0, 0, 0,
     * }
     */
    int64_t dim[] = {5, 4};
    int nDims = 2;
    int size = dim[0] * dim[1];

    int32_t yref[] = {0, 5, 0, 0,
                      0, 0, 5, 0,
                      0, 0, 0, 5,
                      0, 0, 0, 0,
                      0, 0, 0, 0};

    int32_t p0 = 1;
    int32_t p1 = 5;
    hiednnSetTensorValueMode_t mode = HIEDNN_SETTENSOR_DIAGONAL;

    hiednnDataType_t ytype = HIEDNN_DATATYPE_INT32;

    // create cuda handle
    hiednnCudaHandle_t handle;
    CHECK_HIEDNN(hiednnCreateCudaHandle(&handle));

    // default stream
    CHECK_HIEDNN(hiednnSetCudaStream(handle, 0));

    // create tensor descriptor
    hiednnTensorDesc_t yDesc;
    CHECK_HIEDNN(hiednnCreateTensorDesc(&yDesc));

    // init tensor descriptor
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(yDesc, ytype, nDims, dim));

    // allocate device memory for tensor y
    int32_t *dy;
    CHECK_CUDA(cudaMalloc(&dy, size * sizeof(int32_t)));

    printf("HIEDNN_SETTENSOR_DIAGONAL\n");
    CHECK_HIEDNN(hiednnCudaSetTensorValue(handle, mode, &p0, &p1, yDesc, dy));

    int32_t *hy = static_cast<int32_t *>(malloc(size * sizeof(int32_t)));

    // copy tensor y from device to host
    CHECK_CUDA(cudaMemcpy(hy, dy, size * sizeof(int32_t),
                          cudaMemcpyDeviceToHost));

    // check output
    printf("check output tenosr... ");
    for (int i = 0; i < size; ++i) {
        if (hy[i] != yref[i]) {
            printf("FAILED\n");
            exit(1);
        }
    }
    printf("OK\n");

    CHECK_HIEDNN(hiednnDestroyTensorDesc(yDesc));
    CHECK_HIEDNN(hiednnDestroyCudaHandle(handle));

    CHECK_CUDA(cudaFree(dy));

    free(hy);
}

void SetTensorDiagonal2() {
    /*
     * input:
     * p0 = -1, p1 = 5
     *
     * output(2D tensor):
     * tensor y: dim={5, 4}, dataType=int32_t
     *
     * y[*] = {
     *     0, 0, 0, 0,
     *     5, 0, 0, 0,
     *     0, 5, 0, 0,
     *     0, 0, 5, 0,
     *     0, 0, 0, 5,
     * }
     */
    int64_t dim[] = {5, 4};
    int nDims = 2;
    int size = dim[0] * dim[1];

    int32_t yref[] = {0, 0, 0, 0,
                      5, 0, 0, 0,
                      0, 5, 0, 0,
                      0, 0, 5, 0,
                      0, 0, 0, 5};

    int32_t p0 = 1;
    int32_t p1 = 5;
    hiednnSetTensorValueMode_t mode = HIEDNN_SETTENSOR_DIAGONAL;

    hiednnDataType_t ytype = HIEDNN_DATATYPE_INT32;

    // create cuda handle
    hiednnCudaHandle_t handle;
    CHECK_HIEDNN(hiednnCreateCudaHandle(&handle));

    // default stream
    CHECK_HIEDNN(hiednnSetCudaStream(handle, 0));

    // create tensor descriptor
    hiednnTensorDesc_t yDesc;
    CHECK_HIEDNN(hiednnCreateTensorDesc(&yDesc));

    // init tensor descriptor
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(yDesc, ytype, nDims, dim));

    // allocate device memory for tensor y
    int32_t *dy;
    CHECK_CUDA(cudaMalloc(&dy, size * sizeof(int32_t)));

    printf("HIEDNN_SETTENSOR_DIAGONAL\n");
    CHECK_HIEDNN(hiednnCudaSetTensorValue(handle, mode, &p0, &p1, yDesc, dy));

    int32_t *hy = static_cast<int32_t *>(malloc(size * sizeof(int32_t)));

    // copy tensor y from device to host
    CHECK_CUDA(cudaMemcpy(hy, dy, size * sizeof(int32_t),
                          cudaMemcpyDeviceToHost));

    // check output
    printf("check output tenosr... ");
    for (int i = 0; i < size; ++i) {
        if (hy[i] != yref[i]) {
            printf("FAILED\n");
            exit(1);
        }
    }
    printf("OK\n");

    CHECK_HIEDNN(hiednnDestroyTensorDesc(yDesc));
    CHECK_HIEDNN(hiednnDestroyCudaHandle(handle));

    CHECK_CUDA(cudaFree(dy));

    free(hy);
}

int main() {
    SetTensorConst();
    SetTensorRange();
    SetTensorDiagonal1();

    return 0;
}

