/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    14_trilu.cu
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

void Triu() {
    /*
     * input:
     * tensor x: dim={1, 4, 5}, dataType=int32_t
     * param k=0
     * param triluOp=HIEDNN_TRILU_UPPER
     *
     * output:
     * tensor y: dim={1, 4, 5}, dataType=int32_t
     */
    // input:
    int32_t x[] = {1,   2,  3,  4,  5,
                   6,   7,  8,  9, 10,
                   11, 12, 13, 14, 15,
                   16, 17, 18, 19, 20};

    // expected output:
    int32_t yref[] = {1,  2,  3,  4,  5,
                      0,  7,  8,  9, 10,
                      0,  0, 13, 14, 15,
                      0,  0,  0, 19, 20};

    int64_t dim[] = {1, 4, 5};
    int nDims = 3;
    int size = 1 * 4 * 5;
    hiednnDataType_t dataType = HIEDNN_DATATYPE_INT32;

    // create CUDA handle
    hiednnCudaHandle_t handle;
    CHECK_HIEDNN(hiednnCreateCudaHandle(&handle));

    // default stream
    CHECK_HIEDNN(hiednnSetCudaStream(handle, 0));

    // create tensor descriptors
    hiednnTensorDesc_t xDesc, yDesc;
    CHECK_HIEDNN(hiednnCreateTensorDesc(&xDesc));
    CHECK_HIEDNN(hiednnCreateTensorDesc(&yDesc));

    // init tensor descriptors
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(xDesc, dataType, nDims, dim));
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(yDesc, dataType, nDims, dim));

    // allocate device memory for tensors and copy input tensors to device
    int32_t *dX, *dY;
    CHECK_CUDA(cudaMalloc(&dX, size * sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&dY, size * sizeof(int32_t)));
    CHECK_CUDA(cudaMemcpy(dX, x, size * sizeof(int32_t),
                          cudaMemcpyHostToDevice));

    CHECK_HIEDNN(hiednnCudaTrilu(
        handle, xDesc, dX, yDesc, dY, 0, HIEDNN_TRILU_UPPER));

    int32_t *y = static_cast<int32_t *>(malloc(size * sizeof(int32_t)));

    // copy output tensor from device to host
    CHECK_CUDA(cudaMemcpy(y, dY, size * sizeof(int32_t),
                          cudaMemcpyDeviceToHost));

    // check output
    printf("check output tenosr... ");
    for (int i = 0; i < size; ++i) {
        if (y[i] != yref[i]) {
            printf("FAILED\n");
            exit(1);
        }
    }
    printf("OK\n");

    CHECK_HIEDNN(hiednnDestroyTensorDesc(xDesc));
    CHECK_HIEDNN(hiednnDestroyTensorDesc(yDesc));
    CHECK_HIEDNN(hiednnDestroyCudaHandle(handle));

    CHECK_CUDA(cudaFree(dX));
    CHECK_CUDA(cudaFree(dY));

    free(y);
}

void TriuNeg() {
    /*
     * input:
     * tensor x: dim={1, 4, 5}, dataType=int32_t
     * param k=-2
     * param triluOp=HIEDNN_TRILU_UPPER
     *
     * output:
     * tensor y: dim={1, 4, 5}, dataType=int32_t
     */
    // input:
    int32_t x[] = {1,   2,  3,  4,  5,
                   6,   7,  8,  9, 10,
                   11, 12, 13, 14, 15,
                   16, 17, 18, 19, 20};

    // expected output:
    int32_t yref[] = {1,   2,  3,  4,  5,
                      6,   7,  8,  9, 10,
                      11, 12, 13, 14, 15,
                      0,  17, 18, 19, 20};

    int64_t dim[] = {1, 4, 5};
    int nDims = 3;
    int size = 1 * 4 * 5;
    hiednnDataType_t dataType = HIEDNN_DATATYPE_INT32;

    // create CUDA handle
    hiednnCudaHandle_t handle;
    CHECK_HIEDNN(hiednnCreateCudaHandle(&handle));

    // default stream
    CHECK_HIEDNN(hiednnSetCudaStream(handle, 0));

    // create tensor descriptors
    hiednnTensorDesc_t xDesc, yDesc;
    CHECK_HIEDNN(hiednnCreateTensorDesc(&xDesc));
    CHECK_HIEDNN(hiednnCreateTensorDesc(&yDesc));

    // init tensor descriptors
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(xDesc, dataType, nDims, dim));
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(yDesc, dataType, nDims, dim));

    // allocate device memory for tensors and copy input tensors to device
    int32_t *dX, *dY;
    CHECK_CUDA(cudaMalloc(&dX, size * sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&dY, size * sizeof(int32_t)));
    CHECK_CUDA(cudaMemcpy(dX, x, size * sizeof(int32_t),
                          cudaMemcpyHostToDevice));

    CHECK_HIEDNN(hiednnCudaTrilu(
        handle, xDesc, dX, yDesc, dY, -2, HIEDNN_TRILU_UPPER));

    int32_t *y = static_cast<int32_t *>(malloc(size * sizeof(int32_t)));

    // copy output tensor from device to host
    CHECK_CUDA(cudaMemcpy(y, dY, size * sizeof(int32_t),
                          cudaMemcpyDeviceToHost));

    // check output
    printf("check output tenosr... ");
    for (int i = 0; i < size; ++i) {
        if (y[i] != yref[i]) {
            printf("FAILED\n");
            exit(1);
        }
    }
    printf("OK\n");

    CHECK_HIEDNN(hiednnDestroyTensorDesc(xDesc));
    CHECK_HIEDNN(hiednnDestroyTensorDesc(yDesc));
    CHECK_HIEDNN(hiednnDestroyCudaHandle(handle));

    CHECK_CUDA(cudaFree(dX));
    CHECK_CUDA(cudaFree(dY));

    free(y);
}

void TriuPos() {
    /*
     * input:
     * tensor x: dim={1, 4, 5}, dataType=int32_t
     * param k=1
     * param triluOp=HIEDNN_TRILU_UPPER
     *
     * output:
     * tensor y: dim={1, 4, 5}, dataType=int32_t
     */
    // input:
    int32_t x[] = {1,   2,  3,  4,  5,
                   6,   7,  8,  9, 10,
                   11, 12, 13, 14, 15,
                   16, 17, 18, 19, 20};

    // expected output:
    int32_t yref[] = {0,  2,  3,  4,  5,
                      0,  0,  8,  9, 10,
                      0,  0,  0, 14, 15,
                      0,  0,  0,  0, 20};

    int64_t dim[] = {1, 4, 5};
    int nDims = 3;
    int size = 1 * 4 * 5;
    hiednnDataType_t dataType = HIEDNN_DATATYPE_INT32;

    // create CUDA handle
    hiednnCudaHandle_t handle;
    CHECK_HIEDNN(hiednnCreateCudaHandle(&handle));

    // default stream
    CHECK_HIEDNN(hiednnSetCudaStream(handle, 0));

    // create tensor descriptors
    hiednnTensorDesc_t xDesc, yDesc;
    CHECK_HIEDNN(hiednnCreateTensorDesc(&xDesc));
    CHECK_HIEDNN(hiednnCreateTensorDesc(&yDesc));

    // init tensor descriptors
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(xDesc, dataType, nDims, dim));
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(yDesc, dataType, nDims, dim));

    // allocate device memory for tensors and copy input tensors to device
    int32_t *dX, *dY;
    CHECK_CUDA(cudaMalloc(&dX, size * sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&dY, size * sizeof(int32_t)));
    CHECK_CUDA(cudaMemcpy(dX, x, size * sizeof(int32_t),
                          cudaMemcpyHostToDevice));

    CHECK_HIEDNN(hiednnCudaTrilu(
        handle, xDesc, dX, yDesc, dY, 1, HIEDNN_TRILU_UPPER));

    int32_t *y = static_cast<int32_t *>(malloc(size * sizeof(int32_t)));

    // copy output tensor from device to host
    CHECK_CUDA(cudaMemcpy(y, dY, size * sizeof(int32_t),
                          cudaMemcpyDeviceToHost));

    // check output
    printf("check output tenosr... ");
    for (int i = 0; i < size; ++i) {
        if (y[i] != yref[i]) {
            printf("FAILED\n");
            exit(1);
        }
    }
    printf("OK\n");

    CHECK_HIEDNN(hiednnDestroyTensorDesc(xDesc));
    CHECK_HIEDNN(hiednnDestroyTensorDesc(yDesc));
    CHECK_HIEDNN(hiednnDestroyCudaHandle(handle));

    CHECK_CUDA(cudaFree(dX));
    CHECK_CUDA(cudaFree(dY));

    free(y);
}

void Tril() {
    /*
     * input:
     * tensor x: dim={1, 4, 5}, dataType=int32_t
     * param k=0
     * param triluOp=HIEDNN_TRILU_LOWER
     *
     * output:
     * tensor y: dim={1, 4, 5}, dataType=int32_t
     */
    // input:
    int32_t x[] = {1,   2,  3,  4,  5,
                   6,   7,  8,  9, 10,
                   11, 12, 13, 14, 15,
                   16, 17, 18, 19, 20};

    // expected output:
    int32_t yref[] = {1,   0,  0,  0,  0,
                      6,   7,  0,  0,  0,
                      11, 12, 13,  0,  0,
                      16, 17, 18, 19,  0};

    int64_t dim[] = {1, 4, 5};
    int nDims = 3;
    int size = 1 * 4 * 5;
    hiednnDataType_t dataType = HIEDNN_DATATYPE_INT32;

    // create CUDA handle
    hiednnCudaHandle_t handle;
    CHECK_HIEDNN(hiednnCreateCudaHandle(&handle));

    // default stream
    CHECK_HIEDNN(hiednnSetCudaStream(handle, 0));

    // create tensor descriptors
    hiednnTensorDesc_t xDesc, yDesc;
    CHECK_HIEDNN(hiednnCreateTensorDesc(&xDesc));
    CHECK_HIEDNN(hiednnCreateTensorDesc(&yDesc));

    // init tensor descriptors
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(xDesc, dataType, nDims, dim));
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(yDesc, dataType, nDims, dim));

    // allocate device memory for tensors and copy input tensors to device
    int32_t *dX, *dY;
    CHECK_CUDA(cudaMalloc(&dX, size * sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&dY, size * sizeof(int32_t)));
    CHECK_CUDA(cudaMemcpy(dX, x, size * sizeof(int32_t),
                          cudaMemcpyHostToDevice));

    CHECK_HIEDNN(hiednnCudaTrilu(
        handle, xDesc, dX, yDesc, dY, 0, HIEDNN_TRILU_LOWER));

    int32_t *y = static_cast<int32_t *>(malloc(size * sizeof(int32_t)));

    // copy output tensor from device to host
    CHECK_CUDA(cudaMemcpy(y, dY, size * sizeof(int32_t),
                          cudaMemcpyDeviceToHost));

    // check output
    printf("check output tenosr... ");
    for (int i = 0; i < size; ++i) {
        if (y[i] != yref[i]) {
            printf("FAILED\n");
            exit(1);
        }
    }
    printf("OK\n");

    CHECK_HIEDNN(hiednnDestroyTensorDesc(xDesc));
    CHECK_HIEDNN(hiednnDestroyTensorDesc(yDesc));
    CHECK_HIEDNN(hiednnDestroyCudaHandle(handle));

    CHECK_CUDA(cudaFree(dX));
    CHECK_CUDA(cudaFree(dY));

    free(y);
}

void TrilNeg() {
    /*
     * input:
     * tensor x: dim={1, 4, 5}, dataType=int32_t
     * param k=-1
     * param triluOp=HIEDNN_TRILU_LOWER
     *
     * output:
     * tensor y: dim={1, 4, 5}, dataType=int32_t
     */
    // input:
    int32_t x[] = {1,   2,  3,  4,  5,
                   6,   7,  8,  9, 10,
                   11, 12, 13, 14, 15,
                   16, 17, 18, 19, 20};

    // expected output:
    int32_t yref[] = {0,   0,  0,  0,  0,
                      6,   0,  0,  0,  0,
                      11, 12,  0,  0,  0,
                      16, 17, 18,  0,  0};

    int64_t dim[] = {1, 4, 5};
    int nDims = 3;
    int size = 1 * 4 * 5;
    hiednnDataType_t dataType = HIEDNN_DATATYPE_INT32;

    // create CUDA handle
    hiednnCudaHandle_t handle;
    CHECK_HIEDNN(hiednnCreateCudaHandle(&handle));

    // default stream
    CHECK_HIEDNN(hiednnSetCudaStream(handle, 0));

    // create tensor descriptors
    hiednnTensorDesc_t xDesc, yDesc;
    CHECK_HIEDNN(hiednnCreateTensorDesc(&xDesc));
    CHECK_HIEDNN(hiednnCreateTensorDesc(&yDesc));

    // init tensor descriptors
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(xDesc, dataType, nDims, dim));
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(yDesc, dataType, nDims, dim));

    // allocate device memory for tensors and copy input tensors to device
    int32_t *dX, *dY;
    CHECK_CUDA(cudaMalloc(&dX, size * sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&dY, size * sizeof(int32_t)));
    CHECK_CUDA(cudaMemcpy(dX, x, size * sizeof(int32_t),
                          cudaMemcpyHostToDevice));

    CHECK_HIEDNN(hiednnCudaTrilu(
        handle, xDesc, dX, yDesc, dY, -1, HIEDNN_TRILU_LOWER));

    int32_t *y = static_cast<int32_t *>(malloc(size * sizeof(int32_t)));

    // copy output tensor from device to host
    CHECK_CUDA(cudaMemcpy(y, dY, size * sizeof(int32_t),
                          cudaMemcpyDeviceToHost));

    // check output
    printf("check output tenosr... ");
    for (int i = 0; i < size; ++i) {
        if (y[i] != yref[i]) {
            printf("FAILED\n");
            exit(1);
        }
    }
    printf("OK\n");

    CHECK_HIEDNN(hiednnDestroyTensorDesc(xDesc));
    CHECK_HIEDNN(hiednnDestroyTensorDesc(yDesc));
    CHECK_HIEDNN(hiednnDestroyCudaHandle(handle));

    CHECK_CUDA(cudaFree(dX));
    CHECK_CUDA(cudaFree(dY));

    free(y);
}

void TrilPos() {
    /*
     * input:
     * tensor x: dim={1, 4, 5}, dataType=int32_t
     * param k=2
     * param triluOp=HIEDNN_TRILU_LOWER
     *
     * output:
     * tensor y: dim={1, 4, 5}, dataType=int32_t
     */
    // input:
    int32_t x[] = {1,   2,  3,  4,  5,
                   6,   7,  8,  9, 10,
                   11, 12, 13, 14, 15,
                   16, 17, 18, 19, 20};

    // expected output:
    int32_t yref[] = {1,   2,  3,  0,  0,
                      6,   7,  8,  9,  0,
                      11, 12, 13, 14, 15,
                      16, 17, 18, 19, 20};

    int64_t dim[] = {1, 4, 5};
    int nDims = 3;
    int size = 1 * 4 * 5;
    hiednnDataType_t dataType = HIEDNN_DATATYPE_INT32;

    // create CUDA handle
    hiednnCudaHandle_t handle;
    CHECK_HIEDNN(hiednnCreateCudaHandle(&handle));

    // default stream
    CHECK_HIEDNN(hiednnSetCudaStream(handle, 0));

    // create tensor descriptors
    hiednnTensorDesc_t xDesc, yDesc;
    CHECK_HIEDNN(hiednnCreateTensorDesc(&xDesc));
    CHECK_HIEDNN(hiednnCreateTensorDesc(&yDesc));

    // init tensor descriptors
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(xDesc, dataType, nDims, dim));
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(yDesc, dataType, nDims, dim));

    // allocate device memory for tensors and copy input tensors to device
    int32_t *dX, *dY;
    CHECK_CUDA(cudaMalloc(&dX, size * sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&dY, size * sizeof(int32_t)));
    CHECK_CUDA(cudaMemcpy(dX, x, size * sizeof(int32_t),
                          cudaMemcpyHostToDevice));

    CHECK_HIEDNN(hiednnCudaTrilu(
        handle, xDesc, dX, yDesc, dY, 2, HIEDNN_TRILU_LOWER));

    int32_t *y = static_cast<int32_t *>(malloc(size * sizeof(int32_t)));

    // copy output tensor from device to host
    CHECK_CUDA(cudaMemcpy(y, dY, size * sizeof(int32_t),
                          cudaMemcpyDeviceToHost));

    // check output
    printf("check output tenosr... ");
    for (int i = 0; i < size; ++i) {
        if (y[i] != yref[i]) {
            printf("FAILED\n");
            exit(1);
        }
    }
    printf("OK\n");

    CHECK_HIEDNN(hiednnDestroyTensorDesc(xDesc));
    CHECK_HIEDNN(hiednnDestroyTensorDesc(yDesc));
    CHECK_HIEDNN(hiednnDestroyCudaHandle(handle));

    CHECK_CUDA(cudaFree(dX));
    CHECK_CUDA(cudaFree(dY));

    free(y);
}

int main() {
    Triu();
    TriuNeg();
    TriuPos();

    Tril();
    TrilNeg();
    TrilPos();
    return 0;
}
