/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    18_concat.cu
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

int main() {
    /*
     * input:
     * tensor x: dim={2, 3}, dataType=int32_t
     *           dim={2, 4}, dataType=int32_t
     *           dim={2, 5}, dataType=int32_t
     * param inputCount=3
     * param axis=1
     *
     * output:
     * tensor y: dim={2, 3+4+5}, dataType=int32_t
     */
    // input:
    int32_t x0[] = {0, 1, 2,
                    3, 4, 5};
    int32_t x1[] = {6, 7, 8, 9,
                    10, 11, 12, 13};
    int32_t x2[] = {14, 15, 16, 17, 18,
                    19, 20, 21, 22, 23};
    int inputCount = 3;
    int axis = 1;

    // expected output:
    int32_t yref[] = {0, 1, 2, 6, 7, 8, 9, 14, 15, 16, 17, 18,
                      3, 4, 5, 10, 11, 12, 13, 19, 20, 21, 22, 23};

    int nDims = 2;
    int64_t x0Dim[] = {2, 3};
    int64_t x1Dim[] = {2, 4};
    int64_t x2Dim[] = {2, 5};
    int64_t yDim[] = {2, 12};

    int x0Size = 2 * 3;
    int x1Size = 2 * 4;
    int x2Size = 2 * 5;
    int ySize = 2 * 12;

    hiednnDataType_t dataType = HIEDNN_DATATYPE_INT32;

    // create cuda handle
    hiednnCudaHandle_t handle;
    CHECK_HIEDNN(hiednnCreateCudaHandle(&handle));

    // default stream
    CHECK_HIEDNN(hiednnSetCudaStream(handle, 0));

    // create tensor descriptor
    hiednnTensorDesc_t xDesc[3], yDesc;
    CHECK_HIEDNN(hiednnCreateTensorDesc(&xDesc[0]));
    CHECK_HIEDNN(hiednnCreateTensorDesc(&xDesc[1]));
    CHECK_HIEDNN(hiednnCreateTensorDesc(&xDesc[2]));
    CHECK_HIEDNN(hiednnCreateTensorDesc(&yDesc));

    // init tensor descriptor
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(xDesc[0], dataType, nDims, x0Dim));
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(xDesc[1], dataType, nDims, x1Dim));
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(xDesc[2], dataType, nDims, x2Dim));
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(yDesc, dataType, nDims, yDim));

    // allocate device memory for tensor and copy input tensor to device
    int32_t *dx0, *dx1, *dx2;
    CHECK_CUDA(cudaMalloc(&dx0, x0Size * sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&dx1, x1Size * sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&dx2, x2Size * sizeof(int32_t)));
    int32_t *dy;
    CHECK_CUDA(cudaMalloc(&dy, ySize * sizeof(int32_t)));

    CHECK_CUDA(cudaMemcpy(dx0, x0, x0Size * sizeof(int32_t),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dx1, x1, x1Size * sizeof(int32_t),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dx2, x2, x2Size * sizeof(int32_t),
                          cudaMemcpyHostToDevice));

    const void *xPtrs[] = {dx0, dx1, dx2};

    CHECK_HIEDNN(hiednnCudaConcat(
        handle, xDesc, xPtrs, inputCount, axis, yDesc, dy));

    int32_t *y = static_cast<int32_t *>(malloc(ySize * sizeof(int32_t)));
    // copy output tensor from device to host
    CHECK_CUDA(cudaMemcpy(y, dy, ySize * sizeof(int32_t),
                          cudaMemcpyDeviceToHost));

    // check output
    printf("check output tenosr... ");
    for (int i = 0; i < ySize; ++i) {
        if (y[i] != yref[i]) {
            printf("FAILED\n");
            exit(1);
        }
    }
    printf("OK\n");

    CHECK_HIEDNN(hiednnDestroyTensorDesc(xDesc[0]));
    CHECK_HIEDNN(hiednnDestroyTensorDesc(xDesc[1]));
    CHECK_HIEDNN(hiednnDestroyTensorDesc(xDesc[2]));
    CHECK_HIEDNN(hiednnDestroyTensorDesc(yDesc));
    CHECK_HIEDNN(hiednnDestroyCudaHandle(handle));

    CHECK_CUDA(cudaFree(dx0));
    CHECK_CUDA(cudaFree(dx1));
    CHECK_CUDA(cudaFree(dx2));
    CHECK_CUDA(cudaFree(dy));

    free(y);

    return 0;
}

