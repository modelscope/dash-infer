/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    04_expand.cu
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
     * tensor x: dim={2, 1}, dataType=int32_t
     *
     * output:
     * tensor y: dim={3, 2, 2}, dataType=int32_t
     */
    // input:
    int32_t x[] = {1,
                   2};
    // expected output:
    int32_t yref[] = {1, 1,
                      2, 2,
                      1, 1,
                      2, 2,
                      1, 1,
                      2, 2 };
    int xNDims = 2;
    int yNDims = 3;
    int64_t xDim[] = {2, 1};
    int64_t yDim[] = {3, 2, 2};
    int xSize = 2 * 1;
    int ySize = 3 * 2 * 2;
    hiednnDataType_t dtype = HIEDNN_DATATYPE_INT32;

    // create cuda handle
    hiednnCudaHandle_t handle;
    CHECK_HIEDNN(hiednnCreateCudaHandle(&handle));

    // default stream
    CHECK_HIEDNN(hiednnSetCudaStream(handle, 0));

    // create tensor descriptor
    hiednnTensorDesc_t xDesc, yDesc;
    CHECK_HIEDNN(hiednnCreateTensorDesc(&xDesc));
    CHECK_HIEDNN(hiednnCreateTensorDesc(&yDesc));

    // init tensor descriptor
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(xDesc, dtype, xNDims, xDim));
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(yDesc, dtype, yNDims, yDim));

    // allocate device memory for tensor and copy input tensor to device
    int32_t *dx, *dy;
    CHECK_CUDA(cudaMalloc(&dx, xSize * sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&dy, ySize * sizeof(int32_t)));
    CHECK_CUDA(cudaMemcpy(dx, x, xSize * sizeof(int32_t),
                          cudaMemcpyHostToDevice));

    CHECK_HIEDNN(hiednnCudaExpand(handle, xDesc, dx, yDesc, dy));

    int32_t y[3 * 2 * 2];

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

    CHECK_HIEDNN(hiednnDestroyTensorDesc(xDesc));
    CHECK_HIEDNN(hiednnDestroyTensorDesc(yDesc));
    CHECK_HIEDNN(hiednnDestroyCudaHandle(handle));

    CHECK_CUDA(cudaFree(dx));
    CHECK_CUDA(cudaFree(dy));

    return 0;
}

