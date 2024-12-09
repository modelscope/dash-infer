/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    12_pad.cu
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

void PadConst() {
    /*
     * input:
     * tensor x: dim={3, 2}, dataType=int32_t
     * pads: {1, 2, 1, 2}
     * mode: HIEDNN_PAD_CONST
     * param: 10
     *
     * output:
     * tensor y: dim={5, 6}, dataType=int32_t
     */
    // input:
    int32_t x[] = {0, 1,
                   2, 3,
                   4, 5};
    // expected output:
    int32_t yref[] = {10, 10, 10, 10, 10, 10,
                      10, 10, 0,  1,  10, 10,
                      10, 10, 2,  3,  10, 10,
                      10, 10, 4,  5,  10, 10,
                      10, 10, 10, 10, 10, 10};

    int nDims = 2;
    int64_t xDim[] = {3, 2};
    int64_t yDim[] = {5, 6};
    int xSize = 3 * 2;
    int ySize = 5 * 6;
    int64_t pads[] = {1, 2, 1, 2};
    int nPads = 4;
    hiednnDataType_t dtype = HIEDNN_DATATYPE_INT32;
    hiednnPadMode_t mode = HIEDNN_PAD_CONST;
    int32_t param = 10;

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
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(xDesc, dtype, nDims, xDim));
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(yDesc, dtype, nDims, yDim));

    // allocate device memory for tensor and copy input tensor to device
    int32_t *dx, *dy;
    CHECK_CUDA(cudaMalloc(&dx, xSize * sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&dy, ySize * sizeof(int32_t)));
    CHECK_CUDA(cudaMemcpy(dx, x, xSize * sizeof(int32_t),
                          cudaMemcpyHostToDevice));

    printf("PadConst\n");
    CHECK_HIEDNN(hiednnCudaPad(handle, mode, xDesc, dx, pads, nPads,
                               &param, yDesc, dy));

    int32_t y[5 * 6];

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
}

void PadEdge() {
    /*
     * input:
     * tensor x: dim={3, 2}, dataType=int32_t
     * pads: {1, 2, 1, 2}
     * mode: HIEDNN_PAD_EDGE
     *
     * output:
     * tensor y: dim={5, 6}, dataType=int32_t
     */
    // input:
    int32_t x[] = {0, 1,
                   2, 3,
                   4, 5};
    // expected output:
    int32_t yref[] = {0, 0, 0, 1, 1, 1,
                      0, 0, 0, 1, 1, 1,
                      2, 2, 2, 3, 3, 3,
                      4, 4, 4, 5, 5, 5,
                      4, 4, 4, 5, 5, 5};

    int nDims = 2;
    int64_t xDim[] = {3, 2};
    int64_t yDim[] = {5, 6};
    int xSize = 3 * 2;
    int ySize = 5 * 6;
    int64_t pads[] = {1, 2, 1, 2};
    int nPads = 4;
    hiednnDataType_t dtype = HIEDNN_DATATYPE_INT32;
    hiednnPadMode_t mode = HIEDNN_PAD_EDGE;

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
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(xDesc, dtype, nDims, xDim));
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(yDesc, dtype, nDims, yDim));

    // allocate device memory for tensor and copy input tensor to device
    int32_t *dx, *dy;
    CHECK_CUDA(cudaMalloc(&dx, xSize * sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&dy, ySize * sizeof(int32_t)));
    CHECK_CUDA(cudaMemcpy(dx, x, xSize * sizeof(int32_t),
                          cudaMemcpyHostToDevice));

    printf("PadEdge\n");
    CHECK_HIEDNN(hiednnCudaPad(handle, mode, xDesc, dx, pads, nPads,
                               nullptr, yDesc, dy));

    int32_t y[5 * 6];

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
}

void PadReflect() {
    /*
     * input:
     * tensor x: dim={3, 3}, dataType=int32_t
     * pads: {1, 2, 1, 2}
     * mode: HIEDNN_PAD_REFLECT
     *
     * output:
     * tensor y: dim={5, 7}, dataType=int32_t
     */
    // input:
    int32_t x[] = {0, 1, 2,
                   3, 4, 5,
                   6, 7, 8};
    // expected output:
    int32_t yref[] = {5, 4, 3, 4, 5, 4, 3,
                      2, 1, 0, 1, 2, 1, 0,
                      5, 4, 3, 4, 5, 4, 3,
                      8, 7, 6, 7, 8, 7, 6,
                      5, 4, 3, 4, 5, 4, 3};

    int nDims = 2;
    int64_t xDim[] = {3, 3};
    int64_t yDim[] = {5, 7};
    int xSize = 3 * 3;
    int ySize = 5 * 7;
    int64_t pads[] = {1, 2, 1, 2};
    int nPads = 4;
    hiednnDataType_t dtype = HIEDNN_DATATYPE_INT32;
    hiednnPadMode_t mode = HIEDNN_PAD_REFLECT;

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
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(xDesc, dtype, nDims, xDim));
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(yDesc, dtype, nDims, yDim));

    // allocate device memory for tensor and copy input tensor to device
    int32_t *dx, *dy;
    CHECK_CUDA(cudaMalloc(&dx, xSize * sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&dy, ySize * sizeof(int32_t)));
    CHECK_CUDA(cudaMemcpy(dx, x, xSize * sizeof(int32_t),
                          cudaMemcpyHostToDevice));

    printf("PadReflect\n");
    CHECK_HIEDNN(hiednnCudaPad(handle, mode, xDesc, dx, pads, nPads,
                               nullptr, yDesc, dy));

    int32_t y[5 * 7];

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
}

int main() {
    PadConst();
    PadEdge();
    PadReflect();
}


