/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    15_where.cu
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
     * tensor x: dim={1, 2}, dataType=int32_t
     * tensor y: dim={2, 1}, dataType=int32_t
     * tensor cond: dim={2, 2, 1}, datatype=char
     *
     * output:
     * tensor z: dim={2, 2, 2}, dataType=int32_t
     */
    // input:
    int32_t x[] = {1, 2};
    int32_t y[] = {3,
                   4};
    char cond[] = {1,
                   0,

                   0,
                   1};

    // expected output:
    int32_t zref[] = {1, 2,
                      4, 4,

                      3, 3,
                      1, 2};

    int xNDims = 2;
    int yNDims = 2;
    int condNDims = 3;
    int zNDims = 3;
    int64_t xDim[] = {1, 2};
    int64_t yDim[] = {2, 1};
    int64_t condDim[] = {2, 2, 1};
    int64_t zDim[] = {2, 2, 2};
    int xSize = 1 * 2;
    int ySize = 2 * 1;
    int condSize = 2 * 2 * 1;
    int zSize = 2 * 2 * 2;
    hiednnDataType_t dtype = HIEDNN_DATATYPE_INT32;
    hiednnDataType_t condtype = HIEDNN_DATATYPE_BOOL;

    // create cuda handle
    hiednnCudaHandle_t handle;
    CHECK_HIEDNN(hiednnCreateCudaHandle(&handle));

    // default stream
    CHECK_HIEDNN(hiednnSetCudaStream(handle, 0));

    // create tensor descriptor
    hiednnTensorDesc_t xDesc, yDesc, condDesc, zDesc;
    CHECK_HIEDNN(hiednnCreateTensorDesc(&xDesc));
    CHECK_HIEDNN(hiednnCreateTensorDesc(&yDesc));
    CHECK_HIEDNN(hiednnCreateTensorDesc(&condDesc));
    CHECK_HIEDNN(hiednnCreateTensorDesc(&zDesc));

    // init tensor descriptor
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(
        xDesc, dtype, xNDims, xDim));
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(
        yDesc, dtype, yNDims, yDim));
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(
        condDesc, condtype, condNDims, condDim));
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(
        zDesc, dtype, zNDims, zDim));

    // allocate device memory for tensor and copy input tensor to device
    char *dcond;
    CHECK_CUDA(cudaMalloc(&dcond, condSize * sizeof(char)));

    int32_t *dx, *dy, *dz;
    CHECK_CUDA(cudaMalloc(&dx, xSize * sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&dy, ySize * sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&dz, zSize * sizeof(int32_t)));

    CHECK_CUDA(cudaMemcpy(dx, x, xSize * sizeof(int32_t),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dy, y, ySize * sizeof(int32_t),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dcond, cond, condSize * sizeof(char),
                          cudaMemcpyHostToDevice));

    CHECK_HIEDNN(hiednnCudaWhere(
        handle, xDesc, dx, yDesc, dy, condDesc, dcond, zDesc, dz));

    int32_t z[2 * 2 * 2];

    // copy output tensor from device to host
    CHECK_CUDA(cudaMemcpy(z, dz, zSize * sizeof(int32_t),
                          cudaMemcpyDeviceToHost));

    // check output
    printf("check output tenosr... ");
    for (int i = 0; i < zSize; ++i) {
        if (z[i] != zref[i]) {
            printf("FAILED\n");
            exit(1);
        }
    }
    printf("OK\n");

    CHECK_HIEDNN(hiednnDestroyTensorDesc(xDesc));
    CHECK_HIEDNN(hiednnDestroyTensorDesc(yDesc));
    CHECK_HIEDNN(hiednnDestroyTensorDesc(condDesc));
    CHECK_HIEDNN(hiednnDestroyTensorDesc(zDesc));
    CHECK_HIEDNN(hiednnDestroyCudaHandle(handle));

    CHECK_CUDA(cudaFree(dx));
    CHECK_CUDA(cudaFree(dy));
    CHECK_CUDA(cudaFree(dcond));
    CHECK_CUDA(cudaFree(dz));

    return 0;
}

