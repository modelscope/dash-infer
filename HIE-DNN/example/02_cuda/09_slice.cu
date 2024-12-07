/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    09_slice.cu
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
     * tensor x: dim={2, 8}, dataType=int32_t
     * param starts={1}
     * param ends={7}
     * param steps={2}
     * param axes={1}
     *
     * output:
     * tensor y: dim={3, 2, 2}, dataType=int32_t
     */
    // input:
    int32_t x[] = {0, 1, 2, 3, 4, 5, 6, 7,
                   8, 9, 10, 11, 12, 13, 14, 15};
    int64_t starts[] = {1};
    int64_t ends[] = {7};
    int64_t steps[] = {2};
    int axes[] = {1};
    int nParams = 1;

    // expected output:
    int32_t yref[] = {1, 3, 5,
                      9, 11, 13};

    int64_t xDim[] = {2, 8};
    int xNDims = 2;
    int xSize = 2 * 8;
    int64_t yDim[] = {2, 3};
    int yNDims = 2;
    int ySize = 2 * 3;
    hiednnDataType_t dataType = HIEDNN_DATATYPE_INT32;

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
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(xDesc, dataType, xNDims, xDim));
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(yDesc, dataType, yNDims, yDim));

    // allocate device memory for tensor and copy input tensor to device
    int32_t *dx, *dy;
    CHECK_CUDA(cudaMalloc(&dx, xSize * sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&dy, ySize * sizeof(int32_t)));
    CHECK_CUDA(cudaMemcpy(dx, x, xSize * sizeof(int32_t),
                          cudaMemcpyHostToDevice));

    CHECK_HIEDNN(hiednnCudaSlice(
        handle, xDesc, dx, starts, ends, steps, axes, nParams, yDesc, dy));

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

    CHECK_HIEDNN(hiednnDestroyTensorDesc(xDesc));
    CHECK_HIEDNN(hiednnDestroyTensorDesc(yDesc));
    CHECK_HIEDNN(hiednnDestroyCudaHandle(handle));

    CHECK_CUDA(cudaFree(dx));
    CHECK_CUDA(cudaFree(dy));

    free(y);

    return 0;
}

