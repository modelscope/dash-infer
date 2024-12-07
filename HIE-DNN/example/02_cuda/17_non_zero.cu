/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    17_non_zero.cu
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
    hiednnStatus_t err; \
    if ((err = (expr)) != HIEDNN_STATUS_SUCCESS) { \
        int line = __LINE__; \
        printf("hiednn error %d at %d\n", err, line); \
        exit(1); \
    } \
}

void NonZero() {
    /*
     * input:
     * tensor x: dim={2, 2}, dataType=float
     *
     * output:
     * tensor y: dim={8} (actual shape: {2, 3}), dataType=uint64_t
     */
    // input:
    float x[] = {1, 0,
                 1, 1};

    // expected output:
    uint64_t yref[] = {0, 1, 1,
                       0, 0, 1};

    int64_t xDim[] = {2, 2};
    int xNDims = 2;
    int xSize = 2 * 2;

    int64_t yDim[] = {xNDims * xSize};
    int yNDims = 1;
    int ySize = xNDims * xSize;

    hiednnDataType_t dataType = HIEDNN_DATATYPE_FP32;
    hiednnDataType_t indexType = HIEDNN_DATATYPE_UINT64;

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
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(xDesc, dataType, xNDims, xDim));
    // tensor y reserve enough space
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(yDesc, indexType, yNDims, yDim));

    // allocate device memory for tensors and copy input tensors to device
    float *dX;
    uint64_t *dY;
    size_t *dCountPtr;
    void *workspace;

    CHECK_CUDA(cudaMalloc(&dX, xSize * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dY, ySize * sizeof(uint64_t)));
    CHECK_CUDA(cudaMalloc(&dCountPtr, sizeof(size_t)));

    size_t workspaceSize = hiednnCudaNonZeroGetWorkspaceSize(xDesc);
    CHECK_CUDA(cudaMalloc(&workspace, workspaceSize));

    CHECK_CUDA(cudaMemcpy(dX, x, xSize * sizeof(float),
                          cudaMemcpyHostToDevice));

    CHECK_HIEDNN(hiednnCudaNonZero(
        handle, xDesc, dX, yDesc, dY, dCountPtr, workspace, workspaceSize));

    // retrieve count of non-zero elements
    size_t hCount;
    CHECK_CUDA(cudaMemcpy(&hCount, dCountPtr, sizeof(size_t),
                          cudaMemcpyDeviceToHost));

    uint64_t *y = static_cast<uint64_t *>(malloc(
        xNDims * hCount * sizeof(uint64_t)));

    // copy output tensor from device to host
    CHECK_CUDA(cudaMemcpy(y, dY, xNDims * hCount * sizeof(uint64_t),
                          cudaMemcpyDeviceToHost));

    printf("check count of elements... ");
    if (hCount != 3) {
        printf("FAILED\n");
        exit(1);
    }
    printf("OK\n");

    printf("check output tenosr... ");
    for (int i = 0; i < xSize; ++i) {
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
    CHECK_CUDA(cudaFree(dCountPtr));
    CHECK_CUDA(cudaFree(workspace));

    free(y);
}

void FastNonZero() {
    /*
     * input:
     * tensor x: dim={2, 2}, dataType=float
     *
     * output:
     * tensor y: dim={2, 4} (valid length of each row is 3), dataType=uint64_t
     */
    // input:
    float x[] = {1, 0,
                 1, 1};

    // expected output:
    // elements beyond the count of non-zero element in each dim is invalid
    uint64_t yref[] = {0, 1, 1, 0xbad,
                       0, 0, 1, 0xbad};

    int64_t xDim[] = {2, 2};
    int xNDims = 2;
    int xSize = 2 * 2;

    int64_t yDim[] = {xNDims, xSize};
    int yNDims = xNDims;
    int ySize = xNDims * xSize;

    hiednnDataType_t dataType = HIEDNN_DATATYPE_FP32;
    hiednnDataType_t indexType = HIEDNN_DATATYPE_UINT64;

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
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(xDesc, dataType, xNDims, xDim));
    // tensor y reserve enough space
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(yDesc, indexType, yNDims, yDim));

    // allocate device memory for tensors and copy input tensors to device
    float *dX;
    uint64_t *dY;
    size_t *dCountPtr;

    CHECK_CUDA(cudaMalloc(&dX, xSize * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dY, ySize * sizeof(uint64_t)));
    CHECK_CUDA(cudaMalloc(&dCountPtr, sizeof(size_t)));

    CHECK_CUDA(cudaMemcpy(dX, x, xSize * sizeof(float),
                          cudaMemcpyHostToDevice));

    CHECK_HIEDNN(hiednnCudaFastNonZero(
        handle, xDesc, dX, yDesc, dY, dCountPtr));

    uint64_t *y = static_cast<uint64_t *>(malloc(ySize * sizeof(uint64_t)));

    // copy output tensor from device to host
    CHECK_CUDA(cudaMemcpy(y, dY, ySize * sizeof(uint64_t),
                          cudaMemcpyDeviceToHost));

    // retrieve count of non-zero elements
    size_t hCount;
    CHECK_CUDA(cudaMemcpy(&hCount, dCountPtr, sizeof(size_t),
                          cudaMemcpyDeviceToHost));

    printf("check count of elements... ");
    if (hCount != 3) {
        printf("FAILED\n");
        exit(1);
    }
    printf("OK\n");

    printf("check output tenosr... ");
    for (int d = 0; d < xNDims; ++d) {
        for (int i = 0; i < hCount; ++i) {
            if (y[d * xSize + i] != yref[d * xSize + i]) {
                printf("FAILED\n");
                exit(1);
            }
        }
    }
    printf("OK\n");

    CHECK_HIEDNN(hiednnDestroyTensorDesc(xDesc));
    CHECK_HIEDNN(hiednnDestroyTensorDesc(yDesc));
    CHECK_HIEDNN(hiednnDestroyCudaHandle(handle));

    CHECK_CUDA(cudaFree(dX));
    CHECK_CUDA(cudaFree(dY));
    CHECK_CUDA(cudaFree(dCountPtr));

    free(y);
}

int main() {
    NonZero();
    FastNonZero();
    return 0;
}
