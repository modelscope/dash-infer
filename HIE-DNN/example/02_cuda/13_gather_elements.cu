/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    13_gather_elements.cu
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

void GatherElementsAlongAxis() {
    /*
     * input:
     * tensor x: dim={2, 2}, dataType=int32_t
     * tensor indices: dim={2, 2}, dataType=int64_t
     * param axis=1
     *
     * output:
     * tensor y: dim={2, 2}, dataType=int32_t
     */
    // input:
    int32_t x[] = {1, 2,
                   3, 4};
    int64_t indices[] = {0, 0,
                         1, 0};
    int axis = 1;

    // expected output:
    int32_t yref[] = {1, 1,
                      4, 3};

    int64_t xDim[] = {2, 2};
    int xNDims = 2;
    int xSize = 2 * 2;
    int64_t yDim[] = {2, 2};
    int yNDims = 2;
    int ySize = 2 * 2;
    hiednnDataType_t dataType = HIEDNN_DATATYPE_INT32;
    hiednnDataType_t indexType = HIEDNN_DATATYPE_INT64;

    // create CUDA handle
    hiednnCudaHandle_t handle;
    CHECK_HIEDNN(hiednnCreateCudaHandle(&handle));

    // default stream
    CHECK_HIEDNN(hiednnSetCudaStream(handle, 0));

    // create tensor descriptors
    hiednnTensorDesc_t xDesc, yDesc, indicesDesc;
    CHECK_HIEDNN(hiednnCreateTensorDesc(&xDesc));
    CHECK_HIEDNN(hiednnCreateTensorDesc(&yDesc));
    CHECK_HIEDNN(hiednnCreateTensorDesc(&indicesDesc));

    // init tensor descriptors
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(xDesc, dataType, xNDims, xDim));
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(yDesc, dataType, yNDims, yDim));
    // tensor indices is of the same shape as tensor y
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(indicesDesc, indexType,
                                           yNDims, yDim));

    // allocate device memory for tensors and copy input tensors to device
    int32_t *dX, *dY;
    int64_t *dIndices;
    CHECK_CUDA(cudaMalloc(&dX, xSize * sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&dY, ySize * sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&dIndices, ySize * sizeof(int64_t)));
    CHECK_CUDA(cudaMemcpy(dX, x, xSize * sizeof(int32_t),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dIndices, indices, ySize * sizeof(int64_t),
                          cudaMemcpyHostToDevice));

    CHECK_HIEDNN(hiednnCudaGatherElements(handle, xDesc, dX, indicesDesc,
                                          dIndices, yDesc, dY, axis));

    int32_t *y = static_cast<int32_t *>(malloc(ySize * sizeof(int32_t)));

    // copy output tensor from device to host
    CHECK_CUDA(cudaMemcpy(y, dY, ySize * sizeof(int32_t),
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
    CHECK_HIEDNN(hiednnDestroyTensorDesc(indicesDesc));
    CHECK_HIEDNN(hiednnDestroyCudaHandle(handle));

    CHECK_CUDA(cudaFree(dX));
    CHECK_CUDA(cudaFree(dY));
    CHECK_CUDA(cudaFree(dIndices));

    free(y);
}

int main() {
    GatherElementsAlongAxis();
    return 0;
}
