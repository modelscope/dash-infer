/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    16_scatter_elements.cu
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

void ScatterElements() {
    /*
     * input:
     * tensor x: dim={3, 3}, dataType=float
     * tensor indices: dim={2, 3}, dataType=int32_t
     * tensor updates: dim={2, 3}, dataType=float
     * param axis=0
     * param reduction=HIEDNN_SCATTERELEM_REDUCE_NONE
     *
     * output:
     * tensor y: dim={3, 3}, dataType=float
     */
    // input:
    float x[] = {0.0f, 0.0f, 0.0f,
                 0.0f, 0.0f, 0.0f,
                 0.0f, 0.0f, 0.0f};
    int32_t indices[] = {1, 0, 2,
                         0, 2, 1};
    float updates[] = {1.0f, 1.1f, 1.2f,
                       2.0f, 2.1f, 2.2f};
    int axis = 0;

    // expected output:
    float yref[] = {2.0f, 1.1f, 0.0f,
                    1.0f, 0.0f, 2.2f,
                    0.0f, 2.1f, 1.2f};

    int64_t xDim[] = {3, 3};
    int xNDims = 2;
    int xSize = 3 * 3;
    int64_t updatesDim[] = {2, 3};
    int updatesNDims = 2;
    int updatesSize = 2 * 3;
    hiednnDataType_t dataType = HIEDNN_DATATYPE_FP32;
    hiednnDataType_t indexType = HIEDNN_DATATYPE_INT32;

    // create CUDA handle
    hiednnCudaHandle_t handle;
    CHECK_HIEDNN(hiednnCreateCudaHandle(&handle));

    // default stream
    CHECK_HIEDNN(hiednnSetCudaStream(handle, 0));

    // create tensor descriptors
    hiednnTensorDesc_t xDesc, yDesc, updatesDesc, indicesDesc;
    CHECK_HIEDNN(hiednnCreateTensorDesc(&xDesc));
    CHECK_HIEDNN(hiednnCreateTensorDesc(&yDesc));
    CHECK_HIEDNN(hiednnCreateTensorDesc(&updatesDesc));
    CHECK_HIEDNN(hiednnCreateTensorDesc(&indicesDesc));

    // init tensor descriptors
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(xDesc, dataType, xNDims, xDim));
    // tensor y is of the same shape as tensor x
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(yDesc, dataType, xNDims, xDim));

    CHECK_HIEDNN(hiednnSetNormalTensorDesc(updatesDesc, dataType,
                                           updatesNDims, updatesDim));
    // tensor indices is of the same shape as tensor updates
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(indicesDesc, indexType,
                                           updatesNDims, updatesDim));

    // allocate device memory for tensors and copy input tensors to device
    float *dX, *dY, *dUpdates;
    int32_t *dIndices;
    CHECK_CUDA(cudaMalloc(&dX, xSize * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dY, xSize * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dUpdates, updatesSize * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dIndices, updatesSize * sizeof(int32_t)));

    CHECK_CUDA(cudaMemcpy(dX, x, xSize * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dUpdates, updates, updatesSize * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dIndices, indices, updatesSize * sizeof(int32_t),
                          cudaMemcpyHostToDevice));

    CHECK_HIEDNN(hiednnCudaScatterElements(
        handle, xDesc, dX, indicesDesc, dIndices, updatesDesc, dUpdates,
        yDesc, dY, axis, HIEDNN_SCATTERELEM_REDUCE_NONE));

    float *y = static_cast<float *>(malloc(xSize * sizeof(float)));

    // copy output tensor from device to host
    CHECK_CUDA(cudaMemcpy(y, dY, xSize * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // check output
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
    CHECK_HIEDNN(hiednnDestroyTensorDesc(updatesDesc));
    CHECK_HIEDNN(hiednnDestroyTensorDesc(indicesDesc));
    CHECK_HIEDNN(hiednnDestroyCudaHandle(handle));

    CHECK_CUDA(cudaFree(dX));
    CHECK_CUDA(cudaFree(dY));
    CHECK_CUDA(cudaFree(dUpdates));
    CHECK_CUDA(cudaFree(dIndices));

    free(y);
}

int main() {
    ScatterElements();
    return 0;
}
