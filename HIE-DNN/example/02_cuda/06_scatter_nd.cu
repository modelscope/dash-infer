/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    06_scatter_nd.cu
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

void ScatterND1() {
    /*
     * input:
     * tensor x: dim={4, 4}, dataType=int32_t
     * tensor indices: dim={2, 1}, dataType=int32_t
     * tensor updates: dim={2, 4}, dataType=int32_t
     *
     * output:
     * tensor y: dim=x.dim, dataType=int32_t
     */
    // input:
    int32_t x[] = {1, 1, 1, 1,
                   2, 2, 2, 2,
                   3, 3, 3, 3,
                   4, 4, 4, 4};
    int32_t indices[] = {2,
                         0};
    int32_t updates[] = {5, 5, 5, 5,
                         6, 6, 6, 6};

    // expected output
    int32_t yref[] = {6, 6, 6, 6,
                      2, 2, 2, 2,
                      5, 5, 5, 5,
                      4, 4, 4, 4};

    int64_t dataDim[] = {4, 4};
    int dataNDims = 2;
    int dataSize = 4 * 4;
    int64_t indicesDim[] = {2, 1};
    int indicesNDims = 2;
    int indicesSize = 2 * 1;
    int64_t updatesDim[] = {2, 4};
    int updatesNDims = 2;
    int updatesSize = 2 * 4;

    hiednnDataType_t dataType = HIEDNN_DATATYPE_INT32;
    hiednnDataType_t indicesType = HIEDNN_DATATYPE_INT32;

    // create cuda handle
    hiednnCudaHandle_t handle;
    CHECK_HIEDNN(hiednnCreateCudaHandle(&handle));

    // default stream
    CHECK_HIEDNN(hiednnSetCudaStream(handle, 0));

    // create tensor descriptor
    hiednnTensorDesc_t dataDesc, indicesDesc, updatesDesc;
    CHECK_HIEDNN(hiednnCreateTensorDesc(&dataDesc));
    CHECK_HIEDNN(hiednnCreateTensorDesc(&indicesDesc));
    CHECK_HIEDNN(hiednnCreateTensorDesc(&updatesDesc));

    // init tensor descriptor
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(
        dataDesc, dataType, dataNDims, dataDim));
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(
        indicesDesc, indicesType, indicesNDims, indicesDim));
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(
        updatesDesc, dataType, updatesNDims, updatesDim));

    // allocate device memory for tensor and copy input tensor to device
    int32_t *dx, *dIndices, *dUpdates, *dy;
    CHECK_CUDA(cudaMalloc(&dx, dataSize * sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&dIndices, indicesSize * sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&dUpdates, updatesSize * sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&dy, dataSize * sizeof(int32_t)));

    CHECK_CUDA(cudaMemcpy(dx, x, dataSize * sizeof(int32_t),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dIndices, indices, indicesSize * sizeof(int32_t),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dUpdates, updates, updatesSize * sizeof(int32_t),
                          cudaMemcpyHostToDevice));

    CHECK_HIEDNN(hiednnCudaScatterND(
        handle, dataDesc, dx, indicesDesc, dIndices,
        updatesDesc, dUpdates, dataDesc, dy));

    int32_t *y = static_cast<int32_t *>(malloc(dataSize * sizeof(int32_t)));

    // copy output tensor from device to host
    CHECK_CUDA(cudaMemcpy(y, dy, dataSize * sizeof(int32_t),
                          cudaMemcpyDeviceToHost));

    // check output
    printf("check output tenosr... ");
    for (int i = 0; i < dataSize; ++i) {
        if (y[i] != yref[i]) {
            printf("FAILED\n");
            exit(1);
        }
    }
    printf("OK\n");

    CHECK_HIEDNN(hiednnDestroyTensorDesc(dataDesc));
    CHECK_HIEDNN(hiednnDestroyTensorDesc(indicesDesc));
    CHECK_HIEDNN(hiednnDestroyTensorDesc(updatesDesc));
    CHECK_HIEDNN(hiednnDestroyCudaHandle(handle));

    CHECK_CUDA(cudaFree(dx));
    CHECK_CUDA(cudaFree(dIndices));
    CHECK_CUDA(cudaFree(dUpdates));
    CHECK_CUDA(cudaFree(dy));

    free(y);
}

void ScatterND2() {
    /*
     * input:
     * tensor x: dim={4, 4}, dataType=int32_t
     * tensor indices: dim={5, 2}, dataType=int32_t
     * tensor updates: dim={5}, dataType=int32_t
     *
     * output:
     * tensor y: dim=x.dim, dataType=int32_t
     */
    // input:
    int32_t x[] = {1, 1, 1, 1,
                   2, 2, 2, 2,
                   3, 3, 3, 3,
                   4, 4, 4, 4};
    int32_t indices[] = {1, 0,
                         0, 1,
                         0, 2,
                         2, 2,
                         3, 0};
    int32_t updates[] = {5, 6, 7, 8, 9};

    // expected output
    int32_t yref[] = {1, 6, 7, 1,
                      5, 2, 2, 2,
                      3, 3, 8, 3,
                      9, 4, 4, 4};

    int64_t dataDim[] = {4, 4};
    int dataNDims = 2;
    int dataSize = 4 * 4;
    int64_t indicesDim[] = {5, 2};
    int indicesNDims = 2;
    int indicesSize = 5 * 2;
    int64_t updatesDim[] = {5};
    int updatesNDims = 1;
    int updatesSize = 5;

    hiednnDataType_t dataType = HIEDNN_DATATYPE_INT32;
    hiednnDataType_t indicesType = HIEDNN_DATATYPE_INT32;

    // create cuda handle
    hiednnCudaHandle_t handle;
    CHECK_HIEDNN(hiednnCreateCudaHandle(&handle));

    // default stream
    CHECK_HIEDNN(hiednnSetCudaStream(handle, 0));

    // create tensor descriptor
    hiednnTensorDesc_t dataDesc, indicesDesc, updatesDesc;
    CHECK_HIEDNN(hiednnCreateTensorDesc(&dataDesc));
    CHECK_HIEDNN(hiednnCreateTensorDesc(&indicesDesc));
    CHECK_HIEDNN(hiednnCreateTensorDesc(&updatesDesc));

    // init tensor descriptor
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(
        dataDesc, dataType, dataNDims, dataDim));
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(
        indicesDesc, indicesType, indicesNDims, indicesDim));
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(
        updatesDesc, dataType, updatesNDims, updatesDim));

    // allocate device memory for tensor and copy input tensor to device
    int32_t *dx, *dIndices, *dUpdates, *dy;
    CHECK_CUDA(cudaMalloc(&dx, dataSize * sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&dIndices, indicesSize * sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&dUpdates, updatesSize * sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&dy, dataSize * sizeof(int32_t)));

    CHECK_CUDA(cudaMemcpy(dx, x, dataSize * sizeof(int32_t),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dIndices, indices, indicesSize * sizeof(int32_t),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dUpdates, updates, updatesSize * sizeof(int32_t),
                          cudaMemcpyHostToDevice));

    CHECK_HIEDNN(hiednnCudaScatterND(
        handle, dataDesc, dx, indicesDesc, dIndices,
        updatesDesc, dUpdates, dataDesc, dy));

    int32_t *y = static_cast<int32_t *>(malloc(dataSize * sizeof(int32_t)));

    // copy output tensor from device to host
    CHECK_CUDA(cudaMemcpy(y, dy, dataSize * sizeof(int32_t),
                          cudaMemcpyDeviceToHost));

    // check output
    printf("check output tenosr... ");
    for (int i = 0; i < dataSize; ++i) {
        if (y[i] != yref[i]) {
            printf("FAILED\n");
            exit(1);
        }
    }
    printf("OK\n");

    CHECK_HIEDNN(hiednnDestroyTensorDesc(dataDesc));
    CHECK_HIEDNN(hiednnDestroyTensorDesc(indicesDesc));
    CHECK_HIEDNN(hiednnDestroyTensorDesc(updatesDesc));
    CHECK_HIEDNN(hiednnDestroyCudaHandle(handle));

    CHECK_CUDA(cudaFree(dx));
    CHECK_CUDA(cudaFree(dIndices));
    CHECK_CUDA(cudaFree(dUpdates));
    CHECK_CUDA(cudaFree(dy));

    free(y);
}

int main() {
    ScatterND1();
    ScatterND2();
    return 0;
}


