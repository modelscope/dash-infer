/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    08_reduce.cu
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

void Reduce() {
    /*
     * input:
     * tensor x: dim={4, 4}, dataType=int8_t
     * param reduceOp=HIEDNN_REDUCE_MAX
     * param axis=0
     * param alpha=2
     *
     * output:
     * tensor y: dim={1, 4}, dataType=int8_t
     */
    // input:
    int8_t x[] = {12, 1,  2,  3,
                  4,  13, 6,  7,
                  8,  9,  14, 11,
                  0,  5,  10, 15};
    hiednnReduceOp_t reduceOp = HIEDNN_REDUCE_MAX;
    int8_t alpha = 2;
    int axis = 0;

    // expected output
    int8_t yref[] = {24, 26, 28, 30};

    int64_t xDim[] = {4, 4};
    int xNDim = 2;
    int xSize = 4 * 4;
    int64_t yDim[] = {1, 4};
    int yNDim = 2;
    int ySize = 1 * 4;

    hiednnDataType_t dataType = HIEDNN_DATATYPE_INT8;

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
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(xDesc, dataType, xNDim, xDim));
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(yDesc, dataType, yNDim, yDim));

    // allocate device memory for tensor and copy input tensor to device
    int8_t *dx, *dy;
    CHECK_CUDA(cudaMalloc(&dx, xSize * sizeof(int8_t)));
    CHECK_CUDA(cudaMalloc(&dy, ySize * sizeof(int8_t)));

    CHECK_CUDA(cudaMemcpy(dx, x, xSize * sizeof(int8_t),
                          cudaMemcpyHostToDevice));

    // indexType is senseless for reduction without index,
    // just for matching the parameter list
    hiednnDataType_t indexType = HIEDNN_DATATYPE_INT8;

    printf("Reduce\n");
    CHECK_HIEDNN(hiednnCudaReduce(
        handle, reduceOp, &alpha, xDesc, dx, axis, yDesc, dy,
        indexType, nullptr));

    int8_t *y = static_cast<int8_t *>(malloc(ySize * sizeof(int8_t)));

    // copy output tensor from device to host
    CHECK_CUDA(cudaMemcpy(y, dy, ySize * sizeof(int8_t),
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
}

void ReduceWithIndex() {
    /*
     * input:
     * tensor x: dim={4, 4}, dataType=int8_t
     * param reduceOp=HIEDNN_REDUCE_MAX
     * param axis=0
     * param alpha=2
     *
     * output:
     * tensor y: dim={1, 4}, dataType=int8_t
     * tensor index: dim={1, 4}, dataType=uint32_t
     */
    // input:
    int8_t x[] = {12, 1,  2,  3,
                  4,  13, 6,  7,
                  8,  9,  14, 11,
                  0,  5,  10, 15};
    hiednnReduceOp_t reduceOp = HIEDNN_REDUCE_MAX;
    int8_t alpha = 2;
    int axis = 0;

    // expected output
    int8_t yref[] = {24, 26, 28, 30};
    uint32_t indexRef[] = {0, 1, 2, 3};

    int64_t xDim[] = {4, 4};
    int xNDim = 2;
    int xSize = 4 * 4;
    int64_t yDim[] = {1, 4};
    int yNDim = 2;
    int ySize = 1 * 4;

    hiednnDataType_t dataType = HIEDNN_DATATYPE_INT8;
    hiednnDataType_t indexType = HIEDNN_DATATYPE_UINT32;

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
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(xDesc, dataType, xNDim, xDim));
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(yDesc, dataType, yNDim, yDim));

    // allocate device memory for tensor and copy input tensor to device
    int8_t *dx, *dy;
    uint32_t *dIndex;
    CHECK_CUDA(cudaMalloc(&dx, xSize * sizeof(int8_t)));
    CHECK_CUDA(cudaMalloc(&dy, ySize * sizeof(int8_t)));
    CHECK_CUDA(cudaMalloc(&dIndex, ySize * sizeof(uint32_t)));

    CHECK_CUDA(cudaMemcpy(dx, x, xSize * sizeof(int8_t),
                          cudaMemcpyHostToDevice));

    printf("ReduceWithIndex\n");
    CHECK_HIEDNN(hiednnCudaReduce(
        handle, reduceOp, &alpha, xDesc, dx, axis, yDesc, dy,
        indexType, dIndex));

    int8_t *y = static_cast<int8_t *>(malloc(ySize * sizeof(int8_t)));
    uint32_t *index = static_cast<uint32_t *>(malloc(ySize * sizeof(uint32_t)));

    // copy output tensor from device to host
    CHECK_CUDA(cudaMemcpy(y, dy, ySize * sizeof(int8_t),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(index, dIndex, ySize * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));

    // check output
    printf("check output tenosr... ");
    for (int i = 0; i < ySize; ++i) {
        if (y[i] != yref[i] || index[i] != indexRef[i]) {
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
    CHECK_CUDA(cudaFree(dIndex));

    free(y);
    free(index);
}

void MixedPrecisionReduce() {
    /*
     * input:
     * tensor x: dim={4, 4}, dataType=int8_t
     * param reduceOp=HIEDNN_REDUCE_SUM
     * param axis=0
     * param alpha=2.0
     *
     * output:
     * tensor y: dim={1, 4}, dataType=float
     */
    // input:
    int8_t x[] = {0,  1,  2,  3,
                  4,  5,  6,  7,
                  8,  9,  10, 11,
                  12, 13, 14, 15};
    hiednnReduceOp_t reduceOp = HIEDNN_REDUCE_SUM;
    float alpha = 2.0f;
    int axis = 0;

    // expected output
    float yref[] = {48.f, 56.f, 64.f, 72.f};

    int64_t xDim[] = {4, 4};
    int xNDim = 2;
    int xSize = 4 * 4;
    int64_t yDim[] = {1, 4};
    int yNDim = 2;
    int ySize = 1 * 4;

    hiednnDataType_t xType = HIEDNN_DATATYPE_INT8;
    hiednnDataType_t yType = HIEDNN_DATATYPE_FP32;

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
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(xDesc, xType, xNDim, xDim));
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(yDesc, yType, yNDim, yDim));

    // allocate device memory for tensor and copy input tensor to device
    int8_t *dx;
    float *dy;
    CHECK_CUDA(cudaMalloc(&dx, xSize * sizeof(int8_t)));
    CHECK_CUDA(cudaMalloc(&dy, ySize * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(dx, x, xSize * sizeof(int8_t),
                          cudaMemcpyHostToDevice));

    // indexType is senseless for reduction without index,
    // just for matching the parameter list
    hiednnDataType_t indexType = HIEDNN_DATATYPE_INT8;

    printf("MixedPrecisionReduce\n");
    CHECK_HIEDNN(hiednnCudaReduce(
        handle, reduceOp, &alpha, xDesc, dx, axis, yDesc, dy,
        indexType, nullptr));

    float *y = static_cast<float *>(malloc(ySize * sizeof(float)));

    // copy output tensor from device to host
    CHECK_CUDA(cudaMemcpy(y, dy, ySize * sizeof(float),
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
}

int main() {
    Reduce();
    ReduceWithIndex();
    MixedPrecisionReduce();
    return 0;
}


