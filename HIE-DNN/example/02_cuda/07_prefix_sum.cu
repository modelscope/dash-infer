/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    07_prefix_sum.cu
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

void PrefixSum() {
    /*
     * input:
     * tensor x: dim={4, 4}, dataType=int32_t
     * param axis=0
     * param exclusive=0
     * param reverse=0
     *
     * output:
     * tensor y: dim=x.dim, dataType=int32_t
     */
    // input:
    int32_t x[] = {0,  1,  2,  3,
                   4,  5,  6,  7,
                   8,  9,  10, 11,
                   12, 13, 14, 15};
    int axis = 0;
    int exclusive = 0;
    int reverse = 0;

    // expected output
    int32_t yref[] = {0,  1,  2,  3,
                      4,  6,  8,  10,
                      12, 15, 18, 21,
                      24, 28, 32, 36};

    int64_t dataDim[] = {4, 4};
    int dataNDims = 2;
    int dataSize = 4 * 4;

    hiednnDataType_t dataType = HIEDNN_DATATYPE_INT32;

    // create cuda handle
    hiednnCudaHandle_t handle;
    CHECK_HIEDNN(hiednnCreateCudaHandle(&handle));

    // default stream
    CHECK_HIEDNN(hiednnSetCudaStream(handle, 0));

    // create tensor descriptor
    hiednnTensorDesc_t dataDesc;
    CHECK_HIEDNN(hiednnCreateTensorDesc(&dataDesc));

    // init tensor descriptor
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(
        dataDesc, dataType, dataNDims, dataDim));

    // allocate device memory for tensor and copy input tensor to device
    int32_t *dx, *dy;
    CHECK_CUDA(cudaMalloc(&dx, dataSize * sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&dy, dataSize * sizeof(int32_t)));

    CHECK_CUDA(cudaMemcpy(dx, x, dataSize * sizeof(int32_t),
                          cudaMemcpyHostToDevice));

    printf("PrefixSum\n");
    CHECK_HIEDNN(hiednnCudaPrefixSum(
        handle, dataDesc, dx, axis, exclusive, reverse, dataDesc, dy));

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
    CHECK_HIEDNN(hiednnDestroyCudaHandle(handle));

    CHECK_CUDA(cudaFree(dx));
    CHECK_CUDA(cudaFree(dy));

    free(y);
}

void ExclusivePrefixSum() {
    /*
     * input:
     * tensor x: dim={4, 4}, dataType=int32_t
     * param axis=0
     * param exclusive=1
     * param reverse=0
     *
     * output:
     * tensor y: dim=x.dim, dataType=int32_t
     */
    // input:
    int32_t x[] = {0,  1,  2,  3,
                   4,  5,  6,  7,
                   8,  9,  10, 11,
                   12, 13, 14, 15};
    int axis = 0;
    int exclusive = 1;
    int reverse = 0;

    // expected output
    int32_t yref[] = {0,  0,  0,  0,
                      0,  1,  2,  3,
                      4,  6,  8,  10,
                      12, 15, 18, 21};

    int64_t dataDim[] = {4, 4};
    int dataNDims = 2;
    int dataSize = 4 * 4;

    hiednnDataType_t dataType = HIEDNN_DATATYPE_INT32;

    // create cuda handle
    hiednnCudaHandle_t handle;
    CHECK_HIEDNN(hiednnCreateCudaHandle(&handle));

    // default stream
    CHECK_HIEDNN(hiednnSetCudaStream(handle, 0));

    // create tensor descriptor
    hiednnTensorDesc_t dataDesc;
    CHECK_HIEDNN(hiednnCreateTensorDesc(&dataDesc));

    // init tensor descriptor
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(
        dataDesc, dataType, dataNDims, dataDim));

    // allocate device memory for tensor and copy input tensor to device
    int32_t *dx, *dy;
    CHECK_CUDA(cudaMalloc(&dx, dataSize * sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&dy, dataSize * sizeof(int32_t)));

    CHECK_CUDA(cudaMemcpy(dx, x, dataSize * sizeof(int32_t),
                          cudaMemcpyHostToDevice));

    printf("ExclusivePrefixSum\n");
    CHECK_HIEDNN(hiednnCudaPrefixSum(
        handle, dataDesc, dx, axis, exclusive, reverse, dataDesc, dy));

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
    CHECK_HIEDNN(hiednnDestroyCudaHandle(handle));

    CHECK_CUDA(cudaFree(dx));
    CHECK_CUDA(cudaFree(dy));

    free(y);
}

void SuffixSum() {
    /*
     * input:
     * tensor x: dim={4, 4}, dataType=int32_t
     * param axis=0
     * param exclusive=0
     * param reverse=1
     *
     * output:
     * tensor y: dim=x.dim, dataType=int32_t
     */
    // input:
    int32_t x[] = {0,  1,  2,  3,
                   4,  5,  6,  7,
                   8,  9,  10, 11,
                   12, 13, 14, 15};
    int axis = 0;
    int exclusive = 0;
    int reverse = 1;

    // expected output
    int32_t yref[] = {24, 28, 32, 36,
                      24, 27, 30, 33,
                      20, 22, 24, 26,
                      12, 13, 14, 15};

    int64_t dataDim[] = {4, 4};
    int dataNDims = 2;
    int dataSize = 4 * 4;

    hiednnDataType_t dataType = HIEDNN_DATATYPE_INT32;

    // create cuda handle
    hiednnCudaHandle_t handle;
    CHECK_HIEDNN(hiednnCreateCudaHandle(&handle));

    // default stream
    CHECK_HIEDNN(hiednnSetCudaStream(handle, 0));

    // create tensor descriptor
    hiednnTensorDesc_t dataDesc;
    CHECK_HIEDNN(hiednnCreateTensorDesc(&dataDesc));

    // init tensor descriptor
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(
        dataDesc, dataType, dataNDims, dataDim));

    // allocate device memory for tensor and copy input tensor to device
    int32_t *dx, *dy;
    CHECK_CUDA(cudaMalloc(&dx, dataSize * sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&dy, dataSize * sizeof(int32_t)));

    CHECK_CUDA(cudaMemcpy(dx, x, dataSize * sizeof(int32_t),
                          cudaMemcpyHostToDevice));

    printf("SuffixSum\n");
    CHECK_HIEDNN(hiednnCudaPrefixSum(
        handle, dataDesc, dx, axis, exclusive, reverse, dataDesc, dy));

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
    CHECK_HIEDNN(hiednnDestroyCudaHandle(handle));

    CHECK_CUDA(cudaFree(dx));
    CHECK_CUDA(cudaFree(dy));

    free(y);
}

int main() {
    PrefixSum();
    ExclusivePrefixSum();
    SuffixSum();
    return 0;
}


