/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    03_cast.cu
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
     * tensor x: dim={4, 5, 6}, dataType=float
     *
     * output:
     * tensor y: dim={4, 5, 6}, dataType=int32_t
     *
     * y[i] = static_cast<int32_t>(x[i])
     */
    int64_t dim[] = {4, 5, 6};
    int nDims = 3;
    int size = 4 * 5 * 6;

    hiednnDataType_t xtype = HIEDNN_DATATYPE_FP32;
    hiednnDataType_t ytype = HIEDNN_DATATYPE_INT32;

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
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(xDesc, xtype, nDims, dim));
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(yDesc, ytype, nDims, dim));

    // host memory for tensor
    float *hx = static_cast<float *>(malloc(size * sizeof(float)));
    int32_t *hy = static_cast<int32_t *>(malloc(size * sizeof(int32_t)));

    // init input tensor
    for (int i = 0; i < size; ++i) {
        hx[i] = 0.5f * static_cast<float>(i);
    }

    // allocate device memory for tensor and copy input tensor to device
    float *dx;
    int32_t *dy;
    CHECK_CUDA(cudaMalloc(&dx, size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dy, size * sizeof(int32_t)));
    CHECK_CUDA(cudaMemcpy(dx, hx, size * sizeof(float),
                          cudaMemcpyHostToDevice));

    CHECK_HIEDNN(hiednnCudaCast(handle, xDesc, dx, yDesc, dy));

    // copy output tensor from device to host
    CHECK_CUDA(cudaMemcpy(hy, dy, size * sizeof(int32_t),
                          cudaMemcpyDeviceToHost));

    // check output
    printf("check output tenosr... ");
    for (int i = 0; i < size; ++i) {
        if (hy[i] != static_cast<int32_t>(hx[i])) {
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

    free(hx);
    free(hy);

    return 0;
}

