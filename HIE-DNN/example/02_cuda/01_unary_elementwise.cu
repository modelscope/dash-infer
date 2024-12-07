/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    01_unary_elementwise.cu
 */

#include <hiednn.h>
#include <hiednn_cuda.h>

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>

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

bool CheckFP32(const float &ref, const float &y) {
    return std::fabs(y - ref) <= std::fabs(ref) * 1e-5f + 1e-5f;
}

void UnaryMath() {
    /*
     * input:
     * tensor x: dim={4, 5, 6}, dataType=float
     *
     * output:
     * tensor y: dim={4, 5, 6}, dataType=float
     *
     * y[i] = alpha * exp(x[i]) + y[i] * beta
     */
    int64_t dim[] = {4, 5, 6};
    int nDims = 3;
    int size = 4 * 5 * 6;

    hiednnDataType_t dtype = HIEDNN_DATATYPE_FP32;
    hiednnUnaryEltwiseOp_t op = HIEDNN_UNARY_MATH_EXP;
    float alpha = 1.5f;
    float beta = 2.0f;

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
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(xDesc, dtype, nDims, dim));
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(yDesc, dtype, nDims, dim));

    // host memory for tensor
    float *hx = static_cast<float *>(malloc(size * sizeof(float)));
    float *hy = static_cast<float *>(malloc(size * sizeof(float)));

    // init input tensor
    for (int i = 0; i < size; ++i) {
        hx[i] = 0.01f * static_cast<float>(i);
        hy[i] = 0.02f * static_cast<float>(i);
    }

    // allocate device memory for tensor and copy input tensor to device
    float *dx, *dy;
    CHECK_CUDA(cudaMalloc(&dx, size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dy, size * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dx, hx, size * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dy, hy, size * sizeof(float),
                          cudaMemcpyHostToDevice));

    printf("HIEDNN_UNARY_MATH_EXP\n");
    CHECK_HIEDNN(hiednnCudaUnaryElementwiseOp(
        handle, op, &alpha, xDesc, dx, nullptr, nullptr, &beta, yDesc, dy));

    float *ret = static_cast<float *>(malloc(size * sizeof(float)));

    // copy output tensor from device to host
    CHECK_CUDA(cudaMemcpy(ret, dy, size * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // check output
    printf("check output tenosr... ");
    for (int i = 0; i < size; ++i) {
        float ref = alpha * std::exp(hx[i]) + hy[i] * beta;
        if (!CheckFP32(ref, ret[i])) {
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
    free(ret);
}

void UnaryMathWithParameter() {
    /*
     * input:
     * tensor x: dim={4, 5, 6}, dataType=float
     *
     * output:
     * tensor y: dim={4, 5, 6}, dataType=float
     *
     * y[i] = alpha * (x[i] + 2.f) + y[i] * beta
     */
    int64_t dim[] = {4, 5, 6};
    int nDims = 3;
    int size = 4 * 5 * 6;

    hiednnDataType_t dtype = HIEDNN_DATATYPE_FP32;
    hiednnUnaryEltwiseOp_t op = HIEDNN_UNARY_MATH_ADD;
    float extParam1 = 2.f;
    float alpha = 1.5f;
    float beta = 2.0f;

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
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(xDesc, dtype, nDims, dim));
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(yDesc, dtype, nDims, dim));

    // host memory for tensor
    float *hx = static_cast<float *>(malloc(size * sizeof(float)));
    float *hy = static_cast<float *>(malloc(size * sizeof(float)));

    // init input tensor
    for (int i = 0; i < size; ++i) {
        hx[i] = 0.01f * static_cast<float>(i);
        hy[i] = 0.02f * static_cast<float>(i);
    }

    // allocate device memory for tensor and copy input tensor to device
    float *dx, *dy;
    CHECK_CUDA(cudaMalloc(&dx, size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dy, size * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dx, hx, size * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dy, hy, size * sizeof(float),
                          cudaMemcpyHostToDevice));

    printf("HIEDNN_UNARY_MATH_ADD\n");
    CHECK_HIEDNN(hiednnCudaUnaryElementwiseOp(
        handle, op, &alpha, xDesc, dx, &extParam1, nullptr, &beta, yDesc, dy));

    float *ret = static_cast<float *>(malloc(size * sizeof(float)));

    // copy output tensor from device to host
    CHECK_CUDA(cudaMemcpy(ret, dy, size * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // check output
    printf("check output tenosr... ");
    for (int i = 0; i < size; ++i) {
        float ref = alpha * (hx[i] + extParam1) + hy[i] * beta;
        if (!CheckFP32(ref, ret[i])) {
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
    free(ret);
}

void UnaryBoolean() {
    /*
     * input:
     * tensor x: dim={4, 5, 6}, dataType=float
     *
     * output:
     * tensor y: dim={4, 5, 6}, dataType=bool
     *
     * y[i] = isinf(x[i])
     */
    int64_t dim[] = {4, 5, 6};
    int nDims = 3;
    int size = 4 * 5 * 6;

    hiednnDataType_t xtype = HIEDNN_DATATYPE_FP32;
    hiednnDataType_t ytype = HIEDNN_DATATYPE_BOOL;
    hiednnUnaryEltwiseOp_t op = HIEDNN_UNARY_MATH_ISINF;

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
    char *hy = static_cast<char *>(malloc(size * sizeof(char )));

    // init input tensor
    for (int i = 0; i < size; ++i) {
        uint32_t inf = 0x7f800000;
        hx[i] = i % 2 == 0 ? 0.01f * static_cast<float>(i) :
                             reinterpret_cast<float &>(inf);
    }

    // allocate device memory for tensor and copy input tensor to device
    float *dx;
    char *dy;
    CHECK_CUDA(cudaMalloc(&dx, size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dy, size * sizeof(char)));
    CHECK_CUDA(cudaMemcpy(dx, hx, size * sizeof(float),
                          cudaMemcpyHostToDevice));

    printf("HIEDNN_UNARY_MATH_ISINF\n");
    CHECK_HIEDNN(hiednnCudaUnaryElementwiseOp(
        handle, op, nullptr, xDesc, dx, nullptr, nullptr, nullptr, yDesc, dy));

    // copy output tensor from device to host
    CHECK_CUDA(cudaMemcpy(hy, dy, size * sizeof(char),
                          cudaMemcpyDeviceToHost));

    // check output
    printf("check output tenosr... ");
    for (int i = 0; i < size; ++i) {
        char ref = std::isinf(hx[i]);
        if (hy[i] != ref) {
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
}

int main() {
    UnaryMath();
    UnaryMathWithParameter();
    UnaryBoolean();

    return 0;
}

