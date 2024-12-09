/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    02_binary_elementwise.cu
 */

#include <hiednn.h>
#include <hiednn_cuda.h>

#include <cmath>
#include <cstdio>

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

void BinaryMath() {
    /*
     * input:
     * tensor x: dim={4, 5, 6}, dataType=float
     * tensor y: dim={4, 5, 6}, dataType=float
     *
     * output:
     * tensor z: dim={4, 5, 6}, dataType=float
     *
     * z[i] = alpha * (x[i] * y[i]) + beta * z[i]
     */
    constexpr int64_t dim[] = {4, 5, 6};
    constexpr int nDims = 3;
    constexpr int size = 4 * 5 * 6;

    constexpr hiednnDataType_t dtype = HIEDNN_DATATYPE_FP32;
    constexpr hiednnBinaryEltwiseOp_t op = HIEDNN_BINARY_MATH_MUL;
    constexpr float alpha = 0.75f;
    constexpr float beta = 0.25f;

    // create cuda handle
    hiednnCudaHandle_t handle;
    CHECK_HIEDNN(hiednnCreateCudaHandle(&handle));

    // default stream
    CHECK_HIEDNN(hiednnSetCudaStream(handle, 0));

    // create tensor descriptors
    hiednnTensorDesc_t xDesc, yDesc, zDesc;
    CHECK_HIEDNN(hiednnCreateTensorDesc(&xDesc));
    CHECK_HIEDNN(hiednnCreateTensorDesc(&yDesc));
    CHECK_HIEDNN(hiednnCreateTensorDesc(&zDesc));

    // init tensor descriptors
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(xDesc, dtype, nDims, dim));
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(yDesc, dtype, nDims, dim));
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(zDesc, dtype, nDims, dim));

    // host memory for tensor
    float *hx = static_cast<float *>(malloc(size * sizeof(float)));
    float *hy = static_cast<float *>(malloc(size * sizeof(float)));
    float *hz = static_cast<float *>(malloc(size * sizeof(float)));

    // init input tensor
    for (int i = 0; i < size; ++i) {
        hx[i] = 0.5f * static_cast<float>(i);
        hy[i] = 0.01f * static_cast<float>(i);
        hz[i] = 0.05f * static_cast<float>(i);
    }

    // allocate device memory for tensor and copy input tensor to device
    float *dx, *dy, *dz;
    CHECK_CUDA(cudaMalloc(&dx, size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dy, size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dz, size * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dx, hx, size * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dy, hy, size * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dz, hz, size * sizeof(float),
                          cudaMemcpyHostToDevice));

    printf("HIEDNN_BINARY_MATH_MUL\n");
    CHECK_HIEDNN(hiednnCudaBinaryElementwiseOp(
        handle, op, &alpha, xDesc, dx, yDesc, dy, nullptr, &beta, zDesc, dz));

    float *ret = static_cast<float *>(malloc(size * sizeof(float)));

    // copy output tensor from device to host
    CHECK_CUDA(cudaMemcpy(ret, dz, size * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // check output
    printf("check output tenosr... ");
    for (int i = 0; i < size; ++i) {
        float ref = alpha * (hx[i] * hy[i]) + beta * hz[i];
        if (!CheckFP32(ref, ret[i])) {
            printf("FAILED\n");
            exit(1);
        }
    }
    printf("OK\n");

    CHECK_HIEDNN(hiednnDestroyTensorDesc(xDesc));
    CHECK_HIEDNN(hiednnDestroyTensorDesc(yDesc));
    CHECK_HIEDNN(hiednnDestroyTensorDesc(zDesc));
    CHECK_HIEDNN(hiednnDestroyCudaHandle(handle));

    CHECK_CUDA(cudaFree(dx));
    CHECK_CUDA(cudaFree(dy));
    CHECK_CUDA(cudaFree(dz));

    free(hx);
    free(hy);
    free(hz);
    free(ret);
}

void BinaryMathBroadcast() {
    /*
     * input:
     * tensor x: dim={2, 3}, dataType=float
     * tensor y: dim={3}, dataType=float
     *
     * output:
     * tensor z: dim={2, 3}, dataType=float
     *
     * z[i] = alpha * (x[i] / y_broadcast[i]) + beta * z[i]
     */
    constexpr int64_t xDim[] = {2, 3};
    constexpr int xNDims = 2;
    constexpr int xSize = 2 * 3;
    constexpr float hx[] = {1.5f, 2.0f, 3.0f,
                            3.0f, 0.8f, 6.0f};

    constexpr int64_t yDim[] = {3};
    constexpr int yNDims = 1;
    constexpr int ySize = 3;
    constexpr float hy[] = {0.5f, 0.4f, 0.6f};

    constexpr hiednnDataType_t dtype = HIEDNN_DATATYPE_FP32;
    constexpr hiednnBinaryEltwiseOp_t op = HIEDNN_BINARY_MATH_DIV;
    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;

    // expected output: z has the same shape as x
    constexpr int zNDims = xNDims;
    constexpr int zSize = xSize;
    const int64_t (&zDim)[zNDims] = xDim;
    constexpr float hz[] = {3.0f, 5.0f, 5.0f,
                            6.0f, 2.0f, 10.0f};

    // create cuda handle
    hiednnCudaHandle_t handle;
    CHECK_HIEDNN(hiednnCreateCudaHandle(&handle));

    // default stream
    CHECK_HIEDNN(hiednnSetCudaStream(handle, 0));

    // create tensor descriptors
    hiednnTensorDesc_t xDesc, yDesc, zDesc;
    CHECK_HIEDNN(hiednnCreateTensorDesc(&xDesc));
    CHECK_HIEDNN(hiednnCreateTensorDesc(&yDesc));
    CHECK_HIEDNN(hiednnCreateTensorDesc(&zDesc));

    // init tensor descriptors
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(xDesc, dtype, xNDims, xDim));
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(yDesc, dtype, yNDims, yDim));
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(zDesc, dtype, zNDims, zDim));

    // allocate device memory for tensor and copy input tensor to device
    float *dx, *dy, *dz;
    CHECK_CUDA(cudaMalloc(&dx, xSize * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dy, ySize * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dz, zSize * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dx, hx, xSize * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dy, hy, ySize * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dz, hz, zSize * sizeof(float),
                          cudaMemcpyHostToDevice));

    printf("HIEDNN_BINARY_MATH_DIV\n");
    CHECK_HIEDNN(hiednnCudaBinaryElementwiseOp(
        handle, op, &alpha, xDesc, dx, yDesc, dy, nullptr, &beta, zDesc, dz));

    float *ret = static_cast<float *>(malloc(zSize * sizeof(float)));

    // copy output tensor from device to host
    CHECK_CUDA(cudaMemcpy(ret, dz, zSize * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // check output
    printf("check output tenosr... ");
    for (int i = 0; i < zSize; ++i) {
        if (!CheckFP32(hz[i], ret[i])) {
            printf("FAILED\n");
            exit(1);
        }
    }
    printf("OK\n");

    CHECK_HIEDNN(hiednnDestroyTensorDesc(xDesc));
    CHECK_HIEDNN(hiednnDestroyTensorDesc(yDesc));
    CHECK_HIEDNN(hiednnDestroyTensorDesc(zDesc));
    CHECK_HIEDNN(hiednnDestroyCudaHandle(handle));

    CHECK_CUDA(cudaFree(dx));
    CHECK_CUDA(cudaFree(dy));
    CHECK_CUDA(cudaFree(dz));
    free(ret);
}

void BinaryMathWithParameter() {
    /*
     * input:
     * tensor x: dim={2, 3, 4}, dataType=uint32_t
     * tensor y: dim={2, 3, 4}, dataType=uint32_t
     *
     * output:
     * tensor z: dim={2, 3, 4}, dataType=uint32_t
     *
     * z[i] = alpha * (x[i] << y[i]) + beta * z[i]
     */
    constexpr int64_t dim[] = {2, 3, 4};
    constexpr int nDims = 3;
    constexpr int size = 2 * 3 * 4;

    constexpr hiednnDataType_t dtype = HIEDNN_DATATYPE_UINT32;
    constexpr hiednnBinaryEltwiseOp_t op = HIEDNN_BINARY_MATH_BITSHIFT;
    // extParam == 0 indicates logical shift left
    constexpr int extParam = 0;
    constexpr uint32_t alpha = 1U;
    constexpr uint32_t beta = 2U;

    // create cuda handle
    hiednnCudaHandle_t handle;
    CHECK_HIEDNN(hiednnCreateCudaHandle(&handle));

    // default stream
    CHECK_HIEDNN(hiednnSetCudaStream(handle, 0));

    // create tensor descriptor
    hiednnTensorDesc_t xDesc, yDesc, zDesc;
    CHECK_HIEDNN(hiednnCreateTensorDesc(&xDesc));
    CHECK_HIEDNN(hiednnCreateTensorDesc(&yDesc));
    CHECK_HIEDNN(hiednnCreateTensorDesc(&zDesc));

    // init tensor descriptor
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(xDesc, dtype, nDims, dim));
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(yDesc, dtype, nDims, dim));
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(zDesc, dtype, nDims, dim));

    // host memory for tensor
    uint32_t *hx = static_cast<uint32_t *>(malloc(size * sizeof(uint32_t)));
    uint32_t *hy = static_cast<uint32_t *>(malloc(size * sizeof(uint32_t)));
    uint32_t *hz = static_cast<uint32_t *>(malloc(size * sizeof(uint32_t)));

    // init input tensor
    for (int i = 0; i < size; ++i) {
        hx[i] = 1U;
        hy[i] = static_cast<uint32_t>(i);
        hz[i] = static_cast<uint32_t>(size - i);
    }

    // allocate device memory for tensor and copy input tensor to device
    uint32_t *dx, *dy, *dz;
    CHECK_CUDA(cudaMalloc(&dx, size * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&dy, size * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&dz, size * sizeof(uint32_t)));
    CHECK_CUDA(cudaMemcpy(dx, hx, size * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dy, hy, size * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dz, hz, size * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));

    printf("HIEDNN_BINARY_MATH_BITSHIFT\n");
    CHECK_HIEDNN(hiednnCudaBinaryElementwiseOp(
        handle, op, &alpha, xDesc, dx, yDesc, dy, &extParam, &beta, zDesc, dz));

    uint32_t *ret = static_cast<uint32_t *>(malloc(size * sizeof(uint32_t)));

    // copy output tensor from device to host
    CHECK_CUDA(cudaMemcpy(ret, dz, size * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));

    // check output
    printf("check output tenosr... ");
    for (int i = 0; i < size; ++i) {
        uint32_t ref = alpha * (hx[i] << hy[i]) + beta * hz[i];
        if (ref != ret[i]) {
            printf("FAILED\n");
            exit(1);
        }
    }
    printf("OK\n");

    CHECK_HIEDNN(hiednnDestroyTensorDesc(xDesc));
    CHECK_HIEDNN(hiednnDestroyTensorDesc(yDesc));
    CHECK_HIEDNN(hiednnDestroyTensorDesc(zDesc));
    CHECK_HIEDNN(hiednnDestroyCudaHandle(handle));

    CHECK_CUDA(cudaFree(dx));
    CHECK_CUDA(cudaFree(dy));
    CHECK_CUDA(cudaFree(dz));

    free(hx);
    free(hy);
    free(hz);
    free(ret);
}

void BinaryBoolean() {
    /*
     * input:
     * tensor x: dim={4, 5, 6}, dataType=float
     * tensor y: dim={4, 5, 6}, dataType=float
     *
     * output:
     * tensor z: dim={4, 5, 6}, dataType=bool
     *
     * z[i] = x[i] <= y[i]
     */
    constexpr int64_t dim[] = {4, 5, 6};
    constexpr int nDims = 3;
    constexpr int size = 4 * 5 * 6;

    constexpr hiednnDataType_t xtype = HIEDNN_DATATYPE_FP32;
    constexpr hiednnDataType_t ytype = HIEDNN_DATATYPE_FP32;
    constexpr hiednnDataType_t ztype = HIEDNN_DATATYPE_BOOL;
    constexpr hiednnBinaryEltwiseOp_t op = HIEDNN_BINARY_COMPARE_LE;

    // create cuda handle
    hiednnCudaHandle_t handle;
    CHECK_HIEDNN(hiednnCreateCudaHandle(&handle));

    // default stream
    CHECK_HIEDNN(hiednnSetCudaStream(handle, 0));

    // create tensor descriptors
    hiednnTensorDesc_t xDesc, yDesc, zDesc;
    CHECK_HIEDNN(hiednnCreateTensorDesc(&xDesc));
    CHECK_HIEDNN(hiednnCreateTensorDesc(&yDesc));
    CHECK_HIEDNN(hiednnCreateTensorDesc(&zDesc));

    // init tensor descriptors
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(xDesc, xtype, nDims, dim));
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(yDesc, ytype, nDims, dim));
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(zDesc, ztype, nDims, dim));

    // host memory for tensor
    float *hx = static_cast<float *>(malloc(size * sizeof(float)));
    float *hy = static_cast<float *>(malloc(size * sizeof(float)));
    char *hz = static_cast<char *>(malloc(size * sizeof(char)));

    // init input tensor
    for (int i = 0; i < size; ++i) {
        hx[i] = i % 2 == 0 ? 0.1f * static_cast<float>(i) :
                             -0.1f * static_cast<float>(i);
        hx[i] = -hx[i];
    }

    // allocate device memory for tensor and copy input tensor to device
    float *dx, *dy;
    char *dz;
    CHECK_CUDA(cudaMalloc(&dx, size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dy, size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dz, size * sizeof(char)));
    CHECK_CUDA(cudaMemcpy(dx, hx, size * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dy, hy, size * sizeof(float),
                          cudaMemcpyHostToDevice));

    printf("HIEDNN_BINARY_COMPARE_LE\n");
    CHECK_HIEDNN(hiednnCudaBinaryElementwiseOp(
        handle, op, nullptr, xDesc, dx, yDesc, dy,
        nullptr, nullptr, zDesc, dz));

    // copy output tensor from device to host
    CHECK_CUDA(cudaMemcpy(hz, dz, size * sizeof(char),
                          cudaMemcpyDeviceToHost));

    // check output
    printf("check output tenosr... ");
    for (int i = 0; i < size; ++i) {
        char ref = hx[i] <= hy[i];
        if (ref != hz[i]) {
            printf("FAILED\n");
            exit(1);
        }
    }
    printf("OK\n");

    CHECK_HIEDNN(hiednnDestroyTensorDesc(xDesc));
    CHECK_HIEDNN(hiednnDestroyTensorDesc(yDesc));
    CHECK_HIEDNN(hiednnDestroyTensorDesc(zDesc));
    CHECK_HIEDNN(hiednnDestroyCudaHandle(handle));

    CHECK_CUDA(cudaFree(dx));
    CHECK_CUDA(cudaFree(dy));
    CHECK_CUDA(cudaFree(dz));

    free(hx);
    free(hy);
    free(hz);
}

int main() {
    BinaryMath();
    BinaryMathBroadcast();
    BinaryMathWithParameter();
    BinaryBoolean();
    return 0;
}
