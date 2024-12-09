/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    01_unary_elementwise.cpp
 */

#include <hiednn.h>
#include <hiednn_cpp.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>

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

    // create handle
    hiednnCppHandle_t handle;
    CHECK_HIEDNN(hiednnCreateCppHandle(&handle, HIEDNN_ASM_OPTIMIZE_NONE));

    // create tensor descriptor
    hiednnTensorDesc_t xDesc, yDesc;
    CHECK_HIEDNN(hiednnCreateTensorDesc(&xDesc));
    CHECK_HIEDNN(hiednnCreateTensorDesc(&yDesc));

    // init tensor descriptor
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(xDesc, dtype, nDims, dim));
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(yDesc, dtype, nDims, dim));

    // host memory for tensor
    float *x = static_cast<float *>(malloc(size * sizeof(float)));
    float *y = static_cast<float *>(malloc(size * sizeof(float)));

    // init input tensor
    for (int i = 0; i < size; ++i) {
        x[i] = 0.01f * static_cast<float>(i);
        y[i] = 0.02f * static_cast<float>(i);
    }

    float *yref = static_cast<float *>(malloc(size * sizeof(float)));
    memcpy(yref, y, size * sizeof(float));

    printf("HIEDNN_UNARY_MATH_EXP\n");
    CHECK_HIEDNN(hiednnCppUnaryElementwiseOp(
        handle, op, &alpha, xDesc, x, nullptr, nullptr, &beta, yDesc, y));

    // check output
    printf("check output tenosr... ");
    for (int i = 0; i < size; ++i) {
        float ref = alpha * std::exp(x[i]) + yref[i] * beta;
        if (!CheckFP32(ref, y[i])) {
            printf("FAILED\n");
            exit(1);
        }
    }
    printf("OK\n");

    CHECK_HIEDNN(hiednnDestroyTensorDesc(xDesc));
    CHECK_HIEDNN(hiednnDestroyTensorDesc(yDesc));
    CHECK_HIEDNN(hiednnDestroyCppHandle(handle));

    free(x);
    free(y);
    free(yref);
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

    // create handle
    hiednnCppHandle_t handle;
    CHECK_HIEDNN(hiednnCreateCppHandle(&handle, HIEDNN_ASM_OPTIMIZE_NONE));

    // create tensor descriptor
    hiednnTensorDesc_t xDesc, yDesc;
    CHECK_HIEDNN(hiednnCreateTensorDesc(&xDesc));
    CHECK_HIEDNN(hiednnCreateTensorDesc(&yDesc));

    // init tensor descriptor
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(xDesc, dtype, nDims, dim));
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(yDesc, dtype, nDims, dim));

    // host memory for tensor
    float *x = static_cast<float *>(malloc(size * sizeof(float)));
    float *y = static_cast<float *>(malloc(size * sizeof(float)));

    // init input tensor
    for (int i = 0; i < size; ++i) {
        x[i] = 0.01f * static_cast<float>(i);
        y[i] = 0.02f * static_cast<float>(i);
    }

    float *yref = static_cast<float *>(malloc(size * sizeof(float)));
    memcpy(yref, y, size * sizeof(float));

    printf("HIEDNN_UNARY_MATH_ADD\n");
    CHECK_HIEDNN(hiednnCppUnaryElementwiseOp(
        handle, op, &alpha, xDesc, x, &extParam1, nullptr, &beta, yDesc, y));

    // check output
    printf("check output tenosr... ");
    for (int i = 0; i < size; ++i) {
        float ref = alpha * (x[i] + extParam1) + yref[i] * beta;
        if (!CheckFP32(ref, y[i])) {
            printf("FAILED\n");
            exit(1);
        }
    }
    printf("OK\n");

    CHECK_HIEDNN(hiednnDestroyTensorDesc(xDesc));
    CHECK_HIEDNN(hiednnDestroyTensorDesc(yDesc));
    CHECK_HIEDNN(hiednnDestroyCppHandle(handle));

    free(x);
    free(y);
    free(yref);
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

    // create handle
    hiednnCppHandle_t handle;
    CHECK_HIEDNN(hiednnCreateCppHandle(&handle, HIEDNN_ASM_OPTIMIZE_NONE));

    // create tensor descriptor
    hiednnTensorDesc_t xDesc, yDesc;
    CHECK_HIEDNN(hiednnCreateTensorDesc(&xDesc));
    CHECK_HIEDNN(hiednnCreateTensorDesc(&yDesc));

    // init tensor descriptor
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(xDesc, xtype, nDims, dim));
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(yDesc, ytype, nDims, dim));

    // host memory for tensor
    float *x = static_cast<float *>(malloc(size * sizeof(float)));
    char *y = static_cast<char *>(malloc(size * sizeof(char)));

    // init input tensor
    for (int i = 0; i < size; ++i) {
        float inf;
        reinterpret_cast<float &>(inf) = 0x7f800000;
        x[i] = i % 2 == 0 ? 0.01f * static_cast<float>(i) : inf;
    }

    printf("HIEDNN_UNARY_MATH_ISINF\n");
    CHECK_HIEDNN(hiednnCppUnaryElementwiseOp(
        handle, op, nullptr, xDesc, x, nullptr, nullptr, nullptr, yDesc, y));

    // check output
    printf("check output tenosr... ");
    for (int i = 0; i < size; ++i) {
        char ref = std::isinf(x[i]);
        if (y[i] != ref) {
            printf("FAILED\n");
            exit(1);
        }
    }
    printf("OK\n");

    CHECK_HIEDNN(hiednnDestroyTensorDesc(xDesc));
    CHECK_HIEDNN(hiednnDestroyTensorDesc(yDesc));
    CHECK_HIEDNN(hiednnDestroyCppHandle(handle));

    free(x);
    free(y);
}

int main() {
    UnaryMath();
    UnaryMathWithParameter();
    UnaryBoolean();

    return 0;
}

