/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    10_linear_interpolation.cu
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

#define CHECK_32F(y, ref) { \
    if (std::fabs(y - ref) > std::fabs(ref) * 1e-5f) { \
        printf("FAILED\n"); \
        exit(1); \
    } \
}

float Asymmetric(float yCoord, float scale) {
    return yCoord / scale;
}

void LinearInterp2DReference(
        const float *x, const int64_t *xDim, const float *scale,
        float *y, const int64_t *yDim) {
    for (int64_t batch = 0; batch < yDim[0]; ++batch) {
        const float *xPtr = x + batch * xDim[1] * xDim[2];
        float *yPtr = y + batch * yDim[1] * yDim[2];

        for (int64_t row = 0; row < yDim[1]; ++row) {
            for (int64_t col = 0; col < yDim[2]; ++col) {
                float xCoordF[2];
                xCoordF[0] = Asymmetric(static_cast<float>(row), scale[0]);
                xCoordF[1] = Asymmetric(static_cast<float>(col), scale[1]);

                float xCoordFL[2];
                float xCoordFR[2];
                xCoordFL[0] = std::floor(xCoordF[0]);
                xCoordFL[1] = std::floor(xCoordF[1]);
                xCoordFR[0] = std::ceil(xCoordF[0]);
                xCoordFR[1] = std::ceil(xCoordF[1]);
                if (xCoordFR[0] >= xDim[1]) xCoordFR[0] = xDim[1] - 1;
                if (xCoordFR[1] >= xDim[2]) xCoordFR[1] = xDim[2] - 1;

                int64_t xCoordIL[2];
                int64_t xCoordIR[2];
                xCoordIL[0] = static_cast<int64_t>(xCoordFL[0]);
                xCoordIL[1] = static_cast<int64_t>(xCoordFL[1]);
                xCoordIR[0] = static_cast<int64_t>(xCoordFR[0]);
                xCoordIR[1] = static_cast<int64_t>(xCoordFR[1]);

                float xCoordOffset[2];
                xCoordOffset[0] = xCoordF[0] - xCoordFL[0];
                xCoordOffset[1] = xCoordF[1] - xCoordFL[1];

                int64_t xOffset00 = xCoordIL[0] * xDim[2] + xCoordIL[1];
                int64_t xOffset01 = xCoordIL[0] * xDim[2] + xCoordIR[1];
                int64_t xOffset10 = xCoordIR[0] * xDim[2] + xCoordIL[1];
                int64_t xOffset11 = xCoordIR[0] * xDim[2] + xCoordIR[1];

                float x00 = xPtr[xOffset00];
                float x01 = xPtr[xOffset01];
                float x10 = xPtr[xOffset10];
                float x11 = xPtr[xOffset11];

                float x0 = x00 + (x01 - x00) * xCoordOffset[1];
                float x1 = x10 + (x11 - x10) * xCoordOffset[1];

                float yRef = x0 + (x1 - x0) * xCoordOffset[0];
                yPtr[row * yDim[2] + col] = yRef;
            }
        }
    }
}

int main() {
    /*
     * input:
     * tensor x: dim={2, 4, 4}, dataType=float
     * param coorMode=HIEDNN_INTERP_COORD_ASYMMETRIC
     * param scale={1.5, 1.5}
     *
     * output:
     * tensor y: dim={2, 6. 6}, datatype=float
     */
    // input dimension:
    int64_t xDim[] = {2, 4, 4};
    int64_t xSize = 2 * 4 * 4;
    int xNDims = 3;

    // parameter
    hiednnDataType_t dataType = HIEDNN_DATATYPE_FP32;
    hiednnInterpCoordMode_t coordMode = HIEDNN_INTERP_COORD_ASYMMETRIC;
    float scale[] = {1.5, 1.5};
    int scaleSize = 2;

    // output dimension:
    int64_t yDim[] = {2, 6, 6};
    int64_t ySize = 2 * 6 * 6;
    int yNDims = 3;

    // init tensor x
    float x[2 * 4 * 4];
    for (int i = 0; i < 2 * 4 * 4; ++i) {
        x[i] = 0.5f * i;
    }

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
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(xDesc, dataType, xNDims, xDim));
    CHECK_HIEDNN(hiednnSetNormalTensorDesc(yDesc, dataType, yNDims, yDim));

    // allocate device memory for tensor and copy input tensor to device
    float *dx, *dy;
    CHECK_CUDA(cudaMalloc(&dx, xSize * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dy, ySize * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dx, x, xSize * sizeof(float),
                          cudaMemcpyHostToDevice));

    CHECK_HIEDNN(hiednnCudaLinearInterpolation(
        handle, coordMode, xDesc, dx, scale, scaleSize, yDesc, dy));

    float *y = static_cast<float *>(malloc(ySize * sizeof(float)));

    // copy output tensor from device to host
    CHECK_CUDA(cudaMemcpy(y, dy, ySize * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // get expected output
    float *yRef = static_cast<float *>(malloc(ySize * sizeof(float)));
    LinearInterp2DReference(x, xDim, scale, yRef, yDim);

    // check output
    printf("check output tenosr... ");
    for (int i = 0; i < ySize; ++i) {
        CHECK_32F(y[i], yRef[i]);
    }
    printf("OK\n");

    CHECK_HIEDNN(hiednnDestroyTensorDesc(xDesc));
    CHECK_HIEDNN(hiednnDestroyTensorDesc(yDesc));
    CHECK_HIEDNN(hiednnDestroyCudaHandle(handle));

    CHECK_CUDA(cudaFree(dx));
    CHECK_CUDA(cudaFree(dy));

    free(y);
    free(yRef);

    return 0;
}

