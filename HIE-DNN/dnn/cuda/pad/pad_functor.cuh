/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    pad_functor.cuh
 */

#ifndef DNN_CUDA_PAD_PAD_FUNCTOR_CUH_
#define DNN_CUDA_PAD_PAD_FUNCTOR_CUH_

#include <utils.hpp>
#include <cuda/intrinsic/global_memory.hpp>

namespace hiednn {

namespace cuda {

namespace pad {

template <int NDIMS,
          int IMAGE_UNROLL,
          typename OffsetT,
          typename T>
struct PadConst {
    static_assert(IMAGE_UNROLL <= 32, "PadConst: invalid IMAGE_UNROLL");

    uint32_t mask;
    const T *ldgPtr[IMAGE_UNROLL];

    __device__ __forceinline__
    void GetInputPtr(const T *x,
                     const OffsetT (&batchIdx)[IMAGE_UNROLL],
                     const OffsetT (&idx)[IMAGE_UNROLL][NDIMS],
                     const Array<OffsetT, NDIMS> &lPadBound,
                     const Array<OffsetT, NDIMS> &rPadBound,
                     const Array<OffsetT, NDIMS> &xIdxMax,
                     const Array<OffsetT, NDIMS> &xStride,
                     const OffsetT &xBatchStride) {
        mask = 0xffffffff;
        OffsetT ldgOffset[IMAGE_UNROLL];

        #pragma unroll
        for (int i = 0; i < IMAGE_UNROLL; ++i) {
            ldgOffset[i] = 0;
        }

        #pragma unroll
        for (int i = 0; i < NDIMS; ++i) {
            #pragma unroll
            for (int j = 0; j < IMAGE_UNROLL; ++j) {
                if (idx[j][i] < lPadBound[i] || idx[j][i] >= rPadBound[i]) {
                    mask &= ~(1U << j);
                } else {
                    ldgOffset[j] += (idx[j][i] - lPadBound[i]) * xStride[i];
                }
            }
        }

        #pragma unroll
        for (int i = 0; i < IMAGE_UNROLL; ++i) {
            ldgPtr[i] = x + (batchIdx[i] * xBatchStride + ldgOffset[i]);
        }
    }

    __device__ __forceinline__
    T Load(const int &batchIt,
           const int &imgIt,
           const OffsetT &batchLdgStride,
           const T &param) {
        T ret;
        if ((mask & (1U << imgIt)) != 0) {
            Ldg<NC>(&ret, ldgPtr[imgIt] + batchIt * batchLdgStride);
        } else {
            ret = param;
        }
        return ret;
    }
};

template <int NDIMS,
          int IMAGE_UNROLL,
          typename OffsetT,
          typename T>
struct PadEdge {
    const T *ldgPtr[IMAGE_UNROLL];

    __device__ __forceinline__
    void GetInputPtr(const T *x,
                     const OffsetT (&batchIdx)[IMAGE_UNROLL],
                     const OffsetT (&idx)[IMAGE_UNROLL][NDIMS],
                     const Array<OffsetT, NDIMS> &lPadBound,
                     const Array<OffsetT, NDIMS> &rPadBound,
                     const Array<OffsetT, NDIMS> &xIdxMax,
                     const Array<OffsetT, NDIMS> &xStride,
                     const OffsetT &xBatchStride) {
        OffsetT ldgOffset[IMAGE_UNROLL];
        #pragma unroll
        for (int i = 0; i < IMAGE_UNROLL; ++i) {
            ldgOffset[i] = 0;
        }

        #pragma unroll
        for (int i = 0; i < NDIMS; ++i) {
            #pragma unroll
            for (int j = 0; j < IMAGE_UNROLL; ++j) {
                if (idx[j][i] >= rPadBound[i]) {
                    ldgOffset[j] += xIdxMax[i] * xStride[i];
                } else if (idx[j][i] >= lPadBound[i]) {
                    ldgOffset[j] += (idx[j][i] - lPadBound[i]) * xStride[i];
                }
            }
        }

        #pragma unroll
        for (int i = 0; i < IMAGE_UNROLL; ++i) {
            ldgPtr[i] = x + (batchIdx[i] * xBatchStride + ldgOffset[i]);
        }
    }

    __device__ __forceinline__
    T Load(const int &batchIt,
           const int &imgIt,
           const OffsetT &batchLdgStride,
           const T &param) {
        T ret;
        Ldg<NC>(&ret, ldgPtr[imgIt] + batchIt * batchLdgStride);
        return ret;
    }
};

template <int NDIMS,
          int IMAGE_UNROLL,
          typename OffsetT,
          typename T>
struct PadReflect {
    const T *ldgPtr[IMAGE_UNROLL];

    __device__ __forceinline__
    void GetInputPtr(const T *x,
                     const OffsetT (&batchIdx)[IMAGE_UNROLL],
                     const OffsetT (&idx)[IMAGE_UNROLL][NDIMS],
                     const Array<OffsetT, NDIMS> &lPadBound,
                     const Array<OffsetT, NDIMS> &rPadBound,
                     const Array<OffsetT, NDIMS> &xIdxMax,
                     const Array<OffsetT, NDIMS> &xStride,
                     const OffsetT &xBatchStride) {
        OffsetT ldgOffset[IMAGE_UNROLL];
        #pragma unroll
        for (int i = 0; i < IMAGE_UNROLL; ++i) {
            ldgOffset[i] = 0;
        }

        #pragma unroll
        for (int i = 0; i < NDIMS; ++i) {
            #pragma unroll
            for (int j = 0; j < IMAGE_UNROLL; ++j) {
                if (idx[j][i] < lPadBound[i]) {
                    ldgOffset[j] += (lPadBound[i] - idx[j][i]) * xStride[i];
                } else if (idx[j][i] < rPadBound[i]) {
                    ldgOffset[j] += (idx[j][i] - lPadBound[i]) * xStride[i];
                } else {
                    ldgOffset[j] += (2 * xIdxMax[i] - idx[j][i] + lPadBound[i])
                                    * xStride[i];
                }
            }
        }

        #pragma unroll
        for (int i = 0; i < IMAGE_UNROLL; ++i) {
            ldgPtr[i] = x + (batchIdx[i] * xBatchStride + ldgOffset[i]);
        }
    }

    __device__ __forceinline__
    T Load(const int &batchIt,
           const int &imgIt,
           const OffsetT &batchLdgStride,
           const T &param) {
        T ret;
        Ldg<NC>(&ret, ldgPtr[imgIt] + batchIt * batchLdgStride);
        return ret;
    }
};

}  // namespace pad

}  // namespace cuda

}  // namespace hiednn

#endif  // DNN_CUDA_PAD_PAD_FUNCTOR_CUH_


