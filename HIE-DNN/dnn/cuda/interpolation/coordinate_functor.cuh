/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    coordinate_functor.cuh
 */

#ifndef DNN_CUDA_INTERPOLATION_COORDINATE_FUNCTOR_CUH_
#define DNN_CUDA_INTERPOLATION_COORDINATE_FUNCTOR_CUH_

#include <utils.hpp>
#include <cmath_wrapper.hpp>
#include <datatype_extension/datatype_extension.hpp>

namespace hiednn {

namespace cuda {

namespace interpolation {

//--------------------------------------
// Round Functor
//--------------------------------------
template <typename T>
struct RoundNon {
    static __device__ __forceinline__
    T Round(const T &coord) {
        return coord;
    }
};

template <typename T>
struct RoundHalfDown {
    static __device__ __forceinline__
    T Round(const T &coord) {
        return cmath_ceil(coord - T(0.5));
    }
};

template <typename T>
struct RoundHalfUp {
    static __device__ __forceinline__
    T Round(const T &coord) {
        return cmath_floor(coord + T(0.5));
    }
};

template <typename T>
struct RoundFloor {
    static __device__ __forceinline__
    T Round(const T &coord) {
        return cmath_floor(coord);
    }
};

template <typename T>
struct RoundCeil {
    static __device__ __forceinline__
    T Round(const T &coord) {
        return cmath_ceil(coord);
    }
};

//--------------------------------------
// Coordinate Functor
//--------------------------------------
template <template <typename> class RoundFunc, typename T>
struct HalfPixel {
    T scaleRcp;

    HalfPixel() = default;

    HalfPixel(T lengthIn, T lengthOut, T scale) {
        scaleRcp = T(1) / scale;
    }

    template <bool INVALID = true>
    __device__ __forceinline__ T Coordinate(T coordOut) {
        T coordIn = (coordOut + T(0.5)) * scaleRcp - T(0.5);
        coordIn = RoundFunc<T>::Round(coordIn);
        return INVALID ? (coordIn < 0 ? T(0) : coordIn) : coordIn;
    }
};

template <template <typename> class RoundFunc, typename T>
struct PytorchHalfPixel {
    T scaleRcp;

    PytorchHalfPixel() = default;

    PytorchHalfPixel(T lengthIn, T lengthOut, T scale) {
        scaleRcp = lengthOut > 1 ? T(1) / scale: T(0);
    }

    template <bool INVALID = true>
    __device__ __forceinline__ T Coordinate(T coordOut) {
        T coordIn = (coordOut + T(0.5)) * scaleRcp - T(0.5);
        coordIn = RoundFunc<T>::Round(coordIn);
        return INVALID ? (coordIn < 0 ? T(0) : coordIn) : coordIn;
    }
};

template <template <typename> class RoundFunc, typename T>
struct AlignCorner {
    T alignCornerScale;

    AlignCorner() = default;

    AlignCorner(T lengthIn, T lengthOut, T scale) {
        alignCornerScale = lengthOut > 1 ?
                           (lengthIn - 1) / (lengthOut - 1) : T(0);
    }

    template <bool INVALID = true>
    __device__ __forceinline__ T Coordinate(T coordOut) {
        T coordIn = coordOut * alignCornerScale;
        return RoundFunc<T>::Round(coordIn);
    }
};

template <template <typename> class RoundFunc, typename T>
struct Asymmetric {
    T scaleRcp;

    Asymmetric() = default;

    Asymmetric(T lengthIn, T lengthOut, T scale) {
        scaleRcp = T(1) / scale;
    }

    template <bool INVALID = true>
    __device__ __forceinline__ T Coordinate(T coordOut) {
        return RoundFunc<T>::Round(coordOut * scaleRcp);
    }
};

template <typename T>
struct ScaleType {
    using type = T;
};

#ifdef HIEDNN_USE_FP16
template <>
struct ScaleType<half> {
    using type = float;
};
#endif

#ifdef HIEDNN_USE_BF16
template <>
struct ScaleType<bfloat16> {
    using type = float;
};
#endif

}  // namespace interpolation

}  // namespace cuda

}  // namespace hiednn

#endif  // DNN_CUDA_INTERPOLATION_COORDINATE_FUNCTOR_CUH_


