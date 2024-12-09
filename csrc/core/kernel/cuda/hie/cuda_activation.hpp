/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    cuda_activation.hpp
 */

#pragma once

#include "cuda/cuda_common.h"
#define CUDA_HOST_DEVICE __forceinline__ __device__ __host__
#define CUDA_DEVICE __forceinline__ __device__
namespace hie {

namespace activation {

template <typename T>
struct Identity {
  static CUDA_DEVICE T Op(T& value) { return value; }
};

template <typename T>
struct Relu {
  static CUDA_DEVICE T Op(T& value) { return value < T(0) ? T(0) : value; }
};

template <typename T>
struct Tanh {
  static CUDA_DEVICE T Op(T& value) { return T(tanhf(float(value))); }
};

template <typename T>
struct Gelu {
  static CUDA_DEVICE T Op(const T& value) {
    return T(float(value) * 0.5f * (1.0f + erff(float(value) * 0.70710678f)));
  }
};

template <typename T>
struct Silu {
  static CUDA_DEVICE T Op(const T& value) {
    return T(float(value) * 1.0f / (1.0f + expf(float(-value))));
  }
};

template <>
struct Gelu<float> {
  static constexpr float gelu_erf_one_over_sqrt_two = float(0.7071067690849304);
  static constexpr float gelu_erf_approx_const = float(0.32759109139442444);
  static constexpr float gelu_erf_pol_0 = float(0.254829592f);
  static constexpr float gelu_erf_pol_1 = float(-0.284496736f);
  static constexpr float gelu_erf_pol_2 = float(1.421413741f);
  static constexpr float gelu_erf_pol_3 = float(-1.453152027f);
  static constexpr float gelu_erf_pol_4 = float(1.061405429f);

  static CUDA_DEVICE float Op(const float& value) { return Compute(value); }

  template <int UNROLL>
  static CUDA_DEVICE void Op(float (&value)[UNROLL], float (&result)[UNROLL]) {
#pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
      result[i] = Compute(value[i]);
    }
  }

  static CUDA_DEVICE float Compute(float src) {
    float dst, tmp0, tmp1, tmp2, tmp3, tmp4;

    // x = src /sqrt(2)
    src = src * gelu_erf_one_over_sqrt_two;
    tmp3 = src;

    // -exp(-x*x)
    src = src * src;
    src = -src;
    src = __expf(src);
    src = -src;

    // get sign
    tmp0 = tmp3 < 0 ? -1 : 1;

    // abs(x)
    tmp1 = abs(tmp3);

    // t = 1 / (p*x + 1)
    tmp2 = gelu_erf_approx_const * tmp1 + 1;
    // tmp4 = __fdividef(1, tmp2);
    tmp4 = 1.0f / tmp2;

    // -exp(-x*x)*t
    src = src * tmp4;

    // compute polynomialial r
    tmp1 = gelu_erf_pol_4;
    tmp1 = tmp1 * tmp4 + gelu_erf_pol_3;
    tmp1 = tmp1 * tmp4 + gelu_erf_pol_2;
    tmp1 = tmp1 * tmp4 + gelu_erf_pol_1;
    tmp1 = tmp1 * tmp4 + gelu_erf_pol_0;

    // erf = sign * (1 - r * t * exp(-x*x))
    src = src * tmp1 + 1;
    src = src * tmp0;

    // S = 0.5 * s = x / sqrt^2(2)
    tmp3 = tmp3 * gelu_erf_one_over_sqrt_two;

    // GELU = 0.5 * s * (1 + erf) = S + S * erf
    dst = src * tmp3 + tmp3;

    return dst;
  }
};

template <typename T>
struct GeluTanh {
  static CUDA_HOST_DEVICE T Op(const T& value) {
    const float x = float(value);
    return T(
        x * 0.5f *
        (1.0f + tanhf((0.7978845608028654f * (x + 0.044715f * x * x * x)))));
  }
};

}  // namespace activation
}  // namespace hie