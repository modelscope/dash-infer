/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    convert_4bit.h
 */

#pragma once
#include "cuda/cuda_common.h"

template <typename FT, typename QT>
class cvt_4bx2_to_16bx2;

template <typename FT>
struct cvt_4bx2_to_16bx2<FT, uint8_t> {
  static __device__ __forceinline__ void cvt(const uint8_t& idata, FT* fdata) {
    fdata[0] = FT(uint16_t(idata & 0xf));
    fdata[1] = FT(uint16_t(idata >> 4));
  }
};

template <>
struct cvt_4bx2_to_16bx2<__half2, uint8_t> {
  static __device__ __forceinline__ void cvt(const uint8_t& idata,
                                             __half2& fdata) {
    uint32_t res = idata & 0xf;
    res |= (idata & 0xf0) << 12;
    constexpr uint32_t MAGIC_NUM = (25 << 10 << 16) | (25 << 10);
    res = __vadd2(MAGIC_NUM, res);
    fdata = __hsub2(reinterpret_cast<const __half2*>(&res)[0],
                    reinterpret_cast<const __half2*>(&MAGIC_NUM)[0]);
  }
};

template <>
struct cvt_4bx2_to_16bx2<__nv_bfloat162, uint8_t> {
  static __device__ __forceinline__ void cvt(const uint8_t& idata,
                                             __nv_bfloat162& fdata) {
#if (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
    __half2 fval;
    cvt_4bx2_to_16bx2<__half2, uint8_t>::cvt(idata, fval);
    const float2 fval_t = __half22float2(fval);
    fdata = __float22bfloat162_rn(fval_t);
#endif
  }
};

__device__ __forceinline__ void cvt_u4x8_to_halfx8(const uint32_t& idata,
                                                   __half2* fdata) {
  uint32_t low4 = idata & 0x0f0f0f0f;
  uint32_t high4 = (idata & 0xf0f0f0f0) >> 4;
  uint32_t i32x8[8];
  asm volatile(
      "{prmt.b32 %0, 0, %8, 0x4;\n"
      "prmt.b32 %2, 0, %8, 0x5;\n"
      "prmt.b32 %4, 0, %8, 0x6;\n"
      "prmt.b32 %6, 0, %8, 0x7;\n"
      "prmt.b32 %1, 0, %9, 0x4;\n"
      "prmt.b32 %3, 0, %9, 0x5;\n"
      "prmt.b32 %5, 0, %9, 0x6;\n"
      "prmt.b32 %7, 0, %9, 0x7;}\n"
      : "=r"(i32x8[0]), "=r"(i32x8[1]), "=r"(i32x8[2]), "=r"(i32x8[3]),
        "=r"(i32x8[4]), "=r"(i32x8[5]), "=r"(i32x8[6]), "=r"(i32x8[7])
      : "r"(low4), "r"(high4));
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    fdata[i].x = static_cast<half>(i32x8[2 * i]);
    fdata[i].y = static_cast<half>(i32x8[2 * i + 1]);
  }
}

template <typename T>
__device__ __forceinline__ void cvt_4bx8_to_16bx8_v2(const uint32_t& idata,
                                                     T* fdata);

template <>
// optimized 8xu4 to 8xhalf2, author lyh238099
__device__ __forceinline__ void cvt_4bx8_to_16bx8_v2<__half2>(
    const uint32_t& idata, __half2* fdata) {
  uint32_t i6x20 = idata & 0x0f000f0f;
  uint32_t i753x = idata & 0xf0f0f000;
  uint32_t ix4x1 = idata & 0x000f00f0;

  uint32_t i10, i32, i54, i76;
  asm("prmt.b32 %0, %4, %6, 0x2420;"
      "prmt.b32 %1, %4, %5, 0x2521;"
      "prmt.b32 %2, %5, %6, 0x0206;"
      "prmt.b32 %3, %4, %5, 0x2723;"
      : "=r"(i10), "=r"(i32), "=r"(i54), "=r"(i76)
      : "r"(i6x20), "r"(i753x), "r"(ix4x1));

  uint32_t expo = ((6 + 15) << (10 + 16)) | ((10 + 15) << 10);
  const __half2& sub = reinterpret_cast<const __half2&>(expo);

  uint32_t h10 = i10 | expo;
  uint32_t h32 = i32 | expo;
  uint32_t h54 = i54 | expo;
  uint32_t h76 = i76 | expo;

  fdata[0] = __hsub2(reinterpret_cast<const __half2&>(h10), sub);
  fdata[1] = __hsub2(reinterpret_cast<const __half2&>(h32), sub);
  fdata[2] = __hsub2(reinterpret_cast<const __half2&>(h54), sub);
  fdata[3] = __hsub2(reinterpret_cast<const __half2&>(h76), sub);
}

template <>
__device__ __forceinline__ void cvt_4bx8_to_16bx8_v2<__nv_bfloat162>(
    const uint32_t& idata, __nv_bfloat162* fdata) {
#if (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
  __half2 fval[4];
  cvt_4bx8_to_16bx8_v2<__half2>(idata, fval);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 fval_t = __half22float2(fval[i]);
    *(fdata + i) = __float22bfloat162_rn(fval_t);
  }
#endif
}
