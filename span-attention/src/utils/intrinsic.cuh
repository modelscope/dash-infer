/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    intrinsic.cuh
 */

#pragma once

#include <cassert>
#include <cstdint>

#include "common/func_modifier.h"

namespace span {

namespace intrinsic {

template <int S>
DEVICE_FUNC void LdgNCImpl(void* ret, const void* ptr);

template <>
DEVICE_FUNC void LdgNCImpl<1>(void* ret, const void* ptr) {
  uint32_t reg;
  asm volatile("ld.global.nc.b8 %0, [%1];" : "=r"(reg) : "l"(ptr));
  *static_cast<uint8_t*>(ret) = static_cast<uint8_t>(reg);
}

template <>
DEVICE_FUNC void LdgNCImpl<2>(void* ret, const void* ptr) {
  uint32_t reg;
  asm volatile("ld.global.nc.b16 %0, [%1];" : "=r"(reg) : "l"(ptr));
  *static_cast<uint16_t*>(ret) = static_cast<uint16_t>(reg);
}

template <>
DEVICE_FUNC void LdgNCImpl<4>(void* ret, const void* ptr) {
  uint32_t reg;
  asm volatile("ld.global.nc.b32 %0, [%1];" : "=r"(reg) : "l"(ptr));
  *static_cast<uint32_t*>(ret) = reg;
}

template <>
DEVICE_FUNC void LdgNCImpl<8>(void* ret, const void* ptr) {
  uint32_t w0, w1;
  asm volatile("ld.global.nc.v2.b32 {%0, %1}, [%2];"
               : "=r"(w0), "=r"(w1)
               : "l"(ptr));
  static_cast<uint2*>(ret)->x = w0;
  static_cast<uint2*>(ret)->y = w1;
}

template <>
DEVICE_FUNC void LdgNCImpl<16>(void* ret, const void* ptr) {
  uint32_t w0, w1, w2, w3;
  asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
               : "=r"(w0), "=r"(w1), "=r"(w2), "=r"(w3)
               : "l"(ptr));
  static_cast<uint4*>(ret)->x = w0;
  static_cast<uint4*>(ret)->y = w1;
  static_cast<uint4*>(ret)->z = w2;
  static_cast<uint4*>(ret)->w = w3;
}

template <int S>
DEVICE_FUNC void LdgCSImpl(void* ret, const void* ptr);

template <>
DEVICE_FUNC void LdgCSImpl<1>(void* ret, const void* ptr) {
  uint32_t reg;
  asm volatile("ld.global.cs.b8 %0, [%1];" : "=r"(reg) : "l"(ptr));
  *static_cast<uint8_t*>(ret) = static_cast<uint8_t>(reg);
}

template <>
DEVICE_FUNC void LdgCSImpl<2>(void* ret, const void* ptr) {
  uint32_t reg;
  asm volatile("ld.global.cs.b16 %0, [%1];" : "=r"(reg) : "l"(ptr));
  *static_cast<uint16_t*>(ret) = static_cast<uint16_t>(reg);
}

template <>
DEVICE_FUNC void LdgCSImpl<4>(void* ret, const void* ptr) {
  uint32_t reg;
  asm volatile("ld.global.cs.b32 %0, [%1];" : "=r"(reg) : "l"(ptr));
  *static_cast<uint32_t*>(ret) = reg;
}

template <>
DEVICE_FUNC void LdgCSImpl<8>(void* ret, const void* ptr) {
  uint32_t w0, w1;
  asm volatile("ld.global.cs.v2.b32 {%0, %1}, [%2];"
               : "=r"(w0), "=r"(w1)
               : "l"(ptr));
  static_cast<uint2*>(ret)->x = w0;
  static_cast<uint2*>(ret)->y = w1;
}

template <>
DEVICE_FUNC void LdgCSImpl<16>(void* ret, const void* ptr) {
  uint32_t w0, w1, w2, w3;
  asm volatile("ld.global.cs.v4.b32 {%0, %1, %2, %3}, [%4];"
               : "=r"(w0), "=r"(w1), "=r"(w2), "=r"(w3)
               : "l"(ptr));
  static_cast<uint4*>(ret)->x = w0;
  static_cast<uint4*>(ret)->y = w1;
  static_cast<uint4*>(ret)->z = w2;
  static_cast<uint4*>(ret)->w = w3;
}

template <int S>
DEVICE_FUNC void LdgCGImpl(void* ret, const void* ptr);

template <>
DEVICE_FUNC void LdgCGImpl<1>(void* ret, const void* ptr) {
  uint32_t reg;
  asm volatile("ld.global.cg.b8 %0, [%1];" : "=r"(reg) : "l"(ptr));
  *static_cast<uint8_t*>(ret) = static_cast<uint8_t>(reg);
}

template <>
DEVICE_FUNC void LdgCGImpl<2>(void* ret, const void* ptr) {
  uint32_t reg;
  asm volatile("ld.global.cg.b16 %0, [%1];" : "=r"(reg) : "l"(ptr));
  *static_cast<uint16_t*>(ret) = static_cast<uint16_t>(reg);
}

template <>
DEVICE_FUNC void LdgCGImpl<4>(void* ret, const void* ptr) {
  uint32_t reg;
  asm volatile("ld.global.cg.b32 %0, [%1];" : "=r"(reg) : "l"(ptr));
  *static_cast<uint32_t*>(ret) = reg;
}

template <>
DEVICE_FUNC void LdgCGImpl<8>(void* ret, const void* ptr) {
  uint32_t w0, w1;
  asm volatile("ld.global.cg.v2.b32 {%0, %1}, [%2];"
               : "=r"(w0), "=r"(w1)
               : "l"(ptr));
  static_cast<uint2*>(ret)->x = w0;
  static_cast<uint2*>(ret)->y = w1;
}

template <int S>
DEVICE_FUNC void StgCGImpl(void* ptr, const void* v);

template <>
DEVICE_FUNC void StgCGImpl<1>(void* ptr, const void* v) {
  uint32_t reg = *static_cast<const uint8_t*>(v);
  asm volatile("st.global.cg.b8 [%0], %1;" : : "l"(ptr), "r"(reg));
}

template <>
DEVICE_FUNC void StgCGImpl<2>(void* ptr, const void* v) {
  uint32_t reg = *static_cast<const uint16_t*>(v);
  asm volatile("st.global.cg.b16 [%0], %1;" : : "l"(ptr), "r"(reg));
}

template <>
DEVICE_FUNC void StgCGImpl<4>(void* ptr, const void* v) {
  const uint32_t& reg = *static_cast<const uint32_t*>(v);
  asm volatile("st.global.cg.b32 [%0], %1;" : : "l"(ptr), "r"(reg));
}

template <>
DEVICE_FUNC void StgCGImpl<8>(void* ptr, const void* v) {
  const uint32_t& r0 = static_cast<const uint2*>(v)->x;
  const uint32_t& r1 = static_cast<const uint2*>(v)->y;
  asm volatile("st.global.cg.v2.b32 [%0], {%1, %2};"
               :
               : "l"(ptr), "r"(r0), "r"(r1));
}

template <int S>
DEVICE_FUNC void RegSet0Impl(void* regPtr);

template <>
DEVICE_FUNC void RegSet0Impl<1>(void* regPtr) {
  uint32_t reg;
  asm volatile("mov.b32 %0, 0;" : "=r"(reg));
  *static_cast<uint8_t*>(regPtr) = static_cast<uint8_t>(reg);
}

template <>
DEVICE_FUNC void RegSet0Impl<2>(void* regPtr) {
  asm volatile("mov.b16 %0, 0;" : "=h"(*static_cast<uint16_t*>(regPtr)));
}

template <>
DEVICE_FUNC void RegSet0Impl<4>(void* regPtr) {
  asm volatile("mov.b32 %0, 0;" : "=r"(*static_cast<uint32_t*>(regPtr)));
}

template <>
DEVICE_FUNC void RegSet0Impl<8>(void* regPtr) {
  asm volatile("mov.b64 {%0, %1}, 0;"
               : "=r"(static_cast<uint2*>(regPtr)->x),
                 "=r"(static_cast<uint2*>(regPtr)->y));
}

template <>
DEVICE_FUNC void RegSet0Impl<16>(void* regPtr) {
  asm volatile(
      "mov.b64 {%0, %1}, 0;\n"
      "mov.b64 {%2, %3}, 0;"
      : "=r"(static_cast<uint4*>(regPtr)->x),
        "=r"(static_cast<uint4*>(regPtr)->y),
        "=r"(static_cast<uint4*>(regPtr)->z),
        "=r"(static_cast<uint4*>(regPtr)->w));
}

}  // namespace intrinsic

template <typename T>
DEVICE_FUNC void LdgNC(T* ret, const void* ptr) {
  intrinsic::LdgNCImpl<sizeof(T)>(ret, ptr);
}

template <typename T>
DEVICE_FUNC void LdgCS(T* ret, const void* ptr) {
  intrinsic::LdgCSImpl<sizeof(T)>(ret, ptr);
}

template <typename T>
DEVICE_FUNC void LdgCG(T* ret, const void* ptr) {
  intrinsic::LdgCGImpl<sizeof(T)>(ret, ptr);
}

template <typename T>
DEVICE_FUNC void StgCG(void* ptr, const T& r) {
  intrinsic::StgCGImpl<sizeof(T)>(ptr, &r);
}

// specialized register-set-0 function to utilize CS2R instruction
template <typename T>
DEVICE_FUNC void RegSet0(T* regPtr) {
  intrinsic::RegSet0Impl<sizeof(T)>(regPtr);
}

}  // namespace span
