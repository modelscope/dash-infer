/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    pack.cuh
 */

#pragma once

#include <cassert>
#include <cstdint>

#include "common/func_modifier.h"

namespace span {

template <int S>
struct WordType {
  using Type = uint32_t;
};

template <>
struct WordType<1> {
  using Type = uint8_t;
};

template <>
struct WordType<2> {
  using Type = uint16_t;
};

/**
 * extract the idx'th S-byte element from register `word`, and put the
 * output to the lower S-byte of the returned 32-bit register
 */
template <int S>
DEVICE_FUNC uint32_t ByteExtract(uint32_t word, int idx);

template <int S>
DEVICE_FUNC uint32_t ByteExtract<1>(uint32_t word, int idx) {
  uint32_t ret;
  asm("bfe.u32 %0, %1, %2, 8;" : "=r"(ret) : "r"(word), "r"(idx * 8));
  return ret;
}

template <>
DEVICE_FUNC uint32_t ByteExtract<2>(uint32_t word, int idx) {
  uint16_t lo, hi;
  asm("mov.b32 {%0, %1}, %2;" : "=h"(lo), "=h"(hi) : "r"(word));
  return idx == 0 ? lo : hi;
}

template <>
DEVICE_FUNC uint32_t ByteExtract<4>(uint32_t word, int idx) {
  return word;
}

/**
 * Split the packed data to 32-bit word to make the compiler generate
 * efficient unpack instructions, such as R0.B1, R0.B2 or R0.H0_H0
 */
template <int PACK_SIZE, typename T>
struct alignas(PACK_SIZE * sizeof(T)) WordPackT {
  static_assert(sizeof(T) <= 4, "size of type too large");
  static_assert(sizeof(T) * PACK_SIZE <= 16, "PACK_SIZE too large");

  static constexpr int PACK_BYTE = PACK_SIZE * sizeof(T);
  using WordT = typename WordType<PACK_BYTE>::Type;
  static constexpr int WORD_NUM = PACK_BYTE / sizeof(WordT);

  WordT data[WORD_NUM];

  template <typename ComputeT>
  DEVICE_FUNC void Unpack(ComputeT (&ret)[PACK_SIZE]) const {
    constexpr int WORD_PACK = sizeof(WordT) / sizeof(T);
#pragma unroll
    for (int i = 0; i < WORD_NUM; ++i) {
#pragma unroll
      for (int j = 0; j < WORD_PACK; ++j) {
        uint32_t rawData;
        rawData = ByteExtract<sizeof(T)>(data[i], j);
        ret[i * WORD_PACK + j] =
            static_cast<ComputeT>(*reinterpret_cast<T*>(&rawData));
      }
    }
  }

  /**
   * UNDERLYING_SIZE: if T is packed type, the pack size of T itself, else 1.
   *
   * void extractor(uint32_t (&ret)[UNDERLYING_SIZE], uint32_t rawData)
   */
  template <int UNDERLYING_SIZE, typename UnderlyingT = T, typename ComputeT,
            typename ExtractorT>
  DEVICE_FUNC void Unpack(ComputeT (&ret)[PACK_SIZE * UNDERLYING_SIZE],
                          ExtractorT extractor) const {
    static_assert(UNDERLYING_SIZE > 0, "UNDERLYING_SIZE must be positive");
    static_assert(UNDERLYING_SIZE <= 8 * sizeof(T), "invalid UNDERLYING_SIZE");

    constexpr int WORD_PACK = sizeof(WordT) / sizeof(T);
#pragma unroll
    for (int i = 0; i < WORD_NUM; ++i) {
#pragma unroll
      for (int j = 0; j < WORD_PACK; ++j) {
        uint32_t rawData;
        rawData = ByteExtract<sizeof(T)>(data[i], j);

        uint32_t extracted[UNDERLYING_SIZE];
        extractor(extracted, rawData);
#pragma unroll
        for (int k = 0; k < UNDERLYING_SIZE; ++k) {
          ret[(i * WORD_PACK + j) * UNDERLYING_SIZE + k] =
              static_cast<ComputeT>(
                  *reinterpret_cast<UnderlyingT*>(extracted + k));
        }
      }
    }
  }
};

template <int PACK_SIZE, typename T>
struct alignas(PACK_SIZE * sizeof(T)) PackT {
  T data[PACK_SIZE];
};

}  // namespace span
