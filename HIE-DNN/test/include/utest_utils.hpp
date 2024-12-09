/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    utest_utils.hpp
 */

#ifndef TEST_INCLUDE_UTEST_UTILS_HPP_
#define TEST_INCLUDE_UTEST_UTILS_HPP_

#include <gtest/gtest.h>

#include <cmath>
#include <vector>
#include <random>
#include <string>
#include <limits>
#include <algorithm>

#ifdef HIEDNN_USE_FP16
#include "datatype_extension/half/hiednn_half.hpp"
#include "datatype_extension/half/hiednn_half_cmath.hpp"
using half_t = hiednn::half;
#endif  // HIEDNN_USE_FP16

#ifdef HIEDNN_USE_BF16
#include "datatype_extension/bfloat16/hiednn_bfloat16.hpp"
#include "datatype_extension/bfloat16/hiednn_bfloat16_cmath.hpp"
using bf16_t = hiednn::bfloat16;
#endif  // HIEDNN_USE_BF16

#define CHECK_HIEDNN(EXPR) \
    ASSERT_EQ((EXPR), HIEDNN_STATUS_SUCCESS);

#define CHECK_CUDA(EXPR)   \
    ASSERT_EQ((EXPR), cudaSuccess);

template <typename T>
inline void CheckEq(const T &x, const T &y) {
    ASSERT_EQ(x, y);
}

template <>
inline void CheckEq<float>(const float &x, const float &y) {
    ASSERT_NEAR(x, y, std::fabs(y) * 1e-5 + 1e-5f);
}

template <>
inline void CheckEq<double>(const double &x, const double &y) {
    ASSERT_NEAR(x, y, std::fabs(y) * 1e-15 + 1e-15);
}

// gen rand data
template <typename T>
inline void GenRand(size_t bufflen, T bias, T *buff) {
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_int_distribution<T> dis(
        std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
    for (size_t index = 0; index < bufflen; index++) {
        buff[index] = bias + static_cast<T>(dis(mt));
    }
}

template <>
inline void GenRand<uint8_t> (size_t bufflen, uint8_t bias, uint8_t *buff) {
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_int_distribution<int> dis(
        std::numeric_limits<uint8_t>::min(),
        std::numeric_limits<uint8_t>::max());
    for (size_t index = 0; index < bufflen; index++) {
        buff[index] = bias + static_cast<uint8_t>(dis(mt));
    }
}

template <>
inline void GenRand<int8_t> (size_t bufflen, int8_t bias, int8_t *buff) {
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_int_distribution<int> dis(
        std::numeric_limits<int8_t>::min(),
        std::numeric_limits<int8_t>::max());
    for (size_t index = 0; index < bufflen; index++) {
        buff[index] = bias + static_cast<int8_t>(dis(mt));
    }
}

#define GEN_RAND_F(TYPE, BASE) \
template <> \
inline void GenRand<TYPE> (size_t bufflen, TYPE bias, TYPE *buff) { \
    std::random_device rd; \
    std::mt19937 mt(rd()); \
    std::normal_distribution<BASE> dis(0, 2); \
    for (size_t index = 0; index < bufflen; index++) { \
        buff[index] = bias + static_cast<TYPE>(dis(mt)); \
    } \
}

#ifdef HIEDNN_USE_FP16
GEN_RAND_F(half_t, float);
#endif  // HIEDNN_USE_FP16

#ifdef HIEDNN_USE_BF16
GEN_RAND_F(bf16_t, float);
#endif  // HIEDNN_USE_BF16

GEN_RAND_F(float,  float);
GEN_RAND_F(double, double);

#undef GEN_RAND_F

template <typename TYPE>
inline void SetValue(size_t bufflen, TYPE value, TYPE *buff) {
    for (size_t index = 0; index < bufflen; index++) {
        buff[index] = value;
    }
    return;
}

#endif  // TEST_INCLUDE_UTEST_UTILS_HPP_

