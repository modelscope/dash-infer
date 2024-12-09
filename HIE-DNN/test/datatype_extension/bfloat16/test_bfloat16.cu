/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    test_bfloat16.cu
 */

#include <gtest/gtest.h>
#include <cmath>
#include "datatype_extension/bfloat16/hiednn_bfloat16.hpp"
#include "datatype_extension/bfloat16/hiednn_bfloat16_cmath.hpp"

namespace test_bf16_cuda {

template <typename T>
__global__ void cast_to_bf16_kernel(T val, hiednn::bfloat16* dst) {
    *dst = static_cast<hiednn::bfloat16>(val);
}

template <typename T>
__global__ void compute_diff_kernel(T val, hiednn::bfloat16* dst, float* diff) {
    *diff = static_cast<float>(val - static_cast<T>(*dst));
}

template <typename T>
__global__ void
compute_diff_kernel_f(T val, hiednn::bfloat16* dst, double* diff) {
    T hval = static_cast<T>(*dst);
    *diff = fabs(static_cast<double>(val) - hval);
}

template <typename T>
void test_type_conversion(T val) {
    hiednn::bfloat16* dst;
    cudaMalloc(&dst, sizeof(hiednn::bfloat16));
    float* diff_dev;
    cudaMalloc(&diff_dev, sizeof(float));
    cast_to_bf16_kernel<T><<<1, 1, 0, 0>>>(val, dst);
    compute_diff_kernel<T><<<1, 1, 0, 0>>>(val, dst, diff_dev);
    float diff_host;
    cudaMemcpy(&diff_host, diff_dev, sizeof(float), cudaMemcpyDeviceToHost);
    ASSERT_EQ(diff_host, 0);
    cudaFree(dst);
    cudaFree(diff_dev);
}

template <typename T>
void test_type_conversion_f(T val) {
    hiednn::bfloat16* dst;
    cudaMalloc(&dst, sizeof(hiednn::bfloat16));
    double* diff_dev;
    cudaMalloc(&diff_dev, sizeof(double));
    cast_to_bf16_kernel<T><<<1, 1, 0, 0>>>(val, dst);
    compute_diff_kernel_f<T><<<1, 1, 0, 0>>>(val, dst, diff_dev);
    double diff_host;
    cudaMemcpy(&diff_host, diff_dev, sizeof(double), cudaMemcpyDeviceToHost);
    ASSERT_LE(diff_host, fabs(static_cast<double>(val) * 1e-2) + 1e-2);
    cudaFree(dst);
    cudaFree(diff_dev);
}

#define BF16_BINARY_OPERATOR_KERNEL(op, expr) \
template <typename T> \
__global__ void bf16_type_##expr##_kernel(float a, T b, float* res) { \
    hiednn::bfloat16 tmp = static_cast<hiednn::bfloat16>(a); \
    *res = static_cast<float>(tmp op b); \
} \
template <typename T> \
__global__ void type_bf16_##expr##_kernel(float a, T b, float* res) { \
    hiednn::bfloat16 tmp = static_cast<hiednn::bfloat16>(a); \
    *res = static_cast<float>(b op tmp); \
}

BF16_BINARY_OPERATOR_KERNEL(+, add)
BF16_BINARY_OPERATOR_KERNEL(-, sub)
BF16_BINARY_OPERATOR_KERNEL(*, mul)
BF16_BINARY_OPERATOR_KERNEL(/, div)
#undef BF16_BINARY_OPERATOR_KERNEL

#define TEST_BF16_BINARY_OPERATOR(op, expr) \
template <typename T> \
void test_bf16_##expr##_operator(float a, T b) { \
    float res = static_cast<float>(a op b); \
    float* res_dev; \
    cudaMalloc(&res_dev, sizeof(float)); \
    float res_host; \
    bf16_type_##expr##_kernel<T><<<1, 1, 0, 0>>>(a, b, res_dev); \
    cudaMemcpy(&res_host, res_dev, sizeof(float), cudaMemcpyDeviceToHost); \
    ASSERT_NEAR(res, res_host, fabs(res_host * 1e-2f) + 1e-1f); \
    res = static_cast<float>(b op a); \
    type_bf16_##expr##_kernel<T><<<1, 1, 0, 0>>>(a, b, res_dev); \
    cudaMemcpy(&res_host, res_dev, sizeof(float), cudaMemcpyDeviceToHost); \
    ASSERT_NEAR(res, res_host, fabs(res_host * 1e-2f) + 1e-1f); \
    cudaFree(res_dev); \
}

TEST_BF16_BINARY_OPERATOR(+, add)
TEST_BF16_BINARY_OPERATOR(-, sub)
TEST_BF16_BINARY_OPERATOR(*, mul)
TEST_BF16_BINARY_OPERATOR(/, div)
#undef TEST_BF16_BINARY_OPERATOR

#define BF16_CMP_OPERATOR_KERNEL(op, expr) \
template <typename T> \
__global__ void bf16_type_##expr##_kernel(float a, T b, bool* res) { \
    hiednn::bfloat16 tmp = static_cast<hiednn::bfloat16>(a); \
    *res = (tmp op b); \
} \
template <typename T> \
__global__ void type_bf16_##expr##_kernel(float a, T b, bool* res) { \
    hiednn::bfloat16 tmp = static_cast<hiednn::bfloat16>(a); \
    *res = (b op tmp); \
}

BF16_CMP_OPERATOR_KERNEL(==, eq)
BF16_CMP_OPERATOR_KERNEL(!=, neq)
BF16_CMP_OPERATOR_KERNEL(>, gt)
BF16_CMP_OPERATOR_KERNEL(<, lt)
BF16_CMP_OPERATOR_KERNEL(>=, ge)
BF16_CMP_OPERATOR_KERNEL(<=, le)
#undef BF16_CMP_OPERATOR_KERNEL

#define TEST_BF16_CMP_OPERATOR(op, expr) \
template <typename T> \
void test_bf16_##expr##_operator(float a, T b) { \
    bool res = a op b; \
    bool* res_dev; \
    cudaMalloc(&res_dev, sizeof(bool)); \
    bool res_host; \
    bf16_type_##expr##_kernel<T><<<1, 1, 0, 0>>>(a, b, res_dev); \
    cudaMemcpy(&res_host, res_dev, sizeof(bool), cudaMemcpyDeviceToHost); \
    ASSERT_EQ(res, res_host); \
    res = b op a; \
    type_bf16_##expr##_kernel<T><<<1, 1, 0, 0>>>(a, b, res_dev); \
    cudaMemcpy(&res_host, res_dev, sizeof(bool), cudaMemcpyDeviceToHost); \
    ASSERT_EQ(res, res_host); \
    cudaFree(res_dev); \
}

TEST_BF16_CMP_OPERATOR(==, eq)
TEST_BF16_CMP_OPERATOR(!=, neq)
TEST_BF16_CMP_OPERATOR(>, gt)
TEST_BF16_CMP_OPERATOR(<, lt)
TEST_BF16_CMP_OPERATOR(>=, ge)
TEST_BF16_CMP_OPERATOR(<=, le)
#undef TEST_BF16_CMP_OPERATOR

#define BF16_CMATH_FUNCTION_HH_KERNEL(expr) \
__global__ void bf16_##expr##_function_kernel(float a, float* res) { \
    *res = static_cast<float>(expr##_bf16(static_cast<hiednn::bfloat16>(a))); \
}
#define BF16_CMATH_FUNCTION_HHH_KERNEL(expr) \
__global__ void bf16_##expr##_function_kernel(float a, float b, float* res) { \
    *res = static_cast<float>( \
            expr##_bf16(static_cast<hiednn::bfloat16>(a), \
                static_cast<hiednn::bfloat16>(b))); \
}

BF16_CMATH_FUNCTION_HH_KERNEL(fabs)
BF16_CMATH_FUNCTION_HHH_KERNEL(fmod)
BF16_CMATH_FUNCTION_HHH_KERNEL(remainder)
__global__ void
bf16_fma_function_kernel(float a, float b, float c, float* res) {
    *res = static_cast<float>(fma_bf16(static_cast<hiednn::bfloat16>(a),
                                       static_cast<hiednn::bfloat16>(b),
                                       static_cast<hiednn::bfloat16>(c)));
}
BF16_CMATH_FUNCTION_HHH_KERNEL(fmax)
BF16_CMATH_FUNCTION_HHH_KERNEL(fmin)
BF16_CMATH_FUNCTION_HHH_KERNEL(fdim)

BF16_CMATH_FUNCTION_HH_KERNEL(exp)
BF16_CMATH_FUNCTION_HH_KERNEL(exp2)
BF16_CMATH_FUNCTION_HH_KERNEL(expm1)
BF16_CMATH_FUNCTION_HH_KERNEL(log)
BF16_CMATH_FUNCTION_HH_KERNEL(log10)
BF16_CMATH_FUNCTION_HH_KERNEL(log2)
BF16_CMATH_FUNCTION_HH_KERNEL(log1p)

BF16_CMATH_FUNCTION_HHH_KERNEL(pow)
BF16_CMATH_FUNCTION_HH_KERNEL(sqrt)
BF16_CMATH_FUNCTION_HH_KERNEL(rsqrt)
BF16_CMATH_FUNCTION_HH_KERNEL(cbrt)
BF16_CMATH_FUNCTION_HHH_KERNEL(hypot)

BF16_CMATH_FUNCTION_HH_KERNEL(sin)
BF16_CMATH_FUNCTION_HH_KERNEL(cos)
BF16_CMATH_FUNCTION_HH_KERNEL(tan)
BF16_CMATH_FUNCTION_HH_KERNEL(asin)
BF16_CMATH_FUNCTION_HH_KERNEL(acos)
BF16_CMATH_FUNCTION_HH_KERNEL(atan)
BF16_CMATH_FUNCTION_HHH_KERNEL(atan2)

BF16_CMATH_FUNCTION_HH_KERNEL(sinh)
BF16_CMATH_FUNCTION_HH_KERNEL(cosh)
BF16_CMATH_FUNCTION_HH_KERNEL(tanh)
BF16_CMATH_FUNCTION_HH_KERNEL(asinh)
BF16_CMATH_FUNCTION_HH_KERNEL(acosh)
BF16_CMATH_FUNCTION_HH_KERNEL(atanh)

BF16_CMATH_FUNCTION_HH_KERNEL(erf)
BF16_CMATH_FUNCTION_HH_KERNEL(erfc)
BF16_CMATH_FUNCTION_HH_KERNEL(tgamma)
BF16_CMATH_FUNCTION_HH_KERNEL(lgamma)

BF16_CMATH_FUNCTION_HH_KERNEL(ceil)
BF16_CMATH_FUNCTION_HH_KERNEL(floor)
BF16_CMATH_FUNCTION_HH_KERNEL(trunc)
BF16_CMATH_FUNCTION_HH_KERNEL(round)
BF16_CMATH_FUNCTION_HH_KERNEL(nearbyint)
BF16_CMATH_FUNCTION_HH_KERNEL(rint)
#undef BF16_CMATH_FUNCTION_HH_KERNEL
#undef BF16_CMATH_FUNCTION_HHH_KERNEL

#define TEST_BF16_CMATH_FUNCTION_HH(expr) \
void test_bf16_##expr##_function(float a) { \
    float res = expr(a); \
    float* res_dev; \
    cudaMalloc(&res_dev, sizeof(float)); \
    float res_host; \
    bf16_##expr##_function_kernel<<<1, 1, 0, 0>>>(a, res_dev); \
    cudaMemcpy(&res_host, res_dev, sizeof(float), cudaMemcpyDeviceToHost); \
    ASSERT_LT(fabs(static_cast<double>(res) - res_host), \
              fabs(static_cast<double>(res_host) * 1e-2) + 1e-1); \
    cudaFree(res_dev); \
}

#define TEST_BF16_CMATH_FUNCTION_HHH(expr) \
void test_bf16_##expr##_function(float a, float b) { \
    float res = expr(a, b); \
    float* res_dev; \
    cudaMalloc(&res_dev, sizeof(float)); \
    float res_host; \
    bf16_##expr##_function_kernel<<<1, 1, 0, 0>>>(a, b, res_dev); \
    cudaMemcpy(&res_host, res_dev, sizeof(float), cudaMemcpyDeviceToHost); \
    ASSERT_LT(fabs(static_cast<double>(res) - res_host), \
              fabs(static_cast<double>(res_host) * 1e-2) + 1e-1); \
    cudaFree(res_dev); \
}

TEST_BF16_CMATH_FUNCTION_HH(fabs)
TEST_BF16_CMATH_FUNCTION_HHH(fmod)
TEST_BF16_CMATH_FUNCTION_HHH(remainder)
void test_bf16_fma_function(float a, float b, float c) {
    float res = fma(a, b, c);
    float res_host;
    float* res_dev;
    cudaMalloc(&res_dev, sizeof(float));
    bf16_fma_function_kernel<<<1, 1, 0, 0>>>(a, b, c, res_dev);
    cudaMemcpy(&res_host, res_dev, sizeof(float), cudaMemcpyDeviceToHost);
    ASSERT_LT(fabs(static_cast<double>(res) - res_host),
              fabs(static_cast<double>(res_host) * 1e-2) + 1e-1);
    cudaFree(res_dev);
}
TEST_BF16_CMATH_FUNCTION_HHH(fmax)
TEST_BF16_CMATH_FUNCTION_HHH(fmin)
TEST_BF16_CMATH_FUNCTION_HHH(fdim)

TEST_BF16_CMATH_FUNCTION_HH(exp)
TEST_BF16_CMATH_FUNCTION_HH(exp2)
TEST_BF16_CMATH_FUNCTION_HH(expm1)
TEST_BF16_CMATH_FUNCTION_HH(log)
TEST_BF16_CMATH_FUNCTION_HH(log10)
TEST_BF16_CMATH_FUNCTION_HH(log2)
TEST_BF16_CMATH_FUNCTION_HH(log1p)

TEST_BF16_CMATH_FUNCTION_HHH(pow)
TEST_BF16_CMATH_FUNCTION_HH(sqrt)
TEST_BF16_CMATH_FUNCTION_HH(rsqrt)
TEST_BF16_CMATH_FUNCTION_HH(cbrt)
TEST_BF16_CMATH_FUNCTION_HHH(hypot)

TEST_BF16_CMATH_FUNCTION_HH(sin)
TEST_BF16_CMATH_FUNCTION_HH(cos)
TEST_BF16_CMATH_FUNCTION_HH(tan)
TEST_BF16_CMATH_FUNCTION_HH(asin)
TEST_BF16_CMATH_FUNCTION_HH(acos)
TEST_BF16_CMATH_FUNCTION_HH(atan)
TEST_BF16_CMATH_FUNCTION_HHH(atan2)

TEST_BF16_CMATH_FUNCTION_HH(sinh)
TEST_BF16_CMATH_FUNCTION_HH(cosh)
TEST_BF16_CMATH_FUNCTION_HH(tanh)
TEST_BF16_CMATH_FUNCTION_HH(asinh)
TEST_BF16_CMATH_FUNCTION_HH(acosh)
TEST_BF16_CMATH_FUNCTION_HH(atanh)

TEST_BF16_CMATH_FUNCTION_HH(erf)
TEST_BF16_CMATH_FUNCTION_HH(erfc)
TEST_BF16_CMATH_FUNCTION_HH(tgamma)
TEST_BF16_CMATH_FUNCTION_HH(lgamma)

TEST_BF16_CMATH_FUNCTION_HH(ceil)
TEST_BF16_CMATH_FUNCTION_HH(floor)
TEST_BF16_CMATH_FUNCTION_HH(trunc)
TEST_BF16_CMATH_FUNCTION_HH(round)
TEST_BF16_CMATH_FUNCTION_HH(nearbyint)
TEST_BF16_CMATH_FUNCTION_HH(rint)
#undef TEST_BF16_CMATH_FUNCTION_HH
#undef TEST_BF16_CMATH_FUNCTION_HHH

}  // namespace test_bf16_cuda

TEST(datatype_test, bf16_type_conversion_cuda) {
    test_bf16_cuda::test_type_conversion_f<float>(13.4);
    test_bf16_cuda::test_type_conversion_f<double>(13.4);
    test_bf16_cuda::test_type_conversion<int64_t>(17);
    test_bf16_cuda::test_type_conversion<uint64_t>(17);
    test_bf16_cuda::test_type_conversion<int32_t>(17);
    test_bf16_cuda::test_type_conversion<uint32_t>(17);
    test_bf16_cuda::test_type_conversion<int16_t>(17);
    test_bf16_cuda::test_type_conversion<uint16_t>(17);
    test_bf16_cuda::test_type_conversion<unsigned char>(17);
    test_bf16_cuda::test_type_conversion<signed char>(17);
    test_bf16_cuda::test_type_conversion<bool>(1);
}

TEST(datatype_test, bf16_basic_operator_cuda) {
    // +
    test_bf16_cuda::test_bf16_add_operator<float>(3.1, 3.2);
    test_bf16_cuda::test_bf16_add_operator<double>(3.1, 3.2);
    test_bf16_cuda::test_bf16_add_operator<int64_t>(3.1, 3);
    test_bf16_cuda::test_bf16_add_operator<uint64_t>(3.1, 3);
    test_bf16_cuda::test_bf16_add_operator<int32_t>(3.1, 3);
    test_bf16_cuda::test_bf16_add_operator<uint32_t>(3.1, 3);
    test_bf16_cuda::test_bf16_add_operator<int16_t>(3.1, 1);
    test_bf16_cuda::test_bf16_add_operator<uint16_t>(3.1, 10);
    test_bf16_cuda::test_bf16_add_operator<char>(2.5, 2);
    test_bf16_cuda::test_bf16_add_operator<signed char>(2.7, 1);
    test_bf16_cuda::test_bf16_add_operator<unsigned char>(5.5, 3);
    test_bf16_cuda::test_bf16_add_operator<bool>(8.7, 1);
    // -
    test_bf16_cuda::test_bf16_sub_operator<float>(10.1, 2.5);
    test_bf16_cuda::test_bf16_sub_operator<double>(2.9, 11.2);
    test_bf16_cuda::test_bf16_sub_operator<int64_t>(17.4, 8);
    test_bf16_cuda::test_bf16_sub_operator<uint64_t>(2.5, 1);
    test_bf16_cuda::test_bf16_sub_operator<int32_t>(8.8, 4);
    test_bf16_cuda::test_bf16_sub_operator<uint32_t>(9.7, 9);
    test_bf16_cuda::test_bf16_sub_operator<int16_t>(2.5, 2);
    test_bf16_cuda::test_bf16_sub_operator<uint16_t>(3.0, 1);
    test_bf16_cuda::test_bf16_sub_operator<char>(8.8, 9);
    test_bf16_cuda::test_bf16_sub_operator<signed char>(2.6, 2);
    test_bf16_cuda::test_bf16_sub_operator<unsigned char>(19.2, 5);
    test_bf16_cuda::test_bf16_sub_operator<bool>(10.8, 1);
    // *
    test_bf16_cuda::test_bf16_mul_operator<float>(2.6, 2.6);
    test_bf16_cuda::test_bf16_mul_operator<double>(3.7, 9.1);
    test_bf16_cuda::test_bf16_mul_operator<int64_t>(2.8, 4);
    test_bf16_cuda::test_bf16_mul_operator<uint64_t>(7.6, 3);
    test_bf16_cuda::test_bf16_mul_operator<int32_t>(3.4, 6);
    test_bf16_cuda::test_bf16_mul_operator<uint32_t>(3.5, 8);
    test_bf16_cuda::test_bf16_mul_operator<int16_t>(2.1, 2);
    test_bf16_cuda::test_bf16_mul_operator<uint16_t>(2.4, 1);
    test_bf16_cuda::test_bf16_mul_operator<char>(1.6, 5);
    test_bf16_cuda::test_bf16_mul_operator<signed char>(1.2, 4);
    test_bf16_cuda::test_bf16_mul_operator<unsigned char>(3.4, 2);
    test_bf16_cuda::test_bf16_mul_operator<bool>(2.9, 1);
    // /
    test_bf16_cuda::test_bf16_div_operator<float>(2.8, 1.4);
    test_bf16_cuda::test_bf16_div_operator<double>(3.9, 2.6);
    test_bf16_cuda::test_bf16_div_operator<int64_t>(8.1, 3);
    test_bf16_cuda::test_bf16_div_operator<uint64_t>(7.5, 2);
    test_bf16_cuda::test_bf16_div_operator<int32_t>(4.6, 2);
    test_bf16_cuda::test_bf16_div_operator<uint32_t>(2.8, 2);
    test_bf16_cuda::test_bf16_div_operator<int16_t>(9.4, 5);
    test_bf16_cuda::test_bf16_div_operator<uint16_t>(3.9, 3);
    test_bf16_cuda::test_bf16_div_operator<char>(6.6, 3);
    test_bf16_cuda::test_bf16_div_operator<signed char>(2.8, 4);
    test_bf16_cuda::test_bf16_div_operator<unsigned char>(11.5, 2);
    test_bf16_cuda::test_bf16_div_operator<bool>(7.7, 1);
    // ==
    test_bf16_cuda::test_bf16_eq_operator<float>(2.8, 2.2);
    test_bf16_cuda::test_bf16_eq_operator<double>(1.5, 14.2);
    test_bf16_cuda::test_bf16_eq_operator<int64_t>(3.6, 3);
    test_bf16_cuda::test_bf16_eq_operator<uint64_t>(2.0, 2);
    test_bf16_cuda::test_bf16_eq_operator<int32_t>(3.5, 1.2);
    test_bf16_cuda::test_bf16_eq_operator<uint32_t>(1.6, 1.3);
    test_bf16_cuda::test_bf16_eq_operator<int16_t>(1.0, 1);
    test_bf16_cuda::test_bf16_eq_operator<uint16_t>(2.7, 1);
    test_bf16_cuda::test_bf16_eq_operator<char>(13.4, 3);
    test_bf16_cuda::test_bf16_eq_operator<signed char>(0.6, 2);
    test_bf16_cuda::test_bf16_eq_operator<unsigned char>(0.9, 4);
    test_bf16_cuda::test_bf16_eq_operator<bool>(1.0, 1);
    // !=
    test_bf16_cuda::test_bf16_neq_operator<float>(2.8, 2.2);
    test_bf16_cuda::test_bf16_neq_operator<double>(1.5, 14.2);
    test_bf16_cuda::test_bf16_neq_operator<int64_t>(3.6, 3);
    test_bf16_cuda::test_bf16_neq_operator<uint64_t>(2.0, 2);
    test_bf16_cuda::test_bf16_neq_operator<int32_t>(3.5, 1.2);
    test_bf16_cuda::test_bf16_neq_operator<uint32_t>(1.6, 1.3);
    test_bf16_cuda::test_bf16_neq_operator<int16_t>(1.0, 1);
    test_bf16_cuda::test_bf16_neq_operator<uint16_t>(2.7, 1);
    test_bf16_cuda::test_bf16_neq_operator<char>(13.4, 3);
    test_bf16_cuda::test_bf16_neq_operator<signed char>(0.6, 2);
    test_bf16_cuda::test_bf16_neq_operator<unsigned char>(0.9, 4);
    test_bf16_cuda::test_bf16_neq_operator<bool>(1.0, 1);
    // >
    test_bf16_cuda::test_bf16_gt_operator<float>(2.8, 2.2);
    test_bf16_cuda::test_bf16_gt_operator<double>(1.5, 14.2);
    test_bf16_cuda::test_bf16_gt_operator<int64_t>(3.6, 3);
    test_bf16_cuda::test_bf16_gt_operator<uint64_t>(2.0, 2);
    test_bf16_cuda::test_bf16_gt_operator<int32_t>(3.5, 1.2);
    test_bf16_cuda::test_bf16_gt_operator<uint32_t>(1.6, 1.3);
    test_bf16_cuda::test_bf16_gt_operator<int16_t>(1.0, 1);
    test_bf16_cuda::test_bf16_gt_operator<uint16_t>(2.7, 1);
    test_bf16_cuda::test_bf16_gt_operator<char>(13.4, 3);
    test_bf16_cuda::test_bf16_gt_operator<signed char>(0.6, 2);
    test_bf16_cuda::test_bf16_gt_operator<unsigned char>(0.9, 4);
    test_bf16_cuda::test_bf16_gt_operator<bool>(1.0, 1);
    // <
    test_bf16_cuda::test_bf16_lt_operator<float>(2.8, 2.2);
    test_bf16_cuda::test_bf16_lt_operator<double>(1.5, 14.2);
    test_bf16_cuda::test_bf16_lt_operator<int64_t>(3.6, 3);
    test_bf16_cuda::test_bf16_lt_operator<uint64_t>(2.0, 2);
    test_bf16_cuda::test_bf16_lt_operator<int32_t>(3.5, 1.2);
    test_bf16_cuda::test_bf16_lt_operator<uint32_t>(1.6, 1.3);
    test_bf16_cuda::test_bf16_lt_operator<int16_t>(1.0, 1);
    test_bf16_cuda::test_bf16_lt_operator<uint16_t>(2.7, 1);
    test_bf16_cuda::test_bf16_lt_operator<char>(13.4, 3);
    test_bf16_cuda::test_bf16_lt_operator<signed char>(0.6, 2);
    test_bf16_cuda::test_bf16_lt_operator<unsigned char>(0.9, 4);
    test_bf16_cuda::test_bf16_lt_operator<bool>(1.0, 1);
    // >=
    test_bf16_cuda::test_bf16_ge_operator<float>(2.8, 2.2);
    test_bf16_cuda::test_bf16_ge_operator<double>(1.5, 14.2);
    test_bf16_cuda::test_bf16_ge_operator<int64_t>(3.6, 3);
    test_bf16_cuda::test_bf16_ge_operator<uint64_t>(2.0, 2);
    test_bf16_cuda::test_bf16_ge_operator<int32_t>(3.5, 1.2);
    test_bf16_cuda::test_bf16_ge_operator<uint32_t>(1.6, 1.3);
    test_bf16_cuda::test_bf16_ge_operator<int16_t>(1.0, 1);
    test_bf16_cuda::test_bf16_ge_operator<uint16_t>(2.7, 1);
    test_bf16_cuda::test_bf16_ge_operator<char>(13.4, 3);
    test_bf16_cuda::test_bf16_ge_operator<signed char>(0.6, 2);
    test_bf16_cuda::test_bf16_ge_operator<unsigned char>(0.9, 4);
    test_bf16_cuda::test_bf16_ge_operator<bool>(1.0, 1);
    // <=
    test_bf16_cuda::test_bf16_le_operator<float>(2.8, 2.2);
    test_bf16_cuda::test_bf16_le_operator<double>(1.5, 14.2);
    test_bf16_cuda::test_bf16_le_operator<int64_t>(3.6, 3);
    test_bf16_cuda::test_bf16_le_operator<uint64_t>(2.0, 2);
    test_bf16_cuda::test_bf16_le_operator<int32_t>(3.5, 1.2);
    test_bf16_cuda::test_bf16_le_operator<uint32_t>(1.6, 1.3);
    test_bf16_cuda::test_bf16_le_operator<int16_t>(1.0, 1);
    test_bf16_cuda::test_bf16_le_operator<uint16_t>(2.7, 1);
    test_bf16_cuda::test_bf16_le_operator<char>(13.4, 3);
    test_bf16_cuda::test_bf16_le_operator<signed char>(0.6, 2);
    test_bf16_cuda::test_bf16_le_operator<unsigned char>(0.9, 4);
    test_bf16_cuda::test_bf16_le_operator<bool>(1.0, 1);
}

TEST(datatype_test, bf16_cmath_function_cuda) {
    test_bf16_cuda::test_bf16_fabs_function(-1.2f);
    test_bf16_cuda::test_bf16_fmod_function(2.3f, 1.8f);
    test_bf16_cuda::test_bf16_fma_function(2.5f, 1.9f, 0.8f);
    test_bf16_cuda::test_bf16_fmax_function(6.2f, 1.3f);
    test_bf16_cuda::test_bf16_fmin_function(6.2f, 1.3f);
    test_bf16_cuda::test_bf16_fdim_function(9.2f, 6.2f);

    test_bf16_cuda::test_bf16_exp_function(1.3f);
    test_bf16_cuda::test_bf16_exp2_function(1.6f);
    test_bf16_cuda::test_bf16_expm1_function(2.8f);
    test_bf16_cuda::test_bf16_log_function(3.5f);
    test_bf16_cuda::test_bf16_log10_function(9.1f);
    test_bf16_cuda::test_bf16_log2_function(7.6f);
    test_bf16_cuda::test_bf16_log1p_function(1.7f);

    test_bf16_cuda::test_bf16_pow_function(3.2f, 1.7f);
    test_bf16_cuda::test_bf16_sqrt_function(7.1f);
    test_bf16_cuda::test_bf16_rsqrt_function(2.2f);
    test_bf16_cuda::test_bf16_cbrt_function(2.8f);
    test_bf16_cuda::test_bf16_hypot_function(2.9f, 1.3f);

    test_bf16_cuda::test_bf16_sin_function(1.8f);
    test_bf16_cuda::test_bf16_cos_function(1.5f);
    test_bf16_cuda::test_bf16_tan_function(0.6f);
    test_bf16_cuda::test_bf16_asin_function(0.5f);
    test_bf16_cuda::test_bf16_acos_function(0.5f);
    test_bf16_cuda::test_bf16_atan_function(1.2f);
    test_bf16_cuda::test_bf16_atan2_function(1.1f, 2.3f);

    test_bf16_cuda::test_bf16_sinh_function(1.4f);
    test_bf16_cuda::test_bf16_sinh_function(-1.4f);
    test_bf16_cuda::test_bf16_cosh_function(2.8f);
    test_bf16_cuda::test_bf16_cosh_function(-2.8f);
    test_bf16_cuda::test_bf16_tanh_function(1.9f);
    test_bf16_cuda::test_bf16_tanh_function(-1.9f);
    test_bf16_cuda::test_bf16_asinh_function(0.2f);
    test_bf16_cuda::test_bf16_acosh_function(1.9f);
    test_bf16_cuda::test_bf16_atanh_function(0.3f);

    test_bf16_cuda::test_bf16_erf_function(0.7f);
    test_bf16_cuda::test_bf16_erfc_function(0.9f);
    test_bf16_cuda::test_bf16_tgamma_function(1.2f);
    test_bf16_cuda::test_bf16_lgamma_function(0.3f);

    test_bf16_cuda::test_bf16_ceil_function(1.5f);
    test_bf16_cuda::test_bf16_floor_function(1.5f);
    test_bf16_cuda::test_bf16_trunc_function(1.5f);
    test_bf16_cuda::test_bf16_round_function(1.5f);
    test_bf16_cuda::test_bf16_nearbyint_function(1.5f);
    test_bf16_cuda::test_bf16_rint_function(1.5f);
}
