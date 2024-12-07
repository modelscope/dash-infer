/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    scalar_functor.hpp
 */

#ifndef DNN_INCLUDE_SCALAR_FUNCTOR_HPP_
#define DNN_INCLUDE_SCALAR_FUNCTOR_HPP_

#include <utility>

#include <cmath_wrapper.hpp>
#include <device_function_modifier.hpp>

namespace hiednn {

namespace scalar_functor {

// ------------------------------------------------------------------------
// unary-map scalar functor
// ------------------------------------------------------------------------
#define UNARY_SCALAR_FUNCTOR(FUNC, EXPR) \
template <typename T> \
struct FUNC { \
    DEVICE_FUNCTION T operator()(const T &x) { \
        return EXPR; \
    } \
};

#define UNARY_SCALAR_FUNCTOR_EXTPARAM1(FUNC, EXPR) \
template <typename T> \
struct FUNC { \
    T extParam; \
    \
    explicit FUNC(const void *param) \
        : extParam(*static_cast<const T *>(param)) {} \
    \
    DEVICE_FUNCTION T operator()(const T &x) { \
        return EXPR; \
    } \
};

#define UNARY_SCALAR_FUNCTOR_EXTPARAM2(FUNC, EXPR) \
template <typename T> \
struct FUNC { \
    T extParam1; \
    T extParam2; \
    \
    explicit FUNC(const void *param1, const void *param2) \
        : extParam1(*static_cast<const T *>(param1)), \
          extParam2(*static_cast<const T *>(param2)) {} \
    \
    DEVICE_FUNCTION T operator()(const T &x) { \
        return EXPR; \
    } \
};

// input: bool, output: same as input
UNARY_SCALAR_FUNCTOR(Not, !x);

// input: float, output: bool
template <typename T>
struct IsInf {
    DEVICE_FUNCTION bool operator()(const T &x) {
        return cmath_isinf(x);
    }
};

template <typename T>
struct IsNan {
    DEVICE_FUNCTION bool operator()(const T &x) {
        return cmath_isnan(x);
    }
};

// input: float, output: same as input
UNARY_SCALAR_FUNCTOR(Sqrt, cmath_sqrt(x));
UNARY_SCALAR_FUNCTOR(Cbrt, cmath_cbrt(x));
UNARY_SCALAR_FUNCTOR(Exp, cmath_exp(x));
UNARY_SCALAR_FUNCTOR(Erf, cmath_erf(x));
UNARY_SCALAR_FUNCTOR(Log, cmath_log(x));
UNARY_SCALAR_FUNCTOR(Sin, cmath_sin(x));
UNARY_SCALAR_FUNCTOR(Cos, cmath_cos(x));
UNARY_SCALAR_FUNCTOR(Tan, cmath_tan(x));
UNARY_SCALAR_FUNCTOR(Asin, cmath_asin(x));
UNARY_SCALAR_FUNCTOR(Acos, cmath_acos(x));
UNARY_SCALAR_FUNCTOR(Atan, cmath_atan(x));
UNARY_SCALAR_FUNCTOR(Sinh, cmath_sinh(x));
UNARY_SCALAR_FUNCTOR(Cosh, cmath_cosh(x));
UNARY_SCALAR_FUNCTOR(Tanh, cmath_tanh(x));
UNARY_SCALAR_FUNCTOR(Asinh, cmath_asinh(x));
UNARY_SCALAR_FUNCTOR(Acosh, cmath_acosh(x));
UNARY_SCALAR_FUNCTOR(Atanh, cmath_atanh(x));
UNARY_SCALAR_FUNCTOR(Reciprocal, 1 / x);
UNARY_SCALAR_FUNCTOR(Ceil, cmath_ceil(x));
UNARY_SCALAR_FUNCTOR(Floor, cmath_floor(x));
UNARY_SCALAR_FUNCTOR(Round, cmath_round(x));
UNARY_SCALAR_FUNCTOR(Sigmoid, 1 / (1 + cmath_exp(-x)));
UNARY_SCALAR_FUNCTOR(Softplus, cmath_log1p(cmath_exp(x)));
UNARY_SCALAR_FUNCTOR(Softsign, x / (1 + cmath_fabs(x)));

UNARY_SCALAR_FUNCTOR_EXTPARAM1(PowX, cmath_pow(x, extParam));
UNARY_SCALAR_FUNCTOR_EXTPARAM1(LeakyRelu, x < 0 ? x * extParam : x);
UNARY_SCALAR_FUNCTOR_EXTPARAM1(Elu, x < 0 ? extParam * (cmath_exp(x) - 1) : x);
UNARY_SCALAR_FUNCTOR_EXTPARAM1(Celu, cmath_fmax(T(0), x) +
    cmath_fmin(T(0), extParam * (cmath_exp(x / extParam) - 1)));
UNARY_SCALAR_FUNCTOR_EXTPARAM2(Selu,
    x > 0 ? extParam2 * x : extParam2 * (extParam1 * cmath_exp(x) - extParam1));
UNARY_SCALAR_FUNCTOR_EXTPARAM2(HardSigmoid,
    cmath_fmax(T(0), cmath_fmin(T(1), extParam1 * x + extParam2)));
UNARY_SCALAR_FUNCTOR_EXTPARAM2(HardSwish,
    x * cmath_fmax(T(0), cmath_fmin(T(1), extParam1 * x + extParam2)));

// input: signed type, output: same as input
UNARY_SCALAR_FUNCTOR(Neg, -x);
UNARY_SCALAR_FUNCTOR_EXTPARAM1(ThresholdRelu, x > extParam ? x : T(0));

// input: all type, output: same as input
UNARY_SCALAR_FUNCTOR(Pass, x);
UNARY_SCALAR_FUNCTOR(Sign, x > 0 ? 1 : x == 0 ? 0 : -1);
UNARY_SCALAR_FUNCTOR(Abs, x < 0 ? -x : x);
UNARY_SCALAR_FUNCTOR(Square, x * x);
UNARY_SCALAR_FUNCTOR_EXTPARAM1(AddX, x + extParam);
UNARY_SCALAR_FUNCTOR_EXTPARAM1(MulX, x * extParam);
UNARY_SCALAR_FUNCTOR_EXTPARAM1(DivX, x / extParam);
UNARY_SCALAR_FUNCTOR_EXTPARAM2(Shrink,
    x < -extParam1 ? x + extParam2 : x > extParam1 ? x - extParam2 : T(0));
UNARY_SCALAR_FUNCTOR_EXTPARAM2(Clip,
    x < extParam1 ? extParam1 : x > extParam2 ? extParam2 : x);

#undef UNARY_SCALAR_FUNCTOR
#undef UNARY_SCALAR_FUNCTOR_EXTPARAM1
#undef UNARY_SCALAR_FUNCTOR_EXTPARAM2

// ------------------------------------------------------------------------
// binary-map scalar functor
// ------------------------------------------------------------------------
#define BINARY_SCALAR_FUNCTOR(FUNC, EXPR) \
template <typename T> \
struct FUNC { \
    DEVICE_FUNCTION T operator()(const T &x, const T &y) { \
        return EXPR; \
    } \
};

#define BINARY_LOGICAL_FUNCTOR(FUNC, EXPR) \
template <typename T> \
struct FUNC { \
    DEVICE_FUNCTION bool operator()(const T &x, const T &y) { \
        return EXPR; \
    } \
};

BINARY_SCALAR_FUNCTOR(Add, x + y)
BINARY_SCALAR_FUNCTOR(Sub, x - y)
BINARY_SCALAR_FUNCTOR(Mul, x * y)
BINARY_SCALAR_FUNCTOR(Div, x / y)
BINARY_SCALAR_FUNCTOR(Max, x > y ? x : y)
BINARY_SCALAR_FUNCTOR(Min, x < y ? x : y)
BINARY_SCALAR_FUNCTOR(LogicalShiftR, x >> y)
BINARY_SCALAR_FUNCTOR(LogicalShiftL, x << y)

// FMod: the sign of the remainder is same as that of the divisor.
template <typename T>
struct FMod {
    DEVICE_FUNCTION T operator()(const T &x, const T &y) {
        return y == 0 ? 0 : x % y;
    }
};

template <>
struct FMod<float> {
    DEVICE_FUNCTION float operator()(const float &x, const float &y) {
        return cmath_fmod(x, y);
    }
};

template <>
struct FMod<double> {
    DEVICE_FUNCTION double operator()(const double &x, const double &y) {
        return cmath_fmod(x, y);
    }
};

#ifdef HIEDNN_USE_FP16
template <>
struct FMod<half> {
    DEVICE_FUNCTION half operator()(const half &x, const half &y) {
        return cmath_fmod(x, y);
    }
};
#endif

#ifdef HIEDNN_USE_BF16
template <>
struct FMod<bfloat16> {
    DEVICE_FUNCTION bfloat16 operator()(const bfloat16 &x, const bfloat16 &y) {
        return cmath_fmod(x, y);
    }
};
#endif

// Mod: the sign of the remainder is same as that of the dividend.
// only work for integer
template <typename T>
struct Mod {
    DEVICE_FUNCTION T operator()(const T &x, const T &y) {
        if (y == 0) {
            return 0;
        } else {
            T mod = x % y;
            return mod * y >= 0 ? mod : mod + y;
        }
    }
};

// specialized Mod functor for unsigned to aovid
// 'comparison of unsigned integer with zero' warning
#define MOD_FUNCTOR_UNSIGNED(TYPE) \
template <> \
struct Mod<TYPE> { \
    DEVICE_FUNCTION TYPE operator()(const TYPE &x, const TYPE &y) { \
        if (y == 0) { \
            return 0; \
        } else { \
            return x % y; \
        } \
    } \
};
MOD_FUNCTOR_UNSIGNED(uint64_t)
MOD_FUNCTOR_UNSIGNED(uint32_t)
MOD_FUNCTOR_UNSIGNED(uint16_t)
MOD_FUNCTOR_UNSIGNED(uint8_t)
#undef MOD_FUNCTOR_UNSIGNED

BINARY_SCALAR_FUNCTOR(PRelu, x > 0 ? x : x * y)
BINARY_LOGICAL_FUNCTOR(CompareEQ, x == y)
BINARY_LOGICAL_FUNCTOR(CompareGT, x > y)
BINARY_LOGICAL_FUNCTOR(CompareGE, x >= y)
BINARY_LOGICAL_FUNCTOR(CompareLT, x < y)
BINARY_LOGICAL_FUNCTOR(CompareLE, x <= y)
BINARY_LOGICAL_FUNCTOR(LogicalAnd, (x && y))
BINARY_LOGICAL_FUNCTOR(LogicalOr, (x || y))
BINARY_LOGICAL_FUNCTOR(LogicalXor, (!x != !y))

#undef BINARY_SCALAR_FUNCTOR
#undef BINARY_CMP_SCALAR_FUNCTOR

}  // namespace scalar_functor

template <template <typename> class ScalarOp>
struct GetScalarOp {
    template <typename T, typename ...Arg>
    ScalarOp<T> get(Arg&&... args) const {
        return ScalarOp<T>(std::forward<Arg>(args)...);
    }
};

}  // namespace hiednn

#endif  // DNN_INCLUDE_SCALAR_FUNCTOR_HPP_


