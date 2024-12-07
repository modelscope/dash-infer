/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    integer_divmod.hpp
 */

#ifndef DNN_INCLUDE_INTEGER_DIVMOD_HPP_
#define DNN_INCLUDE_INTEGER_DIVMOD_HPP_

#include <cstdint>
#include <device_function_modifier.hpp>

namespace hiednn {

// ------------------------------------------
// fast integer division
// ------------------------------------------
template <typename T>
class IntDiv {
 public:
    T d_;

    template <typename DT>
    static bool OutOfBound(const DT *d, int count) {
        return true;
    }

    IntDiv() {}

    explicit IntDiv(T d) : d_(d) {}

    DEVICE_FUNCTION T Div(T n) const {
        return n / d_;
    }
};

template <>
class IntDiv<uint32_t> {
 public:
    template <typename DT>
    static bool OutOfBound(const DT *d, int count) {
        for (int i = 0; i < count; ++i) {
            if (static_cast<uint64_t>(d[i]) > INT32_MAX) {
                return true;
            }
        }
        return false;
    }

    IntDiv() {}

    explicit IntDiv(uint32_t d) {
        // assert(d >= 1 && d <= INT32_MAX);

        for (shift_ = 0; shift_ < 32; ++shift_) {
            if ((1u << shift_) >= d) {
                break;
            }
        }
        uint64_t tmp_magic = ((1lu << 32) * ((1lu << shift_) - d)) / d + 1;
        magic_ = tmp_magic;  // copy lower 32-bit
        // assert(magic_ != 0 && magic_ == tmp_magic);
    }

    DEVICE_FUNCTION uint32_t Div(uint32_t n) const {
#ifdef __CUDA_ARCH__
        return (__umulhi(n, magic_) + n) >> shift_;
#else
        uint64_t tmp = (static_cast<uint64_t>(n) * magic_) >> 32;
        return (static_cast<uint32_t>(tmp) + n) >> shift_;
#endif
    }

 private:
    uint32_t magic_;
    uint32_t shift_;
};

// ------------------------------------------
// fast integer division and mod
// ------------------------------------------
template <typename T>
struct DivModRet {
    T div;
    T mod;
    DEVICE_FUNCTION DivModRet(T d, T m) : div(d), mod(m) {}
};

template <typename T>
class IntDivMod {
 public:
    uint32_t d_;

    template <typename DT>
    static bool OutOfBound(const DT *d, int count) {
        return true;
    }

    IntDivMod() {}

    explicit IntDivMod(T d) : d_(d) {}

    DEVICE_FUNCTION T Div(T n) const {
        return n / d_;
    }

    DEVICE_FUNCTION T Mod(T n) const {
        return n % d_;
    }

    DEVICE_FUNCTION DivModRet<T> DivMod(T n) const {
        return DivModRet<T>(n / d_, n % d_);
    }
};

template <>
class IntDivMod<uint32_t> {
 public:
    uint32_t d_;

 private:
    uint32_t magic_;
    uint32_t shift_;

 public:
    template <typename DT>
    static bool OutOfBound(const DT *d, int count) {
        for (int i = 0; i < count; ++i) {
            if (static_cast<uint64_t>(d[i]) > INT32_MAX) {
                return true;
            }
        }
        return false;
    }

    IntDivMod() {}

    explicit IntDivMod(uint32_t d) {
        // assert(d >= 1 && d <= INT32_MAX);
        d_ = d;

        for (shift_ = 0; shift_ < 32; ++shift_) {
            if ((1u << shift_) >= d) {
                break;
            }
        }
        uint64_t tmp_magic = ((1lu << 32) * ((1lu << shift_) - d)) / d + 1;
        magic_ = tmp_magic;  // copy lower 32-bit
        // assert(magic_ != 0 && magic_ == tmp_magic);
    }

    DEVICE_FUNCTION uint32_t Div(uint32_t n) const {
#ifdef __CUDA_ARCH__
        return (__umulhi(n, magic_) + n) >> shift_;
#else
        uint64_t tmp = (static_cast<uint64_t>(n) * magic_) >> 32;
        return (static_cast<uint32_t>(tmp) + n) >> shift_;
#endif
    }

    DEVICE_FUNCTION uint32_t Mod(uint32_t n) const {
        return n - Div(n) * d_;
    }

    DEVICE_FUNCTION DivModRet<uint32_t> DivMod(uint32_t n) const {
        uint32_t d = Div(n);
        return DivModRet<uint32_t>(d, n - d_ * d);
    }
};

using U32Div = IntDiv<uint32_t>;
using U32DivMod = IntDivMod<uint32_t>;

}  // namespace hiednn

#endif  // DNN_INCLUDE_INTEGER_DIVMOD_HPP_

