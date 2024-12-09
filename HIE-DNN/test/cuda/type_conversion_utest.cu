/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    type_conversion_utest.cu
 */
#include <hiednn.h>
#include <hiednn_cuda.h>

#include <gtest/gtest.h>
#include <cstdint>
#include <cmath>

#include <utest_utils.hpp>
#include <datatype_extension/datatype_extension.hpp>
#include <cuda/intrinsic/type_conversion.hpp>

namespace {

template <typename D, typename S>
struct RD {
    static __device__ inline D DevFunc(const S &s) {
        return hiednn::cuda::F2I_RD<D>(s);
    }

    static D HostFunc(const S &s) {
        return static_cast<D>(std::floor(static_cast<double>(s)));
    }
};

template <typename D, typename S>
struct RU {
    static __device__ inline D DevFunc(const S &s) {
        return hiednn::cuda::F2I_RU<D>(s);
    }

    static D HostFunc(const S &s) {
        return static_cast<D>(std::ceil(static_cast<double>(s)));
    }
};

template <typename D, typename S>
struct RN {
    static __device__ inline D DevFunc(const S &s) {
        return hiednn::cuda::F2I_RN<D>(s);
    }

    static D HostFunc(const S &s) {
        return static_cast<D>(std::rint(static_cast<double>(s)));
    }
};

template <typename D, typename S>
struct RZ {
    static __device__ inline D DevFunc(const S &s) {
        return hiednn::cuda::F2I_RZ<D>(s);
    }

    static D HostFunc(const S &s) {
        return static_cast<D>(std::trunc(static_cast<double>(s)));
    }
};

template <typename D, typename S, template <typename, typename> class Func>
__global__ void kernel(const S *s, D *d) {
    d[threadIdx.x] = Func<D, S>::DevFunc(s[threadIdx.x]);
}

template <typename T>
bool PassNeg(const double &x) {
    return false;
}

template <>
bool PassNeg<uint8_t>(const double &x) {
    return x < 0;
}

template <>
bool PassNeg<uint16_t>(const double &x) {
    return x < 0;
}

template <>
bool PassNeg<uint32_t>(const double &x) {
    return x < 0;
}

template <>
bool PassNeg<uint64_t>(const double &x) {
    return x < 0;
}

template <typename D, typename S, template <typename, typename> class Func>
void RunTest() {
    const int N = 7;
    S hs[N] = {0, -0.5, 0.5, -1.5, 1.5, -2.5, 2.5};
    S *s;
    D *d;
    CHECK_CUDA(cudaMalloc(&s, N * sizeof(S)));
    CHECK_CUDA(cudaMalloc(&d, N * sizeof(D)));
    CHECK_CUDA(cudaMemcpy(s, hs, N * sizeof(S), cudaMemcpyHostToDevice));

    kernel<D, S, Func><<<1, N>>>(s, d);

    D hd[N];
    CHECK_CUDA(cudaMemcpy(hd, d, N * sizeof(D), cudaMemcpyDeviceToHost));
    for (int i = 0; i < N; ++i) {
        if (PassNeg<D>(hs[i]))
            continue;
        CheckEq(hd[i], Func<D, S>::HostFunc(hs[i]));
    }

    CHECK_CUDA(cudaFree(s));
    CHECK_CUDA(cudaFree(d));
}

}  // anonymous namespace

#define UTEST_TYPE_CONVERSION(TEST_NAME, TYPE, RND_MODE) \
TEST(TypeConversion, TEST_NAME) { \
    RunTest<int8_t, TYPE, RND_MODE>(); \
    RunTest<uint8_t, TYPE, RND_MODE>(); \
    RunTest<int16_t, TYPE, RND_MODE>(); \
    RunTest<uint16_t, TYPE, RND_MODE>(); \
    RunTest<int32_t, TYPE, RND_MODE>(); \
    RunTest<uint32_t, TYPE, RND_MODE>(); \
    RunTest<int64_t, TYPE, RND_MODE>(); \
    RunTest<uint64_t, TYPE, RND_MODE>(); \
}

#ifdef HIEDNN_USE_FP16
UTEST_TYPE_CONVERSION(Half_RD, hiednn::half, RD)
UTEST_TYPE_CONVERSION(Half_RU, hiednn::half, RU)
UTEST_TYPE_CONVERSION(Half_RN, hiednn::half, RN)
UTEST_TYPE_CONVERSION(Half_RZ, hiednn::half, RZ)
#endif

#ifdef HIEDNN_USE_BF16
UTEST_TYPE_CONVERSION(Bfloat16_RD, hiednn::bfloat16, RD)
UTEST_TYPE_CONVERSION(Bfloat16_RU, hiednn::bfloat16, RU)
UTEST_TYPE_CONVERSION(Bfloat16_RN, hiednn::bfloat16, RN)
UTEST_TYPE_CONVERSION(Bfloat16_RZ, hiednn::bfloat16, RZ)
#endif

UTEST_TYPE_CONVERSION(Float_RD, float, RD)
UTEST_TYPE_CONVERSION(Float_RU, float, RU)
UTEST_TYPE_CONVERSION(Float_RN, float, RN)
UTEST_TYPE_CONVERSION(Float_RZ, float, RZ)

UTEST_TYPE_CONVERSION(Double_RD, double, RD)
UTEST_TYPE_CONVERSION(Double_RU, double, RU)
UTEST_TYPE_CONVERSION(Double_RN, double, RN)
UTEST_TYPE_CONVERSION(Double_RZ, double, RZ)


