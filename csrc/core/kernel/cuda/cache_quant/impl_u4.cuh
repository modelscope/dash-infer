/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    impl_u4.cuh
 */

#pragma once

#include <cstdint>
#include <limits>

#include "../utils/pack.cuh"
#include "prototype.cuh"
#include "reduce.cuh"
#include "utils.cuh"

namespace allspark {
namespace cuda {
namespace qcache {

struct uint4x2_t {
  using UnderlyingT = uint8_t;
  static constexpr int UNDERLYING_SIZE = 2;

  UnderlyingT raw;
  __device__ uint4x2_t() : raw(0) {}

  __device__ explicit uint4x2_t(const uint32_t (&val)[UNDERLYING_SIZE]) {
    raw = (val[0] & 0xf) | ((val[1] & 0xf) << 4);
  }

  __device__ void Extract(uint32_t (&ret)[UNDERLYING_SIZE]) const {
    ret[0] = raw & 0xf;
    ret[1] = (raw & 0xf0) >> 4;
    return;
  }
};

template <>
struct QCacheExtractor<span::QuantMode::U4> {
  __device__ void operator()(uint32_t (&ret)[uint4x2_t::UNDERLYING_SIZE],
                             uint32_t packed) const {
    uint4x2_t trunc;
    trunc.raw = packed & 0xff;
    trunc.Extract(ret);
    return;
  }
};
template <typename T>
struct QCacheConfig<span::QuantMode::U4, T> {
  using QuantT = uint4x2_t;
  using ComputeT = float;
  using UnderlyingT = typename QuantT::UnderlyingT;
  using ExtractorT = QCacheExtractor<span::QuantMode::U4>;
  static constexpr int UNDERLYING_SIZE = QuantT::UNDERLYING_SIZE;
};

template <typename T>
struct alignas(2 *
               sizeof(typename QCacheConfig<span::QuantMode::U4, T>::ComputeT))
    QuantParam<span::QuantMode::U4, T> {
  using QT = typename QCacheConfig<span::QuantMode::U4, T>::QuantT;
  using CPT = typename QCacheConfig<span::QuantMode::U4, T>::ComputeT;
  static constexpr int UNDERLYING_SIZE =
      QCacheConfig<span::QuantMode::U4, T>::UNDERLYING_SIZE;

  // members
  CPT zero;
  CPT scale;

  // static descriptors
  static constexpr CPT MAX = static_cast<CPT>(0xf);
  static constexpr CPT MIN = static_cast<CPT>(0);

  static constexpr CPT ORIGIN = MIN;
  static constexpr CPT RANGE = MAX - MIN;

  static constexpr CPT EPS = static_cast<CPT>(1e-5);

  // APIs
  __device__ __forceinline__ void Quant(QT& ret, const T* val) const {
    uint32_t words[UNDERLYING_SIZE];
#pragma unroll
    for (int i = 0; i < UNDERLYING_SIZE; ++i) {
      CPT tmp =
          static_cast<CPT>(zero + impl::Div(static_cast<CPT>(val[i]), scale));
      tmp = min(tmp, MAX);
      // uint natually >= 0
      // tmp = max(tmp, MIN);
      tmp = impl::Rounding(tmp);
      words[i] = static_cast<uint32_t>(tmp);
    }
    ret = QT(words);
    return;
  }

  __device__ __forceinline__ void Dequant(CPT* ret, const QT qVal) const {
    uint32_t words[UNDERLYING_SIZE];
    qVal.Extract(words);
#pragma unroll
    for (int i = 0; i < UNDERLYING_SIZE; ++i) {
      ret[i] = (static_cast<CPT>(words[i]) - static_cast<CPT>(zero)) *
               static_cast<CPT>(scale);
    }
    return;
  }

  // builder
  template <int BLOCK, int UNROLL, int HEADSIZE>
  class Builder {
    using QConfT = QCacheConfig<span::QuantMode::U4, T>;
    using CPT = typename QConfT::ComputeT;
    using Param = QuantParam<span::QuantMode::U4, T>;

    static_assert(BLOCK * UNROLL == HEADSIZE,
                  "QuantConfig requires BLOCK * UNROLL == HEADSIZE");
    static_assert(UNROLL % QConfT::UNDERLYING_SIZE == 0, "");
    static constexpr int QUANT_UNROLL = UNROLL / QConfT::UNDERLYING_SIZE;

    template <template <typename> typename FUNC>
    __device__ __forceinline__ CPT reduce(const CPT (&val)[UNROLL],
                                          const uint32_t& warpId,
                                          const uint32_t& laneId) const;

   public:
    // one warp handles one head
    __device__ __forceinline__ Param operator()(const T (&val)[UNROLL]) const;
  };
};

/* ================= impl ================= */

template <typename T>
template <int BLOCK, int UNROLL, int HEADSIZE>
template <template <typename> typename FUNC>
__device__ __forceinline__ typename QCacheConfig<span::QuantMode::U4,
                                                 T>::ComputeT
QuantParam<span::QuantMode::U4, T>::Builder<BLOCK, UNROLL, HEADSIZE>::reduce(
    const typename QCacheConfig<span::QuantMode::U4, T>::ComputeT (
        &val)[UNROLL],
    const uint32_t& warpId, const uint32_t& laneId) const {
  using CPT = typename QCacheConfig<span::QuantMode::U4, T>::ComputeT;

  // thread reduce
  CPT threadRes = impl::IdentityElem<FUNC, CPT>::v;
#pragma unroll
  for (int i = 0; i < UNROLL; ++i) {
    threadRes = FUNC<CPT>()(threadRes, val[i]);
  }

  // warp reduce
  CPT warpRes = impl::warpReduce<FUNC>(threadRes);

  // lane 0 holds the result
  CPT ret = ShflIdx(0xffffffff, warpRes, 0, impl::WARP_SIZE);
  return ret;
}

// one warp handles one head
template <typename T>
template <int BLOCK, int UNROLL, int HEADSIZE>
__device__ __forceinline__ QuantParam<span::QuantMode::U4, T>
QuantParam<span::QuantMode::U4, T>::Builder<
    BLOCK, UNROLL, HEADSIZE>::operator()(const T (&val)[UNROLL]) const {
  // type case with word pack
  const WordPackT<UNROLL, T>* packPtr =
      reinterpret_cast<const WordPackT<UNROLL, T>*>(val);
  CPT regs[UNROLL];
  packPtr->Unpack(regs);

  const uint32_t warpId = threadIdx.x / impl::WARP_SIZE;
  const uint32_t laneId = threadIdx.x % impl::WARP_SIZE;

  CPT maxVal = reduce<impl::functor::Max>(regs, warpId, laneId);
  CPT minVal = reduce<impl::functor::Min>(regs, warpId, laneId);

  CPT qs = impl::Div(maxVal - minVal, Param::RANGE);
  qs = max(qs, Param::EPS);
  CPT qz = Param::ORIGIN - impl::Div(minVal, qs);
  qz = min(qz, Param::MAX);
  // uint natually >= 0
  // qz = max(qz, Param::MIN);
  qz = impl::Rounding(qz);

  Param ret;
  ret.scale = qs;
  ret.zero = qz;
  return ret;
}

}  // namespace qcache
}  // namespace cuda
}  // namespace allspark
