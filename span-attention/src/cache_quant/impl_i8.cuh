/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    impl_i8.cuh
 */

#pragma once

#include <cstdint>
#include <limits>

#include "prototype.cuh"
#include "reduce.cuh"
#include "utils.cuh"
#include "utils/pack.cuh"

namespace span {

namespace qcache {

template <typename T>
struct QCacheConfig<QuantMode::I8, T> {
  using QuantT = int8_t;
  using ComputeT = float;
  using UnderlyingT = QuantT;
  using ExtractorT = QCacheExtractor<QuantMode::I8>;
  static constexpr int UNDERLYING_SIZE = 1;
};

template <typename T>
struct alignas(2 * sizeof(typename QCacheConfig<QuantMode::I8, T>::ComputeT))
    QuantParam<QuantMode::I8, T> {
  using QT = typename QCacheConfig<QuantMode::I8, T>::QuantT;
  using CPT = typename QCacheConfig<QuantMode::I8, T>::ComputeT;

  // members
  CPT zero;
  CPT scale;

  // static descriptors
  static constexpr CPT ORIGIN =
      static_cast<CPT>(std::numeric_limits<QT>::lowest());
  static constexpr CPT RANGE =
      static_cast<CPT>(std::numeric_limits<QT>::max()) -
      static_cast<CPT>(std::numeric_limits<QT>::lowest());

  static constexpr CPT MAX = static_cast<CPT>(std::numeric_limits<QT>::max());
  static constexpr CPT MIN =
      static_cast<CPT>(std::numeric_limits<QT>::lowest());

  static constexpr CPT EPS = static_cast<CPT>(1e-5);

  // APIs
  DEVICE_FUNC void Quant(QT& ret, const T* val) const {
    CPT tmp = static_cast<CPT>(zero + impl::Div(static_cast<CPT>(*val), scale));
    tmp = min(tmp, MAX);
    tmp = max(tmp, MIN);
    tmp = impl::Rounding(tmp);
    ret = static_cast<QT>(tmp);
    return;
  }

  DEVICE_FUNC void Dequant(CPT* ret, const QT qVal) const {
    *ret = (static_cast<CPT>(qVal) - static_cast<CPT>(zero)) *
           static_cast<CPT>(scale);
    return;
  }

  // builder
  template <int BLOCK, int UNROLL, int HEADSIZE>
  class Builder {
    using CPT = typename QCacheConfig<QuantMode::I8, T>::ComputeT;
    using Param = QuantParam<QuantMode::I8, T>;

    static_assert(BLOCK * UNROLL == HEADSIZE,
                  "QuantConfig requires BLOCK * UNROLL == HEADSIZE");

    template <template <typename> typename FUNC>
    DEVICE_FUNC CPT reduce(const CPT (&val)[UNROLL], const uint32_t& warpId,
                           const uint32_t& laneId) const;

   public:
    // one warp handles one head
    DEVICE_FUNC Param operator()(const T (&val)[UNROLL]) const;
  };
};

/* ================= impl ================= */

template <typename T>
template <int BLOCK, int UNROLL, int HEADSIZE>
template <template <typename> typename FUNC>
DEVICE_FUNC typename QCacheConfig<QuantMode::I8, T>::ComputeT
QuantParam<QuantMode::I8, T>::Builder<BLOCK, UNROLL, HEADSIZE>::reduce(
    const typename QCacheConfig<QuantMode::I8, T>::ComputeT (&val)[UNROLL],
    const uint32_t& warpId, const uint32_t& laneId) const {
  using CPT = typename QCacheConfig<QuantMode::I8, T>::ComputeT;

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
DEVICE_FUNC QuantParam<QuantMode::I8, T>
QuantParam<QuantMode::I8, T>::Builder<BLOCK, UNROLL, HEADSIZE>::operator()(
    const T (&val)[UNROLL]) const {
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
  qz = max(qz, Param::MIN);
  qz = impl::Rounding(qz);

  Param ret;
  ret.scale = qs;
  ret.zero = qz;
  return ret;
}

}  // namespace qcache

}  // namespace span
