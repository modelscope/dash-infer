/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    impl_default.cuh
 */

#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>

#include "common/func_modifier.h"
#include "prototype.cuh"

namespace span {

namespace qcache {

template <QuantMode MODE>
struct QCacheExtractor {
  DEVICE_FUNC void operator()(uint32_t (&ret)[1], uint32_t packed) const {
    ret[0] = packed;
  }
};

template <QuantMode MODE, typename T>
struct QCacheConfig {
  using QuantT = T;
  using ComputeT = float;
  using UnderlyingT = QuantT;
  using ExtractorT = QCacheExtractor<MODE>;
  static constexpr int UNDERLYING_SIZE = 1;
};

/**
 * @brief Cache quantization parameters.
 *
 * Use example:
 * @code{.cpp}
 *  using Param = qcache::QuantParam<MODE, T>;
 *  typename Param::Builder<BLOCK, UNROLL, HEADSIZE> builder;
 *  Param param = builder(regs);
 * @endcode
 *
 * @tparam MODE QuantMode
 * @tparam T original type
 */
template <QuantMode MODE, typename T>
struct QuantParam {
  using QT = typename QCacheConfig<MODE, T>::QuantT;
  using CPT = typename QCacheConfig<MODE, T>::ComputeT;

  /* empty struct */

  /**
   * T (&val)[DescT::UNDERLYING_SIZE]
   */
  DEVICE_FUNC void Quant(QT& ret, const T* val) const {
    ret = static_cast<QT>(*val);
    return;
  }

  /**
   * CPT (&ret)[DescT::UNDERLYING_SIZE]
   */
  DEVICE_FUNC void Dequant(CPT* ret, const QT qVal) const {
    *ret = static_cast<CPT>(qVal);
    return;
  }

  template <int BLOCK, int UNROLL, int HEADSIZE>
  class Builder {
    using Param = QuantParam<MODE, T>;

   public:
    DEVICE_FUNC Param operator()(const T (&val)[UNROLL]) const {
      return Param{};
    }
  };
};

}  // namespace qcache

}  // namespace span
