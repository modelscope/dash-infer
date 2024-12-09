/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    prototype.cuh
 */

#pragma once

#include <span_attn.h>

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace span {

namespace qcache {

template <QuantMode MODE>
struct QCacheExtractor;

template <QuantMode MODE, typename T>
struct QCacheConfig;

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
struct QuantParam;

}  // namespace qcache

}  // namespace span
