/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    prototype.cuh
 */

#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>

#include "cuda/cuda_kernel.h"
#include "cuda/cuda_kernel_span_cache.h"

namespace allspark {
namespace cuda {
namespace qcache {

template <span::QuantMode MODE>
struct QCacheExtractor;

template <span::QuantMode MODE, typename T>
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
 * @tparam MODE span::QuantMode
 * @tparam T original type
 */
template <span::QuantMode MODE, typename T>
struct QuantParam;

}  // namespace qcache
}  // namespace cuda
}  // namespace allspark
