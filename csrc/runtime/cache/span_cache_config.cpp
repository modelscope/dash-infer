/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    span_cache_config.cpp
 */

#if ENABLE_SPAN_ATTENTION
#include <string>

#include "common/engine_runtime.h"
#include "glog/logging.h"

namespace allspark {

SpanCacheConfig::SpanCacheConfig(AsCacheMode mode_, int span_size_,
                                 int span_num_init_, int span_num_grow_)
    : mode(mode_),
#ifdef FIXED_SPAN_SIZE
      span_size((FIXED_SPAN_SIZE)),
#else
      span_size(span_size_),
#endif  // FIXED_SPAN_SIZE
      span_num_init(span_num_init_),
      span_num_grow(span_num_grow_) {
  LOG(INFO) << "Cache config: cache mode set to " << CacheMode2String(mode_);

#ifdef FIXED_SPAN_SIZE
  LOG(WARNING) << "Cache config: ignoring input cache span size, span size set "
                  "to a fixed value: "
               << span_size;
#endif  // FIXED_SPAN_SIZE

  switch (span_size) {
    case 0:
      /* span disabled */
      break;
    case 16:
    case 32:
    case 64:
    case 128:
      /* fall through */
      LOG(INFO) << "Cache config: cache span size set to " << span_size;
      break;
    default:
      LOG(ERROR) << "Cache config: cache span size only support 16, 32, "
                    "64, 128; got "
                 << span_size;
      AS_THROW(AsStatus::ALLSPARK_PARAM_ERROR);
  }

  if (span_size == 0) {
    LOG(INFO) << "Cache config: cache span size set to 0, span cache disabled";
    // skip param check and return
    return;
  }

  if (span_num_init < 0) {
    LOG(ERROR) << "Cache config: cache span num init cannot be negative, got "
               << span_num_init;
    AS_THROW(AsStatus::ALLSPARK_PARAM_ERROR);
  }

  if (span_num_grow < 0) {
    LOG(ERROR) << "Cache config: cache span num grow cannot be negative, got "
               << span_num_grow;
    AS_THROW(AsStatus::ALLSPARK_PARAM_ERROR);
  }
}

SpanCacheConfig::Ptr SpanCacheConfig::Create(AsCacheMode mode, int span_size,
                                             int span_num_init,
                                             int span_num_grow) {
  Ptr ret;
  try {
    ret = std::shared_ptr<SpanCacheConfig>(
        new SpanCacheConfig(mode, span_size, span_num_init, span_num_grow));
  } catch (const AsException& e) {
    if (std::string(e.what()) == "ALLSPARK_PARAM_ERROR") {
      LOG(ERROR) << "Cache config: cache config param error";
      ret = nullptr;
    } else {
      LOG(ERROR) << "Cache config: failed to create cache config: " << e.what();
      ret = nullptr;
    }
  }
  return ret;
}

};  // namespace allspark
#endif
