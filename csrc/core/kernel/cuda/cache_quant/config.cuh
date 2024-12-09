/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    config.cuh
 */

#pragma once

/**
 * @brief If defined, cache quantization will use round-to-nearest-int, i.e.,
 * rintf in CUDA. Otherwise, roundf will be used. RNI generates less
 * instructions, and is the recommended behavior.
 */
#define CONFIG_CACHE_ROUND_RNI
