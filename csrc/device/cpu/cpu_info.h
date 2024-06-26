/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    cpu_info.h
 */

#pragma once

#if defined(__x86_64__) || defined(_M_X64)
#include <cpuid.h>
#endif

namespace allspark {
class CPUInfo final {
 public:
  static bool SupportAVX512F() {
#if defined(__x86_64__) || defined(_M_X64)
    int cpu_info[4];
    __cpuid_count(7, 0, cpu_info[0], cpu_info[1], cpu_info[2], cpu_info[3]);
    return (cpu_info[1] & (1 << 16)) != 0;
#else
    return false;
#endif
  }
};

}  // namespace allspark
