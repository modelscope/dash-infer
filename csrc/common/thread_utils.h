/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    thread_utils.h
 */

#ifndef HIE_ALLSPARK_CSRC_COMMON_THREAD_UTILS_H_
#define HIE_ALLSPARK_CSRC_COMMON_THREAD_UTILS_H_

#include <iostream>
#include <sstream>
#include <thread>

#if defined(__linux__) || defined(__APPLE__) && defined(__MACH__)
#include <pthread.h>

static inline void setThreadName(int id, const std::string& baseName) {
  std::ostringstream threadName;
  threadName << baseName << ":" << id;
  auto handle = pthread_self();
  #if 0
  pthread_setname_np(handle, threadName.str().c_str());
  #else
  pthread_setname_np(threadName.str().c_str());
  #endif
}
#elif defined(_WIN32)

// TODO
static inline void setThreadName(int id, const std::string& baseName) {}
#else
#error "Not support platform"
#endif

#endif  // HIE_ALLSPARK_CSRC_COMMON_THREAD_UTILS_H_
