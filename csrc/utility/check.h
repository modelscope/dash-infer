/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    check.h
 */

#pragma once

#include <execinfo.h>
#include <glog/logging.h>
#include <interface/allspark_check.h>

namespace allspark {

static void print_backtrace() {
  const int backtrace_max_size = 512;
  void* buffer[backtrace_max_size];

  int buffer_size = backtrace(buffer, backtrace_max_size);
  char** symbols = backtrace_symbols(buffer, buffer_size);
  if (symbols) {
    LOG(INFO) << "==================== BackTrace ===================";
    for (int i = 0; i < buffer_size; ++i) {
      LOG(INFO) << buffer[i] << symbols[i];
    }
    free(symbols);
  }
}

inline void ConcatStringInternal(std::stringstream& ss) {}

template <typename T,
          typename std::enable_if<!std::is_enum<
              typename std::remove_reference<T>::type>::value>::type* = nullptr>
void ConcatStringInternal(std::stringstream& ss, T&& t) {
  ss << std::forward<T>(t);
}

template <typename T,
          typename std::enable_if<std::is_enum<
              typename std::remove_reference<T>::type>::value>::type* = nullptr>
void ConcatStringInternal(std::stringstream& ss, T&& t) {
  ss << to_string(t);
}

template <typename T, typename... Args>
void ConcatStringInternal(std::stringstream& ss, T&& t, Args&&... args) {
  ConcatStringInternal(ss, std::forward<T>(t));
  ConcatStringInternal(ss, std::forward<Args>(args)...);
}

template <typename... Args>
std::string ConcatString(Args&&... args) {
  std::stringstream ss;
  ConcatStringInternal(ss, std::forward<Args>(args)...);
  return std::string(ss.str());
}

#define AS_ENFORCE(condition, ...)                          \
  do {                                                      \
    if (!(condition)) {                                     \
      LOG(FATAL) << __FILE__ << ":" << __LINE__             \
                 << ", AllSpark ENFORCE FAILED , "          \
                 << ConcatString(__VA_ARGS__) << std::endl; \
      throw AsException(ConcatString(__VA_ARGS__));         \
    }                                                       \
  } while (false)

#define AS_CHECK_RETVAL(condition, retval, message) \
  do {                                              \
    if (!(condition)) {                             \
      LOG(ERROR) << message << std::endl;           \
      return retval;                                \
    }                                               \
  } while (0)

#define AS_STATUS_OK(status)               \
  (status == AsStatus::ALLSPARK_SUCCESS or \
   status == AsStatus::ALLSPARK_STREAMING)

#define AS_CHECK_STATUS(status)         \
  do {                                  \
    AsStatus err_status = status;       \
    if (not AS_STATUS_OK(err_status)) { \
      LOG(ERROR) << "AllSpark AS_CHECK_STATUS FAILED"; \
      return err_status;                \
    }                                   \
  } while (0)

#define AS_CHECK_STATUS_DO(status, cmd) \
  do {                                  \
    AsStatus err_status = status;       \
    if (not AS_STATUS_OK(err_status)) { \
      cmd;                              \
      return err_status;                \
    }                                   \
  } while (0)

#define AS_CHECK_EXCEPTION(expr)                                \
  do {                                                          \
    try {                                                       \
      expr;                                                     \
    } catch (std::exception & e) {                              \
      LOG(ERROR) << "Failed:  " << __FILE__ << ":" << __LINE__; \
      throw e;                                                  \
    }                                                           \
  } while (0)

}  // namespace allspark
