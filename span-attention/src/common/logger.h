/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    logger.h
 */

#pragma once

#include <iostream>

/**
 * @brief  Logging macro.
 *
 * Supported options for severity: DEBUG, ERROR
 *
 * Example:
 *
 * @code LOG(ERROR) << "This is an error message"; @endcode
 */
#define LOG(severity) __span_logger_##severity.stream(__FUNCTION__)

namespace span {

enum class LogLevel : int {
  DEBUG,
  ERROR,
};

std::string to_string(LogLevel level);

class SpanLoggerWrapper {
  std::ostream& os_;

  LogLevel level_;
  std::string prefix_;

 public:
  SpanLoggerWrapper(LogLevel level, std::ostream& os);
  virtual ~SpanLoggerWrapper() = default;

  std::ostream& stream(const char* caller);
};

class EmptyLogger : public std::ostream {
  template <typename T>
  EmptyLogger& operator<<(const T&) {
    return *this;
  }

  EmptyLogger& operator<<(std::ostream& (*f)(std::ostream&)) {
    // f(std::cout);
    return *this;
  }
};

}  // namespace span

extern ::span::SpanLoggerWrapper __span_logger_ERROR;
extern ::span::SpanLoggerWrapper __span_logger_DEBUG;
