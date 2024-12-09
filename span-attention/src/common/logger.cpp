/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    logger.cpp
 */

#include "common/logger.h"

#include <sstream>

#include "common/error.h"

namespace span {

std::string to_string(LogLevel level) {
  switch (level) {
    case LogLevel::DEBUG:
      return "DEBUG";
    case LogLevel::ERROR:
      return "ERROR";
    default:
      return std::string("<invalid LogLevel: ") +
             std::to_string(static_cast<int>(level)) + ">";
  }
}

SpanLoggerWrapper::SpanLoggerWrapper(LogLevel level, std::ostream& os)
    : os_(os), level_(level), prefix_(to_string(level_)) {}

std::ostream& SpanLoggerWrapper::stream(const char* caller) {
  std::stringstream ss;
  ss << "[" << prefix_ << " " << caller << ":" << __LINE__ << "] ";
  return os_ << ss.str();
}

}  // namespace span

::span::SpanLoggerWrapper __span_logger_ERROR(span::LogLevel::ERROR, std::cerr);

#ifdef NDEBUG
::span::EmptyLogger __empty_logger;
::span::SpanLoggerWrapper __span_logger_DEBUG(span::LogLevel::DEBUG,
                                              __empty_logger);
#else
::span::SpanLoggerWrapper __span_logger_DEBUG(span::LogLevel::DEBUG, std::cerr);
#endif
