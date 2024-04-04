/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    timer.h
 */
#pragma once

#include <chrono>
#include <string>

namespace allspark {

namespace util {
class Timer {
 public:
  Timer() : m_begin(std::chrono::high_resolution_clock::now()) {}
  Timer(std::string str) : m_begin(std::chrono::high_resolution_clock::now()) {
    this->name = str;
  }

  void reset() { m_begin = std::chrono::high_resolution_clock::now(); }

  // default using milliseconds.
  int64_t elapsed() const {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::high_resolution_clock::now() - m_begin)
        .count();
  }

  int64_t elapsed_micro(int epoch = 1) const {
    int64_t t = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::high_resolution_clock::now() - m_begin)
                    .count();
    return t;
  }

  int64_t elapsed_nano(int epoch = 1) const {
    int64_t t = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::high_resolution_clock::now() - m_begin)
                    .count();
    return t;
  }

  int64_t elapsed_seconds() const {
    return std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::high_resolution_clock::now() - m_begin)
        .count();
  }

  int64_t elapsed_minutes() const {
    return std::chrono::duration_cast<std::chrono::minutes>(
               std::chrono::high_resolution_clock::now() - m_begin)
        .count();
  }

  int64_t elapsed_hours() const {
    return std::chrono::duration_cast<std::chrono::hours>(
               std::chrono::high_resolution_clock::now() - m_begin)
        .count();
  }

 private:
  std::chrono::time_point<std::chrono::high_resolution_clock> m_begin;
  std::string name = "default_timer";
};
}  // namespace util
}  // namespace allspark
