/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    mutex_wrapper.h
 */

#ifndef MUTEX_WRAPPER_H
#define MUTEX_WRAPPER_H
#include <mutex>
#include <shared_mutex>

namespace allspark {
template <class T>
class lock_guard_wrapper {
 public:
  explicit lock_guard_wrapper(T& mutex, const char* p = nullptr)
      : mutex_(mutex), lock_guard_(mutex_) {}

  ~lock_guard_wrapper() {}

 private:
  T& mutex_;
  std::lock_guard<T> lock_guard_;
};

template <class T>
class shared_lock_wrapper {
 public:
  explicit shared_lock_wrapper(T& mutex, const char* p = nullptr)
      : mutex_(mutex), lock_guard_(mutex_) {}

  ~shared_lock_wrapper() {}

 private:
  T& mutex_;
  std::shared_lock<T> lock_guard_;
};

template <class T>
class unique_lock_wrapper {
 public:
  explicit unique_lock_wrapper(T& mutex, const char* p = nullptr)
      : mutex_(mutex), unique_lock_(mutex_) {}

  ~unique_lock_wrapper() {}

  T& mutex_;
  std::unique_lock<T> unique_lock_;
};
}  // namespace allspark
#endif
