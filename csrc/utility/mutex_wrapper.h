/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    mutex_wrapper.h
 */

#ifndef MUTEX_WRAPPER_H
#define MUTEX_WRAPPER_H
#include <mutex>
#include <shared_mutex>

namespace allspark {

#ifdef CONFIG_LOCK_DEBUG
template <class T>
class lock_guard_wrapper {
 public:
  explicit lock_guard_wrapper(T& mutex, std::string name = "<unknwon>")
      : mutex_(mutex), lock_guard_(mutex_), name_(name) {
    LOG(INFO) << "Locking the mutex. " << (void*)&mutex_ << " " << name_
              << std::endl;
  }

  ~lock_guard_wrapper() {
    LOG(INFO) << "Unlock the mutex. " << (void*)&mutex_ << " " << name_
              << std::endl;
  }

 private:
  T& mutex_;
  std::lock_guard<T> lock_guard_;
  std::string name_;
};

template <class T>
class unique_lock_wrapper {
 public:
  explicit unique_lock_wrapper(T& mutex, std::string name = "<unknwon>")
      : mutex_(mutex), unique_lock_(mutex_), name_(name) {
    LOG(INFO) << "Locking the mutex. " << (void*)&mutex_ << " " << name_
              << std::endl;
  }

  ~unique_lock_wrapper() {
    LOG(INFO) << "Unlock the mutex. " << (void*)&mutex_ << " " << name_
              << std::endl;
  }

  T& mutex_;
  std::unique_lock<T> unique_lock_;
  std::string name_;
};

template <class T>
class shared_lock_wrapper {
 public:
  explicit shared_lock_wrapper(T& mutex, std::string name = "<unknwon>")
      : mutex_(mutex), shared_lock_(mutex_), name_(name) {
    LOG(INFO) << "Shared Locking the mutex. " << (void*)&mutex_ << " " << name_
              << std::endl;
  }

  ~shared_lock_wrapper() {
    LOG(INFO) << "Shared Unlock the mutex. " << (void*)&mutex_ << " " << name_
              << std::endl;
  }

  T& mutex_;
  std::unique_lock<T> shared_lock_;
  std::string name_;
};

#else
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
}
#endif

#endif
