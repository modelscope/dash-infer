/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    allspark_service_parallel.cpp
 */

#include "allspark_service_parallel.h"

#include <glog/logging.h>
#include <omp.h>

#include <mutex>

namespace allspark {
namespace allspark_service {

class ThreadException {
 public:
  ThreadException() : ptr_(nullptr) {}
  ~ThreadException() = default;
  void ConditionRethrow() {
    if (ptr_) std::rethrow_exception(ptr_);
  }
  void CaptureException() {
    std::unique_lock<std::mutex> guard(lock_);
    ptr_ = std::current_exception();
  }
  template <typename Function, typename... Parameters>
  void Run(Function f, Parameters... params) {
    try {
      f(params...);
    } catch (...) {
      CaptureException();
    }
  }

 private:
  std::exception_ptr ptr_;
  std::mutex lock_;
};

void parallel_loop(const int begin, const int end,
                   const ParallelForBody& body) {
  ThreadException e;
#pragma omp parallel for num_threads(8)
  for (int i = begin; i < end; i++) {
    e.Run([=] { body(i); });
  }
  e.ConditionRethrow();
}
}  // namespace allspark_service
}  // namespace allspark
