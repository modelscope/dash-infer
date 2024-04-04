/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    worker_coodinator.cpp
 */
#include "worker_coodinator.h"

#define DISABLE_COODINATOR 1
namespace allspark {

std::atomic_int allspark::WorkerCoodinator::counter_{0};
std::atomic_flag allspark::WorkerCoodinator::busy_flag_{0};
std::atomic_int allspark::WorkerCoodinator::error_flag_{0};

#define DEFAULT_COODINATOR_TIMEOUT_MS 1000

void WorkerCoodinator::ResetCounter() { counter_.store(0); }

int WorkerCoodinator::GetDefaultTimeout() {
  return DEFAULT_COODINATOR_TIMEOUT_MS;
}

int WorkerCoodinator::StateSyncWithTimeout() {
#if DISABLE_COODINATOR
  return 0;
#endif

  int ret = 0;
  // special case for single card.
  if (total_ranks_ == 1) return 0;

  busy_flag_.test_and_set(std::memory_order_acquire);

  if (error_flag_.load(std::memory_order_acquire)) {
    return -1;
  }

  auto start_time = std::chrono::steady_clock::now();

  counter_.fetch_add(1, std::memory_order_acquire);

  // spinning check if counter already archived target number
  while (counter_.load(std::memory_order_acquire) < total_ranks_) {
    auto cur_time = std::chrono::steady_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                            cur_time - start_time)
                            .count();
    if (elapsed_time > timeout_ms_) {
      ret = -2;
      error_flag_.store(1, std::memory_order_acquire);
      goto out;
    }
  }

out:
  busy_flag_.clear(std::memory_order_release);
  return ret;
}

};  // namespace allspark
