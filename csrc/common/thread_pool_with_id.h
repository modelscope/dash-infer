/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    thread_pool_with_id.h
 */

#ifndef THREAD_POOL_ID_H
#define THREAD_POOL_ID_H

#include <check.h>
#include <mutex_wrapper.h>
#include <pthread.h>

#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <vector>

#include "../utility/blockingconcurrentqueue.h"
#include "thread_utils.h"

// #define THREAD_POOL_FEATURE_NO_BLOCK

namespace allspark {

// spin lock version for faster execute.
class ThreadPoolWithID {
  constexpr static bool feature_use_spin_queue = true;

 public:
  ThreadPoolWithID(size_t threads) : queues(threads), stop(false) {
    LOG(INFO) << "ThreadPoolWithID init with thread number: " << threads;

    for (size_t i = 0; i < threads; ++i)

      workers.emplace_back([this, i] {
        setThreadName(i, "ASWorker");
        for (;;) {
          std::function<void()> task;

          if (this->stop.load()) {
            LOG(INFO) << "Thread Pool with id: " << i << " Exit!!!";
            return;
          }

#ifdef THREAD_POOL_FEATURE_NO_BLOCK
          bool have_new_value = this->queues[i].try_dequeue(task);
          if (!have_new_value)
            continue;
          else {
            task();
          }

#else
          this->queues[i].wait_dequeue(task);

          task();
#endif
        }
      });
  }

  template <class F, class... Args>
  auto enqueue(size_t worker_id, F&& f, Args&&... args)
      -> std::future<typename std::result_of<F(Args...)>::type> {
    using return_type = typename std::result_of<F(Args...)>::type;

    if (worker_id >= this->queues.size())
      throw std::runtime_error("worker submit exceeds thread pool size.");

    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...));

    std::future<return_type> res = task->get_future();
    {
      if (stop) {
        throw std::runtime_error("enqueue on stopped ThreadPool");
      }
      this->queues[worker_id].enqueue([task]() { (*task)(); });
    }
    return res;
  }

  ~ThreadPoolWithID() {
    stop.store(true);

    for (auto& q : queues) {
      q.enqueue([]() { LOG(INFO) << "dummy message for wake up."; });
    }

    for (std::thread& worker : workers) {
      worker.join();
    }
  }

 private:
#ifdef THREAD_POOL_FEATURE_NO_BLOCK
  std::vector<moodycamel::ConcurrentQueue<std::function<void()>>> queues;
#else
  std::vector<moodycamel::BlockingConcurrentQueue<std::function<void()>>>
      queues;
#endif

  std::vector<std::thread> workers;
  std::atomic<bool> stop;
};

}  // namespace allspark

#endif
