/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    thread_pool.h
 */

#ifndef THREAD_POOL_H
#define THREAD_POOL_H

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

#include "thread_utils.h"

namespace allspark {
class ThreadPool {
 public:
  ThreadPool(size_t);
  template <class F, class... Args>
  auto enqueue(F&& f, Args&&... args)
      -> std::future<typename std::result_of<F(Args...)>::type>;
  ~ThreadPool();
  size_t hasTasksRunning() const;

 private:
  // need to keep track of threads so we can join them
  std::vector<std::thread> workers;
  // the task queue
  std::queue<std::function<void()>> tasks;

  // synchronization
  std::mutex queue_mutex;
  std::condition_variable condition;
  std::atomic<bool> stop = false;
  std::atomic<size_t> running_tasks = 0;
};

// the constructor just launches some amount of workers
inline ThreadPool::ThreadPool(size_t threads) : stop(false) {
  LOG(INFO) << "ThreadPool created with: " << threads;
  for (size_t i = 0; i < threads; ++i)
    workers.emplace_back([this, i] {
      setThreadName(i, "ASThreadPool");
      for (;;) {
        std::function<void()> task;

        {
          unique_lock_wrapper<std::mutex> lock(this->queue_mutex, "cond.wait");
          this->condition.wait(lock.unique_lock_, [this] {
            return this->stop || !this->tasks.empty();
          });
          if (this->stop && this->tasks.empty()) return;
          task = std::move(this->tasks.front());
          this->tasks.pop();
        }

        task();
        --running_tasks;
      }
    });
}

// add new work item to the pool
template <class F, class... Args>
auto ThreadPool::enqueue(F&& f, Args&&... args)
    -> std::future<typename std::result_of<F(Args...)>::type> {
  using return_type = typename std::result_of<F(Args...)>::type;

  auto task = std::make_shared<std::packaged_task<return_type()>>(
      std::bind(std::forward<F>(f), std::forward<Args>(args)...));

  std::future<return_type> res = task->get_future();
  {
    unique_lock_wrapper<std::mutex> lock(queue_mutex, "enqueue");

    // don't allow enqueueing after stopping the pool
    if (stop) {
      LOG(INFO) << "Enqueue stopped thread pool. " << this;
      print_backtrace();
      throw std::runtime_error("enqueue on stopped ThreadPool");
    }

    ++running_tasks;

    tasks.emplace([task]() { (*task)(); });
  }
  condition.notify_one();
  return res;
}

// the destructor joins all threads
inline ThreadPool::~ThreadPool() {
  {
    LOG(INFO) << "~ThreadPool called. " << this;
    std::unique_lock<std::mutex> lock(queue_mutex);
    // print_backtrace();
    stop = true;
  }
  condition.notify_all();
  for (std::thread& worker : workers) worker.join();
}

inline size_t ThreadPool::hasTasksRunning() const {
  // LOG(INFO) << "Number of running tasks: " << running_tasks;
  return running_tasks.load();
}
}  // namespace allspark

#endif
