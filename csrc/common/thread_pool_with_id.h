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

#include "thread_utils.h"
namespace allspark {

class ThreadPoolWithID {
 public:
  ThreadPoolWithID(size_t threads)
      : task_queues(threads),
        cond_vec(threads),
        mutex_vec(threads),
        stop(false) {
    for (size_t i = 0; i < threads; ++i)
      workers.emplace_back([this, i] {
        setThreadName(i, "ASThreadPool");
        for (;;) {
          std::function<void()> task;
          {
            unique_lock_wrapper<std::mutex> lock(this->mutex_vec[i],
                                                 "cond.wait");

            this->cond_vec[i].wait(lock.unique_lock_, [this, i] {
              return this->stop || !this->task_queues[i].empty();
            });

            if (this->stop && this->task_queues[i].empty()) {
              LOG(INFO) << "Thread Pool with id: " << i << " Exit!!!";
              return;
            }

            {
              // need lock?
              task = std::move(this->task_queues[i].front());

              this->task_queues[i].pop();
            }
          }
          task();
        }
      });
  }

  template <class F, class... Args>
  auto enqueue(size_t worker_id, F&& f, Args&&... args)
      -> std::future<typename std::result_of<F(Args...)>::type> {
    using return_type = typename std::result_of<F(Args...)>::type;
    if (worker_id >= this->task_queues.size())
      throw std::runtime_error("worker submit exceeds thread pool size.");

    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...));

    std::future<return_type> res = task->get_future();
    {
      if (stop) {
        throw std::runtime_error("enqueue on stopped ThreadPool");
      }
      unique_lock_wrapper<std::mutex> lock(this->mutex_vec[worker_id]);
      this->task_queues[worker_id].emplace([task]() { (*task)(); });
    }
    cond_vec[worker_id].notify_all();
    return res;
  }

  ~ThreadPoolWithID() {
    stop = true;

    for (auto& cond : cond_vec) {
      cond.notify_all();
    }

    for (std::thread& worker : workers) {
      worker.join();
    }
  }

 private:
  std::vector<std::thread> workers;
  std::vector<std::queue<std::function<void()>>> task_queues;

  std::vector<std::condition_variable> cond_vec;
  std::vector<std::mutex> mutex_vec;

  std::mutex queue_mutex;  // mutex put task into queue.
  std::atomic<bool> stop;
};

}  // namespace allspark

#endif
