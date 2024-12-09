/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    worker_coodinator.h
 */

#ifndef WORKER_COODINATOR_H
#define WORKER_COODINATOR_H

#include <atomic>
#include <iostream>
#include <thread>

namespace allspark {

/**
 *  this class can coodinator multiple thread work flow, sync the state among
 * multiple threads, if all thread are ready, enter some function that requires
 * sync between multiple thread, like nccl allgather, allreduce, etc, if some
 * thread is lost
 *
 *  eg:
 *  <code>
 *  auto co = WorkCoodinator(8, n, 2000);
 *  if (co.StateSyncWithTimeout() != 0)
 *     return ERROR;
 *  nccl_allreduce();
 *  // or other staff.
 *  co.ResetCounter();
 *  </code>
 */

class WorkerCoodinator {
 public:
  explicit WorkerCoodinator(int total_rank, int rank_id, int timeout_ms)
      : total_ranks_(total_rank),
        cur_rank_id_(rank_id),
        timeout_ms_(timeout_ms) {}

  /**
   * sync multiple card state with the timeout, if success wait co-workers,
   * return with success, otherwise return with failture.
   *
   * @return 0 means success, otherwise means timeout wait for others.
   */
  int StateSyncWithTimeout();

  /**
   * get default timeout
   * */
  static int GetDefaultTimeout();

  /**
   *  reset the counter, call this function after finish the nccl like task.
   *
   */
  void ResetCounter();

 private:
  int total_ranks_;
  int cur_rank_id_;
  int timeout_ms_;

  static std::atomic_int counter_;
  // set flag as busy if enter wait phase,
  // clear it when timeout or success.
  // why need this flag ?
  //
  // if flag is busy, means some one already enter this state
  static std::atomic_flag busy_flag_;

  // mark this flag to true, if any timeout happends,
  // check this flag before start spinned lock,
  // if true, means you arrived too late, just return error.
  static std::atomic_int error_flag_;
};

};  // namespace allspark

#endif
