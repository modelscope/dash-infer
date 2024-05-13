/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    result_queue.cpp
 */
#include <glog/logging.h>

#include "engine_runtime.h"

namespace allspark {

ResultQueueImpl::ResultQueueImpl() {}

ResultQueueImpl::~ResultQueueImpl() {
  queue_write_lock_t write_lock(queue_mutex_);
  closed_ = true;
  write_lock.unlock();
  cond_var_.notify_all();
}

AsEngine::GenerateRequestStatus ResultQueueImpl::GenerateStatus() {
  queue_read_lock_t read_lock(queue_mutex_);
  return status_;
}

size_t ResultQueueImpl::GeneratedLength() { return generate_length_; }

// wait for new data or new status.
ResultQueueImpl::GenerateElementPtr ResultQueueImpl::Get() {
  DLOG(INFO) << "Get Start, store_queue.size: " << store_queue_.size()
             << ", status: " << static_cast<int>(status_);
  // wait should be wait for new data, needs to add a cond var to wait
  queue_read_lock_t read_lock(queue_mutex_);
  if (status_ == AsEngine::GenerateRequestStatus::GenerateInterrupted) {
    return nullptr;
  }

  ResultQueueImpl::GenerateElementPtr ret;
  if (store_queue_.size() > 0)
    ret = GetNoLock();
  else {
    auto old_status = status_;
    cond_var_.wait(read_lock, [this, &old_status]() {
      DLOG(INFO) << "Get cv wake up, old_status: "
                 << static_cast<int>(old_status)
                 << ", status_: " << static_cast<int>(status_)
                 << ", closed_: " << closed_
                 << ", store_queue.size: " << store_queue_.size();
      // return true if want to break wait.
      // if generating finished, either finsihed, or interrupted, return to
      // user.
      return (store_queue_.size() > 0 ||
              (old_status != status_ &&
               status_ != AsEngine::GenerateRequestStatus::Generating) ||
              closed_ ||
              status_ == AsEngine::GenerateRequestStatus::GenerateFinished ||
              status_ == AsEngine::GenerateRequestStatus::GenerateInterrupted);
    });

    // check once more
    if (status_ == AsEngine::GenerateRequestStatus::GenerateInterrupted) {
      return nullptr;
    }
    ret = GetNoLock();
  }

  DLOG(INFO) << "Get End, store_queue.size: " << store_queue_.size()
             << ", status: " << static_cast<int>(status_)
             << ", ptr: " << ret.get();
  return ret;
}

ResultQueueImpl::GenerateElementPtr ResultQueueImpl::GetNoWait() {
  queue_read_lock_t read_lock(queue_mutex_);
  return GetNoLock();
}

ResultQueueImpl::GenerateElementPtr ResultQueueImpl::GetNoLock() {
  // todo: how to define this function?
  if (store_queue_.size() > 0) {
    auto ret = store_queue_.front();
    store_queue_.pop();
    return ret;
  } else {
    return nullptr;
  }
}

void ResultQueueImpl::SetStatus(AsEngine::GenerateRequestStatus new_status) {
  DLOG(INFO) << "SetStatus Start, new_status: " << static_cast<int>(new_status)
             << ", old_status: " << static_cast<int>(status_);

  queue_write_lock_t write_lock(queue_mutex_);
  status_ = new_status;

  write_lock.unlock();
  cond_var_.notify_all();
}

void ResultQueueImpl::AppendGenerateData(std::vector<int64_t>&& new_tokens) {
  DLOG(INFO) << "AppendGenerateData Start, new_tokens.size: "
             << new_tokens.size()
             << ", store_queue.size: " << store_queue_.size()
             << ", status: " << static_cast<int>(status_);

  queue_write_lock_t write_lock(queue_mutex_);
  generate_length_ += new_tokens.size();
  if (store_queue_.size() > 0) {
    auto end_iter = store_queue_.front()->ids_from_generate.end();

    if (new_tokens.size() > 0) {
      store_queue_.front()->ids_from_generate.insert(
          end_iter, new_tokens.begin(), new_tokens.end());
    }
  } else {
    auto new_ele = std::make_shared<AsEngine::GeneratedElements>();
    new_ele->ids_from_generate = std::move(new_tokens);
    DLOG(INFO) << "new ele: " << new_ele->ids_from_generate.size()
               << ", ptr: " << new_ele.get();
    store_queue_.push(new_ele);
  }
  write_lock.unlock();

  cond_var_.notify_all();
}

void ResultQueueImpl::AppendGenerateElement(
    std::shared_ptr<AsEngine::GeneratedElements> new_ele) {
  std::vector<int64_t> new_tokens = new_ele->ids_from_generate;
  std::vector<std::vector<std::pair<int64_t, float>>> new_prob =
      new_ele->log_probs_list;
  std::vector<float> new_token_logprobs = new_ele->token_logprobs_list;
  DLOG(INFO) << "AppendGenerateElement, new_tokens.size: " << new_tokens.size()
             << ", store_queue.size: " << store_queue_.size()
             << ", status: " << static_cast<int>(status_);

  queue_write_lock_t write_lock(queue_mutex_);
  generate_length_ += new_tokens.size();
  if (store_queue_.size() > 0) {
    auto queue_front = store_queue_.front();
    if (new_tokens.size() > 0) {
      store_queue_.front()->ids_from_generate.insert(
          queue_front->ids_from_generate.end(), new_tokens.begin(),
          new_tokens.end());
      store_queue_.front()->log_probs_list.insert(
          queue_front->log_probs_list.end(), new_prob.begin(), new_prob.end());
      store_queue_.front()->token_logprobs_list.insert(
          queue_front->token_logprobs_list.end(), new_token_logprobs.begin(),
          new_token_logprobs.end());
    }
    // LOG(INFO) << "append ele: "  << new_tokens.size()  << "ptr : " <<
    // store_queue_.front().get();
  } else {
    // auto new_ele = std::make_shared<AsEngine::GeneratedElements>();
    // new_ele->ids_from_generate = std::move(new_tokens);
    DLOG(INFO) << "new ele: " << new_ele->ids_from_generate.size()
               << ", ptr: " << new_ele.get();
    store_queue_.push(new_ele);
  }
  write_lock.unlock();

  cond_var_.notify_all();
}

}  // namespace allspark
