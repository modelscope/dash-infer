/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    result_queue.cpp
 */

#include <glog/logging.h>

#include "engine_runtime.h"
#ifdef ENABLE_CUDA
#include <cuda/cuda_context.h>
#endif

// #define DEBUG_RESULT_QUEUE
namespace allspark {

// 函数计算在给定的时间段内可以尝试获取并立刻释放自旋锁多少次
unsigned long ResultQueueImpl::CountSpinCounts(
    std::chrono::milliseconds duration) {
  std::atomic_flag spinlock = ATOMIC_FLAG_INIT;
  unsigned long spin_attempts = 0;

  auto start = std::chrono::high_resolution_clock::now();
  AsEngine::GeneratedElements ret;
  // 持续尝试获取自旋锁，直到超过测试时长
  while (std::chrono::high_resolution_clock::now() - start < duration) {
    store_queue_.try_dequeue(ret);
    spin_attempts++;
  }

  return spin_attempts;
}

ResultQueueImpl::ResultQueueImpl(const std::string& uuid)
    : store_queue_(1024), request_uuid_(uuid) {
  static unsigned long cached_spin_count = 0;
  bool use_cache = false;

  status_.store(AsEngine::GenerateRequestStatus::Init);

  if (cached_spin_count == 0) {
    auto duration = std::chrono::milliseconds(1);
    // only spin for specified time.
    spin_lock_max_counts_ = CountSpinCounts(duration);
    spin_lock_max_counts_ *= 50;
    cached_spin_count = spin_lock_max_counts_;
  } else {
    spin_lock_max_counts_ = cached_spin_count;
    use_cache = true;
  }

  this->InitStatInfo();

  LOG(INFO) << "ResultQueue, max spin count: " << spin_lock_max_counts_
            << " is cached number: " << use_cache;
}

ResultQueueImpl::~ResultQueueImpl() {
  closed_.store(true);
  auto start = std::chrono::high_resolution_clock::now();
  auto timeout_duration = std::chrono::milliseconds(1 * 1000);

  // only wait for 1 seconds for get block get the close, and make sure blocking
  // user can safely return.
  while (get_user > 0 && (std::chrono::high_resolution_clock::now() - start <
                          timeout_duration)) {
    std::this_thread::yield();
  }
  if (get_user > 0) {
    LOG(ERROR) << " request queue destroy with " << get_user
               << " user still waitting on queue, may cause crash.";
  }
}

void ResultQueueImpl::InitStatInfo(void) {
  std::unique_lock<std::mutex> locker(this->queue_lock_);
  // add a lock to make sure the key's thread safe.

  request_stat_[stat_key::KEY_REQUEST_TIME_TS] = 0;
  request_stat_[stat_key::KEY_FIRST_TOKEN_TIME_TS] = 0;
  request_stat_[stat_key::KEY_LAST_TOKEN_TIME_TS] = 0;

  request_stat_[stat_key::KEY_TTFT_MS] = 0;
  request_stat_[stat_key::KEY_TOTAL_TIME_MS] = 0;
  request_stat_[stat_key::KEY_SCHEDULE_TIME_MS] = 0;

  request_stat_[stat_key::KEY_INPUT_CACHE_LEN] = 0;
  request_stat_[stat_key::KEY_INPUT_LEN] = 0;
  request_stat_[stat_key::KEY_OUTPUT_LEN] = 0;

  request_stat_[stat_key::KEY_CONTEXT_TPS] = 0;
  request_stat_[stat_key::KEY_GENERATE_TPS] = 0;
}

ResultQueueImpl::StatInfo ResultQueueImpl::RequestStatInfo() {
  StatInfo ret;
  for (auto iter = request_stat_.begin(); iter != request_stat_.end(); ++iter) {
    ret[iter->first] = iter->second.load();
  }

  // do the tps calulation.
  auto ttft_ms = ret[stat_key::KEY_TTFT_MS];
  if (ttft_ms > 0) {
    auto queue_ms = ret[stat_key::KEY_SCHEDULE_TIME_MS];
    auto ctx_ms = std::max(0L, ttft_ms - queue_ms);
    ret[stat_key::KEY_CONTEXT_TPS] =
        (ret[stat_key::KEY_INPUT_LEN] / (float)ctx_ms) * 1000.0f;
  }

  auto gen_ms = ret[stat_key::KEY_TOTAL_TIME_MS] - ret[stat_key::KEY_TTFT_MS];
  if (gen_ms > 0) {
    ret[stat_key::KEY_GENERATE_TPS] =
        (ret[stat_key::KEY_OUTPUT_LEN] / (float)gen_ms) * 1000.0f;
  }

  return ret;
}

size_t ResultQueueImpl::GeneratedLength() { return generate_length_; }

ResultQueueImpl::GenerateElementPtr ResultQueueImpl::Get() {
  ResultQueueImpl::GenerateElementPtr ret;
#ifdef DEBUG_RESULT_QUEUE
  LOG(INFO) << "Get: " << request_uuid_ << " Begin ";
#endif

  ret = this->GetWithTimeout(-1);

#ifdef DEBUG_RESULT_QUEUE
  LOG(INFO) << "Get: " << request_uuid_ << " End ";
#endif

  return ret;
}

static
void drainAllElements(moodycamel::ConcurrentQueue<AsEngine::GeneratedElements> &queue,
                      ResultQueueImpl::GenerateElementPtr output) {

  bool have_ele = false;

  while (true) {
    AsEngine::GeneratedElements one_new_token;
    have_ele = queue.try_dequeue(one_new_token);
    if (!have_ele)
      break;
    output->AppendNewSingleElementToEnd(one_new_token);
  }


}

// wait for new data or new status.
ResultQueueImpl::GenerateElementPtr ResultQueueImpl::GetWithTimeout(
    int timeout_ms) {
#ifdef DEBUG_RESULT_QUEUE
  DLOG(INFO) << "Get Start " << request_uuid_
             << ", store_queue.size: " << store_queue_.size()
             << ", status: " << static_cast<int>(status_.load());
#endif
  auto start = std::chrono::high_resolution_clock::now();
  auto warn_timeout_duration = std::chrono::milliseconds(10 * 1000);

  // if timeout duration no new token, print warning log.
  ResultQueueImpl::GenerateElementPtr total_elements =
      std::make_shared<AsEngine::GeneratedElements>();
  unsigned long spin_count = 0;
  get_user++;
  // fetch all generated token even interrupted or finished.
  while (true) {
    if (closed_.load()) break;

    AsEngine::GeneratedElements one_new_token;
    // if have elements, always return the element.
    // if no element in queue, check status, and return it.
    // if status is Finished & interrupt & and no new elements, return nullptr.
    bool have_ele = store_queue_.try_dequeue(one_new_token);

    // DLOG(INFO) << " try dequeue return " << have_ele << " status: " <<
    // ToString(status_.load()) << " spin cnt: " << spin_count;
    if (have_ele) {
#ifdef DEBUG_RESULT_QUEUE
      DLOG(INFO) << " ResultQueue: [Get] get new token: "
                 << one_new_token.ids_from_generate[0];
#endif

      // take this token and all other tokens.
      if (total_elements) {
        total_elements->AppendNewSingleElementToEnd(one_new_token);
      }
      drainAllElements(store_queue_, total_elements);
    } else {

      // no new token.
      if (status_ == AsEngine::GenerateRequestStatus::GenerateFinished ||
          status_ == AsEngine::GenerateRequestStatus::GenerateInterrupted ||
          status_ == AsEngine::GenerateRequestStatus::InternalError) {
        // return nullptr, or throw exception?
        get_user--;

        // if state changed, we should drain the queue in case some missing token.
        // since user may not check the queue again.
        // sometimes lockless queue don't return all write in one fetch.
        drainAllElements(store_queue_, total_elements);
        return total_elements;
      } else {

        // if in generating state, return token now.
        if (total_elements && total_elements->ids_from_generate.size() > 0) {
          get_user--;
          return total_elements;
      }

        spin_count++;

        auto wait_duration = std::chrono::high_resolution_clock::now() - start;

        // check timeout, if longer than timeout, return element so far.
        if (timeout_ms > 0) {
          auto exit_timeout = std::chrono::milliseconds(timeout_ms);
          if (wait_duration >= exit_timeout) {
            LOG(INFO) << "Result queue: " << request_uuid_
                      << " timeout, timeout(ms): " << timeout_ms;
            get_user--;
            return nullptr;
          }
        }

        if (wait_duration >= warn_timeout_duration) {
#ifdef ENABLE_CUDA
          // TBD: tmp way to use nvml to check if gpus are hanging
          auto should_abort =
              allspark::NVMLManager::CheckConsistentZeroUtilization();
          LOG(INFO) << "ResultQueue: " << request_uuid_
                    << " CheckConsistentZeroUtilization should_abort:"
                    << should_abort;
          if (should_abort) {
            LOG(ERROR) << "Aborting due to GPU utilization issue";
            abort();
          }
#endif
          auto seconds =
              std::chrono::duration_cast<std::chrono::seconds>(wait_duration);
          // print some log for debug.
          LOG(INFO) << "ResultQueue: " << request_uuid_ << " wait for "
                    << seconds.count() << " seconds spin count: " << spin_count

                    << " no token generate, status: " << (int)status_.load();
          start = std::chrono::high_resolution_clock::now();
          if (closed_.load()) {
            get_user--;
            return total_elements;
          }
        }
        if (spin_count >= spin_lock_max_counts_) {
          // already spinning for enough time, yield to save some cpu usage.
          usleep(500 * 1000);
#ifdef DEBUG_RESULT_QUEUE
          DLOG(INFO) << " spin count " << spin_count
                     << " max: " << spin_lock_max_counts_ << " yield";
#endif
        } else {
          // loop to beginning do the spin check.
          usleep(20 * 1000);
          continue;
        }
      }
    }
  }
  // should not reach here.
  get_user--;
  return total_elements;
}

ResultQueueImpl::GenerateElementPtr ResultQueueImpl::Get(int timeout_ms) {
  return GetWithTimeout(timeout_ms);
}

ResultQueueImpl::GenerateElementPtr ResultQueueImpl::GetNoWait() {
  ResultQueueImpl::GenerateElementPtr ret =
      std::make_shared<AsEngine::GeneratedElements>();

  while (!closed_.load()) {
    AsEngine::GeneratedElements one_new_token;
    auto have_ele = store_queue_.try_dequeue(one_new_token);
    if (have_ele) {
      ret->AppendNewSingleElementToEnd(one_new_token);
    } else {
      break;
    }
  }
  // if have no element, return an empty id array.
  return ret;
}

AsEngine::GenerateRequestStatus ResultQueueImpl::GenerateStatus() {
  return status_.load();
}

void ResultQueueImpl::SetStatus(AsEngine::GenerateRequestStatus new_status) {
  LOG(INFO) << "ResultQueue: SetStatus Start, new_status: "
            << ToString(new_status)
            << ", old_status: " << ToString(status_.load());
  status_.store(new_status);
}

void ResultQueueImpl::AppendGenerateElement(
    std::shared_ptr<AsEngine::GeneratedElements> new_ele) {
  std::vector<int64_t> new_tokens = new_ele->ids_from_generate;
  std::vector<std::vector<std::pair<int, float>>> new_prob =
      new_ele->log_probs_list;
  std::vector<float> new_token_logprobs = new_ele->token_logprobs_list;

#ifdef DEBUG_RESULT_QUEUE
  LOG(INFO) << "AppendGenerateElement, new_tokens.size: " << new_tokens.size()
            << ", store_queue.size: " << store_queue_.size()
            << ", status: " << ToString(status_.load());
#endif

  generate_length_ += new_tokens.size();
  // auto new_ele = std::make_shared<AsEngine::GeneratedElements>();
  // new_ele->ids_from_generate = std::move(new_tokens);

#ifdef DEBUG_RESULT_QUEUE
  LOG(INFO) << "ResultQueue: Append: new ele: "
            << new_ele->ids_from_generate.size()
            << " id: " << new_ele->ids_from_generate[0]
            << " size: " << new_ele->ids_from_generate.size()
            << ", ptr: " << new_ele.get();
#endif
  auto copy_element = std::make_shared<AsEngine::GeneratedElements>(*new_ele);
  store_queue_.enqueue(*copy_element);
}
}  // namespace allspark
