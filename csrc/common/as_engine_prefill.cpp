/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    as_engine_prefill.cpp
 */

#include "as_engine.h"

namespace allspark {

void AsEngineImpl::PrefillThread(
    std::string model_name, std::shared_ptr<ModelControlState> model_state) {
  bool stop_model = false;
  bool graceful_stop_phase = false;
  const bool is_prefill_worker = true;
  EngineControlMessage graceful_stop_msg{};

  while (!stop_model) {
    bool got_new_message = false;
    EngineControlMessage msg;
    if (model_state->msg_queue_prefill.size_approx() > 0) {
      got_new_message = model_state->msg_queue_prefill.try_dequeue(msg);
      if (!got_new_message) {
        LOG(ERROR) << __FUNCTION__ << ": queue size > 0, but not no message";
      } else {
        DLOG(INFO) << __FUNCTION__
                   << ": receive message: " << ToString(msg.msg);
      }
    }

    if (got_new_message) {
      switch (msg.msg) {
        case EngineControlMessageId::GracefulStopModel: {
          graceful_stop_phase = true;
          graceful_stop_msg = std::move(msg);
          break;
        }
        case EngineControlMessageId::StartRequest: {
          DLOG(INFO) << __FUNCTION__
                     << ": StartRequest add : " << msg.request_uuid
                     << " into queue map , ptr: " << msg.result_queue.get();

          util::Timer t1;

          auto ret = this->StartRequestImpl(
              model_name.c_str(), msg.request_handle, msg.request->config);
          DLOG(INFO) << __FUNCTION__ << ": StartRequestImpl finish "
                     << t1.elapsed() << " ms";
          if (ret != AsStatus::ALLSPARK_SUCCESS) {
            LOG(ERROR) << __FUNCTION__ << ": StartRequest return failed: "
                       << " uuid: " << msg.request_uuid << " " << (int)ret;
          } else {
            std::lock_guard<std::mutex> guard(model_state->map_lock_);
            DLOG(INFO) << "[" << __FUNCTION__ << "] "
                       << "model_state map mutex lock passed";

            auto ret_insert1 = model_state->result_queue_map.emplace(
                msg.request_uuid, msg.result_queue);
            auto ret_insert2 = model_state->request_handle_map.emplace(
                msg.request_uuid, msg.request_handle);
            if (!ret_insert1.second && !ret_insert2.second) {
              // this means the key already in the map.
              LOG(ERROR) << __FUNCTION__
                         << ": StartRequest: with duplicated uuid: "
                         << msg.request_uuid;
            }
          }
          msg.promise->set_value(ret);
          break;
        }
        case EngineControlMessageId::StopRequest: {
          auto uuid = msg.request_uuid;
          LOG(INFO) << __FUNCTION__ << ": StopRequest: " << uuid;
          auto ret = this->StopRequestByRequestID(model_name.c_str(), uuid,
                                                  is_prefill_worker);
          msg.promise->set_value(ret);
          break;
        }
        case EngineControlMessageId::SyncRequest: {
          break;
        }
        case EngineControlMessageId::SyncAllRequest: {
          break;
        }
        case EngineControlMessageId::ReleaseRequest: {
          LOG(INFO) << __FUNCTION__
                    << ": ReleaseRequest received: " << msg.request_uuid;

          auto uuid = msg.request_uuid;

          auto ret_stop = this->StopRequestByRequestID(model_name.c_str(), uuid,
                                                       is_prefill_worker);

          auto ret_release = this->ReleaseRequestByRequestID(
              model_name.c_str(), uuid, is_prefill_worker);

          if (ret_stop != AsStatus::ALLSPARK_SUCCESS)
            msg.promise->set_value(ret_stop);
          else
            msg.promise->set_value(ret_release);

          break;
        }
        default: {
          LOG(WARNING) << __FUNCTION__
                       << ": Warning: unhandle message received: "
                       << (int)msg.msg;
        }
      }
    }

    RunPrefillWorker(model_name);

    if (graceful_stop_phase) {
      int finished_worker = 0;
      for (int i = 0; i < nranks_; ++i) {
        if (workers_[i]->GetUnFinishedRequest() == 0) {
          finished_worker++;
        }
      }
      if (finished_worker == nranks_) {
        stop_model = true;
        graceful_stop_msg.promise->set_value(AsStatus::ALLSPARK_SUCCESS);
      }
    }
  }
}

AsStatus AsEngineImpl::RunPrefillWorker(std::string model_name) {
  DLOG(INFO) << __FUNCTION__ << ", start.";

  AsStatus status = AsStatus::ALLSPARK_SUCCESS;

  if (workers_decode_[0]->GetRunningRequest() >=
      device_ctx_->GetModelMaxBatch()) {
    return status;
  }

  status = this->RunTextGenerationContext(model_name.c_str());
  if (status != AsStatus::ALLSPARK_SUCCESS &&
      status != AsStatus::ALLSPARK_EMPTY_REQUEST) {
    LOG(ERROR) << "RunTextGenerationContext Failed:"
               << AsGetErrorByCode(status);
  }

  DLOG(INFO) << __FUNCTION__ << ", finish.";
  return status;
}

AsStatus AsEngineImpl::RunEngineContext(std::string model_name) {
  AsSchedulingStrategy scheduling_strategy =
      device_ctx_->GetSchedulingStrategy();
  switch (scheduling_strategy) {
    case AsSchedulingStrategy::ContextPriority: {
      LOG(INFO) << "AsSchedulingStrategy::ContextPriority";
      int run_context_count = 0;
      while (true) {
        AsStatus status = this->RunTextGenerationContext(model_name.c_str());
        run_context_count += 1;
        if (status != AsStatus::ALLSPARK_SUCCESS &&
            status != AsStatus::ALLSPARK_EMPTY_REQUEST &&
            status != AsStatus::ALLSPARK_CHUNK_PREFILL) {
          // context error
          return status;
        }
        if (status == AsStatus::ALLSPARK_CHUNK_PREFILL ||
            status == AsStatus::ALLSPARK_EMPTY_REQUEST) {
          // can't do more context
          break;
        }
        // continue context
      }
      return AsStatus::ALLSPARK_SUCCESS;
    }
    case AsSchedulingStrategy::Balance: {
      LOG(INFO) << "AsSchedulingStrategy::Balance";
      // just run one context,must be new context
      AsStatus status = this->RunTextGenerationContext(model_name.c_str());
      return status;
    }
    default: {
      LOG(ERROR) << "not support scheduling_strategy ";
      return AsStatus::ALLSPARK_PARAM_ERROR;
    }
  }
  return AsStatus::ALLSPARK_UNKNOWN_ERROR;
}

AsStatus AsEngineImpl::RunTextGenerationContext(const char* model_name) {
  DLOG(INFO) << "[" << model_name << "] "
             << "AsEngineImpl::RunTextGenerationContext" << std::endl;

  TracerLog t(device_ctx_->GetDeviceType(), "RunCtx", 2);
  // check model registered
  if (model_irs_[model_name] == nullptr) {
    LOG(ERROR) << "[" << model_name << "] "
               << "Invalid model name : " << model_name << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }
  // verify params
  if (model_irs_[model_name]->model_conf().is_generate() == false) {
    LOG(ERROR) << "[" << model_name << "] "
               << "RunTextGenerationContext() is only supported in text "
                  "generation model. Please use RunModel() API."
               << std::endl;
    return AsStatus::ALLSPARK_INVALID_CALL_ERROR;
  }

  // 即使失败、异常，也要让各子线程运行完毕，以保证原子性。在可恢复的情况下，确保下一次请求有干净的环境
  AsStatus failed_ret = AsStatus::ALLSPARK_SUCCESS;
  std::future<AsStatus> result[nranks_];

  int64_t min_free_count = min_free_frame_count_.load();

  std::vector<int> pres_frame(nranks_);
  for (int i = 0; i < nranks_; ++i) {
    result[i] = threadpool_->enqueue(i, [this, i, min_free_count,
                                         &pres_frame]() {
      try {
        return workers_[i]->AllocPrefillMemory(min_free_count, pres_frame[i]);
      } catch (std::exception& e) {
        LOG(ERROR) << "AllocPrefillMemory Failed!"
                   << " status: " << std::string(e.what())
                   << " worker_id: " << i;
        if (std::string(e.what()) == "ALLSPARK_MEMORY_ERROR" ||
            std::string(e.what()) == "ALLSPARK_CACHE_MEMORY_OUT") {
          return AsStatus::ALLSPARK_CACHE_MEMORY_OUT;
        } else if (std::string(e.what()) == "ALLSPARK_EMPTY_REQUEST") {
          return AsStatus::ALLSPARK_EMPTY_REQUEST;
        } else {
          return AsStatus::ALLSPARK_RUNTIME_ERROR;
        }
      }
    });
  }

  bool has_fail = false;
  std::vector<AsStatus> tmp_ret(nranks_);
  for (int i = 0; i < nranks_; ++i) {
    tmp_ret[i] = result[i].get();
    if (tmp_ret[i] != AsStatus::ALLSPARK_SUCCESS) {
      has_fail = true;
    }
  }

  for (int i = 0; i < nranks_; ++i) {
    AsStatus ret = tmp_ret[i];
    if (has_fail) {
      workers_[i]->FreePresFrame(pres_frame[i]);
      if (ret != AsStatus::ALLSPARK_SUCCESS) {
        LOG(ERROR) << "AllocPrefillMemory Failed!"
                   << " status: " << AsStatusToString(ret)
                   << " worker_id: " << i;
        failed_ret = ret;
      } else {
        LOG(INFO) << "AllocPrefillMemory Success."
                  << " status: " << AsStatusToString(ret) << " worker_id: " << i
                  << " free preserved frame: " << pres_frame[i];
      }
    }
  }

  // 任何一个子线程reshape阶段出问题都返回
  if (failed_ret != AsStatus::ALLSPARK_SUCCESS) {
    return failed_ret;
  }

  for (int i = 0; i < nranks_; ++i) {
    result[i] = threadpool_->enqueue(i, [this, i]() {
      try {
        return workers_[i]->RunTextGenerationContext();
      } catch (std::exception& e) {
        LOG(ERROR) << "RunTextGenerationContext Failed:"
                   << std::string(e.what());
        if (std::string(e.what()) == "ALLSPARK_MEMORY_ERROR" ||
            std::string(e.what()) == "ALLSPARK_CACHE_MEMORY_OUT") {
          return AsStatus::ALLSPARK_CACHE_MEMORY_OUT;
        } else if (std::string(e.what()) == "ALLSPARK_EMPTY_REQUEST") {
          return AsStatus::ALLSPARK_EMPTY_REQUEST;
        } else {
          return AsStatus::ALLSPARK_RUNTIME_ERROR;
        }
      }
    });
  }

  for (int i = 0; i < nranks_; ++i) {
    AsStatus ret = result[i].get();
    if (ret != AsStatus::ALLSPARK_SUCCESS) {
      failed_ret = ret;
    }
  }
  return failed_ret;
}

}  // namespace allspark
