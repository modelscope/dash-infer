/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    as_engine_decode.cpp
 */

#include "as_engine.h"

namespace allspark {

void AsEngineImpl::DecodeThread(
    std::string model_name, std::shared_ptr<ModelControlState> model_state) {
  bool stop_model = false;
  bool graceful_stop_phase = false;
  const bool is_prefill_worker = false;
  EngineControlMessage graceful_stop_msg{};

  while (!stop_model) {
    bool got_new_message = false;
    EngineControlMessage msg;
    if (model_state->msg_queue_decode.size_approx() > 0) {
      got_new_message = model_state->msg_queue_decode.try_dequeue(msg);
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
          break;
        }
        case EngineControlMessageId::StopRequest: {
          auto uuid = msg.request_uuid;
          DLOG(INFO) << __FUNCTION__ << ": StopRequest: " << uuid;
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

    RunDecodeWorker(model_name, model_state);

    if (graceful_stop_phase) {
      int finished_worker = 0;
      for (int i = 0; i < nranks_; ++i) {
        if (workers_decode_[i]->GetUnFinishedRequest() == 0) {
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

AsStatus AsEngineImpl::RunDecodeWorker(
    std::string model_name, std::shared_ptr<ModelControlState> model_state) {
  DLOG(INFO) << __FUNCTION__ << ", start.";
  AsStatus status;

  while (true) {
    status = this->RunTextGenerationContinue(model_name.c_str());
    if (status == AsStatus::ALLSPARK_SUCCESS ||
        status == AsStatus::ALLSPARK_EMPTY_REQUEST) {
      break;
    }

    LOG(ERROR) << "RunTextGenerationContinue Failed:"
               << AsGetErrorByCode(status);
    if (status == AsStatus::ALLSPARK_CACHE_MEMORY_OUT) {
      // TOOD: instead of stop request and pop to upper service
      // we can do the recompute as a new request, so
      // it's more likely higher to archive higher performance.
      // 显存不足，从目前正在进行的request中，随机选择一个停止
      std::vector<std::string> request_ids;
      std::vector<int> request_lens;
      {
        std::lock_guard<std::mutex> guard(model_state->map_lock_);
        DLOG(INFO) << "[" << __FUNCTION__ << "] "
                   << "model_state map mutex lock passed";
        for (auto& entry : model_state->request_handle_map) {
          auto& handle = entry.second;
          auto& out_queue = model_state->result_queue_map[handle->request_uuid];
          auto out_queue_impl_ptr =
              std::static_pointer_cast<ResultQueueImpl>(out_queue);
          // maybe ContextFinished also can stop?
          if (out_queue_impl_ptr->GenerateStatus() ==
              AsEngine::GenerateRequestStatus::Generating) {
            request_ids.push_back(handle->request_uuid);
            request_lens.push_back(handle->context_length +
                                   handle->generate_length);
          }
        }
      }
      int running_requests = request_ids.size();
      if (running_requests == 0) {
        LOG(ERROR) << __FUNCTION__ << ": No Generating request!";
        break;
      }

      std::string victim_request_id = this->ChooseVictimRequest(
          request_ids, request_lens, running_requests);
      auto ret = this->StopRequestByRequestID(model_name.c_str(),
                                              victim_request_id, true);

      DLOG(INFO) << __FUNCTION__
                 << ": Memory is running out, choose one request to stop "
                 << " ID: " << victim_request_id;

      if (ret != AsStatus::ALLSPARK_SUCCESS) {
        LOG(INFO) << __FUNCTION__ << ": " << model_name << " StopRequest"
                  << victim_request_id << " failed " << (int)ret;
      }

      {
        std::lock_guard<std::mutex> guard(model_state->map_lock_);
        DLOG(INFO) << "[" << __FUNCTION__ << "] "
                   << "model_state map mutex lock passed";
        auto& out_queue = model_state->result_queue_map[victim_request_id];
        auto out_queue_impl_ptr =
            std::static_pointer_cast<ResultQueueImpl>(out_queue);
        out_queue_impl_ptr->SetStatus(
            AsEngine::GenerateRequestStatus::GenerateInterrupted);
        LOG(INFO) << __FUNCTION__
                  << ": Memory is running out, Request :" << victim_request_id
                  << " GenerateInterruptedInterrupted!";
      }
    } else {
      LOG(ERROR) << "RunTextGenerationContinue Failed:"
                 << AsGetErrorByCode(status);
      // TODO: instead of abort, clean all request and restart.
      abort();
      break;
    }
  }

  DLOG(INFO) << __FUNCTION__ << ", finish.";
  return status;
}

AsStatus AsEngineImpl::RunTextGenerationContinue(const char* model_name) {
  DLOG(INFO) << "[" << model_name << "] "
             << "AsEngineImpl::RunTextGenerationContinue" << std::endl;

  // check model registered
  if (model_irs_[model_name] == nullptr) {
    LOG(ERROR) << "[" << model_name << "] "
               << "Invalid model name : " << model_name << std::endl;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }

  // verify params
  if (!model_irs_[model_name]->model_conf().is_generate()) {
    LOG(ERROR) << "[" << model_name << "] "
               << "RunTextGenerationContinue() is only supported in text "
                  "generation model.Please use RunModel() API."
               << std::endl;
    return AsStatus::ALLSPARK_INVALID_CALL_ERROR;
  }

  // 即使失败、异常，也要让各子线程运行完毕，以保证原子性。在可恢复的情况下，确保下一次请求有干净的环境
  AsStatus failed_ret = AsStatus::ALLSPARK_SUCCESS;
  std::future<AsStatus> result[nranks_];

  int pending_num = workers_decode_[0]->GetPendingDecodeNum();
  int64_t min_free_count = min_free_frame_count_.load();

  std::vector<int> pres_frame(nranks_);
  for (int i = 0; i < nranks_; ++i) {
    result[i] = threadpool_decode_->enqueue(
        i, [this, i, pending_num, min_free_count, &pres_frame]() {
          try {
            return workers_decode_[i]->AllocDecoderMemory(
                pending_num, min_free_count, pres_frame[i]);
          } catch (std::exception& e) {
            LOG(ERROR) << "AllocDecoderMemory Failed!"
                       << " status: " << std::string(e.what())
                       << " worker_id: " << i;
            if (std::string(e.what()) == "ALLSPARK_MEMORY_ERROR" ||
                std::string(e.what()) == "ALLSPARK_CACHE_MEMORY_OUT") {
              return AsStatus::ALLSPARK_CACHE_MEMORY_OUT;
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
      workers_decode_[i]->FreePresFrame(pres_frame[i]);
      if (ret != AsStatus::ALLSPARK_SUCCESS) {
        LOG(ERROR) << "AllocDecoderMemory Failed!"
                   << " status: " << AsStatusToString(ret)
                   << " worker_id: " << i;
        failed_ret = ret;
      } else {
        LOG(INFO) << "AllocDecoderMemory Success."
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
    result[i] = threadpool_decode_->enqueue(i, [this, i]() {
      return workers_decode_[i]->RunTextGenerationContinue();
    });
  }

  for (int i = 0; i < nranks_; ++i) {
    try {
      AsStatus ret = result[i].get();
      if (not AS_STATUS_OK(ret)) failed_ret = ret;
    } catch (std::exception& e) {
      if (std::string(e.what()) == "ALLSPARK_MEMORY_ERROR") {
        LOG(ERROR) << "AsEngineImpl::RunTextGenerationContinue: "
                      "exception caught: ALLSPARK_MEMORY_ERROR";
        throw AsException(("ALLSPARK_MEMORY_ERROR"));
      } else {
        AsSaveError(e.what());
        LOG(ERROR) << "AsEngineImpl::RunTextGenerationContinue: "
                      "exception caught: "
                   << e.what() << ", saved with AsSaveError";
        failed_ret = AsStatus::ALLSPARK_RUNTIME_ERROR;
      }
    }
  }
  return failed_ret;
}

std::string AsEngineImpl::ChooseVictimRequest(
    const std::vector<std::string>& request_ids,
    const std::vector<int>& request_lens, int n) {
  std::string stop_request_id;
  AsEvictionStrategy eviction_strategy = device_ctx_->GetEvictionStrategy();
  switch (eviction_strategy) {
    case AsEvictionStrategy::MaxLength: {
      int x = 0;
      for (int i = 0; i < request_lens.size(); i++) {
        if (request_lens[i] > request_lens[x]) {
          x = i;
        }
      }
      stop_request_id = request_ids[x];
      LOG(INFO) << "ALLSPARK_CACHE_MEMORY_OUT, stop MaxLength request_id = "
                << request_ids[x] << " ,length = " << request_lens[x];
      break;
    }
    case AsEvictionStrategy::Random: {
      auto rand = random_engine();
      int x = rand % n;
      stop_request_id = request_ids[x];
      LOG(INFO) << "ALLSPARK_CACHE_MEMORY_OUT, stop Random request_id = "
                << request_ids[x] << " ,length = " << request_lens[x];
      break;
    }
    default: {
      LOG(ERROR) << "not support EvictionStrategy = ";
      auto rand = random_engine();
      int x = rand % n;
      LOG(INFO) << "ALLSPARK_CACHE_MEMORY_OUT, stop Random request_id = "
                << request_ids[x] << " ,length = " << request_lens[x];
      stop_request_id = request_ids[x];
      break;
    }
  }
  return stop_request_id;
}

}  // namespace allspark
