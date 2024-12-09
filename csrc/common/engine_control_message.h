/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    engine_control_message.h
 */

#ifndef ENGINE_CONTROL_MESSAGE_H
#define ENGINE_CONTROL_MESSAGE_H

#include <future>
#include <memory>

#include "allspark.h"

namespace allspark {

class RequestHandle;

enum class EngineControlMessageId {
  /// model related
  BuildModel = 1,
  StartModel = 2,
  StopModel = 3,
  ReleaseModel = 4,  // deprecated
  GracefulStopModel = 5,

  /// request relalted
  StartRequest = 11,
  SyncRequest = 12,
  StopRequest = 13,
  ReleaseRequest = 14,
  SyncAllRequest = 15,

  // Unknown message
  // this message for default constructor.
  UnknownRequest = 100,
};

inline std::string ToString(EngineControlMessageId id) {
  switch (id) {
    case EngineControlMessageId::BuildModel:
      return "BuildModel";
    case EngineControlMessageId::StartModel:
      return "StartModel";
    case EngineControlMessageId::StopModel:
      return "StopModel";
    case EngineControlMessageId::ReleaseModel:
      return "ReleaseModel";
    case EngineControlMessageId::GracefulStopModel:
      return "GracefulStopModel";
    case EngineControlMessageId::StartRequest:
      return "StartRequest";
    case EngineControlMessageId::SyncRequest:
      return "SyncRequest";
    case EngineControlMessageId::SyncAllRequest:
      return "SyncAllRequest";
    case EngineControlMessageId::StopRequest:
      return "StopRequest";
    case EngineControlMessageId::ReleaseRequest:
      return "ReleaseRequest";
    case EngineControlMessageId::UnknownRequest:
      return "UnknownRequest";
    default:
      return "Unknown";
  }
}

class ResultQueueImpl;

struct alignas(32) EngineControlMessage {
  EngineControlMessageId msg;
  std::shared_ptr<std::promise<AsStatus>> promise;

  std::string request_uuid;

  //   char data[128];
  //   char reserved[64];
  std::shared_ptr<AsEngine::RequestContent> request;

  // only available when msg is StartRequest.
  std::shared_ptr<RequestHandle> request_handle;
  std::shared_ptr<AsEngine::ResultQueue> result_queue;

  EngineControlMessage()
      : msg(EngineControlMessageId::UnknownRequest),
        request_uuid("Control-Unknown-UUID") {}

  EngineControlMessage(EngineControlMessageId msg_id_,
                       const std::shared_ptr<std::promise<AsStatus>>& promise_)
      : msg(msg_id_),
        promise(promise_),
        request_uuid("Control-Unknown-UUID"),
        request{} {}

  EngineControlMessage(EngineControlMessageId msg_id_,
                       const std::shared_ptr<std::promise<AsStatus>>& promise_,
                       const std::string& request_uuid_)
      : msg(msg_id_),
        promise(promise_),
        request_uuid(request_uuid_),
        request{} {}

  EngineControlMessage(
      EngineControlMessageId msg_id_,
      const std::shared_ptr<std::promise<AsStatus>>& promise_,
      const std::string& request_uuid_,
      const std::shared_ptr<AsEngine::RequestContent>& request_)
      : msg(msg_id_),
        promise(promise_),
        request_uuid(request_uuid_),
        request(request_) {}

  EngineControlMessage(
      EngineControlMessageId msg_id_,
      const std::shared_ptr<std::promise<AsStatus>>& promise_,
      const std::string& request_uuid_, std::shared_ptr<RequestHandle> handle_,
      std::shared_ptr<AsEngine::ResultQueue> result_queue_,

      const std::shared_ptr<AsEngine::RequestContent>& request_)
      : msg(msg_id_),
        promise(promise_),
        request_uuid(request_uuid_),
        request_handle(handle_),
        result_queue(result_queue_),
        request(request_) {}

  EngineControlMessage(const EngineControlMessage&) = delete;
  EngineControlMessage& operator=(const EngineControlMessage&) = delete;

  EngineControlMessage(EngineControlMessage&& other) noexcept
      : msg(other.msg),
        promise(std::move(other.promise)),
        request_uuid(other.request_uuid),
        request_handle(other.request_handle),
        result_queue(other.result_queue),
        request(other.request) {
    other.request_handle.reset();
    other.result_queue.reset();
    other.request.reset();
  }

  // 移动赋值运算符
  EngineControlMessage& operator=(EngineControlMessage&& other) noexcept {
    if (this != &other) {
      msg = other.msg;
      promise = std::move(other.promise);
      request_uuid = other.request_uuid;
      request_handle = other.request_handle;
      result_queue = other.result_queue;
      request = other.request;

      other.request_handle.reset();
      other.result_queue.reset();
      other.request.reset();
    }
    return *this;
  }
};

};  // namespace allspark

#endif
