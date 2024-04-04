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
};

struct alignas(32) EngineControlMessage {
  EngineControlMessageId msg;
  std::shared_ptr<std::promise<AsStatus>> promise;

  // only available when msg is request-related.
  std::weak_ptr<RequestHandle> request_handle;

  // only available when msg is StartRequest.
  //   char data[128];
  //   char reserved[64];
  std::shared_ptr<AsEngine::RequestContent> request;

  EngineControlMessage(EngineControlMessageId msg_id_,
                       const std::shared_ptr<std::promise<AsStatus>>& promise_)
      : msg(msg_id_), promise(promise_), request_handle{}, request{} {}

  EngineControlMessage(EngineControlMessageId msg_id_,
                       const std::shared_ptr<std::promise<AsStatus>>& promise_,
                       const std::shared_ptr<RequestHandle>& request_handle_)
      : msg(msg_id_),
        promise(promise_),
        request_handle(request_handle_),
        request{} {}

  EngineControlMessage(
      EngineControlMessageId msg_id_,
      const std::shared_ptr<std::promise<AsStatus>>& promise_,
      const std::shared_ptr<RequestHandle>& request_handle_,
      const std::shared_ptr<AsEngine::RequestContent>& request_)
      : msg(msg_id_),
        promise(promise_),
        request_handle(request_handle_),
        request(request_) {}

  /**
   * @brief Check if request_handle is empty. Useful for SyncRequest.
   */
  bool EmptyRequestHandle() const {
    static const std::weak_ptr<RequestHandle> w_null{};
    return !request_handle.owner_before(w_null) &&
           !w_null.owner_before(request_handle);
  }
};

};  // namespace allspark

#endif
