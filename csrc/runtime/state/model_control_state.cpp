/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    model_control_state.cpp
 */
//
// Created by jiejing.zjj on 4/28/24.
//

#include <allspark_logging.h>
#include <engine_runtime.h>

namespace allspark {
void ModelControlState::StopLoop() {
  LOG(INFO) << "[" << model_name << " ] "
            << " Model Loop going to stop...";
  std::unique_lock<std::mutex> lock(*(this->lock));
  if (loop_thread_) {
    loop_thread_->join();
    loop_thread_.reset();
    model_stopped = true;
  }
}
}  // namespace allspark