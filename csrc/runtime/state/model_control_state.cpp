/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    model_control_state.cpp
 */
#include <allspark_logging.h>
#include <engine_runtime.h>

namespace allspark {
void ModelControlState::StopLoop() {
  LOG(INFO) << "[" << model_name << " ] "
            << " Model Loop going to stop...";
  if (loop_thread_) {
    loop_thread_->join();
    loop_thread_.reset();
    model_stopped = true;
  }
}
}  // namespace allspark
