
/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    request.h
 */

#pragma once
#include <core/tensor/tensor.h>

#include <string>

namespace allspark {

struct Request {
  std::string request_id;
  TensorMap inputs;
  TensorMap outputs;
  GenerateConfig gen_cfg;
  std::vector<std::vector<std::pair<int64_t, float>>> log_probs_list;
  bool finish = false;
  int input_len = 0;
  AsEngine::GenerateRequestStatus status =
      AsEngine::GenerateRequestStatus::Init;
  TensorListMap extra_embedding;
  Request(const std::string& request_id_, const TensorMap& inputs_,
          const TensorMap& outputs_, const GenerateConfig& gen_cfg)
      : request_id(request_id_),
        inputs(inputs_),
        outputs(outputs_),
        gen_cfg(gen_cfg),
        finish(false),
        status(AsEngine::GenerateRequestStatus::Init) {}
};

}  // namespace allspark
