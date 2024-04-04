/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    weight_saver.h
 */
#pragma once

#include <common/common.h>
#include <common/device_context.h>
#include <core/tensor/tensor.h>

#include "utility/string_util.h"

namespace allspark {

class WeightSerialization {
 public:
  WeightSerialization();

  // AsTensor -> allsparky
  void SerializeSingleTensor(const AsTensor* tensor, std::string* bin_data);

  // TensorMap -> allsparkz
  void SerializeMultipleTensor(const TensorMap& tensors, std::string* bin_data);
};

}  // namespace allspark
