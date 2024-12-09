/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    allsparkz_util.h
 */

#pragma once
#include <allspark.h>

#include <map>
#include <string>
#include <vector>

#include "sparse_util.h"
namespace allspark {
namespace util {

std::string save_allsparky(const std::string& bin_data,
                           TensorAttribute& tensor_info);

void save_allsparky_tofile(const std::string& weights_path,
                           const std::string& name, void* data_ptr,
                           int64_t nbytes, TensorAttribute& tensor_info);

void set_global_header(const std::string& weights_path);

template <typename T>
std::vector<char>& operator+=(std::vector<char>& lhs, const T rhs) {
  // write in little endian
  for (size_t byte = 0; byte < sizeof(T); byte++) {
    char val = *((char*)&rhs + byte);
    lhs.push_back(val);
  }
  return lhs;
}

template <>
std::vector<char>& operator+=(std::vector<char>& lhs, const std::string rhs);
template <>
std::vector<char>& operator+=(std::vector<char>& lhs, const char* rhs);
}  // namespace util
}  // namespace allspark
