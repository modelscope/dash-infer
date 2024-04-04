/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    allspark_check.cpp
 */
#include <assert.h>
#include <interface/allspark_check.h>

#include <algorithm>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>

namespace allspark {

static std::vector<std::string> g_errors;
static std::mutex g_errors_lock;

const std::string AsGetErrorByCode(AsStatus error_code) {
  switch (error_code) {
    case AsStatus::ALLSPARK_SUCCESS:
      return "ALLSPARK_SUCCESS";
    case AsStatus::ALLSPARK_STREAMING:
      return "ALLSPARK_STREAMING";
    case AsStatus::ALLSPARK_UNKNOWN_ERROR:
      return "ALLSPARK_UNKNOWN_ERROR";
    case AsStatus::ALLSPARK_PARAM_ERROR:
      return "ALLSPARK_PARAM_ERROR";
    case AsStatus::ALLSPARK_IO_ERROR:
      return "ALLSPARK_IO_ERROR";
    case AsStatus::ALLSPARK_MEMORY_ERROR:
      return "ALLSPARK_MEMORY_ERROR";
    case AsStatus::ALLSPARK_EXCEED_LIMIT_ERROR:
      return "ALLSPARK_EXCEED_LIMIT_ERROR";
    case AsStatus::ALLSPARK_INVALID_CALL_ERROR:
      return "ALLSPARK_INVALID_CALL_ERROR";
    case AsStatus::ALLSPARK_EMPTY_REQUEST:
      return "ALLSPARK_EMPTY_REQUEST";
    case AsStatus::ALLSPARK_ILLEGAL_REQUEST_ID:
      return "ALLSPARK_ILLEGAL_REQUEST_ID";
    case AsStatus::ALLSPARK_CACHE_MEMORY_OUT:
      return "ALLSPARK_CACHE_MEMORY_OUT";
    case AsStatus::ALLSPARK_DEPRECATED:
      return "ALLSPARK_DEPRECATED";
    case AsStatus::ALLSPARK_RUNTIME_ERROR:
      if (g_errors.size())
        return "ALLSPARK_RUNTIME_ERROR" + AsConcatErrors();
      else
        return "ALLSPARK_RUNTIME_ERROR";

    default:
      return "ALLSPARK_UNDEFINED_ERROR_CODE";
  }
}

void AsSaveError(const std::string& err_str) {
  std::lock_guard<std::mutex> guard(g_errors_lock);
  // one err_str per type is enough
  if (std::find(g_errors.begin(), g_errors.end(), err_str) == g_errors.end()) {
    g_errors.emplace_back(err_str);
  }
}

const std::string AsConcatErrors() {
  std::lock_guard<std::mutex> guard(g_errors_lock);
  std::stringstream ss;
  if (g_errors.size()) ss << "|";
  for (auto& err_str : g_errors) {
    ss << err_str << "#";
  }
  return ss.str();
}

void AsClearErrors() {
  std::lock_guard<std::mutex> guard(g_errors_lock);
  g_errors.clear();
}

}  // namespace allspark
