/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    file_util.cpp
 */

#include "file_util.h"  // NOLINT

#include <sys/stat.h>
#include <unistd.h>

#include <fstream>
#include <string>

namespace allspark {
namespace util {

bool IsExists(const std::string& file_path) {
  struct stat buffer;
  return (stat(file_path.c_str(), &buffer) == 0);
}

bool MakeDir(const std::string& dir_path) {
  return mkdir(dir_path.c_str(), 0777) == 0 &&
         chmod(dir_path.c_str(), 0777) == 0;
}

bool MakeDirs(const std::string& path) {
  int status = mkdir(path.c_str(), 0777);
  if (status == 0) {
    return true;
  } else if (errno == ENOENT) {
    std::size_t found = path.find_last_of('/');
    if (found != std::string::npos) {
      std::string parentDir = path.substr(0, found);
      if (!MakeDirs(parentDir)) {
        return false;
      }
      status = mkdir(path.c_str(), 0777);
      return (status == 0);
    }
  }
  return false;
}

}  // namespace util
}  // namespace allspark
