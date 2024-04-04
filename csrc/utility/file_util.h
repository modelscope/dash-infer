/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    file_util.h
 */

#pragma once

#include <common/common.h>
namespace allspark {
namespace util {

bool IsExists(const std::string& file_path);
bool MakeDir(const std::string& dir_path);
bool MakeDirs(const std::string& path);

class Path {
 public:
  Path(const std::string& path) : m_path(path) {}

  std::string filename() const {
    size_t pos = m_path.find_last_of('/');
    if (pos != std::string::npos)
      return m_path.substr(pos + 1);
    else
      return m_path;
  }

  std::string parent_path() const {
    size_t pos = m_path.find_last_of('/');
    if (pos != std::string::npos)
      return m_path.substr(0, pos);
    else
      return "";
  }

  std::string extension() const {
    size_t dotPos = m_path.rfind('.');
    size_t slashPos = m_path.find_last_of('/');

    if (dotPos != std::string::npos &&
        (slashPos == std::string::npos || dotPos > slashPos))
      return m_path.substr(dotPos);
    else
      return "";
  }

  bool is_absolute() const { return (!m_path.empty() && m_path[0] == '/'); }

  std::string get_path() const { return m_path; }

 private:
  std::string m_path;
};

}  // namespace util
}  // namespace allspark
