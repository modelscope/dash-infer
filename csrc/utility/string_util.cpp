/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    string_util.cpp
 */

#include "string_util.h"

#include <assert.h>
#include <errno.h>
#include <memory.h>
#include <stdint.h>

#include <algorithm>

namespace allspark {
namespace util {

void split(std::vector<std::string>& out, std::string& str, std::string delim) {
  out.clear();
  std::string s(str);
  size_t pos;
  while ((pos = s.find(delim)) != std::string::npos) {
    out.emplace_back(s.substr(0, pos));
    s = s.substr(pos + delim.size(), s.size());
  }
  out.emplace_back(s);
}

StringUtil::StringUtil() {}

StringUtil::~StringUtil() {}

void StringUtil::Trim(std::string& str) {
  str.erase(str.find_last_not_of(" \t\r\n") + 1);
  str.erase(0, str.find_first_not_of(" \t\r\n"));
}

char* StringUtil::Trim(const char* szStr) {
  const char* szTmp;
  char *szRet, *szTrim;

  szTmp = szStr;
  while ((*szTmp) == ' ' || (*szTmp) == '\t' || (*szTmp) == '\r' ||
         (*szTmp) == '\n')
    szTmp++;
  size_t len = strlen(szTmp);
  szTrim = replicate(szTmp);
  if (len > 0 && (szTrim[len - 1] == ' ' || szTrim[len - 1] == '\t' ||
                  szTrim[len - 1] == '\r' || szTrim[len - 1] == '\n')) {
    szRet = &szTrim[len - 1];
    while ((*szRet) == ' ' || (*szRet) == '\t' || (*szRet) == '\r' ||
           (*szRet) == '\n')
      szRet--;
    *(++szRet) = '\0';
  }

  return szTrim;
}
std::vector<std::string> StringUtil::Split(const std::string& text,
                                           const char* sepStr) {
  return StringUtil::Split(text.c_str(), sepStr);
}

std::vector<std::string> StringUtil::Split(const char* text,
                                           const char* sepStr) {
  std::vector<std::string> vec;
  std::string str(text);
  std::string sep(sepStr);
  size_t n = 0, old = 0;
  while (n != std::string::npos) {
    n = str.find(sep, n);
    if (n != std::string::npos) {
      if (n != old) vec.push_back(str.substr(old, n - old));
      n += sep.length();
      old = n;
    }
  }
  vec.push_back(str.substr(old, str.length() - old));
  return vec;
}

// split with multiple char delimiters
std::vector<std::string> StringUtil::Split2(const std::string& s,
                                            const std::string& delimiters) {
  std::vector<std::string> tokens;
  std::string::size_type lastPos = s.find_first_not_of(delimiters, 0);
  std::string::size_type pos = s.find_first_of(delimiters, lastPos);
  while (std::string::npos != pos || std::string::npos != lastPos) {
    tokens.push_back(s.substr(lastPos, pos - lastPos));
    lastPos = s.find_first_not_of(delimiters, pos);
    pos = s.find_first_of(delimiters, lastPos);
  }
  return tokens;
}

bool StringUtil::StrToInt8(const char* str, int8_t& value) {
  int32_t v32;
  bool ret = StrToInt32(str, v32);
  if (ret) {
    if (v32 >= INT8_MIN && v32 <= INT8_MAX) {
      value = (int8_t)v32;
      return true;
    }
  }
  return false;
}

bool StringUtil::StrToUInt8(const char* str, uint8_t& value) {
  uint32_t v32;
  bool ret = StrToUInt32(str, v32);
  if (ret) {
    if (v32 <= UINT8_MAX) {
      value = (uint8_t)v32;
      return true;
    }
  }
  return false;
}

bool StringUtil::StrToInt16(const char* str, int16_t& value) {
  int32_t v32;
  bool ret = StrToInt32(str, v32);
  if (ret) {
    if (v32 >= INT16_MIN && v32 <= INT16_MAX) {
      value = (int16_t)v32;
      return true;
    }
  }
  return false;
}

bool StringUtil::StrToUInt16(const char* str, uint16_t& value) {
  uint32_t v32;
  bool ret = StrToUInt32(str, v32);
  if (ret) {
    if (v32 <= UINT16_MAX) {
      value = (uint16_t)v32;
      return true;
    }
  }
  return false;
}

bool StringUtil::StrToInt32(const char* str, int32_t& value) {
  if (NULL == str || *str == 0) {
    return false;
  }
  char* endPtr = NULL;
  errno = 0;

#if __WORDSIZE == 64
  int64_t value64 = strtol(str, &endPtr, 10);
  if (value64 < INT32_MIN || value64 > INT32_MAX) {
    return false;
  }
  value = (int32_t)value64;
#else
  value = (int32_t)strtol(str, &endPtr, 10);
#endif

  if (errno == 0 && endPtr && *endPtr == 0) {
    return true;
  }
  return false;
}

bool StringUtil::StrToUInt32(const char* str, uint32_t& value) {
  if (NULL == str || *str == 0 || *str == '-') {
    return false;
  }
  char* endPtr = NULL;
  errno = 0;

#if __WORDSIZE == 64
  uint64_t value64 = strtoul(str, &endPtr, 10);
  if (value64 > UINT32_MAX) {
    return false;
  }
  value = (int32_t)value64;
#else
  value = (uint32_t)strtoul(str, &endPtr, 10);
#endif

  if (errno == 0 && endPtr && *endPtr == 0) {
    return true;
  }
  return false;
}

bool StringUtil::StrToUInt64(const char* str, uint64_t& value) {
  if (NULL == str || *str == 0 || *str == '-') {
    return false;
  }
  char* endPtr = NULL;
  errno = 0;
  value = (uint64_t)strtoull(str, &endPtr, 10);
  if (errno == 0 && endPtr && *endPtr == 0) {
    return true;
  }
  return false;
}

bool StringUtil::StrToInt64(const char* str, int64_t& value) {
  if (NULL == str || *str == 0) {
    return false;
  }
  char* endPtr = NULL;
  errno = 0;
  value = (int64_t)strtoll(str, &endPtr, 10);
  if (errno == 0 && endPtr && *endPtr == 0) {
    return true;
  }
  return false;
}

bool StringUtil::HexStrToUint64(const char* str, uint64_t& value) {
  if (NULL == str || *str == 0) {
    return false;
  }
  char* endPtr = NULL;
  errno = 0;
  value = (uint64_t)strtoull(str, &endPtr, 16);
  if (errno == 0 && endPtr && *endPtr == 0) {
    return true;
  }
  return false;
}

uint32_t StringUtil::DeserializeUInt32(const std::string& str) {
  assert(str.length() == sizeof(uint32_t));

  uint32_t value = 0;
  for (size_t i = 0; i < str.length(); ++i) {
    value <<= 8;
    value |= (unsigned char)str[i];
  }
  return value;
}

void StringUtil::SerializeUInt32(uint32_t value, std::string& str) {
  char key[4];
  for (int i = (int)sizeof(uint32_t) - 1; i >= 0; --i) {
    key[i] = (char)(value & 0xFF);
    value >>= 8;
  }
  str.assign(key, sizeof(uint32_t));
}

void StringUtil::SerializeUInt64(uint64_t value, std::string& str) {
  char key[8];
  for (int i = (int)sizeof(uint64_t) - 1; i >= 0; --i) {
    key[i] = (char)(value & 0xFF);
    value >>= 8;
  }
  str.assign(key, sizeof(uint64_t));
}

uint64_t StringUtil::DeserializeUInt64(const std::string& str) {
  assert(str.length() == sizeof(uint64_t));

  uint64_t value = 0;
  for (size_t i = 0; i < str.length(); ++i) {
    value <<= 8;
    value |= (unsigned char)str[i];
  }
  return value;
}

bool StringUtil::StrToFloat(const char* str, float& value) {
  if (NULL == str || *str == 0) {
    return false;
  }
  errno = 0;
  char* endPtr = NULL;
  value = strtof(str, &endPtr);
  if (errno == 0 && endPtr && *endPtr == 0) {
    return true;
  }
  return false;
}

bool StringUtil::StrToDouble(const char* str, double& value) {
  if (NULL == str || *str == 0) {
    return false;
  }
  errno = 0;
  char* endPtr = NULL;
  value = strtod(str, &endPtr);
  if (errno == 0 && endPtr && *endPtr == 0) {
    return true;
  }
  return false;
}

void StringUtil::Uint64ToHexStr(uint64_t value, char* hexStr, int len) {
  assert(len > 16);
  snprintf(hexStr, len, "%016lx", value);
}

int8_t StringUtil::StrToInt8WithDefault(const char* str, int8_t defaultValue) {
  int8_t tmp;
  if (StrToInt8(str, tmp)) {
    return tmp;
  }
  return defaultValue;
}

uint8_t StringUtil::StrToUInt8WithDefault(const char* str,
                                          uint8_t defaultValue) {
  uint8_t tmp;
  if (StrToUInt8(str, tmp)) {
    return tmp;
  }
  return defaultValue;
}

int16_t StringUtil::StrToInt16WithDefault(const char* str,
                                          int16_t defaultValue) {
  int16_t tmp;
  if (StrToInt16(str, tmp)) {
    return tmp;
  }
  return defaultValue;
}

uint16_t StringUtil::StrToUInt16WithDefault(const char* str,
                                            uint16_t defaultValue) {
  uint16_t tmp;
  if (StrToUInt16(str, tmp)) {
    return tmp;
  }
  return defaultValue;
}

int32_t StringUtil::StrToInt32WithDefault(const char* str,
                                          int32_t defaultValue) {
  int32_t tmp;
  if (StrToInt32(str, tmp)) {
    return tmp;
  }
  return defaultValue;
}

uint32_t StringUtil::StrToUInt32WithDefault(const char* str,
                                            uint32_t defaultValue) {
  uint32_t tmp;
  if (StrToUInt32(str, tmp)) {
    return tmp;
  }
  return defaultValue;
}

int64_t StringUtil::StrToInt64WithDefault(const char* str,
                                          int64_t defaultValue) {
  int64_t tmp;
  if (StrToInt64(str, tmp)) {
    return tmp;
  }
  return defaultValue;
}

uint64_t StringUtil::StrToUInt64WithDefault(const char* str,
                                            uint64_t defaultValue) {
  uint64_t tmp;
  if (StrToUInt64(str, tmp)) {
    return tmp;
  }
  return defaultValue;
}

float StringUtil::StrToFloatWithDefault(const char* str, float defaultValue) {
  float tmp;
  if (StrToFloat(str, tmp)) {
    return tmp;
  }
  return defaultValue;
}

double StringUtil::StrToDoubleWithDefault(const char* str,
                                          double defaultValue) {
  double tmp;
  if (StrToDouble(str, tmp)) {
    return tmp;
  }
  return defaultValue;
}

void StringUtil::ForWarning() {
  int32_t m_int32_t = 1;
  uint32_t m_uint32_t = 1;
  int64_t m_int64_t = 1;
  uint64_t m_uint64_t = 1;
  int m_int = 1;
  size_t m_size_t = 1;

  char buf[128] = {0};
  snprintf(buf, sizeof(buf), "%d", m_int32_t);
  snprintf(buf, sizeof(buf), "%u", m_uint32_t);
  snprintf(buf, sizeof(buf), "%ld", m_int64_t);
  snprintf(buf, sizeof(buf), "%lu", m_uint64_t);
  snprintf(buf, sizeof(buf), "%016lx", m_uint64_t);
  snprintf(buf, sizeof(buf), "%d", m_int);
  snprintf(buf, sizeof(buf), "%lu", m_size_t);
}

char* StringUtil::replicate(const char* szStr) {
  return replicate(szStr, strlen(szStr));
}

char* StringUtil::replicate(const char* szStr, uint32_t nLength) {
  if (szStr == NULL) return NULL;
  char* szRet = new char[nLength + 1];
  memcpy(szRet, szStr, nLength);
  szRet[nLength] = '\0';
  return szRet;
}

char* StringUtil::mergeString(const char* szStr1, const char* szStr2) {
  if (szStr1 == NULL && szStr2 == NULL) {
    return NULL;
  }

  if (szStr1 == NULL) {
    return replicate(szStr2);
  }

  if (szStr2 == NULL) {
    return replicate(szStr1);
  }

  uint32_t nLength1 = strlen(szStr1);
  uint32_t nLength2 = strlen(szStr2);
  char* szRet = new char[nLength1 + nLength2 + 1];
  memcpy(szRet, szStr1, nLength1);
  memcpy(szRet + nLength1, szStr2, nLength2);
  szRet[nLength1 + nLength2] = '\0';
  return szRet;
}

bool StringUtil::equal(const char* szStr1, const char* szStr2) {
  assert(szStr1 && szStr2);
  return strcmp(szStr1, szStr2) == 0;
}

bool StringUtil::safe_equal(const char* szStr1, const char* szStr2) {
  if (szStr1 == NULL || szStr2 == NULL) return false;
  return strcmp(szStr1, szStr2) == 0;
}

bool StringUtil::equalNoCase(const char* szStr1, const char* szStr2) {
  assert(szStr1 && szStr2);
#ifndef _WIN32
  return strcasecmp(szStr1, szStr2) == 0;
#else
  return stricmp(szStr1, szStr2) == 0;
#endif
}

bool StringUtil::isAscii(const char* str, size_t size) {
  const char* end = str + size;
  for (; str < end; ++str) {
    if (!isascii(*str)) return false;
  }
  return true;
}

bool StringUtil::isAscii(const char* str) {
  for (; *str != '\0'; ++str) {
    if (!isascii(*str)) return false;
  }
  return true;
}

std::string StringUtil::ToLower(const std::string& s) {
  std::string rc = s;
  std::transform(rc.begin(), rc.end(), rc.begin(), ::tolower);
  return rc;
}

std::string StringUtil::ToUpper(const std::string& s) {
  std::string rc = s;
  std::transform(rc.begin(), rc.end(), rc.begin(), ::toupper);
  return rc;
}

}  // namespace util
}  // namespace allspark
