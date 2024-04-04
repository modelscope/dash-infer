/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    string_util.h
 */

#pragma once

#include <common/common.h>
#include <stdint.h>

#include <iostream>
#include <string>
#include <vector>

namespace allspark {
namespace util {

void split(std::vector<std::string>& out, std::string& str, std::string delim);

class StringUtil {
 public:
  StringUtil();
  ~StringUtil();

 public:
  static void Trim(std::string& str);
  static char* Trim(const char* szStr);

  static bool StartsWith(const std::string& value,
                         const std::string& starting) {
    return value.rfind(starting, 0) == 0;
  }

  static bool EndsWith(const std::string& value, const std::string& ending) {
    if (ending.size() > value.size()) return false;
    return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
  }

  static std::vector<std::string> Split(const std::string& text,
                                        const char* sepStr);
  static std::vector<std::string> Split(const char* text, const char* sepStr);

  // split with multiple char delimiters
  static std::vector<std::string> Split2(const std::string& s,
                                         const std::string& delimiters);

  static bool StrToInt8(const char* str, int8_t& value);
  static bool StrToUInt8(const char* str, uint8_t& value);
  static bool StrToInt16(const char* str, int16_t& value);
  static bool StrToUInt16(const char* str, uint16_t& value);
  static bool StrToInt32(const char* str, int32_t& value);
  static bool StrToUInt32(const char* str, uint32_t& value);
  static bool StrToInt64(const char* str, int64_t& value);
  static bool StrToUInt64(const char* str, uint64_t& value);
  static bool StrToFloat(const char* str, float& value);
  static bool StrToDouble(const char* str, double& value);
  static bool HexStrToUint64(const char* str, uint64_t& value);
  static void Uint64ToHexStr(uint64_t value, char* hexStr, int len);

  static uint32_t DeserializeUInt32(const std::string& str);
  static void SerializeUInt32(uint32_t value, std::string& str);

  static uint64_t DeserializeUInt64(const std::string& str);
  static void SerializeUInt64(uint64_t value, std::string& str);

  static int8_t StrToInt8WithDefault(const char* str, int8_t defaultValue);
  static uint8_t StrToUInt8WithDefault(const char* str, uint8_t defaultValue);
  static int16_t StrToInt16WithDefault(const char* str, int16_t defaultValue);
  static uint16_t StrToUInt16WithDefault(const char* str,
                                         uint16_t defaultValue);
  static int32_t StrToInt32WithDefault(const char* str, int32_t defaultValue);
  static uint32_t StrToUInt32WithDefault(const char* str,
                                         uint32_t defaultValue);
  static int64_t StrToInt64WithDefault(const char* str, int64_t defaultValue);
  static uint64_t StrToUInt64WithDefault(const char* str,
                                         uint64_t defaultValue);
  static float StrToFloatWithDefault(const char* str, float defaultValue);
  static double StrToDoubleWithDefault(const char* str, double defaultValue);

  static char* replicate(const char* szStr);
  static char* replicate(const char* szStr, uint32_t nLength);

  static char* mergeString(const char* szStr1, const char* szStr2);

  static void ForWarning();

  static bool equal(const char* szStr1, const char* szStr2);
  static bool safe_equal(const char* szStr1, const char* szStr2);
  static bool equalNoCase(const char* szStr1, const char* szStr2);

  static bool isAscii(const char* str, size_t size);
  static bool isAscii(const char* str);

  static std::string ToLower(const std::string& s);
  static std::string ToUpper(const std::string& s);
};

namespace notstd {
template <class T, bool is_enum_type>
struct adl_helper {
  std::string as_string(T&& t) {
    using std::to_string;
    return to_string(std::forward<T>(t));
  }
};

template <class T>
struct adl_helper<T, true> {
  std::string as_string(T&& t) { return to_string(std::forward<T>(t)); }
};

template <class T>
std::string to_string(T&& t) {
  return adl_helper<
             T, std::is_enum<typename std::remove_reference<T>::type>::value>()
      .as_string(std::forward<T>(t));
}
}  // namespace notstd

}  // namespace util
}  // namespace allspark
