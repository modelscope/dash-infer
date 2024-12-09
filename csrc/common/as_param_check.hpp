/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    as_param_check.hpp
 */

#define MD5_DEBUG 0

#pragma once
#include <git_version.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <sstream>

#include "common/common.h"
#if MD5_DEBUG
#include <chrono>
#endif  // MD5_DEBUG

#ifndef MD5_DIGEST_LENGTH
#define MD5_DIGEST_LENGTH 16
#endif  // MD5_DIGEST_LENGTH

#define DBG_LOG LOG(INFO)
// #define DBG_LOG DLOG(INFO)
// #define DBG_LOG std::cout

#define ERR_LOG LOG(ERROR)
// #define ERR_LOG DLOG(INFO)
// #define ERR_LOG std::cout

namespace allspark {
using byte = unsigned char;
enum WeightCheckLevel {
  CHECKER_DISMISS = 0,
  CHECKER_NORMAL = 1,
  CHECKER_RESTRICT = 2,
  CHECKER_ENUM_TOTAL = 3,
};

// ENV USAGE: export HIE_PARAM_CHECK_LEVEL=2
#define CHECK_LEVEL_ENV "HIE_PARAM_CHECK_LEVEL"
const WeightCheckLevel default_level = CHECKER_NORMAL;

namespace md5_basic {
#define _F(x, y, z) (((x) & (y)) | ((~x) & (z)))
#define _G(x, y, z) (((x) & (z)) | ((y) & (~z)))
#define _H(x, y, z) ((x) ^ (y) ^ (z))
#define _I(x, y, z) ((y) ^ ((x) | (~z)))
#define ROTATELEFT(num, n) (((num) << (n)) | ((num) >> (32 - (n))))
#define FF(a, b, c, d, x, s, ac)         \
  {                                      \
    (a) += _F((b), (c), (d)) + (x) + ac; \
    (a) = ROTATELEFT((a), (s));          \
    (a) += (b);                          \
  }
#define GG(a, b, c, d, x, s, ac)         \
  {                                      \
    (a) += _G((b), (c), (d)) + (x) + ac; \
    (a) = ROTATELEFT((a), (s));          \
    (a) += (b);                          \
  }
#define HH(a, b, c, d, x, s, ac)         \
  {                                      \
    (a) += _H((b), (c), (d)) + (x) + ac; \
    (a) = ROTATELEFT((a), (s));          \
    (a) += (b);                          \
  }
#define II(a, b, c, d, x, s, ac)         \
  {                                      \
    (a) += _I((b), (c), (d)) + (x) + ac; \
    (a) = ROTATELEFT((a), (s));          \
    (a) += (b);                          \
  }

class MD5Impl {
 public:
  const byte* operator()(const byte* msg, size_t msg_len, byte* result) {
    count[0] = count[1] = 0;
    state[0] = 0x67452301;
    state[1] = 0xefcdab89;
    state[2] = 0x98badcfe;
    state[3] = 0x10325476;
    init(msg, msg_len);

    byte bits[8];
    uint32_t old_state[4];
    uint32_t old_count[2];
    uint32_t index = ((count[0] >> 3) & 0x3f);
    uint32_t pad_len = (index < 56) ? (56 - index) : (120 - index);
    static const byte padding[64] = {0x80};

    memcpy(old_state, state, 4 * sizeof(uint32_t));
    memcpy(old_count, count, 2 * sizeof(uint32_t));
    encode(count, bits, 8);

    init(padding, pad_len);
    init(bits, 8);
    encode(state, digest, MD5_DIGEST_LENGTH);
    memcpy(state, old_state, 4 * sizeof(uint32_t));
    memcpy(count, old_count, 2 * sizeof(uint32_t));
    memcpy(result, digest, MD5_DIGEST_LENGTH);
    return result;
  }

 private:
  void init(const byte* input, size_t len) {
    uint32_t i, part;
    uint32_t index = ((count[0] >> 3) & 0x3f);

    if ((count[0] += ((uint32_t)len << 3)) < ((uint32_t)len << 3)) {
      ++count[1];
    }
    count[1] += ((uint32_t)len >> 29);

    part = 64 - index;
    if (len >= part) {
      memcpy(&buffer[index], input, part);
      transform(buffer);
      for (i = part; i + 63 < len; i += 64) {
        transform(&input[i]);
      }
      index = 0;
    } else {
      i = 0;
    }

    memcpy(&buffer[index], &input[i], len - i);
  }

  void transform(const byte block[64]) {
    uint32_t a = state[0], b = state[1], c = state[2], d = state[3], x[16];
    decode(block, x, 64);

    // clang-format off
        /* Round 1 */
        FF(a, b, c, d, x[ 0],  7, 0xd76aa478);
        FF(d, a, b, c, x[ 1], 12, 0xe8c7b756);
        FF(c, d, a, b, x[ 2], 17, 0x242070db);
        FF(b, c, d, a, x[ 3], 22, 0xc1bdceee);
        FF(a, b, c, d, x[ 4],  7, 0xf57c0faf);
        FF(d, a, b, c, x[ 5], 12, 0x4787c62a);
        FF(c, d, a, b, x[ 6], 17, 0xa8304613);
        FF(b, c, d, a, x[ 7], 22, 0xfd469501);
        FF(a, b, c, d, x[ 8],  7, 0x698098d8);
        FF(d, a, b, c, x[ 9], 12, 0x8b44f7af);
        FF(c, d, a, b, x[10], 17, 0xffff5bb1);
        FF(b, c, d, a, x[11], 22, 0x895cd7be);
        FF(a, b, c, d, x[12],  7, 0x6b901122);
        FF(d, a, b, c, x[13], 12, 0xfd987193);
        FF(c, d, a, b, x[14], 17, 0xa679438e);
        FF(b, c, d, a, x[15], 22, 0x49b40821);

        /* Round 2 */
        GG(a, b, c, d, x[ 1],  5, 0xf61e2562);
        GG(d, a, b, c, x[ 6],  9, 0xc040b340);
        GG(c, d, a, b, x[11], 14, 0x265e5a51);
        GG(b, c, d, a, x[ 0], 20, 0xe9b6c7aa);
        GG(a, b, c, d, x[ 5],  5, 0xd62f105d);
        GG(d, a, b, c, x[10],  9,  0x2441453);
        GG(c, d, a, b, x[15], 14, 0xd8a1e681);
        GG(b, c, d, a, x[ 4], 20, 0xe7d3fbc8);
        GG(a, b, c, d, x[ 9],  5, 0x21e1cde6);
        GG(d, a, b, c, x[14],  9, 0xc33707d6);
        GG(c, d, a, b, x[ 3], 14, 0xf4d50d87);
        GG(b, c, d, a, x[ 8], 20, 0x455a14ed);
        GG(a, b, c, d, x[13],  5, 0xa9e3e905);
        GG(d, a, b, c, x[ 2],  9, 0xfcefa3f8);
        GG(c, d, a, b, x[ 7], 14, 0x676f02d9);
        GG(b, c, d, a, x[12], 20, 0x8d2a4c8a);

        /* Round 3 */
        HH(a, b, c, d, x[ 5],  4, 0xfffa3942);
        HH(d, a, b, c, x[ 8], 11, 0x8771f681);
        HH(c, d, a, b, x[11], 16, 0x6d9d6122);
        HH(b, c, d, a, x[14], 23, 0xfde5380c);
        HH(a, b, c, d, x[ 1],  4, 0xa4beea44);
        HH(d, a, b, c, x[ 4], 11, 0x4bdecfa9);
        HH(c, d, a, b, x[ 7], 16, 0xf6bb4b60);
        HH(b, c, d, a, x[10], 23, 0xbebfbc70);
        HH(a, b, c, d, x[13],  4, 0x289b7ec6);
        HH(d, a, b, c, x[ 0], 11, 0xeaa127fa);
        HH(c, d, a, b, x[ 3], 16, 0xd4ef3085);
        HH(b, c, d, a, x[ 6], 23,  0x4881d05);
        HH(a, b, c, d, x[ 9],  4, 0xd9d4d039);
        HH(d, a, b, c, x[12], 11, 0xe6db99e5);
        HH(c, d, a, b, x[15], 16, 0x1fa27cf8);
        HH(b, c, d, a, x[ 2], 23, 0xc4ac5665);

        /* Round 4 */
        II(a, b, c, d, x[ 0],  6, 0xf4292244);
        II(d, a, b, c, x[ 7], 10, 0x432aff97);
        II(c, d, a, b, x[14], 15, 0xab9423a7);
        II(b, c, d, a, x[ 5], 21, 0xfc93a039);
        II(a, b, c, d, x[12],  6, 0x655b59c3);
        II(d, a, b, c, x[ 3], 10, 0x8f0ccc92);
        II(c, d, a, b, x[10], 15, 0xffeff47d);
        II(b, c, d, a, x[ 1], 21, 0x85845dd1);
        II(a, b, c, d, x[ 8],  6, 0x6fa87e4f);
        II(d, a, b, c, x[15], 10, 0xfe2ce6e0);
        II(c, d, a, b, x[ 6], 15, 0xa3014314);
        II(b, c, d, a, x[13], 21, 0x4e0811a1);
        II(a, b, c, d, x[ 4],  6, 0xf7537e82);
        II(d, a, b, c, x[11], 10, 0xbd3af235);
        II(c, d, a, b, x[ 2], 15, 0x2ad7d2bb);
        II(b, c, d, a, x[ 9], 21, 0xeb86d391);
    // clang-format on

    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
  }

  void encode(const uint32_t* input, byte* output, size_t length) {
    for (size_t i = 0, j = 0; j < length; ++i, j += 4) {
      output[j] = (byte)(input[i] & 0xff);
      output[j + 1] = (byte)((input[i] >> 8) & 0xff);
      output[j + 2] = (byte)((input[i] >> 16) & 0xff);
      output[j + 3] = (byte)((input[i] >> 24) & 0xff);
    }
  }

  void decode(const byte* input, uint32_t* output, size_t length) {
    for (size_t i = 0, j = 0; j < length; ++i, j += 4) {
      output[i] = ((uint32_t)input[j]) | (((uint32_t)input[j + 1]) << 8) |
                  (((uint32_t)input[j + 2]) << 16) |
                  (((uint32_t)input[j + 3]) << 24);
    }
  }

 private:
  uint32_t state[4];  // ABCD
  uint32_t count[2];
  byte buffer[64];
  byte digest[MD5_DIGEST_LENGTH];
};

#undef ROTATELEFT
#undef _F
#undef _G
#undef _H
#undef _I
#undef FF
#undef GG
#undef HH
#undef II
}  // namespace md5_basic

class AsParamGuard {
 public:
  std::vector<std::string> weight_hash;
  std::map<std::string, std::string> torch_build_config;

 private:
  std::string prefix_log = "AsParamGuard check level = ";
  std::string engine_ver = "";
  std::string weight_ver = "";
  std::string basic_info = "";
  std::string epilog_err = "";

 public:
  static WeightCheckLevel curr_check_level() {
    const char* env_char = std::getenv(CHECK_LEVEL_ENV);
    const char* deprecated = std::getenv("AS_PARAM_CEHCK_LEVEL");  // prev typo
    int env_int_level = static_cast<int>(default_level);
    if (env_char) {
      env_int_level = atoi(env_char);
    } else if (deprecated) {
      DBG_LOG << "AsParamGuard, environment AS_PARAM_CEHCK_LEVEL is already "
                 "deprecated. please use "
              << CHECK_LEVEL_ENV << " instead. ";
      env_int_level = atoi(deprecated);
    }
    if (env_int_level >=
            static_cast<int>(WeightCheckLevel::CHECKER_ENUM_TOTAL) ||
        env_int_level < 0)
      env_int_level = static_cast<int>(default_level);
    return static_cast<WeightCheckLevel>(env_int_level);
  }

  void append_weight_md5(const char* weight, size_t hash_size) {
    WeightCheckLevel level = AsParamGuard::curr_check_level();
    auto md5_byte2str = [](byte(&md5)[MD5_DIGEST_LENGTH]) -> std::string {
      static const char hex_index[16 + 1] = "0123456789abcdef";
      char convert_str[2 * MD5_DIGEST_LENGTH + 1];
      for (int i = 0; i < MD5_DIGEST_LENGTH; i++) {
        convert_str[i * 2] = hex_index[(md5[i] >> 4) & 0xF];
        convert_str[i * 2 + 1] = hex_index[(md5[i]) & 0xF];
      }
      convert_str[MD5_DIGEST_LENGTH * 2] = '\0';
      return std::string(convert_str);
    };

    if (level == WeightCheckLevel::CHECKER_NORMAL ||
        level == WeightCheckLevel::CHECKER_RESTRICT) {
#if MD5_DEBUG
      auto time0 = std::chrono::high_resolution_clock::now();
#endif

      byte md5_buffer[MD5_DIGEST_LENGTH];
      md5_basic::MD5Impl()((byte*)weight, hash_size,
                           md5_buffer);  // same as openssl/md5 MD5();
      std::string md5_string = md5_byte2str(md5_buffer);
      weight_hash.push_back(md5_string);

#if MD5_DEBUG
      auto time1 = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapse = time1 - time0;
      std::cout << "calculate md5 checksum is: " << md5_string << ", "
                << "\telapse time = " << elapse.count() << ", "
                << "\twith checksum size = " << hash_size / 1024 / 1024 << "M."
                << std::endl;
#endif  // MD5_DEBUG
    }
    return;
  }

  AsStatus operator()(const BuildMetaProto& graph_meta) {
    WeightCheckLevel level = AsParamGuard::curr_check_level();
    bool flag = true;
    switch (level) {
      case WeightCheckLevel::CHECKER_DISMISS:
        flag = check_dismiss(graph_meta);
        break;
      case WeightCheckLevel::CHECKER_RESTRICT:
        flag = check_restrict(graph_meta);
        break;
      case WeightCheckLevel::CHECKER_NORMAL:
      default:
        flag = check_normal(graph_meta);
        break;
    }
    // log here
    std::string version_log = "";
    if (engine_ver.length() > 0)
      version_log += "Engine version = " + engine_ver + " . ";
    if (weight_ver.length() > 0)
      version_log += "Weight version = " + weight_ver + " . ";
    if (flag) {
      DBG_LOG << prefix_log << AsParamGuard::toString(level) << ". "
              << version_log << basic_info << std::endl;
    } else {
      ERR_LOG << prefix_log << AsParamGuard::toString(level) << ". "
              << version_log << basic_info
              << "Param-Guard check failure: " << epilog_err << std::endl;
      DBG_LOG << "You can set Param-Guard check-level in os-environment "
              << CHECK_LEVEL_ENV << ", "
              << "by using int-value levels: CHECKER_DISMISS (0), "
                 "CHECKER_NORMAL (1) and CHECKER_RESTRICT(2). "
              << "For example: export " << CHECK_LEVEL_ENV
              << "=0, to skip Param-Guard. " << std::endl;
    }
    return flag ? AsStatus::ALLSPARK_SUCCESS : AsStatus::ALLSPARK_PARAM_ERROR;
  }

  bool check_dismiss(const BuildMetaProto& graph_meta) {
    return true;  // always true.
  }

  bool check_normal(const BuildMetaProto& graph_meta) {
    return check_version_major(graph_meta) && check_version_minor(graph_meta) &&
           check_weight_hash(graph_meta) &&
           check_torch_build_config(graph_meta, "multigpu_mode");
  }

  bool check_restrict(const BuildMetaProto& graph_meta) {
    return check_version_major(graph_meta) && check_version_minor(graph_meta) &&
           check_version_patch(graph_meta) &&
           check_version_git_commit(graph_meta) &&
           check_weight_hash(graph_meta) &&
           check_torch_build_config(graph_meta, "model_name") &&
           check_torch_build_config(graph_meta, "multigpu_mode");
    // && check_torch_build_config(graph_meta, "derive_type")
  }

  bool get_version(const BuildMetaProto& graph_meta, std::string& version_str) {
    if (graph_meta.has_version()) {
      char buf[255];
      snprintf(buf, 255, "%d.%d.%d", graph_meta.version().major(),
               graph_meta.version().minor(), graph_meta.version().patch());
      version_str = buf;

      return true;
    }
    return false;
  }

 private:
#define ASWeightLevelToString(LEVEL) \
  case LEVEL:                        \
    return #LEVEL;
  static std::string toString(WeightCheckLevel level) {
    switch (level) {
      ASWeightLevelToString(CHECKER_DISMISS);
      ASWeightLevelToString(CHECKER_NORMAL);
      ASWeightLevelToString(CHECKER_RESTRICT);
      default:
        return "Invalid Checker Level";
    }
  }
#undef ASWeightLevelToString

 private:
  bool check_version_major(const BuildMetaProto& graph_meta) {
    if (graph_meta.has_version()) {
      engine_ver += std::string(ALLSPARK_VERSION_MAJOR);
      weight_ver += std::to_string(graph_meta.version().major());
      if (graph_meta.version().major() == std::stoi(ALLSPARK_VERSION_MAJOR)) {
        return true;
      } else {
        epilog_err += "Weight version check failure, version-major not match. ";
        return false;
      }
    } else {
      epilog_err +=
          "Weight version check failure, version-major required but not "
          "found. ";
      return false;
    }
  }

  bool check_version_minor(const BuildMetaProto& graph_meta) {
    if (graph_meta.has_version()) {
      engine_ver += "." + std::string(ALLSPARK_VERSION_MINOR);
      weight_ver += "." + std::to_string(graph_meta.version().minor());
      if (graph_meta.version().minor() == std::stoi(ALLSPARK_VERSION_MINOR)) {
        return true;
      } else {
        epilog_err += "Weight version check failure, version-minor not match. ";
        return false;
      }
    } else {
      epilog_err +=
          "Weight version check failure, version-minor required but not "
          "found. ";
      return false;
    }
  }

  bool check_version_patch(const BuildMetaProto& graph_meta) {
    if (graph_meta.has_version()) {
      engine_ver += "." + std::string(ALLSPARK_VERSION_PATCH);
      weight_ver += "." + std::to_string(graph_meta.version().patch());
      if (graph_meta.version().patch() == std::stoi(ALLSPARK_VERSION_PATCH)) {
        return true;
      } else {
        epilog_err += "Weight version check failure, version-patch not match. ";
        return false;
      }
    } else {
      epilog_err +=
          "Weight version check failure, version-patch required but not "
          "found. ";
      return false;
    }
  }

  bool check_version_git_commit(const BuildMetaProto& graph_meta) {
    if (graph_meta.has_version() &&
        graph_meta.version().git_commit().length() != 0) {
      std::string git_commit_info =
          "Allspark-Engine built commit = " + std::string(kGitHash) + ", " +
          "weights generate by commit = " + graph_meta.version().git_commit() +
          ". ";
      basic_info += git_commit_info;
      std::string param = graph_meta.version().git_commit();
      std::string local = std::string(kGitHash);
      int min_length = std::min(param.length(), local.length());
      if (param.substr(0, min_length) == local.substr(0, min_length)) {
        return true;
      } else {
        epilog_err +=
            "Engine build commit check failure, commit hash mismatch. ";
        return false;
      }
    } else {
      epilog_err +=
          "Engine build commit check failure, commit hash not exist. ";
      return false;
    }
  }

  bool check_weight_hash(const BuildMetaProto& graph_meta) {
    if (weight_hash.size() == 0) return true;  // no hash exist, skip check.
    if (graph_meta.has_weight_hash()) {
      int32_t hash_record_num = graph_meta.weight_hash().hash_size();
      int32_t hash_length_num = graph_meta.weight_hash().hash_length_size();
      std::string param_hash_list_for_log = "";
      std::string local_hash_list_for_log = "";
      for (int i = 0; i < hash_record_num; i++) {
        param_hash_list_for_log += graph_meta.weight_hash().hash(i);
        param_hash_list_for_log += "\t";
      }
      for (int i = 0; i < weight_hash.size(); i++) {
        local_hash_list_for_log += weight_hash[i];
        local_hash_list_for_log += "\t";
      }
      std::string hash_check_info =
          "Weight checksum use algorithm [" +
          graph_meta.weight_hash().algorithm() + "]. Incoming " +
          std::to_string(graph_meta.weight_hash().hash_size()) +
          " record(s) in allspark-params: " + param_hash_list_for_log +
          ". Load weights are splited in " +
          std::to_string(weight_hash.size()) +
          " part(s) with checksum results: " + local_hash_list_for_log + ". ";
      // check hash num
      if (hash_record_num != hash_length_num ||
          hash_record_num != weight_hash.size()) {
        epilog_err += "Weight checksum failure, hash records number mismatch. ";
        return false;
      }
      // check hash
      for (int i = 0; i < hash_record_num; i++) {
        std::string incoming_hash(graph_meta.weight_hash().hash(i));
        std::transform(incoming_hash.begin(), incoming_hash.end(),
                       incoming_hash.begin(),
                       [](unsigned char c) { return std::tolower(c); });
        if (incoming_hash != weight_hash[i]) {
          std::string hash_mismatch_log =
              "Weight checksum failure, hash record[" + std::to_string(i) +
              "] mismatch. ";
          epilog_err += hash_mismatch_log;
          return false;
        }
      }
      return true;
    } else {
      epilog_err += "Weight checksum failure, hash records not found. ";
      return false;
    }
  }

  bool check_torch_build_config(const BuildMetaProto& graph_meta,
                                const std::string& key) {
    auto param = graph_meta.torch_build_config().find(key);
    auto local = torch_build_config.find(key);
    if (local == torch_build_config.end())
      return true;  // key not exist in current build. skip.
    if (param != graph_meta.torch_build_config().end()) {
      std::string build_config_info =
          "Engine init with config " + key + ": " + std::string(local->second) +
          ", and load Allspark build config " + key + ": " +
          std::string(param->second) + " from weight. ";
      if (param->second == local->second) {
        return true;
      } else {
        epilog_err +=
            "Engine build config check failure, key " + key + " mismatch. ";
        return false;
      }
    } else {
      epilog_err += "Engine build config check failure, key " + key +
                    " not found in allspark-params. ";
      return false;
    }
  }
};

}  // namespace allspark

#undef DBG_LOG
#undef ERR_LOG
#undef CHECK_LEVEL_ENV
