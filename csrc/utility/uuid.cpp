/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    uuid.cpp
 */
#include "uuid.h"

#include <algorithm>
#include <climits>
#include <functional>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#define UUID_SEQ_DEBUG \
  1  // use 0,1,2,3... sequence instead uuid, for DEBUG ONLY!

namespace allspark {

static unsigned char random_char() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, 255);
  return static_cast<unsigned char>(dis(gen));
}

static std::string generate_hex(const unsigned int len) {
  std::stringstream ss;
  for (auto i = 0; i < len; i++) {
    auto rc = random_char();
    std::stringstream hexstream;
    hexstream << std::hex << int(rc);
    auto hex = hexstream.str();
    ss << (hex.length() < 2 ? '0' + hex : hex);
  }
  return ss.str();
}

std::string GenNewUUID() {
#if UUID_SEQ_DEBUG
  char buf[32];
  static long uuid_cnt = 0;
  snprintf(buf, 32, "%031ld", uuid_cnt++);
  return std::string(buf);
#else
  return generate_hex(32);
#endif
}

}  // namespace allspark
