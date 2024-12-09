/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    uuid.cpp
 */

#include "uuid.h"

#include <algorithm>
#include <atomic>
#include <climits>
#include <functional>
#include <iostream>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#define UUID_SEQ_DEBUG 1  // use 0,1,2,3... sequence instead uuid

namespace allspark {

#if defined(UUID_SEQ_DEBUG) && UUID_SEQ_DEBUG

// Use std::atomic<uint32_t> to ensure thread-safety of uuid_cnt
static std::atomic<uint32_t> uuid_cnt{0};

// Helper function: convert uint32_t to a numeric string with specified width,
// padding with leading zeros
inline std::string UInt32ToString(uint32_t value, int width) {
  char buffer[11];  // 10 digits + null terminator
  std::snprintf(buffer, sizeof(buffer), "%010u", value);
  std::string str(buffer);

  // Calculate the number of leading zeros needed
  int padding = width - static_cast<int>(str.size());
  if (padding > 0) {
    return std::string(padding, '0') + str;
  } else if (padding < 0) {
    // If the number of digits exceeds the specified width, truncate the high
    // digits (should not happen)
    return str.substr(str.size() - width, width);
  } else {
    return str;
  }
}

std::string GenNewUUID() {
  // Atomically increment counter and get the value before incrementing
  uint32_t cnt = uuid_cnt.fetch_add(1, std::memory_order_relaxed);

  // Define the maximum value as 4,294,967,295 (maximum of uint32_t)
  const uint32_t max_val = std::numeric_limits<uint32_t>::max();

  // Check if the counter has reached the overflow point
  if (cnt == max_val) {
    // Attempt to reset the counter to 0
    uint32_t expected = cnt;
    bool success = uuid_cnt.compare_exchange_strong(expected, 0,
                                                    std::memory_order_relaxed);
    if (!success) {
      // If CAS fails, it means other threads have modified uuid_cnt
      // Continue using the current overflow value, or take other actions as
      // needed
      printf("uuid overflow...");
    }
  }

  // Convert the counter to a 32-digit numeric string, pad with 22 leading zeros
  // and 10 digits from the counter
  std::string uuid_str = UInt32ToString(cnt, 32);
  return uuid_str;
}

#else

// Non-debug mode implementation, assuming generate_hex is thread-safe
std::string generate_hex(int length) {
  static const char hex_chars[] = "0123456789ABCDEF";
  std::string hex;
  hex.reserve(length);
  for (int i = 0; i < length; ++i) {
    hex += hex_chars[rand() % 16];
  }
  return hex;
}

std::string GenNewUUID() {
  // Generate a 32-digit hexadecimal string from the counter value
  return generate_hex(32);
}

#endif

}  // namespace allspark
