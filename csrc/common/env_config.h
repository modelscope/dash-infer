/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    env_config.h
 */

#pragma once

#include <cstdlib>  // for getenv
#include <iostream>
#include <stdexcept>
#include <string>

namespace allspark {

class EnvVarConfig {
 public:
  /**
   * @brief Gets the string value of an environment variable, or returns a
   * default value if not set.
   *
   * @param varName The name of the environment variable.
   * @param defaultValue The default value to return if the environment variable
   * is not set.
   * @return std::string The value of the environment variable, or the default
   * value if not set.
   */
  static std::string GetString(const std::string& varName,
                               const std::string& defaultValue) {
    const char* val = std::getenv(varName.c_str());
    return (val == nullptr) ? defaultValue : std::string(val);
  }

  /**
   * @brief Gets the integer value of an environment variable, or returns a
   * default value if not set or conversion fails.
   *
   * @param varName The name of the environment variable.
   * @param defaultValue The default value to return if the environment variable
   * is not set or conversion fails.
   * @return int The integer value of the environment variable, or the default
   * value if not set or conversion fails.
   */
  static int GetInt(const std::string& varName, int defaultValue) {
    const char* val = std::getenv(varName.c_str());
    if (val == nullptr) {
      // Environment variable is not set, return the default value
      return defaultValue;
    }
    try {
      // Attempt to convert the string to an integer
      return std::stoi(val);
    } catch (const std::invalid_argument& e) {
      // Conversion failed (invalid input), return the default value
      return defaultValue;
    } catch (const std::out_of_range& e) {
      // Conversion failed (out of range), return the default value
      return defaultValue;
    }
  }
};

class AttentionEnvConfig {
 public:
  static int GetFlashThresh() {
    static int env_flash_thresh = -1;
    if (env_flash_thresh == -1) {
      env_flash_thresh = EnvVarConfig::GetInt("AS_FLASH_THRESH", 1024);
    }
    return env_flash_thresh;
  }
};

}  // namespace allspark
