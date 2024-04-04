/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    allspark_logging.cpp
 */
#include "allspark_logging.h"

#include <stdlib.h>

#include <mutex>

namespace allspark {
namespace util {

static std::once_flag g_log_init_once;

void as_init_log() {
  std::call_once(g_log_init_once, []() {
    google::InitGoogleLogging("hie_allspark");
    google::InstallFailureSignalHandler();
    google::EnableLogCleaner(3);

    fLB::FLAGS_timestamp_in_logfile_name = true;
    fLB::FLAGS_alsologtostderr = false;
    fLI::FLAGS_stderrthreshold = google::ERROR;
    fLI::FLAGS_logbuflevel = google::WARNING;
    fLI::FLAGS_logbufsecs = 5;
    fLI::FLAGS_max_log_size = 10;

    const char* log_dir = std::getenv("HIE_LOG_DIR");
    if (not log_dir or std::string(log_dir) == "") {
      fLB::FLAGS_logtostderr = true;
    } else {
      fLS::FLAGS_log_dir = log_dir;
      fLB::FLAGS_logtostderr = false;
    }

    const char* log_level_str = std::getenv("HIE_LOG_LEVEL");
    int log_level = 0;
    if (log_level_str) {
      log_level = atoi(log_level_str);
      log_level = (google::INFO <= log_level and log_level <= google::FATAL)
                      ? log_level
                      : 0;
    }
    fLI::FLAGS_minloglevel = log_level;
  });
}

}  // namespace util
}  // namespace allspark
