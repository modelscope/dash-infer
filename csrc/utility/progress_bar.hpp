/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    progress_bar.hpp
 */
#include <chrono>
#include <iomanip>
#include <iostream>
#include <thread>

namespace allspark {

namespace util {

class ProgressBar {
 public:
  explicit ProgressBar(size_t total_iterations, size_t bar_width = 50)
      : total_iterations_(total_iterations),
        bar_width_(bar_width),
        start_time_(std::chrono::high_resolution_clock::now()) {}

  ~ProgressBar() { std::cout << "\n"; }

  void Update(size_t current_iteration) {
    // Calculate progress
    double progress =
        static_cast<double>(current_iteration) / total_iterations_;

    // Calculate ETA
    auto current_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::seconds>(
        current_time - start_time_);
    int eta_seconds = (elapsed_time.count() / progress) - elapsed_time.count();

    // Format progress bar
    int bar_position = static_cast<int>(progress * bar_width_);
    std::cout << "[";
    for (int i = 0; i < bar_width_; ++i) {
      if (i < bar_position) {
        std::cout << "#";
      } else {
        std::cout << " ";
      }
    }
    std::cout << "] " << std::fixed << std::setprecision(1) << progress * 100
              << "% ";

    // Format ETA
    if (eta_seconds > 0) {
      std::cout << "ETA: ";
      if (eta_seconds >= 3600) {
        std::cout << eta_seconds / 3600 << "h ";
        eta_seconds %= 3600;
      }
      if (eta_seconds >= 60) {
        std::cout << eta_seconds / 60 << "m ";
        eta_seconds %= 60;
      }
      std::cout << eta_seconds << "s";
    }

    std::cout << "\r" << std::flush;
  }

 private:
  size_t total_iterations_;
  size_t bar_width_;
  std::chrono::time_point<std::chrono::high_resolution_clock> start_time_;
};
}  // namespace util
}  // namespace allspark
