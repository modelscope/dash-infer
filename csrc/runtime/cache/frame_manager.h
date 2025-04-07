/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    frame_manager.h
 */

#pragma once

#include <atomic>
#include <memory>
#include <queue>

#include "common/common.h"
#include "common/device_context.h"
#include "utility/concurrentqueue.h"

namespace allspark {

/**
 * @brief A cache frame in the device.
 *
 * The storage is automatically managed.
 */
class CacheFrame {
 public:
  using Ptr = std::shared_ptr<CacheFrame>;
  using ConstPtr = std::shared_ptr<const CacheFrame>;

  /// @brief Create a new frame.
  /// @return The shared pointer to the frame.
  static Ptr Create(DeviceType device_type, int64_t frame_size,
                    const std::string& tag);
  /// @brief Get the pointer.
  void* Data();
  /// @brief Get the size of the frame.
  int64_t Size();
  /// @brief Get the device type of the frame.
  DeviceType GetDeviceType();

  virtual ~CacheFrame() = default;

 protected:
  CacheFrame() {}

 private:
  CacheFrame(const CacheFrame&) = delete;
  CacheFrame(CacheFrame&&) = delete;
};

/**
 * @brief The cache frame manager for one model.
 *
 * Generally, all requests for one model use the same cache config.
 * So each model has its own cache frame manager.
 * Construct the manager when building the model, at which time the device is
 * determined.
 * Initialize the manager in the first call to Reshape(), at which time the
 * frame size are determined.
 */
class CacheFrameManager {
 public:
  using Ptr = std::shared_ptr<CacheFrameManager>;
  using ConstPtr = std::shared_ptr<const CacheFrameManager>;

  explicit CacheFrameManager(DeviceType device_type)
      : device_type_(device_type), initialized_(false) {}
  virtual ~CacheFrameManager() = default;

  bool Inited() const { return initialized_; }
  /// @brief Initialize the manager in the first call to Reshape().
  virtual void Init(int64_t frame_size) = 0;

  /// @brief Grow the number of managed frames until reaching num_frames.
  /// @param num_frames The total number of frames (including free and
  /// occupied) after growth, 0 means growing until no more available memory.
  /// @return true if all required frames claimed.
  /// @return false if failed to claim all required frames.
  virtual bool GrowUntil(size_t num_frames) = 0;
  /// @brief Grow the number of managed frames by num_frames.
  /// @param num_frames The number of frames to grow.
  /// @return true if all required frames claimed.
  /// @return false if failed to claim all required frames.
  virtual bool GrowBy(size_t num_frames) = 0;

  /// @brief Allocate one frame.
  /// @return Shared pointr to the frame, NULL if failed.
  virtual CacheFrame::Ptr AllocFrame() = 0;
  /// @brief Allocate multiple frames.
  /// @return number of allocated frames.
  virtual size_t AllocFrame(std::vector<CacheFrame::Ptr>& out_vec,
                            size_t count) = 0;
  /// @brief Free the frame.
  virtual void FreeFrame(CacheFrame::Ptr frame) = 0;
  /// @brief Free multiple frames.
  virtual void FreeFrame(std::vector<CacheFrame::Ptr>& frame_vec,
                         size_t count) = 0;

  /// @brief Return the number of free frames.
  virtual size_t CountFreeFrame() const = 0;
  /// @brief Return the number of occupied frames.
  virtual size_t CountOccupiedFrame() const = 0;
  /// @brief Return the number of all frames.
  virtual size_t CountFrame() const = 0;

  virtual size_t PresFrame(size_t count) = 0;
  virtual size_t AllocFrameFromPres(std::vector<CacheFrame::Ptr>& out_vec,
                                    size_t count) = 0;
  virtual void FreeFrameToPres(std::vector<CacheFrame::Ptr>& frame_vec,
                               size_t count) = 0;
  virtual size_t FreePresFrame(size_t count) = 0;
  virtual size_t CountPresFrame() const = 0;

  DeviceType GetDeviceType() const { return device_type_; }
  virtual int64_t GetFrameSize() const = 0;
  virtual int GetInitFrameNum() const = 0;
  virtual int GetGrowFrameNum() const = 0;

 protected:
  const DeviceType device_type_;
  bool initialized_;
};

class DefaultCacheFrameManager : public CacheFrameManager {
 public:
  explicit DefaultCacheFrameManager(DeviceType device_type, int init_frames,
                                    int grow_frames)
      : CacheFrameManager(device_type),
        frame_size_(0),
        init_frames_(init_frames),
        grow_frames_(grow_frames) {}

  void Init(int64_t frame_size) override;
  bool GrowUntil(size_t num_frames) override;
  bool GrowBy(size_t num_frames) override;

  CacheFrame::Ptr AllocFrame() override;
  size_t AllocFrame(std::vector<CacheFrame::Ptr>& out_vec,
                    size_t count) override;
  void FreeFrame(CacheFrame::Ptr frame) override;
  void FreeFrame(std::vector<CacheFrame::Ptr>& frame_vec,
                 size_t count) override;

  size_t CountFreeFrame() const override;
  size_t CountOccupiedFrame() const override;
  size_t CountFrame() const override;

  size_t PresFrame(size_t count) override;
  size_t AllocFrameFromPres(std::vector<CacheFrame::Ptr>& out_vec,
                            size_t count) override;
  void FreeFrameToPres(std::vector<CacheFrame::Ptr>& frame_vec,
                       size_t count) override;
  size_t FreePresFrame(size_t count) override;
  size_t CountPresFrame() const override;

  int64_t GetFrameSize() const override { return frame_size_; };
  int GetInitFrameNum() const override { return init_frames_; };
  int GetGrowFrameNum() const override { return grow_frames_; };

 private:
  std::unordered_map<void*, CacheFrame::Ptr> occupied_frames_;
  std::queue<CacheFrame::Ptr> free_frames_;
  int64_t frame_size_;
  const int init_frames_;
  const int grow_frames_;

  bool createFrame();
};

class ConcurrentCacheFrameManager : public CacheFrameManager {
 public:
  explicit ConcurrentCacheFrameManager(DeviceType device_type, int init_frames)
      : CacheFrameManager(device_type),
        free_frames_(init_frames),
        free_frame_count_(0),
        pres_frame_count_(0),
        total_frame_count_(0),
        frame_size_(0),
        init_frames_(init_frames) {}

  void Init(int64_t frame_size) override;
  bool GrowUntil(size_t num_frames) override;
  bool GrowBy(size_t num_frames) override;

  CacheFrame::Ptr AllocFrame() override;
  size_t AllocFrame(std::vector<CacheFrame::Ptr>& out_vec,
                    size_t count) override;
  void FreeFrame(CacheFrame::Ptr frame) override;
  void FreeFrame(std::vector<CacheFrame::Ptr>& frame_vec,
                 size_t count) override;

  size_t CountFreeFrame() const override;
  size_t CountOccupiedFrame() const override;
  size_t CountFrame() const override;

  size_t PresFrame(size_t count) override;
  size_t AllocFrameFromPres(std::vector<CacheFrame::Ptr>& out_vec,
                            size_t count) override;
  void FreeFrameToPres(std::vector<CacheFrame::Ptr>& frame_vec,
                       size_t count) override;
  size_t FreePresFrame(size_t count) override;
  size_t CountPresFrame() const override;

  int64_t GetFrameSize() const override { return frame_size_; };
  int GetInitFrameNum() const override { return init_frames_; };
  int GetGrowFrameNum() const override { return 0; };

 private:
  moodycamel::ConcurrentQueue<CacheFrame::Ptr> free_frames_;
  moodycamel::ConcurrentQueue<CacheFrame::Ptr> pres_frames_;
  std::atomic_int64_t free_frame_count_;
  std::atomic_int64_t pres_frame_count_;
  int64_t total_frame_count_;
  int64_t frame_size_;
  const int init_frames_;

  bool createFrame();
};

}  // namespace allspark
