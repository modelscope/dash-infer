/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    span_manager.h
 */

#pragma once

#include <memory>
#include <unordered_map>

#include "common/common.h"
#include "common/device_context.h"
#include "core/tensor/data.h"
#include "frame_manager.h"

namespace allspark {

using CacheSpanHandle = std::string;

/**
 * @brief A virtual cache span.
 */
class CacheSpan {
 public:
  using Ptr = std::shared_ptr<CacheSpan>;
  using ConstPtr = std::shared_ptr<const CacheSpan>;

  /// @brief Create a new span from a present frame.
  /// @return The shared pointer to the span.
  static Ptr Create(CacheFrame::Ptr frame, std::string tag);
  static Ptr Create(CacheFrame::Ptr frame);
  /// @brief Get the frame pointer.
  CacheFrame::Ptr Frame();
  /// @brief Get the data pointer.
  void* Data();
  /// @brief Get the size of the span.
  int64_t Size();
  /// @brief Get the tag of the span.
  std::string Tag();

  void RefCacheSpan();
  void UnrefCacheSpan();
  bool HasExtraRef();

  virtual ~CacheSpan() = default;

  // TODO: swap, share

 protected:
  CacheSpan() {}

 private:
  CacheSpan(const CacheSpan&) = delete;
  CacheSpan(CacheSpan&&) = delete;
};

/**
 * @brief The cache span manager for one model.
 *
 * Generally, all requests for one model use the same cache config.
 * So each model has its own cache span manager.
 * Construct the manager when building the model, at which time the device is
 * determined.
 * Initialize the manager in the first call to Reshape(), at which time the
 * span size are determined.
 * Init() will also init the frame manager if it is not yet inited.
 */
class CacheSpanManager {
 public:
  using Ptr = std::shared_ptr<CacheSpanManager>;
  using ConstPtr = std::shared_ptr<const CacheSpanManager>;

  static CacheSpanHandle GenHandle(std::string tag);

  explicit CacheSpanManager(const CacheFrameManager::Ptr& frame_manager)
      : frame_manager_(frame_manager), span_size_(0), initialized_(false) {}

  bool Inited() const { return initialized_; }
  /// @brief Initialize the manager in the first call to Reshape().
  virtual void Init(int64_t span_size) = 0;

  /// @brief Request a tagged span.
  /// @param do_alloc If true, allocate the span without checking existence.
  /// @return Shared pointr to the span, NULL if failed.
  virtual CacheSpan::Ptr GetSpan(CacheSpanHandle tag, bool do_alloc) = 0;
  /// @brief Claim multiple tagged spans.
  /// @return Number of claimed spans.
  virtual size_t ClaimSpan(std::vector<CacheSpan::Ptr>& out_vec,
                           CacheSpanHandle tag, size_t count) = 0;
  virtual size_t ClaimSpanFromPres(std::vector<CacheSpan::Ptr>& out_vec,
                                   CacheSpanHandle tag, size_t count) = 0;
  /// @brief Release the span with tag.
  virtual void ReleaseSpan(const CacheSpanHandle& tag) = 0;
  /// @brief Release the span.
  /// @pre span must be managed by this manager.
  virtual void ReleaseSpan(CacheSpan::Ptr span) = 0;
  /// @brief Release multiple span.
  /// @pre span must be managed by this manager.
  virtual void ReleaseSpan(std::vector<CacheSpan::Ptr>& span_vec,
                           size_t count) = 0;

  DeviceType GetDeviceType() const { return frame_manager_->GetDeviceType(); }
  int64_t GetSpanSize() const { return span_size_; }

  CacheFrameManager::ConstPtr GetFrameManager() const { return frame_manager_; }

 protected:
  CacheFrameManager::Ptr frame_manager_;
  int64_t span_size_;
  bool initialized_;
};

class DefaultCacheSpanManager : public CacheSpanManager {
 public:
  explicit DefaultCacheSpanManager(const CacheFrameManager::Ptr& frame_manager)
      : CacheSpanManager(frame_manager) {}

  void Init(int64_t span_size) override;
  CacheSpan::Ptr GetSpan(CacheSpanHandle tag, bool do_alloc) override;
  size_t ClaimSpan(std::vector<CacheSpan::Ptr>& out_vec, CacheSpanHandle tag,
                   size_t count) override;
  size_t ClaimSpanFromPres(std::vector<CacheSpan::Ptr>& out_vec,
                           CacheSpanHandle tag, size_t count) override;
  void ReleaseSpan(const CacheSpanHandle& tag) override;
  void ReleaseSpan(CacheSpan::Ptr span) override;
  void ReleaseSpan(std::vector<CacheSpan::Ptr>& span_vec,
                   size_t count) override;

 private:
  std::unordered_map<CacheSpanHandle, CacheSpan::Ptr> spans_;
};

class ConcurrentCacheSpanManager : public CacheSpanManager {
 public:
  explicit ConcurrentCacheSpanManager(
      const CacheFrameManager::Ptr& frame_manager)
      : CacheSpanManager(frame_manager) {}

  void Init(int64_t span_size) override;
  /// @param do_alloc Always true.
  CacheSpan::Ptr GetSpan(CacheSpanHandle tag, bool do_alloc) override;
  size_t ClaimSpan(std::vector<CacheSpan::Ptr>& out_vec, CacheSpanHandle tag,
                   size_t count) override;
  size_t ClaimSpanFromPres(std::vector<CacheSpan::Ptr>& out_vec,
                           CacheSpanHandle tag, size_t count) override;
  void ReleaseSpan(const CacheSpanHandle& tag) override;
  void ReleaseSpan(CacheSpan::Ptr span) override;
  void ReleaseSpan(std::vector<CacheSpan::Ptr>& span_vec,
                   size_t count) override;
};

}  // namespace allspark
