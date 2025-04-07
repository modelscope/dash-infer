/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    frame_manager.cpp
 */

#include "frame_manager.h"

#include <sstream>

#include "core/kernel/cpu/cpu_common.h"
#include "core/tensor/data.h"

namespace allspark {

class CacheFrameImpl : public CacheFrame {
 public:
  CacheFrameImpl(DeviceType device_type, int64_t frame_size,
                 const std::string& tag)
      : data_(std::make_unique<DenseData>(tag, frame_size, device_type)) {
#ifdef ENABLE_SPAN_DEBUG
    DLOG(INFO) << "CacheFrame created: " << Data();
#endif
  }
  ~CacheFrameImpl() {
#ifdef ENABLE_SPAN_DEBUG
    DLOG(INFO) << "CacheFrame destroyed: " << Data();
#endif
    // data_ will be automatically released by its dtor
  }
  void* Data() const { return data_->GetRawData(); }
  int64_t Size() const { return data_->GetSize(); }
  DeviceType GetDeviceType() const { return data_->GetDeviceType(); }

 private:
  std::unique_ptr<DenseData> data_;
};

namespace {

CacheFrameImpl* GetImpl(CacheFrame* ptr) {
  return dynamic_cast<CacheFrameImpl*>(ptr);
}

}  // anonymous namespace

CacheFrame::Ptr CacheFrame::Create(DeviceType device_type, int64_t frame_size,
                                   const std::string& tag) {
  return std::make_shared<CacheFrameImpl>(device_type, frame_size, tag);
}

void* CacheFrame::Data() { return GetImpl(this)->Data(); }

int64_t CacheFrame::Size() { return GetImpl(this)->Size(); }

DeviceType CacheFrame::GetDeviceType() {
  return GetImpl(this)->GetDeviceType();
}

/* -------------------------------------- */

size_t DefaultCacheFrameManager::CountFreeFrame() const {
  return free_frames_.size();
}

size_t DefaultCacheFrameManager::CountOccupiedFrame() const {
  return occupied_frames_.size();
}

size_t DefaultCacheFrameManager::CountFrame() const {
  return CountFreeFrame() + CountOccupiedFrame();
}

size_t DefaultCacheFrameManager::CountPresFrame() const {
  throw AsException(
      ("DefaultCacheFrameManager::CountPresFrame not implemented."));
}

void DefaultCacheFrameManager::Init(int64_t frame_size) {
  if (initialized_) {
    return;
  }

  if (frame_size <= 0) {
    LOG(ERROR) << "DefaultCacheFrameManager: frame size in bytes must be "
                  "positive, got "
               << frame_size;
    AS_THROW(AsStatus::ALLSPARK_PARAM_ERROR);
  }

  initialized_ = true;
  frame_size_ = frame_size;
  for (int i = 0; i < init_frames_; ++i) {
    if (!createFrame()) {
      break;
    }
  }

  LOG(INFO) << "DefaultCacheFrameManager: init with frame size in bytes: "
            << frame_size << ", frame count: " << CountFrame();
  return;
}

bool DefaultCacheFrameManager::GrowUntil(size_t num_frames) {
  if (num_frames == 0) {
    num_frames = std::numeric_limits<size_t>::max();
  }

  while (CountFrame() < num_frames) {
    if (!createFrame()) {
      return false;
    }
  }
  return true;
}

bool DefaultCacheFrameManager::GrowBy(size_t num_frames) {
  for (size_t i = 0; i < num_frames; ++i) {
    if (!createFrame()) {
      return false;
    }
  }
  return true;
}

CacheFrame::Ptr DefaultCacheFrameManager::AllocFrame() {
  if (CountFreeFrame() == 0) {
    for (int i = 0; i < grow_frames_; ++i) {
      if (!createFrame()) {
        break;
      }
    }
  }

  if (CountFreeFrame() == 0) {
    return nullptr;
  }

  CacheFrame::Ptr frame = free_frames_.front();
  free_frames_.pop();
  occupied_frames_[frame->Data()] = frame;
  return frame;
}

size_t DefaultCacheFrameManager::AllocFrame(
    std::vector<CacheFrame::Ptr>& out_vec, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    auto frame = AllocFrame();
    if (!frame) {
      return i;
    }
    out_vec[i] = std::move(frame);
  }
  return count;
}

size_t DefaultCacheFrameManager::PresFrame(size_t count) {
  throw AsException(("DefaultCacheFrameManager::PresFrame not implemented."));
}

size_t DefaultCacheFrameManager::AllocFrameFromPres(
    std::vector<CacheFrame::Ptr>& out_vec, size_t count) {
  throw AsException(
      ("DefaultCacheFrameManager::AllocFrameFromPres not implemented."));
}

void DefaultCacheFrameManager::FreeFrameToPres(
    std::vector<CacheFrame::Ptr>& frame_vec, size_t count) {
  throw AsException(
      ("DefaultCacheFrameManager::FreeFrameToPres not implemented."));
}

size_t DefaultCacheFrameManager::FreePresFrame(size_t count) {
  throw AsException(
      ("DefaultCacheFrameManager::FreePresFrame not implemented."));
}

void DefaultCacheFrameManager::FreeFrame(CacheFrame::Ptr frame) {
  if (frame == nullptr) {
    return;
  }

  const auto pos = occupied_frames_.find(frame->Data());
  if (pos == occupied_frames_.end()) {
    LOG(WARNING) << "WARNING: DefaultCacheFrameManager: unrecognized frame: "
                 << frame->Data();
    return;
  }

  occupied_frames_.erase(pos);
  free_frames_.push(std::move(frame));

  // TODO: GC
  return;
}

void DefaultCacheFrameManager::FreeFrame(
    std::vector<CacheFrame::Ptr>& frame_vec, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    FreeFrame(frame_vec[i]);
  }
  return;
}

bool DefaultCacheFrameManager::createFrame() {
  std::stringstream ss;
  ss << "frame_" << CountFrame();
  CacheFrame::Ptr frame;
  try {
    frame = CacheFrame::Create(device_type_, frame_size_, ss.str());
  } catch (const std::exception& e) {
    LOG(WARNING) << "WARNING: DefaultCacheFrameManager: failed to allocate "
                    "more frames, current frame count: "
                 << CountFrame();
    return false;
  }

  free_frames_.push(frame);
  return true;
}

/* -------------------------------------- */

size_t ConcurrentCacheFrameManager::CountFreeFrame() const {
  return free_frame_count_.load(std::memory_order_acquire);
}

size_t ConcurrentCacheFrameManager::CountOccupiedFrame() const {
  return CountFrame() - CountFreeFrame();
}

size_t ConcurrentCacheFrameManager::CountFrame() const {
  return total_frame_count_;
}

size_t ConcurrentCacheFrameManager::CountPresFrame() const {
  return pres_frame_count_.load(std::memory_order_acquire);
}

void ConcurrentCacheFrameManager::Init(int64_t frame_size) {
  if (initialized_) {
    return;
  }

  if (frame_size <= 0) {
    LOG(ERROR) << "ConcurrentCacheFrameManager: frame size in bytes must "
                  "be positive, got "
               << frame_size;
    AS_THROW(AsStatus::ALLSPARK_PARAM_ERROR);
  }

  initialized_ = true;
  frame_size_ = frame_size;
  // TODO: lock
  for (int i = 0; i < init_frames_; ++i) {
    if (!createFrame()) {
      break;
    }
  }

  LOG(INFO) << "ConcurrentCacheFrameManager: init with frame size in bytes: "
            << frame_size << ", frame count: " << CountFrame();
  return;
}

bool ConcurrentCacheFrameManager::GrowUntil(size_t num_frames) {
  if (num_frames == 0) {
    num_frames = std::numeric_limits<int64_t>::max();
  }

  int64_t current_num = CountFrame();
  while (current_num++ < num_frames) {
    if (!createFrame()) {
      return false;
    }
  }
  return true;
}

bool ConcurrentCacheFrameManager::GrowBy(size_t num_frames) {
  // TODO: lock
  for (size_t i = 0; i < num_frames; ++i) {
    if (!createFrame()) {
      return false;
    }
  }
  return true;
}

CacheFrame::Ptr ConcurrentCacheFrameManager::AllocFrame() {
  CacheFrame::Ptr ret;
  if (!free_frames_.try_dequeue(ret)) {
    return nullptr;
  }
  free_frame_count_.fetch_sub(1, std::memory_order_relaxed);
  return ret;
}

size_t ConcurrentCacheFrameManager::AllocFrame(
    std::vector<CacheFrame::Ptr>& out_vec, size_t count) {
  size_t finished = free_frames_.try_dequeue_bulk(out_vec.begin(), count);
  // all or none
  if (finished != count) {
    if (!free_frames_.enqueue_bulk(std::make_move_iterator(out_vec.begin()),
                                   finished)) {
      LOG(ERROR) << "ConcurrentCacheFrameManager: failed to bulk enqueue "
                    "free frames";
      AS_THROW(AsStatus::ALLSPARK_RUNTIME_ERROR);
    }
    return 0;
  }
  free_frame_count_.fetch_sub(finished, std::memory_order_relaxed);
  return count;
}

size_t ConcurrentCacheFrameManager::PresFrame(size_t count) {
  std::vector<CacheFrame::Ptr> pres_frame(count);
  size_t finished = free_frames_.try_dequeue_bulk(pres_frame.begin(), count);
  if (finished != count) {
    if (!free_frames_.enqueue_bulk(std::make_move_iterator(pres_frame.begin()),
                                   finished)) {
      LOG(ERROR) << "ConcurrentCacheFrameManager: failed to bulk enqueue "
                    "free frames";
      AS_THROW(AsStatus::ALLSPARK_RUNTIME_ERROR);
    }
    return 0;
  }

  if (!pres_frames_.enqueue_bulk(std::make_move_iterator(pres_frame.begin()),
                                 finished)) {
    LOG(ERROR) << "ConcurrentCacheFrameManager: failed to bulk enqueue "
                  "pres frames";
    AS_THROW(AsStatus::ALLSPARK_RUNTIME_ERROR);
  }
  free_frame_count_.fetch_sub(finished, std::memory_order_relaxed);
  pres_frame_count_.fetch_add(finished, std::memory_order_relaxed);
  return finished;
}

size_t ConcurrentCacheFrameManager::AllocFrameFromPres(
    std::vector<CacheFrame::Ptr>& out_vec, size_t count) {
  size_t finished = pres_frames_.try_dequeue_bulk(out_vec.begin(), count);
  // all or none
  if (finished != count) {
    if (!pres_frames_.enqueue_bulk(std::make_move_iterator(out_vec.begin()),
                                   finished)) {
      LOG(ERROR) << "ConcurrentCacheFrameManager: failed to bulk enqueue "
                    "pres frames";
      AS_THROW(AsStatus::ALLSPARK_RUNTIME_ERROR);
    }
    return 0;
  }
  pres_frame_count_.fetch_sub(finished, std::memory_order_relaxed);
  return finished;
}

void ConcurrentCacheFrameManager::FreeFrameToPres(
    std::vector<CacheFrame::Ptr>& frame_vec, size_t count) {
  if (count == 0) {
    return;
  }

  if (!pres_frames_.enqueue_bulk(std::make_move_iterator(frame_vec.begin()),
                                 count)) {
    LOG(ERROR) << "ConcurrentCacheFrameManager: failed to bulk enqueue "
                  "pres frames";
    AS_THROW(AsStatus::ALLSPARK_RUNTIME_ERROR);
  }
  pres_frame_count_.fetch_add(count, std::memory_order_relaxed);
  return;
}

size_t ConcurrentCacheFrameManager::FreePresFrame(size_t count) {
  if (count == 0) return 0;

  std::vector<CacheFrame::Ptr> pres_frame(count);
  size_t finished = pres_frames_.try_dequeue_bulk(pres_frame.begin(), count);

  if (finished != count) {
    LOG(ERROR) << "ConcurrentCacheFrameManager: free preserved frames failed.";
    AS_THROW(AsStatus::ALLSPARK_RUNTIME_ERROR);
  }

  if (!free_frames_.enqueue_bulk(std::make_move_iterator(pres_frame.begin()),
                                 finished)) {
    LOG(ERROR) << "ConcurrentCacheFrameManager: failed to bulk enqueue "
                  "free frames";
    AS_THROW(AsStatus::ALLSPARK_RUNTIME_ERROR);
  }
  pres_frame_count_.fetch_sub(finished, std::memory_order_relaxed);
  free_frame_count_.fetch_add(finished, std::memory_order_relaxed);
  return finished;
}

void ConcurrentCacheFrameManager::FreeFrame(CacheFrame::Ptr frame) {
  if (frame == nullptr) {
    return;
  }
  if (!free_frames_.enqueue(std::move(frame))) {
    LOG(ERROR) << "ConcurrentCacheFrameManager: failed to enqueue free frame";
    AS_THROW(AsStatus::ALLSPARK_RUNTIME_ERROR);
  }
  free_frame_count_.fetch_add(1, std::memory_order_relaxed);
  return;
}

void ConcurrentCacheFrameManager::FreeFrame(
    std::vector<CacheFrame::Ptr>& frame_vec, size_t count) {
  if (count == 0) {
    return;
  }
  if (!free_frames_.enqueue_bulk(std::make_move_iterator(frame_vec.begin()),
                                 count)) {
    LOG(ERROR) << "ConcurrentCacheFrameManager: failed to bulk enqueue "
                  "free frames";
    AS_THROW(AsStatus::ALLSPARK_RUNTIME_ERROR);
  }
  free_frame_count_.fetch_add(count, std::memory_order_relaxed);
  return;
}

bool ConcurrentCacheFrameManager::createFrame() {
  CacheFrame::Ptr frame;
  try {
    frame = CacheFrame::Create(device_type_, frame_size_, "frame");
  } catch (const std::exception& e) {
    LOG(WARNING) << "WARNING: ConcurrentCacheFrameManager: failed to allocate "
                    "more frames, current frame count: "
                 << CountFrame();
    return false;
  }

  if (!free_frames_.enqueue(std::move(frame))) {
    LOG(ERROR)
        << "ConcurrentCacheFrameManager: failed to enqueue created frame";
    return false;
  }
  total_frame_count_++;
  free_frame_count_.fetch_add(1, std::memory_order_relaxed);
  return true;
}

}  // namespace allspark
