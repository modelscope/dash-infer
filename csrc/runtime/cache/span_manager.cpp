/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    span_manager.cpp
 */

#if ENABLE_SPAN_ATTENTION

#include "span_manager.h"

#include <sstream>

namespace allspark {

class CacheSpanImpl : public CacheSpan {
 public:
  CacheSpanImpl(CacheFrame::Ptr frame, std::string tag)
      : frame_(std::move(frame)), tag_(std::move(tag)) {
#ifdef ENABLE_SPAN_DEBUG
    DLOG(INFO) << "CacheSpan created: " << Tag();
#endif
  }
  explicit CacheSpanImpl(CacheFrame::Ptr frame)
      : frame_(std::move(frame)), tag_() {
#ifdef ENABLE_SPAN_DEBUG
    DLOG(INFO) << "CacheSpan created without tag";
#endif
  }
  ~CacheSpanImpl() {
#ifdef ENABLE_SPAN_DEBUG
    DLOG(INFO) << "CacheSpan destroyed: " << Tag();
#endif
  }

  CacheFrame::Ptr Frame() const { return frame_; }
  void* Data() const { return frame_->Data(); }
  int64_t Size() const { return frame_->Size(); }
  std::string Tag() const { return tag_; }

  void RefCacheSpan() { has_extra_ref_ = true; }
  void UnrefCacheSpan() { has_extra_ref_ = false; }
  bool HasExtraRef() const { return has_extra_ref_; }

 private:
  CacheFrame::Ptr frame_;
  std::string tag_;
  bool has_extra_ref_ = false;
};

namespace {

CacheSpanImpl* GetImpl(CacheSpan* ptr) {
  return dynamic_cast<CacheSpanImpl*>(ptr);
}

}  // anonymous namespace

CacheSpan::Ptr CacheSpan::Create(CacheFrame::Ptr frame, std::string tag) {
  return std::make_shared<CacheSpanImpl>(std::move(frame), std::move(tag));
}

CacheSpan::Ptr CacheSpan::Create(CacheFrame::Ptr frame) {
  return std::make_shared<CacheSpanImpl>(std::move(frame));
}

CacheFrame::Ptr CacheSpan::Frame() { return GetImpl(this)->Frame(); }

void* CacheSpan::Data() { return GetImpl(this)->Data(); }

int64_t CacheSpan::Size() { return GetImpl(this)->Size(); }

std::string CacheSpan::Tag() { return GetImpl(this)->Tag(); }

void CacheSpan::RefCacheSpan() { return GetImpl(this)->RefCacheSpan(); }

void CacheSpan::UnrefCacheSpan() { return GetImpl(this)->UnrefCacheSpan(); }

bool CacheSpan::HasExtraRef() { return GetImpl(this)->HasExtraRef(); }

/* -------------------------------------- */

CacheSpanHandle CacheSpanManager::GenHandle(std::string tag) {
  return CacheSpanHandle(std::move(tag));
}

/* -------------------------------------- */

void DefaultCacheSpanManager::Init(int64_t span_size) {
  if (initialized_) {
    return;
  }

  if (span_size <= 0) {
    LOG(ERROR) << "DefaultCacheSpanManager: span size in bytes must be "
                  "positive, got "
               << span_size;
    AS_THROW(AsStatus::ALLSPARK_PARAM_ERROR);
  }

  if (!frame_manager_->Inited()) {
    frame_manager_->Init(span_size);
  }

  if (span_size != frame_manager_->GetFrameSize()) {
    LOG(ERROR) << "DefaultCacheSpanManager: span size in bytes not match, "
                  "span size: "
               << span_size
               << ", frame size: " << frame_manager_->GetFrameSize();
    AS_THROW(AsStatus::ALLSPARK_PARAM_ERROR);
  }

  initialized_ = true;
  span_size_ = span_size;
  LOG(INFO) << "DefaultCacheSpanManager: init with span size in bytes: "
            << span_size_;
  return;
}

CacheSpan::Ptr DefaultCacheSpanManager::GetSpan(CacheSpanHandle tag,
                                                bool do_alloc) {
  if (!do_alloc) {
    auto it = spans_.find(tag);
    if (it != spans_.end()) {
      return it->second;
    }
  }
  CacheFrame::Ptr frame = frame_manager_->AllocFrame();
  if (!frame) {
    return nullptr;
  }
  CacheSpan::Ptr span = CacheSpan::Create(std::move(frame), tag);
  spans_[std::move(tag)] = span;
  return span;
}

size_t DefaultCacheSpanManager::ClaimSpan(std::vector<CacheSpan::Ptr>& out_vec,
                                          CacheSpanHandle tag, size_t count) {
  std::vector<CacheSpan::Ptr> new_spans(0);
  size_t claimed = 0;
  for (; claimed < count; ++claimed) {
    auto span = GetSpan(tag, true);
    if (!span) {
      break;
    }
    new_spans.emplace_back(span);
  }

  if (claimed < count) {
    for (auto span : new_spans) {
      frame_manager_->FreeFrame(span->Frame());
    }
    return 0;
  }

  out_vec = std::move(new_spans);
  return count;
}

size_t DefaultCacheSpanManager::ClaimSpanFromPres(
    std::vector<CacheSpan::Ptr>& out_vec, CacheSpanHandle /* tag */,
    size_t count) {
  throw AsException(
      ("DefaultCacheSpanManager::ClaimSpanFromPres not implemented."));
}

void DefaultCacheSpanManager::ReleaseSpan(const CacheSpanHandle& tag) {
  auto it = spans_.find(tag);
  if (it != spans_.end()) {
    CacheSpan::Ptr span = it->second;
    if (span->HasExtraRef() == false) {
      spans_.erase(it);
      frame_manager_->FreeFrame(span->Frame());
    }
  } else {
    LOG(WARNING) << "DefaultCacheSpanManager: span not found, tag: " << tag;
  }
  return;
}

void DefaultCacheSpanManager::ReleaseSpan(CacheSpan::Ptr span) {
  if (span->HasExtraRef() == false) {
    spans_.erase(span->Tag());
    frame_manager_->FreeFrame(span->Frame());
  }
  return;
}

void DefaultCacheSpanManager::ReleaseSpan(std::vector<CacheSpan::Ptr>& span_vec,
                                          size_t count) {
  for (size_t i = 0; i < count; ++i) {
    ReleaseSpan(span_vec[i]);
  }
  return;
}

/* -------------------------------------- */

void ConcurrentCacheSpanManager::Init(int64_t span_size) {
  if (initialized_) {
    return;
  }

  if (span_size <= 0) {
    LOG(ERROR) << "ConcurrentCacheSpanManager: span size in bytes must be "
                  "positive, got "
               << span_size;
    AS_THROW(AsStatus::ALLSPARK_PARAM_ERROR);
  }

  if (!frame_manager_->Inited()) {
    frame_manager_->Init(span_size);
  }

  if (span_size != frame_manager_->GetFrameSize()) {
    LOG(ERROR) << "ConcurrentCacheSpanManager: span size in bytes not "
                  "match, span size: "
               << span_size
               << ", frame size: " << frame_manager_->GetFrameSize();
    AS_THROW(AsStatus::ALLSPARK_PARAM_ERROR);
  }

  initialized_ = true;
  span_size_ = span_size;
  LOG(INFO) << "ConcurrentCacheSpanManager: init with span size in bytes: "
            << span_size_;
  return;
}

CacheSpan::Ptr ConcurrentCacheSpanManager::GetSpan(CacheSpanHandle tag,
                                                   bool /* do_alloc */) {
  // [warning] this api may cause bug after implement prefill disaggregation
  CacheFrame::Ptr frame = frame_manager_->AllocFrame();
  if (!frame) {
    return nullptr;
  }
  CacheSpan::Ptr span = CacheSpan::Create(std::move(frame), std::move(tag));
  return span;
}

size_t ConcurrentCacheSpanManager::ClaimSpan(
    std::vector<CacheSpan::Ptr>& out_vec, CacheSpanHandle /* tag */,
    size_t count) {
  std::vector<CacheFrame::Ptr> new_frames(count);
  size_t claimed = frame_manager_->AllocFrame(new_frames, count);
  // all or none
  if (claimed < count) {
    if (claimed > 0) {
      frame_manager_->FreeFrame(new_frames, claimed);
    }
    return 0;
  }
  std::transform(std::make_move_iterator(new_frames.begin()),
                 std::make_move_iterator(new_frames.end()), out_vec.begin(),
                 [](auto it) { return CacheSpan::Create(std::move(it)); });
  return count;
}

size_t ConcurrentCacheSpanManager::ClaimSpanFromPres(
    std::vector<CacheSpan::Ptr>& out_vec, CacheSpanHandle /* tag */,
    size_t count) {
  std::vector<CacheFrame::Ptr> new_frames(count);
  size_t claimed = frame_manager_->AllocFrameFromPres(new_frames, count);
  // all or none
  if (claimed < count) {
    if (claimed > 0) {
      frame_manager_->FreeFrameToPres(new_frames, claimed);
    }
    return 0;
  }
  std::transform(std::make_move_iterator(new_frames.begin()),
                 std::make_move_iterator(new_frames.end()), out_vec.begin(),
                 [](auto it) { return CacheSpan::Create(std::move(it)); });
  return count;
}

void ConcurrentCacheSpanManager::ReleaseSpan(const CacheSpanHandle& tag) {
  LOG(ERROR) << "ConcurrentCacheSpanManager: ReleaseSpan not implemented";
  AS_THROW(AsStatus::ALLSPARK_UNKNOWN_ERROR);
}

void ConcurrentCacheSpanManager::ReleaseSpan(CacheSpan::Ptr span) {
  if (span->HasExtraRef() == false) {
    frame_manager_->FreeFrame(span->Frame());
  }
  return;
}

void ConcurrentCacheSpanManager::ReleaseSpan(
    std::vector<CacheSpan::Ptr>& span_vec, size_t count) {
#if 0
  if (count == 0) {
    return;
  }
  std::vector<CacheFrame::Ptr> frames(count);
  std::transform(std::make_move_iterator(span_vec.begin()),
                 std::make_move_iterator(span_vec.end()), frames.begin(),
                 [](auto it) { return it->Frame(); });
  frame_manager_->FreeFrame(frames, count);
  return;
#else
  for (size_t i = 0; i < count; ++i) {
    ReleaseSpan(span_vec[i]);
  }
#endif
}

}  // namespace allspark
#endif
