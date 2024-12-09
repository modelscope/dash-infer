/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    virtual_cache.cpp
 */

#if ENABLE_SPAN_ATTENTION
#include "virtual_cache.h"

#include <algorithm>
#include <iterator>
#include <sstream>

namespace allspark {

namespace {

template <typename UT>
inline UT uint_div_up(UT x, UT y) {
  return (x + y - 1) / y;
}

template <typename UT>
inline bool is_power_of_2(UT x) {
  return (x & (x - 1)) == 0;
}

}  // anonymous namespace

namespace cache {

CacheArray::CacheArray(SpannedVirtualCache* manager, int layer_id,
                       int max_num_spans)
    : tag_(manager->GetTag() + "_" + std::to_string(layer_id)),
      cache_config_(manager->GetCacheConfig()),
      layer_id_(layer_id),
      span_manager_(manager->GetSpanManager()),
      seq_length_(0) {
  std::stringstream ss;
  ss << tag_ << "_tlb";
  tlb_ =
      std::make_unique<AsTensor>(ss.str(), DeviceType::CPU, DataType::POINTER,
                                 DataMode::DENSE, Shape{max_num_spans});
}

CacheArray::~CacheArray() {
  tlb_->Free();

#ifdef ENABLE_SPAN_DEBUG
  for (auto& span : cache_) {
    DLOG(INFO) << "CacheArray layer[" << layer_id_
               << "] destroy span id=" << span->Tag();
    destroySpan(span);
  }
#else
  destroySpan(cache_, cache_.size());
#endif
}

bool CacheArray::InitContext(int context_length) {
  bool ret = updateSeqLength(context_length);
  if (ret) {
#ifdef ENABLE_SPAN_DEBUG
    DLOG(INFO) << "CacheArray layer[" << layer_id_
               << "] init, context len: " << seq_length_
               << ", num spans: " << cache_.size();
#endif
  }
  return ret;
}

const AsTensor& CacheArray::GetCachePtrVec(int increment) {
  if (increment < 0) {
    LOG(ERROR)
        << "CacheArray::GetCachePtrVec: increment cannot be negative, got "
        << increment;
    AS_THROW(AsStatus::ALLSPARK_PARAM_ERROR);
  }

  if (increment > 0) {
    if (!updateSeqLength(seq_length_ + increment)) {
      LOG(ERROR) << "CacheArray::GetCachePtrVec: cannot achieve required "
                    "cache span increment due to insufficient memory";
      AS_THROW(AsStatus::ALLSPARK_CACHE_MEMORY_OUT);
    }
  }

  return *tlb_;
}

const CacheSpan::Ptr CacheArray::GetCacheSpan(int index) const {
  if (index < 0 || index >= cache_.size()) {
    LOG(ERROR) << "CacheArray::GetCacheSpan: index out of range: " << index;
    AS_THROW(AsStatus::ALLSPARK_PARAM_ERROR);
  }
  return cache_[index];
}

CacheSpan::Ptr CacheArray::newSpan(size_t index) const {
  std::stringstream ss;
  ss << tag_ << "_" << index;
  const auto handle_tag = ss.str();
  const auto handle = CacheSpanManager::GenHandle(handle_tag);
  auto span = span_manager_->GetSpan(handle, true);
  if (span == nullptr) {
    return nullptr;
  }
  return span;
}

size_t CacheArray::newSpan(std::vector<CacheSpan::Ptr>& out_vec,
                           size_t start_index, size_t count) const {
  std::string handle_tag = tag_ + "_" + "span";
  return span_manager_->ClaimSpan(out_vec, std::move(handle_tag), count);
}

void CacheArray::destroySpan(CacheSpan::Ptr span) const {
  if (!span) {
    return;
  }
  span_manager_->ReleaseSpan(std::move(span));
  return;
}

void CacheArray::destroySpan(std::vector<CacheSpan::Ptr>& span_vec,
                             size_t count) const {
  if (count == 0) {
    return;
  }
  span_manager_->ReleaseSpan(span_vec, count);
  return;
}

bool CacheArray::updateSeqLength(size_t new_seq_length) {
  if (new_seq_length == seq_length_) {
    return true;
  } else if (new_seq_length < seq_length_) {
    // TODO: rollback
    LOG(ERROR) << "CacheArray::updateSeqLength: new sequence length "
               << new_seq_length
               << " cannot be less than current sequence length " << seq_length_
               << ", rollback not implemented yet";
    AS_THROW(AsStatus::ALLSPARK_RUNTIME_ERROR);
  }

  // assert: new_seq_length > seq_length_
  const size_t current_spans = GetSpanNum();
  const size_t need_spans = uint_div_up(
      new_seq_length, static_cast<size_t>(cache_config_->span_size));

  if (static_cast<size_t>(tlb_->GetShape()[0]) < need_spans) {
    LOG(ERROR) << "CacheArray::updateSeqLength: TLB overflow, capacity: "
               << tlb_->GetShape()[0] << ", required: " << need_spans;
    AS_THROW(AsStatus::ALLSPARK_EXCEED_LIMIT_ERROR);
  }

  if (need_spans > current_spans) {
    std::vector<CacheSpan::Ptr> new_spans;
    new_spans.resize(need_spans - current_spans);

    size_t num_new_spans = 0;
#ifdef CONFIG_CONCURRENT_SPAN
    num_new_spans =
        newSpan(new_spans, current_spans, need_spans - current_spans);
#else
    for (size_t i = current_spans; i < need_spans; ++i) {
      auto span = newSpan(i);
      if (span == nullptr) {
        break;
      }
      new_spans[i - current_spans] = std::move(span);
      num_new_spans++;
    }
#endif

    if (num_new_spans == need_spans - current_spans) {
      std::vector<void*> new_ptrs(new_spans.size());
      std::transform(new_spans.begin(), new_spans.end(), new_ptrs.begin(),
                     [](const auto& it) { return it->Data(); });
      TensorUtils::DeepCopyFromStdVector(*tlb_, current_spans, new_ptrs);

      cache_.reserve(need_spans);
      cache_.insert(cache_.end(), std::make_move_iterator(new_spans.begin()),
                    std::make_move_iterator(new_spans.end()));
    } else {
      LOG(ERROR)
          << "CacheArray::updateSeqLength: cannot make enough cache span";
      // restore
      destroySpan(new_spans, num_new_spans);
      return false;
    }
  }

  // all spans are available, update seq length
  seq_length_ = new_seq_length;
  return true;
}

}  // namespace cache

size_t CacheUtils::GetSpanSizeInBytes(const SpanCacheConfig& cache_config,
                                      DataType data_type, int num_heads,
                                      int per_head_size) {
  size_t cache_size_data{0};
  size_t cache_size_extra{0};
  switch (cache_config.mode) {
    case AsCacheMode::AsCacheDefault:
      cache_size_data = cache_config.span_size * num_heads * per_head_size *
                        SizeofType(data_type);
      break;
    case AsCacheMode::AsCacheQuantI8:
      cache_size_data = cache_config.span_size * num_heads * per_head_size *
                        SizeofType(DataType::INT8);
      // 2: zero and scale
      cache_size_extra = 2 * cache_config.span_size * num_heads *
                         SizeofType(DataType::FLOAT32);
      break;
    case AsCacheMode::AsCacheQuantU4:
      cache_size_data = cache_config.span_size * num_heads * per_head_size *
                        SizeofType(DataType::UINT8) / 2;
      // 2: zero and scale
      cache_size_extra = 2 * cache_config.span_size * num_heads *
                         SizeofType(DataType::FLOAT32);
      break;
    default:
      // this should never happen
      LOG(ERROR) << "CacheUtils: bad AsCacheMode: " << int(cache_config.mode);
      AS_THROW(AsStatus::ALLSPARK_UNKNOWN_ERROR);
  }
  return cache_size_data + cache_size_extra;
}

span::QuantMode CacheUtils::toQuantMode(AsCacheMode cache_mode) {
  switch (cache_mode) {
    case AsCacheMode::AsCacheDefault:
      return span::QuantMode::NONE;
    case AsCacheMode::AsCacheQuantI8:
      return span::QuantMode::I8;
    case AsCacheMode::AsCacheQuantU4:
      return span::QuantMode::U4;
    default:
      LOG(ERROR) << "bad cache mode: " << static_cast<int>(cache_mode);
      AS_THROW(AsStatus::ALLSPARK_RUNTIME_ERROR);
  }
}

/* -------------------------------------- */

AsStatus SpannedVirtualCache::InitLayer(int layer_id, int num_heads,
                                        int per_head_size, int context_len,
                                        int max_num_spans) {
  // param check
  if (num_heads <= 0 || per_head_size <= 0) {
    LOG(ERROR) << "SpannedVirtualCache::InitLayer: layer[" << layer_id
               << "] bad params: num_heads=" << num_heads
               << ", per_head_size=" << per_head_size;
    return AsStatus::ALLSPARK_PARAM_ERROR;
  }

  if (!is_power_of_2(per_head_size)) {
    LOG(WARNING) << "SpannedVirtualCache::InitLayer: layer[" << layer_id
                 << "] per_head_size is not power of 2, got " << per_head_size;
  }

  if (!span_manager_->Inited()) {
    LOG(ERROR) << "SpannedVirtualCache::InitLayer: uninitialized span_manager_";
    return AsStatus::ALLSPARK_RUNTIME_ERROR;
  }

  // build & init layer
  auto layer = std::make_unique<CacheArray>(this, layer_id, max_num_spans);
  if (!layer->InitContext(context_len)) {
    LOG(ERROR) << "SpannedVirtualCache::InitLayer: layer[" << layer_id
               << "] init failed due to insufficient memory";
    return AsStatus::ALLSPARK_MEMORY_ERROR;
  }
  layer_cache_[layer_id] = std::move(layer);

  return AsStatus::ALLSPARK_SUCCESS;
}

const AsTensor& SpannedVirtualCache::GetCache(int layer_id, int increment) {
  if (layer_id < 0 || layer_id >= layer_num_) {
    LOG(ERROR) << "SpannedVirtualCache::GetCache: layer_id out of range: "
               << layer_id;
    AS_THROW(AsStatus::ALLSPARK_PARAM_ERROR);
  }

  auto& layer = layer_cache_[layer_id];

  return layer->GetCachePtrVec(increment);
}

size_t SpannedVirtualCache::GetSeqLength(int layer_id) const {
  if (layer_id < 0 || layer_id >= layer_num_) {
    LOG(ERROR) << "SpannedVirtualCache::GetSeqLength: layer_id out of range: "
               << layer_id;
    AS_THROW(AsStatus::ALLSPARK_RUNTIME_ERROR);
  }

  return layer_cache_[layer_id]->GetSeqLength();
}

void SpannedVirtualCache::FillCache(std::vector<CacheSpan::Ptr>& cache) {
  int current_span_num = layer_cache_[0]->GetSpanNum();
  for (int layer_idx = 0; layer_idx < layer_num_; layer_idx++) {
    layer_cache_[layer_idx]->GetCacheVector().emplace_back(cache[layer_idx]);
    layer_cache_[layer_idx]->SetSeqLength((current_span_num + 1) *
                                          cache_config_->span_size);
    ((void**)layer_cache_[layer_idx]
         ->GetTLB()
         ->GetDataPtr())[current_span_num] = cache[layer_idx]->Data();
  }
  return;
}
}  // namespace allspark
#endif
