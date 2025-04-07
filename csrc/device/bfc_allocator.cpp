/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    bfc_allocator.cpp
 */

#include <algorithm>
#include <array>
#include <atomic>
#include <deque>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "device/cpu/cpu_allocator.h"
#ifdef ENABLE_CUDA
#include "device/cuda/cuda_allocator.h"
#include "device/cuda/cuda_host_allocator.h"
#endif
#include <utility/check_cuda.h>
#include <utility/mem_registry.h>

#include "check.h"
#include "device/bfc_allocator.h"

// #define DEBUG_BFC

// FIXME: release BFC memory when desc will cause crash.
#define BFC_FREE_MEMORY 1

namespace allspark {

class BFCAllocatorImpl;

class MemoryDump;

// A memory allocator that implements a 'best-fit with coalescing'
// algorithm.  This is essentially a very simple version of Doug Lea's
// malloc (dlmalloc).
//
// The goal of this allocator is to support defragmentation via
// coalescing.  One assumption we make is that the process using this
// allocator owns pretty much all of the memory, and that nearly
// all requests to allocate memory go through this interface.
class BFCAllocatorImpl {
 public:
  BFCAllocatorImpl(std::unique_ptr<SubAllocator> sub_allocator,
                   size_t total_memory, const std::string& name, int device_id,
                   const BFCAllocator::Options& opts);

  ~BFCAllocatorImpl();

  std::string Name() { return name_; }

  void* AllocateRaw(size_t alignment, size_t num_bytes);

  void DeallocateRaw(void* ptr);

  bool TracksAllocationSizes() const;

  size_t RequestedSize(const void* ptr) const;

  size_t AllocatedSize(const void* ptr) const;

  int64_t AllocationId(const void* ptr) const;

  AllocatorStats GetStats();

  bool ClearStats();

  DeviceType GetMemoryType() const;

  bool ShouldRecordOpName() const { return true; }

  int GetDeviceId() { return device_id_; }

 private:
  struct Bin;

  void* AllocateRawInternal(size_t alignment, size_t num_bytes,
                            bool dump_log_on_failure);

  void DeallocateRawInternal(void* ptr);

  // Return the f dlargest free chunk bytes from the largest bin in constant
  // time. The free chunks are sorted by size (and then address) in a bin.
  int64_t LargestFreeChunk();

  // A ChunkHandle is an index into the chunks_ vector in BFCAllocatorImpl
  // kInvalidChunkHandle means an invalid chunk
  typedef size_t ChunkHandle;
  static constexpr ChunkHandle kInvalidChunkHandle = SIZE_MAX;

  typedef int BinNum;
  static constexpr int kInvalidBinNum = -1;
  // The following means that the largest bin'd chunk size is 256 << 21 = 512MB.
  static constexpr int kNumBins = 21;

  // A Chunk points to a piece of memory that's either entirely free or entirely
  // in use by one user memory allocation.
  //
  // An AllocationRegion's memory is split up into one or more disjoint Chunks,
  // which together cover the whole region without gaps.  Chunks participate in
  // a doubly-linked list, and the prev/next pointers point to the physically
  // adjacent chunks.
  //
  // Since a chunk cannot be partially in use, we may need to split a free chunk
  // in order to service a user allocation.  We always merge adjacent free
  // chunks.
  //
  // Chunks contain information about whether they are in use or whether they
  // are free, and contain a pointer to the bin they are in.
  struct Chunk {
    size_t size = 0;  // Full size of buffer.

    // We sometimes give chunks that are larger than needed to reduce
    // fragmentation.  requested_size keeps track of what the client
    // actually wanted so we can understand whether our splitting
    // strategy is efficient.
    size_t requested_size = 0;

    // allocation_id is set to -1 when the chunk is not in use. It is assigned a
    // value greater than zero before the chunk is returned from
    // AllocateRaw, and this value is unique among values assigned by
    // the parent allocator.
    int64_t allocation_id = -1;
    void* ptr = nullptr;  // pointer to granted subbuffer.

    // If not kInvalidChunkHandle, the memory referred to by 'prev' is directly
    // preceding the memory used by this chunk.  E.g., It should start
    // at 'ptr - prev->size'
    ChunkHandle prev = kInvalidChunkHandle;

    // If not kInvalidChunkHandle, the memory referred to by 'next' is directly
    // following the memory used by this chunk.  E.g., It should be at
    // 'ptr + size'
    ChunkHandle next = kInvalidChunkHandle;

    // What bin are we in?
    BinNum bin_num = kInvalidBinNum;

    // Optional count when this chunk was most recently made free.
    uint64_t freed_at_count = 0;

    bool in_use() const { return allocation_id != -1; }

    std::string DebugString(BFCAllocatorImpl* a, bool recurse) {
      std::ostringstream dbg;
      dbg << "  Size: " << size << " | Requested Size: " << requested_size
          << " | in_use: " << in_use() << " | bin_num: " << bin_num;
      if (recurse && prev != kInvalidChunkHandle) {
        Chunk* p = a->ChunkFromHandle(prev);
        dbg << ", prev: " << p->DebugString(a, false);
      }
      if (recurse && next != kInvalidChunkHandle) {
        Chunk* n = a->ChunkFromHandle(next);
        dbg << ", next: ", n->DebugString(a, false);
      }
      return dbg.str();
    }
  };

  // A Bin is a collection of similar-sized free chunks.
  // Allocated chunks are never in a Bin.
  struct Bin {
    // All chunks in this bin have >= bin_size memory.
    size_t bin_size = 0;

    class ChunkComparator {
     public:
      explicit ChunkComparator(BFCAllocatorImpl* allocator)
          : allocator_(allocator) {}
      // Sort first by size and then use pointer address as a tie breaker.
      bool operator()(const ChunkHandle ha, const ChunkHandle hb) const {
        const Chunk* a = allocator_->ChunkFromHandle(ha);
        const Chunk* b = allocator_->ChunkFromHandle(hb);
        if (a->size != b->size) {
          return a->size < b->size;
        }
        return a->ptr < b->ptr;
      }

     private:
      BFCAllocatorImpl* allocator_;  // The parent allocator
    };

    typedef std::set<ChunkHandle, ChunkComparator> FreeChunkSet;
    // List of free chunks within the bin, sorted by chunk size.
    // Chunk * not owned.
    FreeChunkSet free_chunks;
    Bin(BFCAllocatorImpl* allocator, size_t bs)
        : bin_size(bs), free_chunks(ChunkComparator(allocator)) {}
  };

  static constexpr size_t kMinAllocationBits = 8;
  static constexpr size_t kMinAllocationSize = 1 << kMinAllocationBits;

  // BFCAllocatorImpl allocates memory into a collection of disjoint
  // AllocationRegions.  Each AllocationRegion corresponds to one call to
  // SubAllocator::Alloc().  (Actually, if a subsequent call to
  // SubAllocator::Alloc() returns another region immediately adjacent to the
  // last, it will be used to extend the first AllocationRegion, not create a
  // separate one.)
  //
  // An AllocationRegion contains one or more Chunks, covering all of its
  // memory.  Its primary job is to map pointers to ChunkHandles.
  //
  // This class is thread-compatible.
  class AllocationRegion {
   public:
    AllocationRegion(void* ptr, size_t memory_size)
        : ptr_(ptr),
          memory_size_(memory_size),
          end_ptr_(
              static_cast<void*>(static_cast<char*>(ptr_) + memory_size_)) {
      const size_t n_handles =
          (memory_size + kMinAllocationSize - 1) / kMinAllocationSize;
      handles_.resize(n_handles, kInvalidChunkHandle);
    }

    AllocationRegion() = default;
    AllocationRegion(AllocationRegion&& other) { Swap(&other); }
    AllocationRegion& operator=(AllocationRegion&& other) {
      Swap(&other);
      return *this;
    }

    void* ptr() const { return ptr_; }
    void* end_ptr() const { return end_ptr_; }
    size_t memory_size() const { return memory_size_; }
    void extend(size_t size) {
      memory_size_ += size;

      end_ptr_ = static_cast<void*>(static_cast<char*>(end_ptr_) + size);
      const size_t n_handles =
          (memory_size_ + kMinAllocationSize - 1) / kMinAllocationSize;
      handles_.resize(n_handles, kInvalidChunkHandle);
    }
    ChunkHandle get_handle(const void* p) const {
      return handles_[IndexFor(p)];
    }
    void set_handle(const void* p, ChunkHandle h) { handles_[IndexFor(p)] = h; }
    void erase(const void* p) { set_handle(p, kInvalidChunkHandle); }

   private:
    void Swap(AllocationRegion* other) {
      std::swap(ptr_, other->ptr_);
      std::swap(memory_size_, other->memory_size_);
      std::swap(end_ptr_, other->end_ptr_);
      std::swap(handles_, other->handles_);
    }

    size_t IndexFor(const void* p) const {
      std::uintptr_t p_int = reinterpret_cast<std::uintptr_t>(p);
      std::uintptr_t base_int = reinterpret_cast<std::uintptr_t>(ptr_);
      return static_cast<size_t>(((p_int - base_int) >> kMinAllocationBits));
    }

    // Metadata about the allocation region.
    void* ptr_ = nullptr;
    size_t memory_size_ = 0;
    void* end_ptr_ = nullptr;

    // Array of size "memory_size / kMinAllocationSize".  It is
    // indexed by (p-base) / kMinAllocationSize, contains ChunkHandle
    // for the memory allocation represented by "p"
    std::vector<ChunkHandle> handles_;

    DISABLE_COPY_AND_ASSIGN(AllocationRegion);
  };

  // RegionManager aggregates one or more "AllocationRegions" and provides
  // a layer of indirection from pointers to the underlying ChunkHandle,
  // allowing allocation across multiple discontiguous memory regions.
  //
  // This class is thread-compatible.
  class RegionManager {
   public:
    RegionManager() {}
    ~RegionManager() {}

    void AddAllocationRegion(void* ptr, size_t memory_size) {
      // Insert sorted by end_ptr.
      auto entry =
          std::upper_bound(regions_.begin(), regions_.end(), ptr, &Comparator);
      regions_.insert(entry, AllocationRegion(ptr, memory_size));
    }

    // Adds an alloation region for the given ptr and size, potentially
    // extending a region if ptr matches the end_ptr of an existing region.
    // If a region is extended, returns a pointer to the extended region so that
    // the BFC allocator can reason about chunkification.
    AllocationRegion* AddOrExtendAllocationRegion(void* ptr,
                                                  size_t memory_size) {
      // Insert sorted by end_ptr.
      auto entry =
          std::upper_bound(regions_.begin(), regions_.end(), ptr, &Comparator);
      // Check if can be coalesced with preceding region.
      if (entry != regions_.begin()) {
        auto preceding_region = entry - 1;
        if (preceding_region->end_ptr() == ptr) {
          preceding_region->extend(memory_size);
          return &*preceding_region;
        }
      }
      regions_.insert(entry, AllocationRegion(ptr, memory_size));
      return nullptr;
    }

    std::vector<AllocationRegion>::iterator RemoveAllocationRegion(
        std::vector<AllocationRegion>::iterator it) {
      return regions_.erase(it);
    }

    ChunkHandle get_handle(const void* p) const {
      auto* rp = RegionFor(p);
      if (rp) return rp->get_handle(p);
      return 0;
    }

    void set_handle(const void* p, ChunkHandle h) {
      return MutableRegionFor(p)->set_handle(p, h);
    }
    void erase(const void* p) { return MutableRegionFor(p)->erase(p); }

    const std::vector<AllocationRegion>& regions() const { return regions_; }

   private:
    static bool Comparator(const void* ptr, const AllocationRegion& other) {
      return ptr < other.end_ptr();
    }

    AllocationRegion* MutableRegionFor(const void* p) {
      return const_cast<AllocationRegion*>(RegionFor(p));
    }

    const AllocationRegion* RegionFor(const void* p) const {
      auto entry =
          std::upper_bound(regions_.begin(), regions_.end(), p, &Comparator);

      if (entry != regions_.end()) {
        return &(*entry);
      }

      // LOG(ERROR) << "Could not find Region for " << p;
      return nullptr;
    }

   private:
    std::vector<AllocationRegion> regions_;
  };

  // Returns 'bytes' rounded up to the next highest kMinAllocationSize.
  static size_t RoundedBytes(size_t bytes);

  // Try to add a new memory region that can satisfy an allocation of
  // 'rounded_bytes' bytes.  Returns true on success and false on
  // failure.
  bool Extend(size_t alignment, size_t rounded_bytes);

 public:
  // Deallocate free regions to give back the memory to suballocator, so that
  // we can re-allocate a larger region.  The main use scenario of this function
  // is when OOM happens but we have free regions and the sum of sizes of free
  // regions and unallocated bytes is larger than the requested size, implying
  // (external) memory fragmentation.  Returns true if any free regions are
  // found and freed; false otherwise.
  bool DeallocateFreeRegions(size_t rounded_bytes);

  // deallcate all regions, no matter free or not, to reclaim memory.
  bool DeallocateAllRegions();

 private:
  // Helper function to deallocate regions.
  void DeallocateRegions(const std::unordered_set<void*>& region_ptrs);

  // Returns a pointer to an underlying allocated chunk of size
  // 'rounded_bytes'.
  void* FindChunkPtr(BinNum bin_num, size_t rounded_bytes, size_t num_bytes);

  // Splits the chunk specified by 'h' into two chunks, one at least
  // of size 'num_bytes'.
  void SplitChunk(ChunkHandle h, size_t num_bytes);

  // Merges the two chunk handles.  Requires that the chunks are
  // contiguous in their allocation.
  void Merge(ChunkHandle h, ChunkHandle h2);

  // Adds the chunk 'h' to the proper free bin.
  void InsertFreeChunkIntoBin(ChunkHandle h);

  // Removes the free chunk pointed to by 'c' from the set free_chunks.
  void RemoveFreeChunkIterFromBin(Bin::FreeChunkSet* free_chunks,
                                  const Bin::FreeChunkSet::iterator& c);

  // Removes a free chunk from the bin.
  void RemoveFreeChunkFromBin(ChunkHandle h);
  void MaybeRemoveFreeChunkFromBin(ChunkHandle h);

  // Removes the chunk metadata represented by 'h'.
  void DeleteChunk(ChunkHandle h);

  std::string RenderOccupancy();
  void DumpMemoryLog(size_t num_bytes);

  ChunkHandle AllocateChunk();
  void DeallocateChunk(ChunkHandle h);

  Chunk* ChunkFromHandle(ChunkHandle h);
  const Chunk* ChunkFromHandle(ChunkHandle h) const;

  void MarkFree(ChunkHandle h);

  ChunkHandle TryToCoalesce(ChunkHandle h, bool ignore_freed_at);

  // Fragmentation is calculated as the reverse ratio of the largest free chunk
  // size over total free memory, and returns a value within [0, 1].
  double GetFragmentation();

  // Information about a Bin that is useful for debugging.
  struct BinDebugInfo {
    size_t total_bytes_in_use = 0;
    size_t total_bytes_in_bin = 0;
    size_t total_requested_bytes_in_use = 0;
    size_t total_chunks_in_use = 0;
    size_t total_chunks_in_bin = 0;
  };

  // Computes and returns a BinDebugInfo for each Bin.
  std::array<BinDebugInfo, kNumBins> get_bin_debug_info();

  // Structures immutable after construction
  size_t memory_limit_ = 0;

  int device_id_ = -1;

  inline int Log2FloorNonZeroSlow(uint64_t n) {
    int r = 0;
    while (n > 0) {
      r++;
      n >>= 1;
    }
    return r - 1;
  }

  // Returns floor(log2(n)).
  inline int Log2FloorNonZero(uint64_t n) {
#if defined(__GNUC__)
    return 63 ^ __builtin_clzll(n);
#elif defined(PLATFORM_WINDOWS) && (_WIN64)
    unsigned long index;
    _BitScanReverse64(&index, n);
    return index;
#else
    return Log2FloorNonZeroSlow(n);
#endif
  }

  // Map from bin size to Bin
  Bin* BinFromIndex(BinNum index) {
    return reinterpret_cast<Bin*>(&(bins_space_[index * sizeof(Bin)]));
  }
  size_t BinNumToSize(BinNum index) {
    return static_cast<size_t>(256) << index;
  }
  BinNum BinNumForSize(size_t bytes) {
    uint64_t v = std::max<size_t>(bytes, 256) >> kMinAllocationBits;
    int b = std::min(kNumBins - 1, Log2FloorNonZero(v));
    return b;
  }
  Bin* BinForSize(size_t bytes) { return BinFromIndex(BinNumForSize(bytes)); }

  char bins_space_[sizeof(Bin) * kNumBins];

  const BFCAllocator::Options opts_;

  // The size of the current region allocation.
  size_t curr_region_allocation_bytes_;

  // The total number of allocated bytes by the allocator.
  size_t total_region_allocated_bytes_ = 0;

  // An indicator that expansion of a region has hit the limits
  // of the available memory.
  bool started_backpedal_ = false;

  // Whether the allocator will coalesce adjacent sub allocator provided
  // AllocationRegions. This may be disabled if discrete sub allocator
  // regions can't be treated as contiguous (e.g. if the allocation refers to
  // device visible memory which is not adjacent to the other region in the
  // device's address space).
  const bool coalesce_regions_;

 public:
  std::unique_ptr<SubAllocator> sub_allocator_;

 private:
  std::string name_;

  std::atomic<uint64_t> safe_frontier_ = {0};

  // Structures mutable after construction
  mutable std::mutex lock_;
  RegionManager region_manager_;

  std::vector<Chunk> chunks_;

  // Pointer to head of linked list of free Chunks
  ChunkHandle free_chunks_list_;

  // Counter containing the next unique identifier to assign to a
  // newly-created chunk.
  int64_t next_allocation_id_;

  // Stats.
  AllocatorStats stats_;

  DISABLE_COPY_AND_ASSIGN(BFCAllocatorImpl);
};

constexpr BFCAllocatorImpl::ChunkHandle BFCAllocatorImpl::kInvalidChunkHandle;

BFCAllocatorImpl::BFCAllocatorImpl(std::unique_ptr<SubAllocator> sub_allocator,
                                   size_t total_memory, const std::string& name,
                                   int device_id,
                                   const BFCAllocator::Options& opts)
    : opts_(opts),
      coalesce_regions_(sub_allocator->SupportsCoalescing()),
      sub_allocator_(std::move(sub_allocator)),
      name_(name),
      device_id_(device_id),
      free_chunks_list_(kInvalidChunkHandle),
      next_allocation_id_(1) {
  if (opts.allow_growth) {
    // 2MiB smallest initial allocation, unless total memory available
    // is less.
    curr_region_allocation_bytes_ =
        RoundedBytes(std::min(total_memory, size_t{2 << 20}));
  } else {
    curr_region_allocation_bytes_ = RoundedBytes(total_memory);
  }

  // Allocate the requested amount of memory.
  memory_limit_ = total_memory;
  stats_.bytes_limit = static_cast<int64_t>(total_memory);

  // Create a bunch of bins of various good sizes.

  // We create bins to fit all possible ranges that cover the
  // memory_limit_ starting from allocations up to 256 bytes to
  // allocations up to (and including) the memory limit.
  // LOG(DBG) << "Creating new BFCAllocatorImpl named: " << name;
  for (BinNum b = 0; b < kNumBins; b++) {
    size_t bin_size = BinNumToSize(b);
    // LOG(DBG) << "Creating bin of max chunk size " << bin_size;
    new (BinFromIndex(b)) Bin(this, bin_size);
    AS_ENFORCE(BinForSize(bin_size) == BinFromIndex(b));
    AS_ENFORCE(BinForSize(bin_size + 255) == BinFromIndex(b));
    AS_ENFORCE(BinForSize(bin_size * 2 - 1) == BinFromIndex(b));
    if (b + 1 < kNumBins) {
      AS_ENFORCE(BinForSize(bin_size * 2) != BinFromIndex(b));
    }
  }
}

BFCAllocatorImpl::~BFCAllocatorImpl() {
  return;  // 调用该析构说明进程已经结束了，那么意味着什么都不需要做
#if BFC_FREE_MEMORY
  // FIXME: do real bfc free memory!!!
  // Return memory back.

  std::lock_guard<std::mutex> l(lock_);
  LOG(INFO) << "BFC_Allocator: Release All Memory !!! \n Number of regions "
               "allocated: "
            << region_manager_.regions().size();
  for (const auto& region : region_manager_.regions()) {
    sub_allocator_->free(region.ptr(), region.memory_size());
  }

  for (BinNum b = 0; b < kNumBins; b++) {
    BinFromIndex(b)->~Bin();
  }
#endif
}

BFCAllocatorImpl::Chunk* BFCAllocatorImpl::ChunkFromHandle(ChunkHandle h) {
  // DCHECK_GE(h, 0);
  // DCHECK_LT(h, static_cast<int>(chunks_.size()));
  return &(chunks_[h]);
}

const BFCAllocatorImpl::Chunk* BFCAllocatorImpl::ChunkFromHandle(
    ChunkHandle h) const {
  // DCHECK_GE(h, 0);
  // DCHECK_LT(h, static_cast<int>(chunks_.size()));
  return &(chunks_[h]);
}

bool BFCAllocatorImpl::Extend(size_t alignment, size_t rounded_bytes) {
  size_t available_bytes = memory_limit_ - total_region_allocated_bytes_;
  // Rounds available_bytes down to the nearest multiple of kMinAllocationSize.
  available_bytes = (available_bytes / kMinAllocationSize) * kMinAllocationSize;

  // Do we have enough space to handle the client's request?
  // If not, fail immediately.
  if (rounded_bytes > available_bytes) {
    return false;
  }

  // If curr_region_allocation_bytes_ is not enough to satisfy the
  // allocation, keep multiplying by a power of two until that is
  // sufficient.
  bool increased_allocation = false;
  while (rounded_bytes > curr_region_allocation_bytes_) {
    curr_region_allocation_bytes_ *= 2;
    increased_allocation = true;
  }

  // Try allocating.
  size_t bytes = std::min(curr_region_allocation_bytes_, available_bytes);
  size_t bytes_received;
  void* mem_addr = sub_allocator_->allocate(alignment, bytes, &bytes_received);

  if (mem_addr == nullptr) {
    return false;
  }

  if (!increased_allocation) {
    // Increase the region size of the next required allocation.
    curr_region_allocation_bytes_ *= 2;
  }

  // LOG(DBG) << "Extending allocation by " << bytes_received << " bytes for "
  // << Name() << ".";

  total_region_allocated_bytes_ += bytes_received;
  // LOG(DBG) << "Total allocated bytes: " << total_region_allocated_bytes_;

  // LOG(DBG) << "Allocated memory at " << mem_addr << " to "
  // << static_cast<void*>(static_cast<char*>(mem_addr) + bytes_received);

  AllocationRegion* maybe_extended_region = nullptr;
  if (coalesce_regions_) {
    maybe_extended_region =
        region_manager_.AddOrExtendAllocationRegion(mem_addr, bytes_received);
  } else {
    region_manager_.AddAllocationRegion(mem_addr, bytes_received);
  }

  // Create one large chunk for the whole memory space that will
  // be chunked later.
  ChunkHandle h = AllocateChunk();
  BFCAllocatorImpl::Chunk* c = ChunkFromHandle(h);
  c->ptr = mem_addr;
  c->size = bytes_received;
  c->allocation_id = -1;
  c->prev = kInvalidChunkHandle;
  c->next = kInvalidChunkHandle;
  c->freed_at_count = 0;

  region_manager_.set_handle(c->ptr, h);

  // If the region was extended, then there exists a previous chunk that should
  // be linked to the new chunk.
  if (maybe_extended_region != nullptr) {
    ChunkHandle prev =
        maybe_extended_region->get_handle(maybe_extended_region->ptr());
    BFCAllocatorImpl::Chunk* prev_chunk = ChunkFromHandle(prev);
    // Find the last recorded chunk in the extended region.
    while (prev_chunk->next != kInvalidChunkHandle) {
      prev = prev_chunk->next;
      prev_chunk = ChunkFromHandle(prev);
    }
    c->prev = prev;
    prev_chunk->next = h;
  }

  // Maybe merge adjacent chunks and insert the chunk into the right bin.
  InsertFreeChunkIntoBin(TryToCoalesce(h, /*ignore_freed_at=*/false));

  return true;
}

BFCAllocatorImpl::ChunkHandle BFCAllocatorImpl::AllocateChunk() {
  if (free_chunks_list_ != kInvalidChunkHandle) {
    ChunkHandle h = free_chunks_list_;
    Chunk* c = ChunkFromHandle(h);
    free_chunks_list_ = c->next;
    return h;
  } else {
    ChunkHandle h = chunks_.size();
    chunks_.resize(h + 1);
    return h;
  }
}

void BFCAllocatorImpl::DeallocateChunk(ChunkHandle h) {
  Chunk* c = ChunkFromHandle(h);
  c->allocation_id = -1;
  c->bin_num = kInvalidBinNum;
  c->next = free_chunks_list_;
  free_chunks_list_ = h;
}

void* BFCAllocatorImpl::AllocateRaw(size_t unused_alignment, size_t num_bytes) {
  // LOG(DBG) << "AllocateRaw " << Name() << "  " << num_bytes;
#ifdef NDEBUG
  bool dump_log_on_failure = false;
#else
#ifdef BFC_DEBUG
  bool dump_log_on_failure = true;
#else
  bool dump_log_on_failure = false;
#endif
#endif

  void* result =
      AllocateRawInternal(unused_alignment, num_bytes, dump_log_on_failure);
  if (result == nullptr) {
    LOG(ERROR) << "Allocator (" << Name() << ") ran out of memory trying "
               << "to allocate " << num_bytes
               << ", in MB: " << num_bytes / (1024 * 1024)
               << ". The caller indicates that this is not a failure, but"
               << " may mean that there could be performance gains if more"
               << " memory were available.";
  }
  return result;
}

// static
size_t BFCAllocatorImpl::RoundedBytes(size_t bytes) {
  size_t rounded_bytes =
      (kMinAllocationSize *
       ((bytes + kMinAllocationSize - 1) / kMinAllocationSize));
  // DCHECK_EQ(size_t{0}, rounded_bytes % kMinAllocationSize);
  return rounded_bytes;
}

/**
 * free all regions, no matter free or not.
 **/
bool BFCAllocatorImpl::DeallocateAllRegions() {
  LOG(INFO) << "DeallocateAllRegions: all regions will be free.";
  std::unordered_set<void*> region_ptrs;
  for (const AllocationRegion& region : region_manager_.regions()) {
    region_ptrs.insert(region.ptr());
  }

  DeallocateRegions(region_ptrs);
  return true;
}

bool BFCAllocatorImpl::DeallocateFreeRegions(size_t rounded_bytes) {
  // Do nothing if garbage collection is off.
  if (!opts_.garbage_collection) {
    LOG(INFO) << "BFC: DeallocateFreeRegions: ignore because garbage "
                 "collection is not on";
    return false;
  }

  // Searching for free regions.
  std::unordered_set<void*> free_region_ptrs;
  size_t total_free_bytes = 0;
  for (const AllocationRegion& region : region_manager_.regions()) {
    ChunkHandle h = region_manager_.get_handle(region.ptr());
    bool any_use = false;
    while (h != kInvalidChunkHandle) {
      const Chunk* c = ChunkFromHandle(h);
      if (c->in_use()) {
        any_use = true;
        break;
      }
      h = c->next;
    }

    if (!any_use) {
      // LOG(DBG) << "Found free region with ptr = " << region.ptr();
      free_region_ptrs.insert(region.ptr());
      total_free_bytes += region.memory_size();
    }
  }

  if (total_free_bytes == 0) {
    LOG(INFO) << "total free bytes == 0, ignore DeallocateFreeRegions";
    return false;
  }
  LOG(INFO) << "BFC: can reclaim " << total_free_bytes / 1024 << " K Memory";

  // Rough estimation to check whether deallocation can help.
  size_t available_bytes =
      memory_limit_ - total_region_allocated_bytes_ + total_free_bytes;
  if (rounded_bytes > available_bytes) {
    LOG(INFO) << "BFC: rounded_bytes: " << rounded_bytes
              << " available bytes: " << available_bytes << " ignore...";
    return false;
  }

  LOG(WARNING) << "Garbage collection: deallocate free memory regions"
               << " (i.e., allocations) so that we can re-allocate a larger"
               << " region to avoid OOM due to memory fragmentation. If you"
               << " see this message frequently, you are running near the"
               << " threshold of the available device memory and re-allocation"
               << " may incur great performance overhead. You may try smaller"
               << " batch sizes to observe the performance impact."
               << " Set TF_ENABLE_GPU_GARBAGE_COLLECTION=false if you'd like to"
               << " disable this feature.";

  // Deallocate free regions.
  DeallocateRegions(free_region_ptrs);

  return true;
}

void BFCAllocatorImpl::DeallocateRegions(
    const std::unordered_set<void*>& region_ptrs) {
  // Explicitly remove the const qualifier as some compilers disallow passing
  // const_iterator to std::vector::erase(), which is used in
  // RemoveAllocationRegion().
  auto regions =
      const_cast<std::vector<AllocationRegion>*>(&region_manager_.regions());
  auto it = regions->begin();
  while (it != regions->end()) {
    if (!region_ptrs.count(it->ptr())) {
      ++it;
      continue;
    }

    // LOG(DBG) << "Deallocate region with ptr = " << it->ptr();
    // Remove all chunk registrations from Bins.
    ChunkHandle h = region_manager_.get_handle(it->ptr());
    while (h != kInvalidChunkHandle) {
      const Chunk* c = ChunkFromHandle(h);
      if (c->bin_num != kInvalidBinNum) {
        RemoveFreeChunkFromBin(h);
      }
      auto h_to_delete = h;
      h = c->next;
      DeleteChunk(h_to_delete);
    }

    // Deallocate the memory.
    sub_allocator_->free(it->ptr(), it->memory_size());
    total_region_allocated_bytes_ -= it->memory_size();
    it = region_manager_.RemoveAllocationRegion(it);
  }
}

void* BFCAllocatorImpl::AllocateRawInternal(size_t unused_alignment,
                                            size_t num_bytes,
                                            bool dump_log_on_failure) {
  if (num_bytes == 0) {
    // LOG(DBG) << "tried to allocate 0 bytes";
    return nullptr;
  }
  // First, always allocate memory of at least kMinAllocationSize
  // bytes, and always allocate multiples of kMinAllocationSize bytes
  // so all memory addresses are nicely byte aligned.
  size_t rounded_bytes = RoundedBytes(num_bytes);

  // The BFC allocator tries to find the best fit first.
  BinNum bin_num = BinNumForSize(rounded_bytes);

  std::lock_guard<std::mutex> l(lock_);
  void* ptr = FindChunkPtr(bin_num, rounded_bytes, num_bytes);
  if (ptr != nullptr) {
    return ptr;
  }

  // Try to extend
  if (Extend(unused_alignment, rounded_bytes)) {
    ptr = FindChunkPtr(bin_num, rounded_bytes, num_bytes);
    if (ptr != nullptr) {
      return ptr;
    }
  }

  // Reaching this point means that no chunks can satisfy the request. Also,
  // the unallocated bytes cannot satisfy the request. Before giving up, let's
  // try deallocating free regions so that suballocator can combine them with
  // the unallocated bytes and form a larger region.
  if (DeallocateFreeRegions(rounded_bytes) &&
      Extend(unused_alignment, rounded_bytes)) {
    ptr = FindChunkPtr(bin_num, rounded_bytes, num_bytes);
    if (ptr != nullptr) {
      return ptr;
    }
  }

  // We searched all bins for an existing free chunk to use and
  // couldn't find one.  This means we must have run out of memory,
  // Dump the memory log for analysis.
  if (dump_log_on_failure) {
    LOG(WARNING) << "Allocator (" << Name() << ") ran out of memory trying "
                 << "to allocate " << num_bytes << " (rounded to "
                 << rounded_bytes << ")"
                 << "\nCurrent allocation summary follows.";
    DumpMemoryLog(rounded_bytes);
    LOG(WARNING) << RenderOccupancy();
  }
  return nullptr;
}

int64_t BFCAllocatorImpl::LargestFreeChunk() {
  for (int i = kNumBins - 1; i >= 0; i--) {
    if (!BinFromIndex(i)->free_chunks.empty()) {
      return ChunkFromHandle(*BinFromIndex(i)->free_chunks.rbegin())->size;
    }
  }
  return 0;
}

double BFCAllocatorImpl::GetFragmentation() {
  int64_t bytes_available = total_region_allocated_bytes_ - stats_.bytes_in_use;
  // DCHECK_GT(bytes_available, 0);
  return static_cast<double>(bytes_available - LargestFreeChunk()) /
         bytes_available;
}

void* BFCAllocatorImpl::FindChunkPtr(BinNum bin_num, size_t rounded_bytes,
                                     size_t num_bytes) {
  // First identify the first bin that could satisfy rounded_bytes.
  for (; bin_num < kNumBins; bin_num++) {
    // Start searching from the first bin for the smallest chunk that fits
    // rounded_bytes.
    Bin* b = BinFromIndex(bin_num);
    for (auto citer = b->free_chunks.begin(); citer != b->free_chunks.end();
         ++citer) {
      const BFCAllocatorImpl::ChunkHandle h = (*citer);
      BFCAllocatorImpl::Chunk* chunk = ChunkFromHandle(h);
      // DCHECK(!chunk->in_use());
      if (chunk->size >= rounded_bytes) {
        // We found an existing chunk that fits us that wasn't in use, so remove
        // it from the free bin structure prior to using.
        RemoveFreeChunkIterFromBin(&b->free_chunks, citer);

        // If we can break the size of the chunk into two reasonably large
        // pieces, do don't waste more than max_internal_fragmentation_bytes on
        // padding. If this threshold is not set by the user, then use 128MB as
        // the default.
        const int64_t max_internal_fragmentation_bytes =
            (opts_.fragmentation_fraction > 0.0)
                ? opts_.fragmentation_fraction * memory_limit_
                : 128 << 20;

        if (chunk->size >= rounded_bytes * 2 ||
            static_cast<int64_t>(chunk->size) - rounded_bytes >=
                max_internal_fragmentation_bytes) {
          SplitChunk(h, rounded_bytes);
          chunk = ChunkFromHandle(h);  // Update chunk pointer in case it moved
        }

        // The requested size of the returned chunk is what the user
        // has allocated.
        chunk->requested_size = num_bytes;
        // Assign a unique id and increment the id counter, marking the
        // chunk as being in use.
        chunk->allocation_id = next_allocation_id_++;

        // Update stats.
        ++stats_.num_allocs;
        stats_.bytes_in_use += chunk->size;
        // if (stats_.bytes_in_use > stats_.peak_bytes_in_use) {
        // LOG(DBG) << "New Peak memory usage of " << stats_.bytes_in_use << "
        // bytes for "
        // << Name();
        // }
        stats_.peak_bytes_in_use =
            std::max(stats_.peak_bytes_in_use, stats_.bytes_in_use);
        stats_.largest_alloc_size =
            std::max<std::size_t>(stats_.largest_alloc_size, chunk->size);

        // LOG(DBG) << "Returning: " << chunk->ptr;
        // if (VLOG_IS_ON(4)) {
        // LOG(INFO) << "A: " << RenderOccupancy();
        // }
        return chunk->ptr;
      }
    }
  }

  return nullptr;
}

void BFCAllocatorImpl::SplitChunk(BFCAllocatorImpl::ChunkHandle h,
                                  size_t num_bytes) {
  // Allocate the new chunk before we do any ChunkFromHandle
  ChunkHandle h_new_chunk = AllocateChunk();

  Chunk* c = ChunkFromHandle(h);
  AS_ENFORCE(!c->in_use() && (c->bin_num == kInvalidBinNum));

  // Create a new chunk starting num_bytes after c
  BFCAllocatorImpl::Chunk* new_chunk = ChunkFromHandle(h_new_chunk);
  new_chunk->ptr = static_cast<void*>(static_cast<char*>(c->ptr) + num_bytes);
  region_manager_.set_handle(new_chunk->ptr, h_new_chunk);

  // Set the new sizes of the chunks.
  new_chunk->size = c->size - num_bytes;
  c->size = num_bytes;

  // The new chunk is not in use.
  new_chunk->allocation_id = -1;

  // It inherits the freed time.
  new_chunk->freed_at_count = c->freed_at_count;

  // Maintain the pointers.
  // c <-> c_neighbor becomes
  // c <-> new_chunk <-> c_neighbor
  BFCAllocatorImpl::ChunkHandle h_neighbor = c->next;
  new_chunk->prev = h;
  new_chunk->next = h_neighbor;
  c->next = h_new_chunk;
  if (h_neighbor != kInvalidChunkHandle) {
    Chunk* c_neighbor = ChunkFromHandle(h_neighbor);
    c_neighbor->prev = h_new_chunk;
  }

  // Add the newly free chunk to the free bin.
  InsertFreeChunkIntoBin(h_new_chunk);
}

void BFCAllocatorImpl::DeallocateRaw(void* ptr) {
  // LOG(DBG) << "DeallocateRaw " << Name() << " " << (ptr ? RequestedSize(ptr)
  // : 0);
  DeallocateRawInternal(ptr);
}

void BFCAllocatorImpl::DeallocateRawInternal(void* ptr) {
  if (ptr == nullptr) {
    // LOG(DBG) << "tried to deallocate nullptr";
    return;
  }
  std::lock_guard<std::mutex> l(lock_);

  // Find the chunk from the ptr.
  BFCAllocatorImpl::ChunkHandle h = region_manager_.get_handle(ptr);
  if (h == 0) return;
  AS_ENFORCE(h != kInvalidChunkHandle);
  // Record chunk information before it's freed.
  Chunk* chunk = ChunkFromHandle(h);
  void* chunk_ptr = chunk->ptr;
  int64_t req_bytes = chunk->requested_size;
  int64_t alloc_bytes = chunk->size;

  MarkFree(h);

  // Consider coalescing it.
  InsertFreeChunkIntoBin(TryToCoalesce(h, false));

  // if (VLOG_IS_ON(4)) {
  // LOG(INFO) << "F: " << RenderOccupancy();
  // }
}

// Merges h1 and h2 when Chunk(h1)->next is h2 and Chunk(h2)->prev is c1.
// We merge Chunk(h2) into Chunk(h1).
void BFCAllocatorImpl::Merge(BFCAllocatorImpl::ChunkHandle h1,
                             BFCAllocatorImpl::ChunkHandle h2) {
  Chunk* c1 = ChunkFromHandle(h1);
  Chunk* c2 = ChunkFromHandle(h2);
  // We can only merge chunks that are not in use.
  AS_ENFORCE(!c1->in_use() && !c2->in_use());

  // c1's prev doesn't change, still points to the same ptr, and is
  // still not in use.

  // Fix up neighbor pointers
  //
  // c1 <-> c2 <-> c3 should become
  // c1 <-> c3

  BFCAllocatorImpl::ChunkHandle h3 = c2->next;
  c1->next = h3;
  AS_ENFORCE(c2->prev == h1);
  if (h3 != kInvalidChunkHandle) {
    BFCAllocatorImpl::Chunk* c3 = ChunkFromHandle(h3);
    c3->prev = h1;
  }

  // Set the new size
  c1->size += c2->size;

  // Pick latest free time.
  c1->freed_at_count = std::max(c1->freed_at_count, c2->freed_at_count);

  DeleteChunk(h2);
}

void BFCAllocatorImpl::DeleteChunk(ChunkHandle h) {
  // Delete h and cleanup all state
  Chunk* c = ChunkFromHandle(h);
  //  LOG(DBG) << "Removing: " << c->ptr;
  region_manager_.erase(c->ptr);
  DeallocateChunk(h);
}

void BFCAllocatorImpl::InsertFreeChunkIntoBin(BFCAllocatorImpl::ChunkHandle h) {
  Chunk* c = ChunkFromHandle(h);
  AS_ENFORCE(!c->in_use() && (c->bin_num == kInvalidBinNum));
  BinNum bin_num = BinNumForSize(c->size);
  Bin* new_bin = BinFromIndex(bin_num);
  c->bin_num = bin_num;
  new_bin->free_chunks.insert(h);
}

void BFCAllocatorImpl::RemoveFreeChunkIterFromBin(
    BFCAllocatorImpl::Bin::FreeChunkSet* free_chunks,
    const BFCAllocatorImpl::Bin::FreeChunkSet::iterator& citer) {
  ChunkHandle h = *citer;
  Chunk* c = ChunkFromHandle(h);
  AS_ENFORCE(!c->in_use() && (c->bin_num != kInvalidBinNum));
  free_chunks->erase(citer);
  c->bin_num = kInvalidBinNum;
}

void BFCAllocatorImpl::RemoveFreeChunkFromBin(BFCAllocatorImpl::ChunkHandle h) {
  Chunk* c = ChunkFromHandle(h);
  AS_ENFORCE(!c->in_use() && (c->bin_num != kInvalidBinNum));
  AS_ENFORCE(BinFromIndex(c->bin_num)->free_chunks.erase(h) > 0,
             "Could not find chunk in bin");
  c->bin_num = kInvalidBinNum;
}

void BFCAllocatorImpl::MarkFree(BFCAllocatorImpl::ChunkHandle h) {
  Chunk* c = ChunkFromHandle(h);
  AS_ENFORCE(c->in_use() && (c->bin_num == kInvalidBinNum));

  // Mark the chunk as no longer in use.
  c->allocation_id = -1;

  // Updates the stats.
  stats_.bytes_in_use -= c->size;
}

BFCAllocatorImpl::ChunkHandle BFCAllocatorImpl::TryToCoalesce(
    ChunkHandle h, bool ignore_freed_at) {
  Chunk* c = ChunkFromHandle(h);
  if ((!ignore_freed_at) && c->freed_at_count > 0) return h;
  ChunkHandle coalesced_chunk = h;

  // If the next chunk is free, merge it into c and delete it.
  if (c->next != kInvalidChunkHandle && !ChunkFromHandle(c->next)->in_use()) {
    Chunk* n = ChunkFromHandle(c->next);
    if ((n->freed_at_count == 0) || ignore_freed_at) {
      // LOG(DBG) << "Merging c->next " << n->ptr << " with c " << c->ptr;
      RemoveFreeChunkFromBin(c->next);
      Merge(h, c->next);
    }
  }

  // If the previous chunk is free, merge c into it and delete c.
  if (c->prev != kInvalidChunkHandle && !ChunkFromHandle(c->prev)->in_use()) {
    Chunk* n = ChunkFromHandle(c->prev);
    if ((n->freed_at_count == 0) || ignore_freed_at) {
      // LOG(DBG) << "Merging c " << c->ptr << " into c->prev " << n->ptr;
      coalesced_chunk = c->prev;
      RemoveFreeChunkFromBin(c->prev);
      Merge(c->prev, h);
    }
  }

  return coalesced_chunk;
}

bool BFCAllocatorImpl::TracksAllocationSizes() const { return true; }

size_t BFCAllocatorImpl::RequestedSize(const void* ptr) const {
  AS_ENFORCE(ptr);
  std::lock_guard<std::mutex> l(lock_);
  BFCAllocatorImpl::ChunkHandle h = region_manager_.get_handle(ptr);
  AS_ENFORCE(h != kInvalidChunkHandle,
             "Asked for requested size of pointer we never allocated: ", ptr);
  const BFCAllocatorImpl::Chunk* c = ChunkFromHandle(h);
  return c->requested_size;
}

size_t BFCAllocatorImpl::AllocatedSize(const void* ptr) const {
  std::lock_guard<std::mutex> l(lock_);
  BFCAllocatorImpl::ChunkHandle h = region_manager_.get_handle(ptr);
  AS_ENFORCE(h != kInvalidChunkHandle,
             "Asked for allocated size of pointer we never allocated: ", ptr);
  const BFCAllocatorImpl::Chunk* c = ChunkFromHandle(h);
  return c->size;
}

int64_t BFCAllocatorImpl::AllocationId(const void* ptr) const {
  std::lock_guard<std::mutex> l(lock_);
  BFCAllocatorImpl::ChunkHandle h = region_manager_.get_handle(ptr);
  AS_ENFORCE(h != kInvalidChunkHandle,
             "Asked for allocation id of pointer we never allocated: ", ptr);
  const BFCAllocatorImpl::Chunk* c = ChunkFromHandle(h);
  return c->allocation_id;
}

namespace {

void RenderRegion(char* rendered, const size_t resolution,
                  const size_t total_render_size, const size_t offset,
                  const void* base_ptr, const void* ptr, const size_t size,
                  const char c) {
  const char* base_ptr_c = static_cast<const char*>(base_ptr);
  const char* ptr_c = static_cast<const char*>(ptr);

  size_t start_location =
      ((ptr_c - base_ptr_c + offset) * resolution) / total_render_size;
  AS_ENFORCE(start_location >= 0);
  AS_ENFORCE(start_location < resolution);
  size_t end_location =
      ((ptr_c + size - 1 - base_ptr_c + offset) * resolution) /
      total_render_size;
  AS_ENFORCE(end_location >= 0);
  AS_ENFORCE(end_location < resolution);

  for (size_t i = start_location; i <= end_location; ++i) {
    rendered[i] = c;
  }
}

}  // namespace

std::string BFCAllocatorImpl::RenderOccupancy() {
  // Make a buffer for the ASCII-art representation.
  const size_t resolution = 100;
  char rendered[resolution];

  // Compute the total region size to render over
  size_t total_region_size = 0;
  for (const auto& region : region_manager_.regions()) {
    total_region_size += region.memory_size();
  }

  if (total_region_size == 0) {
    return "<allocator contains no memory>";
  }

  // Start out with everything empty
  RenderRegion(rendered, resolution, total_region_size, 0, nullptr, nullptr,
               total_region_size, '_');

  size_t region_offset = 0;
  for (const auto& region : region_manager_.regions()) {
    ChunkHandle h = region_manager_.get_handle(region.ptr());
    // Then render each chunk left to right.
    while (h != kInvalidChunkHandle) {
      Chunk* c = ChunkFromHandle(h);
      if (c->in_use()) {
        // Render the wasted space
        size_t wasted = c->size - c->requested_size;
        if (wasted > 0) {
          RenderRegion(rendered, resolution, total_region_size,
                       region_offset + c->requested_size, region.ptr(), c->ptr,
                       wasted, 'x');
        }
        // Then the occupied space
        RenderRegion(rendered, resolution, total_region_size, region_offset,
                     region.ptr(), c->ptr, c->requested_size, '*');
      }
      h = c->next;
    }
    region_offset += region.memory_size();
  }

  return std::string(rendered, resolution);
}

void BFCAllocatorImpl::DumpMemoryLog(size_t num_bytes) {
  const std::array<BinDebugInfo, kNumBins> bin_infos = get_bin_debug_info();
  LOG(INFO) << "BFCAllocatorImpl dump for " << Name();
  for (BinNum bin_num = 0; bin_num < kNumBins; bin_num++) {
    Bin* b = BinFromIndex(bin_num);
    const BinDebugInfo& bin_info = bin_infos[bin_num];
    AS_ENFORCE(b->free_chunks.size() ==
               bin_info.total_chunks_in_bin - bin_info.total_chunks_in_use);

    LOG(INFO) << "Bin (" << b->bin_size
              << "): \tTotal Chunks: " << bin_info.total_chunks_in_bin
              << ", Chunks in use: " << bin_info.total_chunks_in_use << ". "
              << bin_info.total_bytes_in_bin << " allocated for chunks. "
              << bin_info.total_bytes_in_use << " in use in bin. "
              << bin_info.total_requested_bytes_in_use
              << " client-requested in use in bin.";
  }

  // Find the bin that we would have liked to allocate in, so we
  // can get some further analysis about fragmentation.
  Bin* b = BinForSize(num_bytes);

  LOG(INFO) << "Bin for " << num_bytes << " was " << b->bin_size
            << ", Chunk State: ";

  for (ChunkHandle h : b->free_chunks) {
    Chunk* c = ChunkFromHandle(h);
    LOG(INFO) << c->DebugString(this, true);
  }

  // Next show the chunks that are in use, and also summarize their
  // number by size.
  std::map<size_t, int> in_use_by_size;
  for (const auto& region : region_manager_.regions()) {
    LOG(INFO) << "Next region of size " << region.memory_size();
    ChunkHandle h = region_manager_.get_handle(region.ptr());
    while (h != kInvalidChunkHandle) {
      const Chunk* c = ChunkFromHandle(h);
      if (c->in_use()) {
        in_use_by_size[c->size]++;
      }
      LOG(INFO) << (c->in_use() ? "InUse" : "Free ") << " at " << c->ptr
                << " of size " << c->size << " next " << c->next;
      h = c->next;
    }
  }

  LOG(INFO) << "     Summary of in-use Chunks by size: ";
  size_t total_bytes = 0;
  for (auto& it : in_use_by_size) {
    LOG(INFO) << it.second << " Chunks of size " << it.first << " totalling "
              << it.first * it.second;
    total_bytes += (it.first * it.second);
  }
  LOG(INFO) << "Sum Total of in-use chunks: " << total_bytes;
  LOG(INFO) << "total_region_allocated_bytes_: "
            << total_region_allocated_bytes_
            << " memory_limit_: " << memory_limit_ << " available bytes: "
            << (memory_limit_ - total_region_allocated_bytes_)
            << " curr_region_allocation_bytes_: "
            << curr_region_allocation_bytes_;
  LOG(INFO) << "Stats: \n" << stats_.DebugString();
}

AllocatorStats BFCAllocatorImpl::GetStats() {
  std::lock_guard<std::mutex> l(lock_);
  return stats_;
}

bool BFCAllocatorImpl::ClearStats() {
  std::lock_guard<std::mutex> l(lock_);
  stats_.num_allocs = 0;
  stats_.peak_bytes_in_use = stats_.bytes_in_use;
  stats_.largest_alloc_size = 0;
  return true;
}

std::array<BFCAllocatorImpl::BinDebugInfo, BFCAllocatorImpl::kNumBins>
BFCAllocatorImpl::get_bin_debug_info() {
  std::array<BinDebugInfo, kNumBins> bin_infos;
  for (const auto& region : region_manager_.regions()) {
    ChunkHandle h = region_manager_.get_handle(region.ptr());
    while (h != kInvalidChunkHandle) {
      const Chunk* c = ChunkFromHandle(h);
      BinNum bin_num = BinNumForSize(c->size);
      BinDebugInfo& bin_info = bin_infos[bin_num];
      bin_info.total_bytes_in_bin += c->size;
      bin_info.total_chunks_in_bin++;
      if (c->in_use()) {
        bin_info.total_bytes_in_use += c->size;
        bin_info.total_requested_bytes_in_use += c->requested_size;
        bin_info.total_chunks_in_use++;
      } else {
        Bin* bin = BinFromIndex(bin_num);
        AS_ENFORCE(bin->free_chunks.count(h) == 1);
        AS_ENFORCE(c->bin_num == bin_num);
      }
      h = c->next;
    }
  }
  return bin_infos;
}

DeviceType BFCAllocatorImpl::GetMemoryType() const {
  return sub_allocator_->GetMemoryType();
}

BFCAllocator::BFCAllocator(std::unique_ptr<SubAllocator> sub_allocator,
                           size_t total_memory, const std::string& name,
                           int device_id, const Options& opts)
    : impl_(std::make_unique<BFCAllocatorImpl>(
          std::move(sub_allocator), total_memory, name, device_id, opts)) {}

BFCAllocator::~BFCAllocator() {}

std::string BFCAllocator::name() { return impl_->Name(); }

DeviceType BFCAllocator::memory_type() { return impl_->GetMemoryType(); }

// void* BFCAllocator::allocate(uint64_t size, uint64_t alignment, const
// std::string& tensor_name) noexcept {
AsStatus BFCAllocator::Alloc(void** ptr, int64_t nbytes,
                             const std::string& name) {
#ifdef DEBUG_BFC

  int orig_device_id = -1;
  AS_CHECK_CUDA(cudaGetDevice(&orig_device_id));

  if (orig_device_id != impl_->GetDeviceId()) {
    LOG(ERROR) << "BFC Miss match with device: init with device id: "
               << impl_->GetDeviceId()
               << " , but current on: " << orig_device_id;
  }

#endif

  uint64_t alignment = 0;
  void* pp = impl_->AllocateRaw(alignment, nbytes);
  if (!pp) {
    LOG(ERROR) << "failed alloc " << nbytes << " bytes!" << std::endl;
    print_backtrace();
    return AsStatus::ALLSPARK_MEMORY_ERROR;
  }
  *ptr = pp;
  if (impl_->GetMemoryType() == DeviceType::CUDA) {
    util::RegisterMem(uint64_t(*ptr), name, nbytes, DeviceType::CUDA);
  }
  // LOG(INFO) << "success alloc " << nbytes << " bytes! name = " <<
  // name<<std::endl;
  return AsStatus::ALLSPARK_SUCCESS;
}

// void BFCAllocator::free(void* memory) noexcept {
AsStatus BFCAllocator::Free(void* ptr) {
  // LOG(INFO) << "enter BFCAllocator::Free " << std::hex << ptr << std::endl;
  if (impl_->GetMemoryType() == DeviceType::CUDA) {
    util::UnRegisterMem(uint64_t(ptr));
  }

#ifdef DEBUG_BFC
  int orig_device_id = -1;
  AS_CHECK_CUDA(cudaGetDevice(&orig_device_id));

  if (orig_device_id != impl_->GetDeviceId()) {
    LOG(ERROR) << "BFC Miss match with device: init with device id: "
               << impl_->GetDeviceId()
               << " , but current on: " << orig_device_id;
  }
#endif

  impl_->DeallocateRaw(ptr);
  return AsStatus::ALLSPARK_SUCCESS;
}

AllocatorStats BFCAllocator::GetStats() { return impl_->GetStats(); }

void BFCAllocator::RealFree() { impl_->DeallocateFreeRegions(0); }

void BFCAllocator::FreeAll() { impl_->DeallocateAllRegions(); }

class CPUSubAllocator : public SubAllocator {
 public:
  void* allocate(size_t alignment, size_t num_bytes,
                 size_t* bytes_received) override {
    *bytes_received = num_bytes;
    void* ret = nullptr;
    default_allocator_.Alloc(&ret, num_bytes, BFC_SPACE_NAME);
    return ret;
  }

  void free(void* ptr, size_t num_bytes) override {
    default_allocator_.Free(ptr);
  }

  bool SupportsCoalescing() const override { return false; }

  DeviceType GetMemoryType() const override { return DeviceType::CPU; }

 private:
  CPUAllocator default_allocator_;
};

#ifdef ENABLE_CUDA
class GPUSubAllocator : public SubAllocator {
 public:
  void* allocate(size_t alignment, size_t num_bytes,
                 size_t* bytes_received) override {
    *bytes_received = num_bytes;
    void* ret = nullptr;
    default_allocator_.Alloc(&ret, num_bytes, BFC_SPACE_NAME);
    return ret;
  }

  void free(void* ptr, size_t num_bytes) override {
    default_allocator_.Free(ptr);
  }

  bool SupportsCoalescing() const override { return false; }

  DeviceType GetMemoryType() const override { return DeviceType::CUDA; }

 private:
  CUDAAllocator default_allocator_{false};
};

class CPUPinnedSubAllocator : public SubAllocator {
 public:
  void* allocate(size_t alignment, size_t num_bytes,
                 size_t* bytes_received) override {
    *bytes_received = num_bytes;
    void* ret = nullptr;
    default_allocator_.Alloc(&ret, num_bytes, BFC_SPACE_NAME);
    return ret;
  }

  void free(void* ptr, size_t num_bytes) override {
    default_allocator_.Free(ptr);
  }

  bool SupportsCoalescing() const override { return false; }

  DeviceType GetMemoryType() const override { return DeviceType::CPU_PINNED; }

 private:
  CUDAHostAllocator default_allocator_;
};
#endif

std::unique_ptr<SubAllocator> CreateSubAllocator(DeviceType device_type) {
  if (device_type == DeviceType::CPU)
    return std::unique_ptr<SubAllocator>(new CPUSubAllocator());

#ifdef ENABLE_CUDA
  else if (device_type == DeviceType::CUDA)
    return std::unique_ptr<SubAllocator>(new GPUSubAllocator());
  else if (device_type == DeviceType::CPU_PINNED)
    return std::unique_ptr<SubAllocator>(new CPUPinnedSubAllocator());
#endif

  return nullptr;
}

AsStatus InitBFCAllocator(DeviceType device_type,
                          const std::vector<int>& device_ids) {
  const char* use_bfc = std::getenv("BFC_ALLOCATOR");
  if (use_bfc && std::string(use_bfc) == "OFF")  // default is ON
    return AsStatus::ALLSPARK_SUCCESS;

  std::unique_lock<std::mutex> lock(g_bfc_registry_lock);
  if (g_bfc_allocator_registry.reuse) {
    SweepBFCAllocator();
    // return AsStatus::ALLSPARK_SUCCESS;
  }
  if (g_bfc_allocator_registry.valid) {
    LOG(ERROR) << "Please call DestroyBFCAllocator before Re-InitBFCAllocator."
               << std::endl;
    return AsStatus::ALLSPARK_INVALID_CALL_ERROR;
  }

#ifdef ENABLE_CUDA
  assert(device_type == DeviceType::CUDA);  // tmp only support CUDA
  g_bfc_allocator_registry.device_type = DeviceType::CUDA;
  g_bfc_allocator_registry.device_ids = device_ids;  // save

  int orig_device_id;
  AS_CHECK_CUDA(cudaGetDevice(&orig_device_id));  // 主线程：保存现场
  for (int dev_id : device_ids) {
    size_t free_mem, total_mem;
    AS_CHECK_CUDA(cudaSetDevice(dev_id));
    AS_CHECK_CUDA(cudaMemGetInfo(&free_mem, &total_mem));
    LOG(INFO) << "InitBFCAllocator device_id=" << dev_id
              << ", mem_in_use(MB): " << (total_mem - free_mem) / (1024 * 1024)
              << ", free_mem(MB):" << free_mem / (1024 * 1024)
              << ", total_mem:" << total_mem / (1024 * 1024) << std::endl;
    const char* bfc_leftover_mb = std::getenv("BFC_LEFTOVER_MB");
    uint64_t leftover_bytes = 600ULL * 1024 * 1024;  // for pytorch & default
    if (bfc_leftover_mb) {
      try {
        leftover_bytes =
            std::stoull(std::string(bfc_leftover_mb)) * 1024 * 1024;
      } catch (const std::exception& e) {
        DLOG(WARNING) << "Invalid numeric format for BFC_LEFTOVER_MB, will use "
                         "default val: 0"
                      << std::endl;
      }
    }

    if (total_mem < leftover_bytes || free_mem < leftover_bytes) {
      LOG(ERROR) << "not enough cuda mem for device " << dev_id << std::endl;
      return AsStatus::ALLSPARK_MEMORY_ERROR;
    }
    float ratio = 0.9;
    const char* mem_ratio = std::getenv("BFC_MEM_RATIO");
    if (mem_ratio) {
      try {
        float ratio_tmp = std::stof(std::string(mem_ratio));
        if (ratio_tmp <= 0 || ratio_tmp > 1) {
          LOG(WARNING) << "invalid float range for env var BFC_MEM_RATIO: "
                       << mem_ratio << ", will use BFC_MEM_RATIO=" << ratio
                       << std::endl;
        } else {
          ratio = ratio_tmp;
        }
      } catch (std::exception& e) {
        LOG(WARNING) << "invalid float format for env var BFC_MEM_RATIO: "
                     << mem_ratio << ", will use BFC_MEM_RATIO=" << ratio
                     << std::endl;
      }
    }

    uint64_t max_mem_bytes = uint64_t((total_mem - leftover_bytes) * ratio);
    BFCAllocator::Options opt;
    const char* bfc_allow_growth = std::getenv("BFC_ALLOW_GROWTH");
    if (bfc_allow_growth &&
        std::string(bfc_allow_growth) == "OFF")  // default is ON
      opt.allow_growth = false;
    AsDevice device(device_type, dev_id);
    std::string name = "BfcAllocator_for_device_" + std::to_string(dev_id);
    auto allocator = std::make_shared<BFCAllocator>(
        CreateSubAllocator(device_type), max_mem_bytes, name, dev_id, opt);
    g_bfc_allocator_registry.allocator_map[device] = allocator;
  }
  AS_CHECK_CUDA(cudaSetDevice(orig_device_id));  // 主线程： 恢复现场
#endif

  g_bfc_allocator_registry.valid = true;
  return AsStatus::ALLSPARK_SUCCESS;
}

std::shared_ptr<Allocator> GetBFCAllocator(const DeviceType device_type) {
  assert(device_type == DeviceType::CUDA);  // tmp only support CUDA

  std::unique_lock<std::mutex> lock(g_bfc_registry_lock);

#ifdef ENABLE_CUDA
  int device_id;
  AS_CHECK_CUDA(cudaGetDevice(&device_id));

#ifdef MEM_DEBUG
  DLOG(INFO) << "GetBFCAllocator device_id=" << device_id << std::endl;
#endif
  AsDevice device(device_type, device_id);
  if (g_bfc_allocator_registry.allocator_map.count(device))
    return g_bfc_allocator_registry.allocator_map.at(device);
#endif

  return nullptr;
}

std::shared_ptr<Allocator> GetBFCAllocatorByDeviceId(
    const DeviceType device_type, const int device_id) {
  assert(device_type == DeviceType::CUDA);  // tmp only support CUDA

  std::unique_lock<std::mutex> lock(g_bfc_registry_lock);

#ifdef ENABLE_CUDA
#ifdef MEM_DEBUG
  DLOG(INFO) << "GetBFCAllocator device_id=" << device_id << std::endl;
#endif
  AsDevice device(device_type, device_id);
  if (g_bfc_allocator_registry.allocator_map.count(device))
    return g_bfc_allocator_registry.allocator_map.at(device);
#endif

  return nullptr;
}

void SweepBFCAllocator() {
  std::unique_lock<std::mutex> lock(g_bfc_registry_lock);
  for (int device_id : g_bfc_allocator_registry.device_ids) {
    AsDevice device(g_bfc_allocator_registry.device_type, device_id);
    g_bfc_allocator_registry.allocator_map.at(device)->RealFree();
  }
}

void DestroyBFCAllocator() {
  LOG(INFO) << " DestroyBFCAllocator called";
  std::unique_lock<std::mutex> lock(g_bfc_registry_lock);
  if (!g_bfc_allocator_registry.valid) {
    LOG(WARNING) << "Cannot call DestroyBFCAllocator before InitBFCAllocator."
                 << std::endl;
    return;
  }
#if BFC_FREE_MEMORY
  for (int device_id : g_bfc_allocator_registry.device_ids) {
    AsDevice device(g_bfc_allocator_registry.device_type, device_id);
    //     g_bfc_allocator_registry.allocator_map[device] = nullptr;
    if (g_bfc_allocator_registry.allocator_map[device])
      g_bfc_allocator_registry.allocator_map[device]->FreeAll();
    g_bfc_allocator_registry.allocator_map[device] = nullptr;

    size_t free_mem = 0, total_mem = 0;
#ifdef ENABLE_CUDA
    AS_CHECK_CUDA(cudaSetDevice(device_id));
    AS_CHECK_CUDA(cudaMemGetInfo(&free_mem, &total_mem));
    LOG(INFO) << "DestroyBFCAllocator: memory size after free device_id="
              << device_id << ", mem_in_use: " << total_mem - free_mem
              << ", free_mem=" << free_mem << ", total_mem=" << total_mem
              << std::endl;
#endif
  }
  g_bfc_allocator_registry.allocator_map.clear();
#else
  g_bfc_allocator_registry.reuse = true;
#endif
  g_bfc_allocator_registry.valid = false;
}
}  // namespace allspark
