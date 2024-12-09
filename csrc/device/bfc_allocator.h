/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    bfc_allocator.h
 */

#pragma once

#include <memory>
#include <string>

#include "common/allocator.h"
#include "common/common.h"

namespace allspark {

// Runtime statistics collected by an allocator. Exactly the same as
// stream_executor::AllocatorStats, but independently defined to preserve the
// mutual independence of StreamExecutor and TensorFlow.
struct AllocatorStats {
  int64_t num_allocs;          // Number of allocations.
  int64_t bytes_in_use;        // Number of bytes in use.
  int64_t peak_bytes_in_use;   // The peak bytes in use.
  int64_t largest_alloc_size;  // The largest single allocation seen.

  // The upper limit of bytes of user allocatable device memory, if such a limit
  // is known.
  int64_t bytes_limit;

  // Stats for reserved memory usage.
  int64_t bytes_reserved;       // Number of bytes reserved.
  int64_t peak_bytes_reserved;  // The peak number of bytes reserved.
  // The upper limit on the number bytes of reservable memory,
  // if such a limit is known.
  int64_t bytes_reservable_limit;

  int64_t largest_free_block_bytes;  // Largest free block's size in heap.

  AllocatorStats()
      : num_allocs(0),
        bytes_in_use(0),
        peak_bytes_in_use(0),
        largest_alloc_size(0),
        bytes_limit(0),
        bytes_reserved(0),
        peak_bytes_reserved(0),
        bytes_reservable_limit(0),
        largest_free_block_bytes(0) {}

  std::string DebugString() const {
    std::ostringstream dbg;
    dbg << "Limit:            " << this->bytes_limit
        << "\nInUse:            " << this->bytes_in_use
        << "\naxInUse:         " << this->peak_bytes_in_use
        << "\numAllocs:        " << this->num_allocs
        << "\naxAllocSize:     " << this->largest_alloc_size
        << "\neserved:         " << this->bytes_reserved
        << "\neakReserved:     " << this->peak_bytes_reserved
        << "\nargestFreeBlock: " << this->largest_free_block_bytes;
    return dbg.str();
  }
};

// An object that does the underlying suballoc/free of memory for a higher-level
// allocator.  The expectation is that the higher-level allocator is doing some
// kind of cache or pool management so that it will call SubAllocator::Alloc and
// Free relatively infrequently, compared to the number of times its own
// AllocateRaw and Free methods are called.
class SubAllocator {
 public:
  SubAllocator() {}

  virtual ~SubAllocator() {}
  // Allocates at least num_bytes. Returns actual number of bytes allocated in
  // bytes_received. The caller can safely use the full bytes_received sized
  // buffer following the returend pointer.
  virtual void* allocate(size_t alignment, size_t num_bytes,
                         size_t* bytes_received) = 0;
  virtual void free(void* ptr, size_t num_bytes) = 0;

  // Returns true if the BFC allocator can safely coalesce adjacent regions
  // returned by this allocator.
  virtual bool SupportsCoalescing() const = 0;

  // Returns the type of the memory allocated by this SubAllocator.
  virtual DeviceType GetMemoryType() const = 0;
};

std::unique_ptr<SubAllocator> CreateSubAllocator(DeviceType device_type);

class BFCAllocatorImpl;

class BFCAllocator : public Allocator {
 public:
  struct Options {
    bool allow_growth = true;

    // Whether the allocator will deallocate free regions to avoid OOM due to
    // memory fragmentation.
    bool garbage_collection = true;

    // Controls when a chunk should be split, if its size exceeds the requested
    // allocation size.
    double fragmentation_fraction = 0;
  };

  BFCAllocator(std::unique_ptr<SubAllocator> sub_allocator, size_t total_memory,
               const std::string& name, int device_id, const Options& opts);
  ~BFCAllocator();

  std::string name();

  DeviceType memory_type();

  // virtual void* allocate(uint64_t size, uint64_t alignment = 0) noexcept
  // override;
  virtual AsStatus Alloc(void** ptr, int64_t nbytes,
                         const std::string& name) override;

  // virtual void free(void* memory) noexcept override;
  virtual AsStatus Free(void* ptr) override;

  AllocatorStats GetStats();

  void RealFree();

  void FreeAll();

 private:
  std::unique_ptr<BFCAllocatorImpl> impl_;
};

// FIXME: add destructor for this class, compiler will call destructor
// but will crash when thread exit.
struct BFCAllocatorRegistry {
  std::map<AsDevice, std::shared_ptr<BFCAllocator>> allocator_map;
  DeviceType device_type{DeviceType::CUDA};
  std::vector<int> device_ids;
  bool reuse{false};
  bool valid{false};
};
static BFCAllocatorRegistry g_bfc_allocator_registry;
static std::mutex g_bfc_registry_lock;
AsStatus InitBFCAllocator(DeviceType device_type,
                          const std::vector<int>& device_ids);
std::shared_ptr<Allocator> GetBFCAllocator(const DeviceType device_type);
std::shared_ptr<Allocator> GetBFCAllocatorByDeviceId(
    const DeviceType device_type, const int device_id);
void SweepBFCAllocator();
void DestroyBFCAllocator();

}  // namespace allspark
