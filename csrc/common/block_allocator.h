/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    block_allocator.h
 */

#pragma once
#include <common.h>

#include <functional>
#include <memory>
#include <set>

#include "block.h"

namespace allspark {

template <typename BlockType>
class BlockAllocator final {
 public:
  using BlockTypePtr = std::shared_ptr<BlockType>;

  BlockAllocator()
      : free_blocks_([](const BlockTypePtr& a, const BlockTypePtr& b) {
          return (*a) < (*b);
        }) {}
  ~BlockAllocator() { ResetPools(); }

  template <typename... Args>
  Block::Ptr Alloc(int64_t size, Args&&... args) {
    BlockTypePtr block = nullptr;
    BlockTypePtr search_key =
        std::make_shared<BlockType>(size, std::forward<Args>(args)..., true);
    auto it = free_blocks_.lower_bound(search_key);
    if (it != free_blocks_.end()) {
      block = *it;
      free_blocks_.erase(it);
    } else if (it != free_blocks_.begin()) {
      block = *(--it);
      block->Resize(size);
      free_blocks_.erase(it);
    } else {
      block =
          std::make_shared<BlockType>(size, std::forward<Args>(args)..., false);
    }
    allocated_blocks_.insert(block);
    return block;
  }

  void Free(const Block::Ptr& block) {
    if (block == nullptr) {
      return;
    }
    BlockTypePtr block_typed = std::static_pointer_cast<BlockType>(block);
    auto it = allocated_blocks_.find(block_typed);
    if (it == allocated_blocks_.end()) {
      return;
    }
    allocated_blocks_.erase(it);
    free_blocks_.insert(block_typed);
  }

  void Reset() {
    for (auto it = allocated_blocks_.begin(); it != allocated_blocks_.end();) {
      auto tmp = it++;
      Free(*tmp);
    }
  }

  void ResetPools() {
    free_blocks_.clear();
    allocated_blocks_.clear();
  }

 private:
  std::set<BlockTypePtr,
           std::function<bool(const BlockTypePtr&, const BlockTypePtr&)>>
      free_blocks_;
  std::set<BlockTypePtr> allocated_blocks_;
};

}  // namespace allspark
