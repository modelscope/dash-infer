/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    spmm_config.h
 */

#pragma once

template <typename InputValType_, typename OutputValType_, typename IdxType_,
          int kThreadBlockX_, int kThreadBlockY_, int kSparseBlockLen_ = 0>
struct SpmmConfig {
  typedef InputValType_ InputValType;
  typedef OutputValType_ OutputValType;
  typedef IdxType_ IdxType;

  static constexpr int kThreadBlockX = kThreadBlockX_;
  static constexpr int kThreadBlockY = kThreadBlockY_;
  static constexpr int kSparseBlockLen = kSparseBlockLen_;
  static constexpr int kPackSize = 16 / sizeof(InputValType);
  static constexpr int kThreadBlockSize = kThreadBlockX_ * kThreadBlockY_;

  static_assert(sizeof(InputValType) == 2 || sizeof(InputValType) == 4,
                "invalid input type");
  // static_assert((kThreadBlockX & (kThreadBlockX - 1)) == 0 ||
  //              kThreadBlockX % 32 == 0, "invalid thread block shape");
  static_assert(kThreadBlockSize <= 1024, "invalid thread block size");
};
