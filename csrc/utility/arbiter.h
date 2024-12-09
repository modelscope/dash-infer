/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    arbiter.h
 */

#pragma once

#include <core/operator/operator.h>

#include <string>

namespace allspark {
namespace util {

// 模式1: 保存为numpy文件，串行对比
// 模式2: socket传输，实时对比
// 尚无分布式需求，暂不支持模式2，不支持多卡对比
// (分布式版本支持模式2，以及多卡对比)
#define ALLSPARK_ARBIT_MODE 1

#if ALLSPARK_ARBIT_MODE == 1
bool DumpToNumpyFile(int rank, int nranks, int seq_len, AsOperator* op);
#define DO_ARBITRATE(rank, nranks, seq_len, op)                                \
  do {                                                                         \
    if (!util::DumpToNumpyFile(rank, nranks, seq_len, op.get())) {             \
      LOG(ERROR) << "DUMP failed, rank=" << rank << " for " << op->GetOpName() \
                 << std::endl;                                                 \
      throw AsException("DUMP_ERROR");                                         \
    }                                                                          \
  } while (false);
#elif ALLSPARK_ARBIT_MODE == 2
bool FetchArbitResult(int rank, int nranks, AsOperator* op);
#define DO_ARBITRATE(rank, nranks, seq_len, op)                           \
  do {                                                                    \
    if (!util::FetchArbitResult(rank, nranks, op.get())) {                \
      LOG(ERROR) << "arbitration failed, rank=" << rank << "outputs for " \
                 << op->GetOpName() << std::endl;                         \
      throw AsException("ARBITRATION_ERROR");                             \
    }                                                                     \
  } while (false);
#else
#define DO_ARBITRATE(rank, nranks, seq_len, op)
#endif

}  // namespace util
}  // namespace allspark
