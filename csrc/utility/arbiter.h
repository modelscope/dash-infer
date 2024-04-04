/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    arbiter.h
 */
#pragma once

#include <core/operator/operator.h>

#include <string>

namespace allspark {
namespace util {
bool DumpToNumpyFile(int rank, int nranks, int seq_len, AsOperator* op);
#define DO_ARBITRATE(rank, nranks, seq_len, op)                                \
  do {                                                                         \
    if (!util::DumpToNumpyFile(rank, nranks, seq_len, op.get())) {             \
      LOG(ERROR) << "DUMP failed, rank=" << rank << " for " << op->GetOpName() \
                 << std::endl;                                                 \
      throw AsException("DUMP_ERROR");                                         \
    }                                                                          \
  } while (false);

}  // namespace util
}  // namespace allspark
