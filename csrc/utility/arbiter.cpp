/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    arbiter.cpp
 */
#include "arbiter.h"

#include <check.h>
#include <string.h>
#include <unistd.h>

#include "utility/file_util.h"

namespace allspark {
namespace util {

#define EXEC_ORDER_IN_FNAME 0

const static std::string NUMPY_DUMP_BASE_DIR =
    "/root/workspace/ALLSPARK_DUMP/to_be_verified/";
void process_io(const TensorMap& tmap_to_compare, const std::string& npy_dir,
                int idx = -1) {
  std::vector<std::string> keys;
  for (const auto& pair : tmap_to_compare) {
    keys.push_back(pair.first);
  }
  std::sort(keys.begin(), keys.end());
  for (const auto& key : keys) {
    const auto& t = tmap_to_compare.at(key);
    std::string filename = npy_dir + t->GetName() + ".npy";
    if (idx != -1)
      filename = npy_dir + t->GetName() + "." + std::to_string(idx) + ".npy";
    t->ToNumpy(filename);
  }
}

#if EXEC_ORDER_IN_FNAME
static int op_exec_order = 0;
static int seq_len_last = -1;
#endif
bool DumpToNumpyFile(int rank, int nranks, int seq_len, AsOperator* op) {
  AS_ENFORCE(seq_len >= 0, "seq_len should >= 0");
  if (rank != 0) return true;

#if EXEC_ORDER_IN_FNAME
  if (seq_len_last != seq_len) {
    op_exec_order = 0;
  } else {
    op_exec_order++;
  }
  seq_len_last = seq_len;
#else
  int op_exec_order = -1;
#endif

  MakeDirs(NUMPY_DUMP_BASE_DIR);
  std::string npy_dir =
      NUMPY_DUMP_BASE_DIR +
      (seq_len == 0 ? "context_phase" : "seq_len_" + std::to_string(seq_len)) +
      "/";
  MakeDirs(npy_dir);
  auto tmap_to_compare = op->GetInTensors();
  process_io(tmap_to_compare, npy_dir, op_exec_order);
  tmap_to_compare = op->GetOutTensors();
  process_io(tmap_to_compare, npy_dir, op_exec_order);
  /*
  if (seq_len == 0) {
      tmap_to_compare = op->GetWeights();
      process_io(tmap_to_compare, npy_dir, op_exec_order);
  }
  */
  return true;
}

}  // namespace util
}  // namespace allspark
