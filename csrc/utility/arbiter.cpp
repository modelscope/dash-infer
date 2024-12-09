/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    arbiter.cpp
 */

#include "arbiter.h"

#include <check.h>
#include <string.h>
#include <unistd.h>
#if ALLSPARK_ARBIT_MODE == 2
#include "utility/allspark_socket.h"
#endif
#include "utility/file_util.h"

namespace allspark {

namespace util {

#define EXEC_ORDER_IN_FNAME 0

#if ALLSPARK_ARBIT_MODE == 1  // numpy文件模式
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
  // std::string npy_dir = NUMPY_DUMP_BASE_DIR + "rank" + std::to_string(rank) +
  // "/" + (seq_len == 0 ? "context_phase" : "seq_len_" +
  // std::to_string(seq_len)) + "/";
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
#elif ALLSPARK_ARBIT_MODE == 2  // socket模式
bool compare_iomap(const std::string& arbiter_sock,
                   const TensorMap& tmap_to_compare,
                   const std::string& header_info) {
  std::vector<std::string> keys;
  for (const auto& pair : tmap_to_compare) {
    keys.push_back(pair.first);
  }
  std::sort(keys.begin(), keys.end());
  for (const auto& key : keys) {
    const auto& t = tmap_to_compare.at(key);
    std::vector<char> npy = t->ToNumpy();
    std::string tensor_str(std::make_move_iterator(npy.begin()),
                           std::make_move_iterator(npy.end()));
    std::string str_to_send = header_info + t->GetName() + "|" + tensor_str;
    DLOG(INFO) << "sending to ..." << std::endl;
    int sockfd = SendToUnixSocket(arbiter_sock.c_str(), str_to_send.c_str(),
                                  str_to_send.size());
    DLOG(INFO) << "sent" << std::endl;
    AS_CHECK_RETVAL(sockfd >= 0, false, "send to arbiter socket failed!");
    std::string recv_str(10, 0);
    DLOG(INFO) << "recv from..." << std::endl;
    bool ret =
        RecvFromUnixSocket(sockfd, (char*)recv_str.data(), recv_str.size() - 1);
    DLOG(INFO) << "recv str=" << recv_str.c_str() << std::endl;
    AS_CHECK_RETVAL(ret, false, "recv from arbiter socket failed!");
    AS_CHECK_RETVAL(strncmp(recv_str.c_str(), "OK", 2) == 0, false,
                    "Compare error in op: " + header_info);
  }
  return true;
}

bool FetchArbitResult(int rank, int nranks, AsOperator* op) {
  AS_ENFORCE(nranks == 1, "only support single card for now");
  std::string pid = std::to_string(getpid());
  std::string arbiter_sock = "/tmp/arbiter_socket." + pid;
  std::string op_name = op->GetOpName();
  std::string header_info = std::to_string(rank) + "|" + op_name + "|";
  return compare_iomap(arbiter_sock, op->GetInTensors(), header_info) and
         compare_iomap(arbiter_sock, op->GetOutTensors(), header_info);
}
#endif

}  // namespace util
}  // namespace allspark
