/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    cpu_context.cpp
 */

#include "cpu/cpu_context.h"

#include <libipc/condition.h>
#include <libipc/shm.h>
#include <mpi.h>

#include <csignal>
namespace allspark {

struct CPUShareData {
  int data_send_pre;
  int data_send;
  int data_recv;
  int data_retry;
  int data_recv_msg[];
};

class MPIContext {
 public:
  static MPIContext& GetInstance() {
    static MPIContext myInstance;
    return myInstance;
  }
  MPIContext(MPIContext const&) = delete;             // Copy construct
  MPIContext(MPIContext&&) = delete;                  // Move construct
  MPIContext& operator=(MPIContext const&) = delete;  // Copy assign
  MPIContext& operator=(MPIContext&&) = delete;       // Move assign
  void Init(int& rank_id, int& ranks) {
    /**
     * There has no general and portable way to detect whether a process is
     *launched via mpirun. As a workaround, we can check for environment
     *variables that are set by mpirun.
     **/
    const char* ompi_comm_world_size_str = std::getenv("OMPI_COMM_WORLD_SIZE");
    if (!ompi_comm_world_size_str) {
      return;
    }
    // MPI_Init can only be performed once
    MPI_Initialized(&mpi_init_);
    if (!mpi_init_) {
      mpi_init_ = 1;
      MPI_Init(nullptr, nullptr);
      MPI_Comm_rank(MPI_COMM_WORLD, &rank_id);
      MPI_Comm_size(MPI_COMM_WORLD, &ranks);
      rank_ = rank_id;
      nranks_ = ranks;
      // condition & mutex resources for IPC sync
      cond_send_.open("cpu-send-cond");
      lock_send_.open("cpu-send-mutex");

      cond_recv_.open("cpu-recv-cond");
      lock_recv_.open("cpu-recv-mutex");

      shm_hd_.acquire("cpu-share-memory", 1024);
      auto shared_mem = shm_hd_.get();
      memset(shared_mem, 0, 1024);

      std::signal(SIGTERM, &MPIContext::SignalHandler);
      std::signal(SIGABRT, &MPIContext::SignalHandler);
    }
  }

  static void SignalHandler(int signal) {
    LOG(WARNING) << "process received abort signal: " << signal;
    // need to clear all ipc resources in case of abnormal exit
    std::remove("/dev/shm/cpu-recv-cond");
    std::remove("/dev/shm/cpu-recv-mutex");
    std::remove("/dev/shm/cpu-send-cond");
    std::remove("/dev/shm/cpu-send-mutex");
    std::remove("/dev/shm/cpu-share-memory");
  }

  ipc::sync::mutex& GetSendLock() { return lock_send_; }

  ipc::sync::condition& GetSendCondition() { return cond_send_; }

  ipc::sync::mutex& GetRecvLock() { return lock_recv_; }

  ipc::sync::condition& GetRecvCondition() { return cond_recv_; }

  ipc::shm::handle& GetShareHandle() { return shm_hd_; }

 protected:
  MPIContext() : nranks_(1), rank_(0), mpi_init_(0) {}
  ~MPIContext() {
    if (mpi_init_) {
      MPI_Finalize();
    }
    shm_hd_.release();
  }

 private:
  int nranks_;
  int rank_;
  int mpi_init_;
  ipc::sync::condition cond_send_;
  ipc::sync::mutex lock_send_;
  ipc::sync::condition cond_recv_;
  ipc::sync::mutex lock_recv_;
  ipc::shm::handle shm_hd_;
};

CPUContext::~CPUContext() {}

void CPUContext::InitMCCL(int rank, int nRanks) {
  MPIContext::GetInstance().Init(rank, nRanks);
  nranks_ = nRanks;
  rank_ = rank;
  LOG(INFO) << "CPUContext::InitMCCL() rank: " << rank << " nRanks: " << nRanks;
}
int CPUContext::GetRank() const { return rank_; }
int CPUContext::GetNranks() const { return nranks_; }
void CPUContext::SemPostInterProcess() {
  if (nranks_ == 1) {
    return;
  }
  auto& shm_hd = MPIContext::GetInstance().GetShareHandle();
  auto mem = shm_hd.get();
  CPUShareData* mem_msg = static_cast<CPUShareData*>(mem);
  auto& lock = MPIContext::GetInstance().GetSendLock();
  auto& cond = MPIContext::GetInstance().GetSendCondition();
  std::lock_guard<ipc::sync::mutex> guard{lock};
  mem_msg->data_send_pre += 1;
  if (mem_msg->data_send_pre == nranks_) {
    cond.broadcast(lock);
  } else {
    cond.wait(lock);
  }
}
void CPUContext::SemWaitSendInterProcess() {
  if (nranks_ == 1) {
    return;
  }
  auto& shm_hd = MPIContext::GetInstance().GetShareHandle();
  auto mem = shm_hd.get();
  CPUShareData* mem_msg = static_cast<CPUShareData*>(mem);
  auto& lock = MPIContext::GetInstance().GetSendLock();
  auto& cond = MPIContext::GetInstance().GetSendCondition();
  std::lock_guard<ipc::sync::mutex> guard{lock};
  mem_msg->data_send += 1;
  if (mem_msg->data_send == nranks_) {
    mem_msg->data_send = 0;
    mem_msg->data_send_pre = 0;
    cond.broadcast(lock);
  } else {
    cond.wait(lock);
  }
}

bool CPUContext::SemWaitMsgSynInterProcess(int msg_size) {
  if (nranks_ == 1) {
    return false;
  }
  bool retry = false;
  auto& shm_hd = MPIContext::GetInstance().GetShareHandle();
  auto mem = shm_hd.get();
  CPUShareData* mem_msg = static_cast<CPUShareData*>(mem);
  auto& lock = MPIContext::GetInstance().GetRecvLock();
  auto& cond = MPIContext::GetInstance().GetRecvCondition();
  std::lock_guard<ipc::sync::mutex> guard{lock};
  mem_msg->data_recv += 1;
  mem_msg->data_recv_msg[rank_] += msg_size;
  if (mem_msg->data_recv != nranks_) {
    cond.wait(lock);
    if (mem_msg->data_retry == 1) {
      retry = true;
    }
  } else {
    int res = 0;
    for (int i = 0; i < nranks_; i++) {
      res ^= mem_msg->data_recv_msg[i];
    }
    if (res != 0) {
      retry = true;
      mem_msg->data_retry = 1;
    } else {
      retry = false;
      mem_msg->data_retry = 0;
      memset(mem_msg->data_recv_msg, 0, sizeof(int) * nranks_);
    }
    mem_msg->data_recv = 0;
    cond.broadcast(lock);
  }
  return retry;
}

}  // namespace allspark
