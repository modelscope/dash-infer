/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    cpu_context.h
 */

#pragma once

#include <common/block_allocator.h>
#include <common/block_impl.h>
#include <common/device_context.h>
#include <core/kernel/cpu/cpu_common.h>

#include <dnnl.hpp>
namespace allspark {

class DNNLEngine {
 public:
  static DNNLEngine& GetInstance() {
    static DNNLEngine myInstance;
    return myInstance;
  }
  DNNLEngine(DNNLEngine const&) = delete;             // Copy construct
  DNNLEngine(DNNLEngine&&) = delete;                  // Move construct
  DNNLEngine& operator=(DNNLEngine const&) = delete;  // Copy assign
  DNNLEngine& operator=(DNNLEngine&&) = delete;       // Move assign
  dnnl::engine& GetEngine() { return dnnl_engine_; }

 protected:
  DNNLEngine() : dnnl_engine_(dnnl::engine::kind::cpu, 0) {}
  ~DNNLEngine() {}

 private:
  dnnl::engine dnnl_engine_;
};

class CPUContext : public DeviceContext {
 public:
  CPUContext() : cpu_id_(0), stream_(DNNLEngine::GetInstance().GetEngine()) {
    int nthread = cpu::get_max_threads();
    SetNumThreads(nthread);
  }
  ~CPUContext();
  void SetNumThreads(int num_threads) {
    nthread_ = num_threads;

#if AS_RUNTIME_THREAD == AS_TBB
    global_limit_ = std::make_unique<global_control>(
        global_control::max_allowed_parallelism, num_threads);
#elif AS_RUNTIME_THREAD == AS_OMP
    omp_set_num_threads(num_threads);
#endif
  }

  void SetDeviceId(int device_id) { cpu_id_ = device_id; }
  int GetDeviceId() { return cpu_id_; }

  DeviceType GetDeviceType() const { return DeviceType::CPU; }

  int GetNumThread() const { return nthread_; }

  dnnl::stream& GetStream() { return stream_; }

  const dnnl::stream& GetStream() const { return stream_; }

  void Synchronize() const {
    dnnl::stream* s_ptr = const_cast<dnnl::stream*>(&stream_);
    s_ptr->wait();
  }

  Block::Ptr AllocBlock(int64_t nbytes) { return allocator_.Alloc(nbytes, 0); }

  void FreeBlock(const Block::Ptr& block) { allocator_.Free(block); }
  void ResetBlockPools() { allocator_.ResetPools(); }

  void InitMCCL(int rank, int nRanks);
  int GetRank() const override;
  int GetNranks() const override;
  void SemPostInterProcess() override;
  void SemWaitSendInterProcess() override;
  bool SemWaitMsgSynInterProcess(int msg_size) override;

 private:
  int cpu_id_;
  int nthread_;
#if AS_RUNTIME_THREAD == AS_TBB
  std::unique_ptr<global_control> global_limit_;
#endif
  dnnl::stream stream_;
#define TENSOR_ALIGN_IN_BYTES_CPU (256)
  using CPUBlock = BlockImpl<DeviceType::CPU, TENSOR_ALIGN_IN_BYTES_CPU>;
#undef TENSOR_ALIGN_IN_BYTES_CPU

  BlockAllocator<CPUBlock> allocator_;
  int nranks_ = 1;
  int rank_ = 0;
};

}  // namespace allspark
