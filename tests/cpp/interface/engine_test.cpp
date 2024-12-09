/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    engine_test.cpp
 */

#include <test_common.h>

#ifdef ENABLE_CUDA
#include <cuda/cuda_context.h>
#endif

#if 0
TEST(AsEngine, SetDeviceType) {
    allspark::AsEngine as_engine;
    AS_CHECK(as_engine.SetDeviceType("CPU"));
}

TEST(AsEngine, SetNumThreads) {
    allspark::AsEngine as_engine;
    AS_CHECK(as_engine.SetNumThreads(8));
}
#endif

#ifdef ENABLE_CUDA

TEST(AsEngine, GetDeviceSMVersion) {
  int sm_version = allspark::CUDAContext::GetStreamProcessorVersion(0);
  EXPECT_GE(sm_version, allspark::CUDASMDef::SM_Volta);
  EXPECT_LE(sm_version, allspark::CUDASMDef::SM_90);
}
#endif
