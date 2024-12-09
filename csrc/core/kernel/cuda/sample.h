/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    sample.h
 */

#pragma once

namespace allspark {
namespace cuda {
// constants from
// (https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications)
// The maximum number of threads per multiprocessor is 1024 for Turing
// architecture (7.5), 1536 for Geforce Ampere (8.6), and 2048 for all other
// architectures. You'll get warnings if you exceed these constants. Hence, the
// following macros adjust the input values from the user to resolve potential
// warnings.
#if __CUDA_ARCH__ == 750
constexpr uint32_t CUDA_MAX_THREADS_PER_SM = 1024;
#elif __CUDA_ARCH__ == 860
constexpr uint32_t CUDA_MAX_THREADS_PER_SM = 1536;
#else
constexpr uint32_t CUDA_MAX_THREADS_PER_SM = 2048;
#endif

// CUDA_MAX_THREADS_PER_BLOCK is same for all architectures currently
constexpr uint32_t CUDA_MAX_THREADS_PER_BLOCK = 1024;
// CUDA_THREADS_PER_BLOCK_FALLBACK is the "canonical fallback" choice of block
// size. 256 is a good number for this fallback and should give good occupancy
// and versatility across all architectures.
constexpr uint32_t CUDA_THREADS_PER_BLOCK_FALLBACK = 256;

#define MAX_THREADS_PER_BLOCK(val)               \
  (((val) <= CUDA_MAX_THREADS_PER_BLOCK) ? (val) \
                                         : CUDA_THREADS_PER_BLOCK_FALLBACK)

#define MIN_BLOCKS_PER_SM(threads_per_block, blocks_per_sm)            \
  ((((threads_per_block) * (blocks_per_sm) <= CUDA_MAX_THREADS_PER_SM) \
        ? (blocks_per_sm)                                              \
        : ((CUDA_MAX_THREADS_PER_SM + (threads_per_block)-1) /         \
           (threads_per_block))))

#define LAUNCH_BOUNDS(max_threads_per_block, min_blocks_per_sm) \
  __launch_bounds__(                                            \
      (MAX_THREADS_PER_BLOCK((max_threads_per_block))),         \
      (MIN_BLOCKS_PER_SM((max_threads_per_block), (min_blocks_per_sm))))

// launch bounds used for kernels utilizing TensorIterator
const uint32_t block_size_bound = 256;
const uint32_t grid_size_bound = 4;
// number of randoms given by distributions like curand_uniform4,
// curand_uniform2_double used in calculating philox offset.
const uint32_t curand4_engine_calls =
    4;  // using curand_uniform4, so we should unroll for 4

struct PhiloxCudaState {
  PhiloxCudaState() = default;
  // Called if graph capture is not underway
  PhiloxCudaState(uint64_t seed, uint64_t offset) {
    seed_ = seed;
    offset_ = offset;
  }

  // Public members
  uint64_t seed_ = 0;
  uint64_t offset_ = 0;
};

}  // namespace cuda
}  // namespace allspark
