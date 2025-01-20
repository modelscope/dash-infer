/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    sample.cu
 */

#include <curand_kernel.h>
#include <float.h>

#include <random>

#include "cuda_common.h"  // NOLINT
#include "cuda_kernel.h"
#include "reduce.cuh"
#include "sample.h"
#include "utility/check_cuda.h"
namespace allspark {
namespace cuda {
__global__ void curand_init_kernel(curandState_t* state,
                                   unsigned long long random_seed) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  curand_init((unsigned long long)(random_seed), 0, 0, &state[tid]);
}
void SampleKernelInitLauncher(
    void* state, unsigned long long seed,
    int batch_size,  // for later refactor...  == 1 for now
    cudaStream_t stream) {
  curandState_t* curand_state = (curandState_t*)state;
  curand_init_kernel<<<1, 1, 0, stream>>>(curand_state, seed);
}
template <typename T>
__device__ T exponential_func(T val) {
  auto f_val = static_cast<float>(val);
  auto log = f_val >= 1.0 - FLT_EPSILON / 2 ? -FLT_EPSILON / 2 : __logf(f_val);

  const float lambda = 1.0;
  auto q = -1.0 / lambda * log;
  return static_cast<T>(q);
}

// for-batch 实现
template <typename T>
__global__ void sample_kernel_batch(int64_t* out, const T* in,
                                    const int* indice, curandState_t** state,
                                    int batch_size, int* num_arr, int stride) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < batch_size) {
    int64_t max_idx = 0;
    float max_val = -1e20;
    for (int i = 0; i < num_arr[tid]; i++) {
      float r = (float)curand_uniform(state[tid]);
      float q = (float)exponential_func(r);
      float val = (float)in[tid * stride + i] / q;
      if (val > max_val) {
        max_val = val;
        max_idx = indice[tid * stride + i];
      }
    }
    out[tid] = max_idx;
  }
}

template <typename T>
void SampleKernelLauncher(int64_t* out, void** states, T* in, const int* indice,
                          int batch_size, int* num_arr, int stride,
                          cudaStream_t stream, void*) {
  curandState_t** curand_state = (curandState_t**)(states);
  const int block_num = (batch_size + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
  sample_kernel_batch<<<block_num, THREAD_PER_BLOCK, 0, stream>>>(
      out, in, indice, curand_state, batch_size, num_arr, stride);
}
template void SampleKernelLauncher<float>(int64_t* out, void** states,
                                          float* in, const int* indice,
                                          int batch_size, int* num_arr,
                                          int stride, cudaStream_t stream,
                                          void*);
#ifdef ENABLE_FP16
template void SampleKernelLauncher<half>(int64_t* out, void** states, half* in,
                                         const int* indice, int batch_size,
                                         int* num_arr, int stride,
                                         cudaStream_t stream, void*);
#endif
template void SampleKernelLauncher<hie::bfloat16>(
    int64_t* out, void** states, hie::bfloat16* in, const int* indice,
    int batch_size, int* num_arr, int stride, cudaStream_t stream, void*);

// utility function that calculates proper philox_offset
// for distributions utilizing TensorIterator. For distributions using
// TensorIterator, we are using a grid-stride loop with each
// thread yielding one element per thread. For the edge of the grid-stride
// loop, if the tensor size is large, the unroll loop will kick in and the
// float4 from curand4 will start getting utilized (for common tensor sizes, we
// end up using rand.x from each thread). Hence, the philox_offset is (number of
// elements per thread * number of engine calls), which makes sure that philox
// offset increment is not less than the number of randoms used in each thread.
std::tuple<uint64_t, dim3, dim3> calc_execution_policy(
    int64_t total_elements, cudaDeviceProp* device_prop) {
  const uint64_t numel = static_cast<uint64_t>(total_elements);
  const uint32_t block_size = block_size_bound;
  const uint32_t unroll = curand4_engine_calls;
  dim3 dim_block(block_size);
  dim3 grid((numel + block_size - 1) / block_size);
  uint32_t blocks_per_sm =
      device_prop->maxThreadsPerMultiProcessor / block_size;
  grid.x = std::min(
      static_cast<uint32_t>(device_prop->multiProcessorCount) * blocks_per_sm,
      grid.x);
  // number of times random will be generated per thread, to offset philox
  // counter in thc random state
  uint64_t counter_offset =
      ((numel - 1) / (block_size * grid.x * unroll) + 1) * curand4_engine_calls;
  return std::make_tuple(counter_offset, grid, dim_block);
}

// grid stride loop kernel for distributions
template <typename T, int unroll_factor, typename dist_t, typename transform_t>
LAUNCH_BOUNDS(block_size_bound, grid_size_bound)
__global__ void distribution_elementwise_grid_stride_kernel(
    int numel, const PhiloxCudaState philox_args, const int* indice,
    const dist_t dist_func, const transform_t transform_func) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  curandStatePhilox4_32_10_t state;
  auto subseq = indice[idx];
  curand_init(philox_args.seed_, subseq, philox_args.offset_, &state);

  int rounded_size =
      ((numel - 1) / (blockDim.x * gridDim.x * unroll_factor) + 1) *
      blockDim.x * gridDim.x * unroll_factor;
  for (int linear_index = idx; linear_index < rounded_size;
       linear_index += blockDim.x * gridDim.x * unroll_factor) {
    auto rand = dist_func(&state);
#pragma unroll
    for (int ii = 0; ii < unroll_factor; ii++) {
      int li = linear_index + blockDim.x * gridDim.x * ii;
      if (li < numel) {
        transform_func(li, static_cast<T>((&rand.x)[ii]));
      }
    }
    __syncthreads();
  }
}

void SampleTorchKernelInitLauncher(
    void* states, unsigned long long seed,
    int batch_size,  // for later refactor...  == 1 for now
    cudaStream_t stream) {
  assert(batch_size == 1);
  for (int i = 0; i < batch_size; i++) {
    auto& philox_state = ((PhiloxCudaState*)states)[i];
    philox_state.seed_ = seed;
  }
}

__global__ void transform_index_kernel(int64_t* out, int* tmp_indice,
                                       const int numel, const int* indice) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < numel) {
    out[tid] = indice[tmp_indice[tid]];
  }
}

template <typename T>
void SampleTorchKernelLauncher(int64_t* out, void** states, T* in,
                               const int* indice, int batch_size, int* num_arr,
                               int stride, cudaStream_t stream,
                               void* device_prop) {
  int num_arr_cpu[batch_size];
  // AS_CHECK_CUDA(cudaMemcpy(num_arr_cpu, num_arr, batch_size * sizeof(int),
  // cudaMemcpyDeviceToHost));
  AS_CHECK_CUDA(cudaMemcpyAsync(num_arr_cpu, num_arr, sizeof(int) * batch_size,
                                cudaMemcpyDeviceToHost, stream));
  AS_CHECK_CUDA(cudaStreamSynchronize(stream));

  for (auto i = 0; i < batch_size;
       i++) {  // batch_size for later refactor...  == 1 for now
    PhiloxCudaState& philox_state = *(((PhiloxCudaState**)states)[i]);
    int numel = num_arr_cpu[i];  // number of elements
    auto execution_policy =
        calc_execution_policy(numel, (cudaDeviceProp*)device_prop);
    auto counter_offset = std::get<0>(execution_policy);
    auto grid = std::get<1>(execution_policy);
    auto block = std::get<2>(execution_policy);
    auto r_out = in + stride * i;  // inplace, transform into multinomial
                                   // distribution of probablities
    distribution_elementwise_grid_stride_kernel<T, curand4_engine_calls>
        <<<grid, block, 0, stream>>>(
            numel, philox_state, indice + stride * i,
            [] __device__(curandStatePhilox4_32_10_t * state) {
              return curand_uniform4(state);
            },
            [=] __device__(int idx, T rand) {
              auto* logits_prob = (T*)&in[idx + stride * i];
              float q = exponential_func(rand);  // q ~ Exp(1), float
              *logits_prob =
                  static_cast<T>(float(*logits_prob) / q);  //  ~ Multinomial
            });
    int* tmp_indice = (int*)(out + i);
    philox_state.offset_ += counter_offset;
    TopKKernelLauncher(r_out, tmp_indice, r_out, 1, numel, 1, stream);
    transform_index_kernel<<<grid, block, 0, stream>>>(out + i, tmp_indice, 1,
                                                       indice + i * stride);
  }
}
template void SampleTorchKernelLauncher<float>(int64_t* out, void** states,
                                               float* in, const int* indice,
                                               int batch_size, int* num_arr,
                                               int stride, cudaStream_t stream,
                                               void*);
#ifdef ENABLE_FP16
template void SampleTorchKernelLauncher<half>(int64_t* out, void** states,
                                              half* in, const int* indice,
                                              int batch_size, int* num_arr,
                                              int stride, cudaStream_t stream,
                                              void*);
#endif

#ifdef ENABLE_BF16
template void SampleTorchKernelLauncher<hie::bfloat16>(
    int64_t* out, void** states, hie::bfloat16* in, const int* indice,
    int batch_size, int* num_arr, int stride, cudaStream_t stream, void*);
#endif

template <typename T>
__global__ static void batch_softmax_kernel(T* input, int* len_list, int stride,
                                            float temperature) {
  float tmp = -1e20f;
  float in = 0.0f;
  int offset = blockIdx.x * stride + threadIdx.x;
  int step = len_list[blockIdx.x];
  __shared__ float s_sum, s_max;
  if (threadIdx.x < step) {
    in = static_cast<float>(input[offset]);
    tmp = in / temperature;
  }
  float max_val = tmp;
  blockReduce<float, ReduceOp::kMAX>(&max_val);
  if (threadIdx.x == 0) {
    s_max = max_val;
  }
  __syncthreads();

  if (threadIdx.x < step) {
    in = expf(tmp - s_max);
  }
  float sum_val = in;
  blockReduce<float, ReduceOp::kSUM>(&sum_val);
  if (threadIdx.x == 0) {
    s_sum = sum_val + 1e-12f;
  }
  __syncthreads();
  if (threadIdx.x < step) {
    input[offset] = in / s_sum;
  }
}
template <typename T>
void BatchSoftmaxKernelLauncher(T* input, int* len_list, int batch_size,
                                int stride, float temperature,
                                cudaStream_t stream) {
  dim3 block, grid;
  grid.x = batch_size;
  if (stride <= 1024) {
    block.x = (stride + 31) / 32 * 32;
    batch_softmax_kernel<<<grid, block, 0, stream>>>(input, len_list, stride,
                                                     temperature);
  }
  // else if (stride <= 2048) {
  //     const int unroll = 2;
  //     block.x = 1024;
  //     softmax_kernel_UNROLL<T, 1024, unroll>
  //         <<<grid, block, 0, stream>>>(qk_buf, mask, beam_size, step);
  // }
}
template void BatchSoftmaxKernelLauncher<float>(float* input, int* len_list,
                                                int batch_size, int stride,
                                                float temperature,
                                                cudaStream_t stream);
#ifdef ENABLE_FP16
template void BatchSoftmaxKernelLauncher<half>(half* input, int* len_list,
                                               int batch_size, int stride,
                                               float temperature,
                                               cudaStream_t stream);
#endif
template void BatchSoftmaxKernelLauncher<hie::bfloat16>(
    hie::bfloat16* input, int* len_list, int batch_size, int stride,
    float temperature, cudaStream_t stream);
}  // namespace cuda
}  // namespace allspark
