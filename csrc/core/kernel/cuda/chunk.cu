/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    chunk.cu
 */

#include <math.h>

#include "allspark.pb.h"
#include "cuda_common.h"  // NOLINT
#include "cuda_kernel.h"
namespace allspark {
namespace cuda {

template <typename T>
__global__ void chunk_kernel(int N, T* out, T* in, int batch_size, int seq_len,
                             int hidden_size, int chunk_split) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < N) {
    int bs = tid / hidden_size;
    int hid = tid % hidden_size;
    int split_hid = hidden_size / chunk_split;
    int split_part = hid / split_hid;
    int now = hid % split_hid;
    if (now < split_hid / 2) {
      T x = in[bs * hidden_size + hid + split_hid / 2];
      T cdf = 0.5f * (1.0f + tanhf(((T)0.7978845608028654f *
                                    (T)(x + (T)0.044715f * x * x * x))));
      x = x * cdf;
      out[bs * hidden_size / 2 + split_part * split_hid / 2 + now] =
          in[bs * hidden_size + hid] * x;
    }
  }
}
template <typename T>
void Chunk(T* output, T* input, int batch, int seq_len, int hidden_size,
           int chunk_split, cudaStream_t stream) {
  int N = batch * seq_len * hidden_size;
  const int block_num = (N + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
  chunk_kernel<<<block_num, THREAD_PER_BLOCK, 0, stream>>>(
      N, output, input, batch, seq_len, hidden_size, chunk_split);
}

template void Chunk<float>(float* output, float* input, int batch, int seq_len,
                           int hidden_size, int chunk_split,
                           cudaStream_t cu_stream);

#ifdef ENABLE_FP16
template void Chunk<half>(half* output, half* input, int batch, int seq_len,
                          int hidden_size, int chunk_split,
                          cudaStream_t cu_stream);
#endif
template void Chunk<hie::bfloat16>(hie::bfloat16* output, hie::bfloat16* input,
                                   int batch, int seq_len, int hidden_size,
                                   int chunk_split, cudaStream_t cu_stream);

template <typename T>
__global__ void swiglu_kernel(int N, T* out, T* in, int batch_size, int seq_len,
                              int hidden_size, int chunk_split) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < N) {
    int bs = tid / hidden_size;
    int hid = tid % hidden_size;
    int split_hid = hidden_size / chunk_split;
    int split_part = hid / split_hid;
    int now = hid % split_hid;
    if (now < split_hid / 2) {
      T x = in[bs * hidden_size + hid];
      x = x * (1.0f / (1.0f + expf((T)(-x))));
      out[bs * hidden_size / 2 + split_part * split_hid / 2 + now] =
          x * in[bs * hidden_size + hid + split_hid / 2];
    }
  }
}

template <typename T>
void ChunkBinary(T* output, T* input, int batch, int seq_len, int hidden_size,
                 int chunk_split, int type, cudaStream_t stream) {
  int N = batch * seq_len * hidden_size;
  const int block_num = (N + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
  switch (type) {
    case BinaryType::SWIGLU:
      swiglu_kernel<<<block_num, THREAD_PER_BLOCK, 0, stream>>>(
          N, output, input, batch, seq_len, hidden_size, chunk_split);
      // elementwise::Binary(SIGLUFunctor<T>(), count, out, in1, in2,
      // stream);
      break;
    case BinaryType::GEGLU:
      // elementwise::Binary(GeGLUFunctor<T>(), count, out, in1, in2,
      // stream);
      break;

    default:
      return;
  }
}

template void ChunkBinary<float>(float* output, float* input, int batch,
                                 int seq_len, int hidden_size, int chunk_split,
                                 int type, cudaStream_t cu_stream);

#ifdef ENABLE_FP16
template void ChunkBinary<half>(half* output, half* input, int batch,
                                int seq_len, int hidden_size, int chunk_split,
                                int type, cudaStream_t cu_stream);
#endif
template void ChunkBinary<hie::bfloat16>(hie::bfloat16* output,
                                         hie::bfloat16* input, int batch,
                                         int seq_len, int hidden_size,
                                         int chunk_split, int type,
                                         cudaStream_t cu_stream);

}  // namespace cuda
}  // namespace allspark