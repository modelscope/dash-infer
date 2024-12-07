/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    rotary.cu
 */

#include <math.h>

#include <cmath>

#include "cuda_common.h"  // NOLINT
#include "cuda_kernel.h"
#include "gemm_lowp/gemm_lowp_utils.cuh"
namespace allspark {
namespace cuda {

using namespace hie;

/*
 * T is half, head_size % 8 = 0
 * T is float, head_size % 4 = 0
 */
template <typename T, int UNROLL, int BLOCK>
__global__ void __launch_bounds__(BLOCK, 2)
    rotary_opt_kernel(int head_size, T* out, const T* in, const float* inv_freq,
                      const int* batch_offset, int batch_size, int seq_len,
                      int num_head, int size_per_head, int* step_list,
                      int seq_stride, int xlogn,
                      const Array<U32DivMod, 2> nDivMod, int* positions,
                      int mrope_size, int* mrope_section) {
  extern __shared__ char smem[];
  float* smem_inv = reinterpret_cast<float*>(smem);
  T* smem_in = reinterpret_cast<T*>(smem + size_per_head / 2 * 4 /*ELEM SIZE*/);

  auto dm = nDivMod[0].DivMod(blockIdx.x);
  int blk_batch_idx = dm.div;
  int blk_seq_idx = dm.mod;

  const T* data_in = in + blockIdx.x * seq_stride;
  T* data_out = out + blockIdx.x * seq_stride;

  const int load_iter_in = (head_size + BLOCK * UNROLL - 1) / (BLOCK * UNROLL);

  if (inv_freq == nullptr) {
    for (int ld_idx = 0; ld_idx < load_iter_in; ++ld_idx) {
      int ld_offset = threadIdx.x * UNROLL + ld_idx * UNROLL * BLOCK;
      if (ld_offset < head_size) {
        *reinterpret_cast<float4*>(data_out + ld_offset) =
            *reinterpret_cast<const float4*>(data_in + ld_offset);
      }
    }
  } else {
    // here imply size_per_head must be multiples of 8
    // load inv_freq from GMEM to SMEM
    if (threadIdx.x < size_per_head / 8) {
      int ld_inv_offset = threadIdx.x * 4;
      *reinterpret_cast<float4*>(smem_inv + ld_inv_offset) =
          *reinterpret_cast<const float4*>(inv_freq + ld_inv_offset +
                                           blk_batch_idx * size_per_head / 2);
    }

    // load in from GM to SMEM
    for (int ld_idx = 0; ld_idx < load_iter_in; ++ld_idx) {
      int ld_offset = threadIdx.x * UNROLL + ld_idx * UNROLL * BLOCK;
      if (ld_offset < head_size) {
        *reinterpret_cast<float4*>(smem_in + ld_offset) =
            *reinterpret_cast<const float4*>(data_in + ld_offset);
      }
    }
    __syncthreads();
    int offset = 0;
    if (batch_offset != nullptr) {
      offset = batch_offset[blk_batch_idx];
    }
    int pos = blk_seq_idx + step_list[blk_batch_idx] + offset;
    float scale = 1.0;
    if (xlogn > 0 && pos > xlogn) {
      scale = logf(pos) / logf(xlogn);
    }
    int inner_loop = (size_per_head + BLOCK - 1) / BLOCK;
    for (int outer_idx = 0; outer_idx < num_head; ++outer_idx) {
      for (int inner_idx = 0; inner_idx < inner_loop; ++inner_idx) {
        int inner_offset = threadIdx.x + BLOCK * inner_idx;
        int outer_offset = inner_offset + outer_idx * size_per_head;
        if (inner_offset < size_per_head) {
          int real_pos = pos;
          if (positions != nullptr) {
            int x = nDivMod[1].Mod(inner_offset);
            for (int section = 0; section < mrope_size; section++) {
              if (x >= mrope_section[section] &&
                  x < mrope_section[section + 1]) {
                real_pos = positions[pos + section * seq_len];
                break;
              }
            }
          }
          float tmp = smem_inv[nDivMod[1].Mod(inner_offset)] * real_pos;
          float sin_ = __sinf(tmp);
          float cos_ = __cosf(tmp);
          float v1 = static_cast<float>(smem_in[outer_offset]) * cos_;
          float v2;
          if (inner_offset < (size_per_head / 2)) {
            v2 = -1.f *
                 static_cast<float>(smem_in[outer_offset + size_per_head / 2]) *
                 sin_;
          } else {
            v2 = static_cast<float>(smem_in[outer_offset - size_per_head / 2]) *
                 sin_;
          }
          data_out[outer_offset] = static_cast<T>((v1 + v2) * scale);
        }
      }
    }
  }
}

template <typename T>
__global__ void rotary_kernel(int N, T* out, T* in, float* inv_freq,
                              int* batch_offset, int batch_size, int seq_len,
                              int num_head, int inner, int step, int stride) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < N) {
    int batch = tid / (seq_len * num_head);
    int seq_pos = tid / num_head % seq_len;
    int head = tid % num_head;
    T* data_in =
        in + batch * seq_len * stride + seq_pos * stride + head * inner;
    T* data_out =
        out + batch * seq_len * stride + seq_pos * stride + head * inner;
    int offset = 0;
    if (batch_offset != nullptr) {
      offset = batch_offset[batch];
    }
    int pos = seq_pos + step + offset;
    if (inv_freq == nullptr) {
      for (int i = 0; i < inner; i++) {
        data_out[i] = data_in[i];
      }
    } else {
      for (int i = 0; i < inner; i++) {
        float sin_ = std::sin(inv_freq[i % (inner / 2)] * pos);
        float cos_ = std::cos(inv_freq[i % (inner / 2)] * pos);
        float v1 = data_in[i] * cos_;
        float v2;
        if (i < inner / 2) {
          v2 = -data_in[i + inner / 2] * sin_;
        } else {
          v2 = data_in[i - inner / 2] * sin_;
        }
        data_out[i] = static_cast<T>(v1 + v2);
      }
    }
  }
}
template <typename T>
__global__ void rotary2D_kernel(int N, T* out, T* in, float* inv_freq,
                                int* batch_offset, int batch_size, int seq_len,
                                int num_head, int inner, int step, int stride,
                                int input_len) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < N) {
    int batch = tid / (seq_len * num_head);
    int seq_pos = tid / num_head % seq_len;
    int head = tid % num_head;
    T* data_in =
        in + batch * seq_len * stride + seq_pos * stride + head * inner;
    T* data_out =
        out + batch * seq_len * stride + seq_pos * stride + head * inner;
    int offset = 0;
    if (batch_offset != nullptr) {
      offset = batch_offset[batch];
    }
    if (inv_freq == nullptr) {
      for (int i = 0; i < inner; i++) {
        data_out[i] = data_in[i];
      }
    } else {
      int half_inner = inner / 2;

      int pos_1 = seq_pos + step + offset;
      if (pos_1 > input_len - 2) {
        pos_1 = input_len - 2;
      }
      for (int i = 0; i < half_inner; i++) {
        float sin_ = std::sin(inv_freq[i % (half_inner / 2)] * pos_1);
        float cos_ = std::cos(inv_freq[i % (half_inner / 2)] * pos_1);
        float v1 = data_in[i] * cos_;
        float v2;
        if (i < half_inner / 2) {
          v2 = -data_in[i + half_inner / 2] * sin_;
        } else {
          v2 = data_in[i - half_inner / 2] * sin_;
        }
        data_out[i] = static_cast<T>(v1 + v2);
      }
      int pos_2 = seq_pos + step + offset - (input_len - 2);
      if (pos_2 < 0) {
        pos_2 = 0;
      }

      for (int i = half_inner; i < inner; i++) {
        float sin_ =
            std::sin(inv_freq[(i - half_inner) % (half_inner / 2)] * pos_2);
        float cos_ =
            std::cos(inv_freq[(i - half_inner) % (half_inner / 2)] * pos_2);
        float v1 = data_in[i] * cos_;
        float v2;
        if ((i - half_inner) < half_inner / 2) {
          v2 = -data_in[i + half_inner / 2] * sin_;
        } else {
          v2 = data_in[i - half_inner / 2] * sin_;
        }
        data_out[i] = static_cast<T>(v1 + v2);
      }
    }
  }
}
template <typename T>
__global__ void rotary_pct_kernel(int N, T* out, T* in, float* inv_freq,
                                  int* batch_offset, int batch_size,
                                  int seq_len, int num_head, int inner,
                                  int step, int stride, float pct) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < N) {
    int batch = tid / (seq_len * num_head);
    int seq_pos = tid / num_head % seq_len;
    int head = tid % num_head;
    T* data_in =
        in + batch * seq_len * stride + seq_pos * stride + head * inner;
    T* data_out =
        out + batch * seq_len * stride + seq_pos * stride + head * inner;
    int offset = 0;
    if (batch_offset != nullptr) {
      offset = batch_offset[batch];
    }
    int pos = seq_pos + step + offset;
    if (inv_freq == nullptr) {
      for (int i = 0; i < inner; i++) {
        data_out[i] = data_in[i];
      }
    } else {
      int real_inner = round(inner * pct);
      for (int i = 0; i < real_inner; i++) {
        float sin_ = std::sin(inv_freq[i % (real_inner / 2)] * pos);
        float cos_ = std::cos(inv_freq[i % (real_inner / 2)] * pos);
        float v1 = data_in[i] * cos_;
        float v2;
        if (i < real_inner / 2) {
          v2 = -data_in[i + real_inner / 2] * sin_;
        } else {
          v2 = data_in[i - real_inner / 2] * sin_;
        }
        data_out[i] = static_cast<T>(v1 + v2);
      }
      for (int i = real_inner; i < inner; i++) {
        data_out[i] = data_in[i];
      }
    }
  }
}
template <typename T>
__global__ void rotary_half_inner_kernel(int N, T* out, T* in, float* inv_freq,
                                         int* batch_offset, int batch_size,
                                         int seq_len, int num_head, int inner,
                                         int* step_list, int stride) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < N) {
    int batch = tid / (seq_len * num_head);
    int seq_pos = tid / num_head % seq_len;
    int head = tid % num_head;
    T* data_in =
        in + batch * seq_len * stride + seq_pos * stride + head * inner;
    T* data_out =
        out + batch * seq_len * stride + seq_pos * stride + head * inner;
    int offset = 0;
    if (batch_offset != nullptr) {
      offset = batch_offset[batch];
    }
    if (inv_freq == nullptr) {
      for (int i = 0; i < inner; i++) {
        data_out[i] = data_in[i];
      }
    } else {
      int half_inner = inner / 2;
      int pos = seq_pos + step_list[batch] + offset;
      for (int i = 0; i < half_inner / 2; i++) {
        float sin_ = std::sin(inv_freq[i] * (float)pos);
        float cos_ = std::cos(inv_freq[i] * (float)pos);
        float v1 = data_in[i * 2];
        float v2 = data_in[i * 2 + 1];
        data_out[i * 2] = static_cast<T>(v1 * cos_ - v2 * sin_);
        data_out[i * 2 + 1] = static_cast<T>(v2 * cos_ + v1 * sin_);
      }
      for (int i = half_inner; i < inner; i++) {
        data_out[i] = data_in[i];
      }
    }
  }
}
template <typename T>
void RotaryEmbedding(T* output, T* input, float* inv_freq, int* batch_offset,
                     int batch, int seq_len, int head, int size_per_head,
                     int step, int stride, cudaStream_t stream) {
  int N = batch * seq_len * head;
  const int block_num = (N + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
  rotary_kernel<<<block_num, THREAD_PER_BLOCK, 0, stream>>>(
      N, output, input, inv_freq, batch_offset, batch, seq_len, head,
      size_per_head, step, stride);
}

template <typename T>
void RotaryEmbedding2D(T* output, T* input, float* inv_freq, int* batch_offset,
                       int batch, int seq_len, int head, int size_per_head,
                       int step, int stride, int input_len,
                       cudaStream_t stream) {
  int N = batch * seq_len * head;
  const int block_num = (N + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
  rotary2D_kernel<<<block_num, THREAD_PER_BLOCK, 0, stream>>>(
      N, output, input, inv_freq, batch_offset, batch, seq_len, head,
      size_per_head, step, stride, input_len);
}

template <typename T>
void RotaryEmbeddingHalfInner(T* output, T* input, float* inv_freq,
                              int* batch_offset, int batch, int seq_len,
                              int head, int size_per_head, int* step_list,
                              int stride, cudaStream_t stream) {
  int N = batch * seq_len * head;
  const int block_num = (N + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
  rotary_half_inner_kernel<<<block_num, THREAD_PER_BLOCK, 0, stream>>>(
      N, output, input, inv_freq, batch_offset, batch, seq_len, head,
      size_per_head, step_list, stride);
}

template <typename T>
void RotaryPctEmbedding(T* output, T* input, float* inv_freq, int* batch_offset,
                        int batch, int seq_len, int head, int size_per_head,
                        int step, int stride, float pct, cudaStream_t stream) {
  int N = batch * seq_len * head;
  const int block_num = (N + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
  rotary_pct_kernel<<<block_num, THREAD_PER_BLOCK, 0, stream>>>(
      N, output, input, inv_freq, batch_offset, batch, seq_len, head,
      size_per_head, step, stride, pct);
}

template <typename T>
__global__ void rotary_multi_section_kernel(int N, T* out, T* in,
                                            float* inv_freq, int seq_len,
                                            int num_head, int inner, int stride,
                                            int* positions, int mrope_size,
                                            int* mrope_section) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < N) {
    int batch = tid / (seq_len * num_head);
    int seq_pos = tid / num_head % seq_len;
    int head = tid % num_head;
    T* data_in =
        in + batch * seq_len * stride + seq_pos * stride + head * inner;
    T* data_out =
        out + batch * seq_len * stride + seq_pos * stride + head * inner;
    if (inv_freq == nullptr) {
      for (int i = 0; i < inner; i++) {
        data_out[i] = data_in[i];
      }
    } else {
      for (int section = 0; section < mrope_size; section++) {
        for (int i = mrope_section[section]; i < mrope_section[section + 1];
             i++) {
          int pos = positions[seq_pos + section * seq_len];
          // printf("seq_pos=%d,i=%d,pos=%d\n",seq_pos,i,pos);
          float sin_ = std::sin(inv_freq[i % (inner / 2)] * pos);
          float cos_ = std::cos(inv_freq[i % (inner / 2)] * pos);
          float v1 = data_in[i] * cos_;
          float v2;
          if (i < inner / 2) {
            v2 = -data_in[i + inner / 2] * sin_;
          } else {
            v2 = data_in[i - inner / 2] * sin_;
          }
          data_out[i] = static_cast<T>(v1 + v2);
        }
        for (int i = inner / 2 + mrope_section[section];
             i < inner / 2 + mrope_section[section + 1]; i++) {
          int pos = positions[seq_pos + section * seq_len];
          // printf("seq_pos=%d,i=%d,pos=%d\n",seq_pos,i,pos);
          float sin_ = std::sin(inv_freq[i % (inner / 2)] * pos);
          float cos_ = std::cos(inv_freq[i % (inner / 2)] * pos);
          float v1 = data_in[i] * cos_;
          float v2;
          if (i < inner / 2) {
            v2 = -data_in[i + inner / 2] * sin_;
          } else {
            v2 = data_in[i - inner / 2] * sin_;
          }
          data_out[i] = static_cast<T>(v1 + v2);
        }
      }
    }
  }
}

template <typename T>
void RotaryMultimodalSections(T* output, T* input, float* inv_freq, int batch,
                              int seq_len, int head, int size_per_head,
                              int stride, int* positions, int mrope_size,
                              int* mrope_section, cudaStream_t stream) {
  int N = batch * seq_len * head;
  const int block_num = (N + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
  rotary_multi_section_kernel<<<block_num, THREAD_PER_BLOCK, 0, stream>>>(
      N, output, input, inv_freq, seq_len, head, size_per_head, stride,
      positions, mrope_size, mrope_section);
}
template void RotaryMultimodalSections<float>(
    float* output, float* input, float* inv_freq, int batch, int seq_len,
    int head, int size_per_head, int stride, int* positions, int mrope_size,
    int* mrope_section, cudaStream_t stream);
#ifdef ENABLE_FP16
template void RotaryMultimodalSections<half>(
    half* output, half* input, float* inv_freq, int batch, int seq_len,
    int head, int size_per_head, int stride, int* positions, int mrope_size,
    int* mrope_section, cudaStream_t stream);
#endif
#ifdef ENABLE_BF16
template void RotaryMultimodalSections<hie::bfloat16>(
    hie::bfloat16* output, hie::bfloat16* input, float* inv_freq, int batch,
    int seq_len, int head, int size_per_head, int stride, int* positions,
    int mrope_size, int* mrope_section, cudaStream_t stream);
#endif
template <typename T>
void RotaryOptEmbedding(T* output, T* input, float* inv_freq, int* batch_offset,
                        int batch, int seq_len, int head, int size_per_head,
                        int* step_list, int stride, int xlogn, int* positions,
                        int mrope_size, int* mrope_section,
                        cudaStream_t stream) {
  int packSize = GetPackSize(input);
  const int thread_per_block = 128;
  const int block_num = batch * seq_len;
  const int hidden_size = head * size_per_head;
  Array<U32DivMod, 2> nDivMod;
  nDivMod[0] = U32DivMod(seq_len);
  nDivMod[1] = U32DivMod(size_per_head / 2);

  if (packSize * sizeof(T) != 16) {
    LOG(ERROR) << "In/Out ptr must be 16B aligned for RotaryOptEmbedding "
               << std::endl;
    return;
  }
  if (size_per_head % 8 != 0) {
    LOG(ERROR) << "RotaryOptEmbedding now only supports size_per_head % 8 == 0"
               << std::endl;
    return;
  }

  switch (packSize) {
    case 8:
      rotary_opt_kernel<T, 8, thread_per_block>
          <<<block_num, thread_per_block,
             hidden_size * sizeof(T) + (size_per_head / 2) * sizeof(float),
             stream>>>(hidden_size, output, input, inv_freq, batch_offset,
                       batch, seq_len, head, size_per_head, step_list, stride,
                       xlogn, nDivMod, positions, mrope_size, mrope_section);
      break;
    case 4:
      rotary_opt_kernel<T, 4, thread_per_block>
          <<<block_num, thread_per_block,
             hidden_size * sizeof(T) + (size_per_head / 2) * sizeof(float),
             stream>>>(hidden_size, output, input, inv_freq, batch_offset,
                       batch, seq_len, head, size_per_head, step_list, stride,
                       xlogn, nDivMod, positions, mrope_size, mrope_section);
      break;
    default:
      LOG(ERROR) << "No Kernel for RotaryOptEmbedding : " << std::endl;
      break;
  }
}

template void RotaryEmbedding<float>(float* output, float* input,
                                     float* inv_freq, int* batch_offset,
                                     int batch, int seq_len, int head,
                                     int size_per_head, int step, int stride,
                                     cudaStream_t stream);
template void RotaryEmbedding2D<float>(float* output, float* input,
                                       float* inv_freq, int* batch_offset,
                                       int batch, int seq_len, int head,
                                       int size_per_head, int step, int stride,
                                       int input_len, cudaStream_t stream);
template void RotaryEmbeddingHalfInner<float>(float* output, float* input,
                                              float* inv_freq,
                                              int* batch_offset, int batch,
                                              int seq_len, int head,
                                              int size_per_head, int* step_list,
                                              int stride, cudaStream_t stream);
template void RotaryOptEmbedding<float>(float* output, float* input,
                                        float* inv_freq, int* batch_offset,
                                        int batch, int seq_len, int head,
                                        int size_per_head, int* step_list,
                                        int stride, int xlogn, int* positions,
                                        int mrope_size, int* mrope_section,
                                        cudaStream_t stream);
template void RotaryPctEmbedding<float>(float* output, float* input,
                                        float* inv_freq, int* batch_offset,
                                        int batch, int seq_len, int head,
                                        int size_per_head, int step, int stride,
                                        float pct, cudaStream_t stream);
#ifdef ENABLE_FP16
template void RotaryEmbedding<half>(half* output, half* input, float* inv_freq,
                                    int* batch_offset, int batch, int seq_len,
                                    int head, int size_per_head, int step,
                                    int stride, cudaStream_t stream);
template void RotaryEmbedding2D<half>(half* output, half* input,
                                      float* inv_freq, int* batch_offset,
                                      int batch, int seq_len, int head,
                                      int size_per_head, int step, int stride,
                                      int input_len, cudaStream_t stream);
template void RotaryOptEmbedding<half>(half* output, half* input,
                                       float* inv_freq, int* batch_offset,
                                       int batch, int seq_len, int head,
                                       int size_per_head, int* step_list,
                                       int stride, int xlogn, int* positions,
                                       int mrope_size, int* mrope_section,
                                       cudaStream_t stream);
template void RotaryEmbeddingHalfInner<half>(half* output, half* input,
                                             float* inv_freq, int* batch_offset,
                                             int batch, int seq_len, int head,
                                             int size_per_head, int* step_list,
                                             int stride, cudaStream_t stream);
template void RotaryPctEmbedding<half>(half* output, half* input,
                                       float* inv_freq, int* batch_offset,
                                       int batch, int seq_len, int head,
                                       int size_per_head, int step, int stride,
                                       float pct, cudaStream_t stream);
#endif
#ifdef ENABLE_BF16
template void RotaryEmbedding<hie::bfloat16>(hie::bfloat16* output,
                                             hie::bfloat16* input,
                                             float* inv_freq, int* batch_offset,
                                             int batch, int seq_len, int head,
                                             int size_per_head, int step,
                                             int stride, cudaStream_t stream);
template void RotaryEmbedding2D<hie::bfloat16>(
    hie::bfloat16* output, hie::bfloat16* input, float* inv_freq,
    int* batch_offset, int batch, int seq_len, int head, int size_per_head,
    int step, int stride, int input_len, cudaStream_t stream);
template void RotaryOptEmbedding<hie::bfloat16>(
    hie::bfloat16* output, hie::bfloat16* input, float* inv_freq,
    int* batch_offset, int batch, int seq_len, int head, int size_per_head,
    int* step_list, int stride, int xlogn, int* positions, int mrope_size,
    int* mrope_section, cudaStream_t stream);
template void RotaryEmbeddingHalfInner<hie::bfloat16>(
    hie::bfloat16* output, hie::bfloat16* input, float* inv_freq,
    int* batch_offset, int batch, int seq_len, int head, int size_per_head,
    int* step_list, int stride, cudaStream_t stream);
template void RotaryPctEmbedding<hie::bfloat16>(
    hie::bfloat16* output, hie::bfloat16* input, float* inv_freq,
    int* batch_offset, int batch, int seq_len, int head, int size_per_head,
    int step, int stride, float pct, cudaStream_t stream);
#endif
}  // namespace cuda
}  // namespace allspark
