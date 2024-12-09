/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    process_id.cu
 */

#include "cuda_common.h"  // NOLINT
#include "cuda_kernel.h"
#include "utility/check_cuda.h"

namespace allspark {
namespace cuda {

template <typename T>
void CopyMatrix(const int M, const int N, const T* A, const int lda, T* B,
                const int ldb, cudaStream_t stream) {
  AS_CHECK_CUDA(cudaMemcpy2DAsync(B, ldb * sizeof(T), A, lda * sizeof(T),
                                  N * sizeof(T), M, cudaMemcpyDeviceToDevice,
                                  stream));
}
template void CopyMatrix<float>(const int M, const int N, const float* A,
                                const int lda, float* B, const int ldb,
                                cudaStream_t stream);
#ifdef ENABLE_FP16
template void CopyMatrix<half>(const int M, const int N, const half* A,
                               const int lda, half* B, const int ldb,
                               cudaStream_t stream);
#endif
template void CopyMatrix<hie::bfloat16>(const int M, const int N,
                                        const hie::bfloat16* A, const int lda,
                                        hie::bfloat16* B, const int ldb,
                                        cudaStream_t stream);
template <typename T>
__global__ static void preprocess_from_bos(T* dst1, T* dst2, T bos, int stride,
                                           int N) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < N) {
    dst1[tid] = bos;
    dst2[tid * stride] = bos;
  }
}
template <typename T>
__global__ static void preprocess_from_input(T* dst, const T* src,
                                             int dst_stride, int seq_len,
                                             int N) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < N) {
    dst[tid] = src[tid];
  }
}
template <typename T>
void PreProcessForGeneration(T* dec_ids, T* max_dec_ids, const T* in_ids, T bos,
                             int batch_size, int num_beam, int max_len,
                             int in_len, cudaStream_t stream) {
  if (bos != -1) {
    int N = batch_size * num_beam;
    const int block_num = (N + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
    preprocess_from_bos<<<block_num, THREAD_PER_BLOCK, 0, stream>>>(
        dec_ids, max_dec_ids, bos, max_len, N);
  } else {
    int N = batch_size * num_beam * in_len;
    const int block_num = (N + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
    preprocess_from_input<<<block_num, THREAD_PER_BLOCK, 0, stream>>>(
        dec_ids, in_ids, in_len * num_beam, in_len, N);
  }
}
template void PreProcessForGeneration(int64_t* dec_ids, int64_t* max_dec_ids,
                                      const int64_t* in_ids, int64_t bos,
                                      int batch_size, int num_beam, int max_len,
                                      int seq_len, cudaStream_t stream);

// template <typename T>
// __global__ static void update_id_beam_kernel(T* dst, const T* src,
//                                         const int* beam_idx, int beam_size,
//                                         int max_length, int step, int N) {
//     uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
//     if (tid < N) {
//         int batch = tid / (beam_size * step);
//         int beam = tid % (beam_size * step) / step;
//         int idx2 = tid % step;
//         dst[(batch * beam_size + beam)* max_length + idx2] = src[(batch *
//         beam_size + beam_idx[batch * beam_size + beam]) * max_length + idx2];
//     }
// }
template <typename T>
__global__ static void update_id_kernel(T* dst, const T* src, int length,
                                        int max_length, int* step_list, int N) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < N) {
    int batch = tid / (length);
    int seq_len = tid % length;
    dst[batch * max_length + step_list[batch] + seq_len] =
        src[batch * length + seq_len];
  }
}
template <typename T>
void UpdateId(T* out, const T* in, const int* beam_idx, T* tmp_id,
              int batch_size, int beam_size, int* step_list, int max_length,
              int seq_len, cudaStream_t stream) {
  // if (beam_idx != nullptr) {
  //     CopyMatrix(batch_size * beam_size, step, out, max_length, tmp_id,
  //     max_length, stream); int N = batch_size * beam_size * step; const int
  //     block_num = (N + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
  //     update_id_beam_kernel<<<block_num, THREAD_PER_BLOCK, 0, stream>>>(
  //         out, tmp_id, beam_idx, beam_size, max_length, step, N);
  // }
  int N = batch_size * seq_len;
  const int block_num = (N + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
  update_id_kernel<<<block_num, THREAD_PER_BLOCK, 0, stream>>>(
      out, in, seq_len, max_length, step_list, N);
}

template void UpdateId<int64_t>(int64_t* out, const int64_t* in,
                                const int* beam_idx, int64_t* tmp_id,
                                int batch_size, int beam_size, int* step_list,
                                int max_length, int seq_len,
                                cudaStream_t stream);

template <typename T>
void PostProcessId(T* out_ids, const T* in_ids, int batch_size, int in_stride,
                   int out_stride, cudaStream_t stream) {
  CopyMatrix(batch_size, out_stride, in_ids, in_stride, out_ids, out_stride,
             stream);
}

template void PostProcessId<int64_t>(int64_t* out_ids, const int64_t* in_ids,
                                     int batch_size, int in_stride,
                                     int out_stride, cudaStream_t stream);

}  // namespace cuda
}  // namespace allspark
