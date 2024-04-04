/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    gemm_kernel_pack_input.cpp
 */

#ifdef ENABLE_ARM_V84_V9
#include <arm_sve.h>
#include <math.h>

#include <common/hie_bfloat16.hpp>

#include "../../cpu_common.h"
#include "gemm_kernel_impl.h"

namespace allspark {
namespace cpu {
void pack_input_impl_simd(int M, int N, int K, int lda, int K_pack,
                          float* a_fp32, hie::bfloat16* a_bf16) {
#define LABEL_FOR_LOOP_M "0"
#define LABEL_FOR_LOOP_K "1"
#define LABEL_m_EQ_M_1 "2"

  float* a_fp32_ptr1 = a_fp32 + 0 * lda;
  float* a_fp32_ptr2 = a_fp32 + 1 * lda;
  hie::bfloat16* a_bf16_ptr = a_bf16;
  int a_fp32_offset = 2 * lda * sizeof(float);
  int a_bf16_offset = 2 * K_pack * sizeof(hie::bfloat16);

  // clang-format off
    asm volatile(
        "ptrue   p0.b                                    \n"
        "sub     x1,    %[M], #1                         \n"  // M - 1
        "mov     x2,    #0                               \n"  // m

        "" LABEL_FOR_LOOP_M
        ":\n"
        "mov     x3,    %[a_fp32_ptr1]                   \n"
        "mov     x4,    %[a_fp32_ptr2]                   \n"
        "mov     x5,    %[a_bf16_ptr]                    \n"

        "prfw    pldl1strm, p0, [x3,    #0, MUL VL]      \n"
        "prfw    pldl1strm, p0, [x4,    #0, MUL VL]      \n"

        "mov     x0,    #0                               \n"
        "whilelt p1.s,  x0,   %[K_MAX]                   \n"  // compare kk and
                                                              // K_MAX

        "" LABEL_FOR_LOOP_K
        ":\n"
        "ld1w   z0.s, p1/z, [x3,    #0, MUL VL]          \n"
        "dup    z1.h, #0                                 \n"
        "cmp    x2, x1                                   \n"  // compare m, M -
                                                              // 1
        "b.none  " LABEL_m_EQ_M_1
        "f                     \n"
        "ld1w   z1.s, p1/z, [x4,    #0, MUL VL]          \n"  // load, when m !=
                                                              // M - 1

        "" LABEL_m_EQ_M_1
        ":\n"
        "add     x3, x3, #16                             \n"
        "add     x4, x4, #16                             \n"

        "prfw    pldl1strm, p0, [x3,    #0, MUL VL]      \n"
        "prfw    pldl1strm, p0, [x4,    #0, MUL VL]      \n"

        "bfcvt   z0.h, p0/m, z0.s                        \n"  // fp32 -> bf16
        "bfcvt   z1.h, p0/m, z1.s                        \n"
        "uzp1    z2.h, z0.h, z1.h                        \n"  // combine bf16

        "uzp1    p3.h, p1.h, p1.h                        \n"
        "st1h    z2.h, p3,   [x5, #0, MUL VL]            \n"  // store bf16 data
        "add     x5, x5, #16                             \n"

        //   "prfw    pstl1keep, p0, [x5,    #0, MUL VL]      \n"

        "add     x0,    x0,   #4                         \n"  // kk += 4
        "whilelt p1.s,  x0,   %[K_MAX]                   \n"  // compare kk and
                                                              // K_MAX
        "b.tstop " LABEL_FOR_LOOP_K
        "b                   \n"  // if k < K_MAX, go to label

        "add     %[a_fp32_ptr1], %[a_fp32_ptr1], %[a_fp32_offset] \n"
        "add     %[a_fp32_ptr2], %[a_fp32_ptr2], %[a_fp32_offset] \n"
        "add     %[a_bf16_ptr],  %[a_bf16_ptr],  %[a_bf16_offset] \n"
        "add     x2,    x2,   #2                         \n"  // m += 2
        "cmp     x2, %[M]                                \n"  // compare m, M
        "b.tstop " LABEL_FOR_LOOP_M
        "b                   \n"  // if m < M, go to label

        : /* empty OutputOperands */
        : [a_fp32_ptr1] "r"(a_fp32_ptr1), [a_fp32_ptr2] "r"(a_fp32_ptr2),
          [a_bf16_ptr] "r"(a_bf16_ptr), [K_MAX] "r"(K), [M] "r"(M),
          [a_fp32_offset] "r"(a_fp32_offset), [a_bf16_offset] "r"(a_bf16_offset)
        : "x0", "x1", "x2", "x3", "x4", "x5", "p0", "p1", "p3", "z0", "z1",
          "z2", "cc", "memory");
  // clang-format on
  return;
}

void pack_input_impl_parallel_simd(int M, int N, int K, int lda, int K_pack,
                                   float* a_fp32, hie::bfloat16* a_bf16) {
#define LABEL_FOR_LOOP_M "0"
#define LABEL_FOR_LOOP_K "1"
#define LABEL_m_EQ_M_1 "2"
  int k_tile = 1024;  // empirical var: 1024, 5120
  int k_thread = std::ceil(K * 1.0 / k_tile);

  // printf("k_tile: %d, k_thread: %d\n", k_tile, k_thread);

  parallel_for(k_thread, [&](int k) {
    float* a_fp32_ptr1 = a_fp32 + 0 * lda + k * k_tile;
    float* a_fp32_ptr2 = a_fp32 + 1 * lda + k * k_tile;
    hie::bfloat16* a_bf16_ptr = a_bf16 + k * k_tile * 2;
    int a_fp32_offset = 2 * lda * sizeof(float);
    int a_bf16_offset = 2 * K_pack * sizeof(hie::bfloat16);
    int kk = k * k_tile;
    int kk_max = (k + 1) * k_tile < K ? (k + 1) * k_tile : K;

    // clang-format off
        asm volatile(
            "ptrue   p0.b                                    \n"
            "sub     x1,    %[M], #1                         \n"  // M - 1
            "mov     x2,    #0                               \n"  // m

            "" LABEL_FOR_LOOP_M
            ":\n"
            "mov     x3,    %[a_fp32_ptr1]                   \n"
            "mov     x4,    %[a_fp32_ptr2]                   \n"
            "mov     x5,    %[a_bf16_ptr]                    \n"

            "prfw    pldl1strm, p0, [x3,    #0, MUL VL]      \n"
            "prfw    pldl1strm, p0, [x4,    #0, MUL VL]      \n"

            "mov     x0,    %[kk]                            \n"
            "whilelt p1.s,  x0,   %[kk_max]                  \n"  // compare kk
                                                                  // and kk_max

            "" LABEL_FOR_LOOP_K
            ":\n"
            "ld1w   z0.s, p1/z, [x3,    #0, MUL VL]          \n"
            "dup    z1.h, #0                                 \n"
            "cmp    x2, x1                                   \n"  // compare m,
                                                                  // M - 1
            "b.none  " LABEL_m_EQ_M_1
            "f                     \n"
            "ld1w   z1.s, p1/z, [x4,    #0, MUL VL]          \n"  // load, when
                                                                  // m != M - 1

            "" LABEL_m_EQ_M_1
            ":\n"
            "add     x3, x3, #16                             \n"
            "add     x4, x4, #16                             \n"

            "prfw    pldl1strm, p0, [x3,    #0, MUL VL]      \n"
            "prfw    pldl1strm, p0, [x4,    #0, MUL VL]      \n"

            "bfcvt   z0.h, p0/m, z0.s                        \n"  // fp32 ->
                                                                  // bf16
            "bfcvt   z1.h, p0/m, z1.s                        \n"
            "uzp1    z2.h, z0.h, z1.h                        \n"  // combine
                                                                  // bf16

            "uzp1    p3.h, p1.h, p1.h                        \n"
            "st1h    z2.h, p3,   [x5, #0, MUL VL]            \n"  // store bf16
                                                                  // data
            "add     x5, x5, #16                             \n"

            //   "prfw    pstl1keep, p0, [x5,    #0, MUL VL]      \n"

            "add     x0,    x0,   #4                         \n"  // kk += 4
            "whilelt p1.s,  x0,   %[kk_max]                  \n"  // compare kk
                                                                  // and kk_max
            "b.tstop " LABEL_FOR_LOOP_K
            "b                   \n"  // if k < K_MAX, go to label

            "add     %[a_fp32_ptr1], %[a_fp32_ptr1], %[a_fp32_offset] \n"
            "add     %[a_fp32_ptr2], %[a_fp32_ptr2], %[a_fp32_offset] \n"
            "add     %[a_bf16_ptr],  %[a_bf16_ptr],  %[a_bf16_offset] \n"
            "add     x2,    x2,   #2                         \n"  // m += 2
            "cmp     x2, %[M]                                \n"  // compare m,
                                                                  // M
            "b.tstop " LABEL_FOR_LOOP_M
            "b                   \n"  // if m < M, go to label

            : /* empty OutputOperands */
            : [a_fp32_ptr1] "r"(a_fp32_ptr1), [a_fp32_ptr2] "r"(a_fp32_ptr2),
              [a_bf16_ptr] "r"(a_bf16_ptr), [kk] "r"(kk), [kk_max] "r"(kk_max),
              [M] "r"(M), [a_fp32_offset] "r"(a_fp32_offset),
              [a_bf16_offset] "r"(a_bf16_offset)
            : "x0", "x1", "x2", "x3", "x4", "x5", "p0", "p1", "p2", "p3", "z0",
              "z1", "z2", "cc", "memory");
    // clang-format on
  });

  return;
}

}  // namespace cpu
}  // namespace allspark
#endif
