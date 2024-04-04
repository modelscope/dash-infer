/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    gemm_kernel_a16w8_m4_fp16.cpp
 */

#ifdef ENABLE_ARM_V84_V9
#include <arm_sve.h>

#include <common/hie_bfloat16.hpp>

#include "activation_const.hpp"
#include "gemm_kernel_impl.h"
#include "gemm_microkernel_macro.h"
#include "gemm_microkernel_macro_m4.h"

namespace allspark {
namespace cpu {

// clang-format off
#define ASM_BLOCK_BFMMLA                           \
      "bfmmla  z0.s, z12.h, z16.h             \n"  \
      "bfmmla  z1.s, z12.h, z18.h             \n"  \
      "bfmmla  z2.s, z14.h, z16.h             \n"  \
      "bfmmla  z3.s, z14.h, z18.h             \n"  \
                                                   \
      "bfmmla  z0.s, z13.h, z17.h             \n"  \
      "bfmmla  z1.s, z13.h, z19.h             \n"  \
      "bfmmla  z2.s, z15.h, z17.h             \n"  \
      "bfmmla  z3.s, z15.h, z19.h             \n"
// clang-format on

/*********************************************************/

void thread_block_u8_m4_fp16(GemmParam<hie::bfloat16, uint8_t, float16_t>& p,
                             int m, int n, int k, int k_tile) {
#define LABEL_FOR_LOOP_K "1"
#define LABEL_KEEP_DQ_PARAM "2"
#define LABEL_SKIP_PRF_1 "7"
#define LABEL_SKIP_PRF_2 "8"
#define LABEL_UPDATE_DQ_PARAM_2 "9"

  int M = p.M;
  int N = p.N;
  int GroupSize = p.GroupSize;

  hie::bfloat16* a_bf16_ptr1 = p.a_ptr + (m + 0) * p.K_pack + k * 2;
  hie::bfloat16* a_bf16_ptr2 = p.a_ptr + (m + 2) * p.K_pack + k * 2;
  uint8_t* b_u8_ptr1 = p.b_ptr + (n + 0) * p.K_pack + k * 2;
  uint8_t* b_u8_ptr2 = p.b_ptr + (n + 2) * p.K_pack + k * 2;

  uint64_t c_fp32_ptr = reinterpret_cast<uint64_t>(p.c_ptr + (m + 0) * N + n);

  int next_line_offset = N * sizeof(float);
  int next_line_offset_param = ((N + 3) / 4 * 8) * sizeof(float16_t);
  float16_t* wei_scale_ptr =
      p.wei_scale + (n * 2) + (k / GroupSize / 2) * ((N + 3) / 4 * 8);
  float16_t* wei_scaleXzp_ptr =
      p.wei_scaleXzp + (n * 2) + (k / GroupSize / 2) * ((N + 3) / 4 * 8);
  int subch_cnt = k % GroupSize + GroupSize;

  float* bias_ptr = p.bias_ptr + n;

  int k_init = k * 2;
  int K_MAX = (k + k_tile) * 2;
  K_MAX = K_MAX < p.K_pack * 2 ? K_MAX : p.K_pack * 2;

  activation_const_t constant;

  // clang-format off
  asm volatile(
      "ptrue   p0.b                               \n"

      ASM_BLOCK_PREFETCH_PART_0

      "mov     x0, %[k_init]                      \n" // k
      "mov     x1, %[subch_cnt]                   \n" // subch_cnt
      "mov     x2, %x[m]                          \n"
      "mov     x3, %x[n]                          \n"
      "whilelt p1.s, x3, %x[N]                    \n" // compare n, N

      "mov     x7, #0                             \n"

      "mov     x4, #4                             \n"
      "mov     x5, #8                             \n"
      "whilelt p2.h, x4,  x5                      \n" // for prepare dq param
      "mov     x6, #0                             \n"

      /* clear bfmmla result regs */
      "dup     z0.s, #0                           \n"
      "dup     z1.s, #0                           \n"
      "dup     z2.s, #0                           \n"
      "dup     z3.s, #0                           \n"

      " " LABEL_FOR_LOOP_K ":\n"
      ASM_BLOCK_UPDATE_DQ_PARAM_FP16(/* SRC1, SRC2 */ z24, z25,
                                     /* TMP1, TMP2 */ z12, z13,
                                     /* SC1, SC2, ZP1, ZP2 */ z8,  z9, z10, z11)

      /* load bf16 input */
      "mov     x4,    x0                           \n"
      "whilelt p5.h,  x4,   %[K_MAX]               \n" // compare k and K_MAX
      "add     x4,    x0,   #8                     \n"
      "whilelt p4.h,  x4,   %[K_MAX]               \n"

      "ld1h    z12.h, p5/z, [%[a_bf16_ptr1], #0, MUL VL] \n"
      "ld1h    z13.h, p4/z, [%[a_bf16_ptr1], #1, MUL VL] \n"
      "ld1h    z14.h, p5/z, [%[a_bf16_ptr2], #0, MUL VL] \n"
      "ld1h    z15.h, p4/z, [%[a_bf16_ptr2], #1, MUL VL] \n"

      "add     %[a_bf16_ptr1], %[a_bf16_ptr1], #32 \n"
      "add     %[a_bf16_ptr2], %[a_bf16_ptr2], #32 \n"

      /* load u8 weight */
      "mov     x4,    x0                         \n"
      "whilelt p3.b,  x4,     %[K_MAX]           \n"
      "ld1b    z16.b, p3/z,   [%[b_u8_ptr1]]     \n"
      "ld1b    z21.b, p3/z,   [%[b_u8_ptr2]]     \n"

      "add     %[b_u8_ptr1],   %[b_u8_ptr1], #16 \n"
      "add     %[b_u8_ptr2],   %[b_u8_ptr2], #16 \n"

      ASM_BLOCK_PREFETCH_PART_1

      /* weight u8->bf16 */
      ASM_BLOCK_WEIGHT_U8_TO_FP16_TO_BF16(/* SRC1, SRC2             */ z16, z21,
                                          /* DST1, DST2, DST3, DST4 */ z16, z17, z18, z19,
                                          /* TMP1, TMP2, TMP3, TMP4 */ z16, z17, z18, z19,
                                          /* TMP5, TMP6, TMP7       */ z21, z22, z20,
                                          /* SC1,  SC2,  ZP1,  ZP2  */ z8,  z9,  z10, z11)

      /* matmul */
      ASM_BLOCK_BFMMLA

      "add     x0,    x0,   #16                \n" // k += 16
      "whilelt p5.h,  x0,   %[K_MAX]           \n" // compare k and K_MAX
      "b.tstop " LABEL_FOR_LOOP_K "b           \n" // if k < K_MAX, go to label

      /* reorder mmla output */
      ASM_BLOCK_REORDER_BFMMLA_OUTPUT

      : /* empty OutputOperands */
      : [a_bf16_ptr1] "r"(a_bf16_ptr1), [a_bf16_ptr2] "r"(a_bf16_ptr2),
        [b_u8_ptr1] "r"(b_u8_ptr1), [b_u8_ptr2] "r"(b_u8_ptr2),
        [subCh_groupsize] "r"(GroupSize), [subch_cnt] "r"(subch_cnt),
        [scale_p] "r"(wei_scale_ptr), [scaleXzp_p] "r"(wei_scaleXzp_ptr),
        [next_line_offset_param] "r"(next_line_offset_param),
        [m] "r"(m), [n] "r"(n), [k_init] "r"(k_init),
        [M] "r"(M), [N] "r"(N), [K_MAX] "r"(K_MAX)
      : "p0", "p1", "p2", "p3", "p4", "p5",
        "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8",
        "z0", "z1", "z2", "z3", /* "z4", "z5", "z6", "z7", */ "z8", "z9",
        "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19",
        "z20", "z21", "z22", "z23", "z24", "z25",
        "cc", "memory");

  if (p.with_bias && k == 0) {
    ASM_BLOCK_ADD_BIAS
  }

  if (LIKELY(k != 0)) {
    ASM_BLOCK_C_ACCUMULATE
  }

  if (p.do_act == 1) {
    switch (p.actType) {
    case UnaryType::UNARYTYPE_UNDEFINED: {
      break;
    }
    case UnaryType::RELU: {
      ASM_BLOCK_ACTIVE_RELU
      break;
    }
    case UnaryType::SILU: {
      ASM_BLOCK_ACTIVE_SILU
      break;
    }
    case UnaryType::TANH: {
      ASM_BLOCK_ACTIVE_TANH
      break;
    }
    case UnaryType::GELU_ERF: {
      ASM_BLOCK_ACTIVE_GELU_ERF
      break;
    }
    case UnaryType::GELU_TANH: {
      ASM_BLOCK_ACTIVE_GELU_TANH
      break;
    }
    default:
      break;
    }
  }

  ASM_BLOCK_C_STORE

// clang-format on
#undef LABEL_FOR_LOOP_K
#undef LABEL_KEEP_DQ_PARAM
#undef LABEL_SKIP_PRF_1
#undef LABEL_SKIP_PRF_2
#undef LABEL_UPDATE_DQ_PARAM_2
  return;
}

/*********************************************************/

void thread_block_u8_m4_fp16_res(
    GemmParam<hie::bfloat16, uint8_t, float16_t>& p, int m, int n, int k,
    int k_tile) {
#define LABEL_FOR_LOOP_K "1"
#define LABEL_KEEP_DQ_PARAM "2"
#define LABEL_SKIP_STORE "3"
#define LABEL_SKIP_LD_A1 "4"
#define LABEL_SKIP_LD_A2 "5"
#define LABEL_SKIP_LD_B1 "6"
#define LABEL_SKIP_PRF_1 "7"
#define LABEL_SKIP_PRF_2 "8"
#define LABEL_UPDATE_DQ_PARAM_2 "9"
#define LABEL_SKIP_ACCUMULATE "10"

  int M = p.M;
  int N = p.N;
  int GroupSize = p.GroupSize;

  hie::bfloat16* a_bf16_ptr1 = p.a_ptr + (m + 0) * p.K_pack + k * 2;
  hie::bfloat16* a_bf16_ptr2 = p.a_ptr + (m + 2) * p.K_pack + k * 2;
  uint8_t* b_u8_ptr1 = p.b_ptr + (n + 0) * p.K_pack + k * 2;
  uint8_t* b_u8_ptr2 = p.b_ptr + (n + 2) * p.K_pack + k * 2;

  uint64_t c_fp32_ptr = reinterpret_cast<uint64_t>(p.c_ptr + (m + 0) * N + n);

  int next_line_offset = N * sizeof(float);
  int next_line_offset_param = ((N + 3) / 4 * 8) * sizeof(float16_t);
  float16_t* wei_scale_ptr =
      p.wei_scale + (n * 2) + (k / GroupSize / 2) * ((N + 3) / 4 * 8);
  float16_t* wei_scaleXzp_ptr =
      p.wei_scaleXzp + (n * 2) + (k / GroupSize / 2) * ((N + 3) / 4 * 8);
  int subch_cnt = k % GroupSize + GroupSize;

  float* bias_ptr = p.bias_ptr + n;

  int k_init = k * 2;
  int K_MAX = (k + k_tile) * 2;
  K_MAX = K_MAX < p.K_pack * 2 ? K_MAX : p.K_pack * 2;

  activation_const_t constant;

  // clang-format off
  asm volatile(
      "ptrue   p0.b                               \n"

      ASM_BLOCK_PREFETCH_PART_0

      "mov     x0, %[k_init]                      \n" // k
      "mov     x1, %[subch_cnt]                   \n" // subch_cnt
      "mov     x2, %x[m]                          \n"
      "mov     x3, %x[n]                          \n"
      "whilelt p1.s, x3, %x[N]                    \n" // compare n, N

      "mov     x7, #0                             \n"

      "mov     x4, #4                             \n"
      "mov     x5, #8                             \n"
      "whilelt p2.h, x4,  x5                      \n" // for prepare dq param
      "mov     x6, #0                             \n"

      /* clear bfmmla result regs */
      "dup     z0.s, #0                           \n"
      "dup     z1.s, #0                           \n"
      "dup     z2.s, #0                           \n"
      "dup     z3.s, #0                           \n"

      " " LABEL_FOR_LOOP_K ":\n"
      ASM_BLOCK_UPDATE_DQ_PARAM_FP16(/* SRC1, SRC2 */ z24, z25,
                                     /* TMP1, TMP2 */ z12, z13,
                                     /* SC1, SC2, ZP1, ZP2 */ z8,  z9, z10, z11)

      /* load bf16 input */
      "mov     x4,    x0                           \n"
      "whilelt p5.h,  x4,   %[K_MAX]               \n" // compare k and K_MAX
      "add     x4,    x0,   #8                     \n"
      "whilelt p4.h,  x4,   %[K_MAX]               \n"

      "ld1h    z12.h, p5/z, [%[a_bf16_ptr1], #0, MUL VL] \n"
      "ld1h    z13.h, p4/z, [%[a_bf16_ptr1], #1, MUL VL] \n"
      "dup     z14.h, #0                           \n"
      "dup     z15.h, #0                           \n"

      "add     x5,    x2, #2                       \n" // m + 2
      "cmp     x5,    %[M]                         \n"
      "b.tcont " LABEL_SKIP_LD_A1 "f               \n" // if (m + 2) > M, go to label
      "ld1h    z14.h, p5/z, [%[a_bf16_ptr2], #0, MUL VL] \n"
      "ld1h    z15.h, p4/z, [%[a_bf16_ptr2], #1, MUL VL] \n"
      " " LABEL_SKIP_LD_A1 ":\n"
      "add     %[a_bf16_ptr1], %[a_bf16_ptr1], #32 \n"
      "add     %[a_bf16_ptr2], %[a_bf16_ptr2], #32 \n"

      /* load u8 weight */
      "mov     x4,    x0                         \n"
      "whilelt p3.b,  x4,     %[K_MAX]           \n"
      "ld1b    z16.b, p3/z,   [%[b_u8_ptr1]]     \n"

      "dup     z21.b, #0                         \n"
      "add     x5,    x3, #2                     \n" // n + 2
      "cmp     x5,    %[N]                       \n"
      "b.tcont " LABEL_SKIP_LD_B1 "f             \n" // if (n + 2) > N, go to label
      "ld1b    z21.b, p3/z,   [%[b_u8_ptr2]]     \n"
      " " LABEL_SKIP_LD_B1 ":\n"

      "add     %[b_u8_ptr1],   %[b_u8_ptr1], #16 \n"
      "add     %[b_u8_ptr2],   %[b_u8_ptr2], #16 \n"

      ASM_BLOCK_PREFETCH_PART_1

      /* weight u8->bf16 */
      ASM_BLOCK_WEIGHT_U8_TO_FP16_TO_BF16(/* SRC1, SRC2             */ z16, z21,
                                          /* DST1, DST2, DST3, DST4 */ z16, z17, z18, z19,
                                          /* TMP1, TMP2, TMP3, TMP4 */ z16, z17, z18, z19,
                                          /* TMP5, TMP6, TMP7       */ z21, z22, z20,
                                          /* SC1,  SC2,  ZP1,  ZP2  */ z8,  z9,  z10, z11)

      /* matmul */
      ASM_BLOCK_BFMMLA

      "add     x0,    x0,   #16                \n" // k += 16
      "whilelt p5.h,  x0,   %[K_MAX]           \n" // compare k and K_MAX
      "b.tstop " LABEL_FOR_LOOP_K "b           \n" // if k < K_MAX, go to label

      /* reorder mmla output */
      ASM_BLOCK_REORDER_BFMMLA_OUTPUT

      : /* empty OutputOperands */
      : [a_bf16_ptr1] "r"(a_bf16_ptr1), [a_bf16_ptr2] "r"(a_bf16_ptr2),
        [b_u8_ptr1] "r"(b_u8_ptr1), [b_u8_ptr2] "r"(b_u8_ptr2),
        [subCh_groupsize] "r"(GroupSize), [subch_cnt] "r"(subch_cnt),
        [scale_p] "r"(wei_scale_ptr), [scaleXzp_p] "r"(wei_scaleXzp_ptr),
        [next_line_offset_param] "r"(next_line_offset_param),
        [m] "r"(m), [n] "r"(n), [k_init] "r"(k_init),
        [M] "r"(M), [N] "r"(N), [K_MAX] "r"(K_MAX)
      : "p0", "p1", "p2", "p3", "p4", "p5",
        "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8",
        "z0", "z1", "z2", "z3", /* "z4", "z5", "z6", "z7", */ "z8", "z9",
        "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19",
        "z20", "z21", "z22", "z23", "z24", "z25",
        "cc", "memory");

  if (p.with_bias && k == 0) {
    ASM_BLOCK_ADD_BIAS
  }

  if (LIKELY(k != 0)) {
    ASM_BLOCK_C_RES_ACCUMULATE
  }

  if (p.do_act == 1) {
    switch (p.actType) {
    case UnaryType::UNARYTYPE_UNDEFINED: {
      break;
    }
    case UnaryType::RELU: {
      ASM_BLOCK_ACTIVE_RELU
      break;
    }
    case UnaryType::SILU: {
      ASM_BLOCK_ACTIVE_SILU
      break;
    }
    case UnaryType::TANH: {
      ASM_BLOCK_ACTIVE_TANH
      break;
    }
    case UnaryType::GELU_ERF: {
      ASM_BLOCK_ACTIVE_GELU_ERF
      break;
    }
    case UnaryType::GELU_TANH: {
      ASM_BLOCK_ACTIVE_GELU_TANH
      break;
    }
    default:
      break;
    }
  }

  ASM_BLOCK_C_RES_STORE

  // clang-format on

#undef LABEL_FOR_LOOP_K
#undef LABEL_KEEP_DQ_PARAM
#undef LABEL_SKIP_STORE
#undef LABEL_SKIP_LD_A1
#undef LABEL_SKIP_LD_A2
#undef LABEL_SKIP_LD_B1
#undef LABEL_SKIP_PRF_1
#undef LABEL_SKIP_PRF_2
#undef LABEL_UPDATE_DQ_PARAM_2
#undef LABEL_SKIP_ACCUMULATE
  return;
}

}  // namespace cpu
}  // namespace allspark
#endif
