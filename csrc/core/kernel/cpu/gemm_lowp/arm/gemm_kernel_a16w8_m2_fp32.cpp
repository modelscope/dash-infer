/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    gemm_kernel_a16w8_m2_fp32.cpp
 */

#ifdef ENABLE_ARM_V84_V9
#include <arm_sve.h>

#include <common/hie_bfloat16.hpp>

#include "activation_const.hpp"
#include "gemm_kernel_impl.h"
#include "gemm_microkernel_macro.h"
#include "gemm_microkernel_macro_m2.h"

namespace allspark {
namespace cpu {
void thread_block_u8_m2_fp32(GemmParam<hie::bfloat16, uint8_t, float>& p, int m,
                             int n, int k, int k_tile, int is_res) {
#define LABEL_FOR_LOOP_K "1"
#define LABEL_KEEP_DQ_PARAM "2"
#define LABEL_SKIP_STORE "3"
#define LABEL_SKIP_LD_B1 "4"
#define LABEL_SKIP_PRF_1 "5"
#define LABEL_SKIP_ACCUMULATE "6"

  int M = p.M;
  int N = p.N;
  int GroupSize = p.GroupSize;

  hie::bfloat16* a_bf16_ptr = p.a_ptr + (m + 0) * p.K_pack + k * 2;
  uint8_t* b_u8_ptr1 = p.b_ptr + (n + 0) * p.K_pack + k * 2;
  uint8_t* b_u8_ptr2 = p.b_ptr + (n + 2) * p.K_pack + k * 2;

  uint64_t c_fp32_ptr = reinterpret_cast<uint64_t>(p.c_ptr + (m + 0) * N + n);

  int next_line_offset = N * sizeof(float);
  float* wei_scale_ptr = p.wei_scale + n + k / GroupSize * N;
  float* wei_scaleXzp_ptr = p.wei_scaleXzp + n + k / GroupSize * N;
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

      /* clear bfmmla result regs */
      "dup     z0.s, #0                           \n"
      "dup     z1.s, #0                           \n"
      "dup     z2.s, #0                           \n"
      "dup     z3.s, #0                           \n"

      " " LABEL_FOR_LOOP_K ":\n"
      ASM_BLOCK_UPDATE_DQ_PARAM(/* SRC1, SRC2 */ z10, z11, 
                                /* SC1, SC2, SC3, SC4 */ z4, z5, z6, z7,
                                /* ZP1, ZP2, ZP3, ZP4 */ z8, z9, z10, z11)

      /* load bf16 input */
      "mov     x4,    x0                         \n"
      "whilelt p5.h,  x4,   %[K_MAX]             \n" // compare k and K_MAX
      "add     x4,    x0,   #8                   \n"
      "whilelt p4.h,  x4,   %[K_MAX]             \n"

      "ld1h    z12.h, p5/z, [%[a_bf16_ptr], #0, MUL VL] \n"
      "ld1h    z13.h, p4/z, [%[a_bf16_ptr], #1, MUL VL] \n"

      "add     %[a_bf16_ptr], %[a_bf16_ptr], #32 \n"

      /* load u8 weight */
      "mov     x4,    x0                         \n"
      "whilelt p3.b,  x4,     %[K_MAX]           \n"
      "ld1b    z14.b, p3/z,   [%[b_u8_ptr1]]     \n"

      "dup     z18.b, #0                         \n"
      "add     x5,    x3, #2                     \n" // n + 2
      "cmp     x5,    %[N]                       \n"
      "b.tcont " LABEL_SKIP_LD_B1 "f             \n" // if (n + 2) > N, go to label
      "ld1b    z18.b, p3/z,   [%[b_u8_ptr2]]     \n"
      " " LABEL_SKIP_LD_B1 ":\n"

      "add     %[b_u8_ptr1],   %[b_u8_ptr1], #16 \n"
      "add     %[b_u8_ptr2],   %[b_u8_ptr2], #16 \n"

      ASM_BLOCK_PREFETCH_PART_1

      /* weight u8->bf16 */
      ASM_BLOCK_WEIGHT_U8_TO_BF16(/* SRC1, SRC2             */ z14, z18, 
                                  /* DST1, DST2, DST3, DST4 */ z14, z15, z16, z17,
                                  /* TMP1, TMP2, TMP3, TMP4 */ z14, z15, z16, z17,
                                  /* TMP5, TMP6, TMP7, TMP8 */ z18, z19, z20, z21,
                                  /* SC1,  SC2,  SC3,  SC4  */ z4, z5, z6, z7, 
                                  /* ZP1,  ZP2,  ZP3,  ZP4  */ z8, z9, z10, z11)

      /* matmul */
      "bfmmla  z0.s,  z12.h, z14.h             \n"
      "bfmmla  z2.s,  z12.h, z16.h             \n"
      
      "bfmmla  z1.s,  z13.h, z15.h             \n"
      "bfmmla  z3.s,  z13.h, z17.h             \n"

      "add     x0,    x0,   #16                \n" // k += 16
      "whilelt p5.h,  x0,   %[K_MAX]           \n" // compare k and K_MAX
      "b.tstop " LABEL_FOR_LOOP_K "b           \n" // if k < K_MAX, go to label

      /* reorder mmla output */
      "fadd    z0.s,  z0.s,  z1.s              \n" // get matmul result
      "fadd    z1.s,  z2.s,  z3.s              \n"

      "trn1    z4.s,  z0.s,  z1.s              \n"
      "trn2    z5.s,  z0.s,  z1.s              \n"
      "zip1    z0.s,  z4.s,  z5.s              \n"
      "zip2    z1.s,  z4.s,  z5.s              \n"

      : /* empty OutputOperands */
      : [a_bf16_ptr] "r"(a_bf16_ptr),
        [b_u8_ptr1] "r"(b_u8_ptr1), [b_u8_ptr2] "r"(b_u8_ptr2),
        [subCh_groupsize] "r"(GroupSize), [subch_cnt] "r"(subch_cnt),
        [scale_p] "r"(wei_scale_ptr), [scaleXzp_p] "r"(wei_scaleXzp_ptr),
        [next_line_offset] "r"(next_line_offset),
        [m] "r"(m), [n] "r"(n), [k_init] "r"(k_init),
        [M] "r"(M), [N] "r"(N), [K_MAX] "r"(K_MAX)
      : "p0", "p1", "p3", "p4", "p5",
        "x0", "x1", "x2", "x3", "x4", "x5", "x7", "x8", 
        "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9",
        "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19",
        "z20",
        "cc", "memory");

  if (p.with_bias && k == 0) {
    ASM_BLOCK_ADD_BIAS
  }

  if (!is_res) {
    if (UNLIKELY(k != 0)) {
      ASM_BLOCK_C_ACCUMULATE
    }
  } else {
    if (UNLIKELY(k != 0)) {
      ASM_BLOCK_C_RES_ACCUMULATE
    }
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

  if (!is_res) {
    ASM_BLOCK_C_STORE
  } else {
    ASM_BLOCK_C_RES_STORE
  }

  // clang-format on

#undef LABEL_FOR_LOOP_K
#undef LABEL_KEEP_DQ_PARAM
#undef LABEL_SKIP_STORE
#undef LABEL_SKIP_LD_B1
#undef LABEL_SKIP_PRF_1
#undef LABEL_SKIP_ACCUMULATE
  return;
}

}  // namespace cpu
}  // namespace allspark
#endif
