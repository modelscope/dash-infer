/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    gemm_microkernel_macro_m8.h
 */

#pragma once

#ifdef ENABLE_ARM_V84_V9
#include <arm_sve.h>

#include "activation_macro.h"

// clang-format off
/***********************/

#define ASM_BLOCK_REORDER_BFMMLA_OUTPUT             \
      "trn1    z8.s,  z0.s, z1.s               \n"  \
      "trn2    z9.s,  z0.s, z1.s               \n"  \
      "zip1    z10.s, z8.s, z9.s               \n"  \
      "zip2    z11.s, z8.s, z9.s               \n"  \
                                                    \
      "trn1    z8.s,  z2.s, z3.s               \n"  \
      "trn2    z9.s,  z2.s, z3.s               \n"  \
      "zip1    z12.s, z8.s, z9.s               \n"  \
      "zip2    z13.s, z8.s, z9.s               \n"  \
                                                    \
      "trn1    z8.s,  z4.s, z5.s               \n"  \
      "trn2    z9.s,  z4.s, z5.s               \n"  \
      "zip1    z14.s, z8.s, z9.s               \n"  \
      "zip2    z15.s, z8.s, z9.s               \n"  \
                                                    \
      "trn1    z8.s,  z6.s, z7.s               \n"  \
      "trn2    z9.s,  z6.s, z7.s               \n"  \
      "zip1    z16.s, z8.s, z9.s               \n"  \
      "zip2    z17.s, z8.s, z9.s               \n"

/***********************/

#define ASM_BLOCK_PREFETCH_PART_0                              \
      "prfw    pldl2keep, p0, [%[scale_p],    #0, MUL VL]  \n" \
      "prfw    pldl2keep, p0, [%[scaleXzp_p], #0, MUL VL]  \n" \
                                                               \
      "prfw    pldl2keep, p0, [%[a_bf16_ptr1], #0, MUL VL] \n" \
      "prfw    pldl2keep, p0, [%[a_bf16_ptr2], #0, MUL VL] \n" \
      "prfw    pldl2keep, p0, [%[a_bf16_ptr3], #0, MUL VL] \n" \
      "prfw    pldl2keep, p0, [%[a_bf16_ptr4], #0, MUL VL] \n" \
                                                               \
      "prfw    pldl2keep, p0, [%[b_u8_ptr1], #0, MUL VL]   \n" \
      "prfw    pldl2keep, p0, [%[b_u8_ptr2], #0, MUL VL]   \n"

#define ASM_BLOCK_PREFETCH_PART_1                      \
      "add     x7,    x7,    #1                    \n" \
      "cmp     x7,    #2                           \n" \
      "b.any   " LABEL_SKIP_PRF_1 "f               \n" \
      "add     x8,    %[a_bf16_ptr1],  #64         \n" \
      "prfw    pldl2keep, p0, [x8,     #0, MUL VL] \n" \
      "add     x8,    %[a_bf16_ptr2],  #64         \n" \
      "prfw    pldl2keep, p0, [x8,     #0, MUL VL] \n" \
      "add     x8,    %[b_u8_ptr1],    #96         \n" \
      "prfw    pldl2keep, p0, [x8,     #0, MUL VL] \n" \
      "add     x8,    %[b_u8_ptr2],    #96         \n" \
      "prfw    pldl2keep, p0, [x8,     #0, MUL VL] \n" \
      " " LABEL_SKIP_PRF_1 ":                      \n"

#define ASM_BLOCK_PREFETCH_PART_2                      \
      "b.any   " LABEL_SKIP_PRF_2 "f               \n" \
      "add     x8,    %[a_bf16_ptr3],  #64         \n" \
      "prfw    pldl2keep, p0, [x8,     #0, MUL VL] \n" \
      "add     x8,    %[a_bf16_ptr4],  #64         \n" \
      "prfw    pldl2keep, p0, [x8,     #0, MUL VL] \n" \
      "mov     x7,    #0                           \n" \
      " " LABEL_SKIP_PRF_2 ":                      \n"

/***********************/

#define ASM_BLOCK_ADD_BIAS                                            \
    asm volatile(                                                     \
        "ld1w    z8.s,  p1/z, [%[bias_p], #0, MUL VL] \n"              \
        "fadd    z10.s, z10.s, z8.s             \n"                   \
        "fadd    z11.s, z11.s, z8.s             \n"                   \
        "fadd    z12.s, z12.s, z8.s             \n"                   \
        "fadd    z13.s, z13.s, z8.s             \n"                   \
        "fadd    z14.s, z14.s, z8.s             \n"                   \
        "fadd    z15.s, z15.s, z8.s             \n"                   \
        "fadd    z16.s, z16.s, z8.s             \n"                   \
        "fadd    z17.s, z17.s, z8.s             \n"                   \
        : /* empty OutputOperands */                                  \
        : [bias_p] "r"(bias_ptr)                                      \
        : "p1", "z8", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", \
          "cc", "memory");

/***********************/

#define ASM_BLOCK_LOAD_A(REG1, REG2, REG3, REG4,                 \
                         PTR1, PTR2)                             \
      "ld1h    "#REG1".h, p5/z, [%["#PTR1"], #0, MUL VL] \n"     \
      "ld1h    "#REG2".h, p4/z, [%["#PTR1"], #1, MUL VL] \n"     \
      "ld1h    "#REG3".h, p5/z, [%["#PTR2"], #0, MUL VL] \n"     \
      "ld1h    "#REG4".h, p4/z, [%["#PTR2"], #1, MUL VL] \n"     \
                                                                 \
      "add     %["#PTR1"], %["#PTR1"], #32 \n"                   \
      "add     %["#PTR2"], %["#PTR2"], #32 \n"

#define ASM_BLOCK_LOAD_A_RES_1(REG1, REG2, REG3, REG4)           \
      "ld1h    "#REG1".h, p5/z, [%[a_bf16_ptr1], #0, MUL VL] \n" \
      "ld1h    "#REG2".h, p4/z, [%[a_bf16_ptr1], #1, MUL VL] \n" \
      "dup     "#REG3".h, #0                       \n"           \
      "dup     "#REG4".h, #0                       \n"           \
                                                                 \
        /* if (m + 2) > M, go to label (skip load) */            \
      "add     x5,    x2, #2                       \n"           \
      "cmp     x5,    %[M]                         \n"           \
      "b.tcont " LABEL_SKIP_LD_A1 "f               \n"           \
      "ld1h    "#REG3".h, p5/z, [%[a_bf16_ptr2], #0, MUL VL] \n" \
      "ld1h    "#REG4".h, p4/z, [%[a_bf16_ptr2], #1, MUL VL] \n" \
      " " LABEL_SKIP_LD_A1 ":\n"                                 \
                                                                 \
      "add     %[a_bf16_ptr1], %[a_bf16_ptr1], #32 \n"           \
      "add     %[a_bf16_ptr2], %[a_bf16_ptr2], #32 \n"

#define ASM_BLOCK_LOAD_A_RES_2(REG1, REG2, REG3, REG4)           \
      "dup     "#REG1".h, #0                       \n"           \
      "dup     "#REG2".h, #0                       \n"           \
      "dup     "#REG3".h, #0                       \n"           \
      "dup     "#REG4".h, #0                       \n"           \
                                                                 \
        /* if (m + 4) > M, go to label (skip load) */            \
      "add     x5,    x2, #4                       \n"           \
      "cmp     x5,    %[M]                         \n"           \
      "b.tcont " LABEL_SKIP_LD_A2 "f               \n"           \
      "ld1h    "#REG1".h, p5/z, [%[a_bf16_ptr3], #0, MUL VL] \n" \
      "ld1h    "#REG2".h, p4/z, [%[a_bf16_ptr3], #1, MUL VL] \n" \
      " " LABEL_SKIP_LD_A2 ":\n"                                 \
                                                                 \
        /* if (m + 6) > M, go to label (skip load) */            \
      "add     x5,    x2, #6                       \n"           \
      "cmp     x5,    %[M]                         \n"           \
      "b.tcont " LABEL_SKIP_LD_A2 "f               \n"           \
      "ld1h    "#REG3".h, p5/z, [%[a_bf16_ptr4], #0, MUL VL] \n" \
      "ld1h    "#REG4".h, p4/z, [%[a_bf16_ptr4], #1, MUL VL] \n" \
      " " LABEL_SKIP_LD_A2 ":\n"                                 \
                                                                 \
      "add     %[a_bf16_ptr3], %[a_bf16_ptr3], #32 \n"           \
      "add     %[a_bf16_ptr4], %[a_bf16_ptr4], #32 \n"

/***********************/

#define ASM_BLOCK_LOAD_B(REG1, REG2)                 \
      "mov     x4,    x0                         \n" \
      "whilelt p3.b,  x4,       %[K_MAX]         \n" \
      "ld1b    "#REG1".b, p3/z, [%[b_u8_ptr1]]   \n" \
      "ld1b    "#REG2".b, p3/z, [%[b_u8_ptr2]]   \n" \
                                                     \
      "add     %[b_u8_ptr1],   %[b_u8_ptr1], #16 \n" \
      "add     %[b_u8_ptr2],   %[b_u8_ptr2], #16 \n"

#define ASM_BLOCK_LOAD_B_RES(REG1, REG2)             \
      "mov     x4,    x0                         \n" \
      "whilelt p3.b,  x4,     %[K_MAX]           \n" \
      "ld1b    "#REG1".b, p3/z,   [%[b_u8_ptr1]] \n" \
                                                     \
      /* if (n + 2) > N, go to label (skip load) */  \
      "dup     "#REG2".b, #0                     \n" \
      "add     x5,    x3, #2                     \n" \
      "cmp     x5,    %[N]                       \n" \
      "b.tcont " LABEL_SKIP_LD_B1 "f             \n" \
      "ld1b    "#REG2".b, p3/z,   [%[b_u8_ptr2]] \n" \
      " " LABEL_SKIP_LD_B1 ":\n"                     \
                                                     \
      "add     %[b_u8_ptr1],   %[b_u8_ptr1], #16 \n" \
      "add     %[b_u8_ptr2],   %[b_u8_ptr2], #16 \n"

/***********************/

#define ASM_BLOCK_C_STORE                                                \
    asm volatile(                                                        \
        "mov     x9,    %[c_fp32_ptr]                \n"                 \
        "st1w    z10.s, p1,   [x9, #0, MUL VL]       \n"                 \
                                                                         \
        "add     x9,    x9,   %[next_line_offset]    \n"                 \
        "st1w    z11.s, p1,   [x9, #0, MUL VL]       \n"                 \
                                                                         \
        "add     x9,    x9,   %[next_line_offset]    \n"                 \
        "st1w    z12.s, p1,   [x9, #0, MUL VL]       \n"                 \
                                                                         \
        "add     x9,    x9,   %[next_line_offset]    \n"                 \
        "st1w    z13.s, p1,   [x9, #0, MUL VL]       \n"                 \
                                                                         \
        "add     x9,    x9,   %[next_line_offset]    \n"                 \
        "st1w    z14.s, p1,   [x9, #0, MUL VL]       \n"                 \
                                                                         \
        "add     x9,    x9,   %[next_line_offset]    \n"                 \
        "st1w    z15.s, p1,   [x9, #0, MUL VL]       \n"                 \
                                                                         \
        "add     x9,    x9,   %[next_line_offset]    \n"                 \
        "st1w    z16.s, p1,   [x9, #0, MUL VL]       \n"                 \
                                                                         \
        "add     x9,    x9,   %[next_line_offset]    \n"                 \
        "st1w    z17.s, p1,   [x9, #0, MUL VL]       \n"                 \
        : /* empty OutputOperands */                                     \
        : [c_fp32_ptr] "r"(c_fp32_ptr),                                  \
          [next_line_offset] "r"(next_line_offset),                      \
          [M] "r"(M)                                                     \
        : "p1", "x9",                                                    \
          "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17",        \
          "cc", "memory");

#define ASM_BLOCK_C_ACCUMULATE                                           \
    asm volatile(                                                        \
        "mov     x9,    %[c_fp32_ptr]                \n"                 \
        "ld1w    z0.s,  p1/z, [x9, #0, MUL VL]       \n"                 \
        "fadd    z10.s, z10.s, z0.s                  \n"                 \
                                                                         \
        "add     x9,    x9,   %[next_line_offset]    \n"                 \
        "ld1w    z1.s,  p1/z, [x9, #0, MUL VL]       \n"                 \
        "fadd    z11.s, z11.s, z1.s                  \n"                 \
                                                                         \
        "add     x9,    x9,   %[next_line_offset]    \n"                 \
        "ld1w    z2.s,  p1/z, [x9, #0, MUL VL]       \n"                 \
        "fadd    z12.s, z12.s, z2.s                  \n"                 \
                                                                         \
        "add     x9,    x9,   %[next_line_offset]    \n"                 \
        "ld1w    z3.s,  p1/z, [x9, #0, MUL VL]       \n"                 \
        "fadd    z13.s, z13.s, z3.s                  \n"                 \
                                                                         \
        "add     x9,    x9,   %[next_line_offset]    \n"                 \
        "ld1w    z4.s,  p1/z, [x9, #0, MUL VL]       \n"                 \
        "fadd    z14.s, z14.s, z4.s                  \n"                 \
                                                                         \
        "add     x9,    x9,   %[next_line_offset]    \n"                 \
        "ld1w    z5.s,  p1/z, [x9, #0, MUL VL]       \n"                 \
        "fadd    z15.s, z15.s, z5.s                  \n"                 \
                                                                         \
        "add     x9,    x9,   %[next_line_offset]    \n"                 \
        "ld1w    z6.s,  p1/z, [x9, #0, MUL VL]       \n"                 \
        "fadd    z16.s, z16.s, z6.s                  \n"                 \
                                                                         \
        "add     x9,    x9,   %[next_line_offset]    \n"                 \
        "ld1w    z7.s,  p1/z, [x9, #0, MUL VL]       \n"                 \
        "fadd    z17.s, z17.s, z7.s                  \n"                 \
        : /* empty OutputOperands */                                     \
        : [c_fp32_ptr] "r"(c_fp32_ptr),                                  \
          [next_line_offset] "r"(next_line_offset),                      \
          [M] "r"(M)                                                     \
        : "p1", "x9",                                                    \
          "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7",                \
          "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17",        \
          "cc", "memory");

#define ASM_BLOCK_C_RES_STORE                                            \
    asm volatile(                                                        \
        "mov     x9,    %[c_fp32_ptr]                \n"                 \
        "st1w    z10.s, p1,   [x9, #0, MUL VL]       \n"                 \
                                                                         \
        /* if (m + 1) > M, go to label (skip store) */                   \
        "add     x5,    x2,   #1                     \n"                 \
        "cmp     x5,    %[M]                         \n"                 \
        "b.tcont " LABEL_SKIP_STORE "f               \n"                 \
        "add     x9,    x9,   %[next_line_offset]    \n"                 \
        "st1w    z11.s, p1,   [x9, #0, MUL VL]       \n"                 \
                                                                         \
        /* if (m + 2) > M, go to label (skip store) */                   \
        "add     x5,    x2,   #2                     \n"                 \
        "cmp     x5,    %[M]                         \n"                 \
        "b.tcont "  LABEL_SKIP_STORE "f              \n"                 \
        "add     x9,    x9,   %[next_line_offset]    \n"                 \
        "st1w    z12.s, p1,   [x9, #0, MUL VL]       \n"                 \
                                                                         \
        /* if (m + 3) > M, go to label (skip store) */                   \
        "add     x5,    x2,   #3                     \n"                 \
        "cmp     x5,    %[M]                         \n"                 \
        "b.tcont "  LABEL_SKIP_STORE "f              \n"                 \
        "add     x9,    x9,   %[next_line_offset]    \n"                 \
        "st1w    z13.s, p1,   [x9, #0, MUL VL]       \n"                 \
                                                                         \
        /* if (m + 4) > M, go to label (skip store) */                   \
        "add     x5,    x2,   #4                     \n"                 \
        "cmp     x5,    %[M]                         \n"                 \
        "b.tcont "  LABEL_SKIP_STORE "f              \n"                 \
        "add     x9,    x9,   %[next_line_offset]    \n"                 \
        "st1w    z14.s, p1,   [x9, #0, MUL VL]       \n"                 \
                                                                         \
        /* if (m + 5) > M, go to label (skip store) */                   \
        "add     x5,    x2,   #5                     \n"                 \
        "cmp     x5,    %[M]                         \n"                 \
        "b.tcont "  LABEL_SKIP_STORE "f              \n"                 \
        "add     x9,    x9,   %[next_line_offset]    \n"                 \
        "st1w    z15.s, p1,   [x9, #0, MUL VL]       \n"                 \
                                                                         \
        /* if (m + 6) > M, go to label (skip store) */                   \
        "add     x5,    x2,   #6                     \n"                 \
        "cmp     x5,    %[M]                         \n"                 \
        "b.tcont "  LABEL_SKIP_STORE "f              \n"                 \
        "add     x9,    x9,   %[next_line_offset]    \n"                 \
        "st1w    z16.s, p1,   [x9, #0, MUL VL]       \n"                 \
                                                                         \
        /* if (m + 7) > M, go to label (skip store) */                   \
        "add     x5,    x2,   #7                     \n"                 \
        "cmp     x5,    %[M]                         \n"                 \
        "b.tcont "  LABEL_SKIP_STORE "f              \n"                 \
        "add     x9,    x9,   %[next_line_offset]    \n"                 \
        "st1w    z17.s, p1,   [x9, #0, MUL VL]       \n"                 \
                                                                         \
        " " LABEL_SKIP_STORE ":\n"                                       \
        : /* empty OutputOperands */                                     \
        : [c_fp32_ptr] "r"(c_fp32_ptr),                                  \
          [next_line_offset] "r"(next_line_offset),                      \
          [M] "r"(M)                                                     \
        : "p1", "x2", "x5", "x9",                                        \
          "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17",        \
          "cc", "memory");

#define ASM_BLOCK_C_RES_ACCUMULATE                                       \
    asm volatile(                                                        \
        "mov     x9,    %[c_fp32_ptr]                \n"                 \
        "ld1w    z0.s,  p1/z, [x9, #0, MUL VL]       \n"                 \
        "fadd    z10.s, z10.s, z0.s                  \n"                 \
                                                                         \
        /* if (m + 1) > M, go to label (skip store) */                   \
        "add     x5,    x2,   #1                     \n"                 \
        "cmp     x5,    %[M]                         \n"                 \
        "b.tcont " LABEL_SKIP_ACCUMULATE "f          \n"                 \
        "add     x9,    x9,   %[next_line_offset]    \n"                 \
        "ld1w    z1.s,  p1/z, [x9, #0, MUL VL]       \n"                 \
        "fadd    z11.s, z11.s, z1.s                  \n"                 \
                                                                         \
        /* if (m + 2) > M, go to label (skip store) */                   \
        "add     x5,    x2,   #2                     \n"                 \
        "cmp     x5,    %[M]                         \n"                 \
        "b.tcont " LABEL_SKIP_ACCUMULATE "f          \n"                 \
        "add     x9,    x9,   %[next_line_offset]    \n"                 \
        "ld1w    z2.s,  p1/z, [x9, #0, MUL VL]       \n"                 \
        "fadd    z12.s, z12.s, z2.s                  \n"                 \
                                                                         \
        /* if (m + 3) > M, go to label (skip store) */                   \
        "add     x5,    x2,   #3                     \n"                 \
        "cmp     x5,    %[M]                         \n"                 \
        "b.tcont " LABEL_SKIP_ACCUMULATE "f          \n"                 \
        "add     x9,    x9,   %[next_line_offset]    \n"                 \
        "ld1w    z3.s,  p1/z, [x9, #0, MUL VL]       \n"                 \
        "fadd    z13.s, z13.s, z3.s                  \n"                 \
                                                                         \
        /* if (m + 4) > M, go to label (skip store) */                   \
        "add     x5,    x2,   #4                     \n"                 \
        "cmp     x5,    %[M]                         \n"                 \
        "b.tcont " LABEL_SKIP_ACCUMULATE "f          \n"                 \
        "add     x9,    x9,   %[next_line_offset]    \n"                 \
        "ld1w    z4.s,  p1/z, [x9, #0, MUL VL]       \n"                 \
        "fadd    z14.s, z14.s, z4.s                  \n"                 \
                                                                         \
        /* if (m + 5) > M, go to label (skip store) */                   \
        "add     x5,    x2,   #5                     \n"                 \
        "cmp     x5,    %[M]                         \n"                 \
        "b.tcont " LABEL_SKIP_ACCUMULATE "f          \n"                 \
        "add     x9,    x9,   %[next_line_offset]    \n"                 \
        "ld1w    z5.s,  p1/z, [x9, #0, MUL VL]       \n"                 \
        "fadd    z15.s, z15.s, z5.s                  \n"                 \
                                                                         \
        /* if (m + 6) > M, go to label (skip store) */                   \
        "add     x5,    x2,   #6                     \n"                 \
        "cmp     x5,    %[M]                         \n"                 \
        "b.tcont " LABEL_SKIP_ACCUMULATE "f          \n"                 \
        "add     x9,    x9,   %[next_line_offset]    \n"                 \
        "ld1w    z6.s,  p1/z, [x9, #0, MUL VL]       \n"                 \
        "fadd    z16.s, z16.s, z6.s                  \n"                 \
                                                                         \
        /* if (m + 7) > M, go to label (skip store) */                   \
        "add     x5,    x2,   #7                     \n"                 \
        "cmp     x5,    %[M]                         \n"                 \
        "b.tcont " LABEL_SKIP_ACCUMULATE "f          \n"                 \
        "add     x9,    x9,   %[next_line_offset]    \n"                 \
        "ld1w    z7.s,  p1/z, [x9, #0, MUL VL]       \n"                 \
        "fadd    z17.s, z17.s, z7.s                  \n"                 \
                                                                         \
         " " LABEL_SKIP_ACCUMULATE ":\n"                                 \
        : /* empty OutputOperands */                                     \
        : [c_fp32_ptr] "r"(c_fp32_ptr),                                  \
          [next_line_offset] "r"(next_line_offset),                      \
          [M] "r"(M)                                                     \
        : "p1", "x2", "x5", "x9",                                        \
          "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7",                \
          "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17",        \
          "cc", "memory");

/***********************/

#define ASM_BLOCK_ACTIVE_RELU               \
    asm volatile(                           \
      "fmax    z10.s, p0/m, z10.s, #0.0 \n" \
      "fmax    z11.s, p0/m, z11.s, #0.0 \n" \
      "fmax    z12.s, p0/m, z12.s, #0.0 \n" \
      "fmax    z13.s, p0/m, z13.s, #0.0 \n" \
      "fmax    z14.s, p0/m, z14.s, #0.0 \n" \
      "fmax    z15.s, p0/m, z15.s, #0.0 \n" \
      "fmax    z16.s, p0/m, z16.s, #0.0 \n" \
      "fmax    z17.s, p0/m, z17.s, #0.0 \n" \
      : /* empty OutputOperands */          \
      : /* empty InputOperands */           \
      : "p0", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", \
        "cc", "memory");

#define ASM_BLOCK_ACTIVE_SILU                              \
    asm volatile(                                          \
      "ld1w    z0.s,  p0/z, [%[exp_const], #0, MUL VL] \n" \
      "ld1w    z1.s,  p0/z, [%[exp_const], #1, MUL VL] \n" \
      "ld1w    z2.s,  p0/z, [%[exp_const], #2, MUL VL] \n" \
                                                           \
      ASM_BLOCK_SILU_MICRO(z10, z10, z0, z1, z2,           \
                           z3, z4, z5, z6, z7,             \
                           p0, p2)                         \
                                                           \
      ASM_BLOCK_SILU_MICRO(z11, z11, z0, z1, z2,           \
                           z3, z4, z5, z6, z7,             \
                           p0, p2)                         \
                                                           \
      ASM_BLOCK_SILU_MICRO(z12, z12, z0, z1, z2,           \
                           z3, z4, z5, z6, z7,             \
                           p0, p2)                         \
                                                           \
      ASM_BLOCK_SILU_MICRO(z13, z13, z0, z1, z2,           \
                           z3, z4, z5, z6, z7,             \
                           p0, p2)                         \
                                                           \
      ASM_BLOCK_SILU_MICRO(z14, z14, z0, z1, z2,           \
                           z3, z4, z5, z6, z7,             \
                           p0, p2)                         \
                                                           \
      ASM_BLOCK_SILU_MICRO(z15, z15, z0, z1, z2,           \
                           z3, z4, z5, z6, z7,             \
                           p0, p2)                         \
                                                           \
      ASM_BLOCK_SILU_MICRO(z16, z16, z0, z1, z2,           \
                           z3, z4, z5, z6, z7,             \
                           p0, p2)                         \
                                                           \
      ASM_BLOCK_SILU_MICRO(z17, z17, z0, z1, z2,           \
                           z3, z4, z5, z6, z7,             \
                           p0, p2)                         \
                                                           \
      : /* empty OutputOperands */                         \
      : [exp_const] "r"(constant.exp_const)                \
      : "p0", "p2",                                        \
        "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7",    \
        "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", \
        "cc", "memory");


#define ASM_BLOCK_ACTIVE_TANH                              \
    asm volatile(                                          \
      "ld1w    z0.s,  p0/z, [%[exp_const], #0, MUL VL] \n" \
      "ld1w    z1.s,  p0/z, [%[exp_const], #1, MUL VL] \n" \
      "ld1w    z2.s,  p0/z, [%[exp_const], #2, MUL VL] \n" \
                                                           \
      "mov     z3.s, p0/m, z10.s                       \n" \
      ASM_BLOCK_TANH(z3, z10, z0, z1, z2,                  \
                     z4, z5, z6, p0, p2)                   \
                                                           \
      "mov     z3.s, p0/m, z11.s                       \n" \
      ASM_BLOCK_TANH(z3, z11, z0, z1, z2,                  \
                     z4, z5, z6, p0, p2)                   \
                                                           \
      "mov     z3.s, p0/m, z12.s                       \n" \
      ASM_BLOCK_TANH(z3, z12, z0, z1, z2,                  \
                     z4, z5, z6, p0, p2)                   \
                                                           \
      "mov     z3.s, p0/m, z13.s                       \n" \
      ASM_BLOCK_TANH(z3, z13, z0, z1, z2,                  \
                     z4, z5, z6, p0, p2)                   \
                                                           \
      "mov     z3.s, p0/m, z14.s                       \n" \
      ASM_BLOCK_TANH(z3, z14, z0, z1, z2,                  \
                     z4, z5, z6, p0, p2)                   \
                                                           \
      "mov     z3.s, p0/m, z15.s                       \n" \
      ASM_BLOCK_TANH(z3, z15, z0, z1, z2,                  \
                     z4, z5, z6, p0, p2)                   \
                                                           \
      "mov     z3.s, p0/m, z16.s                       \n" \
      ASM_BLOCK_TANH(z3, z16, z0, z1, z2,                  \
                     z4, z5, z6, p0, p2)                   \
                                                           \
      "mov     z3.s, p0/m, z17.s                       \n" \
      ASM_BLOCK_TANH(z3, z17, z0, z1, z2,                  \
                     z4, z5, z6, p0, p2)                   \
                                                           \
      : /* empty OutputOperands */                         \
      : [exp_const] "r"(constant.exp_const)                \
      : "p0", "p2",                                        \
        "z0", "z1", "z2", "z3", "z4", "z5", "z6",          \
        "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", \
        "cc", "memory");

#define ASM_BLOCK_ACTIVE_GELU_ERF                                \
    asm volatile(                                                \
      "ld1w    z0.s,  p0/z, [%[exp_const], #0, MUL VL] \n"       \
      "ld1w    z1.s,  p0/z, [%[exp_const], #1, MUL VL] \n"       \
      "ld1w    z2.s,  p0/z, [%[exp_const], #2, MUL VL] \n"       \
                                                                 \
      "ld1w    z3.s,  p0/z, [%[erf_const], #0, MUL VL] \n"       \
      "ld1w    z4.s,  p0/z, [%[erf_const], #1, MUL VL] \n"       \
                                                                 \
      ASM_BLOCK_GELU_ERF_MICRO(z10, z10,                         \
                               z0, z1, z2, z3, z4,               \
                               z5, z6, z7, z8, z9, z18, z19,     \
                               p0, p2)                           \
                                                                 \
      ASM_BLOCK_GELU_ERF_MICRO(z11, z11,                         \
                               z0, z1, z2, z3, z4,               \
                               z5, z6, z7, z8, z9, z18, z19,     \
                               p0, p2)                           \
                                                                 \
      ASM_BLOCK_GELU_ERF_MICRO(z12, z12,                         \
                               z0, z1, z2, z3, z4,               \
                               z5, z6, z7, z8, z9, z18, z19,     \
                               p0, p2)                           \
                                                                 \
      ASM_BLOCK_GELU_ERF_MICRO(z13, z13,                         \
                               z0, z1, z2, z3, z4,               \
                               z5, z6, z7, z8, z9, z18, z19,     \
                               p0, p2)                           \
                                                                 \
      ASM_BLOCK_GELU_ERF_MICRO(z14, z14,                         \
                               z0, z1, z2, z3, z4,               \
                               z5, z6, z7, z8, z9, z18, z19,     \
                               p0, p2)                           \
                                                                 \
      ASM_BLOCK_GELU_ERF_MICRO(z15, z15,                         \
                               z0, z1, z2, z3, z4,               \
                               z5, z6, z7, z8, z9, z18, z19,     \
                               p0, p2)                           \
                                                                 \
      ASM_BLOCK_GELU_ERF_MICRO(z16, z16,                         \
                               z0, z1, z2, z3, z4,               \
                               z5, z6, z7, z8, z9, z18, z19,     \
                               p0, p2)                           \
                                                                 \
      ASM_BLOCK_GELU_ERF_MICRO(z17, z17,                         \
                               z0, z1, z2, z3, z4,               \
                               z5, z6, z7, z8, z9, z18, z19,     \
                               p0, p2)                           \
                                                                 \
      : /* empty OutputOperands */                               \
      : [exp_const] "r"(constant.exp_const),                     \
        [erf_const] "r"(constant.erf_const),                     \
        [inv_sqrt] "r"(constant.inv_sqrt)                        \
      : "p0", "p2",                                              \
        "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9",           \
        "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", \
        "cc", "memory");

#define ASM_BLOCK_ACTIVE_GELU_TANH                         \
    asm volatile(                                          \
      "ld1w    z0.s,  p0/z, [%[exp_const], #0, MUL VL] \n" \
      "ld1w    z1.s,  p0/z, [%[exp_const], #1, MUL VL] \n" \
      "ld1w    z2.s,  p0/z, [%[exp_const], #2, MUL VL] \n" \
                                                           \
      ASM_BLOCK_GELU_TANH_MICRO(z10, z10,                  \
                                z0, z1, z2,                \
                                z3, z4, z5, z6, z7,        \
                                p0, p2)                    \
                                                           \
      ASM_BLOCK_GELU_TANH_MICRO(z11, z11,                  \
                                z0, z1, z2,                \
                                z3, z4, z5, z6, z7,        \
                                p0, p2)                    \
                                                           \
      ASM_BLOCK_GELU_TANH_MICRO(z12, z12,                  \
                                z0, z1, z2,                \
                                z3, z4, z5, z6, z7,        \
                                p0, p2)                    \
                                                           \
      ASM_BLOCK_GELU_TANH_MICRO(z13, z13,                  \
                                z0, z1, z2,                \
                                z3, z4, z5, z6, z7,        \
                                p0, p2)                    \
                                                           \
      ASM_BLOCK_GELU_TANH_MICRO(z14, z14,                  \
                                z0, z1, z2,                \
                                z3, z4, z5, z6, z7,        \
                                p0, p2)                    \
                                                           \
      ASM_BLOCK_GELU_TANH_MICRO(z15, z15,                  \
                                z0, z1, z2,                \
                                z3, z4, z5, z6, z7,        \
                                p0, p2)                    \
                                                           \
      ASM_BLOCK_GELU_TANH_MICRO(z16, z16,                  \
                                z0, z1, z2,                \
                                z3, z4, z5, z6, z7,        \
                                p0, p2)                    \
                                                           \
      ASM_BLOCK_GELU_TANH_MICRO(z17, z17,                  \
                                z0, z1, z2,                \
                                z3, z4, z5, z6, z7,        \
                                p0, p2)                    \
                                                           \
      : /* empty OutputOperands */                         \
      : [exp_const] "r"(constant.exp_const),               \
        [erf_const] "r"(constant.erf_const),               \
        [const1] "r"(constant.gelu_tanh_const[0]),         \
        [const2] "r"(constant.gelu_tanh_const[1])          \
      : "p0", "p2",                                        \
        "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7",    \
        "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", \
        "cc", "memory");
// clang-format on
#endif
