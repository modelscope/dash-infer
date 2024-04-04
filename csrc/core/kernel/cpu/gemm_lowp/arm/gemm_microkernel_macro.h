/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    gemm_microkernel_macro.h
 */

#pragma once

#ifdef ENABLE_ARM_V84_V9
#include <arm_sve.h>

// clang-format off
/***********************/

#define ASM_BLOCK_UPDATE_DQ_PARAM(SRC1, SRC2,                     \
                                  SC1, SC2, SC3, SC4,             \
                                  ZP1, ZP2, ZP3, ZP4)             \
      /* compare subCh_cnt and subCh_groupsize */                 \
      "cmp     x1,    %[subCh_groupsize]          \n"             \
      "add     x1,    x1,    #8                   \n"             \
                                                                  \
      /* if subCh_cnt < subCh_groupsize, go to label */           \
      "b.tstop " LABEL_KEEP_DQ_PARAM "f           \n"             \
                                                                  \
      /* load weight dequantize param */                          \
      "ld1w   "#SRC1".s, p1/z, [%[scale_p],    #0, MUL VL] \n"    \
      "ld1w   "#SRC2".s, p1/z, [%[scaleXzp_p], #0, MUL VL] \n"    \
                                                                  \
      /* broadcast scale */                                       \
      "dup    "#SC1".s,  "#SRC1".s[0]             \n"             \
      "dup    "#SC2".s,  "#SRC1".s[1]             \n"             \
      "dup    "#SC3".s,  "#SRC1".s[2]             \n"             \
      "dup    "#SC4".s,  "#SRC1".s[3]             \n"             \
                                                                  \
      /* broadcast (- scale * zp) */                              \
      "dup    "#ZP1".s,  "#SRC2".s[0]             \n"             \
      "dup    "#ZP2".s,  "#SRC2".s[1]             \n"             \
      "dup    "#ZP3".s,  "#SRC2".s[2]             \n"             \
      "dup    "#ZP4".s,  "#SRC2".s[3]             \n"             \
                                                                  \
      /* update weight dequantize param */                        \
      "add     %[scale_p],    %[scale_p],    %[next_line_offset]   \n" \
      "add     %[scaleXzp_p], %[scaleXzp_p], %[next_line_offset]   \n" \
                                                                  \
      "prfw    pldl2keep, p0, [%[scale_p],    #0, MUL VL] \n"     \
      "prfw    pldl2keep, p0, [%[scaleXzp_p], #0, MUL VL] \n"     \
                                                                  \
      "sub     x1,  x1, %[subCh_groupsize]      \n"               \
      " " LABEL_KEEP_DQ_PARAM ":\n"

/***********************/

#define ASM_BLOCK_WEIGHT_U8_TO_BF16(SRC1, SRC2,             \
                                    DST1, DST2, DST3, DST4, \
                                    TMP1, TMP2, TMP3, TMP4, \
                                    TMP5, TMP6, TMP7, TMP8, \
                                    SC1,  SC2,  SC3,  SC4,  \
                                    ZP1,  ZP2,  ZP3,  ZP4)  \
      /* uint8 -> uint16 */                         \
      "uunpklo "#TMP2".h, "#SRC1".b            \n"  \
      "uunpkhi "#TMP3".h, "#SRC1".b            \n"  \
                                                    \
      /* uint16 -> uint32 */                        \
      "uunpklo "#TMP1".s, "#TMP2".h            \n"  \
      "uunpkhi "#TMP4".s, "#TMP2".h            \n"  \
      "uunpklo "#TMP2".s, "#TMP3".h            \n"  \
      "uunpkhi "#TMP3".s, "#TMP3".h            \n"  \
                                                    \
      /* uint32 -> fp32 */                          \
      "ucvtf   "#TMP1".s, p0/m,  "#TMP1".s     \n"  \
      "ucvtf   "#TMP4".s, p0/m,  "#TMP4".s     \n"  \
      "ucvtf   "#TMP2".s, p0/m,  "#TMP2".s     \n"  \
      "ucvtf   "#TMP3".s, p0/m,  "#TMP3".s     \n"  \
                                                    \
      /* uint8 -> uint16 */                         \
      "uunpklo "#TMP6".h, "#SRC2".b            \n"  \
      "uunpkhi "#TMP7".h, "#SRC2".b            \n"  \
                                                    \
      /* uint16 -> uint32 */                        \
      "uunpklo "#TMP5".s, "#TMP6".h            \n"  \
      "uunpkhi "#TMP8".s, "#TMP6".h            \n"  \
      "uunpklo "#TMP6".s, "#TMP7".h            \n"  \
      "uunpkhi "#TMP7".s, "#TMP7".h            \n"  \
                                                    \
      /* uint32 -> fp32 */                          \
      "ucvtf   "#TMP5".s, p0/m,  "#TMP5".s     \n"  \
      "ucvtf   "#TMP8".s, p0/m,  "#TMP8".s     \n"  \
      "ucvtf   "#TMP6".s, p0/m,  "#TMP6".s     \n"  \
      "ucvtf   "#TMP7".s, p0/m,  "#TMP7".s     \n"  \
                                                    \
      /* dequantize weight */                       \
      /* qdata * scale + (- scale * zp) */          \
      "fmad    "#TMP1".s, p0/m, "#SC1".s, "#ZP1".s  \n"  \
      "fmad    "#TMP4".s, p0/m, "#SC2".s, "#ZP2".s  \n"  \
      "fmad    "#TMP2".s, p0/m, "#SC1".s, "#ZP1".s  \n"  \
      "fmad    "#TMP3".s, p0/m, "#SC2".s, "#ZP2".s  \n"  \
                                                    \
      "fmad    "#TMP5".s, p0/m, "#SC3".s, "#ZP3".s  \n"  \
      "fmad    "#TMP8".s, p0/m, "#SC4".s, "#ZP4".s  \n"  \
      "fmad    "#TMP6".s, p0/m, "#SC3".s, "#ZP3".s  \n"  \
      "fmad    "#TMP7".s, p0/m, "#SC4".s, "#ZP4".s  \n"  \
                                                    \
      /* fp32 -> bf16 */                            \
      "bfcvt   "#TMP1".h, p0/m, "#TMP1".s      \n"  \
      "bfcvt   "#TMP4".h, p0/m, "#TMP4".s      \n"  \
      "bfcvt   "#TMP2".h, p0/m, "#TMP2".s      \n"  \
      "bfcvt   "#TMP3".h, p0/m, "#TMP3".s      \n"  \
                                                    \
      "uzp1    "#DST1".h, "#TMP1".h, "#TMP4".h \n"  \
      "uzp1    "#DST2".h, "#TMP2".h, "#TMP3".h \n"  \
                                                    \
      "bfcvt   "#TMP5".h, p0/m, "#TMP5".s      \n"  \
      "bfcvt   "#TMP8".h, p0/m, "#TMP8".s      \n"  \
      "bfcvt   "#TMP6".h, p0/m, "#TMP6".s      \n"  \
      "bfcvt   "#TMP7".h, p0/m, "#TMP7".s      \n"  \
                                                    \
      "uzp1    "#DST3".h, "#TMP5".h, "#TMP8".h \n"  \
      "uzp1    "#DST4".h, "#TMP6".h, "#TMP7".h \n"

/***********************/

#define ASM_BLOCK_UPDATE_DQ_PARAM_FP16(SRC1, SRC2, TMP1, TMP2,    \
                                       SC1, SC2, ZP1, ZP2)        \
      /* compare subCh_cnt and subCh_groupsize */                 \
      "cmp     x1,    %[subCh_groupsize]          \n"             \
      "add     x1,    x1,    #8                   \n"             \
      /* if subCh_cnt < subCh_groupsize, go to label */           \
      "b.tstop " LABEL_KEEP_DQ_PARAM "f           \n"             \
                                                                  \
      /* update dequantize param */                               \
      "add    x6,   x6, #1                    \n"                 \
      "cmp    x6,   #2                        \n"                 \
      "b.tcont " LABEL_UPDATE_DQ_PARAM_2 "f   \n"                 \
                                                                  \
      /* load weight dequantize param */                          \
      "ld1h   "#SRC1".h, p0/z, [%[scale_p],    #0, MUL VL] \n"    \
      "ld1h   "#SRC2".h, p0/z, [%[scaleXzp_p], #0, MUL VL] \n"    \
                                                                  \
      /* broadcast scale */                                       \
      "dup    "#SC1".h,  "#SRC1".h[0]                 \n"         \
      "dup    "#TMP1".h, "#SRC1".h[1]                 \n"         \
      "dup    "#SC2".h,  "#SRC1".h[2]                 \n"         \
      "dup    "#TMP2".h, "#SRC1".h[3]                 \n"         \
                                                                  \
      "sel    "#SC1".h, p2, "#SC1".h, "#TMP1".h       \n"         \
      "sel    "#SC2".h, p2, "#SC2".h, "#TMP2".h       \n"         \
                                                                  \
      /* broadcast (- scale * zp) */                              \
      "dup    "#ZP1".h,  "#SRC2".h[0]                 \n"         \
      "dup    "#TMP1".h, "#SRC2".h[1]                 \n"         \
      "dup    "#ZP2".h,  "#SRC2".h[2]                 \n"         \
      "dup    "#TMP2".h, "#SRC2".h[3]                 \n"         \
                                                                  \
      "sel    "#ZP1".h, p2, "#ZP1".h, "#TMP1".h       \n"         \
      "sel    "#ZP2".h, p2, "#ZP2".h, "#TMP2".h       \n"         \
                                                                  \
      "sub    x1,  x1, %[subCh_groupsize]     \n"                 \
                                                                  \
      "b " LABEL_KEEP_DQ_PARAM "f             \n"                 \
                                                                  \
      /* use loaded weight dequantize param */                    \
      " " LABEL_UPDATE_DQ_PARAM_2 ":\n"                           \
                                                                  \
      /* broadcast scale */                                       \
      "dup    "#SC1".h,  "#SRC1".h[4]                 \n"         \
      "dup    "#TMP1".h, "#SRC1".h[5]                 \n"         \
      "dup    "#SC2".h,  "#SRC1".h[6]                 \n"         \
      "dup    "#TMP2".h, "#SRC1".h[7]                 \n"         \
                                                                  \
      "sel    "#SC1".h, p2, "#SC1".h, "#TMP1".h       \n"         \
      "sel    "#SC2".h, p2, "#SC2".h, "#TMP2".h       \n"         \
                                                                  \
      /* broadcast (- scale * zp) */                              \
      "dup    "#ZP1".h,  "#SRC2".h[4]                 \n"         \
      "dup    "#TMP1".h, "#SRC2".h[5]                 \n"         \
      "dup    "#ZP2".h,  "#SRC2".h[6]                 \n"         \
      "dup    "#TMP2".h, "#SRC2".h[7]                 \n"         \
                                                                  \
      "sel    "#ZP1".h, p2, "#ZP1".h, "#TMP1".h       \n"         \
      "sel    "#ZP2".h, p2, "#ZP2".h, "#TMP2".h       \n"         \
                                                                  \
      /* update weight dequantize param */                        \
      "add    %[scale_p],    %[scale_p],    %[next_line_offset_param] \n" \
      "add    %[scaleXzp_p], %[scaleXzp_p], %[next_line_offset_param] \n" \
                                                                  \
      "prfw    pldl2keep, p0, [%[scale_p],    #0, MUL VL] \n"     \
      "prfw    pldl2keep, p0, [%[scaleXzp_p], #0, MUL VL] \n"     \
                                                                  \
      "mov    x6,   #0                        \n"                 \
                                                                  \
      "sub    x1,  x1, %[subCh_groupsize]     \n"                 \
                                                                  \
      " " LABEL_KEEP_DQ_PARAM ":\n"

/***********************/

#define ASM_BLOCK_WEIGHT_U8_TO_FP16_TO_BF16(SRC1, SRC2,             \
                                            DST1, DST2, DST3, DST4, \
                                            TMP1, TMP2, TMP3, TMP4, \
                                            TMP5, TMP6, TMP7,       \
                                            SC1,  SC2,  ZP1,  ZP2)  \
      /* uint8 -> uint16 */                         \
      "uunpklo "#TMP2".h, "#SRC1".b            \n"  \
      "uunpkhi "#TMP3".h, "#SRC1".b            \n"  \
                                                    \
      /* uint16 -> fp16 */                          \
      "ucvtf   "#TMP1".h, p0/m,  "#TMP2".h     \n"  \
      "ucvtf   "#TMP2".h, p0/m,  "#TMP3".h     \n"  \
                                                    \
      /* dequantize weight */                       \
      /* qdata * scale + (- scale * zp) */          \
      "fmad    "#TMP1".h, p0/m, "#SC1".h, "#ZP1".h \n"  \
      "fmad    "#TMP2".h, p0/m, "#SC1".h, "#ZP1".h \n"  \
                                                    \
      /* fp16 -> bf16 */                            \
      "mov     "#TMP3".d, "#TMP1".d            \n"  \
      "mov     "#TMP4".d, "#TMP2".d            \n"  \
      "and     "#TMP1".h, "#TMP1".h, #0x7fff   \n"  \
      "and     "#TMP2".h, "#TMP2".h, #0x7fff   \n"  \
                                                    \
      "asr     "#TMP1".h, "#TMP1".h, #3        \n"  \
      "asr     "#TMP2".h, "#TMP2".h, #3        \n"  \
                                                    \
      "add     "#TMP1".h, "#TMP1".h, #0x3800   \n"  \
      "add     "#TMP2".h, "#TMP2".h, #0x3800   \n"  \
                                                    \
      "mov     "#TMP7".h, #0x7fff              \n"  \
      "bsl     "#DST1".d, "#TMP1".d, "#TMP3".d, "#TMP7".d  \n"  \
      "bsl     "#DST2".d, "#TMP2".d, "#TMP4".d, "#TMP7".d  \n"  \
                                                    \
      /* uint8 -> uint16 */                         \
      "uunpklo "#TMP3".h, "#SRC2".b            \n"  \
      "uunpkhi "#TMP4".h, "#SRC2".b            \n"  \
                                                    \
      /* uint16 -> fp16 */                          \
      "ucvtf   "#TMP3".h, p0/m,  "#TMP3".h     \n"  \
      "ucvtf   "#TMP4".h, p0/m,  "#TMP4".h     \n"  \
                                                    \
      /* dequantize weight */                       \
      /* qdata * scale + (- scale * zp) */          \
      "fmad    "#TMP3".h, p0/m, "#SC2".h, "#ZP2".h \n"  \
      "fmad    "#TMP4".h, p0/m, "#SC2".h, "#ZP2".h \n"  \
                                                    \
      /* fp16 -> bf16 */                            \
      "mov     "#TMP5".d, "#TMP3".d            \n"  \
      "mov     "#TMP6".d, "#TMP4".d            \n"  \
      "and     "#TMP3".h, "#TMP3".h, #0x7fff   \n"  \
      "and     "#TMP4".h, "#TMP4".h, #0x7fff   \n"  \
                                                    \
      "asr     "#TMP3".h, "#TMP3".h, #3        \n"  \
      "asr     "#TMP4".h, "#TMP4".h, #3        \n"  \
                                                    \
      "add     "#TMP3".h, "#TMP3".h, #0x3800   \n"  \
      "add     "#TMP4".h, "#TMP4".h, #0x3800   \n"  \
                                                    \
      "mov     "#TMP7".h, #0x7fff              \n"  \
      "bsl     "#DST3".d, "#TMP3".d, "#TMP5".d, "#TMP7".d  \n"  \
      "bsl     "#DST4".d, "#TMP4".d, "#TMP6".d, "#TMP7".d  \n"
// clang-format on
#endif
