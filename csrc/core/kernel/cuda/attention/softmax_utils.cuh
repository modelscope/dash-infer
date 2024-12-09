/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    softmax_utils.cuh
 */

#ifndef __MHA_SOFTMAX_UTILS__
#define __MHA_SOFTMAX_UTILS__
#include "attention_utils.cuh"

namespace allspark {
namespace cuda {
namespace attention {
namespace softmax {

enum MaskMode {
  MaskDirectAdd = 0,
  MaskWith10 = 1,    // 0 for masked index
  MaskWith0Inf = 2,  // inf for masked index
};

namespace debug {
template <typename T>
struct type2string {
  static constexpr std::string_view str = "unknown_type";
};
template <>
struct type2string<float> {
  static constexpr std::string_view str = "float";
};
template <>
struct type2string<half> {
  static constexpr std::string_view str = "half";
};
template <>
struct type2string<bf16> {
  static constexpr std::string_view str = "bf16";
};
template <>
struct type2string<int32_t> {
  static constexpr std::string_view str = "int32_t";
};
template <>
struct type2string<int16_t> {
  static constexpr std::string_view str = "int16_t";
};
template <bool B = false>
struct bool2string {
  static constexpr std::string_view str = "false";
};
template <>
struct bool2string<true> {
  static constexpr std::string_view str = "true";
};
std::string toString(bool b) { return b ? "true" : "false"; }
std::string toString(MaskMode m) {
  switch (m) {
    case MaskMode::MaskDirectAdd:
      return "MaskDirectAdd";
    case MaskMode::MaskWith10:
      return "MaskWith10";
    case MaskMode::MaskWith0Inf:
      return "MaskWith0Inf";
    default:
      return "MaskUnknown";
  }
}
}  // namespace debug

constexpr float invalid_const = -1e15;

template <typename FT, bool SoftmaxLog = false>
class DefaultSoftmax {
 public:
  constexpr static bool fuse_log = SoftmaxLog;
  using Ptr_t = FT;
  using Params = struct {
    __host__ std::string toString() {
      std::stringstream ss;
      ss << "DefaultSoftmax<dt=" << debug::type2string<FT>::str << ", "
         << "SoftmaxLog=" << debug::bool2string<SoftmaxLog>::str << "> "
         << std::endl
         << "\t\t"
         << "alpha: " << alpha << ", batch: " << batch << ", align: " << align
         << ", "
         << "iptr: " << input_ptr << ", optr: " << output_ptr;
      return ss.str();
    }
    const FT* input_ptr;
    FT* output_ptr;
    float alpha;
    int batch;
    int align;  // [batch, align]
  };

 public:
  __host__ __device__ __forceinline__ DefaultSoftmax(const Params& p,
                                                     const IDX& block)
      : scale(p.alpha),
        in(p.input_ptr),
        out(p.output_ptr),
        batch(p.batch),
        align(p.align),
        block_bidx(block) {
    // if (threadIdx.x == 0) printf("G[%3d] - b[%3d], when batch=%3d, align=%3d,
    // scale = %f\n",
    //     blockIdx.x, int(block_bidx), int(batch), int(align), scale);
  }
  __device__ __forceinline__ bool is_valid_block() const {
    return block_bidx < batch;
  }
  __device__ __forceinline__ bool is_valid_align(const IDX& aidx) const {
    return aidx < align;
  }
  __device__ __forceinline__ IDX index(const IDX& aidx) {
    return block_bidx * align + aidx;
  }

 public:
  float scale = 1.f;
  const FT* in;
  FT* out;
  IDX batch, align;
  IDX block_bidx;
};  // DefaultSoftmax

template <typename FT, bool SoftmaxLog = false>
class AttnLognSoftmax {
 public:
  constexpr static bool fuse_log = SoftmaxLog;
  using Ptr_t = FT;
  using Params = struct {
    __host__ std::string toString() {
      std::stringstream ss;
      ss << "AttnLognSoftmax<dt=" << debug::type2string<FT>::str << ", "
         << "SoftmaxLog=" << debug::bool2string<SoftmaxLog>::str << "> "
         << std::endl
         << "\t\t"
         << "alpha: " << alpha << ", batch: " << batch << ", alogn: " << alogn
         << ", "
         << "nhead: " << nhead << ", align: " << align << ", "
         << "logn_base: " << logn_base << "iptr: " << input_ptr
         << ", optr: " << output_ptr;
      return ss.str();
    }
    const FT* input_ptr;
    FT* output_ptr;
    float alpha;
    int batch, alogn, nhead, align;
    int logn_base;
  };

 public:
  __host__ __device__ __forceinline__ AttnLognSoftmax(const Params& p,
                                                      const IDX& block)
      : scale(p.alpha),
        in(p.input_ptr),
        out(p.output_ptr),
        batch(p.batch),
        alogn(p.alogn),
        nhead(p.nhead),
        align(p.align) {
    block_bidx = block / (p.alogn * p.nhead);
    IDX div_ln = block % (p.alogn * p.nhead);
    block_lidx = div_ln / p.nhead;
    block_nidx = div_ln % p.nhead;
    // check: 1. valid alogn dim, 2. valid logn base length, 3. in logn zone.
    if (p.alogn > 1 && p.logn_base != 0 && block_lidx > p.logn_base) {
      scale = p.alpha * logf(static_cast<float>(block_lidx)) /
              logf(static_cast<float>(p.logn_base));
    }
    // if (threadIdx.x == 0) printf("G[%3d] - b[%3d]l[%3d]n[%3d], when
    // batch=%3d, alogn=%3d, nhead=%3d, align=%3d, scale = %f\n",
    //     blockIdx.x, int(block_bidx), int(block_lidx), int(block_nidx),
    //     int(batch), int(alogn), int(nhead), int(align), scale);
  }
  __device__ __forceinline__ bool is_valid_block() const {
    return block_bidx < batch && block_lidx < alogn;
  }  //  no need check nhead
  __device__ __forceinline__ bool is_valid_align(const IDX& aidx) const {
    return aidx < align;
  }
  __device__ __forceinline__ IDX index(const IDX& aidx) {
    return block_bidx * alogn * nhead * align + block_lidx * nhead * align +
           block_nidx * align + aidx;
  }

 public:
  float scale = 1.f;
  const FT* in;
  FT* out;
  IDX batch, alogn, nhead, align;
  IDX block_bidx, block_lidx, block_nidx;
};  // AttnLognSoftmax

template <typename MT, typename CPT, int Regs>
class NoMask {
 public:
  using Ptr_t = MT;
  using Params = struct {
    __host__ std::string toString() {
      std::stringstream ss;
      ss << "NoMask<dt=" << debug::type2string<MT>::str << ", "
         << "Regs=" << Regs << ">. ";
      return ss.str();
    }
  };

 public:
  __host__ __device__ __forceinline__ NoMask(const Params& p,
                                             const IDX& block) {}
  __device__ __forceinline__ CPT update_with_mask(CPT load_data,
                                                  const IDX& aidx) {
    return load_data;
  }
};  // NoMask

template <typename MT, typename CPT, int Regs, MaskMode Mode>
class DefaultMask {
 public:
  using Ptr_t = MT;
  using Params = struct {
    __host__ std::string toString() {
      std::stringstream ss;
      ss << "DefaultMask<dt=" << debug::type2string<MT>::str << ", "
         << "cpt=" << debug::type2string<CPT>::str << ", "
         << "Regs=" << Regs << ", Mode=" << debug::toString(Mode) << "> "
         << std::endl
         << "\t\t"
         << "decoder mask: " << debug::toString(mask_decoder) << ", "
         << "batch: " << batch << ", xseql: " << xseql << ", align: " << align
         << ", "
         << "beam-batch: " << beam_batch << ", beam-xseql: " << beam_xseql
         << ", "
         << "loop-nhead: " << loop_nhead << ", "
         << "mask-ptr: " << mask_ptr << ". ";
      return ss.str();
    }
    const MT* mask_ptr = nullptr;
    bool mask_decoder =
        false;  // decoder style mask, only use last index of xseql mask dim.
    int batch, xseql, align;     // [batch, xseql, align]
    int beam_batch, beam_xseql;  // beam_batch = data_batch / mask_batch,
                                 // beam_xseql = data_xseql / mask_xseql
    int loop_nhead;              // loop nhead times after xseql dim.
  };

 public:
  __host__ __device__ __forceinline__ DefaultMask(const Params& p,
                                                  const IDX& block)
      : ptr(p.mask_ptr),
        mask_batch(p.batch),
        mask_xseql(p.xseql),
        mask_align(p.align) {
    IDX non_head_index = block / p.loop_nhead;
    if (p.mask_decoder) {
      block_bidx = non_head_index / p.beam_batch;
      block_midx =
          p.xseql - 1;  // xseql dim always use [..., xseql - 1, ...] index.
    } else {
      IDX data_xseql = p.beam_xseql * p.xseql;
      IDX data_batch_index = non_head_index / data_xseql;
      IDX data_xseql_index = non_head_index % data_xseql;
      block_bidx = data_batch_index / p.beam_batch;
      block_midx = data_xseql_index /
                   p.beam_xseql;  // continuous xseql belong to same beam.
    }
    // if (threadIdx.x == 0) printf("G[%3d] - b[%3d]m[%3d], when batch=%3d,
    // xseql=%3d, align=%3d\n",
    //     blockIdx.x, int(block_bidx), int(block_midx), int(mask_batch),
    //     int(mask_xseql), int(mask_align));
  }

  __device__ __forceinline__ IDX index(const IDX& aidx) {
    return block_bidx * mask_xseql * mask_align + block_midx * mask_align +
           aidx;
  }

  __device__ __forceinline__ bool enable_mask(const IDX& aidx) const {
    return block_bidx < mask_batch && block_midx < mask_xseql &&
           aidx < mask_align;
  }

  __device__ __forceinline__ CPT update_with_mask(CPT load_data,
                                                  const IDX& aidx) {
    if (MaskMode::MaskDirectAdd == Mode && enable_mask(aidx)) {
      return load_data + static_cast<CPT>(ptr[index(aidx)]);
    }
    if (MaskMode::MaskWith0Inf == Mode && enable_mask(aidx)) {
      return load_data + static_cast<CPT>(ptr[index(aidx)]);
    }
    if (MaskMode::MaskWith10 == Mode && enable_mask(aidx)) {
      return load_data + static_cast<CPT>(1.f - ptr[index(aidx)]) *
                             static_cast<CPT>(invalid_const);
    }
    return load_data;
  }

 public:
  const MT* ptr;
  IDX mask_batch, mask_xseql, mask_align;
  IDX block_bidx, block_midx;  // batch, xseql index.
};                             // DefaultMask

}  // namespace softmax
}  // namespace attention
}  // namespace cuda
}  // namespace allspark

#endif  // __MHA_SOFTMAX_UTILS__
