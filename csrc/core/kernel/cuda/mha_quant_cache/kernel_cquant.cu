/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    kernel_cquant.cu
 */

#include "attn_kv_cache.hpp"
#include "kernel_utils.cuh"

namespace allspark {
namespace cuda {
namespace mha_quant_cache {
namespace quant_and_cache {
constexpr int32_t warp_num = 4;
constexpr int32_t block_size = warp_num * utils::warp_size;

// single regs group for single channel
// TODO(zhangyufei): optimize later
// qkv                  [batch, xseql, 3, nhead, phead]
// kcache               [batch, nhead, cache, phead]

template <typename T, int32_t hAlign = 192>
struct CoorpCtrl {
  constexpr static int32_t Loop =
      hAlign % 3 == 0 ? 3
                      : 4;  // align = 192, unroll = 3, align = 128, urnoll = 4
  constexpr static int32_t Pack =
      sizeof(float4) / sizeof(T);  // float 4,  half 8
  constexpr static int32_t Coorp =
      hAlign / Loop /
      Pack;  // float-256/192, 16, float-128/96 8, float-64/48 4, float-32/24 2
             //                    half-256/192 8, half-128/96 4,  half-64/48 2
  constexpr static int32_t Once =
      Pack *
      Coorp;  // for boundary check. 256/192 64, 128/96 32. 64/48 16, 32/24 8,
};

template <typename FT, typename QT, int32_t QTPack, int32_t hAlign = 64,
          int32_t QKV = 0, typename ZT = float, typename CPT = float>
__global__ void quant_to_cache_context_kernel(
    const FT* data,  // input float,            [batch, xseql, 3, nhead, phead]
    QT* quant,       // output quant            [batch, nhead, cache, phead]
    ZT* zero, CPT* scale,  // output param            [batch, nhead, cache]
    int32_t batch, int32_t nhead, int32_t phead, int32_t cache, int32_t xseql,
    u32div_t div_nhead, u32div_t div_cache) {
  static_assert(QTPack == 1 || QTPack == 2, "QTPack value error!");
  using ctrl_t = CoorpCtrl<FT, hAlign>;
  using packld_t = utils::packed_data<ctrl_t::Pack, FT>;
  using packst_t = utils::packed_data<ctrl_t::Pack / QTPack, QT>;
  constexpr float min_eps = 1e-5;
  CPT regs[ctrl_t::Loop][ctrl_t::Pack] = {0};

  int32_t tidx = blockIdx.x * blockDim.x + threadIdx.x;
  int32_t pidt = (tidx % ctrl_t::Coorp) * ctrl_t::Pack;
  auto cdivmod = div_cache.divmod(tidx / ctrl_t::Coorp);
  int32_t cidx = cdivmod.mod;  // use cache id, not xseql(xidx).
  auto ndivmod = div_nhead.divmod(cdivmod.div);
  int32_t nidx = ndivmod.mod;
  int32_t bidx = ndivmod.div;

  // load and cal.
  CPT fmax = static_cast<CPT>(-INFINITY);
  CPT fmin = static_cast<CPT>(+INFINITY);
#pragma unroll
  for (int32_t loop = 0; loop < ctrl_t::Loop; loop++) {
    int32_t pidx = loop * ctrl_t::Once + pidt;
    if (bidx < batch && nidx < nhead && cidx < xseql && pidx < phead) {
      int32_t iidx = bidx * xseql * 3 * nhead * phead +
                     cidx * 3 * nhead * phead + QKV * nhead * phead +
                     nidx * phead + pidx;
      packld_t ldg = reinterpret_cast<const packld_t*>(data + iidx)[0];
#pragma unroll
      for (int32_t pack = 0; pack < ctrl_t::Pack; pack++) {
        regs[loop][pack] = static_cast<CPT>(ldg.pack[pack]);
        fmax = max(fmax, regs[loop][pack]);
        fmin = min(fmin, regs[loop][pack]);
      }
    }
  }
  fmax = utils::ReduceThread<utils::MaxOp, CPT, ctrl_t::Coorp>(fmax);
  fmin = utils::ReduceThread<utils::MinOp, CPT, ctrl_t::Coorp>(fmin);
  float qs = __fdividef(
      static_cast<float>(fmax - fmin < min_eps ? min_eps : fmax - fmin),
      utils::quant_tag<QT, QTPack>::quant_div_scale);
  float qz = utils::quant_tag<QT, QTPack>::quant_zero -
             __fdividef(static_cast<float>(fmin), qs);
  qz = qz > utils::quant_tag<QT, QTPack>::quant_max
           ? utils::quant_tag<QT, QTPack>::quant_max
           : qz;
  qz = qz < utils::quant_tag<QT, QTPack>::quant_min
           ? utils::quant_tag<QT, QTPack>::quant_min
           : qz;
  qz = roundf(qz);

  // extract and pack QTPack CPT regs into a QT reg
  auto pack_data_regs_func = [](CPT* reg_ptr) {
    auto extract_func = [](CPT val) {
      uint8_t bit_reg = 0;
      if (std::is_same<QT, int8_t>::value && QTPack == 2) {
        int8_t tmp_reg = static_cast<int8_t>(val) << 4;
        bit_reg = reinterpret_cast<uint8_t&>(tmp_reg) >> 4;
      } else if (std::is_same<QT, int8_t>::value && QTPack == 1) {
        int8_t tmp_reg = static_cast<int8_t>(val);
        bit_reg = reinterpret_cast<uint8_t&>(tmp_reg);
      } else if (std::is_same<QT, uint8_t>::value) {
        bit_reg = static_cast<uint8_t>(val);
      }
      return bit_reg;
    };

    uint8_t dst_reg = 0;
    dst_reg = extract_func(reg_ptr[0]);
    if (QTPack == 2) {
      dst_reg |= extract_func(reg_ptr[1]) << 4;
    }
    QT ret_reg = reinterpret_cast<QT&>(dst_reg);
    return ret_reg;
  };

// quant
#pragma unroll
  for (int32_t loop = 0; loop < ctrl_t::Loop; loop++) {
    int32_t pidx = loop * ctrl_t::Once + pidt;
    int32_t oidx = (bidx * nhead * cache * phead + nidx * cache * phead +
                    cidx * phead + pidx) /
                   QTPack;
    if (bidx < batch && nidx < nhead && pidx < phead) {
      packst_t stg;
      if (cidx < xseql) {
#pragma unroll
        for (int32_t pack = 0; pack < ctrl_t::Pack / QTPack; pack++) {
#pragma unroll
          for (int32_t qt_pack = 0; qt_pack < QTPack; ++qt_pack) {
            int32_t pack_idx = pack * QTPack + qt_pack;
            regs[loop][pack_idx] = static_cast<CPT>(
                qz + __fdividef(static_cast<float>(regs[loop][pack_idx]), qs));
            regs[loop][pack_idx] =
                regs[loop][pack_idx] > utils::quant_tag<QT, QTPack>::quant_max
                    ? utils::quant_tag<QT, QTPack>::quant_max
                    : regs[loop][pack_idx];
            regs[loop][pack_idx] =
                regs[loop][pack_idx] < utils::quant_tag<QT, QTPack>::quant_min
                    ? utils::quant_tag<QT, QTPack>::quant_min
                    : regs[loop][pack_idx];
            regs[loop][pack_idx] = roundf(regs[loop][pack_idx]);
          }
          stg.pack[pack] = pack_data_regs_func(&(regs[loop][pack * QTPack]));
        }
      } else {
// clear cache zone
#pragma unroll
        for (int32_t pack = 0; pack < ctrl_t::Pack / QTPack; pack++) {
          stg.pack[pack] = 0;
        }
      }
      reinterpret_cast<packst_t*>(quant + oidx)[0] = stg;
    }
  }

  // params.
  int32_t qidx = bidx * nhead * cache + nidx * cache + cidx;
  if (bidx < batch && nidx < nhead) {
    if (scale && pidt == 0)
      scale[qidx] = cidx < xseql ? static_cast<CPT>(qs) : static_cast<CPT>(0.f);
    if (zero && pidt == 0)
      zero[qidx] = cidx < xseql ? static_cast<ZT>(qz) : static_cast<ZT>(0.f);
  }
}

template <typename FT, typename QT>
struct quant_to_kvcache_context_impl {
  // input float,            [batch, xseql, 3, nhead, phead]
  // output quant            [batch, nhead, cache, phead]
  // output param            [batch, nhead, cache]
  void operator()(cudaStream_t stream, const void* data, QT* kc, float* kz,
                  float* ks, QT* vc, float* vz, float* vs, int32_t batch,
                  int32_t nhead, int32_t phead, int32_t cache, int32_t xseql,
                  QuantType quant_type) {
    const FT* fval = reinterpret_cast<const FT*>(data);

#define DISPATCH_PHEAD_LESS_THAN(HEAD_ALIGN)                              \
  if (phead <= HEAD_ALIGN) {                                              \
    _alignment_dispatch<HEAD_ALIGN>(stream, fval, kc, kz, ks, vc, vz, vs, \
                                    batch, nhead, phead, cache, xseql,    \
                                    quant_type);                          \
    return;                                                               \
  }
    DISPATCH_PHEAD_LESS_THAN(48);
    DISPATCH_PHEAD_LESS_THAN(64);
    DISPATCH_PHEAD_LESS_THAN(96);
    DISPATCH_PHEAD_LESS_THAN(128);
    DISPATCH_PHEAD_LESS_THAN(192);
    DISPATCH_PHEAD_LESS_THAN(256);
    DISPATCH_PHEAD_LESS_THAN(512);
#undef DISPATCH_PHEAD_LESS_THAN
    LOG(ERROR) << "head size larger than 512 not support";
  }

  template <int32_t kAlign = 192>
  void _alignment_dispatch(cudaStream_t stream, const FT* data, QT* kc,
                           float* kz, float* ks, QT* vc, float* vz, float* vs,
                           int32_t batch, int32_t nhead, int32_t phead,
                           int32_t cache, int32_t xseql, QuantType quant_type) {
    constexpr int32_t kBlock = 128;
    constexpr int32_t kCoorp = CoorpCtrl<FT, kAlign>::Coorp;
    constexpr int32_t kPack = CoorpCtrl<FT, kAlign>::Pack;
    constexpr int32_t perBlock = kBlock / kCoorp;
    int32_t kGrid = utils::cal_ceil(batch * cache * nhead, perBlock);
    if (phead % kPack != 0) {
      LOG(ERROR) << "not support non-aligned head size.";
      return;
    }

    if (quant_type == QuantType::INT8) {
      constexpr int QTPack = 1;
      quant_to_cache_context_kernel<FT, QT, QTPack, kAlign, 1>
          <<<kGrid, kBlock, 0, stream>>>(data, kc, kz, ks, batch, nhead, phead,
                                         cache, xseql, u32div_t(nhead),
                                         u32div_t(cache));
      quant_to_cache_context_kernel<FT, QT, QTPack, kAlign, 2>
          <<<kGrid, kBlock, 0, stream>>>(data, vc, vz, vs, batch, nhead, phead,
                                         cache, xseql, u32div_t(nhead),
                                         u32div_t(cache));
    } else if (quant_type == QuantType::UINT4) {
      constexpr int QTPack = 2;
      quant_to_cache_context_kernel<FT, QT, QTPack, kAlign, 1>
          <<<kGrid, kBlock, 0, stream>>>(data, kc, kz, ks, batch, nhead, phead,
                                         cache, xseql, u32div_t(nhead),
                                         u32div_t(cache));
      quant_to_cache_context_kernel<FT, QT, QTPack, kAlign, 2>
          <<<kGrid, kBlock, 0, stream>>>(data, vc, vz, vs, batch, nhead, phead,
                                         cache, xseql, u32div_t(nhead),
                                         u32div_t(cache));
    }
  }
};

// single warp for single channel
// TODO(zhangyufei): optimize later
// qkv                  [batch,     3, nhead, phead]
// kcache               [batch, nhead, cache, phead]
// kcache offset                         offset,

template <typename FT, typename QT, int32_t QTPack, int32_t UNROLL,
          int32_t QKV = 0, typename ZT = float, typename RT = float,
          typename CPT = float>
__global__ void quant_to_cache_decoder_kernel(
    const FT* data,  // input float,     assume [batch,     3, nhead, phead]
    QT* quant,       // output quant,    assume [batch, nhead, cache, phead]
    ZT* zero, RT* reduce,
    CPT* scale,  // output param,    assume [batch, nhead, cache]
    int32_t batch, int32_t nhead, int32_t phead, int32_t cache, int32_t offset,
    u32div_t div_nhead) {
  static_assert((UNROLL % 2 == 1 && QTPack == 2) == false, "");
  // PACK INFER
  constexpr int32_t PACK = UNROLL % 4 == 0 ? 4 : UNROLL % 2 == 0 ? 2 : 1;
  constexpr int32_t LOOP = UNROLL / PACK;
  using packld_t = utils::packed_data<PACK, FT>;
  using packst_t = utils::packed_data<PACK / QTPack, QT>;
  CPT data_regs[UNROLL] = {0};

  int32_t warp = threadIdx.x / utils::warp_size;
  int32_t lane = threadIdx.x % utils::warp_size;
  int32_t cidx = blockIdx.x * warp_num + warp;  // channel index
  auto ndivmod = div_nhead.divmod(cidx);
  int32_t bidx = ndivmod.div;  // batch index
  int32_t nidx = ndivmod.mod;  // nhead index

  // load and min / max
  CPT fmax = static_cast<CPT>(-INFINITY);
  CPT fmin = static_cast<CPT>(+INFINITY);
#pragma unroll
  for (int32_t loop = 0; loop < LOOP; loop++) {
    int32_t pidx = loop * utils::warp_size * PACK + lane * PACK;
    int32_t iidx =
        bidx * 3 * nhead * phead + QKV * nhead * phead + nidx * phead + pidx;
    if (pidx < phead && nidx < nhead && bidx < batch) {
      packld_t data_ldg = reinterpret_cast<const packld_t*>(data + iidx)[0];
#pragma unroll
      for (int32_t pack = 0; pack < PACK; pack++) {
        int32_t ridx = loop * PACK + pack;
        data_regs[ridx] = static_cast<CPT>(data_ldg.pack[pack]);
        fmax = max(fmax, data_regs[ridx]);
        fmin = min(fmin, data_regs[ridx]);
      }
    }
  }
  fmax = utils::ReduceThread<utils::MaxOp, CPT>(fmax);
  fmin = utils::ReduceThread<utils::MinOp, CPT>(fmin);

  // cal param
  float qscale = __fdividef(static_cast<float>(fmax - fmin),
                            utils::quant_tag<QT, QTPack>::quant_div_scale);
  float qzero = utils::quant_tag<QT, QTPack>::quant_zero -
                __fdividef(static_cast<float>(fmin), qscale);
  qzero = qzero > utils::quant_tag<QT, QTPack>::quant_max
              ? utils::quant_tag<QT, QTPack>::quant_max
              : qzero;
  qzero = qzero < utils::quant_tag<QT, QTPack>::quant_min
              ? utils::quant_tag<QT, QTPack>::quant_min
              : qzero;
  qzero = roundf(qzero);

  // store scale and zero, quant param index
  int32_t qidx = bidx * nhead * cache + nidx * cache + offset;
  if (nidx < nhead && bidx < batch) {
    if (scale && lane == 0) scale[qidx] = static_cast<CPT>(qscale);
    if (zero && lane == 0) zero[qidx] = static_cast<ZT>(qzero);
  }
  // reduce sum
  CPT qrsum = 0;
#pragma unroll
  for (int32_t loop = 0; loop < LOOP; loop++) {
#pragma unroll
    for (int32_t pack = 0; pack < PACK; pack++) {
      int32_t pidx = loop * utils::warp_size * PACK + lane * PACK;
      int32_t ridx = loop * PACK + pack;
      if (pidx < phead && nidx < nhead && bidx < batch) {
        data_regs[ridx] = static_cast<CPT>(
            qzero + __fdividef(static_cast<float>(data_regs[ridx]), qscale));
        data_regs[ridx] =
            data_regs[ridx] > utils::quant_tag<QT, QTPack>::quant_max
                ? utils::quant_tag<QT, QTPack>::quant_max
                : data_regs[ridx];
        data_regs[ridx] =
            data_regs[ridx] < utils::quant_tag<QT, QTPack>::quant_min
                ? utils::quant_tag<QT, QTPack>::quant_min
                : data_regs[ridx];
        data_regs[ridx] = roundf(data_regs[ridx]);
        qrsum += data_regs[ridx];
      }
    }
  }
  qrsum = utils::ReduceThread<utils::SumOp, CPT>(qrsum);
  if (nidx < nhead && bidx < batch) {
    if (reduce && lane == 0) reduce[qidx] = static_cast<RT>(qrsum);
  }
  // extract and pack QTPack CPT regs into a QT reg
  auto pack_data_regs_func = [](CPT* reg_ptr) {
    auto extract_func = [](CPT val) {
      uint8_t bit_reg = 0;
      if (std::is_same<QT, int8_t>::value && QTPack == 2) {
        int8_t tmp_reg = static_cast<int8_t>(val) << 4;
        bit_reg = reinterpret_cast<uint8_t&>(tmp_reg) >> 4;
      } else if (std::is_same<QT, int8_t>::value && QTPack == 1) {
        int8_t tmp_reg = static_cast<int8_t>(val);
        bit_reg = reinterpret_cast<uint8_t&>(tmp_reg);
      } else if (std::is_same<QT, uint8_t>::value) {
        bit_reg = static_cast<uint8_t>(val);
      }
      return bit_reg;
    };

    uint8_t dst_reg = 0;
    dst_reg = extract_func(reg_ptr[0]);
    if (QTPack == 2) {
      dst_reg |= extract_func(reg_ptr[1]) << 4;
    }
    QT ret_reg = reinterpret_cast<QT&>(dst_reg);
    return ret_reg;
  };

// store
#pragma unroll
  for (int32_t loop = 0; loop < LOOP; loop++) {
    int32_t pidx = loop * utils::warp_size * PACK + lane * PACK;
    int32_t oidx = (bidx * nhead * cache * phead + nidx * cache * phead +
                    offset * phead + pidx) /
                   QTPack;
    packst_t data_stg;
#pragma unroll
    for (int32_t pack = 0; pack < PACK / QTPack; pack++) {
      int32_t ridx = loop * PACK + pack * QTPack;
      data_stg.pack[pack] = pack_data_regs_func(&(data_regs[ridx]));
    }
    if (pidx < phead && nidx < nhead && bidx < batch) {
      reinterpret_cast<packst_t*>(quant + oidx)[0] = data_stg;
    }
  }
}

#define LOAD_KV_CACHE_WHEN(UR)                                                \
  if (kUNROLL <= UR) {                                                        \
    constexpr int QTPack = 1;                                                 \
    quant_to_cache_decoder_kernel<FT, QT, QTPack, UR, 1, float, float, float> \
        <<<kGrid, kBlock, 0, stream>>>(tdata, kcache, kzero, kreduce, kscale, \
                                       batch, nhead, phead, cache, offset,    \
                                       div_nhead);                            \
    quant_to_cache_decoder_kernel<FT, QT, QTPack, UR, 2, float, float, float> \
        <<<kGrid, kBlock, 0, stream>>>(tdata, vcache, vzero, vreduce, vscale, \
                                       batch, nhead, phead, cache, offset,    \
                                       div_nhead);                            \
    return;                                                                   \
  }

#define LOAD_U4_KV_CACHE_WHEN(UR)                                             \
  if (kUNROLL <= UR) {                                                        \
    constexpr int QTPack = 2;                                                 \
    quant_to_cache_decoder_kernel<FT, QT, QTPack, UR, 1, float, float, float> \
        <<<kGrid, kBlock, 0, stream>>>(tdata, kcache, kzero, kreduce, kscale, \
                                       batch, nhead, phead, cache, offset,    \
                                       div_nhead);                            \
    quant_to_cache_decoder_kernel<FT, QT, QTPack, UR, 2, float, float, float> \
        <<<kGrid, kBlock, 0, stream>>>(tdata, vcache, vzero, vreduce, vscale, \
                                       batch, nhead, phead, cache, offset,    \
                                       div_nhead);                            \
    return;                                                                   \
  }

template <typename FT, typename QT>
struct load_and_quant_to_kvcache_impl {
  // data     [batch, 3, nhead, phead]
  // k/vcache [batch, 1, nhead, cache, phead]
  // k/vparam [batch, 1, nhead, cache]
  void operator()(cudaStream_t stream, const void* data, QT* kcache,
                  float* kzero, float* kscale, QT* vcache, float* vzero,
                  float* vscale, int32_t batch, int32_t nhead, int32_t phead,
                  int32_t cache, int32_t offset, QuantType quant_type) {
    float* kreduce = nullptr;
    float* vreduce = nullptr;
    int32_t kUNROLL = utils::cal_ceil(phead, utils::warp_size);
    if (phead % utils::warp_size != 0) {
      // TODO(zhangyufei): print warning.
      // printf("head size better aligned to %d, current is %d\n",
      // utils::warp_size, phead);
    }
    int32_t kGrid = utils::cal_ceil(batch * nhead, warp_num);
    constexpr int32_t kBlock = block_size;
    u32div_t div_nhead(nhead);
    const FT* tdata = reinterpret_cast<const FT*>(data);

    if (quant_type == QuantType::INT8) {
      LOAD_KV_CACHE_WHEN(1);
      LOAD_KV_CACHE_WHEN(2);
      LOAD_KV_CACHE_WHEN(3);  // phead = 96,  m6-3b
      LOAD_KV_CACHE_WHEN(4);  // phead = 128  m6-13b
      LOAD_KV_CACHE_WHEN(5);
      LOAD_KV_CACHE_WHEN(6);
      LOAD_KV_CACHE_WHEN(8);   // phead = 256
      LOAD_KV_CACHE_WHEN(12);  // phead = 384
      LOAD_KV_CACHE_WHEN(16);  // phead = 512
      LOG(ERROR) << "head size larger than 512 not support";
    } else if (quant_type == QuantType::UINT4) {
      LOAD_U4_KV_CACHE_WHEN(2);
      LOAD_U4_KV_CACHE_WHEN(4);
      LOAD_U4_KV_CACHE_WHEN(6);
      LOAD_U4_KV_CACHE_WHEN(8);
      LOAD_U4_KV_CACHE_WHEN(12);
      LOAD_U4_KV_CACHE_WHEN(16);
      LOG(ERROR) << "head size larger than 512 not support";
    }
  }
};
#undef LOAD_U4_KV_CACHE_WHEN
#undef LOAD_KV_CACHE_WHEN

#define LOAD_QUERY_WHEN(UR)                                                    \
  if (kUNROLL <= UR) {                                                         \
    quant_to_cache_decoder_kernel<TYPE, int8_t, 1, UR, 0, float, float, float> \
        <<<kGrid, kBlock, 0, stream>>>(tdata, query, zero, reduce, scale,      \
                                       batch, nhead, phead, 1, 0, div_nhead);  \
    return;                                                                    \
  }

template <typename TYPE>
struct load_and_quant_query_impl {
  // data     [batch, 3, nhead, phead]
  // quant    [batch, 1, nhead, phead]
  // param    [batch, 1, nhead]
  void operator()(cudaStream_t stream, const void* data, int8_t* query,
                  float* zero, float* reduce, float* scale, int32_t batch,
                  int32_t nhead, int32_t phead) {
    int32_t kUNROLL = utils::cal_ceil(phead, utils::warp_size);
    if (phead % utils::warp_size != 0) {
      // TODO(zhangyufei): print warning.
      // printf("head size better aligned to %d, current is %d\n",
      // utils::warp_size, phead);
    }
    int32_t kGrid = utils::cal_ceil(batch * nhead, warp_num);
    constexpr int32_t kBlock = block_size;
    u32div_t div_nhead(nhead);
    const TYPE* tdata = reinterpret_cast<const TYPE*>(data);

    LOAD_QUERY_WHEN(1);
    LOAD_QUERY_WHEN(2);
    LOAD_QUERY_WHEN(3);
    LOAD_QUERY_WHEN(4);  // phead = 128
    LOAD_QUERY_WHEN(5);
    LOAD_QUERY_WHEN(6);
    LOAD_QUERY_WHEN(8);   // phead = 256
    LOAD_QUERY_WHEN(12);  // phead = 384
    LOAD_QUERY_WHEN(16);  // phead = 512
    LOG(ERROR) << "head size larger than 512 not support";
  }
};
#undef LOAD_QUERY_WHEN

}  // namespace quant_and_cache

#define LOAD_AND_QUANT_TO_KV_CACHE_CONTEXT_IMPL(FT, QT)                    \
  template <>                                                              \
  void load_and_quant_to_kv_cache_context<FT, QT>(                         \
      cudaStream_t stream, const FT* qkv, QT* kc, float* kz, float* ks,    \
      QT* vc, float* vz, float* vs, int32_t batch, int32_t nhead,          \
      int32_t phead, int32_t cache, int32_t xseql, QuantType quant_type) { \
    quant_and_cache::quant_to_kvcache_context_impl<FT, QT>()(              \
        stream, qkv, kc, kz, ks, vc, vz, vs, batch, nhead, phead, cache,   \
        xseql, quant_type);                                                \
  }

#define LOAD_AND_QUANT_TO_KV_CACHE_DECODER_IMPL(FT, QT)                     \
  template <>                                                               \
  void load_and_quant_to_kv_cache_decoder<FT, QT>(                          \
      cudaStream_t stream, const FT* qkv, QT* kc, float* kz, float* ks,     \
      QT* vc, float* vz, float* vs, int32_t batch, int32_t nhead,           \
      int32_t phead, int32_t cache, int32_t offset, QuantType quant_type) { \
    quant_and_cache::load_and_quant_to_kvcache_impl<FT, QT>()(              \
        stream, qkv, kc, kz, ks, vc, vz, vs, batch, nhead, phead, cache,    \
        offset, quant_type);                                                \
  }

#define LOAD_AND_QUANT_TO_I8_QUERY_DECODER_IMPL(FT)                            \
  template <>                                                                  \
  void load_and_quant_to_i8_query_decoder<FT>(                                 \
      cudaStream_t stream, const FT* qkv, int8_t* qq, float* qz, float* qr,    \
      float* qs, int32_t batch, int32_t nhead, int32_t phead) {                \
    quant_and_cache::load_and_quant_query_impl<FT>()(stream, qkv, qq, qz, qr,  \
                                                     qs, batch, nhead, phead); \
  }

LOAD_AND_QUANT_TO_KV_CACHE_CONTEXT_IMPL(float, int8_t);
LOAD_AND_QUANT_TO_KV_CACHE_CONTEXT_IMPL(float, uint8_t);
LOAD_AND_QUANT_TO_KV_CACHE_DECODER_IMPL(float, int8_t);
LOAD_AND_QUANT_TO_KV_CACHE_DECODER_IMPL(float, uint8_t);
LOAD_AND_QUANT_TO_I8_QUERY_DECODER_IMPL(float);
#if ENABLE_FP16
LOAD_AND_QUANT_TO_KV_CACHE_CONTEXT_IMPL(half_t, int8_t);
LOAD_AND_QUANT_TO_KV_CACHE_CONTEXT_IMPL(half_t, uint8_t);
LOAD_AND_QUANT_TO_KV_CACHE_DECODER_IMPL(half_t, int8_t);
LOAD_AND_QUANT_TO_KV_CACHE_DECODER_IMPL(half_t, uint8_t);
LOAD_AND_QUANT_TO_I8_QUERY_DECODER_IMPL(half_t);
#endif  // ENABLE_FP16
#if ENABLE_BF16
LOAD_AND_QUANT_TO_KV_CACHE_CONTEXT_IMPL(__hie_buildin::bfloat16, int8_t);
LOAD_AND_QUANT_TO_KV_CACHE_CONTEXT_IMPL(__hie_buildin::bfloat16, uint8_t);
LOAD_AND_QUANT_TO_KV_CACHE_DECODER_IMPL(__hie_buildin::bfloat16, int8_t);
LOAD_AND_QUANT_TO_KV_CACHE_DECODER_IMPL(__hie_buildin::bfloat16, uint8_t);
LOAD_AND_QUANT_TO_I8_QUERY_DECODER_IMPL(__hie_buildin::bfloat16);
#endif  // ENABLE_BF16

}  // namespace mha_quant_cache
}  // namespace cuda
}  // namespace allspark
