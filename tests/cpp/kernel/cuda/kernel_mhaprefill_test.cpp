/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    kernel_mhaprefill_test.cpp
 */

#include <core/kernel/cuda/flashv2/flashv2.h>
#include <core/kernel/cuda/trivial_mha/trivial_mha.h>
#include <core/kernel/cuda/xformer_mha/xformer_mha.h>
#include <test_common.h>

#include <algorithm>
#include <cmath>
#include <common.hpp>
#include <iostream>
#include <vector>
#if 0
void reference_prefill_attention(
        const allspark::cuda::trivial_t& param,
        const float* concat,  // batch, seqlen, 3, nhead, phead
        float* output,        // batch, seqlen, nhead, phead
        float alpha, bool causal) {
    // printf("reference prefill, batch = %d, nhead = %d, phead = %d, seqlen = %d\n",
    //     param.batch, param.nhead, param.phead, param.seqlen);
    for (size_t bidx = 0; bidx < param.batch; bidx++) {
        for (size_t nidx = 0; nidx < param.nhead; nidx++) {
            size_t query_max_index[param.seqlen];
            float query_max[param.seqlen];
            float query_sum[param.seqlen];
            float score[param.seqlen * param.seqlen];
            // gemm.qxk
            size_t concat_bn_offset =
                bidx * param.seqlen * 3 * param.nhead * param.phead +
                nidx * param.phead;
            const float* cbnp = concat + concat_bn_offset;
            for (size_t qidx = 0; qidx < param.seqlen; qidx++) {
                query_max[qidx] = -INFINITY;
                for (size_t kidx = 0; kidx < param.seqlen; kidx++) {
                    if (causal && kidx > qidx) {
                        // masking score.
                        score[qidx * param.seqlen + kidx] = -INFINITY;
                    } else {
                        float qxk_phead_sum = 0.f;
                        for (size_t pidx = 0; pidx < param.phead; pidx++) {
                            qxk_phead_sum += alpha *
                                cbnp[qidx * 3 * param.nhead * param.phead + pidx] *
                                cbnp[kidx * 3 * param.nhead * param.phead + pidx +
                                            1 * param.nhead * param.phead];
                        }
                        query_max_index[qidx] =
                                query_max[qidx] > qxk_phead_sum ?
                                query_max_index[qidx] : kidx;
                        query_max[qidx] =
                                query_max[qidx] > qxk_phead_sum ?
                                query_max[qidx] : qxk_phead_sum;
                        score[qidx * param.seqlen + kidx] = qxk_phead_sum;
                    }
                }
            }
            // softmax.exp&sum
            for (size_t qidx = 0; qidx < param.seqlen; qidx++) {
                query_sum[qidx] = 0.f;
                for (size_t kidx = 0; kidx < param.seqlen; kidx++) {
                    score[qidx * param.seqlen + kidx] = expf(score[qidx * param.seqlen + kidx] - query_max[qidx]);
                    query_sum[qidx] += score[qidx * param.seqlen + kidx];
                }
            }
            // softmax.div
            for (size_t qidx = 0; qidx < param.seqlen; qidx++) {
                for (size_t kidx = 0; kidx < param.seqlen; kidx++) {
                    score[qidx * param.seqlen + kidx] /= query_sum[qidx];
                }
            }
            // debug
            for (size_t qidx = 0; qidx < param.seqlen; qidx++) {
                size_t softmax_t0_index = 0;
                size_t softmax_t1_index = 0;
                float softmax_t0_value = -INFINITY;
                float softmax_t1_value = -INFINITY;
                for (size_t kidx = 0; kidx < param.seqlen; kidx++) {
                    float score_qk = score[qidx * param.seqlen + kidx];
                    if (score_qk > softmax_t0_value) {
                        softmax_t1_value = softmax_t0_value;
                        softmax_t1_index = softmax_t0_index;
                        softmax_t0_value = score_qk;
                        softmax_t0_index = kidx;
                    } else if (score_qk > softmax_t1_value) {
                        softmax_t1_value = score_qk;
                        softmax_t1_index = kidx;
                    }
                }
                // printf("bn[%3d|%3d]x[%5d] softmax max = %2.3f, sum = %2.3f -> t0 = %2.3f / %3d, t1 = %2.3f / %3d\n",
                //     bidx, nidx, qidx, query_max[qidx], query_sum[qidx], softmax_t0_value, softmax_t0_index, softmax_t1_value, softmax_t1_index);
            }
            // gemm.sxv
            size_t output_bn_offset =
                bidx * param.seqlen * param.nhead * param.phead +
                nidx * param.phead;
            float* obnp = output + output_bn_offset;
            for (size_t qidx = 0; qidx < param.seqlen; qidx++) {
                for (size_t pidx = 0; pidx < param.phead; pidx++) {
                    float sxv_seqlen_sum = 0.f;
                    for (size_t kidx = 0; kidx < param.seqlen; kidx++) {
                        sxv_seqlen_sum +=
                            score[qidx * param.seqlen + kidx] *
                            cbnp[kidx * 3 * param.nhead * param.phead + pidx +
                                        2 * param.nhead * param.phead];
                    }
                    obnp[qidx * param.nhead * param.phead + pidx] = sxv_seqlen_sum;
                }
            }
        }
    }
}
#endif  // 0

#if 1
template <typename TYPE>
bool pefill_check_with_reference(
    const allspark::cuda::trivial_t& param,
    const float* concat,  // batch, seqlen, 3, nhead, phead
    const TYPE* output,   // batch, seqlen, 1, nhead, phead.  output for check.
    // float* output,        // batch, seqlen, nhead, phead
    float alpha, bool causal, float feps = 1e-3) {
  std::vector<size_t> nan_list;
  std::vector<size_t> err_list;
  std::vector<size_t> wrn_list;
  // printf("reference prefill, batch = %d, nhead = %d, phead = %d, seqlen =
  // %d\n",
  //     param.batch, param.nhead, param.phead, param.seqlen);
  for (size_t bidx = 0; bidx < param.batch; bidx++) {
    for (size_t nidx = 0; nidx < param.nhead; nidx++) {
      std::vector<std::vector<float> > score;
      std::vector<size_t> query_max_index;
      std::vector<float> query_max;
      std::vector<float> query_sum;
      std::vector<std::vector<float> > reference;
      std::vector<float> softmax_top0;
      std::vector<size_t> softmax_top0_index;
      std::vector<float> softmax_top1;
      std::vector<size_t> softmax_top1_index;
      bool warning_only[param.seqlen];  // <-- for given q-seqlen, if softmax
                                        // top0 / top1 too close or top1 <
                                        // threshold, print warning only

      // input
      size_t concat_bn_offset =
          bidx * param.seqlen * 3 * param.nhead * param.phead +
          nidx * param.phead;
      const float* cbnp = concat + concat_bn_offset;

      // gemm.qxk
      for (size_t qidx = 0; qidx < param.seqlen; qidx++) {
        float local_query_max = -INFINITY;
        size_t local_query_max_index = 0;
        std::vector<float> score_query(param.seqlen, 0.f);
        for (size_t kidx = 0; kidx < param.seqlen; kidx++) {
          if (causal && kidx > qidx) {
            // masking score.
            score_query[kidx] = -INFINITY;
          } else {
            float qxk_phead_sum = 0.f;
            for (size_t pidx = 0; pidx < param.phead; pidx++) {
              qxk_phead_sum +=
                  alpha * cbnp[qidx * 3 * param.nhead * param.phead + pidx] *
                  cbnp[kidx * 3 * param.nhead * param.phead + pidx +
                       1 * param.nhead * param.phead];
            }
            local_query_max_index =
                local_query_max > qxk_phead_sum ? local_query_max_index : kidx;
            local_query_max = local_query_max > qxk_phead_sum ? local_query_max
                                                              : qxk_phead_sum;
            score_query[kidx] = qxk_phead_sum;
          }
        }
        query_max.push_back(local_query_max);
        query_max_index.push_back(local_query_max_index);
        score.push_back(score_query);
      }
      // softmax.exp&sum
      for (size_t qidx = 0; qidx < param.seqlen; qidx++) {
        float local_query_sum = 0.f;
        for (size_t kidx = 0; kidx < param.seqlen; kidx++) {
          score[qidx][kidx] = expf(score[qidx][kidx] - query_max[qidx]);
          local_query_sum += score[qidx][kidx];
        }
        query_sum.push_back(local_query_sum);
      }
      // softmax.div
      for (size_t qidx = 0; qidx < param.seqlen; qidx++) {
        for (size_t kidx = 0; kidx < param.seqlen; kidx++) {
          score[qidx][kidx] /= query_sum[qidx];
        }
      }
      // warning only
      for (size_t qidx = 0; qidx < param.seqlen; qidx++) {
        float qst0 = 0.f;
        float qst1 = 0.f;
        size_t qsti0 = 0;
        size_t qsti1 = 0;

        // printf("BNQ[%3d,%3d,%3d] softmax max = %2.3f, sum = %2.3f\n",
        //     bidx, nidx, qidx, query_max[qidx], query_sum[qidx]);
        // printf("BNQ[%3d,%3d,%3d] softmax = ", bidx, nidx, qidx);
        for (size_t kidx = 0; kidx < param.seqlen; kidx++) {
          // float score_qk = score[qidx * param.seqlen + kidx];
          float score_qk = score[qidx][kidx];
          // printf("\t%2.3f,", score_qk);
          if (score_qk > qst0) {
            qst1 = qst0;
            qsti1 = qsti0;
            qst0 = score_qk;
            qsti0 = kidx;
          } else if (score_qk > qst1) {
            qst1 = score_qk;
            qsti1 = kidx;
          }
        }
        // printf("\n");
        softmax_top0.push_back(qst0);
        softmax_top0_index.push_back(qsti0);
        softmax_top1.push_back(qst1);
        softmax_top1_index.push_back(qsti1);
        warning_only[qidx] =
            (qst0 < 0.6) || (qst1 > 0.15) || (qst0 - qst1 < 0.55);
      }

      // gemm.sxv
      for (int64_t qidx = 0; qidx < param.seqlen; qidx++) {
        std::vector<float> reference_phead(param.phead, 0.f);
        for (int64_t pidx = 0; pidx < param.phead; pidx++) {
          float sxv_seqlen_sum = 0.f;
          for (int64_t kidx = 0; kidx < param.seqlen; kidx++) {
            sxv_seqlen_sum +=
                score[qidx][kidx] * cbnp[kidx * 3 * param.nhead * param.phead +
                                         pidx + 2 * param.nhead * param.phead];
          }
          reference_phead[pidx] = sxv_seqlen_sum;
        }
        reference.push_back(reference_phead);
      }

      // check
      for (size_t qidx = 0; qidx < param.seqlen; qidx++) {
        std::vector<size_t> local_error;
        std::vector<size_t> local_warning;

        size_t output_index = bidx * param.seqlen * param.nhead * param.phead +
                              qidx * param.nhead * param.phead +
                              nidx * param.phead;
        for (size_t pidx = 0; pidx < param.phead; pidx++) {
          if (std::isnan(static_cast<float>(output[output_index + pidx]))) {
            nan_list.push_back(output_index + pidx);
            local_error.push_back(pidx);
            continue;
          }
          float ref = reference[qidx][pidx];
          float val = static_cast<float>(output[output_index + pidx]);
          float eps = fabs(ref) * feps > feps ? fabs(ref) * feps : feps;
          // printf("[diff][b%3d,n%3d,q%3d][p%3d] \t%2.5f, \t%2.5f\n",
          //     bidx, nidx, qidx, pidx, ref, val);
          if (fabs(ref - val) < eps) {
            continue;
          } else {
            if (warning_only[qidx]) {
              wrn_list.push_back(output_index + pidx);
              local_warning.push_back(pidx);
            } else {
              err_list.push_back(output_index + pidx);
              local_error.push_back(pidx);
            }
          }
        }

        if (local_error.size()) {
          printf("\n[ERR] top0 index[%3d] = %2.3f top1 index[%3d] = %2.3f\n",
                 int(softmax_top0_index[qidx]), softmax_top0[qidx],
                 int(softmax_top1_index[qidx]), softmax_top1[qidx]);
        }
        for (size_t eidx = 0; eidx < local_error.size(); eidx++) {
          float ref = reference[qidx][local_error[eidx]];
          float val =
              static_cast<float>(output[output_index + local_error[eidx]]);
          float eps = fabs(ref) * feps > feps ? fabs(ref) * feps : feps;
          printf("[ERR] B[%3d]N[%3d] SeqId = %3d, index = %3d, ", bidx, nidx,
                 qidx, local_error[eidx]);
          printf("\tref = %2.4f, \tval = %2.4f, where eps = \t%2.4f / %2.4f\n",
                 ref, val, fabs(ref - val), eps);
        }

        // if (local_warning.size()) {
        //     printf("\n[WARN] top0 index[%3d] = %2.3f top1 index[%3d] =
        //     %2.3f\n",
        //         int(softmax_top0_index[qidx]), softmax_top0[qidx],
        //         int(softmax_top1_index[qidx]), softmax_top1[qidx]);
        // }
        // for (size_t widx = 0; widx < local_warning.size(); widx++) {
        //     float ref = reference[qidx][local_warning[widx]];
        //     float val = static_cast<float>(output[output_index +
        //     local_warning[widx]]); float eps = fabs(ref) * feps > feps ?
        //     fabs(ref) * feps : feps; printf("[WARN] B[%3d]N[%3d] SeqId = %3d,
        //     index = %3d, ",
        //         bidx, nidx, qidx, local_warning[widx]);
        //     printf("\tref = %2.4f, \tval = %2.4f, where abs = \t%2.4f /
        //     %2.4f\n",
        //         ref, val, fabs(ref-val), eps);
        // }
      }
    }
  }

  // print
  if (nan_list.size()) printf("[ERR] num nan = %d\n", nan_list.size());
  if (err_list.size()) printf("[ERR] num error = %d\n", err_list.size());
  if (wrn_list.size()) printf("[WARN] num warning = %d\n", wrn_list.size());
  return nan_list.size() == 0 && err_list.size() == 0;
}
#endif  // 1

class MHAPrefillTest : public ::testing::Test {
 public:
#if CUDA_VERSION >= 11080
  template <typename FT>
  void test_flashv2(size_t batch, size_t seqlen, size_t nhead, size_t phead,
                    float alpha = 1.f, bool causal = true, float feps = 1e-3) {
    // tensor map
    allspark::TensorMap tm;
    cudaStream_t stream =
        static_cast<const CUDAContext*>(device.get())->GetStream();

    // param
    int device_id;
    cudaDeviceProp dprop_;
    cudaGetDevice(&device_id);
    cudaGetDeviceProperties(&dprop_, device_id);
    int sm_version = dprop_.major << 8 | dprop_.minor;
    if (sm_version < 0x0800) {
      printf("current device not support flash attention. skip testing.\n");
      return;
    }
    // if (sm_version <= 0x0705 &&
    //     common::toDataType<FT>::dt == allspark::DataType::BFLOAT16) {
    //     printf("current device not support bfloat16. skip testing.\n");
    //     return;
    // }

    allspark::cuda::flashv2_t flash_param;
    allspark::cuda::flashv2_clear_param(flash_param);
    allspark::cuda::flashv2_set_static_param(
        flash_param, dprop_, common::toDataType<FT>::cuda_t, batch, seqlen,
        seqlen, nhead, nhead, phead, cuda::FlashQKVFormat::INTERLEAVED, causal);
    size_t flash_wss = allspark::cuda::flashv2_wss(flash_param);

    allspark::cuda::trivial_t trivial_param;
    trivial_param.dtype = common::toDataType<FT>::dt;
    trivial_param.maxlen = seqlen;
    trivial_param.batch = batch;
    trivial_param.nhead = nhead;
    trivial_param.phead = phead;
    trivial_param.seqlen = seqlen;

    // tensor
    CREATE_TENSOR(tm, input_concat, FT, int64_t(batch), int64_t(seqlen),
                  int64_t(3), int64_t(nhead), int64_t(phead));
    CREATE_TENSOR(tm, output, FT, int64_t(batch), int64_t(seqlen),
                  int64_t(nhead), int64_t(phead));
    CREATE_WORKSPACE(tm, flash_wss);
    device->Synchronize();

    // rand
    std::vector<float> data_concat_float = common::rand_normal_float<float>(
        batch * seqlen * 3 * nhead * phead, 2.5f);
    std::vector<FT> data_concat_cast(batch * seqlen * 3 * nhead * phead, 0.f);
    for (size_t i = 0; i < data_concat_float.size(); i++)
      data_concat_cast[i] = static_cast<FT>(data_concat_float[i]);
    common::AsyncH2D(data_concat_cast.data(), tm.at("input_concat").get(),
                     data_concat_cast.size() * sizeof(FT), stream);

    // run.
    FT* qptr = static_cast<FT*>(tm.at("input_concat")->GetDataPtr());
    FT* kptr = qptr + nhead * phead;
    FT* vptr = kptr + nhead * phead;
    FT* optr = static_cast<FT*>(tm.at("output")->GetDataPtr());
    FT* wptr = static_cast<FT*>(tm.at("workspace")->GetDataPtr());
    device->Synchronize();
    allspark::cuda::flashv2_set_runtime_param(flash_param, qptr, kptr, vptr,
                                              optr, wptr, alpha);
    allspark::cuda::flashv2_dispatch(flash_param, stream);
    std::vector<FT> data_run(batch * seqlen * nhead * phead,
                             static_cast<FT>(0.f));
    device->Synchronize();
    common::AsyncD2H(tm.at("output").get(), data_run.data(),
                     data_run.size() * sizeof(FT), stream);
    device->Synchronize();

    // ref.
    bool check_pass =
        pefill_check_with_reference<FT>(trivial_param, data_concat_float.data(),
                                        data_run.data(), alpha, causal, feps);
    EXPECT_EQ(check_pass, true);

    return;
  }
#endif

  template <typename FT>
  void test_xformer(size_t batch, size_t seqlen, size_t nhead, size_t phead,
                    float alpha = 1.f, bool causal = true, float feps = 1e-3) {
    // tensor map
    allspark::TensorMap tm;
    cudaStream_t stream =
        static_cast<const CUDAContext*>(device.get())->GetStream();

    // param
    int device_id;
    cudaDeviceProp dprop_;
    cudaGetDevice(&device_id);
    cudaGetDeviceProperties(&dprop_, device_id);
    int sm_version = dprop_.major << 8 | dprop_.minor;
    if (sm_version <= 0x0705 &&
        common::toDataType<FT>::dt == allspark::DataType::BFLOAT16) {
      printf("current device not support bfloat16. skip testing.\n");
      return;
    }

#ifdef XFORMER_FMHA
    allspark::cuda::xformer_t xformer_param;
    xformer_param.qkv_format = allspark::cuda::XformerQKVFormat::INTERLEAVED;
    xformer_param.causal = causal;
    xformer_param.batch = batch;
    xformer_param.nhead_kv = nhead;
    xformer_param.nhead = nhead;
    xformer_param.phead = phead;
    xformer_param.seqlen_q = seqlen;
    xformer_param.seqlen_k = seqlen;
    xformer_param.sm_version = sm_version;
    xformer_param.dtype = common::toDataType<FT>::dt;
    size_t xformer_wss =
        allspark::cuda::xformer_prefill_attention_workspace_inbytes(
            xformer_param);
#endif  // XFORMER_FMHA

    allspark::cuda::trivial_t trivial_param;
    trivial_param.dtype = common::toDataType<FT>::dt;
    trivial_param.maxlen = seqlen;
    trivial_param.batch = batch;
    trivial_param.nhead = nhead;
    trivial_param.phead = phead;
    trivial_param.seqlen = seqlen;

    // tensor
    CREATE_TENSOR(tm, input_concat, FT, int64_t(batch), int64_t(seqlen),
                  int64_t(3), int64_t(nhead), int64_t(phead));
    CREATE_TENSOR(tm, output, FT, int64_t(batch), int64_t(seqlen),
                  int64_t(nhead), int64_t(phead));
    CREATE_WORKSPACE(tm, xformer_wss);
    device->Synchronize();

    // rand
    std::vector<float> data_concat_float = common::rand_normal_float<float>(
        batch * seqlen * 3 * nhead * phead, 2.5f);
    std::vector<FT> data_concat_cast(batch * seqlen * 3 * nhead * phead, 0.f);
    for (size_t i = 0; i < data_concat_float.size(); i++)
      data_concat_cast[i] = static_cast<FT>(data_concat_float[i]);
    common::AsyncH2D(data_concat_cast.data(), tm.at("input_concat").get(),
                     data_concat_cast.size() * sizeof(FT), stream);

    // run.
    FT* cptr = static_cast<FT*>(tm.at("input_concat")->GetDataPtr());
    FT* qptr = cptr;
    FT* kptr = qptr + nhead * phead;
    FT* vptr = kptr + nhead * phead;
    FT* optr = static_cast<FT*>(tm.at("output")->GetDataPtr());
    FT* wptr = static_cast<FT*>(tm.at("workspace")->GetDataPtr());
    device->Synchronize();
    allspark::cuda::xformer_prefill_attention(xformer_param, qptr, kptr, vptr,
                                              optr, wptr, alpha, stream);
    std::vector<FT> data_run(batch * seqlen * nhead * phead,
                             static_cast<FT>(0.f));
    device->Synchronize();
    common::AsyncD2H(tm.at("output").get(), data_run.data(),
                     data_run.size() * sizeof(FT), stream);
    device->Synchronize();

    // ref.
    bool check_pass =
        pefill_check_with_reference<FT>(trivial_param, data_concat_float.data(),
                                        data_run.data(), alpha, causal, feps);
    EXPECT_EQ(check_pass, true);

    return;
  }

  template <typename FT>
  void test_trivial(size_t batch, size_t seqlen, size_t maxlen, size_t nhead,
                    size_t phead, float alpha = 1.f, bool causal = true,
                    float feps = 1e-3) {
    // tensor map
    allspark::TensorMap tm;
    cudaStream_t stream =
        static_cast<const CUDAContext*>(device.get())->GetStream();
    cublasHandle_t cublas_handle =
        static_cast<const CUDAContext*>(device.get())->GetCublasHandle();

    int device_id;
    cudaDeviceProp dprop_;
    cudaGetDevice(&device_id);
    cudaGetDeviceProperties(&dprop_, device_id);
    int sm_version = dprop_.major << 8 | dprop_.minor;
    if (sm_version <= 0x0705 &&
        common::toDataType<FT>::dt == allspark::DataType::BFLOAT16) {
      printf("current device not support bfloat16. skip testing.\n");
      return;
    }

    // param
    allspark::cuda::trivial_t trivial_param;
    trivial_param.dtype = common::toDataType<FT>::dt;
    trivial_param.maxlen = seqlen;
    trivial_param.batch = batch;
    trivial_param.nhead = nhead;
    trivial_param.phead = phead;
    trivial_param.seqlen = seqlen;
    size_t tws = trivial_param.workspace_inbytes();

    // tensor
    CREATE_TENSOR(tm, input_concat, FT, int64_t(batch), int64_t(seqlen),
                  int64_t(3), int64_t(nhead), int64_t(phead));
    CREATE_TENSOR(tm, input_kcache, FT, int64_t(batch), int64_t(maxlen),
                  int64_t(nhead), int64_t(phead));
    CREATE_TENSOR(tm, input_vcache, FT, int64_t(batch), int64_t(maxlen),
                  int64_t(nhead), int64_t(phead));
    CREATE_TENSOR(tm, input_mask, float, int64_t(batch), int64_t(seqlen),
                  int64_t(seqlen));
    CREATE_TENSOR(tm, output, FT, int64_t(batch), int64_t(seqlen),
                  int64_t(nhead), int64_t(phead));
    CREATE_WORKSPACE(tm, int64_t(tws));
    device->Synchronize();

    // mask, float 0 for mask 1 for valid.
    std::vector<float> data_mask(batch * seqlen * seqlen, 1.f);
    if (causal) {
      for (size_t bidx = 0; bidx < batch; bidx++) {
        for (size_t qidx = 0; qidx < seqlen; qidx++) {
          for (size_t kidx = 0; kidx < seqlen; kidx++) {
            if (kidx > qidx)
              data_mask[bidx * seqlen * seqlen + qidx * seqlen + kidx] = 0.f;
          }
        }
      }
    }
    common::AsyncH2D(data_mask.data(), tm.at("input_mask").get(),
                     data_mask.size() * sizeof(float), stream);

    // rand
    std::vector<float> data_concat_float = common::rand_normal_float<float>(
        batch * seqlen * 3 * nhead * phead, 2.5f);
    std::vector<FT> data_concat_cast(batch * seqlen * 3 * nhead * phead, 0.f);
    for (size_t i = 0; i < data_concat_float.size(); i++)
      data_concat_cast[i] = static_cast<FT>(data_concat_float[i]);
    common::AsyncH2D(data_concat_cast.data(), tm.at("input_concat").get(),
                     data_concat_cast.size() * sizeof(FT), stream);

    // kvcache
    std::vector<FT> data_kcache_cast(batch * maxlen * nhead * phead, 0.f);
    std::vector<FT> data_vcache_cast(batch * maxlen * nhead * phead, 0.f);
    for (size_t bidx = 0; bidx < batch; bidx++) {
      for (size_t kidx = 0; kidx < maxlen; kidx++) {
        for (size_t nidx = 0; nidx < nhead; nidx++) {
          for (size_t pidx = 0; pidx < phead; pidx++) {
            size_t tidx = bidx * maxlen * nhead * phead + kidx * nhead * phead +
                          nidx * phead + pidx;
            size_t kcid = bidx * seqlen * 3 * nhead * phead +
                          kidx * 3 * nhead * phead + 1 * nhead * phead +
                          nidx * phead + pidx;
            size_t vcid = bidx * seqlen * 3 * nhead * phead +
                          kidx * 3 * nhead * phead + 2 * nhead * phead +
                          nidx * phead + pidx;
            data_kcache_cast[tidx] =
                kidx < seqlen ? data_concat_cast[kcid] : static_cast<FT>(0.f);
            data_vcache_cast[tidx] =
                kidx < seqlen ? data_concat_cast[vcid] : static_cast<FT>(0.f);
          }
        }
      }
    }
    common::AsyncH2D(data_kcache_cast.data(), tm.at("input_kcache").get(),
                     data_kcache_cast.size() * sizeof(FT), stream);
    common::AsyncH2D(data_vcache_cast.data(), tm.at("input_vcache").get(),
                     data_vcache_cast.size() * sizeof(FT), stream);
    device->Synchronize();

    // run.
    FT* cptr = static_cast<FT*>(tm.at("input_concat")->GetDataPtr());
    FT* kptr = static_cast<FT*>(tm.at("input_kcache")->GetDataPtr());
    FT* vptr = static_cast<FT*>(tm.at("input_vcache")->GetDataPtr());
    float* mptr = static_cast<float*>(tm.at("input_mask")->GetDataPtr());
    FT* optr = static_cast<FT*>(tm.at("output")->GetDataPtr());
    FT* wptr = static_cast<FT*>(tm.at("workspace")->GetDataPtr());
    device->Synchronize();
    allspark::cuda::trivial_prefill_attention(trivial_param, cublas_handle,
                                              stream, cptr, mptr, optr, kptr,
                                              vptr, wptr, 1, alpha);
    std::vector<FT> data_run(batch * seqlen * nhead * phead,
                             static_cast<FT>(0.f));
    device->Synchronize();
    common::AsyncD2H(tm.at("output").get(), data_run.data(),
                     data_run.size() * sizeof(FT), stream);
    device->Synchronize();

    // check & ref.
    bool check_pass =
        pefill_check_with_reference<FT>(trivial_param, data_concat_float.data(),
                                        data_run.data(), alpha, causal, feps);
    EXPECT_EQ(check_pass, true);
    return;
  }

 protected:
  void SetUp() override {
    device = allspark::DeviceContextFactory::CreateCUDAContext();
    device->SetDeviceId(0);
    return;
  }
  void TearDown() override {}

 protected:
  std::shared_ptr<allspark::DeviceContext> device;
};  //

using bf16 = hie::bfloat16;
#define TestFlashv2PrefillBasic(DTYPE, BATCH, XSEQL, MSEQL, NHEAD, PHEAD,                   \
                                ALPHA, CAUSAL, EPS)                                         \
  TEST_F(                                                                                   \
      MHAPrefillTest,                                                                       \
      flashv2_##DTYPE##_B##BATCH##_X##XSEQL##_M##MSEQL##_N##NHEAD##_P##PHEAD##_C##CAUSAL) { \
    test_flashv2<DTYPE>(BATCH, XSEQL, NHEAD, PHEAD, ALPHA, CAUSAL, EPS);                    \
  }
#define TestXformerPrefillBasic(DTYPE, BATCH, XSEQL, MSEQL, NHEAD, PHEAD,                   \
                                ALPHA, CAUSAL, EPS)                                         \
  TEST_F(                                                                                   \
      MHAPrefillTest,                                                                       \
      xformer_##DTYPE##_B##BATCH##_X##XSEQL##_M##MSEQL##_N##NHEAD##_P##PHEAD##_C##CAUSAL) { \
    test_xformer<DTYPE>(BATCH, XSEQL, NHEAD, PHEAD, ALPHA, CAUSAL, EPS);                    \
  }
#define TestTrivialPrefillBasic(DTYPE, BATCH, XSEQL, MSEQL, NHEAD, PHEAD,                   \
                                ALPHA, CAUSAL, EPS)                                         \
  TEST_F(                                                                                   \
      MHAPrefillTest,                                                                       \
      trivial_##DTYPE##_B##BATCH##_X##XSEQL##_M##MSEQL##_N##NHEAD##_P##PHEAD##_C##CAUSAL) { \
    test_trivial<DTYPE>(BATCH, XSEQL, MSEQL, NHEAD, PHEAD, ALPHA, CAUSAL,                   \
                        EPS);                                                               \
  }

// clang-format off
//                          DTYPE,  BATCH,  XSEQL,  MSEQL,  NHEAD,  PHEAD,  ALPHA,  CAUSAL, EPS
#if 0
TestTrivialPrefillBasic(    float,   1,     1234,   1234,   3,      20,     1.f,    false,  1e-2);
TestTrivialPrefillBasic(    float,   1,     1234,   1234,   3,      20,     1.f,    true,   1e-2);
TestTrivialPrefillBasic(    float,   1,     12,     12,     64,     20,     1.f,    false,  3e-3);
TestTrivialPrefillBasic(    float,   8,     256,    256,    5,      64,     1.f,    false,  7e-2);
TestTrivialPrefillBasic(    float,   8,     256,    256,    5,      64,     1.f,    true,   9e-2);
TestTrivialPrefillBasic(    float,   3,     345,    345,    1,      256,    1.f,    false,  7e-2);
TestTrivialPrefillBasic(    float,   3,     345,    345,    1,      256,    1.f,    true,   9e-2);
TestTrivialPrefillBasic(    float,   1,     8192,   8192,   1,      16,     1.f,    false,  1e-2);
TestTrivialPrefillBasic(    float,   1,     8192,   8192,   1,      16,     1.f,    true,   1e-2);
#ifdef  ENABLE_FP16
TestTrivialPrefillBasic(    half,    1,     1234,   1234,   3,      20,     1.f,    false,  1e-1);
TestTrivialPrefillBasic(    half,    1,     1234,   1234,   3,      20,     1.f,    true,   1e-1);
TestTrivialPrefillBasic(    half,    1,     12,     12,     64,     20,     1.f,    false,  3e-2);
TestTrivialPrefillBasic(    half,    8,     256,    256,    5,      64,     1.f,    false,  0.2);
TestTrivialPrefillBasic(    half,    8,     256,    256,    5,      64,     1.f,    true,   0.2);
TestTrivialPrefillBasic(    half,    3,     345,    345,    1,      256,    1.f,    false,  0.2);
TestTrivialPrefillBasic(    half,    3,     345,    345,    1,      256,    1.f,    true,   0.2);
TestTrivialPrefillBasic(    half,    1,     8192,   8192,   1,      16,     1.f,    false,  2e-1);
TestTrivialPrefillBasic(    half,    1,     8192,   8192,   1,      16,     1.f,    true,   2e-1);
#endif  // ENABLE_FP16
#ifdef  ENABLE_BF16
TestTrivialPrefillBasic(    bf16,    1,     64,     64,     1,      16,     1.f,    false,  0.2);
/* B[  0]N[  0] SeqId =  41, index =   5, 	ref = -1.4864, 	val = -1.2891, where eps = 	0.1973 / 0.1486 */
TestTrivialPrefillBasic(    bf16,    1,     1234,   1234,   3,      20,     1.f,    false,  0.5);
TestTrivialPrefillBasic(    bf16,    1,     1234,   1234,   3,      20,     1.f,    true,   0.5);
TestTrivialPrefillBasic(    bf16,    1,     12,     12,     64,     20,     1.f,    false,  0.3);
/* TestTrivialPrefillBasic(    bf16,    8,     256,    256,    5,      64,     1.f,    false,  1.0);  // >=1 */
/* TestTrivialPrefillBasic(    bf16,    8,     256,    256,    5,      64,     1.f,    true,   1.0);  // >=1 */
/* TestTrivialPrefillBasic(    bf16,    3,     345,    345,    1,      256,    1.f,    false,  1.0);  // >=1 */
/* TestTrivialPrefillBasic(    bf16,    3,     345,    345,    1,      256,    1.f,    true,   1.0);  // >=1 */
TestTrivialPrefillBasic(    bf16,    1,     8192,   8192,   1,      16,     1.f,    false,  0.5);
/* TestTrivialPrefillBasic(    bf16,    1,     8192,   8192,   1,      16,     1.f,    true,   0.7);  // ??? */
#endif  // ENABLE_BF16
#endif

#if CUDA_VERSION >= 11080
#ifdef ENABLE_FP16
TestFlashv2PrefillBasic(    half,   3,      12,     12,     34,     16,     1.f,    false,  3e-2);
TestFlashv2PrefillBasic(    half,   3,      12,     12,     34,     16,     1.f,    true,   3e-2);
TestFlashv2PrefillBasic(    half,   1,      1234,   1234,   3,      32,     1.f,    false,  5e-2);
TestFlashv2PrefillBasic(    half,   1,      1234,   1234,   3,      32,     1.f,    true,   5e-2);
TestFlashv2PrefillBasic(    half,   2,      345,    345,    5,      128,    1.f,    true,   7e-2);
TestFlashv2PrefillBasic(    half,   1,      8192,   8192,   1,      16,     1.f,    false,  7e-2);
TestFlashv2PrefillBasic(    half,   1,      8192,   8192,   1,      16,     1.f,    true,   7e-2);
#endif  // ENABLE_FP16
#ifdef  ENABLE_BF16
TestFlashv2PrefillBasic(    bf16,   3,      12,     12,     34,     16,     1.f,    false,  0.1);
TestFlashv2PrefillBasic(    bf16,   3,      12,     12,     34,     16,     1.f,    true,   0.1);
TestFlashv2PrefillBasic(    bf16,   1,      1234,   1234,   3,      32,     1.f,    false,  0.3);
TestFlashv2PrefillBasic(    bf16,   1,      1234,   1234,   3,      32,     1.f,    true,   0.3);
/* TestFlashv2PrefillBasic(    bf16,   2,      345,    345,    5,      128,    1.f,    true,   0.7);   // ??? */
TestFlashv2PrefillBasic(    bf16,   1,      8192,   8192,   1,      16,     1.f,    false,  0.3);
TestFlashv2PrefillBasic(    bf16,   1,      8192,   8192,   1,      16,     1.f,    true,   0.3);
#endif  // ENABLE_BF16
#endif

TestXformerPrefillBasic(    float,   1,     1234,   1234,   3,      20,     1.f,    false,  1e-3);
TestXformerPrefillBasic(    float,   1,     1234,   1234,   3,      20,     1.f,    true,   1e-3);
TestXformerPrefillBasic(    float,   1,     12,     12,     64,     20,     1.f,    false,  1e-3);
TestXformerPrefillBasic(    float,   8,     256,    256,    5,      64,     1.f,    false,  1e-3);
TestXformerPrefillBasic(    float,   8,     256,    256,    5,      64,     1.f,    true,   1e-3);
TestXformerPrefillBasic(    float,   3,     345,    345,    1,      256,    1.f,    false,  1e-3);
TestXformerPrefillBasic(    float,   3,     345,    345,    1,      256,    1.f,    true,   1e-3);
TestXformerPrefillBasic(    float,   1,     8192,   8192,   1,      16,     1.f,    false,  1e-3);
TestXformerPrefillBasic(    float,   1,     8192,   8192,   1,      16,     1.f,    true,   1e-3);
#ifdef ENABLE_FP16
TestXformerPrefillBasic(    half,   3,      12,     12,     34,     16,     1.f,    false,  3e-2);
TestXformerPrefillBasic(    half,   3,      12,     12,     34,     16,     1.f,    true,   3e-2);
TestXformerPrefillBasic(    half,   1,      1234,   1234,   3,      32,     1.f,    false,  7e-2);
TestXformerPrefillBasic(    half,   1,      1234,   1234,   3,      32,     1.f,    true,   7e-2);
TestXformerPrefillBasic(    half,   2,      345,    345,    5,      128,    1.f,    true,   8e-2);
TestXformerPrefillBasic(    half,   1,      8192,   8192,   1,      16,     1.f,    false,  7e-2);
TestXformerPrefillBasic(    half,   1,      8192,   8192,   1,      16,     1.f,    true,   7e-2);
#endif  // ENABLE_FP16
#ifdef  ENABLE_BF16
TestXformerPrefillBasic(    bf16,   3,      12,     12,     34,     16,     1.f,    false,  0.1);
TestXformerPrefillBasic(    bf16,   3,      12,     12,     34,     16,     1.f,    true,   0.1);
TestXformerPrefillBasic(    bf16,   1,      1234,   1234,   3,      32,     1.f,    false,  0.3);
TestXformerPrefillBasic(    bf16,   1,      1234,   1234,   3,      32,     1.f,    true,   0.3);
/* TestXformerPrefillBasic(    bf16,   2,      345,    345,    5,      128,    1.f,    true,   0.7);   // ??? */
TestXformerPrefillBasic(    bf16,   1,      8192,   8192,   1,      16,     1.f,    false,  0.3);
TestXformerPrefillBasic(    bf16,   1,      8192,   8192,   1,      16,     1.f,    true,   0.3);
#endif  // ENABLE_BF16

#undef TestFlashv2PrefillBasic
#undef TestXformerPrefillBasic
#undef TestTrivialPrefillBasic
// clang-format on
