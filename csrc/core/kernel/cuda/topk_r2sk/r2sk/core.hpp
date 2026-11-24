// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace r2sk {
// Finite FP32 only. Zeros may be ordered by their key; they are numeric ties.
#ifdef __CUDACC__
#define R2SK_HD __host__ __device__
#else
#define R2SK_HD
#endif
R2SK_HD inline uint32_t float_bits(float value) {
  uint32_t bits;
#ifdef __CUDA_ARCH__
  bits = __float_as_uint(value);
#else
  std::memcpy(&bits, &value, sizeof(bits));
#endif
  return bits;
}
R2SK_HD inline uint32_t ordered_key(float value) {
  uint32_t bits = float_bits(value);
  return bits ^ ((0u - (bits >> 31)) | 0x80000000u);
}
R2SK_HD inline bool finite_key(float value) {
  return (float_bits(value) & 0x7f800000u) != 0x7f800000u;
}
struct NonFiniteInput {};
R2SK_HD inline float shift_value(float value, float reference) {
#ifdef __CUDA_ARCH__
  return __fsub_rn(value, reference);
#else
  volatile float result = value - reference;
  return result;
#endif
}
// A bit-exact round trip is a sufficient injectivity certificate. Checking
// only the current candidates at the last shifted digit is conservative.
R2SK_HD inline bool shift_reversible(float value, float shifted,
                                     float reference) {
#ifdef __CUDA_ARCH__
  float recovered = __fadd_rn(shifted, reference);
#else
  volatile float recovered = shifted + reference;
#endif
  return float_bits(recovered) == float_bits(value);
}
R2SK_HD inline int digit_low(int pass) {
  return pass % 3 == 0 ? 20 : pass % 3 == 1 ? 8 : 0;
}
R2SK_HD inline int digit_width(int pass) { return pass % 3 == 2 ? 8 : 12; }
R2SK_HD inline uint32_t digit(float x, int pass, float reference = 0.f,
                              bool shifted = false) {
  if (shifted && pass < 3) x = shift_value(x, reference);
  return (ordered_key(x) >> digit_low(pass)) & ((1u << digit_width(pass)) - 1u);
}
inline uint32_t padded(uint32_t n) {
  uint32_t p = 1;
  while (p < n) p <<= 1;
  return p;
}
enum class Method { reference, winners, early };
enum class WinnerSort { deferred, serial, overlap };
enum class ScaleMode { off, shift };
inline const char* name(ScaleMode mode) {
  return mode == ScaleMode::off ? "off" : "shift";
}
// Cumulative primitive ablations. The device API keeps its original default;
// benchmark runners explicitly select radik and compare all winner modes.
enum class PrimitiveMode { scalar, packed, buffered, scan, radik };
inline const char* name(PrimitiveMode mode) {
  switch (mode) {
    case PrimitiveMode::scalar:
      return "scalar";
    case PrimitiveMode::packed:
      return "packed";
    case PrimitiveMode::buffered:
      return "buffered";
    case PrimitiveMode::scan:
      return "scan";
    case PrimitiveMode::radik:
      return "radik";
  }
  return "invalid";
}
inline const char* name(WinnerSort mode) {
  return mode == WinnerSort::deferred ? "deferred"
         : mode == WinnerSort::serial ? "serial"
                                      : "overlap";
}
inline const char* name(Method m) {
  return m == Method::reference ? "our_reference_filter"
         : m == Method::winners ? "r2sk_winners"
                                : "r2sk_early";
}
struct Options {
  bool reject_nonfinite = false;  // batch-only guard, before any output writes
  Method method = Method::early;
  uint32_t sort_limit =
      1024;  // actual superset M, padded to a supported capacity
  uint32_t block_cap = 1280;
  bool entry_sort = true;
  bool debug = false;
  bool instrument = false;
  WinnerSort winner_sort =
      WinnerSort::deferred;  // saved-group sorting; optional workspace
                             // resources required
  uint32_t capture_rank =
      0;  // 0: off; otherwise retain bins >= each 256-value tile's q-th digit
  uint32_t capture_capacity = 0;  // 0: current candidate count; smaller values
                                  // exercise overflow fallback
  PrimitiveMode primitives = PrimitiveMode::scalar;
  ScaleMode scale =
      ScaleMode::off;  // key-only x-input[0]; raw floats stay in all buffers
};
inline void validate(uint32_t n, uint32_t k, const Options& o) {
  if (!n || !k || k > n || n > 0x7fffffffu)
    throw std::invalid_argument("require 1 <= k <= N <= INT_MAX");
  if (!o.block_cap || o.block_cap > 65535)
    throw std::invalid_argument("block_cap must be in [1,65535]");
  if (!o.sort_limit || o.sort_limit > 4096 ||
      padded(o.sort_limit) != o.sort_limit)
    throw std::invalid_argument(
        "sort_limit must be a power of two in [1,4096]");
  if (o.method == Method::reference && o.winner_sort != WinnerSort::deferred)
    throw std::invalid_argument(
        "winner-run sorting requires a winner-preserving method");
  if (o.capture_rank > 256 || (o.method == Method::reference && o.capture_rank))
    throw std::invalid_argument(
        "capture requires a winner-preserving method and rank in [1,256]");
  if (!o.capture_rank && o.capture_capacity)
    throw std::invalid_argument(
        "capture capacity requires capture to be enabled");
  if (o.primitives < PrimitiveMode::scalar ||
      o.primitives > PrimitiveMode::radik)
    throw std::invalid_argument("unknown primitive mode");
  if (o.scale != ScaleMode::off && o.scale != ScaleMode::shift)
    throw std::invalid_argument("unknown scale mode");
}
template <class Value>
struct ValueView {
  const Value* v = nullptr;
  const uint32_t* i = nullptr;  // null means implicit original positions
  const Value* shift_reference =
      nullptr;  // original input[0], never a compacted C[0]
};
template <class Value>
struct ValueBuffer {
  Value* v = nullptr;
  uint32_t* i = nullptr;
  operator ValueView<Value>() const { return {v, i}; }
};
using View = ValueView<float>;
using Buffer = ValueBuffer<float>;
template <class Value>
inline ValueBuffer<Value> offset(ValueBuffer<Value> b, uint64_t n) {
  return {b.v + n, b.i ? b.i + n : nullptr};
}
struct Decision {
  int bin = -1;
  uint32_t h = 0, B = 0, total = 0;
  uint32_t captured = 0,
           captured_retained = 0;  // logical counts include overflowed stores
  uint32_t scale_unsafe =
      0;  // final shifted histogram: some C failed the bit-exact round trip
};
struct Pass {
  uint32_t candidates, winners, residual;
  int digit, bin;
  uint32_t h, boundary, potential;
  std::string route;
  uint32_t captured = 0, captured_retained = 0, capture_capacity = 0,
           gather_count = 0;
  bool capture_attempted = false, capture_success = false,
       capture_overflow = false;
  bool shifted = false, scale_unsafe = false;
};
struct Stage {
  std::string name;
  float ms;
};
struct Stats {
  std::vector<Pass> trace;
  std::vector<Stage> stages;
  uint32_t final_sorted = 0;
  uint32_t histogram_passes = 0;
  uint64_t estimated_value_bytes =
      0;  // analytical, excludes histogram, indices, sort
  uint64_t sorted_values =
      0;  // sum of sort input lengths, including trivial one-value runs
  std::vector<uint32_t>
      winner_runs;  // nonempty G groups sorted before the final tail
  uint32_t capture_attempts = 0, capture_hits = 0, capture_fallbacks = 0,
           capture_overflows = 0;
  uint64_t captured_values = 0, capture_gather_values = 0;
  uint32_t raw_refinement_passes = 0;
};

// Backend owns allocation, primitives, and optional audits. This controller is
// used unchanged by the CUDA implementation and executable CPU state model.
template <class Backend>
Stats run(Backend& b, View original, uint32_t n, uint32_t k, Buffer out,
          Options o) {
  validate(n, k, o);
  Stats s;
  b.begin(o, &s);
  original.shift_reference = o.scale == ScaleMode::shift ? original.v : nullptr;
  View C = original;  // no initializing N-element copy
  uint32_t count = n, w = 0, r = k, prior_better = 0, prefix = 0;
  int next = 0;
  const bool sorted_runs = o.winner_sort != WinnerSort::deferred;
  auto finish = [&](uint32_t m, uint32_t prefix = 0) {
    s.final_sorted = m;
    s.sorted_values += m;
    b.sort(b.potential(), m, offset(out, prefix), k - prefix);
    if (prefix) {
      b.join_winners();
      b.copy(b.winners(), prefix, out);
    }
    s.estimated_value_bytes +=
        8ull * k;  // final output extraction, excludes sorting
  };
  const int passes = o.scale == ScaleMode::shift ? 6 : 3;
  for (int pass = 0; pass < passes; ++pass) {
    const bool shifted = o.scale == ScaleMode::shift && pass < 3;
    if (pass == 3) prefix = 0;  // collision refinement uses ORIGINAL key bits
    if (!(r > 0 && r <= count) || (o.method != Method::reference && w + r != k))
      throw std::logic_error("entry rank invariant");
    if (o.method == Method::early && o.entry_sort &&
        uint64_t(w) + count <= o.sort_limit) {
      uint32_t prefix = sorted_runs ? w : 0;
      if (!sorted_runs) b.copy(b.winners(), w, b.potential());
      b.copy(C, count, offset(b.potential(), w - prefix));
      s.estimated_value_bytes += 8ull * (w - prefix + count);
      s.trace.push_back(
          {count, w, r, pass, -1, 0, count, w + count, "entry_sort"});
      finish(w - prefix + count, prefix);
      return s;
    }
    uint32_t capture_cap =
        o.capture_capacity ? std::min(count, o.capture_capacity) : count;
    Decision d = o.capture_rank
                     ? b.capture_histogram(C, count, pass, r, b.candidate(next),
                                           capture_cap, o.capture_rank)
                     : b.histogram(C, count, pass, r);
    ++s.histogram_passes;
    if (pass >= 3) ++s.raw_refinement_passes;
    s.estimated_value_bytes += 4ull * count;
    if (d.total != count || d.bin < 0 || !d.B || d.h >= r || r > d.h + d.B)
      throw std::logic_error("histogram/rank invariant");
    uint32_t M = w + d.h + d.B;
    prefix |= uint32_t(d.bin) << digit_low(pass);
    Pass record{count, w, r, pass, d.bin, d.h, d.B, M, "refine"};
    record.shifted = shifted;
    record.scale_unsafe = d.scale_unsafe != 0;
    const bool exhausted = pass % 3 == 2 && (!shifted || !d.scale_unsafe);
    if (shifted && pass == 2 && d.scale_unsafe) record.route = "raw_refine";
    View gather_src = C;
    uint32_t gather_count = count;
    int destination = next;
    record.gather_count = count;
    if (o.capture_rank) {
      if (d.captured > count || d.captured_retained > d.h + d.B ||
          d.captured_retained > d.captured)
        throw std::logic_error("capture population invariant");
      record.capture_attempted = true;
      record.captured = d.captured;
      record.captured_retained = d.captured_retained;
      record.capture_capacity = capture_cap;
      record.capture_overflow = d.captured > capture_cap;
      record.capture_success =
          !record.capture_overflow && d.captured_retained == d.h + d.B;
      ++s.capture_attempts;
      s.captured_values += d.captured;
      s.estimated_value_bytes += 4ull * std::min(d.captured, capture_cap);
      if (record.capture_success) {
        ++s.capture_hits;
        gather_src = b.candidate(next);
        gather_count = d.captured;
        gather_src.shift_reference = original.shift_reference;
        // The captured source occupies next. The old candidate buffer
        // can become the destination only AFTER coverage succeeds.
        destination = next ^ 1;
        s.capture_gather_values += gather_count;
      } else {
        ++s.capture_fallbacks;
        if (record.capture_overflow) ++s.capture_overflows;
      }
      record.gather_count = gather_count;
    }
    if (o.method == Method::reference) {
      prior_better += d.h;
      r -= d.h;
      record.potential = 0;  // W is not preserved in the reference
      if (exhausted || d.B == 1) {
        uint32_t pivot = prefix;
        if (shifted || (d.B == 1 && pass % 3 != 2)) {
          b.partition(C, count, pass, d.bin, {}, 0, b.candidate(next), 1, d.h,
                      d.B);
          s.estimated_value_bytes += 4ull * count + 4ull;
          pivot = b.first_key(b.candidate(next));
        }
        b.filter(original, n, pivot, prior_better, r, b.potential());
        s.estimated_value_bytes += 4ull * n + 4ull * k;
        record.route = "pivot_filter";
        s.trace.push_back(record);
        finish(k);
        return s;
      }
      b.partition(C, count, pass, d.bin, {}, 0, b.candidate(next), d.B, d.h,
                  d.B);
      s.estimated_value_bytes += 4ull * count + 4ull * d.B;
      C = b.candidate(next);
      C.shift_reference = original.shift_reference;
      count = d.B;
      next ^= 1;
    } else {
      const bool exact = d.h + d.B == r;
      const bool small = o.method == Method::early && M <= o.sort_limit;
      if (exact || small || exhausted) {
        // Last digit: all E keys are identical. Keep exactly r-h
        // occurrences without sorting or materializing a huge E.
        uint32_t keep_e = exhausted && !small && !exact ? r - d.h : d.B;
        uint32_t prefix = sorted_runs ? w : 0;
        uint32_t m = w - prefix + d.h + keep_e;
        if (!sorted_runs) b.copy(b.winners(), w, b.potential());
        b.partition(gather_src, gather_count, pass, d.bin,
                    offset(b.potential(), w - prefix), d.h,
                    offset(b.potential(), w - prefix + d.h), keep_e, d.h, d.B);
        s.estimated_value_bytes +=
            4ull * gather_count + 4ull * (d.h + keep_e + 2ull * (w - prefix));
        record.route = exact ? "exact" : small ? "early_sort" : "ties";
        s.trace.push_back(record);
        finish(m, prefix);
        return s;
      }
      b.partition(gather_src, gather_count, pass, d.bin, offset(b.winners(), w),
                  d.h, b.candidate(destination), d.B, d.h, d.B);
      if (sorted_runs && d.h) {
        b.sort_winner_run(w, d.h);
        s.winner_runs.push_back(d.h);
        s.sorted_values += d.h;
      }
      s.estimated_value_bytes += 4ull * gather_count + 4ull * (d.h + d.B);
      w += d.h;
      r -= d.h;
      C = b.candidate(destination);
      C.shift_reference = original.shift_reference;
      count = d.B;
      next = destination ^ 1;
    }
    s.trace.push_back(record);
    if (o.debug) {
      if (sorted_runs)
        b.join_winners();  // validation may read W; excluded from timing
      b.audit(original, n, C, count, w, r, k, prior_better, o.method);
    }
  }
  throw std::logic_error("unreachable: all configured key digits exhausted");
}
}  // namespace r2sk
