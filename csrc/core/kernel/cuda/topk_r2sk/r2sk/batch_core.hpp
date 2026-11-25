// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "core.hpp"

namespace r2sk {
struct BatchStats {
  std::vector<Stats> tasks;
  std::vector<Stage> stages;  // whole batch phases, not sums of task timings
  uint32_t decision_rounds = 0;
};
template <class Value>
struct HistJobT {
  uint32_t task;
  ValueView<Value> src;
  uint32_t n;
  int pass;
  uint32_t rank;
};
template <class Value>
struct CopyJobT {
  uint32_t task;
  ValueView<Value> src;
  uint32_t n;
  ValueBuffer<Value> dst;
};
template <class Value>
struct PartitionJobT {
  uint32_t task;
  ValueView<Value> src;
  uint32_t n;
  int pass;
  uint32_t bin;
  ValueBuffer<Value> G;
  uint32_t gcap;
  ValueBuffer<Value> E;
  uint32_t ecap;
  uint32_t expected_g, expected_e;
};
template <class Value>
struct FilterJobT {
  uint32_t task;
  ValueView<Value> src;
  uint32_t n, pivot;
  ValueView<Value>
      pivot_value;  // nonnull: singleton gathered on device, no scalar D2H
  uint32_t greater, ties;
  ValueBuffer<Value> dst;
};
template <class Value>
struct SortJobT {
  uint32_t task;
  ValueBuffer<Value> src;
  uint32_t n;
  ValueBuffer<Value> out;
  uint32_t k;
};
using HistJob = HistJobT<float>;
using CopyJob = CopyJobT<float>;
using PartitionJob = PartitionJobT<float>;
using FilterJob = FilterJobT<float>;
using SortJob = SortJobT<float>;
struct WinnerJob {
  uint32_t task, base, n;
};
inline std::vector<uint64_t> batch_offsets(const std::vector<uint32_t>& lengths,
                                           uint32_t k, Options o) {
  if (lengths.empty() || lengths.size() > 65535)
    throw std::invalid_argument("batch size must be in [1,65535]");
  std::vector<uint64_t> offsets(lengths.size() + 1, 0);
  for (size_t t = 0; t < lengths.size(); ++t) {
    validate(lengths[t], k, o);
    offsets[t + 1] = offsets[t] + lengths[t];
  }
  return offsets;
}
// Independent states, shared rounds. Backend methods consume whole job lists;
// the CUDA backend launches grid.y over jobs, never calls topk() per row.
template <class Backend, class Value = float>
BatchStats run_batch(Backend& b, ValueView<Value> input,
                     const std::vector<uint32_t>& lengths, uint32_t k,
                     ValueBuffer<Value> output, Options o) {
  using View = ValueView<Value>;
  using Buffer = ValueBuffer<Value>;
  using HistJob = HistJobT<Value>;
  using CopyJob = CopyJobT<Value>;
  using PartitionJob = PartitionJobT<Value>;
  using FilterJob = FilterJobT<Value>;
  using SortJob = SortJobT<Value>;
  if (o.capture_rank || o.capture_capacity)
    throw std::invalid_argument(
        "local capture supports single-array calls only");
  auto offsets = batch_offsets(lengths, k, o);
  BatchStats result;
  result.tasks.resize(lengths.size());
  b.begin(o, &result);
  struct State {
    View original, C;
    Buffer output;
    uint32_t n, count, w = 0, r, better = 0, prefix = 0;
    int next = 0;
    bool done = false;
  };
  std::vector<State> states;
  for (uint32_t t = 0; t < lengths.size(); ++t) {
    View row{input.v + offsets[t], input.i ? input.i + offsets[t] : nullptr};
    row.shift_reference = o.scale == ScaleMode::shift ? row.v : nullptr;
    states.push_back({row, row, offset(output, uint64_t(t) * k), lengths[t],
                      lengths[t], 0, k, 0, 0, 0, false});
  }
  std::vector<CopyJob> copies;
  std::vector<SortJob> sorts;
  std::vector<CopyJob> prefixes;
  const bool sorted_runs = o.winner_sort != WinnerSort::deferred;
  auto finish = [&](uint32_t t, uint32_t m, uint32_t prefix = 0) {
    result.tasks[t].final_sorted = m;
    result.tasks[t].estimated_value_bytes += (2ull * sizeof(Value)) * k;
    result.tasks[t].sorted_values += m;
    sorts.push_back(
        {t, b.potential(t), m, offset(states[t].output, prefix), k - prefix});
    if (prefix) prefixes.push_back({t, b.winners(t), prefix, states[t].output});
    states[t].done = true;
  };
  const int passes = o.scale == ScaleMode::shift ? 6 : 3;
  for (int pass = 0; pass < passes; ++pass) {
    const bool shifted = o.scale == ScaleMode::shift && pass < 3;
    std::vector<HistJob> hist;
    std::vector<CopyJob> entry_winners;
    for (uint32_t t = 0; t < states.size(); ++t) {
      auto& x = states[t];
      if (x.done) continue;
      if (pass == 3) x.prefix = 0;
      if (!x.r || x.r > x.count ||
          (o.method != Method::reference && x.w + x.r != k))
        throw std::logic_error("batch entry invariant");
      if (o.method == Method::early && o.entry_sort &&
          uint64_t(x.w) + x.count <= o.sort_limit) {
        uint32_t prefix = sorted_runs ? x.w : 0;
        if (!sorted_runs && x.w)
          entry_winners.push_back({t, b.winners(t), x.w, b.potential(t)});
        copies.push_back(
            {t, x.C, x.count, offset(b.potential(t), x.w - prefix)});
        result.tasks[t].estimated_value_bytes +=
            (2ull * sizeof(Value)) * (x.w - prefix + x.count);
        result.tasks[t].trace.push_back({x.count, x.w, x.r, pass, -1, 0,
                                         x.count, x.w + x.count, "entry_sort"});
        finish(t, x.w - prefix + x.count, prefix);
        continue;
      }
      hist.push_back({t, x.C, x.count, pass, x.r});
    }
    b.copies(entry_winners);
    b.copies(copies);
    b.sorts(sorts);
    copies.clear();
    sorts.clear();
    if (hist.empty()) break;
    auto decisions = b.histograms(hist);
    if (decisions.size() != hist.size())
      throw std::logic_error("batch metadata size");
    ++result.decision_rounds;
    std::vector<PartitionJob> partitions;
    std::vector<FilterJob> filters;
    std::vector<WinnerJob> winner_jobs;
    for (size_t j = 0; j < hist.size(); ++j) {
      uint32_t t = hist[j].task;
      auto& x = states[t];
      auto& s = result.tasks[t];
      auto d = decisions[j];
      if (d.total != x.count || d.bin < 0 || !d.B || d.h >= x.r ||
          x.r > d.h + d.B)
        throw std::logic_error("batch histogram invariant");
      ++s.histogram_passes;
      s.estimated_value_bytes += uint64_t(sizeof(Value)) * x.count;
      if (pass >= 3) ++s.raw_refinement_passes;
      x.prefix |= uint32_t(d.bin) << digit_low(pass);
      uint32_t M = x.w + d.h + d.B;
      Pass trace{x.count, x.w, x.r, pass, d.bin, d.h, d.B, M, "refine"};
      trace.gather_count = x.count;
      trace.shifted = shifted;
      trace.scale_unsafe = d.scale_unsafe != 0;
      const bool exhausted = pass % 3 == 2 && (!shifted || !d.scale_unsafe);
      if (shifted && pass == 2 && d.scale_unsafe) trace.route = "raw_refine";
      auto partition = [&](Buffer G, uint32_t gc, Buffer E, uint32_t ec) {
        partitions.push_back(
            {t, x.C, x.count, pass, uint32_t(d.bin), G, gc, E, ec, d.h, d.B});
        s.estimated_value_bytes +=
            uint64_t(sizeof(Value)) * x.count +
            uint64_t(sizeof(Value)) * ((G.v ? gc : 0) + ec);
      };
      if (o.method == Method::reference) {
        x.better += d.h;
        x.r -= d.h;
        trace.potential = 0;
        if (exhausted || d.B == 1) {
          View pivot{};
          if (shifted || (d.B == 1 && pass % 3 != 2)) {
            partition({}, 0, b.candidate(t, x.next), 1);
            pivot = b.candidate(t, x.next);
          }
          filters.push_back({t, x.original, x.n, x.prefix, pivot, x.better, x.r,
                             b.potential(t)});
          s.estimated_value_bytes +=
              uint64_t(sizeof(Value)) * x.n + uint64_t(sizeof(Value)) * k;
          trace.route = "pivot_filter";
          finish(t, k);
        } else {
          partition({}, 0, b.candidate(t, x.next), d.B);
          x.C = b.candidate(t, x.next);
          x.C.shift_reference = x.original.shift_reference;
          x.count = d.B;
          x.next ^= 1;
        }
      } else {
        bool exact = d.h + d.B == x.r,
             small = o.method == Method::early && M <= o.sort_limit;
        if (exact || small || exhausted) {
          uint32_t ec = exhausted && !exact && !small ? x.r - d.h : d.B;
          uint32_t prefix = sorted_runs ? x.w : 0;
          if (!sorted_runs && x.w) {
            copies.push_back({t, b.winners(t), x.w, b.potential(t)});
            s.estimated_value_bytes += (2ull * sizeof(Value)) * x.w;
          }
          partition(offset(b.potential(t), x.w - prefix), d.h,
                    offset(b.potential(t), x.w - prefix + d.h), ec);
          trace.route = exact ? "exact" : small ? "early_sort" : "ties";
          finish(t, x.w - prefix + d.h + ec, prefix);
        } else {
          partition(offset(b.winners(t), x.w), d.h, b.candidate(t, x.next),
                    d.B);
          if (sorted_runs && d.h) {
            winner_jobs.push_back({t, x.w, d.h});
            s.winner_runs.push_back(d.h);
            s.sorted_values += d.h;
          }
          x.w += d.h;
          x.r -= d.h;
          x.C = b.candidate(t, x.next);
          x.C.shift_reference = x.original.shift_reference;
          x.count = d.B;
          x.next ^= 1;
        }
      }
      s.trace.push_back(trace);
    }
    b.copies(copies);
    b.partitions(partitions);
    b.winner_sorts(winner_jobs);
    b.filters(filters);
    b.sorts(sorts);
    copies.clear();
    sorts.clear();
    if (o.debug) {
      b.join_winners();
      for (uint32_t t = 0; t < states.size(); ++t) {
        auto& x = states[t];
        if (!x.done)
          b.audit(t, x.original, x.n, x.C, x.count, x.w, x.r, k, x.better,
                  o.method);
      }
    }
  }
  for (auto& x : states)
    if (!x.done) throw std::logic_error("batch failed to terminate");
  // Finished rows may wait for other rows without blocking their refinement.
  // All winner-prefix copies occur after the final cross-stream dependency.
  b.join_winners();
  b.copies(prefixes);
  return result;
}
}  // namespace r2sk
