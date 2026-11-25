<!-- SPDX-License-Identifier: Apache-2.0 -->

# BF16 R2S-K CUDA Graph experiment

This branch adds a selector API for a fixed shape and stable device addresses.
It uses the same packed histogram and partition primitives as the ordinary
BF16 path. All route decisions and descriptors stay in device memory and
are rebuilt for every invocation. The caller creates the optional split
stream and its dependency events before capture.

Pass one consumes the high 12 bits of the ordered BF16 key. If the retained
superset fits the sort limit, a CTA sorts it and writes exactly k outputs.
Otherwise, a packed gather writes confirmed winners to the output prefix and
the boundary candidates to a single scratch buffer. The winner stream sorts
the prefix while the main stream refines the candidates using the last four
BF16 key bits. A final CTA sorts just the selected tail and writes it after
the prefix. The caller stream joins the winner sort before using the output
or recycling scratch. Completed rows skip the second pass using their
device descriptors. The single-stream control uses the same schedule.

The selector handles finite BF16, signed zeros and infinities. It preserves
raw values and row-relative occurrence indices. NaN rows leave output
untouched and expose an invalid flag through the optional device report.
Callers omitting the report must ensure there are no NaNs.

## Experimental decode integration

Set `AS_ENABLE_R2SK_SAMPLING=1` and `AS_ENABLE_R2SK_DECODE_GRAPH=1` on
an R2S-K-enabled build. Eligible single-GPU BF16 decode steps using the
native sampler and `2 <= max_k <= 256` capture the selector, sorted
top-p/softmax, and token draw together. Prefill, log-probability requests,
the Torch-compatible sampler, and unsupported ranks retain existing dispatch.
Before capture or replay, `Prepare()` checks the logits on the GPU and
copies a nonfinite flag to the host. The existing pre-sampling host
synchronization completes that readback.

GenerateOp still updates the device logits, top-p parameters and RNG
pointer table between launches. Cached graphs are keyed by the active
shape and addresses. Each graph reads the current buffers on replay. The
request bookkeeping and generated-token copy stay outside the graph. The
integration owns this sampler graph. Both `Prepare()` and eager `Try()`
decline when the caller is already capturing, so whole-decode capture cannot
dispatch R2S-K through these entry points. The model forward pass and host
scheduler also remain outside the sampler graph.
HIE-DNN uses external CUDA event nodes for the prefix-sum workspace dependency
so eager calls and graph replays can share the same handle.

`AS_TRACE_R2SK_DECODE_GRAPH=1` reports graph creation and cache hits. The
decode graph test compares the graph with the eager RadiK plus native
sampling pipeline using matched RNG state and changing logits, temperatures
and top-p cutoffs.

With a build configured for CUDA and `ENABLE_R2SK_SAMPLING=ON`:

```sh
ctest --test-dir build -R '^cpp_r2sk_graph_test$' --output-on-failure
ctest --test-dir build -R '^cpp_r2sk_decode_graph_test$' --output-on-failure
compute-sanitizer --tool memcheck --error-exitcode 1 build/bin/cpp_r2sk_sampling_test --graph-test
build/bin/cpp_r2sk_sampling_test --graph-replay /path/to/capture.bin 1 151936 20
```

The test replays different data through one instantiated graph for each
shape and schedule, including concurrent rows that finish on different
passes. The benchmark compares eager RadiK, the engine's device-controlled
eager wrapper, a bare device selector benchmark, and R2S-K graph replay
with and without winner overlap. The engine's eager wrapper handles ranks
through 256. See the [historical results and limits](r2sk_review_validation.md).
