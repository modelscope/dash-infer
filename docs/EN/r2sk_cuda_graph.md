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
native sampler capture the selector, sorted top-p/softmax, and token draw
together. Prefill, log-probability requests, the Torch-compatible sampler,
and unsupported ranks retain their existing dispatch. A GPU finite-input
check uses the existing pre-sampling host synchronization and preserves
the fallback for nonfinite logits.

GenerateOp still updates the device logits, top-p parameters and RNG
pointer table between launches. Cached graphs are keyed by the active
shape and addresses. Each graph reads the current buffers on replay. The
initial capture, request bookkeeping and generated-token copy stay outside
the graph. This captures the decode sampling GPU stage within the engine.
It is separate from whole-decode graph capture including the model forward
pass and host scheduler.

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
passes. The benchmark compares RadiK eager, the PR selector eager, the
device controller eager, and graph replay with and without winner overlap.
