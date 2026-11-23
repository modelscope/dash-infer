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

This API is experimental. Enabling the existing DashInfer feature still
selects the PR's host-controlled backend. Capturing the full sampling call
also requires resolving the synchronization in GenerateOp and defining
how the engine handles invalid graph rows.

With a build configured for CUDA and `ENABLE_R2SK_SAMPLING=ON`:

```sh
ctest --test-dir build -R '^cpp_r2sk_graph_test$' --output-on-failure
compute-sanitizer --tool memcheck --error-exitcode 1 build/bin/cpp_r2sk_sampling_test --graph-test
build/bin/cpp_r2sk_sampling_test --graph-replay /path/to/capture.bin 1 151936 20
```

The test replays different data through one instantiated graph for each
shape and schedule, including concurrent rows that finish on different
passes. The benchmark compares RadiK eager, the PR selector eager, the
device controller eager, and graph replay with and without winner overlap.
