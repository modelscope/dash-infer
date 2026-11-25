<!-- SPDX-License-Identifier: Apache-2.0 -->

# Experimental R2S-K token sampling

R2S-K refines a radix boundary while saving confirmed winners. When the
complete set of possible results fits a block sort, it sorts the set and
returns the first k entries. The backend produces descending BF16 values and
row-relative 32-bit token indices for DashInfer's existing top-p sampler.
Equal values may have different token order or membership at the cutoff.

## Enable it

On a Linux CUDA development machine, follow the [source build setup](installation.md#build-from-source).
Set `ENABLE_R2SK_SAMPLING=ON` in CMake, or build with the following environment.
Set `AS_CUDA_VERSION` and `AS_CUDA_SM` to match your toolkit and GPU. This example
targets an L40S with CUDA 12.4.

```sh
AS_BUILD_R2SK_SAMPLING=ON AS_CUDA_VERSION=12.4 AS_CUDA_SM=89 bash build.sh
```

Set `AS_ENABLE_R2SK_SAMPLING=1` before starting the source-built engine and
loading a BF16 model. Leave it unset for the RadiK default in the same build.
The eager backend applies to token sampling with compute capability 8.0 or
newer, finite BF16 logits, and `2 <= max_k <= 256` where
`max_k < vocabulary`. The bound applies to the largest requested k in
the active batch. The existing launcher uses RadiK when eager R2S-K
declines the call.
The first histogram checks for NaN and infinities and sends that batch to
RadiK before writing any R2S-K output. Greedy, full-vocabulary and non-BF16
requests retain the existing dispatch.

The operator borrows the shared top-k/top-p arena and reserves space for
its configured maximum batch. The eager path runs the BF16 selector's two
radix passes with per-row device decisions. Its first histogram checks
input finiteness, and the flag is read during the engine's existing
pre-sampling synchronization. A nonfinite flag leaves the output untouched
for RadiK fallback. A pinned flag, winner stream and events are allocated
on first use. The caller stream joins the winner stream before reusing scratch.

Set `AS_ENABLE_R2SK_DECODE_GRAPH=1` in addition to the build and runtime
opt-ins to enable the [experimental sampler graph](r2sk_cuda_graph.md).
The graph applies to eligible single-GPU decode with the native sampler
and no requested log probabilities, with `2 <= max_k <= 256`.
All three settings default to off.

## Validate and replay

Build with `BUILD_UTEST=ON`, then run:

```sh
ctest --test-dir build --no-tests=error -R '^cpp_r2sk_(sampling|graph|decode_graph)_test$' --output-on-failure
compute-sanitizer --tool memcheck --error-exitcode 1 build/bin/cpp_r2sk_sampling_test
compute-sanitizer --tool synccheck --error-exitcode 1 build/bin/cpp_r2sk_sampling_test
```

Run the source-built engine on a BF16 generation request with `top_k=20`,
first with `AS_ENABLE_R2SK_SAMPLING=1`, then with the variable unset. Exercise
concurrent requests with different k and a log-probability request as separate
integration checks. A source-built Python run needs bindings from the same
checkout if the application uses Python.

For the B200 selector workload, run the standalone build and benchmark on
a Linux CUDA machine with `nvcc`, `protoc`, protobuf development headers
and libraries, and glog development headers and libraries. Pass the CUDA
target for the GPU, such as `90` for H100 or `100` for B200.

```sh
bash scripts/bench_r2sk_selector.sh 100 results-b200
```

The harness generates BF16 Gaussian logits with seed 42 for `k=256` and
batches 1, 16 and 64. It checks eager dispatch and selector graph replay,
then saves raw timing CSVs for eager RadiK, integrated eager R2S-K, the
bare device selector, and graph replay with and without overlap. The
earlier host-controlled eager arm is recorded in the historical results.
This benchmark runs without model downloads, external fixtures or a Modal
account.

The same test binary can replay captured pre-top-k logits for eager-eligible
shapes as a dense, little-endian `[batch,vocabulary]` array of BF16 words.
It validates both outputs before timing and reports complete host call
latency including stream completion, plus the scratch requirement for each
selector. Use a capture you provide and pass its shape:

```sh
build/bin/cpp_r2sk_sampling_test --replay /path/to/capture_0.bin 1 151936 20 > replay.csv
```

The source-built replay measures selection only. Full generation also
includes the model forward pass, top-p and token sampling.

## Attribution

The controller and CUDA primitives are ported from Siraaj Rizvi's private
`radik-design` prototype, revision
`eae1d0c25341e47b0b76970e24c6bb8731e6d4d9`. The prototype has no
repository-wide license. The code contributed here carries DashInfer's
Apache-2.0 license. The low-level histogram, scan and compaction primitives
were informed by RadiK revision
`c934b15d45655bec1cf4fdc4fcb04ba4f31511cf`. See the retained Alibaba copyright
and MIT notice in `csrc/core/kernel/cuda/topk_r2sk/LICENSE.RadiK`.
