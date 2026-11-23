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
The backend applies to token sampling with compute capability 8.0 or newer,
finite BF16 logits, and `2 <= max_k <= 256` where `max_k < vocabulary`.
An unsupported shape, scratch layout, or stream-capture state uses RadiK.
The first histogram checks for NaN and infinities and sends that batch to
RadiK before writing any R2S-K output. The existing greedy, full-vocabulary,
non-BF16 and log-probability paths keep their current selectors.

The operator reserves scratch for its configured maximum batch and the
bounded rank range. It borrows the arena used by top-k and top-p. Per-operator
pinned descriptors, a second CUDA stream and events are allocated on first
use. The caller stream waits for the winner stream before subsequent work
reuses scratch. The experiment retains host decisions between radix passes.

## Validate and replay

Build with `BUILD_UTEST=ON`, then run:

```sh
ctest --test-dir build -R '^cpp_r2sk_sampling_test$' --output-on-failure
compute-sanitizer --tool memcheck --error-exitcode 1 build/bin/cpp_r2sk_sampling_test
compute-sanitizer --tool synccheck --error-exitcode 1 build/bin/cpp_r2sk_sampling_test
```

Run the source-built engine on a BF16 generation request with `top_k=20`,
first with `AS_ENABLE_R2SK_SAMPLING=1`, then with the variable unset. Exercise
concurrent requests with different k and a log-probability request as separate
integration checks. A source-built Python run needs bindings from the same
checkout if the application uses Python.

The same test binary can replay captured pre-top-k logits as a dense,
little-endian `[batch,vocabulary]` array of BF16 words. It validates both
outputs before timing and reports complete host call latency including stream
completion, plus the scratch requirement for each selector. For example,
using the `capture_0.bin` fixture from `radik-design/integration/dashinfer/fixtures/qwen-bf16`:

```sh
build/bin/cpp_r2sk_sampling_test --replay /path/to/capture_0.bin 1 151936 20 > replay.csv
```

The source-built replay measures selection only. Full generation also
includes the model forward pass, top-p and token sampling.

## Attribution

The controller and CUDA primitives are ported from Siraaj Rizvi's `radik-design`
revision `eae1d0c25341e47b0b76970e24c6bb8731e6d4d9`. The low-level histogram,
scan and compaction primitives were informed by RadiK revision
`c934b15d45655bec1cf4fdc4fcb04ba4f31511cf`. See the retained Alibaba copyright
and MIT notice in `csrc/core/kernel/cuda/topk_r2sk/LICENSE.RadiK`.
