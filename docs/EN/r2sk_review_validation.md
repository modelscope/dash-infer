<!-- SPDX-License-Identifier: Apache-2.0 -->

# R2S-K review validation on H100 and B200

## Device-controlled eager on H100

The opt-in eager sampling path in `b65f08df24bfb2d037a058b1b79c7d270869d411`
uses the two-pass BF16 device controller through `max_k=256`. The first
histogram collects the nonfinite flag, and DashInfer reads it during its
existing pre-sampling synchronization. Nonfinite inputs fall back to RadiK.
The separate sampler graph still requires its own opt-in.

### Selector sweep

One NVIDIA H100 80 GB HBM3 ran the CUDA 12.4.131 build targeting `sm_90a`.
The sweep used the reviewer's shape matrix with independently generated,
seed-42 Gaussian and uniform BF16 logits. Vocabulary sizes were 151,936
and 248,320, batch sizes were 1, 16, 64 and 256, and ranks were 20, 50
and 256. Every case passed the CPU oracle for descending values and
row-relative occurrence indices.

| k | Cases | RadiK median / integrated eager median |
| ---: | ---: | ---: |
| 20 | 16 | 1.55–2.18× |
| 50 | 16 | 1.45–2.14× |
| 256 | 16 | 1.31–1.96× |
| All | 48 | 1.31–2.18× (median ratio 1.65×) |

Each method used 51 complete host-call timings after 12 warmups with
stream completion and preallocated scratch. The integrated eager arm
includes the first histogram's finite check and the flag synchronization.
Its timing IQR was below RadiK's in all 48 cases, and its selector scratch
was 24.0–24.5% smaller. For example, Gaussian logits at vocabulary 151,936,
batch 16 and `k=256` took 98.125 µs with RadiK and 56.498 µs with integrated
eager R2S-K. These are selector results on Hopper. The GB300 retest remains
open.

To replay the matrix, build the CUDA test binary with
[`ENABLE_R2SK_SAMPLING=ON`](r2sk_sampling.md#validate-and-replay).
The loop below writes a CSV for each case.

```sh
set -e
mkdir -p results-r2sk-review
for distribution in gaussian uniform; do
  for vocabulary in 151936 248320; do
    for batch in 1 16 64 256; do
      for k in 20 50 256; do
        build/bin/cpp_r2sk_sampling_test --graph-replay \
          "$distribution" "$batch" "$vocabulary" "$k" \
          > "results-r2sk-review/${distribution}_n${vocabulary}_b${batch}_k${k}.csv"
      done
    done
  done
done
```

The `radik_eager` and `r2sk_device_eager` rows are the matched eager
comparison. The `device_eager` and graph rows measure the bare selector.

### Streaming Qwen generation

The same H100 ran Qwen2.5-1.5B-Instruct BF16 at model revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. Arms used the same source build,
prompts, weights, and sampling settings (top-p 0.8, temperature 0.7,
repetition penalty 1.1). Each configuration had 20 timed cohorts across
four blocks with rotated order and 96 output tokens per request. TTFT
and TPOT use engine result-queue delivery timestamps.

| k | Concurrency | Backend | TTFT p50 ms | TPOT p50 ms | Cohort p50 ms | Output tokens/s |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 20 | 1 | Eager RadiK | 4.789 | 3.893 | 374.587 | 256.3 |
| 20 | 1 | Eager R2S-K | 4.803 | 3.880 | 373.565 | 257.0 |
| 20 | 1 | Graph R2S-K | 4.650 | 3.764 | 362.303 | 265.0 |
| 20 | 16 | Eager RadiK | 55.644 | 4.468 | 510.178 | 3010.7 |
| 20 | 16 | Eager R2S-K | 52.508 | 4.352 | 495.128 | 3102.2 |
| 20 | 16 | Graph R2S-K | 52.213 | 4.331 | 497.491 | 3087.5 |
| 256 | 1 | Eager RadiK | 4.641 | 3.810 | 366.699 | 261.8 |
| 256 | 1 | Eager R2S-K | 4.560 | 3.771 | 362.798 | 264.6 |
| 256 | 1 | Graph R2S-K | 4.586 | 3.740 | 359.968 | 266.7 |
| 256 | 16 | Eager RadiK | 52.034 | 4.398 | 502.603 | 3056.1 |
| 256 | 16 | Eager R2S-K | 50.721 | 4.331 | 493.792 | 3110.6 |
| 256 | 16 | Graph R2S-K | 54.455 | 4.382 | 500.222 | 3070.6 |

At `k=20`, concurrency 16, eager R2S-K reduced median cohort latency by
2.95%, with lower medians in all four blocks and separated cohort timing
IQRs. The eager-vs-eager cohort timing IQRs overlap in the other three
configurations. Separate traces confirmed the new eager selector and
sampler graph both dispatched at active batches 1 and 16 for `k=256`.
The serving trial excludes HTTP transport and did not measure answer
quality for this revision.

The eager selector, graph replay and decode sampling parity tests all
passed on the same H100. The [Modal run](https://modal.com/apps/siraajr/main/ap-TmaWgTSrJkrMAUOZ4D56W4)
stored the 12,240 selector timing samples and 2,040 per-request timing
rows under `dash-decode-20261125T085342Z` in the `r2sk-results` volume.
The exact model prompts and serving runner are in the separate
`radik-design` workspace.

## Earlier H100 validation

The earlier H100 model tests used DashInfer commit `71592904387d42e792bd3dd296525cbb89f454b8`,
CUDA 12.4.1 targeting `sm_90a`, and one NVIDIA H100 80 GB HBM3. The model
was Qwen2.5-1.5B-Instruct BF16 at revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. Each backend used the same DashInfer
binary and model files. All backends used `top_k=20`, `top_p=0.8`, temperature
`0.7`, and repetition penalty `1.1`.

### Full request latency

Each sample measured warmed request submission through completion, including
prefill, decode, and sampling. Every request generated 96 tokens. Prefix
caching, early stopping, and dispatch tracing were disabled during timing.
Each arm had 20 measured cohorts across four blocks with rotated arm order.
Separate traces verified cache-hit graph launches at active batch 1 and 16.

| Concurrent requests | Backend | Median ms | Middle 50% ms | Output tokens/s |
| ---: | --- | ---: | ---: | ---: |
| 1 | Native RadiK | 396.65 | 394.83–404.46 | 242.0 |
| 1 | Eager R2S-K | 390.05 | 386.96–395.63 | 246.1 |
| 1 | Graph R2S-K | 389.09 | 386.91–390.47 | 246.7 |
| 16 | Native RadiK | 584.18 | 575.11–596.18 | 2629.3 |
| 16 | Eager R2S-K | 577.21 | 571.32–593.42 | 2661.1 |
| 16 | Graph R2S-K | 578.62 | 571.44–589.80 | 2654.6 |

Graph R2S-K reduced median latency relative to RadiK by 1.90% at batch 1 and
0.95% at batch 16. It had lower medians in four of four and three of four
paired blocks, respectively. The batch-16 IQRs overlap, and eager R2S-K had
the lowest batch-16 median by 1.4 ms. This run leaves a graph-specific
end-to-end improvement over eager R2S-K unresolved.

### Sampled GSM8K accuracy

The matched evaluation used 200 questions and two seeds, 1234 and 5678, with
requested concurrency 16. Native A and B were independent runs of the same
RadiK backend. Output was scored by the same automatic numeric extractor in
every arm. The extractor was fixed before generation. Unresolved answers
counted as incorrect. The graph arm recorded 3,182 cached launches at active
batch 16 during GSM8K generation.

| Backend | Correct | Accuracy | Unresolved answers |
| --- | ---: | ---: | ---: |
| Native A | 174/400 | 43.50% | 35 |
| Native B | 183/400 | 45.75% | 38 |
| Graph R2S-K | 177/400 | 44.25% | 27 |

Graph R2S-K differed from native A by +0.75 percentage points (95% paired
question-bootstrap interval −2.50 to +3.75) and from native B by −1.50
points (−5.25 to +2.25). Native B differed from native A by +2.25 points
(−0.50 to +5.00). These intervals leave both regression and improvement
possible. Quality equivalence remains unverified.

## Build and CI

The CUDA package CI matrix enables `ENABLE_R2SK_SAMPLING` and verifies the
selector and decode parity executables were built. This is compile-only
coverage. The earlier CPU builder marked the selector CTests as skipped.
The standalone selector, selector graph, and integrated sampler parity tests
passed on Modal H100.

The manually dispatched **R2S-K GPU correctness** workflow targets a
self-hosted `gpu` runner with NVIDIA container support and an Ampere, Ada or
Hopper GPU. It directly runs all three test entry points, so a skipped test
fails the job. GPU execution in GitHub Actions remains unverified.

The selector and graph sources passed the repository's clang-format rules.
Clang's raw token streams matched before and after formatting for all 11
changed sources. Prior HIE-DNN mixed-count initcheck findings remain open.

## H100 model experiment artifacts

The historical full-model experiment used a separate `radik-design` runner
and cached public Qwen and GSM8K inputs. Its exact prompt set, runner and
raw artifacts are outside this checkout. The [selector reproduction](r2sk_sampling.md#validate-and-replay)
uses generated inputs and runs from this repository.

The [Modal run](https://modal.com/apps/siraajr/main/ap-RQTqTAc3MEhBQQP9pP9xla)
saved build logs, all 120 timing samples, and all 1,200 scored responses under
`dash-decode-20261124T040747Z` in the `r2sk-results` volume.

## Historical large-k selector validation on B200

The standalone selector suite ran on one B200 with CUDA 13.1.115 targeting
`sm_100`, from source revision `779a20e267ea9077d389d7939999448f20dcb5ce`.
The standalone build omitted an unused NCCL include from the shared CUDA
header. The selector and graph correctness tests completed, and each timed
variant passed the CPU oracle for sorted BF16 scores and indices.

Inputs were BF16 Gaussian logits generated with seed 42, vocabulary 151,936,
and `k=256`. Results are complete-call host medians in microseconds, with
51 repetitions per method after 12 warmups. Median timings include stream
completion. Graph instantiation is outside the timed region.

| Batch | Native RadiK | Original eager R2S-K | Device-controlled eager R2S-K | Graph R2S-K with overlap |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 86.357 | 65.821 | 37.507 | 32.473 |
| 16 | 93.605 | 97.485 | 48.158 | 43.569 |
| 64 | 122.300 | 150.834 | 86.600 | 82.565 |

Before the dispatch cutoff, eager `R2skSampling::Try` was 4.1% slower than
RadiK at batch 16 and 23.3% slower at batch 64 on this workload. Commit
`a4cd8ce` capped the old eager path at 50 to avoid that regression.
The device-controlled eager column measured the bare selector. It reduced latency
relative to eager RadiK by 56.6%, 48.6%, and 29.2% at batches 1, 16, and
64, respectively, with separated timing IQRs and about 24% less selector
scratch.

The graph column compares R2S-K replay with eager RadiK. The newer eager
engine wrapper reuses this device controller and includes a finite-input
check in its first histogram. The B200 numbers above predate that wrapper.
The reviewer's GB300 retest remains outstanding.
