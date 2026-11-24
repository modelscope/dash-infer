# R2S-K review validation on H100 and B200

The tests used DashInfer commit `71592904387d42e792bd3dd296525cbb89f454b8`,
CUDA 12.4.1 targeting `sm_90a`, and one NVIDIA H100 80 GB HBM3. The model
was Qwen2.5-1.5B-Instruct BF16 at revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. Each backend used the same DashInfer
binary and model files. All backends used `top_k=20`, `top_p=0.8`, temperature
`0.7`, and repetition penalty `1.1`.

## Full request latency

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

## Sampled GSM8K accuracy

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

The CUDA package CI matrix enables `ENABLE_R2SK_SAMPLING` and invokes the
selector and graph selector CTests. Both tests built in the full SM90a engine
build. The CPU builder recorded both CTests as skipped because it had no GPU.
The built binaries passed both tests on H100, and the integrated sampler
matched native RadiK and sampling on the focused 18-call parity test.
Device execution in GitHub Actions remains to be verified with a
CUDA-visible runner.

The selector and graph sources passed the repository's clang-format rules.
Clang's raw token streams matched before and after formatting for all 11
changed sources. Prior HIE-DNN mixed-count initcheck findings remain open.

## Reproduction

The experiment ran from the `radik-design` workspace with the checked-out
graph branch and cached public Qwen and GSM8K inputs.

```sh
git -C .cache/dash-infer-r2sk-graph diff --binary 06669fa 7159290 > .cache/dash-infer-r2sk-decode-source.patch
modal run --detach modal_dash_decode_graph.py --arch 90a --source-revision 7159290
```

The [Modal run](https://modal.com/apps/siraajr/main/ap-RQTqTAc3MEhBQQP9pP9xla)
saved build logs, all 120 timing samples, and all 1,200 scored responses under
`dash-decode-20261124T040747Z` in the `r2sk-results` volume.

## Large-k selector validation on B200

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

The original eager path regressed by 4.1% at batch 16 and 23.3% at batch
64. The device-controlled eager path reduced latency relative to RadiK by
56.6%, 48.6%, and 29.2% at batches 1, 16, and 64. Its sample IQR was fully
below RadiK's at each batch, and it used about 24% less selector scratch.
These measurements reproduce the large-k regression pattern and show its
reversal on B200. The reviewer's GB300 workload remains to be retested.
