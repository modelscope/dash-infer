#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

if [[ $# != 2 || ! $1 =~ ^[0-9]+[a-z]?$ ]]; then
  echo "Usage: bash $0 <CUDA arch, e.g. 90 or 100> <new output directory>" >&2
  exit 2
fi
src=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
arch=$1
mkdir -- "$2"
out=$(cd -- "$2" && pwd)
build="$out/build"
mkdir -p "$build/utility"

{
  for tool in nvcc nvidia-smi protoc; do
    command -v "$tool" >/dev/null || { echo "Missing required tool: $tool" >&2; exit 1; }
  done
  git -C "$src" rev-parse HEAD
  printf 'CUDA target sm_%s\nCUDA_VISIBLE_DEVICES=%s\n' "$arch" "${CUDA_VISIBLE_DEVICES:-<unset>}"
  nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv
  nvcc --version
  protoc --version

  # The single-GPU selector doesn't use NCCL. Override the include in a
  # build-local copy so no NCCL headers or library are needed.
  sed '/^#include <nccl\.h>$/d' "$src/csrc/utility/check_cuda.h" > "$build/utility/check_cuda.h"
  protoc -I "$src/csrc/proto" --cpp_out="$build" "$src/csrc/proto/allspark.proto"

  kernel="$src/csrc/core/kernel/cuda"
  flags=(-std=c++17 -O3 -DENABLE_CUDA -DENABLE_BF16 "-arch=sm_$arch"
    "-I$build" "-I$src/csrc" "-I$src/csrc/common" "-I$src/csrc/device"
    "-I$src/csrc/device/cuda" "-I$src/csrc/utility" "-I$src/HIE-DNN/include" "-I$kernel/..")
  objects=()
  for source in topk_r2sk/r2sk.cu topk_r2sk/sampling.cu topk_radix/topk_radix_bf16.cu; do
    obj="$build/${source##*/}.o"
    nvcc "${flags[@]}" --extended-lambda --expt-relaxed-constexpr -lineinfo \
      -c "$kernel/$source" -o "$obj"
    objects+=("$obj")
  done
  binary="$build/cpp_r2sk_sampling_test"
  nvcc "${flags[@]}" -cudart=shared "$src/tests/cpp/kernel/cuda_r2sk_sampling_test.cpp" \
    "${objects[@]}" -lglog -lprotobuf -o "$binary"

  "$binary"
  "$binary" --graph-test
  for batch in 1 16 64; do
    echo "Checking and timing BF16 Gaussian, batch=$batch, vocab=151936, k=256"
    "$binary" --graph-replay gaussian "$batch" 151936 256 > "$out/b${batch}_n151936_k256.csv"
  done
} 2>&1 | tee "$out/run.log"
