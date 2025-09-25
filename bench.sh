#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")" && pwd)
BUILD_DIR="$ROOT_DIR/build"

TPB_LIST=(1 2 4 8 16)
ITERS=${NTT_BENCH_ITERS:-200}

echo "mode,tpb,avg_ms,min_ms,max_ms,gpu_mem_MB"

configure_and_build() {
  local extra_flags="$1"
  rm -rf "$BUILD_DIR"
  cmake -S "$ROOT_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_CUDA_FLAGS="$extra_flags" >/dev/null
  cmake --build "$BUILD_DIR" -j >/dev/null
}

run_case() {
  local mode="$1"; shift
  local tpb="$1"; shift
  local flags="$1"; shift
  configure_and_build "$flags"
  local out
  out=$(NTT_BENCH_ITERS=$ITERS NTT_DEBUG_TW=0 "$BUILD_DIR/ntt_demo" | sed -n 's/.*{"N":.*/&/p')
  # out is JSON-like: {"N":...,"iters":...,"avg_ms":X,"min_ms":Y,"max_ms":Z,"gpu_mem_used_MB":M}
  local avg=$(echo "$out" | sed -n 's/.*"avg_ms":\([^,}]*\).*/\1/p')
  local min=$(echo "$out" | sed -n 's/.*"min_ms":\([^,}]*\).*/\1/p')
  local max=$(echo "$out" | sed -n 's/.*"max_ms":\([^,}]*\).*/\1/p')
  local mem=$(echo "$out" | sed -n 's/.*"gpu_mem_used_MB":\([^,}]*\).*/\1/p')
  echo "$mode,$tpb,$avg,$min,$max,$mem"
}

# Barrett
for t in "${TPB_LIST[@]}"; do
  run_case "barrett" "$t" "-DTILES_PER_BLOCK_S01=$t -DTILES_PER_BLOCK_S23=$t"
done

# Montgomery (상수 인코딩 포함)
for t in "${TPB_LIST[@]}"; do
  run_case "montgomery" "$t" "-DTILES_PER_BLOCK_S01=$t -DTILES_PER_BLOCK_S23=$t -DUSE_MONTGOMERY=1 -DMONT_CONST_ENCODE=1"
done


