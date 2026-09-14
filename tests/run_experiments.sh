#!/usr/bin/env bash
# Builds the project (Release) and runs the correctness test followed by the
# full benchmark sweep, writing results/benchmark.csv and results/device_info.csv.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build}"
OUT_CSV="${OUT_CSV:-$ROOT_DIR/results/benchmark.csv}"

cmake -S "$ROOT_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" --config Release -j

ctest --test-dir "$BUILD_DIR" --build-config Release --output-on-failure

# Single-config generators (Makefiles/Ninja) place the binary directly under
# BUILD_DIR; multi-config generators (Visual Studio) nest it under Release/.
BENCH_BIN="$BUILD_DIR/benchmark"
[ -x "$BENCH_BIN" ] || BENCH_BIN="$BUILD_DIR/Release/benchmark"
[ -x "$BENCH_BIN" ] || BENCH_BIN="$BUILD_DIR/benchmark.exe"
[ -x "$BENCH_BIN" ] || BENCH_BIN="$BUILD_DIR/Release/benchmark.exe"

"$BENCH_BIN" --out "$OUT_CSV" "$@"
