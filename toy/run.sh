#!/usr/bin/env bash
# Run a chapter's demo commands against the test inputs in ../test_Example.
#
# Usage:
#   ./run.sh ch1        # run chapter 1's demo
#   ./run.sh ch6        # run chapter 6's five emission modes + JIT
#   ./run.sh all        # run every chapter in order
#
# Binaries are looked up in build/bin/ (superbuild) first, then in
# ChN/build/ (standalone chapter build) as a fallback.
set -euo pipefail
cd "$(dirname "$0")"

TESTS=../test_Example/Toy

bin() {
  local n=$1
  if [[ -x "build/bin/toyc-ch$n" ]]; then
    echo "build/bin/toyc-ch$n"
  elif [[ -x "Ch$n/build/toyc-ch$n" ]]; then
    echo "Ch$n/build/toyc-ch$n"
  else
    echo "error: toyc-ch$n not found — build it with ./build.sh ch$n" >&2
    return 1
  fi
}

run_ch1() {
  local B; B=$(bin 1)
  "$B" "$TESTS/Ch1/ast.toy" -emit=ast
}

run_ch2() {
  local B; B=$(bin 2)
  "$B" Ch2/codegen.toy -emit=mlir -mlir-print-debuginfo

  # Round trip: emit MLIR to a file, then re-parse that .mlir and emit again.
  # If the second output parses and prints cleanly, the dialect's custom
  # printers and parsers are consistent with each other.
  local tmp
  tmp=$(mktemp -t toy-ch2-codegen).mlir
  "$B" Ch2/codegen.toy -emit=mlir -mlir-print-debuginfo 2> "$tmp"
  "$B" "$tmp" -emit=mlir
  rm -f "$tmp"
}

run_ch3() {
  local B; B=$(bin 3)
  "$B" "$TESTS/Ch3/transpose_transpose.toy" -emit=mlir -opt
  "$B" "$TESTS/Ch3/trivial_reshape.toy" -emit=mlir -opt
}

run_ch4() {
  local B; B=$(bin 4)
  "$B" "$TESTS/Ch4/codegen.toy" -emit=mlir -opt
}

run_ch5() {
  local B; B=$(bin 5)
  "$B" "$TESTS/Ch5/affine-lowering.mlir" -emit=mlir -opt
  "$B" "$TESTS/Ch5/affine-lowering.mlir" -emit=mlir-affine
  echo ""
  echo "---------"
  echo "with -opt"
  echo "---------"
  "$B" "$TESTS/Ch5/affine-lowering.mlir" -emit=mlir-affine -opt
}

run_ch6() {
  local B; B=$(bin 6)
  local prog='def main() { print([[1, 2], [3, 4]]); }'
  local mode
  for mode in mlir llvm mlir-affine mlir-llvm jit; do
    echo "== -emit=$mode =="
    echo "$prog" | "$B" -emit="$mode"
  done
}

run_ch7() {
  local B; B=$(bin 7)
  "$B" "$TESTS/Ch7/struct-codegen.toy" -emit=mlir
}

case "${1:-}" in
  ch[1-7])
    "run_${1}"
    ;;
  all)
    for n in 1 2 3 4 5 6 7; do
      echo "########## Chapter $n ##########"
      "run_ch$n"
      echo ""
    done
    ;;
  *)
    echo "usage: ./run.sh <ch1..ch7|all>" >&2
    exit 1
    ;;
esac
