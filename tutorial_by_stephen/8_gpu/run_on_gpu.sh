#!/usr/bin/env bash
# REFERENCE ONLY — do NOT expect this to run on macOS/Apple Silicon.
#
# This is the command you'd use on a Linux box with an NVIDIA GPU and an MLIR
# built with the CUDA runner enabled (-DMLIR_ENABLE_CUDA_RUNNER=ON), e.g. the
# reference series' docker image  ghcr.io/sdiehl/docker-mlir-cuda:main.
#
# It continues past build.sh's inspection: after lowering to the gpu/NVVM dialect
# it serializes the kernel to a GPU binary (gpu-module-to-binary), lowers the host
# side to LLVM, and JIT-runs it with mlir-runner linked against the CUDA runtime.
set -euo pipefail

CHIP="${1:-sm_80}"   # set to your GPU's architecture (sm_75 Turing, sm_80 Ampere, sm_90 Hopper, ...)

mlir-opt square.mlir \
  --gpu-map-parallel-loops \
  --convert-parallel-loops-to-gpu \
  --gpu-kernel-outlining \
  --convert-gpu-to-nvvm \
  "--nvvm-attach-target=chip=${CHIP}" \
  --gpu-to-llvm \
  --gpu-module-to-binary \
  --convert-to-llvm \
  -o square_gpu_ready.mlir

# Then execute via the MLIR CPU/GPU JIT runner, linking the CUDA runtime wrapper
# shipped with a CUDA-enabled MLIR build:
#
# mlir-runner square_gpu_ready.mlir \
#   --shared-libs=$LLVM/lib/libmlir_cuda_runtime.so \
#   --shared-libs=$LLVM/lib/libmlir_runner_utils.so \
#   --entry-point-result=void
