#!/usr/bin/env bash
# INSPECT-ONLY on this machine: lower square.mlir from a high-level parallel loop
# down to GPU code, saving each stage so you can read the transformation.
#
# These passes are pure IR->IR transforms and run fine on any CPU (including this
# Mac). What you CANNOT do here is *execute* the result — that needs an NVIDIA GPU
# and an MLIR built with the CUDA runner (see run_on_gpu.sh). Apple Silicon has no
# NVIDIA GPU, so we stop at inspecting the generated device IR.
set -euo pipefail
rm -rf build
mkdir -p build

echo "== stage 1: scf.parallel  ->  gpu.module (outlined kernel + host launch) =="
mlir-opt square.mlir \
  --gpu-map-parallel-loops \
  --convert-parallel-loops-to-gpu \
  --gpu-kernel-outlining \
  -o build/square_gpu.mlir
echo "   look for: gpu.launch_func, gpu.module @square_kernel, gpu.block_id/thread_id"

echo "== stage 2: device kernel  ->  NVVM dialect (PTX-level) =="
mlir-opt build/square_gpu.mlir --convert-gpu-to-nvvm -o build/square_nvvm.mlir
echo "   look for: nvvm.read.ptx.sreg.tid.x / ctaid.x  (= threadIdx / blockIdx)"

echo "== stage 3: attach a target GPU architecture (e.g. sm_80 = Ampere) =="
mlir-opt build/square_gpu.mlir "--nvvm-attach-target=chip=sm_80" \
  -o build/square_target.mlir
echo "   look for: #nvvm.target<chip = \"sm_80\"> on the gpu.module"

echo
echo "Wrote build/square_gpu.mlir, build/square_nvvm.mlir, build/square_target.mlir."
echo "To actually RUN on a GPU, see run_on_gpu.sh (needs CUDA hardware + MLIR CUDA runner)."
