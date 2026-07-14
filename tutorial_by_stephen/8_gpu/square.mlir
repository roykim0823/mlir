// The same "square each element" kernel as square.cu, but in MLIR — and written
// at a HIGH level, as a plain parallel loop. We don't write thread/block index
// math by hand; we say "these iterations are independent" with scf.parallel and
// let the GPU lowering passes manufacture the kernel.
//
// build.sh lowers this in stages so you can watch it become GPU code:
//   scf.parallel  --(map + convert-parallel-loops-to-gpu + gpu-kernel-outlining)-->
//   a gpu.module containing a gpu.func kernel + a host-side gpu.launch_func,
//   then --convert-gpu-to-nvvm--> the NVVM dialect (MLIR's mirror of NVIDIA's
//   device IR, where the loop index becomes nvvm.read.ptx.sreg.tid/ctaid — the
//   PTX special registers, i.e. threadIdx / blockIdx from square.cu).
//
// NOTE: this is the same `scf.parallel` you'd hand to OpenMP in Chapter 3. The
// only difference is the *target* the lowering passes pick — CPU threads there,
// GPU threads here. One high-level program, two backends.
func.func @square(%a: memref<1024xf32>) {
  %c0    = arith.constant 0    : index
  %c1    = arith.constant 1    : index
  %c1024 = arith.constant 1024 : index
  scf.parallel (%i) = (%c0) to (%c1024) step (%c1) {
    %v = memref.load %a[%i] : memref<1024xf32>
    %t = arith.mulf %v, %v : f32        // square this element
    memref.store %t, %a[%i] : memref<1024xf32>
    scf.reduce
  }
  return
}
