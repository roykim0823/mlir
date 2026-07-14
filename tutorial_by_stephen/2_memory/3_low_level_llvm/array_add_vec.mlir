// Vectorized array_add — the same kernel as ../2_array_add but
// using vector.load / vector.store so each loop iteration processes
// 8 floats at a time instead of one. On x86 this lowers to a single
// AVX vaddps; on ARM it becomes two NEON adds.
//
// `llvm.emit_c_interface` makes the AOT-compiled .dylib callable from
// Python via the `_mlir_ciface_array_add` symbol.
module {
  func.func @array_add(%arg0: memref<1024xf32>,
                       %arg1: memref<1024xf32>,
                       %arg2: memref<1024xf32>)
      attributes {llvm.emit_c_interface} {
    %c0    = arith.constant 0    : index
    %c1024 = arith.constant 1024 : index
    %c8    = arith.constant 8    : index

    scf.for %i = %c0 to %c1024 step %c8 {
      %va = vector.load %arg0[%i] : memref<1024xf32>, vector<8xf32>
      %vb = vector.load %arg1[%i] : memref<1024xf32>, vector<8xf32>
      %vc = arith.addf %va, %vb : vector<8xf32>
      vector.store %vc, %arg2[%i] : memref<1024xf32>, vector<8xf32>
    }
    return
  }
}
