// array_add: c[i] = a[i] + b[i] over three 1-D memref<1024xf32> buffers.
//
// A `memref` is a concrete, mutable memory buffer (pointer + shape/strides),
// the counterpart to a value-typed tensor. This step compiles the kernel and
// drives it from Python two ways: aot_main.py loads the pre-built .dylib, and
// jit_main.py JIT-compiles this file at runtime with llvmlite. Both pass NumPy
// arrays in by wrapping them in MemRef-descriptor structs
// (see ../common/np_memref.py) — no data copying.
//
// `llvm.emit_c_interface` emits the `_mlir_ciface_array_add` wrapper that
// both drivers call. The three memref<1024xf32> args become a, b, and the
// output c (written in place).
module {
  func.func @array_add(%arg0: memref<1024xf32>,    // input  a
                       %arg1: memref<1024xf32>,     // input  b
                       %arg2: memref<1024xf32>)     // output c (written in place)
      attributes {llvm.emit_c_interface} {
    // Loop bounds are `index`-typed (platform-sized integers).
    %c0    = arith.constant 0    : index   // start
    %c1024 = arith.constant 1024 : index   // end (exclusive)
    %c1    = arith.constant 1    : index   // step

    // One element per iteration: load a[i] and b[i], add, store into c[i].
    scf.for %arg3 = %c0 to %c1024 step %c1 {
      %0 = memref.load %arg0[%arg3] : memref<1024xf32>   // a[i]
      %1 = memref.load %arg1[%arg3] : memref<1024xf32>   // b[i]
      %2 = arith.addf %0, %1 : f32                        // a[i] + b[i]
      memref.store %2, %arg2[%arg3] : memref<1024xf32>   // c[i] = ...
    }
    return
  }
}
