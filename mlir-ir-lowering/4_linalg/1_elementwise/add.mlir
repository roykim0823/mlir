// The same elementwise add as generic_add.mlir, but using the NAMED op
// `linalg.add` — the whole linalg.generic boilerplate collapses to one line.
//
// Linalg ships named specializations for the common elementwise maps:
//   linalg.add   linalg.sub   linalg.mul   linalg.div   (and linalg.map { op })
//
// We operate directly on memrefs (the `outs` buffer is written in place), so no
// bufferization is needed — `-convert-linalg-to-loops` lowers it straight to an
// scf loop. Exported as `_mlir_ciface_addv` (see aot_main.py).
func.func @addv(%a: memref<10xf32>, %b: memref<10xf32>, %c: memref<10xf32>)
    attributes {llvm.emit_c_interface} {
  linalg.add ins(%a, %b : memref<10xf32>, memref<10xf32>)
             outs(%c : memref<10xf32>)
  return
}
