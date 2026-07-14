// Row-wise softmax — the operation at the heart of attention.
//
//   out[i, j] = exp(x[i, j] - max_j x[i, :]) / sum_j exp(x[i, :] - max_j x[i, :])
//
// In a transformer this turns the scaled attention scores QKᵀ/√d into a
// probability distribution over tokens (see gpt2/model.py `softmax`/`attention`).
// We subtract the row max first for numerical stability — exp of a large number
// overflows, and softmax is invariant to that shift.
//
// Each row is handled by three sequential reductions/passes:
//   1. find the row max          (scf.for with a maximumf accumulator)
//   2. write exp(x - max), summing as we go
//   3. divide each element by the row sum
//
// `math.exp` lowers to a libm call (clang links libm by default). Exported as
// `_mlir_ciface_softmax` (see aot_main.py).
func.func @softmax(%x: memref<?x?xf32>, %out: memref<?x?xf32>)
    attributes {llvm.emit_c_interface} {
  %c0   = arith.constant 0 : index
  %c1   = arith.constant 1 : index
  %N    = memref.dim %x, %c0 : memref<?x?xf32>
  %M    = memref.dim %x, %c1 : memref<?x?xf32>
  %ninf = arith.constant -3.40282347E+38 : f32   // -FLT_MAX, the max identity
  %zero = arith.constant 0.0 : f32

  scf.for %i = %c0 to %N step %c1 {
    // 1. row maximum
    %mx = scf.for %j = %c0 to %M step %c1 iter_args(%m = %ninf) -> (f32) {
      %v  = memref.load %x[%i, %j] : memref<?x?xf32>
      %mm = arith.maximumf %m, %v : f32
      scf.yield %mm : f32
    }
    // 2. exp(x - max), stored into out, accumulating the row sum
    %sum = scf.for %j = %c0 to %M step %c1 iter_args(%s = %zero) -> (f32) {
      %v = memref.load %x[%i, %j] : memref<?x?xf32>
      %d = arith.subf %v, %mx : f32
      %e = math.exp %d : f32
      memref.store %e, %out[%i, %j] : memref<?x?xf32>
      %ss = arith.addf %s, %e : f32
      scf.yield %ss : f32
    }
    // 3. normalize the row by its sum
    scf.for %j = %c0 to %M step %c1 {
      %e = memref.load %out[%i, %j] : memref<?x?xf32>
      %r = arith.divf %e, %sum : f32
      memref.store %r, %out[%i, %j] : memref<?x?xf32>
    }
  }
  return
}
