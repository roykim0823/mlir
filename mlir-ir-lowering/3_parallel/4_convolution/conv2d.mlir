// 2-D convolution (really cross-correlation) with the affine dialect.
//
//   output[i, j] = sum over the filter window of  filter[fi, fj] * input[i+fi, j+fj]
//
// The workhorse of image processing and CNNs. Slide the filter over the input; at
// each position, multiply the overlapping cells elementwise and sum them into one
// output cell. Dimensions are dynamic, so one compiled kernel handles any size;
// the output shape is (in_h - k_h + 1, in_w - k_w + 1), which the caller allocates.
//
// Exported as `_mlir_ciface_conv_2d` (see aot_main.py).
func.func @conv_2d(%input: memref<?x?xf32>, %filter: memref<?x?xf32>, %output: memref<?x?xf32>)
    attributes {llvm.emit_c_interface} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %OH = memref.dim %output, %c0 : memref<?x?xf32>   // output rows  = in_h - k_h + 1
  %OW = memref.dim %output, %c1 : memref<?x?xf32>   // output cols  = in_w - k_w + 1
  %KH = memref.dim %filter, %c0 : memref<?x?xf32>   // filter rows  (window height)
  %KW = memref.dim %filter, %c1 : memref<?x?xf32>   // filter cols  (window width)

  // ---- outer loops: one iteration per OUTPUT cell (i, j) ---------------------
  // Independent — each writes a distinct output[i, j] — so `--affine-parallelize`
  // turns these two into `affine.parallel` (see build/conv2d_parallel.mlir).
  affine.for %i = 0 to %OH {
    affine.for %j = 0 to %OW {
      %zero = arith.constant 0.0 : f32

      // ---- inner loops: a REDUCTION over the filter window (fi, fj) ----------
      // Accumulate the dot product of the filter with the input window anchored
      // at (i, j). `iter_args` threads the running sum (%a, then %b) through the
      // nest and `affine.yield` hands it to the next iteration — Chapter 1's
      // loop-carried value, now on affine.for. These loops build ONE value, so
      // they stay sequential (a reduction, not a parallel band).
      %acc = affine.for %fi = 0 to %KH iter_args(%a = %zero) -> (f32) {
        %acc2 = affine.for %fj = 0 to %KW iter_args(%b = %a) -> (f32) {
          %fv = affine.load %filter[%fi, %fj] : memref<?x?xf32>          // filter[fi, fj]
          %iv = affine.load %input[%i + %fi, %j + %fj] : memref<?x?xf32> // input[i+fi, j+fj]  (affine index!)
          %p  = arith.mulf %iv, %fv : f32                               // multiply
          %n  = arith.addf %b, %p : f32                                 // add into running sum
          affine.yield %n : f32                                        // carry sum to next fj
        }
        affine.yield %acc2 : f32                                        // carry sum to next fi
      }
      affine.store %acc, %output[%i, %j] : memref<?x?xf32>   // write the finished output[i, j]
    }
  }
  return
}
