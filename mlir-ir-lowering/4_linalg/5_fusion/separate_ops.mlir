// Kernel fusion (RUNNABLE).
//
// Here `addmul` computes (a + b) * c as TWO separate Linalg ops. Each op is its
// own loop: the add writes a whole intermediate tensor to memory, then the mul
// reads it back. That's two passes over the data and an extra allocation.
//
// Fusion combines them into a SINGLE loop that reads each input once and writes
// the result directly — eliminating the intermediate. In Python terms:
//
//   unfused:  t=[a[i]+b[i] for i]; r=[t[i]*c[i] for i]   # two loops, temp array
//   fused:    r=[(a[i]+b[i])*c[i] for i]                 # one loop, no temp
//
// Fusion is a TENSOR-level optimization: producer/consumer ops fuse before any
// buffers exist, which is why this is written on tensors (not memrefs). build.sh
// first runs the fusion pipeline and writes build/fused_ops.mlir — you'll see the
// two named ops collapse into one linalg.generic whose body does both the addf and
// the mulf before yielding — then bufferizes and compiles the fused form to check
// it against NumPy's (a + b) * c.
//
// The func returns a tensor; to run it, build.sh bufferizes it and converts the
// returned tensor into an in-place `out` buffer (-buffer-results-to-out-params), so
// the C interface matches the other steps: addmul(a, b, c, out).
module {
  // This is the unfused version, which we will fuse with a pass.
  func.func @addmul(%a: tensor<10xf32>, %b: tensor<10xf32>, %c: tensor<10xf32>) -> tensor<10xf32>
      attributes {llvm.emit_c_interface} {
    // 1. Create an empty tensor to hold the intermediate result of the addition.
    %0 = tensor.empty() : tensor<10xf32>

    // 2. Add the two input tensors and store the result in the intermediate tensor.
    %1 = linalg.add ins(%a, %b : tensor<10xf32>, tensor<10xf32>)
                    outs(%0 : tensor<10xf32>) -> tensor<10xf32>

    // 3. Create an empty tensor to hold the final result.
    %2 = tensor.empty() : tensor<10xf32>

    // 4. Multiply the intermediate result by the third input tensor and store the result.
    %3 = linalg.mul ins(%1, %c : tensor<10xf32>, tensor<10xf32>)
                    outs(%2 : tensor<10xf32>) -> tensor<10xf32>
    return %3 : tensor<10xf32>
  }
}
