// OpenMP dialect — explicit parallelism, lowered to the OpenMP runtime.
//
// The affine dialect parallelizes *automatically*; the `omp` dialect is the
// other end of the spectrum — you spell out the parallelism by hand, the way you
// would with `#pragma omp parallel for` in C. This kernel doubles a 10-element
// array across a team of threads, then main() prints the result.
//
// Three core OpenMP ops appear here:
//   omp.parallel   — fork a team of threads; its region runs on every thread
//   omp.wsloop     — a worksharing loop: split the iterations across the team
//   omp.barrier    — make every thread wait until all have arrived
// Every OpenMP region must end with omp.terminator (and each loop body with
// omp.yield).
//
// This is a standalone EXECUTABLE: main() returns 0 and prints via the C printf.
// Note the printf call passes f64 — C varargs promote float to double, so we
// arith.extf the f32 element first (the format string is "%f\n").
memref.global constant @input : memref<10xf32> =
  dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]>

llvm.func @printf(!llvm.ptr, ...) -> i32
llvm.mlir.global private constant @fmt("%f\0A\00") {addr_space = 0 : i32}

func.func private @kernel(%input: memref<10xf32>, %output: memref<10xf32>) {
  %ub   = llvm.mlir.constant(9 : i32) : i32     // inclusive upper bound
  %lb   = llvm.mlir.constant(0 : i32) : i32
  %step = llvm.mlir.constant(1 : i32) : i32
  omp.parallel {
    omp.wsloop {
      omp.loop_nest (%i) : i32 = (%lb) to (%ub) inclusive step (%step) {
        %ix  = arith.index_cast %i : i32 to index
        %v   = memref.load %input[%ix] : memref<10xf32>
        %two = arith.constant 2.0 : f32
        %r   = arith.mulf %v, %two : f32
        memref.store %r, %output[%ix] : memref<10xf32>
        omp.yield
      }
    }
    omp.barrier
    omp.terminator
  }
  return
}

func.func @main() -> i32 {
  %input  = memref.get_global @input : memref<10xf32>
  %output = memref.alloc() : memref<10xf32>
  call @kernel(%input, %output) : (memref<10xf32>, memref<10xf32>) -> ()

  %lb   = index.constant 0
  %ub   = index.constant 10
  %step = index.constant 1
  %fs   = llvm.mlir.addressof @fmt : !llvm.ptr
  scf.for %iv = %lb to %ub step %step {
    %el  = memref.load %output[%iv] : memref<10xf32>
    %eld = arith.extf %el : f32 to f64           // promote for printf varargs
    llvm.call @printf(%fs, %eld) vararg(!llvm.func<i32 (ptr, ...)>)
      : (!llvm.ptr, f64) -> i32
  }
  %z = arith.constant 0 : i32
  return %z : i32
}
