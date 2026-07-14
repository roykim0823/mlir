# 8 — GPU Compilation with MLIR

### The endgame: from a parallel loop to GPU threads

Everything has led here. Chapter 7 left us with a transformer whose hot ops are
matmul and softmax; Chapter 4 showed how to tile them; Chapter 3 showed how to
mark independent iterations parallel. This chapter takes that last step —
**mapping parallel work onto a GPU's thousands of cores** — which is what makes
modern deep learning practical.

The beautiful part is how little new you need. A GPU kernel is, conceptually,
*one iteration of a parallel loop, run on one thread*. You already wrote
`scf.parallel` in Chapter 3 and handed it to OpenMP for CPU threads; here you hand
the *same* loop to a different set of lowering passes and it becomes a GPU kernel.
One high-level program, two backends.

> **⚠️ This chapter is inspect-only on this machine.** GPU execution needs an
> **NVIDIA GPU + CUDA**, and an MLIR built with the CUDA runner
> (`-DMLIR_ENABLE_CUDA_RUNNER=ON`) — *"sorry, no macOS"*, as the reference puts
> it. The lowering *passes* are pure IR→IR transforms and run fine on Apple
> Silicon, so you can watch a loop **turn into** GPU code; you just can't run it
> here. [`run_on_gpu.sh`](run_on_gpu.sh) has the commands for a real CUDA box.
>
> Based on Stephen Diehl's *"MLIR Part 8 — GPU Compilation with MLIR"*
> ([`../reference/`](../reference/)).

---

## How the GPU sees the work

A GPU runs a **grid** of **thread blocks**, each block a group of **threads**. You
launch a *kernel* across this grid (`kernel<<<blocks, threads>>>(...)` in CUDA),
and every thread runs the same code but on different data, locating its element
from its coordinates:

```text
   grid ───────────────────────────────────────────────
   ┌── block 0 ──┐ ┌── block 1 ──┐ ┌── block 2 ──┐ ...
   │ t0 t1 … t255│ │ t0 t1 … t255│ │ t0 t1 … t255│
   └─────────────┘ └─────────────┘ └─────────────┘
       each thread computes ONE element:
       tid = blockDim.x · blockIdx.x + threadIdx.x
             └ block size ┘ └ which block ┘ └ within block ┘
```

```c
int tid = blockDim.x * blockIdx.x + threadIdx.x;   // this thread's global index
if (tid < n) array[tid] = array[tid] * array[tid];
```

That index arithmetic — built from `blockIdx`, `blockDim`, `threadIdx` — is the
heart of GPU programming. [`square.cu`](square.cu) is the full CUDA-C version; its
kernel is exactly the snippet above:

*square.cu* (the CUDA kernel)
```c
__global__ void square(float *array, int n) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;   // this thread's global index
  if (tid < n)                                       // guard the array bounds
    array[tid] = array[tid] * array[tid];
}
```

### The CUDA compilation chain

NVIDIA's toolchain has its own IR ladder, parallel to LLVM's:

| Stage | What it is |
| --- | --- |
| CUDA C++ | `__global__` kernels, launched with `<<<…>>>` |
| **PTX** | architecture-neutral parallel-thread assembly (stable across GPU generations) |
| **CUBIN** | device-specific machine code for one `sm_XX` architecture |
| **SASS** | the actual GPU shader assembly CUBIN contains |
| FATBIN | a bundle of PTX + several CUBINs, so one binary supports many GPUs |

GPU architectures are named `sm_<N>`: `sm_75` (Turing), `sm_80` (Ampere, A100),
`sm_90` (Hopper, H100), etc. A CUBIN runs only on its generation; PTX is
JIT-compiled to fill the gaps.

---

## The MLIR `gpu` dialect

MLIR doesn't make you write thread-index math. You express parallelism at a high
level (`scf.parallel`) and a sequence of passes manufactures the kernel. The
`gpu` dialect mirrors the CUDA model — `gpu.module` (a container of device
kernels), `gpu.func`, `gpu.launch_func` (the host-side launch), and
`gpu.block_id` / `gpu.thread_id` (the coordinate ops). Lowering further produces
the **NVVM** dialect — MLIR's mirror of NVIDIA's device IR, where the loop index
becomes `nvvm.read.ptx.sreg.tid.x` / `ctaid.x` (exactly `threadIdx` / `blockIdx`).

[`square.mlir`](square.mlir) is the high-level kernel — the same elementwise
square as `square.cu`, but written as one `scf.parallel` loop with **no index
math at all**. This is the identical op you handed to OpenMP in Chapter 3:

*square.mlir*
```mlir
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
```

### The lowering pipeline (`build.sh`)

`build.sh` lowers it in stages, saving each so you can read the transformation.

**Stage 1 — `scf.parallel` → a GPU kernel + host launch.** Three passes map the
loop onto the grid, then *outline* its body into a separate `gpu.func`:

```bash
$ mlir-opt square.mlir \
    --gpu-map-parallel-loops \
    --convert-parallel-loops-to-gpu \
    --gpu-kernel-outlining
```

The loop is gone. Its body is now a `gpu.func @square_kernel` indexed by
`gpu.block_id`, launched from the host by a `gpu.launch_func` — and the compiler
synthesized the `blockIdx·blockDim + threadIdx` index math for you (the
`affine.apply #map1` inside the kernel):

*build/square_gpu.mlir*
```mlir
#map = affine_map<(d0)[s0, s1] -> ((d0 - s0) ceildiv s1)>
#map1 = affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>
module attributes {gpu.container_module} {
  func.func @square(%arg0: memref<1024xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1024 = arith.constant 1024 : index
    %c1_0 = arith.constant 1 : index
    %0 = affine.apply #map(%c1024)[%c0, %c1]
    gpu.launch_func  @square_kernel::@square_kernel blocks in (%0, %c1_0, %c1_0) threads in (%c1_0, %c1_0, %c1_0)  args(%c1 : index, %c0 : index, %arg0 : memref<1024xf32>)
    return
  }
  gpu.module @square_kernel {
    gpu.func @square_kernel(%arg0: index, %arg1: index, %arg2: memref<1024xf32>) kernel attributes {known_block_size = array<i32: 1, 1, 1>} {
      %block_id_x = gpu.block_id  x
      %block_id_y = gpu.block_id  y
      %block_id_z = gpu.block_id  z
      %thread_id_x = gpu.thread_id  x
      %thread_id_y = gpu.thread_id  y
      %thread_id_z = gpu.thread_id  z
      %grid_dim_x = gpu.grid_dim  x
      %grid_dim_y = gpu.grid_dim  y
      %grid_dim_z = gpu.grid_dim  z
      %block_dim_x = gpu.block_dim  x
      %block_dim_y = gpu.block_dim  y
      %block_dim_z = gpu.block_dim  z
      %0 = affine.apply #map1(%block_id_x)[%arg0, %arg1]
      %1 = memref.load %arg2[%0] : memref<1024xf32>
      %2 = arith.mulf %1, %1 : f32
      memref.store %2, %arg2[%0] : memref<1024xf32>
      gpu.return
    }
  }
}
```

**Stage 2 — device kernel → NVVM (PTX-level).** Taking that `gpu.module` and
converting to NVVM turns the coordinate ops into PTX special registers and the
memref into raw pointer math:

```bash
$ mlir-opt build/square_gpu.mlir --convert-gpu-to-nvvm
```

`gpu.block_id x` has become `nvvm.read.ptx.sreg.ctaid.x` (literally the `blockIdx`
register), the load/store are `llvm.getelementptr` + `llvm.load`/`store`, and the
kernel is now an `llvm.func` tagged `nvvm.kernel`:

*build/square_nvvm.mlir*
```mlir
#map = affine_map<(d0)[s0, s1] -> ((d0 - s0) ceildiv s1)>
#map1 = affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>
module attributes {gpu.container_module} {
  func.func @square(%arg0: memref<1024xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1024 = arith.constant 1024 : index
    %c1_0 = arith.constant 1 : index
    %0 = affine.apply #map(%c1024)[%c0, %c1]
    gpu.launch_func  @square_kernel::@square_kernel blocks in (%0, %c1_0, %c1_0) threads in (%c1_0, %c1_0, %c1_0)  args(%c1 : index, %c0 : index, %arg0 : memref<1024xf32>)
    return
  }
  gpu.module @square_kernel {
    llvm.func @square_kernel(%arg0: i64, %arg1: i64, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: i64, %arg5: i64, %arg6: i64) attributes {gpu.kernel, gpu.known_block_size = array<i32: 1, 1, 1>, nvvm.kernel, nvvm.maxntid = array<i32: 1, 1, 1>} {
      %0 = builtin.unrealized_conversion_cast %arg1 : i64 to index
      %1 = builtin.unrealized_conversion_cast %arg0 : i64 to index
      %2 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      %3 = llvm.insertvalue %arg2, %2[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
      %4 = llvm.insertvalue %arg3, %3[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
      %5 = llvm.insertvalue %arg4, %4[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
      %6 = llvm.insertvalue %arg5, %5[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
      %7 = llvm.insertvalue %arg6, %6[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
      %8 = nvvm.read.ptx.sreg.ctaid.x : i32
      %9 = llvm.sext %8 : i32 to i64
      %10 = builtin.unrealized_conversion_cast %9 : i64 to index
      %11 = affine.apply #map1(%10)[%1, %0]
      %12 = builtin.unrealized_conversion_cast %11 : index to i64
      %13 = llvm.extractvalue %7[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
      %14 = llvm.getelementptr %13[%12] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %15 = llvm.load %14 : !llvm.ptr -> f32
      %16 = llvm.fmul %15, %15 : f32
      %17 = llvm.extractvalue %7[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
      %18 = llvm.getelementptr %17[%12] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      llvm.store %16, %18 : f32, !llvm.ptr
      llvm.return
    }
  }
}
```

**Stage 3 — pin a target architecture.** Also starting from the stage-1
`gpu.module`, this tags it with the GPU generation to compile for (here Ampere),
which is what the later CUBIN-emission step reads:

```bash
$ mlir-opt build/square_gpu.mlir --nvvm-attach-target=chip=sm_80
```

The only change is an attribute on the module — `#nvvm.target<...>` — now attached
so a backend knows which `sm_XX` machine code to emit:

*build/square_target.mlir* (the tagged module line)
```mlir
  gpu.module @square_kernel [#nvvm.target<chip = "sm_80">] {
```

**Run (inspect):** `cd 8_gpu && bash build.sh`, then read the files in `build/`.

```
== stage 1: scf.parallel  ->  gpu.module (outlined kernel + host launch) ==
   look for: gpu.launch_func, gpu.module @square_kernel, gpu.block_id/thread_id
== stage 2: device kernel  ->  NVVM dialect (PTX-level) ==
   look for: nvvm.read.ptx.sreg.tid.x / ctaid.x  (= threadIdx / blockIdx)
== stage 3: attach a target GPU architecture (e.g. sm_80 = Ampere) ==
   look for: #nvvm.target<chip = "sm_80"> on the gpu.module
```

### Actually running it (on a CUDA box)

On a GPU machine the pipeline continues past NVVM: `--gpu-module-to-binary`
serializes the kernel to a CUBIN, `--gpu-to-llvm` lowers the host side, and
`mlir-runner` JIT-executes it linked against `libmlir_cuda_runtime`. That's
[`run_on_gpu.sh`](run_on_gpu.sh) — reference only here.

---

## Where this leaves the series

Trace one operation all the way down and you've seen the whole stack:

```
transformer softmax / matmul   (Ch 7 / Ch 4, in tensor + linalg)
   → tiled for the memory hierarchy        (Ch 4)
   → marked parallel                       (Ch 3, scf.parallel)
   → outlined into a GPU kernel            (this chapter, gpu dialect)
   → NVVM → PTX → CUBIN → running on thousands of cores
```

That is exactly what a production ML compiler (IREE, Triton, XLA) does — and it's
all the same MLIR machinery you built up from `func.func @main` returning 42 in
Chapter 1.

## Key takeaways

- **A GPU kernel is one parallel-loop iteration per thread.** The same
  `scf.parallel` that fed OpenMP (Ch 3) lowers to a GPU kernel — the backend is
  just a different choice of passes.
- **The `gpu` dialect mirrors CUDA** (`gpu.module`, `gpu.launch_func`,
  `gpu.thread_id`); **NVVM** mirrors NVIDIA's device IR, where `gpu.block_id x`
  becomes `nvvm.read.ptx.sreg.ctaid.x`, lowering to PTX → CUBIN.
- **MLIR writes the thread-index math for you** — the `affine.apply` for
  `blockDim·blockIdx + threadIdx` is synthesized by `--gpu-kernel-outlining`, never
  hand-written.
- **Execution needs NVIDIA hardware + CUDA-enabled MLIR**; on this Mac we inspect
  the generated GPU IR, which is itself the payoff — watching a plain loop become
  a GPU kernel.

**The end.** You've gone from raw LLVM IR (Ch 1) to a transformer compiled toward
GPU kernels (Ch 8) — the full arc of the modern ML compiler stack. See the
[top-level README](../README.md) for the whole map.
```

