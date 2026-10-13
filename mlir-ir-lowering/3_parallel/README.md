# 3 — Parallelism: Affine Dialect & OpenMP

### A loop says more than it means

Write a `for` loop and you have, without meaning to, made a promise: *iteration 0
happens, then 1, then 2, …*, strictly in order. But look at a matrix multiply, or
blurring an image, or scaling a vector — the iterations don't actually *depend* on
each other. Computing `C[0][0]` doesn't need `C[0][1]` to have happened first. The
sequential order is an **accident of syntax**, not a requirement of the math. The
loop says more than it means.

That gap is where all parallel performance lives. A modern CPU has many cores, a
GPU has thousands; the entire game is to notice which work is *independent* and let
it run at the same time. Chapter 1 walked computation *down* the staircase
(loops → branches → registers); Chapter 2 asked *where data lives* (value vs.
address). **This chapter is about a third question: _when_ does work run, and what
is allowed to run at once?**

MLIR gives you two routes to answer it, and this chapter walks both:

- **Let the compiler find the parallelism** — the **`affine` dialect**. By
  restricting loops and array accesses to *affine* (quasi-linear) expressions of
  the loop indices, affine makes loops "first-class" objects the compiler can
  reason about geometrically. It can then *prove* iterations are independent and
  rewrite the nest automatically — parallelize it, tile it for cache, fuse
  adjacent loops, reorder for locality.
- **State the parallelism yourself** — the **`omp` (OpenMP) dialect**. Sometimes
  you'd rather just say "run these iterations on a team of threads" — the
  directive-based model familiar from `#pragma omp parallel for` in C. MLIR has it
  as first-class ops.

> Based on Stephen Diehl's *"MLIR Part 3 — Affine Dialect and OpenMP"*
> ([`../reference/`](../reference/)). Every example here is **built and run /
> inspected on this toolchain** (Homebrew LLVM 20.1.8, Apple Silicon); the
> OpenMP step additionally links Homebrew's `libomp`.

---

## The polyhedral model — loops as geometry

Why can a compiler reason about an `affine.for` nest but not a general
`scf.for`/C loop? Because affine constraints let it adopt the **polyhedral
model**: the idea of representing a loop nest as a *geometric object*.

Each execution of a statement inside a 2-D loop nest is a point `(i, j)`; the set
of all such points is a polygon (in higher dimensions, a polyhedron) carved out by
the loop bounds. Once your program is a shape, loop optimization becomes
*geometry*: parallelizing is slicing the shape into independent pieces, tiling is
gridding it into blocks, interchange is reflecting it, skewing is shearing it. The
model has three pieces:

- **Iteration domains** — the set of points a loop nest visits, bounded by affine
  inequalities (e.g. `0 ≤ i < M`).
- **Access relations** — which memory each point touches (e.g. point `(i, j)`
  reads `A[i, k]`).
- **Scheduling functions** — the order points execute in; a *valid* reordering is
  one that preserves every dependence.

MLIR builds three concepts on top of this:

| Concept | What it is | Example |
| --- | --- | --- |
| **Affine map** | A quasi-linear function from dims/symbols to results | `(d0, d1)[s0] -> (d0 + d1, s0 * d1)` |
| **Integer set** | Affine constraints carving out a domain | `(i)[N] : (i >= 0, N - i - 1 >= 0)` = `0 ≤ i < N` |
| **Affine ops** | `affine.for`, `affine.if`, `affine.parallel`, `affine.apply` | loop nests & index math |

The catch — and the price of all this power — is that affine expressions may use
**only** `+`, `-`, `*` by a constant, and `floordiv`/`ceildiv`/`mod` by a
constant. No data-dependent indices (`A[B[i]]`), no multiplying two loop
variables. That restriction is exactly what keeps the geometry tractable.

### `affine` vs `scf`

| | `scf` (Chapter 1) | `affine` (this chapter) |
| --- | --- | --- |
| Loop op | `scf.for` / `scf.parallel` | `affine.for` / `affine.parallel` |
| Index/access form | anything | affine expressions only |
| Memory ops | `memref.load`/`store` | `affine.load`/`store` (analyzable) |
| Compiler can auto-tile / fuse / parallelize? | no | **yes** |
| Lower it with | (already low) | `-lower-affine` → `scf` |

You write high-level `affine`, let the optimizer transform it, then `-lower-affine`
drops it to `scf`, and from there it's the same pipeline as every earlier chapter.

---

## Chapter layout

| Step | Directory | Topic | Runnable? |
| --- | --- | --- | --- |
| 1 | [`1_affine_matmul/`](1_affine_matmul/) | `affine.for` matmul + `--affine-parallelize` | ✅ Python |
| 2 | [`2_affine_maps/`](2_affine_maps/) | `affine.apply`, affine maps, tiled indexing | 🧩 inspect |
| 3 | [`3_affine_opts/`](3_affine_opts/) | the affine optimization passes + manual interchange/skewing | 🧩 inspect + ✅ |
| 4 | [`4_convolution/`](4_convolution/) | 2-D convolution with `iter_args` reductions | ✅ Python |
| 5 | [`5_openmp/`](5_openmp/) | the `omp` dialect + `-convert-scf-to-openmp` | ✅ executable |

Shared helper: [`common/np_memref2d.py`](common/np_memref2d.py) — a **2-D** memref
descriptor + NumPy adapter (Chapter 2's `np_memref.py` was 1-D), imported by the
matmul and conv2d drivers.

**Legend:** ✅ runnable · 🧩 snippet (lowered/transformed with `mlir-opt`, not run).

---

## Step 1 — Affine matmul, and automatic parallelization (`1_affine_matmul/`) · ✅

**Goal:** meet `affine.for` / `affine.load` / `affine.store`, and watch the
compiler turn an ordinary loop nest into a parallel one.

The classic product $C_{ij} = \sum_k A_{ik} B_{kj}$ as a three-deep affine nest:

*1_affine_matmul/matmul.mlir* (the loop nest)
```mlir
  affine.for %i = 0 to %M {
    affine.for %j = 0 to %N {
      affine.for %k = 0 to %K {
        %a = affine.load %A[%i, %k] : memref<?x?xf32>
        %b = affine.load %B[%k, %j] : memref<?x?xf32>
        %c = affine.load %C[%i, %j] : memref<?x?xf32>
        %prod = arith.mulf %a, %b : f32
        %sum  = arith.addf %c, %prod : f32
        affine.store %sum, %C[%i, %j] : memref<?x?xf32>
      }
    }
  }
```

The `%i` and `%j` loops are **independent** — each iteration writes a distinct
`C[i, j]`. The compiler can *see* this (affine accesses make the dependence
analysis exact), so `--affine-parallelize` rewrites those two loops into
`affine.parallel`, leaving the `k`-reduction sequential:

*build/matmul_parallel.mlir*
```mlir
affine.parallel (%arg3) = (0) to (symbol(%dim)) {          // outer (i): parallel
  affine.parallel (%arg4) = (0) to (symbol(%dim_0)) {      // inner (j): parallel
    affine.for %arg5 = 0 to %dim_1 {                       // the k-reduction stays sequential
      %0 = affine.load %arg0[%arg3, %arg5] : memref<?x?xf32>
      %1 = affine.load %arg1[%arg5, %arg4] : memref<?x?xf32>
      %2 = affine.load %arg2[%arg3, %arg4] : memref<?x?xf32>
      %3 = arith.mulf %0, %1 : f32
      %4 = arith.addf %2, %3 : f32
      affine.store %4, %arg2[%arg3, %arg4] : memref<?x?xf32>
    }
  }
}
```

`affine.parallel` says "these iterations may run in any order, or all at once."
Two notes on what you're seeing:

- **Two nested `affine.parallel`s, not one 2-D band.** The op *can* express a
  hyper-rectangular band like `affine.parallel (%i, %j) = (0,0) to (%M,%N)` (one
  op, two induction variables), but `--affine-parallelize` parallelizes each loop
  in place, so you get an outer parallel loop wrapping an inner one. They mean the
  same thing — both say the `i` and `j` iterations are independent.
- **No `affine.yield` here.** This `affine.parallel` returns *no results*: each
  iteration accumulates straight into the `C` buffer with `affine.store`, so
  there's nothing to combine across iterations. `affine.yield` only appears when a
  loop itself *produces* a value — i.e. a true parallel **reduction**, where
  `affine.parallel` carries reduction results that get combined (sum, max, …) at
  the end. (Step 4's convolution uses the value-carrying form, via `iter_args` on
  `affine.for`.) The `k`-loop here reduces too, but it does so the same
  through-memory way, which is why it has no yield either.

`build.sh` emits that parallelized IR to `build/matmul_parallel.mlir` for you to
read, then — on a **separate branch** — lowers the original *sequential*
`matmul.mlir` with `-lower-affine` → `scf` → `llvm`, compiles it, and runs
`aot_main.py`, which checks the result against NumPy's `@`.

> **The compiled matmul runs single-threaded.** `--affine-parallelize` is shown
> here only as an IR *rewrite*; the parallel version is **not** the one we
> compile. And the default lowering wouldn't preserve it anyway: `-lower-affine`
> turns `affine.parallel` into `scf.parallel`, but `-convert-scf-to-cf` then
> *serializes* `scf.parallel` into ordinary branches. To get real parallel
> execution you must route `scf.parallel` to a backend that keeps it —
> `-convert-scf-to-openmp` for CPU threads (Step 5) or the GPU passes
> (Chapter 8). So `matmul_opt.mlir` has no threads, no SIMD, and no unrolling —
> it's a plain sequential triple loop (one `llvm.fmul` in the body); all its bulk
> is *lowering*, not parallelization.

**Run:** `cd 1_affine_matmul && bash build.sh`

```
Affine matmul successful! (max abs error 1.19e-07)
```

(The tiny error is ordinary float rounding — the kernel and NumPy sum in different
orders.)

### What the lowering produces (`build/matmul_opt.mlir`)

`build.sh` also lowers the kernel all the way to the `llvm` dialect and leaves the
result in `build/matmul_opt.mlir`. It's verbose, but every part is something you've
already seen — it's worth opening once to connect the dots:

**The signature explodes.** `@matmul` takes **21 scalar arguments**, not three
memrefs: `-finalize-memref-to-llvm` unrolls each `memref<?x?xf32>` into
`(allocated_ptr, aligned_ptr, offset, size0, size1, stride0, stride1)`. The first
thing the body does is `llvm.insertvalue` those scalars back into a descriptor
struct — exactly the one from Chapter 2:

```mlir
llvm.func @matmul(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, /* ... */ %arg20: i64) {
  %0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  %1 = llvm.insertvalue %arg14, %0[0] : !llvm.struct<...>   // C: allocated ptr
  %2 = llvm.insertvalue %arg15, %1[1] : !llvm.struct<...>   // C: aligned ptr
  // ... [2]=offset, [3,0..1]=sizes, [4,0..1]=strides; then again for B, then A
```

**The `?` dimensions become runtime loads.** `memref.dim` is gone; each `?` size
is pulled out of the descriptor's size array (`extractvalue ...[3]`) at runtime:

```mlir
%27 = llvm.extractvalue %23[3] : !llvm.struct<...>   // A's [size0, size1] array
// ... indexed + loaded to give %M, %N, %K as i64 loop bounds
```

**The loops became blocks and branches.** No loop ops remain — the three
`affine.for`s are `^bb1`…`^bb9` wired with `llvm.cond_br`/`llvm.br` (the
`affine → scf → cf` lowering, the same shape as Chapter 1's loop). We lower the
**sequential** kernel, so it's a plain nest, not the parallel version above:

```mlir
  llvm.br ^bb1(%41 : i64)
^bb1(%43: i64):                          // the i loop
  %44 = llvm.icmp "slt" %43, %30 : i64   // i < M ?
  llvm.cond_br %44, ^bb2, ^bb9
// ^bb3(%47) is the j loop, ^bb5(%51) the k loop, each the same shape
```

**`affine.load/store` became pointer math.** Each access is a `llvm.mul`/`add`
computing the row-major offset `i*stride0 + k` (Chapter 2), a `getelementptr`, and
a `load`/`store`; `arith.mulf`/`addf` are `llvm.fmul`/`fadd`:

```mlir
%53 = llvm.extractvalue %23[1]              // A aligned ptr
%54 = llvm.extractvalue %23[4, 0]           // A stride0
%55 = llvm.mul %43, %54 : i64               // i * stride0
%56 = llvm.add %55, %51 : i64               // + k
%57 = llvm.getelementptr %53[%56] : (!llvm.ptr, i64) -> !llvm.ptr, f32   // &A[i,k]
%58 = llvm.load %57 : !llvm.ptr -> f32      // A[i,k]
// ... likewise B[k,j], C[i,j]; then:
%71 = llvm.fmul %58, %64 : f32              // A[i,k] * B[k,j]
%72 = llvm.fadd %70, %71 : f32              // C[i,j] + ...
```

**There are two functions.** The real `@matmul` (21 unrolled args) plus a generated
`@_mlir_ciface_matmul` wrapper that takes three descriptor *pointers*, loads each
struct, unpacks its fields, and calls `@matmul` — the C-interface marshalling from
Chapter 2's Step 4, and what `aot_main.py` actually calls:

```mlir
llvm.func @_mlir_ciface_matmul(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) {
  %0 = llvm.load %arg0 : !llvm.ptr -> !llvm.struct<...>   // load A's descriptor
  %1 = llvm.extractvalue %0[0] : !llvm.struct<...>        // unpack its 7 fields
  // ... unpack B and C too, then:
  llvm.call @matmul(%1, %2, /* ...21 scalars... */) : (...) -> ()
}
```

In short: affine math → scalarized loops over raw pointers, ready for
`mlir-translate` to hand to LLVM.

---

## Step 2 — Affine maps & `affine.apply` (`2_affine_maps/`) · 🧩

**Goal:** the index arithmetic that makes the whole dialect tick.

Affine maps are the affine dialect's workhorse — multi-dimensional affine
transformations that spell out *how loop indices map to the memory each iteration
touches*. That makes them the lever for the optimizations this chapter is about:
once the compiler knows the access pattern as an affine map, it can reason about
memory layout and locality, which is exactly what high-performance code on CPUs
and GPUs lives or dies by.

Concretely, an **affine map** is a named function from loop indices (*dimensions*)
and runtime-constant parameters (*symbols*) to `index` results; `affine.apply`
evaluates one. The dims-vs-symbols split — `(d0, d1)[s0]` — matters: symbols are
values that are loop-invariant, so the analysis treats them as constants.

`affine_apply.mlir` works through tiling, symbols, multi-dim maps, and composition
— the whole file:

*2_affine_maps/affine_apply.mlir*
```mlir
// Reusable named maps (the `#name` attribute-alias sigil from Chapter 1):
// Define some resuable affine maps
#tile_map    = affine_map<(d0) -> (d0 floordiv 32)>          // which 32-wide tile?
#offset_map  = affine_map<(d0)[s0] -> (d0 + s0)>            // shift by a symbol
#complex_map = affine_map<(d0, d1)[s0] -> (d0 * 2 + d1 floordiv 4 + s0)>

func.func @affine_apply_examples() {
  // create some test indices
  %c0   = arith.constant 0   : index
  %c42  = arith.constant 42  : index
  %c128 = arith.constant 128 : index

  // Example 1. Simple Tiling Calculation: 42 floordiv 32 = 1 (element 42 lives in tile #1).
  %tile = affine.apply #tile_map(%c42)

  // Example 2. An inline affine (anonymous) map for an offset: 42 + 10 = 52.
  %off  = affine.apply affine_map<(i) -> (i + 10)>(%c42)

  // Example 3. Using a symbol parameter: 42 + 128 = 170.  Dims in (), symbols in [].
  %shift = affine.apply #offset_map(%c42)[%c128]

  // Example 4. Multiple dims and a symbol: 42*2 + 128 floordiv 4 + 0 = 116.
  %cplx = affine.apply #complex_map(%c42, %c128)[%c0]

  // Example 5. Composition of affine maps — feed one apply's result into the next.
  %t     = affine.apply affine_map<(i) -> (i * 2)>(%c42)   // 84
  %final = affine.apply affine_map<(i) -> (i + 5)>(%t)     // 89

  return
}
```

`tiled_loop.mlir` shows the real use — `affine.apply` turning a
`(tile, within-tile)` pair into a flat array index inside a tiled loop nest:

*2_affine_maps/tiled_loop.mlir*
```mlir
// Example showing how affine.apply can be used in a practical loop context
func.func @tiled_loop(%arg0: memref<256xf32>) {
  affine.for %i = 0 to 256 step 32 {       // outer: one iteration per tile
    affine.for %j = 0 to 32 {              // inner: elements within the tile
      %idx = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%i, %j)
      %val = memref.load %arg0[%idx] : memref<256xf32>
      // ... process %val ...
      memref.store %val, %arg0[%idx] : memref<256xf32>
    }
  }
  return
}
```

This is `affine.apply` doing real work: a tiled traversal of an array. The outer
loop strides by the tile size and the inner loop walks one tile, and `affine.apply`
is what turns those two loop counters into the flat element index the loads/stores
use — the everyday pattern behind cache tiling.

A few rules that pin down what an affine map can be:

- **`floordiv` is integer division rounding toward −∞** (and `ceildiv` rounds up);
  `mod` is the matching remainder.
- **`(d0)[s0]` = one dimension `d0` plus one symbol `s0`** — dims in `()`, symbols
  in `[]`.
- **Only a fixed vocabulary is allowed:** `+`, `-`, multiplication *by a constant*,
  and division / `mod` / `floordiv` / `ceildiv` *by a constant*. No two variables
  multiplied, no data-dependent terms — that linearity is what makes the maps
  analyzable.
- **The result is always `index`.** Maps compute positions, never data.

`build.sh` parses both files and writes four outputs into `build/`:

| File | Pass | What it shows |
| --- | --- | --- |
| `affine_apply_verified.mlir` | (parse + print) | the input round-tripped; note the *inline* maps get hoisted into named `#map`s and deduplicated |
| `affine_apply_folded.mlir` | `-canonicalize` | the body collapses to just `return` — every `affine.apply` is **side-effect-free and its result is unused**, so it's dead code and gets deleted (the same value-semantics point as tensors in Chapter 2) |
| `tiled_loop_verified.mlir` | (parse + print) | the tiled loop round-tripped, still in the `affine` dialect |
| `tiled_loop_lowered.mlir` | `-lower-affine` | the affine ops expand: `affine.for` → `scf.for` (bounds/step spelled out as `arith.constant`s) and `affine.apply<(d0,d1)->(d0+d1)>` → a plain `arith.addi` |

So the pair shows both directions: `affine.apply` is *erasable* when its result
isn't used (the folded file), and *lowers to ordinary index arithmetic* when it is
(the lowered file).

**Run:** `cd 2_affine_maps && bash build.sh` (then read the files in `build/`).

---

## Step 3 — Affine optimization passes (`3_affine_opts/`) · 🧩 + ✅

**Goal:** the payoff of first-class loops. Because the `affine` dialect lets the
compiler treat a loop nest as analyzable geometry, MLIR ships a rich catalog of
passes that **automatically rewrite the nest to run faster** — without changing
what it computes. Each subsection below pairs a minimal source file with the exact
command that transforms it and the **real** `mlir-opt` output. `build.sh` applies
every pass and writes the results to `build/<name>_after.mlir`, so you can also
`diff <name>.mlir build/<name>_after.mlir` yourself.

### Loop-invariant code motion

Code inside the loop that doesn't depend on the loop index is recomputed every
iteration for no reason. LICM lifts it out — and by clearing index-independent work
out of the body, it often exposes parallelism too. Here `%x = 42.0` is constant
across all iterations, so it hops above the loop:

*3_affine_opts/licm.mlir*
```mlir
func.func @licm(%A: memref<10xf32>, %B: memref<10xf32>) {
  affine.for %i = 0 to 10 {
    %x = arith.constant 42.0 : f32           // loop-invariant — gets hoisted
    %v = affine.load %A[%i] : memref<10xf32>
    %s = arith.addf %v, %x : f32
    affine.store %s, %B[%i] : memref<10xf32>
  }
  return
}
```

Run the pass:

```bash
$ mlir-opt 3_affine_opts/licm.mlir -affine-loop-invariant-code-motion
```

The `arith.constant` now sits **before** the loop — computed once instead of ten
times:

```mlir
module {
  func.func @licm(%arg0: memref<10xf32>, %arg1: memref<10xf32>) {
    %cst = arith.constant 4.200000e+01 : f32
    affine.for %arg2 = 0 to 10 {
      %0 = affine.load %arg0[%arg2] : memref<10xf32>
      %1 = arith.addf %0, %cst : f32
      affine.store %1, %arg1[%arg2] : memref<10xf32>
    }
    return
  }
}
```

> The `module { … }` wrapper in these outputs is MLIR's **implicit top-level
> module**, which every `.mlir` file has (Chapter 1) — the source files omit the
> keyword, but `mlir-opt` always prints it explicitly. The "after" blocks below
> are the verbatim `build/<name>_after.mlir` output.

### Tiling

Tiling splits a loop into cache-sized **blocks**: instead of sweeping a whole row
before touching the next, you fully process an 8×8 tile (which fits in cache) and
only then move on.

*3_affine_opts/tiling.mlir*
```mlir
func.func @tiling(%A: memref<32x32xf32>) {
  affine.for %i = 0 to 32 {
    affine.for %j = 0 to 32 {
      %v = affine.load %A[%i, %j] : memref<32x32xf32>
      %t = arith.addf %v, %v : f32
      affine.store %t, %A[%i, %j] : memref<32x32xf32>
    }
  }
  return
}
```

Run the pass:

```bash
$ mlir-opt 3_affine_opts/tiling.mlir -affine-loop-tile="tile-size=8"
```

The 2-deep nest becomes 4-deep — outer *tile* loops stepping by 8, inner *point*
loops (bounds `#map = (d0)->(d0)` and `#map1 = (d0)->(d0+8)`) walking the 8
elements of each tile:

```mlir
#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0 + 8)>
module {
  func.func @tiling(%arg0: memref<32x32xf32>) {
    affine.for %arg1 = 0 to 32 step 8 {
      affine.for %arg2 = 0 to 32 step 8 {
        affine.for %arg3 = #map(%arg1) to #map1(%arg1) {
          affine.for %arg4 = #map(%arg2) to #map1(%arg2) {
            %0 = affine.load %arg0[%arg3, %arg4] : memref<32x32xf32>
            %1 = arith.addf %0, %0 : f32
            affine.store %1, %arg0[%arg3, %arg4] : memref<32x32xf32>
          }
        }
      }
    }
    return
  }
}
```

### Unrolling

Replicating the body N times cuts loop overhead (fewer bound checks and branches)
and hands the CPU more independent work to pipeline.

*3_affine_opts/unroll.mlir*
```mlir
func.func @unroll(%A: memref<8xf32>) {
  affine.for %i = 0 to 8 {
    %v = affine.load %A[%i] : memref<8xf32>
    %t = arith.addf %v, %v : f32
    affine.store %t, %A[%i] : memref<8xf32>
  }
  return
}
```

Run the pass:

```bash
$ mlir-opt 3_affine_opts/unroll.mlir -affine-loop-unroll="unroll-factor=4"
```

The loop now steps by 4 (2 trips instead of 8) and the body appears four times,
the copies indexing `i`, `i+1`, `i+2`, `i+3` via `affine.apply`:

```mlir
#map = affine_map<(d0) -> (d0 + 1)>
#map1 = affine_map<(d0) -> (d0 + 2)>
#map2 = affine_map<(d0) -> (d0 + 3)>
module {
  func.func @unroll(%arg0: memref<8xf32>) {
    affine.for %arg1 = 0 to 8 step 4 {
      %0 = affine.load %arg0[%arg1] : memref<8xf32>
      %1 = arith.addf %0, %0 : f32
      affine.store %1, %arg0[%arg1] : memref<8xf32>
      %2 = affine.apply #map(%arg1)
      %3 = affine.load %arg0[%2] : memref<8xf32>
      %4 = arith.addf %3, %3 : f32
      affine.store %4, %arg0[%2] : memref<8xf32>
      %5 = affine.apply #map1(%arg1)
      %6 = affine.load %arg0[%5] : memref<8xf32>
      %7 = arith.addf %6, %6 : f32
      affine.store %7, %arg0[%5] : memref<8xf32>
      %8 = affine.apply #map2(%arg1)
      %9 = affine.load %arg0[%8] : memref<8xf32>
      %10 = arith.addf %9, %9 : f32
      affine.store %10, %arg0[%8] : memref<8xf32>
    }
    return
  }
}
```

### Fusion

Two adjacent loops over the same range — the first writes `B` from `A`, the second
reads `B` to write `C` — are merged into one, so each `B[i]` is produced and
immediately consumed while it's still hot. (This is the loop-level cousin of the
`linalg` op fusion in Chapter 4.)

*3_affine_opts/fusion.mlir*
```mlir
func.func @fusion(%A: memref<10xf32>, %B: memref<10xf32>, %C: memref<10xf32>) {
  affine.for %i = 0 to 10 {
    %v1 = affine.load %A[%i] : memref<10xf32>
    %v2 = arith.mulf %v1, %v1 : f32
    affine.store %v2, %B[%i] : memref<10xf32>
  }
  affine.for %i = 0 to 10 {
    %v3 = affine.load %B[%i] : memref<10xf32>
    %v4 = arith.addf %v3, %v3 : f32
    affine.store %v4, %C[%i] : memref<10xf32>
  }
  return
}
```

Run the pass:

```bash
$ mlir-opt 3_affine_opts/fusion.mlir -affine-loop-fusion
```

The two loops collapse into one, the second body folded in after the first:

```mlir
module {
  func.func @fusion(%arg0: memref<10xf32>, %arg1: memref<10xf32>, %arg2: memref<10xf32>) {
    affine.for %arg3 = 0 to 10 {
      %0 = affine.load %arg0[%arg3] : memref<10xf32>
      %1 = arith.mulf %0, %0 : f32
      affine.store %1, %arg1[%arg3] : memref<10xf32>
      %2 = affine.load %arg1[%arg3] : memref<10xf32>
      %3 = arith.addf %2, %2 : f32
      affine.store %3, %arg2[%arg3] : memref<10xf32>
    }
    return
  }
}
```

### Coalescing

The opposite of tiling: *flatten* a perfectly-nested loop into one, recovering the
original indices with `floordiv`/`mod`. Handy when you want a single flat iteration
space — e.g. to map onto a 1-D thread grid.

*3_affine_opts/coalescing.mlir*
```mlir
func.func @coalescing(%A: memref<4x8xf32>) {
  affine.for %i = 0 to 4 {
    affine.for %j = 0 to 8 {
      %v = affine.load %A[%i, %j] : memref<4x8xf32>
      %t = arith.addf %v, %v : f32
      affine.store %t, %A[%i, %j] : memref<4x8xf32>
    }
  }
  return
}
```

Run the pass:

```bash
$ mlir-opt 3_affine_opts/coalescing.mlir -affine-loop-coalescing
```

The 4×8 nest becomes one loop to 32, with the row/col rebuilt inside via `mod` /
`floordiv` (`#map3`/`#map4`):

```mlir
#map = affine_map<() -> (4)>
#map1 = affine_map<() -> (8)>
#map2 = affine_map<(d0)[s0] -> (d0 * s0)>
#map3 = affine_map<(d0)[s0] -> (d0 mod s0)>
#map4 = affine_map<(d0)[s0] -> (d0 floordiv s0)>
module {
  func.func @coalescing(%arg0: memref<4x8xf32>) {
    %0 = affine.apply #map()
    %1 = affine.apply #map1()
    %2 = affine.apply #map2(%0)[%1]
    affine.for %arg1 = 0 to %2 {
      %3 = affine.apply #map3(%arg1)[%1]
      %4 = affine.apply #map4(%arg1)[%1]
      %5 = affine.load %arg0[%4, %3] : memref<4x8xf32>
      %6 = arith.addf %5, %5 : f32
      affine.store %6, %arg0[%4, %3] : memref<4x8xf32>
    }
    return
  }
}
```

### Normalization

Rewrite a loop to start at 0 and step 1, folding the old bounds into the index.
Many passes assume normalized loops, so this is a common pre-pass.

*3_affine_opts/normalize.mlir*
```mlir
func.func @normalize(%A: memref<100xf32>) {
  affine.for %i = 10 to 100 step 5 {
    %v = affine.load %A[%i] : memref<100xf32>
    %t = arith.addf %v, %v : f32
    affine.store %t, %A[%i] : memref<100xf32>
  }
  return
}
```

Run the pass:

```bash
$ mlir-opt 3_affine_opts/normalize.mlir -affine-loop-normalize
```

`10 to 100 step 5` has `(100−10)/5 = 18` iterations, so it becomes `0 to 18`, with
the real index recovered as `i*5 + 10` (`#map`):

```mlir
#map = affine_map<(d0) -> (d0 * 5 + 10)>
module {
  func.func @normalize(%arg0: memref<100xf32>) {
    affine.for %arg1 = 0 to 18 {
      %0 = affine.apply #map(%arg1)
      %1 = affine.load %arg0[%0] : memref<100xf32>
      %2 = arith.addf %1, %1 : f32
      affine.store %2, %arg0[%0] : memref<100xf32>
    }
    return
  }
}
```

### Scalar replacement — `-affine-scalrep`

When a value is stored and then loaded straight back from the same place, the load
is redundant — the analysis forwards the stored value and deletes the load.

*3_affine_opts/scalrep.mlir*
```mlir
func.func @scalrep(%A: memref<10xf32>) {
  affine.for %i = 0 to 10 {
    %c = arith.constant 1.0 : f32
    affine.store %c, %A[%i] : memref<10xf32>         // store ...
    %v = affine.load %A[%i] : memref<10xf32>         // ... and load right back
    %t = arith.addf %v, %v : f32
    affine.store %t, %A[%i] : memref<10xf32>
  }
  return
}
```

Run the pass:

```bash
$ mlir-opt 3_affine_opts/scalrep.mlir -affine-scalrep
```

The load is gone — `%c` is used directly — and the now-dead first store is dropped,
leaving one store per iteration:

```mlir
module {
  func.func @scalrep(%arg0: memref<10xf32>) {
    affine.for %arg1 = 0 to 10 {
      %cst = arith.constant 1.000000e+00 : f32
      %0 = arith.addf %cst, %cst : f32
      affine.store %0, %arg0[%arg1] : memref<10xf32>
    }
    return
  }
}
```

### Super-vectorization

The affine dialect's own route to SIMD: a scalar loop becomes a vector one (Chapter
2 met the `vector` dialect directly).

*3_affine_opts/vectorize.mlir*
```mlir
func.func @vectorize(%A: memref<256xf32>, %B: memref<256xf32>) {
  affine.for %i = 0 to 256 {
    %a = affine.load %A[%i] : memref<256xf32>
    %b = affine.load %B[%i] : memref<256xf32>
    %s = arith.addf %a, %b : f32
    affine.store %s, %A[%i] : memref<256xf32>
  }
  return
}
```

Run the pass:

```bash
$ mlir-opt 3_affine_opts/vectorize.mlir -affine-super-vectorize="virtual-vector-size=8"
```

The loop now steps by 8, and the elementwise loads/adds become
`vector.transfer_read` / `arith.addf : vector<8xf32>` / `transfer_write` — 8
elements per iteration:

```mlir
module {
  func.func @vectorize(%arg0: memref<256xf32>, %arg1: memref<256xf32>) {
    affine.for %arg2 = 0 to 256 step 8 {
      %cst = arith.constant 0.000000e+00 : f32
      %0 = vector.transfer_read %arg0[%arg2], %cst : memref<256xf32>, vector<8xf32>
      %cst_0 = arith.constant 0.000000e+00 : f32
      %1 = vector.transfer_read %arg1[%arg2], %cst_0 : memref<256xf32>, vector<8xf32>
      %2 = arith.addf %0, %1 : vector<8xf32>
      vector.transfer_write %2, %arg0[%arg2] : vector<8xf32>, memref<256xf32>
    }
    return
  }
}
```

The remaining catalog entries: `-affine-parallelize` you already saw in Step 1
(it produces `affine.parallel`); `-affine-loop-unroll-jam` is unrolling that also
fuses the resulting copies of an inner loop; and `-affine-pipeline-data-transfer`
is specialized — it overlaps `affine.dma` transfers with compute, so it only does
anything on code that already uses the DMA ops.

The full pass catalog (all available in `mlir-opt`):

| Pass | Transformation |
| --- | --- |
| `-affine-loop-coalescing` | Flatten nested loops into one |
| `-affine-loop-fusion` | Merge adjacent loops for locality |
| `-affine-loop-invariant-code-motion` | Hoist code that doesn't depend on the loop index |
| `-affine-loop-normalize` | Normalize bounds/steps to start at 0, step 1 |
| `-affine-loop-tile` | Block a loop into cache-friendly tiles |
| `-affine-loop-unroll` / `-unroll-jam` | Replicate the body (and jam nested loops) |
| `-affine-parallelize` | Turn independent loops into `affine.parallel` |
| `-affine-pipeline-data-transfer` | Pipeline DMA between memory levels |
| `-affine-scalrep` | Forward stores to loads, kill redundant loads |
| `-affine-simplify-structures` | Simplify affine expressions in maps/sets and normalize memrefs |
| `-affine-super-vectorize` | Vectorize to an n-D vector abstraction |

There's also `-convert-affine-for-to-gpu`, which turns affine loops straight into
GPU kernels — the bridge to a later chapter.

### Done by hand, not by a pass: interchange, skewing & the wavefront · ✅

Two transformations the PDF discusses have **no one-flag affine pass** in
`mlir-opt`, so we perform them by rewriting the loop order / bounds / index maps
ourselves. (Other compiler-driven routes exist — the Transform dialect and C++
utilities — see the callout at the end.) They live in two files; `build.sh`
compiles each and runs `aot_main.py` to **confirm every version produces identical
results**.

#### Interchange (`interchange_manual.mlir`)

Interchange swaps the loop nesting order. Visiting `(i, j)` in a different order
can't change a copy `B = A` — every element is written once and the iterations are
independent — but row-major (`i` outer) has far better cache locality than
column-major.

*3_affine_opts/interchange_manual.mlir* — before (`j` outer, column-major walk):
```mlir
func.func @copy_ji(%A: memref<4x5xf32>, %B: memref<4x5xf32>)
    attributes {llvm.emit_c_interface} {
  affine.for %j = 0 to 5 {
    affine.for %i = 0 to 4 {
      %v = affine.load %A[%i, %j] : memref<4x5xf32>
      affine.store %v, %B[%i, %j] : memref<4x5xf32>
    }
  }
  return
}
```

after (`i` outer, row-major walk — the interchanged version):
```mlir
func.func @copy_ij(%A: memref<4x5xf32>, %B: memref<4x5xf32>)
    attributes {llvm.emit_c_interface} {
  affine.for %i = 0 to 4 {
    affine.for %j = 0 to 5 {
      %v = affine.load %A[%i, %j] : memref<4x5xf32>
      affine.store %v, %B[%i, %j] : memref<4x5xf32>
    }
  }
  return
}
```

#### Skewing & the wavefront (`skewing_manual.mlir`)

Skewing shears the iteration space. In the stencil `A[i,j] = A[i-1,j] + A[i,j-1]`
each cell needs its **left** (`i, j-1`) and **upper** (`i-1, j`) neighbour, so
**both** loops are sequential — nothing is parallel yet. This file has three
versions that all compute the same array.

*3_affine_opts/skewing_manual.mlir* — the plain sequential stencil (reference):
```mlir
func.func @stencil(%A: memref<8x8xf32>)
    attributes {llvm.emit_c_interface} {
  affine.for %i = 1 to 8 {
    affine.for %j = 1 to 8 {
      %a = affine.load %A[%i - 1, %j] : memref<8x8xf32>
      %b = affine.load %A[%i, %j - 1] : memref<8x8xf32>
      %s = arith.addf %a, %b : f32
      affine.store %s, %A[%i, %j] : memref<8x8xf32>
    }
  }
  return
}
```

Skewing *alone* remaps the inner index (`j_new = j - i`, bounds shifted by `i`),
visiting the same `(i, j_new)` cells with the same update:
```mlir
func.func @stencil_skewed(%A: memref<8x8xf32>)
    attributes {llvm.emit_c_interface} {
  affine.for %i = 1 to 8 {
    affine.for %j = affine_map<(d) -> (d + 1)>(%i) to affine_map<(d) -> (d + 8)>(%i) {
      %jn = affine.apply affine_map<(d0, d1) -> (d1 - d0)>(%i, %j)   // j_new = j - i
      %a = affine.load %A[%i - 1, %jn] : memref<8x8xf32>
      %b = affine.load %A[%i, %jn - 1] : memref<8x8xf32>
      %s = arith.addf %a, %b : f32
      affine.store %s, %A[%i, %jn] : memref<8x8xf32>
    }
  }
  return
}
```

> **Skewing alone gives no speedup** — it's *just a reindexing*: `i` is still the
> outer loop and the inner loop still reads the cell the previous inner step wrote
> (`A[i, j_new-1]`), so it stays sequential. Skewing's real job is to set up a
> **follow-up loop interchange** — that pair is the wavefront, below.

Skew **+ interchange = the wavefront (the actual payoff)**: make the skewed
dimension — the anti-diagonal `t = i + j` — the *outer* loop, and iterate the cells
*on* each diagonal in the inner loop. Every cell `(i, t−i)` depends only on diagonal
`t−1`, so no two cells of the same diagonal depend on each other — the **inner loop
is now genuinely parallel** (`affine.parallel`):
```mlir
func.func @stencil_wavefront(%A: memref<8x8xf32>)
    attributes {llvm.emit_c_interface} {
  affine.for %t = 2 to 15 {                                  // outer: over diagonals t = i + j
    // cells on diagonal t: i in [max(1, t-7), min(7, t-1)] so j = t-i stays in [1,7]
    affine.parallel (%i) = (max(1, %t - 7)) to (min(8, %t)) {   // inner: PARALLEL
      %j = affine.apply affine_map<(d0, d1) -> (d1 - d0)>(%i, %t)   // j = t - i
      %a = affine.load %A[%i - 1, %j] : memref<8x8xf32>
      %b = affine.load %A[%i, %j - 1] : memref<8x8xf32>
      %s = arith.addf %a, %b : f32
      affine.store %s, %A[%i, %j] : memref<8x8xf32>
    }
  }
  return
}
```

The outer `affine.for` walks the diagonals in order (diagonal `t` still needs
`t−1`); the inner `affine.parallel` is the reward — a whole diagonal at once. (As
with Step 1's matmul, `-lower-affine` here still *serializes* the `affine.parallel`;
real parallel execution needs the OpenMP/GPU route. The point is that the loop is
now *legally* parallel, which the sequential nest never was.)

**The picture.** You're exactly right that it sweeps the `"/"` (NE↔SW) diagonals.
Lay the grid out with row `i` going *down* and column `j` going *right*, and label
each cell with its wavefront `t = i + j`:

```text
         j=1  j=2  j=3  j=4
  i=1  [  2    3    4    5 ]        cells with the SAME label lie on one
  i=2  [  3    4    5    6 ]        anti-diagonal — a "/" line running NE↔SW.
  i=3  [  4    5    6    7 ]        that line is one `affine.parallel` (all
  i=4  [  5    6    7    8 ]        its cells at once).
        (rows i ↓, cols j →)

  the outer loop marches t = 2, 3, 4, … : the "/" wavefront slides from the
  NW corner (1,1)  ──►  the SE corner (N,N).
```

Why every cell on a `"/"` is independent of the others *on the same line*: cell
`(i,j)` reads only its **North** neighbour `A[i-1,j]` and its **West** neighbour
`A[i,j-1]` — and both of those have `i+j = t-1`, i.e. they live on the *previous*
diagonal (up and to the left), never on the current one:

```text
        A[i-1,j]        <- North neighbour, on diagonal t-1
           │
 A[i,j-1] ─┼─► A[i,j]   <- West neighbour, on diagonal t-1
           (current cell, on diagonal t)
```

So each diagonal depends only on the one before it (that's why the *outer* `t`
loop stays sequential), while the cells within a diagonal touch disjoint data
(that's why the *inner* loop is parallel).

> **Why `affine.apply (j - i)` and not `arith.subi %j, %i`?** Inside the affine
> dialect every index passed to `affine.load`/`store` must be a *valid affine
> value* — a loop induction variable, a symbol, or an `affine.apply` result. A
> plain `arith.subi` result is opaque to the affine machinery, so the access would
> be rejected (*"index must be a valid dimension or symbol identifier"*).
> `affine.apply` is the sanctioned way to materialize an index expression. For a
> one-off you don't even need it — because `%i` and `%j` are affine dims you can
> fold the subtraction straight into the access, `affine.load %A[%i - 1, %j - %i]`;
> we use `affine.apply` here only to name `j_new` once and reuse it in all three
> accesses.

**Run:** `cd 3_affine_opts && bash build.sh` — the last step prints:

```
interchange:  copy_ij == copy_ji == A  ✓
skewing:      stencil == stencil_skewed  ✓
wavefront:    stencil == stencil_wavefront (parallel inner loop)  ✓
```

> **So is there really no pass?** Not a stock one-flag *affine* pass — there's no
> `-affine-loop-interchange` or `-affine-loop-skew` in `mlir-opt`. Both are only
> *legal* when they preserve every data dependence, and only *profitable* in
> specific cases, so MLIR treats them as building blocks a scheduler invokes after
> analysis, not blind global rewrites. Compiler-driven routes still exist, though:
>
> - **Interchange** — the **Transform dialect** offers
>   `transform.structured.interchange`, which permutes a *linalg* op's iterator
>   order *before* it lowers to loops (the modern, scriptable way to choose loop
>   order — Chapter 4). MLIR also has C++ utilities (`interchangeLoops` /
>   `permuteLoops`) a custom pass can call, and LLVM runs its own automatic
>   `LoopInterchange` late in the back end.
> - **Skewing** — no user-facing pass *or* transform op; it exists only as an
>   internal C++ utility (`affineForOpBodySkew`, used by
>   `-affine-pipeline-data-transfer`). So from `mlir-opt` alone you do it by hand,
>   as shown here.

---

## Step 4 — 2-D convolution (`4_convolution/`) · ✅

**Goal:** a real kernel — and the affine version of Chapter 1's loop-carried
value. We'll see it in two forms the build produces: the source, and the
`--affine-parallelize`d form.

### How 2-D convolution works

Convolution (really *cross-correlation*) slides a small **filter** over an
**input**. At each position it lays the filter over a window of the input,
multiplies the overlapping cells, and sums them into a single **output** cell:

$$\text{output}[i,j] = \sum_{fi}\sum_{fj} \text{filter}[fi,fj]\cdot\text{input}[i+fi,\, j+fj]$$

```text
   input (in_h × in_w)                filter (KH × KW)        output (OH × OW)
  ┌───────────────────────┐          ┌───────────────┐       ┌───────────────┐
  │┌───────┐              │          │ f00 f01 f02   │       │ o00 ...       │
  ││a  b  c│ d  e  ...    │          │ f10 f11 f12   │  ==>  │  ↑            │
  ││g  h  i│ j  k  ...    │    ⊛     │ f20 f21 f22   │       │ o00 = a·f00 + b·f01 + c·f02
  ││m  n  o│ p  q  ...    │          └───────────────┘       │      + g·f10 + h·f11 + i·f12
  │└───────┘              │                                  │      + m·f20 + n·f21 + o·f22
  │ ...                   │        the boxed KH×KW window     │
  └───────────────────────┘        (anchored at i,j) dotted  └───────────────┘
                                    with the filter → output[i,j]

  slide the window one step right → output[i, j+1]; down → output[i+1, j].
  output size = (in_h − KH + 1) × (in_w − KW + 1)   (only positions where the
                                                     window fully fits).
```

So there are **four** loops: two *outer* over the output cells `(i, j)`, and two
*inner* that reduce over the filter window `(fi, fj)` to build each cell's sum.

### The kernel (`conv2d.mlir`)

The inner reduction accumulates with **`iter_args`** — the same loop-carried-value
mechanism `scf.for` used in Chapter 1, now on `affine.for`, each loop passing its
running sum on with `affine.yield`. The full source:

*4_convolution/conv2d.mlir*
```mlir
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
```

The index `%input[%i + %fi, %j + %fj]` is an affine function of the loop
variables — that's what keeps even a sliding-window access analyzable.

### Parallelizing it (`build/conv2d_parallel.mlir`)

The two *output* loops are independent (distinct `output[i,j]` each), so
`--affine-parallelize` turns them into an `affine.parallel` band; the filter
reduction keeps its `iter_args` and stays a sequential `affine.for`:

*build/conv2d_parallel.mlir*
```mlir
affine.parallel (%arg3) = (0) to (symbol(%dim)) {          // output rows: parallel
  affine.parallel (%arg4) = (0) to (symbol(%dim_0)) {      // output cols: parallel
    %cst = arith.constant 0.000000e+00 : f32
    %0 = affine.for %arg5 = 0 to %dim_1 iter_args(%arg6 = %cst) -> (f32) {   // filter rows: reduce
      %1 = affine.for %arg7 = 0 to %dim_2 iter_args(%arg8 = %arg6) -> (f32) { // filter cols: reduce
        %2 = affine.load %arg1[%arg5, %arg7] : memref<?x?xf32>
        %3 = affine.load %arg0[%arg3 + %arg5, %arg4 + %arg7] : memref<?x?xf32>
        %4 = arith.mulf %3, %2 : f32
        %5 = arith.addf %arg8, %4 : f32
        affine.yield %5 : f32
      }
      affine.yield %1 : f32
    }
    affine.store %0, %arg2[%arg3, %arg4] : memref<?x?xf32>
  }
}
```

(As in Step 1, this is inspect-only — the *runnable* build lowers the sequential
source, and the default pipeline serializes `affine.parallel` anyway; real
parallelism needs the OpenMP/GPU route.)

From there `build.sh` lowers the sequential kernel with `-lower-affine` → `scf` →
`llvm` and compiles it, the same path as every earlier chapter (Step 1's
`matmul_opt.mlir` walks that lowering through in detail). `aot_main.py` then checks
the result against a NumPy reference.

**Run:** `cd 4_convolution && bash build.sh`

```
Affine conv2d successful! (output 8x8, max abs error 3.81e-06)
```

---

## Step 5 — OpenMP (`5_openmp/`) · ✅

**Goal:** the other end of the spectrum — *explicit* parallelism, lowered to a
real threading runtime.

The affine dialect had the *compiler* discover parallelism for you. OpenMP is the
opposite instinct: sometimes you want **direct, explicit control** over how work is
parallelized. OpenMP is a **directive-based** model — you annotate ordinary loops
with pragmas and a runtime spreads the iterations across a team of threads — and
it's a mainstay of high-performance computing.

Its native form is C. This kernel doubles a 10-element array: `#pragma omp
parallel` opens a team of threads, and `#pragma omp for` shares the loop's
iterations among them:

*5_openmp/omp_double.c*
```c
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

int kernel(float *input, float *output) {
#pragma omp parallel
  {
#pragma omp for
    for (int i = 0; i < 10; i++) {
      output[i] = input[i] * 2.0f;
    }
  }
}

int main() {
  float input[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  float output[10];
  kernel(input, output);
  for (int i = 0; i < 10; i++) {
    printf("%f ", output[i]);
  }
  return 0;
}
```

Compiled with OpenMP support (`clang -fopenmp`), that loop runs across multiple
threads. MLIR exposes the *same* primitives as first-class ops in the **`omp`
dialect**, so the identical computation can be written in MLIR — which is the rest
of this step. (`build.sh` compiles and runs both the C version above and the MLIR
version below, so you can see them print the same result.)

The `omp` dialect ops:

| Op | Meaning |
| --- | --- |
| `omp.parallel` | Fork a team of threads; its region runs on each |
| `omp.wsloop` + `omp.loop_nest` | A worksharing loop — split iterations across the team |
| `omp.barrier` | Make all threads wait until everyone arrives |
| `omp.terminator` | Closes an `omp.parallel` (and similar) region — the "end of this block" marker |
| `omp.yield` | Closes the `omp.loop_nest` body — marks the end of *one loop iteration* |

MLIR regions must end in a terminator op (a block can't just "fall through"), so
`omp.yield` and `omp.terminator` are both bookkeeping, not computation. The
distinction is which region they close: **`omp.yield`** ends the body of the
`omp.loop_nest` — i.e. "this iteration is done" — analogous to `scf.yield` /
`affine.yield` ending a loop body earlier in the chapter (here it carries no value,
since the loop just writes to memory). **`omp.terminator`** ends the outer
`omp.parallel` region. So in `omp_double.mlir` you'll see one `omp.yield` inside the
innermost loop and one `omp.terminator` closing the parallel block.

`omp_double.mlir` doubles a 10-element array across a thread team, then `main`
prints it — a standalone **executable**:

*5_openmp/omp_double.mlir* (the parallel region)
```mlir
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
```

**You rarely write this by hand.** The usual route is to produce `scf.parallel`
(by writing it, or via `--affine-parallelize` + `-lower-affine`) and let
`-convert-scf-to-openmp` rewrite it into `omp.parallel`/`omp.wsloop`. That's all
`scf_parallel.mlir` is — a plain parallel loop:

*5_openmp/scf_parallel.mlir*
```mlir
func.func @add(%A: memref<100xf32>, %B: memref<100xf32>) {
  %c0   = arith.constant 0   : index
  %c1   = arith.constant 1   : index
  %c100 = arith.constant 100 : index
  scf.parallel (%i) = (%c0) to (%c100) step (%c1) {
    %v = memref.load %A[%i] : memref<100xf32>
    %r = arith.addf %v, %v : f32
    memref.store %r, %B[%i] : memref<100xf32>
    scf.reduce
  }
  return
}
```

Run the conversion:

```bash
$ mlir-opt 5_openmp/scf_parallel.mlir -convert-scf-to-openmp
```

The single `scf.parallel` becomes an `omp.parallel` region wrapping an
`omp.wsloop`/`omp.loop_nest`, with the bounds/step carried over (`build.sh` saves
this to `build/scf_parallel_omp.mlir`):

*build/scf_parallel_omp.mlir*
```mlir
module {
  func.func @add(%arg0: memref<100xf32>, %arg1: memref<100xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c100 = arith.constant 100 : index
    %0 = llvm.mlir.constant(1 : i64) : i64
    omp.parallel {
      omp.wsloop {
        omp.loop_nest (%arg2) : index = (%c0) to (%c100) step (%c1) {
          memref.alloca_scope  {
            %1 = memref.load %arg0[%arg2] : memref<100xf32>
            %2 = arith.addf %1, %1 : f32
            memref.store %2, %arg1[%arg2] : memref<100xf32>
          }
          omp.yield
        }
      }
      omp.terminator
    }
    return
  }
}
```

A real pipeline then continues with `-convert-openmp-to-llvm` (as `omp_double.mlir`
above does) to reach the OpenMP runtime.

> **Aside — variadic `printf`.** `main` prints via C's `printf`, declared
> `llvm.func @printf(!llvm.ptr, ...) -> i32`, called with
> `llvm.call @printf(%fmt, %x) vararg(...)`. One gotcha baked into the example: C
> varargs promote `float` to `double`, so we `arith.extf` each `f32` to `f64`
> before the call (the format string is `"%f\n"`).

**Toolchain note.** OpenMP needs a runtime, and `llvm@20` doesn't bundle one, so
`build.sh` links Homebrew's `libomp` (`brew install libomp`) — for both the C
(`clang -fopenmp`) and the MLIR builds — and bakes its path in with `-Wl,-rpath`.
(A harmless linker warning about the macOS version may appear.)

**Run:** `cd 5_openmp && bash build.sh` — builds and runs the C version, then the
MLIR one; both print the same doubled array:

```
=== C version (clang -fopenmp) ===
2.000000 4.000000 6.000000 8.000000 10.000000 12.000000 14.000000 16.000000 18.000000 20.000000
=== running (input doubled, one value per line) ===
2.000000
4.000000
... (6, 8, 10, 12, 14, 16, 18)
20.000000
```

---

## Run everything

```bash
export PATH="/opt/homebrew/opt/llvm@20/bin:$PATH"   # llvm@20 is keg-only
brew install libomp                                  # for Step 5
# Python deps for the drivers:  pip install numpy

for d in 1_affine_matmul 2_affine_maps 3_affine_opts 4_convolution 5_openmp; do
  echo "=== $d ==="; ( cd "$d" && bash build.sh ) 2>&1 | tail -n 3
done
```

Each step writes its intermediates (`*_opt.mlir`, `*.ll`, `*.o`, `*.dylib`,
executables) into a local `build/` directory.

## Key takeaways

- **A loop over-specifies order.** Parallelization is recovering the independence
  the sequential syntax hid — and the affine dialect is what lets the compiler
  *see* that independence.
- **`affine` = analyzable loops.** Restricting indices to affine expressions buys
  the polyhedral model, and with it automatic parallelize / tile / fuse /
  interchange. Write `affine`, transform it, then `-lower-affine` to `scf` and
  reuse the Chapter 1–2 pipeline.
- **`affine.parallel` / `iter_args`** express independent iterations and
  loop-carried reductions; `affine.apply` does the affine index math.
- **OpenMP is the explicit alternative.** `omp.parallel` + `omp.wsloop` declare
  parallelism directly; `-convert-scf-to-openmp` gets you there from `scf.parallel`
  without hand-writing the dialect.
- **Two philosophies, one backend.** Automatic (affine) and manual (OpenMP) both
  lower through the same `scf`/`llvm` machinery you already know.

**Next:** Part 4 — Linear algebra and the `linalg` dialect (see
[`../reference/`](../reference/)).
