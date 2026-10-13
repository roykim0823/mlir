# Chapter H: Reproducing Halide Schedule

> **Goal:** Recreate a hand-tuned schedule from the Halide DSL (domain-specific language) — a channeled 2D convolution followed by ReLU (rectified linear unit) — as a Transform dialect script driving Linalg structured transformations, then lower the result all the way to a native executable and check it against a C reference — based on the official tutorial [Chapter H: Reproducing Halide Schedule](https://mlir.llvm.org/docs/Tutorials/transform/ChH/).

---

## 1. Overview

[Halide](http://halide-lang.org) separates *what* a program computes (the algorithm, written as "mathematical functions") from *how* it is computed (the schedule: loop splitting, reordering, vectorization, unrolling, fusion). The Transform dialect provides the same separation inside MLIR: the payload IR says what to compute, and a transform script — itself MLIR — says how to rewrite it. This chapter takes one real Halide program and its schedule and rebuilds both in MLIR:

1. Map the Halide algorithm onto Linalg operations on tensors (section 2).
2. Map each Halide scheduling directive onto a transform operation (section 3).
3. Rebuild the schedule step by step — tiling, fusion, reduction tiling, unrolling, vectorization, bufferization — looking at the real IR after each step (sections 4–7).
4. Replace the last, Halide-faithful steps with multi-dimensional vectors, which is what the final schedule in `test_Example/transform/ChH/full.mlir` does, and lower it to LLVM IR, compile it, run it, and inspect the generated AArch64 assembly (section 8).

This chapter has no C++ of its own: its tool is the stock `mlir-opt` of Homebrew LLVM 20, plus `mlir-translate`, `llc`, `clang` and `FileCheck` from the same installation. The step-by-step schedule lives in complete, runnable stage files under `stages/` (each one contains the previous one plus the new step), and `driver/` holds the C driver that calls the compiled convolution and the lowering pipeline to LLVM.

Sections 1–10 are the essential path; section 9 shows how to derive the schedule yourself, from the payload ops to the lowering passes. The appendices hold the reference material: how this chapter replaces the two test-only passes that the test's `RUN` line uses (A), the C driver and the LLVM lowering pipeline (B), why the ReLU stays scalar and how to vectorize it (C), and the script that runs every command in this README (D).

---

## 2. Channeled Convolution

### 2.1 Why Halide, and two ways to bring it to MLIR

The Transform dialect is a substrate for implementing "transformation directive" DSLs in MLIR. Such a DSL, at least its scheduling part, can target the transform operations, which the compiler later applies. Sets of transform operations — or even whole new dialects using the same interfaces and infrastructure — can be added to support a specific DSL for a particular scheduling model. Halide is the DSL that (re)popularized writing schedules separately from the algorithm, originally for image processing.

There are two ways to bring Halide to the Transform dialect:

* Create a new dialect for the computational part of Halide, and define transformations wrapped into transform operations for the scheduling part.
* Map the Halide abstractions onto existing MLIR abstractions, for both parts.

This chapter takes the second approach, because the computational part maps directly onto the structured operations of the Linalg dialect. It also shows how Linalg transformations on structured operations are similar to, or differ from, classic loop transformations.

### 2.2 The Halide program

The example is the 2D channeled convolution from Halide's [`conv_layer` application](https://github.com/halide/Halide/tree/294f80c49bf3bb8582446613c25fcce03b82bcd8/apps/conv_layer) (Halide C++, not this repo's code):

```cpp
// Sizes of the problem.
const int N = 5, CI = 128, CO = 128, W = 100, H = 80;

// Sized inputs. Note that the order of dimensions is
// inverted in Halide with respect to C++, so the last dimension
// in the list (N for input, CI for filter) is the least
// frequently varying. The C++ equivalent is input[N][H+2][W+2][CI].
Buffer<float, 4> input({CI, W+2, H+2, N}, "input");
Buffer<float, 4> filter({CO, 3, 3, CI}, "filter");
Buffer<float, 1> bias(std::vector<int>{CO}, "bias");

// ... data initialization happens here ...

// Declarations of "mathematical functions" for convolution and relu.
Func conv("conv"), relu("relu");

// Iterators/subscripts.
Var x("x"), y("y"), c("c"), n("n");

// 3D reduction domain (channels and 2 window dimensions),
// dimensions are later referred to as r.x, r.y, r.z.
RDom r(0, CI, 0, 3, 0, 3);

// Core convolution with the result initialized to the bias value.
// Note that the order of iterators is inverted in Halide DSL,
// i.e. `n` corresponds to the lest frequently-varying (outermost) dimension
// here and below.
conv(c, x, y, n) = bias(c);
conv(c, x, y, n) += filter(c, r.y, r.z, r.x) * input(r.x, x + r.y, y + r.z, n);

// ReLU rectification, an elementwise operation.
relu(c, x, y, n) = max(0, conv(c, x, y, n));
```

`conv` has a *pure* definition (`= bias(c)`) and an *update* (`+= ...`) over the reduction domain `r`: `r.x` runs over the 128 input channels, `r.y` and `r.z` over the 3×3 window.

### 2.3 The same computation in Linalg

The program converts almost directly to Linalg operations on tensors, which are conceptually closest to Halide's "mathematical functions" and are where most transformations are available. The test file declares the fixed-size types first:

***test_Example/transform/ChH/full.mlir***
```mlir
// Fixed-size tensor types to be used in convolution.
// Named sizes are: N=5 OH=80 OW=100 F=C=128 KH=KW=3.
// Input is NHWC.
// Filter is CHWF.
// Ouptut is NHWF.
!tinput = tensor<5x82x102x128xf32>
!tfilter = tensor<128x3x3x128xf32>
!tbias = tensor<128xf32>
!toutput = tensor<5x80x100x128xf32>
```

and then the body of `@conv`:

***test_Example/transform/ChH/full.mlir***
```mlir
  // Bias. Using a named Linalg operation for brevity.
  %bias_init = tensor.empty() : !toutput
  %biased = linalg.broadcast ins(%bias : !tbias)
    outs(%bias_init : !toutput) dimensions = [0, 1, 2]

  // Convolution proper. While Linalg has named operations for 2D convolutions,
  // the one in the Halide example has an uncommon order of filter dimensions
  // and is not supported. It also takes the fitler as first argument. This
  // code recreates it faithfully using the generic form.
  %convolved = linalg.generic {
    iterator_types = ["parallel", "parallel", "parallel", "parallel",
                      "reduction", "reduction", "reduction"],
    indexing_maps = [
      affine_map<(n, y, x, c, rz, ry, rx) -> (rx, rz, ry, c)>,
      affine_map<(n, y, x, c, rz, ry, rx) -> (n, y+rz, x+ry, rx)>,
      affine_map<(n, y, x, c, rz, ry, rx) -> (n, y, x, c)>
    ]
  } ins(%filter, %input: !tfilter, !tinput) outs(%biased : !toutput) {
  ^bb0(%in: f32, %f: f32, %b: f32):
    // Note the fastmath attributes that allow operations to be recombined into
    //   %0 = math.fma %in, %f, %b : f32
    // later on and to reorder reductions.
    %m1 = arith.mulf %in, %f  {fastmath = #arith.fastmath<fast>} : f32
    %0 = arith.addf %b, %m1  {fastmath = #arith.fastmath<fast>} : f32
    linalg.yield %0 : f32
  } -> !toutput

  // ReLU is just a max(0, x).
  %c0 = arith.constant 0.0 : f32
  %relued = linalg.generic {
    iterator_types = ["parallel", "parallel", "parallel", "parallel"],
    indexing_maps = [
      affine_map<(d0, d1, d2, d3) -> ()>,
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
    ]
  } ins(%c0, %convolved : f32, !toutput)
    outs(%output : !toutput) {
  ^bb0(%cst: f32, %in: f32, %out: f32):
    %0 = llvm.intr.maxnum(%cst, %in) : (f32, f32) -> f32
    linalg.yield %0 : f32
  } -> !toutput

  return %relued : !toutput
```

Reading it against the Halide program:

- **Bias** is the pure part of `conv`: `linalg.broadcast` copies the 128-element `%bias` along the three new dimensions `[0, 1, 2]` (n, y, x) into a fresh `tensor.empty`.
- **Convolution** is the update. Its seven implicit loops are `(n, y, x, c, rz, ry, rx)`; the first four are `parallel` and the last three, the reduction domain, are `reduction`. The indexing maps say which elements each loop iteration touches: the filter at `(rx, rz, ry, c)`, the input at `(n, y+rz, x+ry, rx)` — the window slides over the input — and the accumulator at `(n, y, x, c)`. Its `outs` is `%biased`, so the accumulation starts from the bias, exactly like Halide's `=` followed by `+=`. The `fastmath<fast>` flags allow the multiply and add to be fused into an FMA (fused multiply-add) later and the reduction to be reordered.
- **ReLU** is `max(0, x)` as a four-dimensional elementwise `linalg.generic`; the scalar `0.0` is passed with the empty indexing map `()`, i.e. the same value for every iteration.

The function's arguments carry `bufferization.*` attributes that tell bufferization (section 7.1) the buffers can be used as they are, and `llvm.emit_c_interface` asks for a C-callable wrapper, `_mlir_ciface_conv`, which the driver in section 8.3 calls.

### 2.4 Fully fused versus fully distributed loops

In Halide, a function such as `conv` may consist of two parts, a "functional" initialization and an in-place update for reductions. They are two C++ statements in the embedded DSL, but one object internally. Linalg has no such capability, so the initialization and the update are two distinct, unconnected Linalg operations. Furthermore, Halide's `x`, `y`, `c`, `n` variables are implicit loops over the corresponding objects, and functions sharing these variables in their definitions also share the loops: **the loop equivalent of a Halide definition starts in a fully fused form**. Linalg is the opposite: each structured operation is its own loop nest, **a fully distributed form**. This decides how the schedule is built later on.

The loop structure of the Halide computation resembles the following (adapted from Halide's debug dump with `HL_DEBUG_CODEGEN=1`; pseudo-code):

```python
for n
  for y
    for x
      for c
        conv[n, y, x, c] = bias[c]
        for rz
          for ry
            for rx
              conv[n, y, x, c] += filter[rx, rz, ry, c] * input[n, y+rz, x+ry, rx]
        relu[n, y, x, c] = max(0, conv[n, y, x, c])
```

The loop structure of the Linalg computation is what you get by generalizing the named op, bufferizing, and turning every structured op into loops:

```bash
cd /Users/roy/study/mlir/transform/ChH
sed -n '8,86p' ../../test_Example/transform/ChH/full.mlir |
  mlir-opt --linalg-generalize-named-ops --empty-tensor-to-alloc-tensor \
           --one-shot-bufferize --convert-linalg-to-loops
```

Real output **[1]** (the function signature is abridged):

```mlir
#map = affine_map<(d0, d1) -> (d0 + d1)>
module {
  func.func @conv(%arg0: tensor<5x82x102x128xf32> {...}, ...) -> tensor<5x80x100x128xf32> attributes {llvm.emit_c_interface} {
    %c3 = arith.constant 3 : index
    %c128 = arith.constant 128 : index
    %c100 = arith.constant 100 : index
    %c80 = arith.constant 80 : index
    %c1 = arith.constant 1 : index
    %c5 = arith.constant 5 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = bufferization.to_memref %arg0 : tensor<5x82x102x128xf32> to memref<5x82x102x128xf32, strided<[?, ?, ?, ?], offset: ?>>
    %1 = bufferization.to_memref %arg1 : tensor<128x3x3x128xf32> to memref<128x3x3x128xf32, strided<[?, ?, ?, ?], offset: ?>>
    %2 = bufferization.to_memref %arg2 : tensor<128xf32> to memref<128xf32, strided<[?], offset: ?>>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<5x80x100x128xf32>
    scf.for %arg4 = %c0 to %c5 step %c1 {
      scf.for %arg5 = %c0 to %c80 step %c1 {
        scf.for %arg6 = %c0 to %c100 step %c1 {
          scf.for %arg7 = %c0 to %c128 step %c1 {
            %4 = memref.load %2[%arg7] : memref<128xf32, strided<[?], offset: ?>>
            memref.store %4, %alloc[%arg4, %arg5, %arg6, %arg7] : memref<5x80x100x128xf32>
          }
        }
      }
    }
    scf.for %arg4 = %c0 to %c5 step %c1 {
      scf.for %arg5 = %c0 to %c80 step %c1 {
        scf.for %arg6 = %c0 to %c100 step %c1 {
          scf.for %arg7 = %c0 to %c128 step %c1 {
            scf.for %arg8 = %c0 to %c3 step %c1 {
              scf.for %arg9 = %c0 to %c3 step %c1 {
                scf.for %arg10 = %c0 to %c128 step %c1 {
                  %4 = memref.load %1[%arg10, %arg8, %arg9, %arg7] : memref<128x3x3x128xf32, strided<[?, ?, ?, ?], offset: ?>>
                  %5 = affine.apply #map(%arg5, %arg8)
                  %6 = affine.apply #map(%arg6, %arg9)
                  %7 = memref.load %0[%arg4, %5, %6, %arg10] : memref<5x82x102x128xf32, strided<[?, ?, ?, ?], offset: ?>>
                  %8 = memref.load %alloc[%arg4, %arg5, %arg6, %arg7] : memref<5x80x100x128xf32>
                  %9 = arith.mulf %4, %7 fastmath<fast> : f32
                  %10 = arith.addf %8, %9 fastmath<fast> : f32
                  memref.store %10, %alloc[%arg4, %arg5, %arg6, %arg7] : memref<5x80x100x128xf32>
                }
              }
            }
          }
        }
      }
    }
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<5x80x100x128xf32>
    scf.for %arg4 = %c0 to %c5 step %c1 {
      scf.for %arg5 = %c0 to %c80 step %c1 {
        scf.for %arg6 = %c0 to %c100 step %c1 {
          scf.for %arg7 = %c0 to %c128 step %c1 {
            %4 = memref.load %alloc[%arg4, %arg5, %arg6, %arg7] : memref<5x80x100x128xf32>
            %5 = llvm.intr.maxnum(%cst, %4) : (f32, f32) -> f32
            memref.store %5, %alloc_0[%arg4, %arg5, %arg6, %arg7] : memref<5x80x100x128xf32>
          }
        }
      }
    }
    %3 = bufferization.to_tensor %alloc_0 : memref<5x80x100x128xf32> to tensor<5x80x100x128xf32>
    return %3 : tensor<5x80x100x128xf32>
  }
}
```

- `sed -n '8,86p'` cuts the payload — the type aliases and `@conv` — out of the test file; lines 88–377 are the transform script, used from section 8 on.
- Three separate loop nests, one per Linalg op: the bias broadcast (4 loops), the convolution (7 loops, with `affine.apply #map` computing `y+rz` and `x+ry`) and the ReLU (4 loops), each writing a whole `5x80x100x128` buffer (`%alloc`, `%alloc_0`) before the next one reads it. In pseudo-code:

```python
for n
  for y
    for x
      for c
        init[n, y, x, c] = bias[c]
for n
  for y
    for x
      for c
        for rz
          for ry
            for rx
              conv[n, y, x, c] += filter[rx, rz, ry, c] * input[n, y+rz, x+ry, rx]
for n
  for y
    for x
      for c
        relu[n, y, x, c] = max(0, conv[n, y, x, c])
```

The schedule therefore has to *fuse* where Halide has to *split*: everything Halide gets for free by sharing loop variables, Linalg gets by producer/consumer fusion (section 4.3).

---

## 3. Mapping Halide Scheduling Primitives to Linalg Structured Transforms

### 3.1 The Halide schedule

The complete Halide schedule of the example is (Halide C++, not this repo's code):

```cpp
Var co, ci, xo, xi;
relu.split(c, co, ci, vec * tile_w)
  .split(x, xo, xi, tile_h)
  .reorder(ci, xi, xo, y, n, co)
  .vectorize(ci, vec)
  .unroll(ci)
  .unroll(xi)
  .parallel(y)
  .parallel(n)
  .parallel(co);

conv.compute_at(relu, xo)
  .vectorize(c, vec)
  .unroll(c)
  .unroll(x)
  .unroll(y)
  .update()
  .reorder(c, x, y, r.x, r.y, r.z, n)
  .vectorize(c, vec)
  .unroll(c)
  .unroll(x)
  .unroll(y)
  .unroll(r.x, 2);
```

with `tile_w = 4`, `tile_h = 5`, `vec = 16` (16 floats = one 512-bit AVX-512 register). Only the case without parallelization is considered, to avoid comparing the different parallel runtimes Halide and MLIR use, so the `.parallel(...)` directives are dropped (the comment in `full.mlir`, lines 91–117, lists the schedule without them). What remains is a sequence of loop manipulations, unrolling and vectorization.

### 3.2 Directive by directive

* **`split`** decomposes a loop dimension into two immediately nested loops, the inner one with at most the given number of iterations. This is loop *strip-mining*, or a degenerate case of tiling a single dimension, which any of the `transform.structured.tile_*` operations can do. This chapter uses `transform.structured.tile_using_forall`: `scf.forall` is the loop best supported by bufferization, and it can become a parallel loop later. Unlike Halide, tiling doesn't add new dimensions to the original operation; it creates a loop around it and rewrites the operation itself to work on a subset (slice) of the original data.
* **`reorder`** rearranges loops arbitrarily. In Linalg, loops are implicit and are meant to stay implicit as long as possible, to target microkernels. The order of the implicit loops of a `linalg.generic` can be changed with `transform.structured.interchange`, but not on named operations, which must first be "generalized" with `transform.structured.generalize` (section 3.3 shows both). And interchange can only reorder *implicit* dimensions, not the explicit loops that tiling has materialized — those can no longer be "folded" back into the operation. The schedule turns this around: it **materializes the loops directly in the desired order**, by "tiling" with size 1.
* **`vectorize`** says the given dimension should be vectorized with the given factor; if the loop is longer than the factor, it is effectively split in two and the inner part vectorized. Linalg vectorization, in contrast, is a global transformation applied to all suitable operations in, e.g., a function: `transform.structured.vectorize_children_and_apply_patterns`. It relies on MLIR's multidimensional vectors to map multidimensional tensors directly; those are decomposed into operations on smaller, hardware-compatible vectors during lowering.
* **`unroll`** unrolls a loop, fully or by the given factor: `transform.loop.unroll`.
* **`compute_at`** says a function's value must be computed inside the given loop of another function. Depending on how the loops surrounding the two functions relate, this is either loop distribution or producer/consumer fusion. Since Linalg starts fully distributed (section 2.4), it is a sequence of `transform.structured.fuse_into_containing_op`, on `scf.forall` loops materialized by tiling beforehand.

### 3.3 Reordering implicit loops: generalize, then interchange

`transform.structured.interchange` permutes the implicit loops of a `linalg.generic`. Applied to the named `linalg.broadcast`, it is rejected:

***stages/reorder-named.mlir***
```mlir
// Interchange reorders the implicit loops of a linalg.generic only: applied
// to the named linalg.broadcast, it fails.
module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %bias = transform.structured.match ops{["linalg.broadcast"]} in %arg0
      : (!transform.any_op) -> !transform.any_op
    //                                             [n  y  x  c] -> [c  n  y  x]
    transform.structured.interchange %bias iterator_interchange = [3, 0, 1, 2]
      : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
```

```bash
cd /Users/roy/study/mlir/transform/ChH
sed -n '8,86p' ../../test_Example/transform/ChH/full.mlir |
  mlir-opt --transform-preload-library=transform-library-paths=stages/reorder-named.mlir \
           --transform-interpreter
```

Real output **[2]** (exit status 1):

```
stages/reorder-named.mlir:8:5: error: transform applied to the wrong op kind
    transform.structured.interchange %bias iterator_interchange = [3, 0, 1, 2]
    ^
<stdin>:37:13: note: when applied to this op
  %biased = linalg.broadcast ins(%bias : !tbias)
            ^
```

`--transform-preload-library` parses the script file and makes its named sequences available to `--transform-interpreter`, which runs `@__transform_main` on the payload read from stdin; section 4.1 explains why every stage is run this way. The error points at the transform op and, in a note, at the payload op it was applied to.

Generalizing first turns the broadcast into an equivalent `linalg.generic`, which interchange accepts:

***stages/reorder-generic.mlir***
```mlir
// Generalize the named linalg.broadcast first; the resulting linalg.generic
// can then have its implicit loops reordered.
module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %bias = transform.structured.match ops{["linalg.broadcast"]} in %arg0
      : (!transform.any_op) -> !transform.any_op
    %generic = transform.structured.generalize %bias
      : (!transform.any_op) -> !transform.any_op
    //                                             [n  y  x  c] -> [c  n  y  x]
    transform.structured.interchange %generic iterator_interchange = [3, 0, 1, 2]
      : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
```

```bash
cd /Users/roy/study/mlir/transform/ChH
sed -n '8,86p' ../../test_Example/transform/ChH/full.mlir |
  mlir-opt --transform-preload-library=transform-library-paths=stages/reorder-generic.mlir \
           --transform-interpreter |
  mlir-opt --empty-tensor-to-alloc-tensor --one-shot-bufferize --convert-linalg-to-loops |
  grep -m4 'scf.for'
```

Real output **[3]**:

```mlir
    scf.for %arg4 = %c0 to %c128 step %c1 {
      scf.for %arg5 = %c0 to %c5 step %c1 {
        scf.for %arg6 = %c0 to %c80 step %c1 {
          scf.for %arg7 = %c0 to %c100 step %c1 {
```

The first loop nest — the bias — now runs `c` (128) outermost, then `n` (5), `y` (80), `x` (100), where section 2.4 had `n, y, x, c`. That is all interchange can do: permute the loops *inside* one operation. Moving `co` outside the `n, y, x` loops of *all three* operations, as the Halide `reorder` requires, needs explicit loops, which is why the schedule materializes them by tiling instead.

---

## 4. Recreating the Loop Structure

### 4.1 Running a stage

Each step below is a stage file under `stages/`: a complete transform module whose `@__transform_main` contains everything up to that step (so `diff` between two consecutive stages shows exactly one step). Every stage is applied to the same payload, cut out of the test file, with one command:

```sh
sed -n '8,86p' ../../test_Example/transform/ChH/full.mlir |
  mlir-opt --transform-preload-library=transform-library-paths=stages/stepK.mlir \
           --transform-interpreter
```

(a template, not a numbered command). Keeping the script in its own file, preloaded as a *library*, keeps it out of the output: only the transformed payload is printed. If the script were embedded in the payload file, as in `full.mlir`, the interpreter would leave the script in the output, next to the payload (appendix A.1).

Most outputs below are long, so from step 3 on they are reduced to a **loop outline** by two filters appended to the command:

```sh
grep -E '= (scf|linalg|tensor\.empty)' | sed -E 's/ (ins|outs|iter_args|shared_outs|\{indexing_maps)[ (=].*//'
```

The `grep` keeps the lines that define loops, Linalg operations and empty tensors, and the `sed` cuts each line before its operand list, so what is left is the loop nest with its bounds and the operations inside it, indented by nesting depth.

### 4.2 Step 1: tile the ReLU — `split` and `reorder`

The first three directives for `relu` in the Halide schedule aim at this loop structure (pseudo-code):

```python
for co
  for n
    for y
      for xo
        for xi
          for ci
            relu[n, y, xo*tile_h + xi, co*tile_w*vec + ci] = ...
```

The outer part of `c` is hoisted out of all the surrounding loops. The implicit loop order of the operation is `n, y, x, c`, so the `co` loop must be materialized **first** to get it outermost; the remaining dimensions are already in order and can be materialized as loops in one transformation:

***stages/step1-tile-relu.mlir***
```mlir
// Stage 1: materialize the co loop, then the n, y, xo loops, around the ReLU.
module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %bias = transform.structured.match ops{["linalg.broadcast"]} in %arg0
      : (!transform.any_op) -> !transform.any_op
    %generics = transform.structured.match ops{["linalg.generic"]} in %arg0
      : (!transform.any_op) -> !transform.any_op
    %conv, %relu = transform.split_handle %generics
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    //                                                             [n  y  x  c]
    %relu2, %co = transform.structured.tile_using_forall %relu
                                                        tile_sizes [0, 0, 0, 64]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %relu3, %n_y_xo = transform.structured.tile_using_forall %relu2
                                                        tile_sizes [1, 1, 5, 0]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
```

- `transform.structured.match` finds payload operations by name and returns a handle to them. There are two `linalg.generic`s, so `transform.split_handle` splits that handle into one per op, in IR order: the convolution, then the ReLU. Matching by name is simplistic and works when the IR has few operations; larger cases use the `transform.match.*` operations.
- A tile size of 0 leaves a dimension untiled; a non-zero size `s` makes a loop over that dimension with step `s`. `[0, 0, 0, 64]` makes the `co` loop: 128 / 64 = 2 iterations, each on 64 channels (64 = `vec * tile_w` = 16 × 4). `[1, 1, 5, 0]` then makes one three-dimensional loop over `n` (5 / 1), `y` (80 / 1) and `xo` (100 / 5 = 20).
- `tile_using_forall` returns the tiled op and the new `scf.forall`; the order of those results is `(tiled op, loop)`.

```bash
cd /Users/roy/study/mlir/transform/ChH
sed -n '8,86p' ../../test_Example/transform/ChH/full.mlir |
  mlir-opt --transform-preload-library=transform-library-paths=stages/step1-tile-relu.mlir \
           --transform-interpreter
```

Real output **[4]** (the function signature is abridged):

```mlir
#map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d6, d4, d5, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
#map3 = affine_map<(d0) -> (d0 * 64)>
#map4 = affine_map<(d0) -> (d0 * 5)>
#map5 = affine_map<(d0, d1, d2, d3) -> ()>
#map6 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module {
  func.func @conv(%arg0: tensor<5x82x102x128xf32> {...}, ...) -> tensor<5x80x100x128xf32> attributes {llvm.emit_c_interface} {
    %0 = tensor.empty() : tensor<5x80x100x128xf32>
    %broadcasted = linalg.broadcast ins(%arg2 : tensor<128xf32>) outs(%0 : tensor<5x80x100x128xf32>) dimensions = [0, 1, 2] 
    %1 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg1, %arg0 : tensor<128x3x3x128xf32>, tensor<5x82x102x128xf32>) outs(%broadcasted : tensor<5x80x100x128xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %3 = arith.mulf %in, %in_0 fastmath<fast> : f32
      %4 = arith.addf %out, %3 fastmath<fast> : f32
      linalg.yield %4 : f32
    } -> tensor<5x80x100x128xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %2 = scf.forall (%arg4) in (2) shared_outs(%arg5 = %arg3) -> (tensor<5x80x100x128xf32>) {
      %3 = affine.apply #map3(%arg4)
      %extracted_slice = tensor.extract_slice %1[0, 0, 0, %3] [5, 80, 100, 64] [1, 1, 1, 1] : tensor<5x80x100x128xf32> to tensor<5x80x100x64xf32>
      %extracted_slice_0 = tensor.extract_slice %arg5[0, 0, 0, %3] [5, 80, 100, 64] [1, 1, 1, 1] : tensor<5x80x100x128xf32> to tensor<5x80x100x64xf32>
      %4 = scf.forall (%arg6, %arg7, %arg8) in (5, 80, 20) shared_outs(%arg9 = %extracted_slice_0) -> (tensor<5x80x100x64xf32>) {
        %5 = affine.apply #map4(%arg8)
        %extracted_slice_1 = tensor.extract_slice %extracted_slice[%arg6, %arg7, %5, 0] [1, 1, 5, 64] [1, 1, 1, 1] : tensor<5x80x100x64xf32> to tensor<1x1x5x64xf32>
        %extracted_slice_2 = tensor.extract_slice %arg9[%arg6, %arg7, %5, 0] [1, 1, 5, 64] [1, 1, 1, 1] : tensor<5x80x100x64xf32> to tensor<1x1x5x64xf32>
        %6 = linalg.generic {indexing_maps = [#map5, #map6, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst, %extracted_slice_1 : f32, tensor<1x1x5x64xf32>) outs(%extracted_slice_2 : tensor<1x1x5x64xf32>) {
        ^bb0(%in: f32, %in_3: f32, %out: f32):
          %7 = llvm.intr.maxnum(%in, %in_3) : (f32, f32) -> f32
          linalg.yield %7 : f32
        } -> tensor<1x1x5x64xf32>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %6 into %arg9[%arg6, %arg7, %5, 0] [1, 1, 5, 64] [1, 1, 1, 1] : tensor<1x1x5x64xf32> into tensor<5x80x100x64xf32>
        }
      }
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %4 into %arg5[0, 0, 0, %3] [5, 80, 100, 64] [1, 1, 1, 1] : tensor<5x80x100x64xf32> into tensor<5x80x100x128xf32>
      }
    }
    return %2 : tensor<5x80x100x128xf32>
  }
}
```

- `scf.forall (%arg4) in (2)` is `co` and `scf.forall (%arg6, %arg7, %arg8) in (5, 80, 20)` is `n, y, xo`. `affine.apply #map3`/`#map4` scale the induction variables by the tile sizes (`d0 * 64`, `d0 * 5`).
- `scf.forall` has *tensor-insertion* semantics: `shared_outs(%arg5 = %arg3)` names the tensor the loop produces, each iteration takes its part with `tensor.extract_slice` and gives back its result with `tensor.parallel_insert_slice` in the `scf.forall.in_parallel` terminator.
- The ReLU itself still exists, unchanged except for its operand types: it now works on `tensor<1x1x5x64xf32>`. Its implicit dimensions are `[ni=0:1, yi=0:1, xi=0:5, ci=0:64]`: **structured tiling never drops the dimensions that became trivial** (size 1), so the structure is technically `(co, n, y, xo, [ni, yi, xi, ci])`, brackets marking the loops still implicit in the operation.
- The bias and the convolution (`linalg.broadcast` and the first `linalg.generic`) are untouched and still compute the whole tensor outside the loops.

### 4.3 Step 2: fuse the convolution and the bias — `compute_at`

The next loop-restructuring directives are `compute_at` and `reorder` on `conv`. They must come before unrolling and vectorization destroy the loops, and they aim at the final loop structure (pseudo-code):

```python
for co
  for n
    for y
      for xo
        for xi
          for ci
            conv[n, y, x*tile_h + xi, co*tile_w*vec + ci] = ...
        for rz
          for ry
            for rx
              for xi
                for ci
                  conv[n, y, x*tile_h + xi, co*tile_w*vec + ci] += ...
        for xi
          for ci
            relu[n, y, xo*tile_h + xi, co*tile_w*vec + ci] = ...
```

In practice this means fusing the convolution's initialization and update into the `co` and `n, y, xo` loops that tiling created. Structured-op fusion fuses the *producer* of a value into its *consumer*, so fusion happens in two stages:

* first the convolution update is fused into the ReLU that uses it and already has loops;
* then the bias initialization is fused into the convolution+ReLU loop nest.

Each stage fuses the operation into the outer loop, then into the inner one:

***stages/step2-fuse.mlir***
```mlir
// Stage 2: stage 1 + fuse the convolution, then the bias, into the ReLU loops.
...
    %conv2, %co2 = transform.structured.fuse_into_containing_op %conv into %co
      : (!transform.any_op, !transform.any_op)
      -> (!transform.any_op, !transform.any_op)
    %conv3, %n_y_xo2 = transform.structured.fuse_into_containing_op %conv2
      into %n_y_xo
      : (!transform.any_op, !transform.any_op)
      -> (!transform.any_op, !transform.any_op)

    %bias2, %co3 = transform.structured.fuse_into_containing_op %bias into %co2
      : (!transform.any_op, !transform.any_op)
      -> (!transform.any_op, !transform.any_op)
    %bias3, %n_y_xo3 = transform.structured.fuse_into_containing_op %bias2
      into %n_y_xo2
      : (!transform.any_op, !transform.any_op)
      -> (!transform.any_op, !transform.any_op)

    // Fusion clones the producer into the loop; an empty pattern sweep runs
    // dead-code elimination on the now-unused originals.
    %f00 = transform.structured.match ops{["func.func"]} in %arg0
      : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %f00 {
    } : !transform.any_op
    transform.yield
  }
}
```

- `fuse_into_containing_op %producer into %loop` returns the fused op and the loop. It consumes both handles and produces new ones, which is why each step chains `%conv2`, `%co2`, ... into the next.
- Fusion mechanically clones the producer into the loop and leaves the original in place. The original is then dead — no uses, no side effects — and the empty `transform.apply_patterns` block removes it: a pattern-driven rewrite always runs DCE (dead-code elimination), even with no patterns. This step is optional (the final IR is the same without it) but makes the intermediate IR readable.

```bash
cd /Users/roy/study/mlir/transform/ChH
sed -n '8,86p' ../../test_Example/transform/ChH/full.mlir |
  mlir-opt --transform-preload-library=transform-library-paths=stages/step2-fuse.mlir \
           --transform-interpreter
```

Real output **[5]** (the function signature is abridged):

```mlir
#map = affine_map<(d0) -> (d0 * 64)>
#map1 = affine_map<(d0) -> (d0 * 5)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d6, d4, d5, d3)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>
#map4 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
#map5 = affine_map<(d0, d1, d2, d3) -> ()>
#map6 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module {
  func.func @conv(%arg0: tensor<5x82x102x128xf32> {...}, ...) -> tensor<5x80x100x128xf32> attributes {llvm.emit_c_interface} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<5x80x100x128xf32>
    %1 = scf.forall (%arg4) in (2) shared_outs(%arg5 = %arg3) -> (tensor<5x80x100x128xf32>) {
      %2 = affine.apply #map(%arg4)
      %3 = affine.apply #map(%arg4)
      %extracted_slice = tensor.extract_slice %arg1[0, 0, 0, %3] [128, 3, 3, 64] [1, 1, 1, 1] : tensor<128x3x3x128xf32> to tensor<128x3x3x64xf32>
      %4 = affine.apply #map(%arg4)
      %5 = affine.apply #map(%arg4)
      %extracted_slice_0 = tensor.extract_slice %arg2[%4] [64] [1] : tensor<128xf32> to tensor<64xf32>
      %extracted_slice_1 = tensor.extract_slice %0[0, 0, 0, %5] [5, 80, 100, 64] [1, 1, 1, 1] : tensor<5x80x100x128xf32> to tensor<5x80x100x64xf32>
      %extracted_slice_2 = tensor.extract_slice %arg5[0, 0, 0, %2] [5, 80, 100, 64] [1, 1, 1, 1] : tensor<5x80x100x128xf32> to tensor<5x80x100x64xf32>
      %6 = scf.forall (%arg6, %arg7, %arg8) in (5, 80, 20) shared_outs(%arg9 = %extracted_slice_2) -> (tensor<5x80x100x64xf32>) {
        %7 = affine.apply #map1(%arg8)
        %8 = affine.apply #map1(%arg8)
        %extracted_slice_3 = tensor.extract_slice %arg0[%arg6, %arg7, %8, 0] [1, 3, 7, 128] [1, 1, 1, 1] : tensor<5x82x102x128xf32> to tensor<1x3x7x128xf32>
        %9 = affine.apply #map1(%arg8)
        %extracted_slice_4 = tensor.extract_slice %extracted_slice_1[%arg6, %arg7, %9, 0] [1, 1, 5, 64] [1, 1, 1, 1] : tensor<5x80x100x64xf32> to tensor<1x1x5x64xf32>
        %broadcasted = linalg.broadcast ins(%extracted_slice_0 : tensor<64xf32>) outs(%extracted_slice_4 : tensor<1x1x5x64xf32>) dimensions = [0, 1, 2] 
        %10 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%extracted_slice, %extracted_slice_3 : tensor<128x3x3x64xf32>, tensor<1x3x7x128xf32>) outs(%broadcasted : tensor<1x1x5x64xf32>) {
        ^bb0(%in: f32, %in_6: f32, %out: f32):
          %12 = arith.mulf %in, %in_6 fastmath<fast> : f32
          %13 = arith.addf %out, %12 fastmath<fast> : f32
          linalg.yield %13 : f32
        } -> tensor<1x1x5x64xf32>
        %extracted_slice_5 = tensor.extract_slice %arg9[%arg6, %arg7, %7, 0] [1, 1, 5, 64] [1, 1, 1, 1] : tensor<5x80x100x64xf32> to tensor<1x1x5x64xf32>
        %11 = linalg.generic {indexing_maps = [#map5, #map6, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst, %10 : f32, tensor<1x1x5x64xf32>) outs(%extracted_slice_5 : tensor<1x1x5x64xf32>) {
        ^bb0(%in: f32, %in_6: f32, %out: f32):
          %12 = llvm.intr.maxnum(%in, %in_6) : (f32, f32) -> f32
          linalg.yield %12 : f32
        } -> tensor<1x1x5x64xf32>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %11 into %arg9[%arg6, %arg7, %7, 0] [1, 1, 5, 64] [1, 1, 1, 1] : tensor<1x1x5x64xf32> into tensor<5x80x100x64xf32>
        }
      }
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %6 into %arg5[0, 0, 0, %2] [5, 80, 100, 64] [1, 1, 1, 1] : tensor<5x80x100x64xf32> into tensor<5x80x100x128xf32>
      }
    }
    return %1 : tensor<5x80x100x128xf32>
  }
}
```

- Everything now happens inside the two loops: per `(co, n, y, xo)` iteration, `linalg.broadcast` initializes a `1x1x5x64` tile with 64 bias values, the convolution `linalg.generic` accumulates into it, and the ReLU reads it. No full-size intermediate tensor is left: the `tensor.empty` remains only as the (never-read) destination the bias slice is taken from.
- Fusion computed each operand slice by inverting the indexing maps. For a `1x1x5x64` output tile, the filter slice is `128x3x3x64` (all channels and window positions, 64 output channels) and the input slice is `[1, 3, 7, 128]` — 3 rows and 7 columns, because `y+rz` and `x+ry` reach 2 beyond the 1×5 tile. Neighboring tiles overlap by 2 columns, so those input elements are read again in the next iteration: fusion trades recomputation and re-reading for not storing the intermediate.

**Producer/consumer fusion does what Halide's shared loop variables give for free, and it works because each fused op recomputes exactly the slice its consumer needs** — the slice is derived from the indexing maps, not from loop analysis.

### 4.4 Step 3: materialize the reduction loops

The `rz, ry, rx` loops must go outside the "tile" loops `xi, ci`. They can be materialized from the convolution by tiling — but they are **reduction** loops, and materializing them as intrinsically parallel `scf.forall` loops would be wrong. The dedicated *reduction tiling* produces sequential `scf.for` loops instead. (`scf.forall` can express parallel reductions too, but its transformation doesn't handle reductions along more than one dimension.)

***stages/step3-tile-reduction.mlir***
```mlir
// Stage 3: stage 2 + materialize the rz, ry, rx reduction loops as scf.for.
...
    //            n  y  x  c  rz ry rx
    %red_fill, %conv4, %comb, %rz_ry_rx
      = transform.structured.tile_reduction_using_for %conv3
        by tile_sizes=[0, 0, 0, 0, 1, 1, 1]
      : (!transform.any_op)
      -> (!transform.any_op, !transform.any_op, !transform.any_op,
          !transform.any_op)
    transform.yield
  }
}
```

The four results are, in this order: the op that fills the partial-result buffer with the neutral element, the tiled ("split") op inside the loops, the combining op after the loops, and the loops.

```bash
cd /Users/roy/study/mlir/transform/ChH
sed -n '8,86p' ../../test_Example/transform/ChH/full.mlir |
  mlir-opt --transform-preload-library=transform-library-paths=stages/step3-tile-reduction.mlir \
           --transform-interpreter |
  grep -E '= (scf|linalg|tensor\.empty)' | sed -E 's/ (ins|outs|iter_args|shared_outs|\{indexing_maps)[ (=].*//'
```

Real output **[6]**:

```mlir
    %0 = tensor.empty() : tensor<5x80x100x128xf32>
    %1 = scf.forall (%arg4) in (2)
      %6 = scf.forall (%arg6, %arg7, %arg8) in (5, 80, 20)
        %broadcasted = linalg.broadcast
        %10 = tensor.empty() : tensor<1x1x5x64x1x1x1xf32>
        %11 = linalg.fill
        %12 = scf.for %arg10 = %c0 to %c3 step %c1
          %14 = scf.for %arg12 = %c0_6 to %c3_8 step %c1_9
            %15 = scf.for %arg14 = %c0_7 to %c128 step %c1_10
              %16 = linalg.generic
        %reduced = linalg.reduce
        %13 = linalg.generic
```

Reduction tiling does more than make loops:

- **`%11 = linalg.fill`** initializes a *partial-result* tensor `tensor<1x1x5x64x1x1x1xf32>` — the output tile extended with one dimension per tiled reduction dimension, of the tile size (here 1) — with `0.0`, the neutral element of addition.
- **The three `scf.for`** (3, 3, 128 iterations) carry that tensor through `iter_args` and contain the convolution `linalg.generic`, whose seven iterator types have all become `parallel`: inside the loops, each iteration accumulates into its own partial result.
- **`%reduced = linalg.reduce`**, the *combiner*, adds the partial results (dimensions 4, 5, 6) into the bias-initialized tile after the loops.

In general, the transformed code performs `tile_size` partial reductions of `N / tile_size` elements each — potentially in parallel, since the dimension kind changed — and then a final reduction of the partial results. With `tile_size = 1` along all three dimensions, the generated loops do the whole reduction, and the combiner only adds the result to the initial (bias) value. This changes the order of floating-point additions, as reduction tiling with non-unit sizes would, and may change the result's last bits, since floating-point addition is not associative; the `fastmath` flags explicitly allow it. Halide also emits LLVM IR with full `fastmath` flags.

### 4.5 Step 4: materialize the `xi` and `ci` loops

The innermost loops `xi` and `ci` are still implicit. The next step is vectorization along `ci`, and Linalg vectorization doesn't pick a loop and a vector size: it **substitutes multidimensional vector types for tensor types** and updates the operations accordingly. So the tensor type must not become trivial (size 1): it must keep a `vector_size`-sized dimension along `ci`. That is what tiling with `vector_size` = 16 in that dimension, and 1 in `xi`, gives — on all four computations, the combiner from reduction tiling included:

***stages/step4-tile-xi-ci.mlir***
```mlir
// Stage 4: stage 3 + materialize the xi and ci loops of every computation.
...
    // Materialize xi (size 1) and ci (size 16 = the vector length) as scf.for
    // loops around each of the four computations.
    //                                                                n  y  xi ci
    %conv5, %conv_xi, %conv_ci = transform.structured.tile_using_for %conv4
                                                           tile_sizes [0, 0, 1, 16]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %bias4, %bias_xi, %bias_ci = transform.structured.tile_using_for %bias3
                                                           tile_sizes [0, 0, 1, 16]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %relu4, %relu_xi, %relu_ci = transform.structured.tile_using_for %relu3
                                                           tile_sizes [0, 0, 1, 16]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %comb2, %comb_xi, %comb_ci = transform.structured.tile_using_for %comb
                                                           tile_sizes [0, 0, 1, 16]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}
```

This stage uses `transform.structured.tile_using_for`, which makes one `scf.for` per tiled dimension and returns a handle to each (`%conv_xi`, `%conv_ci`, ...), because section 5 unrolls these loops and `transform.loop.unroll` takes an `scf.for`. (`tile_using_forall [0, 0, 1, 16]` would make a single two-dimensional `scf.forall` instead; section 7.5 uses that form.)

```bash
cd /Users/roy/study/mlir/transform/ChH
sed -n '8,86p' ../../test_Example/transform/ChH/full.mlir |
  mlir-opt --transform-preload-library=transform-library-paths=stages/step4-tile-xi-ci.mlir \
           --transform-interpreter |
  grep -E '= (scf|linalg|tensor\.empty)' | sed -E 's/ (ins|outs|iter_args|shared_outs|\{indexing_maps)[ (=].*//'
```

Real output **[7]**:

```mlir
    %0 = tensor.empty() : tensor<5x80x100x128xf32>
    %1 = scf.forall (%arg4) in (2)
      %6 = scf.forall (%arg6, %arg7, %arg8) in (5, 80, 20)
        %10 = scf.for %arg10 = %c0 to %c5 step %c1
          %16 = scf.for %arg12 = %c0_5 to %c64 step %c16
            %broadcasted = linalg.broadcast
        %11 = tensor.empty() : tensor<1x1x5x64x1x1x1xf32>
        %12 = linalg.fill
        %13 = scf.for %arg10 = %c0_7 to %c3 step %c1_11
          %16 = scf.for %arg12 = %c0_8 to %c3_10 step %c1_12
            %17 = scf.for %arg14 = %c0_9 to %c128 step %c1_13
              %18 = scf.for %arg16 = %c0_30 to %c5_32 step %c1_34
                %19 = scf.for %arg18 = %c0_31 to %c64_33 step %c16_35
                  %20 = linalg.generic
        %14 = scf.for %arg10 = %c0_14 to %c5_16 step %c1_18
          %16 = scf.for %arg12 = %c0_15 to %c64_17 step %c16_19
            %reduced = linalg.reduce
        %15 = scf.for %arg10 = %c0_21 to %c5_23 step %c1_25
          %16 = scf.for %arg12 = %c0_22 to %c64_24 step %c16_26
            %17 = linalg.generic
```

This is the target structure of section 4.3, line for line: inside `co` and `n, y, xo`, the bias in `xi` (5) × `ci` (64 step 16 = 4) loops; the reduction loops `rz, ry, rx` around `xi, ci` and the convolution; then the combiner and the ReLU, each in its own `xi, ci` loops. Every Linalg op now works on `1x1x1x16` slices (`1x1x1x16x1x1x1` for the convolution's partial result). **The whole Halide loop structure was produced by tiling alone — tiling by 1 to materialize a loop at the right position, and by the vector length to keep a vectorizable dimension.**

---

## 5. Explicit Loop Unrolling

The remaining loop transformation is unrolling. Unrolling is requested for the innermost loops, which form the 4×5 tile of 16-element vector operations: it makes a contiguous sequence of `vfma` instructions that use 20 512-bit vector registers as accumulators. The schedule also requests `unroll(y)` and `unroll(r.x, 2)`, but these *have no practical effect*: the code, and every intermediate representation, Halide produces without them is strictly identical to the code with the full schedule. So only the loops that Halide actually unrolls, along `xi` and `ci`, are unrolled.

Tiling returned handles to the loops it created, so unrolling them is just a matter of chaining the transformation. The inner loop must be unrolled first: unrolling the outer loop clones the inner one and invalidates the handle to it.

***stages/step5-unroll.mlir***
```mlir
// Stage 5: stage 4 + fully unroll the xi (5) and ci (4) loops.
...
    // Unroll the inner (ci) loop first: unrolling the outer (xi) loop clones
    // the inner one and would invalidate the handle to it.
    transform.loop.unroll %bias_ci {factor = 4} : !transform.any_op
    transform.loop.unroll %bias_xi {factor = 5} : !transform.any_op
    transform.loop.unroll %conv_ci {factor = 4} : !transform.any_op
    transform.loop.unroll %conv_xi {factor = 5} : !transform.any_op
    transform.loop.unroll %relu_ci {factor = 4} : !transform.any_op
    transform.loop.unroll %relu_xi {factor = 5} : !transform.any_op
    transform.loop.unroll %comb_ci {factor = 4} : !transform.any_op
    transform.loop.unroll %comb_xi {factor = 5} : !transform.any_op
    transform.yield
  }
}
```

Unrolled code is long, so instead of an outline this run counts the loops and Linalg ops that are left:

```bash
cd /Users/roy/study/mlir/transform/ChH
sed -n '8,86p' ../../test_Example/transform/ChH/full.mlir |
  mlir-opt --transform-preload-library=transform-library-paths=stages/step5-unroll.mlir \
           --transform-interpreter |
  grep -oE '(scf\.forall|scf\.for|linalg\.[a-z_]+) ' | sort | uniq -c
```

Real output **[8]**:

```
  20 linalg.broadcast 
   1 linalg.fill 
  40 linalg.generic 
  20 linalg.reduce 
  60 linalg.yield 
   3 scf.for 
   2 scf.forall 
```

- Each loop with a factor equal to its trip count disappeared: the four computations now appear 5 × 4 = 20 times each (`linalg.broadcast`, `linalg.reduce`, and 20 + 20 `linalg.generic` for the convolution and the ReLU).
- The 3 remaining `scf.for` are the `rz, ry, rx` reduction loops; the 2 `scf.forall` are `co` and `n, y, xo`. The `linalg.fill` of the partial result was never tiled, so it stays one op; `linalg.yield` counts the bodies of the 40 generics and the 20 reduces (`linalg.broadcast` prints without one).

---

## 6. Vectorization

### 6.1 Simplify first

The loop structure is in place; now vectorize. Tiling and fusion have produced many operations computing tensor subsets and loop bounds, some duplicated or needlessly complex, so simplify first. Canonicalization, CSE (common subexpression elimination), LICM (loop-invariant code motion) and various rewrite patterns are all available from the Transform dialect. Moreover, **an arbitrary combination of rewrite patterns can be applied in one sweep** to a given scope — something conventional passes can't do, since each applies its own group of patterns (short of writing a new pass for each combination):

***stages/step6-vectorize.mlir***
```mlir
// Stage 6: stage 5 + simplify, fold unit dimensions, vectorize, simplify.
...
    // Simplify: canonicalize, CSE, hoist loop invariants.
    transform.apply_patterns to %f00 {
      transform.apply_patterns.canonicalization
      transform.apply_patterns.linalg.tiling_canonicalization
    } : !transform.any_op
    transform.apply_cse to %f00 : !transform.any_op
    %all_loops = transform.structured.match interface{LoopLikeInterface}
      in %arg0
      : (!transform.any_op) -> !transform.any_op
    transform.apply_licm to %all_loops : !transform.any_op
```

- `%f00` is the `func.func` handle matched in step 2; `transform.apply_patterns` applies the listed pattern sets together, to a fixed point, inside it.
- `match interface{LoopLikeInterface}` matches by interface instead of name — every `scf.for` and `scf.forall` — and `transform.apply_licm` hoists their invariant operations.

### 6.2 Fold unit dimensions, vectorize, simplify again

One last simplification is needed for good vector code. Tiling by one produced structured ops on 4-D types where only one dimension isn't 1, e.g. `tensor<1x1x1x16xf32>` (16 being the AVX-512 vector size), because structured tiling preserves the operation's rank. The computation is the same, but the code can end up more complicated than necessary, especially when multidimensional vectors are decomposed into the one-dimensional vectors hardware supports. The unit dimensions can be folded away by a dedicated pattern set before vectorization:

***stages/step6-vectorize.mlir***
```mlir
    // Fold the unit dimensions left by tiling-by-one, then vectorize.
    transform.apply_patterns to %f00 {
      transform.apply_patterns.linalg.fold_unit_extent_dims_via_reshapes
    } : !transform.any_op
    %fv = transform.structured.vectorize_children_and_apply_patterns %f00
      : (!transform.any_op) -> !transform.any_op
```

This produces arithmetic on `vector<16xf32>`, which the downstream compiler can lower directly to AVX-512 instructions. Vectorization creates new simplification opportunities — in particular, tensor subsetting can be combined with vector slicing — so another round follows:

***stages/step6-vectorize.mlir***
```mlir
    // Simplify again after vectorization.
    transform.apply_patterns to %fv {
      transform.apply_patterns.canonicalization
      transform.apply_patterns.tensor.fold_tensor_subset_ops_into_vector_transfers
    } : !transform.any_op
    transform.apply_cse to %fv : !transform.any_op
    transform.structured.hoist_redundant_vector_transfers %fv
      : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
```

- `fold_tensor_subset_ops_into_vector_transfers` folds `tensor.extract_slice`/`insert_slice` into the indices of the `vector.transfer_read`/`transfer_write` that use them.
- `hoist_redundant_vector_transfers` moves a read/write pair of the same location out of a loop, keeping the value in a register (an `iter_args` value) across iterations.

```bash
cd /Users/roy/study/mlir/transform/ChH
sed -n '8,86p' ../../test_Example/transform/ChH/full.mlir |
  mlir-opt --transform-preload-library=transform-library-paths=stages/step6-vectorize.mlir \
           --transform-interpreter |
  grep -oE '(scf\.forall|scf\.for|linalg\.[a-z_]+|vector\.[a-z_]+) |vector<[0-9x]+xf32>' |
  sort | uniq -c | sort -rn
```

Real output **[9]**:

```
 217 vector<16xf32>
 133 vector.transfer_read 
 121 vector.transfer_write 
 104 vector<1x1x1x16xf32>
  40 vector<1x1x1x16x1x1x1xf32>
  20 vector.shape_cast 
  20 linalg.yield 
  20 linalg.generic 
   9 vector.broadcast 
   3 scf.for 
   2 vector<1x1x5x64x1x1x1xf32>
   2 scf.forall 
```

- The convolution, the bias and the combiner are vectors now: `vector.transfer_read`/`transfer_write` move `vector<16xf32>` values in and out of tensors, and the arithmetic (`arith.mulf`/`addf`, not counted here) works on `vector<16xf32>`; `vector.shape_cast` and the `1x1x1x16` types are the folded unit dimensions being reshaped at the boundaries.
- The two `vector<1x1x5x64x1x1x1xf32>` are the untiled `linalg.fill` of the partial result, vectorized whole: a constant zero vector and its write.
- **20 `linalg.generic` remain**: the 20 ReLU tiles were not vectorized. The next command shows one.

```bash
cd /Users/roy/study/mlir/transform/ChH
sed -n '8,86p' ../../test_Example/transform/ChH/full.mlir |
  mlir-opt --transform-preload-library=transform-library-paths=stages/step6-vectorize.mlir \
           --transform-interpreter |
  grep -m1 -A4 'linalg.generic'
```

Real output **[10]**:

```mlir
        %220 = linalg.generic {indexing_maps = [#map6, #map7, #map7], iterator_types = ["parallel"]} ins(%cst_0, %extracted_slice_162 : f32, tensor<16xf32>) outs(%extracted_slice_163 : tensor<16xf32>) {
        ^bb0(%in: f32, %in_202: f32, %out: f32):
          %280 = llvm.intr.maxnum(%in, %in_202) : (f32, f32) -> f32
          linalg.yield %280 : f32
        } -> tensor<16xf32>
```

The ReLU body is `llvm.intr.maxnum`, an LLVM dialect intrinsic. The Linalg vectorizer only vectorizes body operations it knows how to widen elementwise (those with the `ElementwiseMappable` trait, such as `arith` ops), and `llvm.intr.maxnum` isn't one, so the vectorizer leaves the op alone and `vectorize_children_and_apply_patterns` still succeeds. The ReLU stays scalar all the way to the LLVM lowering, where `convert-linalg-to-loops` turns it into loops. It costs little here (the ReLU is 5·80·100·128 maxima against ~5.9·10⁹ flops of convolution); appendix C shows that `arith.maxnumf` vectorizes.

---

## 7. Lowering to LLVM and The Bufferization Hurdle

### 7.1 Bufferization and deallocation

With the loops restructured, the program must become executable. The first step is *bufferization*: associating a memory buffer with every tensor in the payload. MLIR's one-shot bufferization is available as a transform op. It doesn't create deallocations, which could leak, so the buffer deallocation pass pipeline runs next. **The Transform dialect runs named passes and pass pipelines directly** — complex `--pass-pipeline` expressions could be replaced by transform ops — and it applies this one to the functions rather than the module, so that it doesn't run on the transform IR the module may contain. Stage 7 appends the second half of `full.mlir`'s `@lower` sequence (lines 319–373) to stage 6:

***stages/step7-bufferize.mlir***
```mlir
// Stage 7: stage 6 + bufferize and lower the vector operations.
...
    // Bufferize, then lower the vector operations (same as full.mlir).
    %arg1 = transform.bufferization.one_shot_bufferize %arg0 {
      bufferize_function_boundaries = true,
      function_boundary_type_conversion = 1 : i32 }
      : (!transform.any_op) -> !transform.any_op
    %f = transform.structured.match ops{["func.func"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    transform.apply_registered_pass "buffer-deallocation-pipeline" to %f
      : (!transform.any_op) -> !transform.any_op
```

- `bufferize_function_boundaries = true` rewrites the function signature too (tensor arguments become memrefs); `function_boundary_type_conversion = 1` is `identity-layout-map`: argument memrefs get the plain row-major layout, matching the `bufferization.buffer_layout` attributes of section 2.3, so a caller can pass ordinary C arrays.
- `transform.apply_registered_pass` runs any registered pass or pipeline, by name, on the ops a handle points to.

Here the transformed IR could be bufferized directly. That is not always the case: some operations, in particular `tensor.empty`, may not be bufferizable. They must be removed first, which sufficient fusion often does (as here: the only `tensor.empty` left is never read), or with `transform.bufferization.eliminate_empty_tensors`, which removes `tensor.empty` ops that only define the size of a computation, or `transform.bufferization.empty_tensor_to_alloc_tensor`, which materializes a new temporary buffer to be used as a local cache.

### 7.2 Lowering the vector operations

***stages/step7-bufferize.mlir***
```mlir
    // Apply general canonicalization and CSE to each function after
    // bufferization as new simplification opportunities may have appeared.
    %fb = transform.structured.match ops{["func.func"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %fb {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %fb : !transform.any_op

    // Lower complex, multidimensional vector operations into simpler
    // primitives. This particular selection of the pattern groups corresponds
    // to vector dialect operations present in the payload IR at this stage.
    // Many of these groups can be parameterized to use different strategies or
    // lower-level primitives offering performance trade-offs. In this case, we
    // are selecting the simplest strategies.
    transform.apply_patterns to %fb {
      transform.apply_patterns.vector.lower_contraction
        lowering_strategy = parallelarith
      transform.apply_patterns.vector.lower_transfer
        max_transfer_rank = 1
      transform.apply_patterns.vector.lower_transpose
        lowering_strategy = eltwise
      transform.apply_patterns.vector.lower_shape_cast
    } : !transform.any_op

    // These patterns apply in a separate sweep to avoid transfer-to-scf
    // patterns overlap with lower-transfer patterns as they apply to the same
    // kind of operations. These patterns may produce local allocations to act
    // as temporary caches deep inside loops, which could lead to catastrophic
    // performance. Such allocations are moved onto the stack and hoisted from
    // all the surrounding loops.
    transform.apply_patterns to %fb {
      transform.apply_patterns.vector.transfer_to_scf
      transform.apply_patterns.memref.alloc_to_alloca
      } : !transform.any_op
    transform.bufferization.buffer_loop_hoisting %fb : !transform.any_op

    // A final round of cleanups additionally includes patterns to simplify
    // buffer aliasing operations that may have been introduced during
    // bufferization and could result in excessively complex address
    // computation.
    transform.apply_patterns to %fb {
      transform.apply_patterns.memref.fold_memref_alias_ops
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %fb : !transform.any_op
    transform.yield
  }
}
```

- `lower_transfer max_transfer_rank = 1` rewrites `vector.transfer_read`/`write` of rank ≤ 1 into plain `vector.load`/`store` where the memory is contiguous; `transfer_to_scf` turns the remaining higher-rank transfers into loops over lower-rank ones, which may need a temporary buffer — `alloc_to_alloca` puts it on the stack and `buffer_loop_hoisting` moves allocations out of `scf.for` loops.
- `lower_contraction`, `lower_transpose`, `lower_shape_cast` pick a strategy for each high-level vector op; `parallelarith` and `eltwise` are the simplest ones.

```bash
cd /Users/roy/study/mlir/transform/ChH
sed -n '8,86p' ../../test_Example/transform/ChH/full.mlir |
  mlir-opt --transform-preload-library=transform-library-paths=stages/step7-bufferize.mlir \
           --transform-interpreter |
  grep -oE '(scf\.forall|scf\.for|linalg\.[a-z_]+|vector\.[a-z_]+|memref\.[a-z_]+|bufferization\.clone) ' |
  sort | uniq -c | sort -rn
```

Real output **[11]**:

```
 320 vector.insert 
 320 vector.extract 
 306 vector.type_cast 
 101 memref.load 
  83 memref.subview 
  81 memref.store 
  80 vector.store 
  68 vector.load 
  62 memref.expand_shape 
  45 vector.transfer_read 
  45 scf.for 
  41 vector.transfer_write 
  20 vector.insertelement 
  20 linalg.yield 
  20 linalg.generic 
   9 vector.broadcast 
   2 scf.forall 
   2 memref.copy 
   1 bufferization.clone 
```

- No tensors remain: data now moves with `memref.load`/`store`, `vector.load`/`store` and `memref.subview`/`expand_shape` (views into buffers). The `vector.type_cast`, `vector.extract`/`insert` and 42 of the 45 `scf.for` (stage 6 had 3, the reduction loops) come from `transfer_to_scf` lowering the multidimensional transfers into loops over smaller ones.
- `bufferization.clone` and `memref.copy` come from the deallocation pipeline and bufferization; `convert-bufferization-to-memref` lowers the clone later (appendix B.2).

Due to its inter-procedural nature, one-shot bufferization processes the entire payload module and therefore **invalidates all previously created handles**. It is typically a late step, after which precise targeting is no longer needed: what follows are module- or function-wide rewrites, mostly pattern-based lowerings. This part of the script can be seen as a pass pipeline written in the Transform dialect, with pattern-based lowering passes built on the fly from named groups of patterns.

### 7.3 To LLVM IR — and where this version stops

The resulting IR can then be lowered completely to the LLVM dialect, translated to LLVM IR, and compiled into an executable or JIT-compiled. This chapter's lowering pipeline is `driver/lower-to-llvm.txt` (appendix B.2), passed to `mlir-opt --pass-pipeline`:

```bash
cd /Users/roy/study/mlir/transform/ChH
sed -n '8,86p' ../../test_Example/transform/ChH/full.mlir |
  mlir-opt --transform-preload-library=transform-library-paths=stages/step7-bufferize.mlir \
           --transform-interpreter |
  mlir-opt --pass-pipeline="$(tr -d ' \n' < driver/lower-to-llvm.txt)" |
  mlir-translate --mlir-to-llvmir > /dev/null
```

Real output **[12]** (exit status 1; the note's dialect list and link are abridged):

```
<stdin>:2440:13: error: Dialect `vector' not found for custom op 'vector.transfer_read' 
    %2349 = vector.transfer_read %2339[%88, %88, %88, %88], %89 : memref<1x1x5x1xf32, strided<[640, 640, 128, 128], offset: ?>>, vector<f32>
            ^
<stdin>:2440:13: note: Registered dialects: acc, amx, ... ; for more info on dialect registration see ...
```

`mlir-opt` succeeds but leaves 0-D (zero-dimensional) `vector.transfer_read` ops behind, and `mlir-translate`, which only knows the LLVM dialect, fails on the first one. They read one element from a `memref<1x1x5x1xf32, strided<[640, 640, 128, 128]>>` view of the input: tiling `ci` by 16 made the convolution read one input value at a time (the input is indexed by `rx`, not `ci`), and the generic lowering turns that read into a 0-D transfer on a non-contiguous view, which neither `convert-vector-to-scf` nor `convert-vector-to-llvm` lowers in MLIR 20. So this Halide-faithful version stays at the IR level on this toolchain; section 8 removes the step that produces these slivers.

### 7.4 Performance: the cost of unrolling early

Measured on an Intel processor with the Skylake microarchitecture at 2.0 GHz, the Halide-faithful code runs in about 420 ms. The computation performs 5 · 80 · 100 · 128 · (2 · 3 · 3 · 128 + 2) ≈ 5.9 · 10⁹ floating-point operations, so that is ~14 GFLOP/s. With one FMA unit, the single-core peak of that processor is 64 GFLOP/s (16 · 2 · 2 · 10⁹, 16 being the vector width), so only 22% of peak is reached. The code Halide produces runs in ~120 ms on the same processor, 3.5× faster and 77% of peak. (These are reported figures for x86 with AVX-512; this machine is an Apple M3 Pro, so they are not reproduced here — section 8.3 measures the final version on it.)

The difference shows in the assembly around the convolution's multiply-adds. Both use AVX-512 `vfmadd231ps` on 512-bit `%zmm` registers, but in the MLIR-generated code they are interspersed with memory accesses that load *two* of the three FMA operands before each operation, increasing latency (x86 assembly, not produced on this machine):

```asm
vmovups       -192(%r10), %zmm0
vbroadcastss  -1536(%rdi,%r9), %zmm1
vmovups       112(%rsp), %zmm2
vfmadd231ps   %zmm1, %zmm0, %zmm2     # zmm2 = (zmm0 * zmm1) + zmm2
vmovups       %ymm2, 112(%rsp)
vextractf64x4 $1, %zmm2, 144(%rsp)
// 19 more blocks of either
//  (a) vmovups,vbroadcast,vfma(z,z),vextract,
//  (b) vbroadcast,vfma(z,mem),vextract
```

Halide's code has compact blocks of `vfmadd231ps` and `vbroadcastss` that load one operand while the other two stay in registers, loaded before the FMAs:

```asm
vbroadcastss    -1536(%rsi,%rbx), %zmm25
vmovups         -192(%rdi), %zmm26
vmovups         -128(%rdi), %zmm27
vmovups         -64(%rdi), %zmm28
vmovups         (%rdi), %zmm29
vfmadd231ps     %zmm25, %zmm26, %zmm24  # zmm24 = (zmm26 * zmm25) + zmm24
vfmadd231ps     %zmm25, %zmm27, %zmm23  # zmm23 = (zmm27 * zmm25) + zmm23
vfmadd231ps     %zmm25, %zmm28, %zmm22  # zmm22 = (zmm28 * zmm25) + zmm22
vfmadd231ps     %zmm25, %zmm29, %zmm21  # zmm21 = (zmm29 * zmm25) + zmm21
vbroadcastss    -1024(%rsi,%rbx), %zmm25
vfmadd231ps     %zmm25, %zmm26, %zmm20  # zmm20 = (zmm26 * zmm25) + zmm20
vfmadd231ps     %zmm25, %zmm27, %zmm19  # zmm19 = (zmm27 * zmm25) + zmm19
vfmadd231ps     %zmm25, %zmm28, %zmm18  # zmm18 = (zmm28 * zmm25) + zmm18
vfmadd231ps     %zmm25, %zmm29, %zmm17  # zmm17 = (zmm29 * zmm25) + zmm17
vbroadcastss    -512(%rsi,%rbx), %zmm25

// 3 more blocks of 4 vfmadd231 followed by a vbroadcast
```

Following the IR through the stages shows the load (transfer)/FMA interleaving at every level from the schedule's result on. The repeated tensor subsetting operations — later vector transfers, then vector loads — **come from the loop unrolling the schedule explicitly asked for**. The single-assignment model of tensors (and vectors) turns unrolled code into long, complex chains of access and update operations, so long that lower-level transformations and the downstream compiler can no longer simplify them. Unrolling early can cause all sorts of compile-time and performance problems, including the compiler giving up on some optimizations because the code is too long. Section 5's 20 copies of every op are where this starts.

### 7.5 Unrolling late, by matching again after bufferization

It is better to unroll later, after bufferization and the simplifications that go with it. But bufferization invalidates every handle, including those to the loops to unroll. The way around it is to **match the payload again after bufferization** to get new handles. To have fewer ops to match, the `[0, 0, 1, 16]` tiling makes `scf.forall` loops (`tile_using_forall` instead of `tile_using_for`); after bufferization, the script matches all `scf.forall` ops and turns the four inner ones into `scf.for` loops with `transform.loop.forall_to_for`. The unrolling then moves to the end of the script. Stage 8 is stage 7 with these changes:

***stages/step8-late-unroll.mlir***
```mlir
// Stage 8: stage 7 with the xi/ci loops materialized as scf.forall, converted
// to scf.for and unrolled only after bufferization.
...
    // Materialize xi (size 1) and ci (size 16 = the vector length) as one
    // two-dimensional scf.forall around each of the four computations.
    //                                                                n  y  xi ci
    %conv5, %conv_xi_ci = transform.structured.tile_using_forall %conv4
                                                           tile_sizes [0, 0, 1, 16]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
...
    // Bufferization invalidated every handle: match the loops again. Matching
    // walks the IR in post-order, so inner loops come before the loops that
    // contain them: bias, conv, combiner, ReLU, then (n, y, xo) and co.
    %foralls = transform.structured.match ops{["scf.forall"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %xi_ci_bias, %xi_ci_conv, %xi_ci_comb, %xi_ci_relu, %n_y_xo_l, %co_l
      = transform.split_handle %foralls
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op,
          !transform.any_op, !transform.any_op, !transform.any_op,
          !transform.any_op)
    %xi_bias, %ci_bias = transform.loop.forall_to_for %xi_ci_bias
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %xi_conv, %ci_conv = transform.loop.forall_to_for %xi_ci_conv
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %xi_comb, %ci_comb = transform.loop.forall_to_for %xi_ci_comb
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %xi_relu, %ci_relu = transform.loop.forall_to_for %xi_ci_relu
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Unrolling now happens on buffers, after all the tensor-level rewrites.
    transform.loop.unroll %ci_bias {factor = 4} : !transform.any_op
    transform.loop.unroll %xi_bias {factor = 5} : !transform.any_op
    transform.loop.unroll %ci_conv {factor = 4} : !transform.any_op
    transform.loop.unroll %xi_conv {factor = 5} : !transform.any_op
    transform.loop.unroll %ci_comb {factor = 4} : !transform.any_op
    transform.loop.unroll %xi_comb {factor = 5} : !transform.any_op
    transform.loop.unroll %ci_relu {factor = 4} : !transform.any_op
    transform.loop.unroll %xi_relu {factor = 5} : !transform.any_op
    transform.yield
  }
}
```

(The other three `tile_using_forall` ops, for the bias, the ReLU and the combiner, change the same way, and stage 5's early unrolling block is gone.) The order of `split_handle`'s results matters: `forall_to_for` must produce one `scf.for` per induction variable, so a handle that landed on the three-dimensional `n, y, xo` loop fails with `op expects as many results (2) as payload has induction variables (3)` — the reason the comment spells out the post-order walk.

```bash
cd /Users/roy/study/mlir/transform/ChH
T=$(mktemp -d)
sed -n '8,86p' ../../test_Example/transform/ChH/full.mlir |
  mlir-opt --transform-preload-library=transform-library-paths=stages/step8-late-unroll.mlir \
           --transform-interpreter > $T/step8.mlir
echo "interpreter exit status: $?"
grep -oE '(scf\.forall|scf\.for|linalg\.[a-z_]+) ' $T/step8.mlir | sort | uniq -c
mlir-opt $T/step8.mlir --pass-pipeline="$(tr -d ' \n' < driver/lower-to-llvm.txt)" |
  mlir-translate --mlir-to-llvmir > /dev/null
echo "mlir-translate exit status: ${PIPESTATUS[1]}"
rm -r $T
```

Real output **[13]** (the note's dialect list and link are abridged):

```
interpreter exit status: 0
  20 linalg.generic 
  20 linalg.yield 
  45 scf.for 
   2 scf.forall 
<stdin>:2303:13: error: Dialect `vector' not found for custom op 'vector.transfer_read' 
    %2235 = vector.transfer_read %2225[%92, %92, %92, %92], %93 : memref<1x1x5x1xf32, strided<[640, 640, 128, 128], offset: ?>>, vector<f32>
            ^
<stdin>:2303:13: note: Registered dialects: acc, amx, ... ; for more info on dialect registration see ...
mlir-translate exit status: 1
```

- The script runs: only the `co` and `n, y, xo` `scf.forall`s are left, and the inner loops were converted and unrolled after bufferization (the 20 ReLU `linalg.generic`s, now on memrefs, are the unrolled ReLU tiles).
- Lowering stops at the same 0-D transfers as section 7.3: they come from tiling `ci` by 16, not from when the unrolling happens.

Compiled to assembly, this version gives exactly the same core computation around `vfmadd231ps` as Halide's, differing only slightly in allocated registers, and runs in roughly 120 ms on the Skylake machine. **Re-matching after bufferization recovers the handles that a module-wide transformation invalidated**, at the price of finding the loops by name and position again.

---

## 8. Multi-Dimensional Vectors to the Rescue

### 8.1 The final schedule: stop tiling at `5x64`

Section 7.5 matched the generated loops again after bufferization, which partly defeats the purpose of chaining transformations through handles. Luckily it isn't needed: it only served as an exercise in producing the Halide loop structure.

Multidimensional operations on vectors are lowered to target-sized vectors by unrolling and splitting. For example, an elementwise operation on `vector<5x64xf32>` is replaced by 5 operations on `vector<64xf32>`, plus vector value manipulations to rebuild the required type at the MLIR level. Each of those is then split into 4 operations on `vector<16xf32>` at the LLVM level, where the target's vector width is known. Together, that has exactly the same effect as materializing the 5×4 loop nest and fully unrolling it. **So the last tiling stage, the re-matching and the unrolling can all be dropped from the schedule.** This is the schedule in `full.mlir`: it is stage 3, followed by two small steps and the lowering sequence of sections 6–7, factored into a named sequence:

***test_Example/transform/ChH/full.mlir***
```mlir
    // At this point, the inner Linalg operations have implicit iteration spaces
    // of 5x64 size, with some additional unit-size dimensions. Completely
    // replicating Halide schedule would require materializing the loops with
    // 5 and 4 iterations, respectively, unrolling those loops and marking the
    // remaining 16-point iteration space for vectorization.
    //
    // This is unnecessary in MLIR that supports multi-dimensional vectors,
    // which will be decomposed into target-specific sizes during the lowering.
    // Therefore, this schedule stops here.

    // Transform the named broadcast operation used for bias into the generic
    // form before vectorization to prevent special cases from kicking in.
    transform.structured.generalize %bias3
      : (!transform.any_op) -> !transform.any_op

    // Use the named macro to perform most of the lowering.
    transform.include @lower failures(propagate) (%arg0)
      : (!transform.any_op) -> ()
    transform.yield
  }

  // Named sequence of transformations is a macro-like object that can be
  // included from another place in the transform dialect, but doesn't allow for
  // recursion. This can be reused in other scenarios.
  transform.named_sequence @lower(
      %arg0: !transform.any_op {transform.consumed}) {
    %f00 = transform.structured.match ops{["func.func"]} in %arg0
      : (!transform.any_op) -> !transform.any_op
...
```

- `transform.structured.generalize %bias3` turns the fused `linalg.broadcast` into a `linalg.generic` before vectorization, so it is vectorized by the same generic path as the other ops.
- `transform.include @lower failures(propagate) (%arg0)` calls another named sequence like a macro (no recursion). `{transform.consumed}` on its argument declares that `@lower` invalidates the handle it receives — it bufferizes the whole module — so the caller can't use `%arg0` afterwards.
- `@lower` is sections 6.1–6.2 and 7.1–7.2, in that order. Its unit-dimension folding now turns `tensor<1x1x5x64xf32>` into `tensor<5x64xf32>`, which vectorizes to `vector<5x64xf32>`.

### 8.2 Down to the LLVM dialect

The full script, from line 90 of `full.mlir` to the end of its transform module (line 377), is copied into a temporary file and preloaded; the transformed payload goes through the lowering pipeline; and `FileCheck` checks the result against the `CHECK` lines at the end of the test file, which require five adjacent `llvm.intr.fma` on `vector<64xf32>`:

***test_Example/transform/ChH/full.mlir***
```mlir
// The core computation, at the LLVM dialect level, must correspond to five
// immediately adjacent fma on vector<64xf32>.

// CHECK:      %[[R0:.+]] = llvm.mlir.undef : !llvm.array<5 x vector<64xf32>>

// CHECK:      %[[V:.+]] = llvm.load %{{.*}} : !llvm.ptr -> !llvm.array<5 x vector<64xf32>>
// CHECK-NEXT: %[[LINE0:.+]] = llvm.extractvalue %[[V]][0] : !llvm.array<5 x vector<64xf32>>
// CHECK-NEXT: %[[FMA0:.+]] = llvm.intr.fma(%{{.*}}, %{{.*}}, %[[LINE0]])
// CHECK-SAME: -> vector<64xf32>
// CHECK-NEXT: %[[R1:.+]] = llvm.insertvalue %[[FMA0]], %[[R0]][0]
...
```

```bash
cd /Users/roy/study/mlir/transform/ChH
F=../../test_Example/transform/ChH/full.mlir
T=$(mktemp -d)
sed -n '90,377p' $F > $T/schedule.mlir
sed -n '8,86p' $F |
  mlir-opt --transform-preload-library=transform-library-paths=$T/schedule.mlir \
           --transform-interpreter |
  mlir-opt --pass-pipeline="$(tr -d ' \n' < driver/lower-to-llvm.txt)" > $T/conv.llvm.mlir
FileCheck $F --input-file=$T/conv.llvm.mlir && echo "FileCheck: PASS"
grep -c 'llvm.intr.fma' $T/conv.llvm.mlir
grep -A14 'llvm.load .* -> !llvm.array<5 x vector<64xf32>>' $T/conv.llvm.mlir | sed -E 's/^ +//'
rm -r $T
```

Real output **[14]**:

```mlir
FileCheck: PASS
5
%843 = llvm.load %842 : !llvm.ptr -> !llvm.array<5 x vector<64xf32>>
%844 = llvm.extractvalue %843[0] : !llvm.array<5 x vector<64xf32>> 
%845 = llvm.intr.fma(%473, %820, %844) {fastmathFlags = #llvm.fastmath<fast>} : (vector<64xf32>, vector<64xf32>, vector<64xf32>) -> vector<64xf32>
%846 = llvm.insertvalue %845, %42[0] : !llvm.array<5 x vector<64xf32>> 
%847 = llvm.extractvalue %843[1] : !llvm.array<5 x vector<64xf32>> 
%848 = llvm.intr.fma(%473, %821, %847) {fastmathFlags = #llvm.fastmath<fast>} : (vector<64xf32>, vector<64xf32>, vector<64xf32>) -> vector<64xf32>
%849 = llvm.insertvalue %848, %846[1] : !llvm.array<5 x vector<64xf32>> 
%850 = llvm.extractvalue %843[2] : !llvm.array<5 x vector<64xf32>> 
%851 = llvm.intr.fma(%473, %822, %850) {fastmathFlags = #llvm.fastmath<fast>} : (vector<64xf32>, vector<64xf32>, vector<64xf32>) -> vector<64xf32>
%852 = llvm.insertvalue %851, %849[2] : !llvm.array<5 x vector<64xf32>> 
%853 = llvm.extractvalue %843[3] : !llvm.array<5 x vector<64xf32>> 
%854 = llvm.intr.fma(%473, %823, %853) {fastmathFlags = #llvm.fastmath<fast>} : (vector<64xf32>, vector<64xf32>, vector<64xf32>) -> vector<64xf32>
%855 = llvm.insertvalue %854, %852[3] : !llvm.array<5 x vector<64xf32>> 
%856 = llvm.extractvalue %843[4] : !llvm.array<5 x vector<64xf32>> 
%857 = llvm.intr.fma(%473, %824, %856) {fastmathFlags = #llvm.fastmath<fast>} : (vector<64xf32>, vector<64xf32>, vector<64xf32>) -> vector<64xf32>
```

- `FileCheck` passes: the multiply-add of the convolution is five `llvm.intr.fma` on `vector<64xf32>` — one per row of the `5x64` accumulator tile — between an `llvm.extractvalue` of the loaded row and an `llvm.insertvalue` back into the `!llvm.array<5 x vector<64xf32>>`. `math-uplift-to-fma` (in the pipeline) turned the `fastmath<fast>` `mulf` + `addf` into `math.fma`, and `convert-math-to-llvm` into `llvm.intr.fma`.
- These five operations are the whole 4×5 tile of 16-wide FMAs that Halide's `unroll(ci)`, `unroll(xi)` and `vectorize(ci, 16)` produce — just not yet split to the hardware width.

### 8.3 Running it

`driver/driver.c` (appendix B.1) fills the input, filter and bias with deterministic pseudo-random values, calls `_mlir_ciface_conv` three times and keeps the best time, then checks **every** output element against a plain C implementation of `relu(bias + conv)` and times that too:

```bash
cd /Users/roy/study/mlir/transform/ChH
F=../../test_Example/transform/ChH/full.mlir
T=$(mktemp -d)
sed -n '90,377p' $F > $T/schedule.mlir
sed -n '8,86p' $F |
  mlir-opt --transform-preload-library=transform-library-paths=$T/schedule.mlir \
           --transform-interpreter |
  mlir-opt --pass-pipeline="$(tr -d ' \n' < driver/lower-to-llvm.txt)" |
  mlir-translate --mlir-to-llvmir -o $T/conv.ll
clang -O3 -Wno-override-module driver/driver.c $T/conv.ll -o $T/conv && $T/conv
rm -r $T
```

Real output **[15]** (the times vary from run to run):

```
checksum 5803713.322, max abs error vs reference 1.34e-05: PASS
conv: best of 3 1512 ms, 7.8 GFLOP/s
C reference (clang): 398 ms, 29.7 GFLOP/s
```

- **Correct**: the result matches the C reference to within 1.34·10⁻⁵ (the summation order differs, which `fastmath` allows), over all 5 120 000 outputs.
- **Not fast on this machine**: under 8 GFLOP/s, about a quarter of the speed of the plain C reference compiled by `clang -O3`. On the Skylake machine the same schedule runs in about 110 ms, ~53.7 GFLOP/s or 84% of the single-core peak — an 8% improvement over both the section 7.5 version and Halide, because the IR is shorter and simpler in the presence of large vector operations, which allows more aggressive address computation and load placement. There, the assembly has all `vbroadcast`s grouped before the `vfmadd231ps` but otherwise the same structure as Halide's; the grouping comes from each multidimensional vector operation being "unrolled" separately. Section 8.4 shows why this machine is different.

### 8.4 The generated assembly

```bash
cd /Users/roy/study/mlir/transform/ChH
F=../../test_Example/transform/ChH/full.mlir
T=$(mktemp -d)
sed -n '90,377p' $F > $T/schedule.mlir
sed -n '8,86p' $F |
  mlir-opt --transform-preload-library=transform-library-paths=$T/schedule.mlir \
           --transform-interpreter |
  mlir-opt --pass-pipeline="$(tr -d ' \n' < driver/lower-to-llvm.txt)" |
  mlir-translate --mlir-to-llvmir | llc -O3 -o $T/conv.s
echo "fmla: $(grep -c fmla $T/conv.s), spills: $(grep -c 'Folded Spill' $T/conv.s), reloads: $(grep -c 'Folded Reload' $T/conv.s)"
grep -m1 -B6 -A6 'fmla' $T/conv.s
rm -r $T
```

Real output **[16]**:

```asm
fmla: 80, spills: 282, reloads: 289
	ldp	q2, q13, [x27, #128]
	ldr	q3, [x27, #1056]
	ldr	q20, [x27, #1040]
	str	q1, [x19, #2832]                ; 16-byte Folded Spill
	ldr	q1, [x27, #1168]
	ldr	q22, [x27, #1024]
	fmla.4s	v20, v18, v29
	str	q0, [x19, #3312]                ; 16-byte Folded Spill
	str	q1, [x19, #2864]                ; 16-byte Folded Spill
	fmla.4s	v22, v18, v23
	ldr	q1, [x27, #1184]
	ldp	q0, q15, [x27, #768]
	str	q2, [x19, #2512]                ; 16-byte Folded Spill
```

`llc` targets the host, AArch64, whose vector registers (NEON) are 128 bits wide: 4 floats. The tile was sized for AVX-512, where a `5x64` f32 accumulator is 5 × 64 / 16 = **20** of the 32 `zmm` registers. On NEON the same tile is 5 × 64 / 4 = **80** registers, but there are only 32. So the 80 `fmla.4s` (one per 4 floats of the tile) are interleaved with stack spills and reloads — `str q0, [x19, #3312] ; 16-byte Folded Spill` — the same latency problem as the x86 listing of section 7.4, for a different reason.

### 8.5 Retuning the tile for NEON

The fix is in the schedule, not in the compiler: keep the register budget, change the tile. Tiling `c` by 16 instead of 64 makes a `5x16` accumulator, 5 × 16 / 4 = 20 NEON registers. The run patches the copied script with `sed`:

```bash
cd /Users/roy/study/mlir/transform/ChH
F=../../test_Example/transform/ChH/full.mlir
T=$(mktemp -d)
sed -n '90,377p' $F | sed 's/tile_sizes \[0, 0, 0, 64\]/tile_sizes [0, 0, 0, 16]/' > $T/schedule.mlir
sed -n '8,86p' $F |
  mlir-opt --transform-preload-library=transform-library-paths=$T/schedule.mlir \
           --transform-interpreter |
  mlir-opt --pass-pipeline="$(tr -d ' \n' < driver/lower-to-llvm.txt)" |
  mlir-translate --mlir-to-llvmir -o $T/conv.ll
llc -O3 $T/conv.ll -o $T/conv.s
echo "fmla: $(grep -c fmla $T/conv.s), spills: $(grep -c 'Folded Spill' $T/conv.s), reloads: $(grep -c 'Folded Reload' $T/conv.s)"
clang -O3 -Wno-override-module driver/driver.c $T/conv.ll -o $T/conv && $T/conv
rm -r $T
```

Real output **[17]** (the times vary from run to run):

```
fmla: 20, spills: 43, reloads: 48
checksum 5803713.322, max abs error vs reference 1.34e-05: PASS
conv: best of 3 906 ms, 13.0 GFLOP/s
C reference (clang): 407 ms, 29.0 GFLOP/s
```

20 `fmla.4s` per window position, far fewer spills, and the same checksum — and roughly 1.7× faster than section 8.3. (It is still slower than the C reference; closing that gap would take more retuning — the `xo` tile, the order of the reduction loops, the input broadcast — which is beyond this chapter.) **A schedule encodes a target: its tile sizes are register-file arithmetic, so porting it means redoing that arithmetic, not touching the algorithm.**

---

## 9. Deriving the Halide schedule yourself

Sections 3–8 explain the schedule in `full.mlir` as it is. This section shows how to arrive at it without having it. The Halide schedule (section 3.1) says *what* loop structure to build, and section 3.2 names the transform operation for each directive. The rest is in the installed MLIR, and the tools can list it: which payload ops there are to target, the order of each transform op's results, which pattern sets `transform.apply_patterns` accepts and with which options, and which passes take the result down to the LLVM dialect.

| Part of the schedule | Lines | Where it comes from |
|---|---|---|
| `@__transform_main`: matching the bias, the convolution and the ReLU | `full.mlir` 118–131 | The payload's ops, listed by the generic printer (section 9.1) |
| Tiling, fusion, reduction tiling, generalization | `full.mlir` 147–174, 212–217, 231–232 | One transform op per Halide directive (section 3.2); tile sizes from the schedule's `tile_w`, `tile_h`, `vec` (sections 4.2–4.4); the order of each op's results from its ODS record (sections 9.2, 9.3) |
| The empty `transform.apply_patterns` sweep | `full.mlir` 189–192 | Dead-code elimination after fusion (section 4.3) |
| `transform.include @lower` and `{transform.consumed}` | `full.mlir` 235–236, 243–244 | A named sequence used as a macro (section 8.1) |
| Simplification and vectorization in `@lower` | `full.mlir` 258–297 | The pattern-set ops of the installed `*TransformOps.td` files (section 9.4) |
| Bufferization and `apply_registered_pass "buffer-deallocation-pipeline"` | `full.mlir` 319–326 | A pipeline listed by `mlir-opt --help` (section 9.6) |
| The vector lowering pattern sets and their strategies | `full.mlir` 332–373 | The pattern-set ops (section 9.4) and the strategies their parser accepts (section 9.5) |
| `driver/lower-to-llvm.txt` | all | The ops left after the schedule, each needing a stock pass (section 9.6) |

Each step can be checked the way sections 4–7 do it: write one stage file at a time, so that `diff` between consecutive stages shows one step (section 4.1), and look at the payload after it.

### 9.1 Step 1: list the payload ops the schedule must target

The schedule starts by finding its targets by name. The generic printer (`--mlir-print-op-generic`) writes every op under its full quoted name, so `grep` can count them; a second `grep` shows the order of the Linalg ops in the payload:

```bash
cd /Users/roy/study/mlir/transform/ChH
F=../../test_Example/transform/ChH/full.mlir
sed -n '8,86p' $F | mlir-opt --mlir-print-op-generic | grep -oE '"[a-z_]+\.[a-z_]+"' | sort | uniq -c
sed -n '8,86p' $F | grep -nE '= linalg\.'
```

Real output **[18]**:

```
   1 "arith.addf"
   1 "arith.constant"
   1 "arith.mulf"
   1 "builtin.module"
   1 "func.func"
   1 "func.return"
   1 "linalg.broadcast"
   2 "linalg.generic"
   3 "linalg.yield"
   1 "tensor.empty"
37:  %biased = linalg.broadcast ins(%bias : !tbias)
44:  %convolved = linalg.generic {
64:  %relued = linalg.generic {
```

- Three structured ops do the work: one `linalg.broadcast` (the bias) and two `linalg.generic`s (the convolution and the ReLU, section 2.3). The three `linalg.yield`s end their bodies; the broadcast has one in its generic form too.
- The broadcast is the only op of its name, so `structured.match ops{["linalg.broadcast"]}` gives a handle to it alone. The two generics share a name, so one match gives one handle to both, and `transform.split_handle` splits it in payload order: the convolution (line 44) first, then the ReLU (line 64).

### 9.2 Step 2: find the order of each transform op's results

Section 3.2 picks the transform op for each directive. Chaining them through handles also needs the order of each op's results, which the op's ODS (Operation Definition Specification) record in the installed `.td` file declares:

```bash
cd /Users/roy/study/mlir/transform/ChH
I=/opt/homebrew/opt/llvm@20/include/mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.td
awk '/Op<Transform_Dialect, "structured\.(tile_using_forall|tile_using_for|tile_reduction_using_for|fuse_into_containing_op)"/ {
       match($0, /"[a-z_.]+"/); op = substr($0, RSTART, RLENGTH); want = 1 }
     want && /let results = \(outs/ {
       r = $0; while (r !~ /\);/) { getline l; r = r l }
       gsub(/[ \t]+/, " ", r); gsub(/TransformHandleTypeInterface:/, "", r); print op ":" r; want = 0 }' $I
```

Real output **[19]**:

```
"structured.fuse_into_containing_op": let results = (outs $fused_op, $new_containing_op);
"structured.tile_reduction_using_for": let results = (outs Variadic<TransformHandleTypeInterface>:$fill_op, $split_op, $combining_op, $for_op);
"structured.tile_using_for": let results = (outs $tiled_linalg_op, Variadic<TransformHandleTypeInterface>:$loops);
"structured.tile_using_forall": let results = (outs $tiled_op, $forall_op);
```

- Every tiling op returns **the tiled op first, then the loops**: `%relu2, %co = ... tile_using_forall` (section 4.2), and `tile_using_for` returns one handle per loop it creates, so tiling `xi` and `ci` gives `%conv5, %conv_xi, %conv_ci` (section 4.5).
- `fuse_into_containing_op` returns the fused op, then the containing loop again: fusion consumes the loop handle, so the next fusion into the same loop uses the new one (section 4.3).
- `tile_reduction_using_for` returns the four results of section 4.4, in this order: fill, split op, combiner, loop.

### 9.3 Step 3: let the interpreter reject swapped results

Getting the order wrong isn't caught when the script is parsed: every result has the same type, `!transform.any_op`. It fails when the next op runs on the wrong payload. A copy of stage 1 with the first tiling's results swapped (by `sed`, in a temporary directory, run from there so the file name in the error is stable):

```bash
cd /Users/roy/study/mlir/transform/ChH
T=$(mktemp -d)
sed 's/%relu2, %co = transform.structured.tile_using_forall %relu$/%co, %relu2 = transform.structured.tile_using_forall %relu/' \
  stages/step1-tile-relu.mlir > $T/swapped.mlir
sed -n '8,86p' ../../test_Example/transform/ChH/full.mlir |
  (cd $T && mlir-opt --transform-preload-library=transform-library-paths=swapped.mlir \
                     --transform-interpreter > /dev/null); echo "mlir-opt exit status: $?"
rm -r $T
```

Real output **[20]**:

```
swapped.mlir:15:23: error: only TilingInterface ops are supported
    %relu3, %n_y_xo = transform.structured.tile_using_forall %relu2
                      ^
<stdin>:64:13: note: target op
  %relued = linalg.generic {
            ^
mlir-opt exit status: 1
```

- `%relu2` now holds the `scf.forall` loop, and the second tiling rejects it: a loop doesn't implement `TilingInterface`, the interface every tileable structured op implements.
- The note points to the ReLU's line because the loops that tiling creates take the location of the op they were made from. The error names the transform op that failed, not the one that bound the wrong handle, so the ODS record of section 9.2 is the place to check the order.

### 9.4 Step 4: find the pattern sets for `transform.apply_patterns`

`transform.apply_patterns` accepts only pattern-set ops, each a named group of rewrite patterns that a dialect exposes. Their mnemonics all start with `apply_patterns.`, so the installed `.td` files list them; then, for each pattern set the schedule uses, the file that defines it:

```bash
cd /Users/roy/study/mlir/transform/ChH
F=../../test_Example/transform/ChH/full.mlir
I=/opt/homebrew/opt/llvm@20/include
grep -rhoE '"apply_patterns\.[a-z_.]+"' $I/mlir --include='*.td' | sort -u |
  sed -E 's/"apply_patterns\.([a-z_]+)\..*/\1/; s/"apply_patterns\.[a-z_]+"$/(transform)/' | sort | uniq -c | sort -rn
for p in $(grep -oE 'transform\.apply_patterns\.[a-z_.]+' $F | sed 's/^transform\.//' | sort -u); do
  printf '%-66s %s\n' "$p" "$(grep -rlE "\"$p\"" $I/mlir --include='*.td' | sed "s|$I/||")"
done
```

Real output **[21]**:

```
  26 vector
   9 tensor
   8 linalg
   6 memref
   3 gpu
   1 scf
   1 (transform)
apply_patterns.canonicalization                                    mlir/Dialect/Transform/IR/TransformOps.td
apply_patterns.linalg.fold_unit_extent_dims_via_reshapes           mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.td
apply_patterns.linalg.tiling_canonicalization                      mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.td
apply_patterns.memref.alloc_to_alloca                              mlir/Dialect/MemRef/TransformOps/MemRefTransformOps.td
apply_patterns.memref.fold_memref_alias_ops                        mlir/Dialect/MemRef/TransformOps/MemRefTransformOps.td
apply_patterns.tensor.fold_tensor_subset_ops_into_vector_transfers mlir/Dialect/Tensor/TransformOps/TensorTransformOps.td
apply_patterns.vector.lower_contraction                            mlir/Dialect/Vector/TransformOps/VectorTransformOps.td
apply_patterns.vector.lower_shape_cast                             mlir/Dialect/Vector/TransformOps/VectorTransformOps.td
apply_patterns.vector.lower_transfer                               mlir/Dialect/Vector/TransformOps/VectorTransformOps.td
apply_patterns.vector.lower_transpose                              mlir/Dialect/Vector/TransformOps/VectorTransformOps.td
apply_patterns.vector.transfer_to_scf                              mlir/Dialect/Vector/TransformOps/VectorTransformOps.td
```

- MLIR 20 installs 54 pattern sets, and the schedule uses 11. Each lives with the dialect whose ops it rewrites; `vector` has the most, because lowering vector ops is mostly pattern-based (section 7.2).
- `canonicalization` is the Transform dialect's own: it collects the canonicalization patterns of all loaded dialects and ops, the ones the `canonicalize` pass runs.
- The `description` of each pattern set in its `.td` file says what it rewrites, which is how to choose among the 54.

### 9.5 Step 5: let the parser list a pattern set's strategies

Some pattern sets take an option, such as `lowering_strategy` (section 7.2). An invalid value makes the parser list the valid ones:

```bash
cd /Users/roy/study/mlir/transform/ChH
mlir-opt <<'EOF'
module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    transform.apply_patterns to %arg0 {
      transform.apply_patterns.vector.lower_contraction lowering_strategy = ?
    } : !transform.any_op
    transform.yield
  }
}
EOF
mlir-opt <<'EOF'
module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    transform.apply_patterns to %arg0 {
      transform.apply_patterns.vector.lower_transpose lowering_strategy = ?
    } : !transform.any_op
    transform.yield
  }
}
EOF
```

Real output **[22]** (exit status 1):

```
<stdin>:4:77: error: custom op 'transform.apply_patterns.vector.lower_contraction' expected string or keyword containing one of the following enum values for attribute 'lowering_strategy' [dot, matmulintrinsics, outerproduct, parallelarith]
      transform.apply_patterns.vector.lower_contraction lowering_strategy = ?
                                                                            ^
<stdin>:4:75: error: custom op 'transform.apply_patterns.vector.lower_transpose' expected string or keyword containing one of the following enum values for attribute 'lowering_strategy' [eltwise, flat_transpose, shuffle_1d, shuffle_16x16]
      transform.apply_patterns.vector.lower_transpose lowering_strategy = ?
                                                                          ^
```

The schedule takes `parallelarith` and `eltwise`, the simplest strategy of each (section 7.2); the others are the knobs to try when retuning for a target (section 8.5).

### 9.6 Step 6: let the ops left after the schedule pick the lowering passes

The schedule ends with mixed-dialect IR on buffers, and `driver/lower-to-llvm.txt` (appendix B.2) takes it to the LLVM dialect. Which passes it needs follows from what is left. The command prints the ops left after the full schedule, grouped by dialect; then the registered pipeline the schedule runs by name; then a check that every pass in `driver/lower-to-llvm.txt` is a stock one:

```bash
cd /Users/roy/study/mlir/transform/ChH
F=../../test_Example/transform/ChH/full.mlir
T=$(mktemp -d)
sed -n '90,377p' $F > $T/schedule.mlir
sed -n '8,86p' $F |
  mlir-opt --transform-preload-library=transform-library-paths=$T/schedule.mlir \
           --transform-interpreter --mlir-print-op-generic |
  grep -oE '"[a-z_]+\.[a-z_.]+"' | tr -d '"' | sort -u |
  awk -F. '{ ops[$1] = ops[$1] " " substr($0, length($1) + 2) } END { for (d in ops) print d ":" ops[d] }' | sort
rm -r $T
mlir-opt --help | grep -E -- '--buffer-deallocation-pipeline '
P=$(tr -d ' \n' < driver/lower-to-llvm.txt | tr ',()' '\n\n\n' | grep -vE '^(builtin\.module|func\.func)?$' | sort -u)
n=0; for p in $P; do mlir-opt --help | grep -qE -- "--$p " && n=$((n + 1)); done
echo "$n of the $(echo "$P" | wc -l | tr -d ' ') distinct passes in driver/lower-to-llvm.txt are listed by mlir-opt --help"
```

Real output **[23]**:

```
affine: apply
arith: addf constant mulf
bufferization: clone
builtin: module
func: func return
linalg: generic yield
llvm: intr.maxnum
memref: alloca collapse_shape copy expand_shape load store subview
scf: for forall forall.in_parallel yield
vector: broadcast extract insert insertelement load transfer_read transfer_write type_cast
      --buffer-deallocation-pipeline                         -   The default pipeline for automatically inserting deallocation operations after one-shot bufferization. Deallocation operations (except `memref.realloc`) may not be present already.
19 of the 19 distinct passes in driver/lower-to-llvm.txt are listed by mlir-opt --help
```

`transform.apply_registered_pass` takes any name `mlir-opt --help` lists, pipelines included, which is where `"buffer-deallocation-pipeline"` comes from (section 7.1). For the lowering, each dialect left needs a pass, and each pass must run after the passes that create its input:

- **`scf.forall`** → `scf-forall-to-for`, then `buffer-loop-hoisting` for the `memref.alloca`s inside it (section 10, Pitfalls); `scf.for` → `convert-scf-to-cf`, and the `cf` branches it creates → `convert-cf-to-llvm`.
- **`bufferization.clone`** → `convert-bufferization-to-memref`.
- **`vector.transfer_read`/`transfer_write`** → `convert-vector-to-scf`, which creates more `scf` loops, so it runs before `convert-scf-to-cf`; the other vector ops → `convert-vector-to-llvm`.
- **`linalg.generic`**, the scalar ReLU (section 6.2) → `convert-linalg-to-loops`, again before `convert-scf-to-cf`.
- **`arith.mulf` + `addf`** → `math-uplift-to-fma`, which creates the `math.fma` that `convert-math-to-llvm` lowers (section 8.2); the remaining `arith` → `convert-arith-to-llvm`.
- **`affine.apply`** → `lower-affine`, run twice: `expand-strided-metadata`, which turns the `memref` views (`subview`, `expand_shape`, `collapse_shape`) into explicit address arithmetic before `finalize-memref-to-llvm`, creates `affine.apply` ops again.
- **`func`** → `convert-func-to-llvm`; `convert-index-to-llvm` and `reconcile-unrealized-casts` clean up after the conversions.

**A lowering pipeline is derived from what the schedule leaves behind: list the dialects that remain, give each its stock conversion pass, and order the passes so that each runs after the ones that create its input.**

---

## 10. Key Takeaways & Pitfalls

**Takeaways**

1. **Halide starts fused, Linalg starts distributed** (section 2.4): Halide's shared loop variables fuse everything; each Linalg op is its own loop nest, so `compute_at` becomes producer/consumer fusion, which recomputes exactly the slice the consumer needs (section 4.3).
2. **Loops are materialized in order by tiling** (sections 4.2, 4.5): `interchange` only permutes the implicit loops of one generic (section 3.3); tiling by 1 creates an explicit loop at the position you want, and tiling by the vector length keeps a vectorizable dimension.
3. **Reduction tiling builds partial results plus a combiner** (section 4.4), turning reduction dimensions into parallel ones inside the loops; it reorders floating-point additions, which `fastmath` allows.
4. **Patterns, passes and lowerings compose inside the script** (sections 6–7): any combination of pattern sets in one sweep, registered passes by name, bufferization as a transform op.
5. **Module-wide transformations invalidate handles**; matching again recovers them (section 7.5) — but multidimensional vectors make the whole tile-and-unroll stage unnecessary (section 8.1).
6. **A schedule encodes a target** (section 8.5): the AVX-512 tile spills on NEON; changing one tile size fixes it.
7. **The schedule is assembled from what the tools list** (section 9): the generic printer lists the ops to match, the ODS records give each transform op's result order, the `.td` files list the pattern sets and the parser their strategies, and the ops left after the schedule pick the lowering passes.

**Pitfalls**

- **The test's `RUN` line uses two test-only passes** that the installed `mlir-opt` doesn't have: preload the script as a library instead of erasing it, and spell out the lowering pipeline (appendix A).
- **`allocas` in parallel loops overflow the stack.** The `alloc_to_alloca` buffers sit inside the `scf.forall` bodies, which `buffer_loop_hoisting` doesn't hoist out of (they are parallel). Lowered to sequential loops, the 16 000 iterations each grow the stack and the executable crashes with `EXC_BAD_ACCESS`; this chapter's pipeline therefore runs `scf-forall-to-for` and then `buffer-loop-hoisting` first (appendix B.2).
- **`split_handle` order is post-order** for `structured.match` (section 7.5): inner ops come before the ops that contain them.
- **Tiling to `1x16` leaves 0-D transfers on strided views** that MLIR 20's stock pipeline can't lower (section 7.3), so the Halide-faithful stages 7 and 8 stop at the IR level.
- **`llvm.intr.maxnum` blocks vectorization** of the ReLU (section 6.2, appendix C).
- **Unroll inner loops before outer ones** while you hold handles to both (section 5).

---

## Appendix A: Replacing the test-only passes

The test's `RUN` line is:

***test_Example/transform/ChH/full.mlir***
```mlir
// RUN: mlir-opt %s --transform-interpreter \
// RUN:             --test-transform-dialect-erase-schedule \
// RUN:             --math-uplift-to-fma \
// RUN:             --convert-bufferization-to-memref \
// RUN:             --test-lower-to-llvm |\
// RUN: FileCheck %s
```

`--test-transform-dialect-erase-schedule` and `--test-lower-to-llvm` are registered only in the `mlir-opt` built with MLIR's test libraries:

```bash
cd /Users/roy/study/mlir/transform/ChH
mlir-opt ../../test_Example/transform/ChH/full.mlir --transform-interpreter \
  --test-transform-dialect-erase-schedule --test-lower-to-llvm > /dev/null
```

Real output **[24]** (exit status 1):

```
mlir-opt: Unknown command line argument '--test-transform-dialect-erase-schedule'.  Try: 'mlir-opt --help'
mlir-opt: Did you mean '--transform-dialect-check-uses'?
mlir-opt: Unknown command line argument '--test-lower-to-llvm'.  Try: 'mlir-opt --help'
mlir-opt: Did you mean '--convert-to-llvm'?
```

### A.1 Instead of erasing the schedule: preload it

`--test-transform-dialect-erase-schedule` walks the module after the interpreter has run and erases every op implementing the transform op interface, so that the lowering passes and `mlir-translate` only see the payload. Without it, the script stays in the output:

```bash
cd /Users/roy/study/mlir/transform/ChH
mlir-opt ../../test_Example/transform/ChH/full.mlir --transform-interpreter |
  grep -E '^ *(func\.func|module|transform\.named_sequence)' | cut -c1-80
```

Real output **[25]**:

```mlir
module {
  func.func @conv(%arg0: memref<5x82x102x128xf32> {bufferization.access = "read"
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    transform.named_sequence @lower(%arg0: !transform.any_op {transform.consumed
```

This chapter keeps the script out of the payload instead. `--transform-preload-library=transform-library-paths=<file>` parses a file of named sequences into the Transform dialect's library module before the interpreter runs, and `--transform-interpreter` finds its entry point (`@__transform_main` by default) there when the payload has none. The library file must hold the named sequences at its top level — a `module attributes {transform.with_named_sequence}` on its own, as in `stages/*.mlir` and lines 90–377 of `full.mlir` — and it must be a regular file (a `<(...)` process substitution is rejected as "neither a file nor a directory"), hence the temporary copy in sections 8.2–8.5.

### A.2 Instead of `--test-lower-to-llvm`: an explicit pipeline

`--test-lower-to-llvm` is a pass pipeline registered by the test library. In MLIR 20 it runs: `convert-vector-to-scf` and `convert-linalg-to-loops` on functions, `lower-affine`, `convert-scf-to-cf`, `canonicalize`, `cse`, `convert-vector-to-llvm`, `convert-math-to-llvm` on functions, `expand-strided-metadata`, `lower-affine`, `finalize-memref-to-llvm`, `convert-func-to-llvm`, `convert-arith-to-llvm`, `convert-cf-to-llvm`, `convert-index-to-llvm` and `reconcile-unrealized-casts`. `driver/lower-to-llvm.txt` (appendix B.2) is that list, preceded by the two passes of the `RUN` line and by `scf-forall-to-for` + `buffer-loop-hoisting` (section 10, Pitfalls); `FileCheck` accepts its output (section 8.2).

---

## Appendix B: The driver and the lowering pipeline

### B.1 `driver/driver.c`

With `llvm.emit_c_interface`, `func-to-llvm` emits, next to `@conv`, a wrapper `_mlir_ciface_conv` that takes each memref as a pointer to a *descriptor* struct and returns the result memref through a pointer passed first:

***driver/driver.c***
```c
// Memref descriptors as MLIR lowers them to LLVM (identity layout).
typedef struct {
  float *allocated, *aligned;
  int64_t offset, sizes[1], strides[1];
} Memref1D;
typedef struct {
  float *allocated, *aligned;
  int64_t offset, sizes[4], strides[4];
} Memref4D;

void _mlir_ciface_conv(Memref4D *result, Memref4D *input, Memref4D *filter,
                       Memref1D *bias, Memref4D *output);
```

- A descriptor holds the pointer `malloc` returned (`allocated`, for freeing), the aligned data pointer, an element offset, and the sizes and strides of each dimension. With the identity layout (section 7.1) the strides are the row-major products, which `wrap4` fills in for plain C arrays.
- The function may return a buffer other than `output` (bufferization decides), so the driver reads the result through `result.aligned + result.offset` and frees `result.allocated` when it isn't `output`.

***driver/driver.c***
```c
  // Reference: relu(bias[f] + sum_{kh,kw,c} filter[c][kh][kw][f] * input[n][y+kh][x+kw][c]),
  // with f innermost so that the C compiler can vectorize it.
  for (int n = 0; n < N; ++n)
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) {
        float acc[F];
        for (int f = 0; f < F; ++f) acc[f] = bias[f];
        for (int kh = 0; kh < KH; ++kh)
          for (int kw = 0; kw < KW; ++kw)
            for (int c = 0; c < C; ++c) {
              float v = in[((n * (H + 2) + y + kh) * (W + 2) + x + kw) * C + c];
              const float *frow = &fil[((c * KH + kh) * KW + kw) * F];
              for (int f = 0; f < F; ++f) acc[f] += frow[f] * v;
            }
        for (int f = 0; f < F; ++f)
          ref[((n * H + y) * W + x) * F + f] = acc[f] > 0 ? acc[f] : 0;
      }
```

The reference is the same formula as the Linalg payload (filter indexed `[c][kh][kw][f]` = `(rx, rz, ry, c)`, input `[n][y+kh][x+kw][c]`), written with the output channel innermost: `clang -O3` vectorizes that loop, so the reference doubles as a baseline of what a C compiler does on its own.

`clang -Wno-override-module` silences clang's warning that it overrides the target triple of the module `mlir-translate` produced with the host's (`arm64-apple-macosx...`).

### B.2 `driver/lower-to-llvm.txt`

***driver/lower-to-llvm.txt***
```
builtin.module(
  func.func(scf-forall-to-for, buffer-loop-hoisting),
  math-uplift-to-fma,
  convert-bufferization-to-memref,
  func.func(convert-vector-to-scf, convert-linalg-to-loops),
  lower-affine,
  convert-scf-to-cf,
  canonicalize,
  cse,
  convert-vector-to-llvm,
  func.func(convert-math-to-llvm),
  expand-strided-metadata,
  lower-affine,
  finalize-memref-to-llvm,
  convert-func-to-llvm,
  convert-arith-to-llvm,
  convert-cf-to-llvm,
  convert-index-to-llvm,
  reconcile-unrealized-casts
)
```

The commands pass it as `--pass-pipeline="$(tr -d ' \n' < driver/lower-to-llvm.txt)"`: `tr` deletes the spaces and newlines, giving the one-line textual pipeline `mlir-opt` expects. Passes that run on functions are nested in `func.func(...)`: `buffer-loop-hoisting`, for one, refuses to be added to the module-level pass manager.

- **`scf-forall-to-for`, `buffer-loop-hoisting`**: the bufferized code has `memref.alloca` buffers inside the `scf.forall` bodies (section 10, Pitfalls). Once the loops are sequential, `buffer-loop-hoisting` moves those allocations before the loops, so the stack holds one copy instead of one per iteration.
- **`math-uplift-to-fma`, `convert-bufferization-to-memref`**: from the `RUN` line; the first fuses `fastmath` `mulf`+`addf` into `math.fma`, the second lowers `bufferization.clone` to `memref.alloc` + `memref.copy`.
- **The rest** is `--test-lower-to-llvm` (appendix A.2): leftover vector transfers and Linalg ops (the ReLU) become loops, loops become branches, and every dialect is converted to the LLVM dialect.

Without the first line, the executable builds but crashes: `sed` deletes the line from the pipeline before it is used.

```bash
cd /Users/roy/study/mlir/transform/ChH
F=../../test_Example/transform/ChH/full.mlir
T=$(mktemp -d)
sed -n '90,377p' $F > $T/schedule.mlir
sed -n '8,86p' $F |
  mlir-opt --transform-preload-library=transform-library-paths=$T/schedule.mlir \
           --transform-interpreter |
  mlir-opt --pass-pipeline="$(sed '/scf-forall-to-for/d' driver/lower-to-llvm.txt | tr -d ' \n')" |
  mlir-translate --mlir-to-llvmir -o $T/conv.ll
clang -O3 -Wno-override-module driver/driver.c $T/conv.ll -o $T/conv
$T/conv; echo "conv exit status: $?"
rm -r $T
```

Real output **[26]** (bash's own `Segmentation fault: 11` report, which carries a process ID, is left out):

```
conv exit status: 139
```

Exit status 139 is 128 + 11, signal 11, `SIGSEGV`: the stack overflowed. The `memref.alloca`s inside the `co` and `n, y, xo` loop bodies are lowered to `llvm.alloca`s that are only released when `@conv` returns, so each of the 16 000 iterations takes another few kilobytes of stack, far beyond the default 8 MB.

---

## Appendix C: Vectorizing the ReLU

Section 6.2 showed the ReLU staying a scalar `linalg.generic` because its body is `llvm.intr.maxnum`. Replacing it with `arith.maxnumf` — the `arith` op with the same semantics (IEEE `maxNum`: the non-NaN operand wins) — lets the vectorizer widen it. The run patches the payload with `sed` and counts what is left after the full schedule:

```bash
cd /Users/roy/study/mlir/transform/ChH
F=../../test_Example/transform/ChH/full.mlir
T=$(mktemp -d)
sed -n '90,377p' $F > $T/schedule.mlir
sed -n '8,86p' $F | sed 's/llvm.intr.maxnum(%cst, %in) : (f32, f32) -> f32/arith.maxnumf %cst, %in : f32/' |
  mlir-opt --transform-preload-library=transform-library-paths=$T/schedule.mlir \
           --transform-interpreter > $T/out.mlir
echo "linalg.generic left: $(grep -c 'linalg.generic' $T/out.mlir)"
grep -oE 'arith.maxnumf .*' $T/out.mlir
rm -r $T
```

Real output **[27]**:

```
linalg.generic left: 0
arith.maxnumf %661, %cst_2 : vector<5x64xf32>
```

No `linalg.generic` is left, and the maximum is computed on a whole `vector<5x64xf32>` at once. The vectorizer works from what it knows about each op, not from what an op means: two ops with the same semantics can vectorize differently.

---

## Appendix D: `examples.sh`

`examples.sh` (a symlink to `transform/examples.sh`) runs the numbered commands of this README, reading them straight from the `bash` blocks:

```
./examples.sh          # run every command, in README order
./examples.sh 15       # run command [15] only (several numbers are fine)
./examples.sh -l       # list the numbered commands without running them
```

Real output of `./examples.sh -l`:

```
[ 1] 2.4   sed -n '8,86p' ../../test_Example/transform/ChH/full.mlir |
[ 2] 3.3   sed -n '8,86p' ../../test_Example/transform/ChH/full.mlir |
[ 3] 3.3   sed -n '8,86p' ../../test_Example/transform/ChH/full.mlir |
[ 4] 4.2   sed -n '8,86p' ../../test_Example/transform/ChH/full.mlir |
[ 5] 4.3   sed -n '8,86p' ../../test_Example/transform/ChH/full.mlir |
[ 6] 4.4   sed -n '8,86p' ../../test_Example/transform/ChH/full.mlir |
[ 7] 4.5   sed -n '8,86p' ../../test_Example/transform/ChH/full.mlir |
[ 8] 5     sed -n '8,86p' ../../test_Example/transform/ChH/full.mlir |
[ 9] 6.2   sed -n '8,86p' ../../test_Example/transform/ChH/full.mlir |
[10] 6.2   sed -n '8,86p' ../../test_Example/transform/ChH/full.mlir |
[11] 7.2   sed -n '8,86p' ../../test_Example/transform/ChH/full.mlir |
[12] 7.3   sed -n '8,86p' ../../test_Example/transform/ChH/full.mlir |
[13] 7.5   T=$(mktemp -d)
[14] 8.2   F=../../test_Example/transform/ChH/full.mlir
[15] 8.3   F=../../test_Example/transform/ChH/full.mlir
[16] 8.4   F=../../test_Example/transform/ChH/full.mlir
[17] 8.5   F=../../test_Example/transform/ChH/full.mlir
[18] 9.1   F=../../test_Example/transform/ChH/full.mlir
[19] 9.2   I=/opt/homebrew/opt/llvm@20/include/mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.td
[20] 9.3   T=$(mktemp -d)
[21] 9.4   F=../../test_Example/transform/ChH/full.mlir
[22] 9.5   mlir-opt <<'EOF'
[23] 9.6   F=../../test_Example/transform/ChH/full.mlir
[24] A     mlir-opt ../../test_Example/transform/ChH/full.mlir --transform-interpreter \
[25] A.1   mlir-opt ../../test_Example/transform/ChH/full.mlir --transform-interpreter |
[26] B.2   F=../../test_Example/transform/ChH/full.mlir
[27] C     F=../../test_Example/transform/ChH/full.mlir
```

The commands of sections 8.3–8.5 and appendix B.2 compile and run the convolution and its C reference several times; each takes a few seconds.

---

## Links

- [Chapter H: Reproducing Halide Schedule](https://mlir.llvm.org/docs/Tutorials/transform/ChH/) (official tutorial)
- [Halide `conv_layer` application](https://github.com/halide/Halide/tree/294f80c49bf3bb8582446613c25fcce03b82bcd8/apps/conv_layer)
- [Transform dialect reference](https://mlir.llvm.org/docs/Dialects/Transform/)
- [Linalg dialect rationale](https://mlir.llvm.org/docs/Rationale/RationaleLinalgDialect/)
- [Bufferization](https://mlir.llvm.org/docs/Bufferization/)
