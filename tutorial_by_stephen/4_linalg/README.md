# 4 — Linear Algebra: the `linalg` Dialect

### Stop writing loops. Name the operation.

Chapter 3 made loops *first-class* so the compiler could optimize them — but you
still **wrote the loops**. You spelled out the three nested `affine.for`s of a
matrix multiply, and only then did the optimizer get to work. `linalg` takes the
next step and asks a sharper question: why write loops at all? A matrix multiply
*is* a matrix multiply. Say **that**, and let the compiler generate the loops,
choose their order, tile them, and fuse them with their neighbours.

That is the whole idea of the `linalg` dialect: a small set of **structured
operations** that describe linear-algebra computations *declaratively* — the
**what**, not the **how**. `linalg.matmul`, `linalg.add`, `linalg.broadcast`,
convolutions, reductions — each is a single op that still carries enough structure
for the compiler to reason about. This is the level real ML compilers actually
target, because chains of these ops (a neural network is little else) can be fused
and tiled *as a group* in ways hand-written loops never could.

Think of it as one more rung above Chapter 3. There, an `affine.for` nest was a
*shape* the polyhedral model could slice and tile — but the shape was still
implied by loops you typed out. Here the operation names itself, and the loop
nest is something the compiler *derives on demand*. You have climbed from "loops
the compiler can analyze" to "loops the compiler writes for you."

> Based on Stephen Diehl's *"MLIR Part 4 — Linear Algebra in MLIR"*
> ([`../reference/`](../reference/)). Every runnable example is built and checked
> against NumPy on this toolchain (Homebrew LLVM 20.1.8, Apple Silicon).

---

## What a structured op actually is

A `linalg` op is **payload + structure**. The *payload* is the scalar math (an
`addf`, a `mulf`) written in a region; the *structure* is the iteration space and
how each tensor is indexed within it. Nothing in between — no loop keywords, no
index arithmetic. The op *is* the pair:

```text
   ┌─────────────────────── linalg.generic ───────────────────────┐
   │  STRUCTURE                          PAYLOAD                    │
   │  indexing_maps  = [...]   ─┐        ^bb0(%a, %b, %out):        │
   │  iterator_types = [...]    ├──────►   %s = arith.addf %a, %b   │
   │  ins(...) outs(...)       ─┘          linalg.yield %s          │
   │  "which element, in what order"     "what to compute per elt"  │
   └───────────────────────────────────────────────────────────────┘
      the compiler expands this into a loop nest on demand
```

The most general op, `linalg.generic`, spells the structure out explicitly with
two attributes:

- **`indexing_maps`** — one affine map per operand (inputs *and* outputs), saying
  how that tensor is indexed by the loop variables. For elementwise add, all three
  maps are `(d0) -> (d0)`: element `i` of each tensor maps to loop index `i`. For
  matmul they are `(i,j,k)->(i,k)`, `(i,j,k)->(k,j)`, `(i,j,k)->(i,j)` — exactly
  $A_{ik}$, $B_{kj}$, $C_{ij}$. (These are the same affine maps from Chapter 3,
  now used to *describe* an op rather than index a hand-written loop.)
- **`iterator_types`** — one per loop dimension, classifying it:

  | Iterator | Meaning | Example |
  | --- | --- | --- |
  | `parallel` | iterations are independent | elementwise; the `i, j` of matmul |
  | `reduction` | iterations accumulate into one value | the `k` sum of matmul |
  | `window` | iterations slide a window | convolution |

Everything else in the dialect is `linalg.generic` with the boilerplate hidden:

| Level | Op | What it is |
| --- | --- | --- |
| Most general | `linalg.generic` | the raw payload+structure form |
| Elementwise | `linalg.map { op }`, `linalg.add`/`sub`/`mul`/`div` | apply a scalar op per element |
| Reduction | `linalg.reduce { op }` | collapse a dimension (sum, product, …) |
| Contraction | `linalg.matmul` (and friends) | the named matrix multiply |
| Shape | `linalg.broadcast` | stretch a smaller tensor to a larger shape |

Because the structure is explicit, lowering is just *generating the loop nest the
maps imply* — `-convert-linalg-to-loops` (→ `scf`) or
`-convert-linalg-to-affine-loops` (→ `affine`, Chapter 3). From there it's the
same pipeline as every earlier chapter.

### Tensors vs. memrefs at this level

Two of these steps operate on **tensors** (SSA values, Chapter 2) and two on
**memrefs** (buffers). The rule is the one from Chapter 2: a named op writing
into a caller-provided memref `outs` buffer works *in place* and needs no
allocation, so it lowers straight to loops. A `linalg` op that *returns* a
`tensor` must first be **bufferized** (tensors → memrefs) before it can run —
which is exactly why the *runnable* named ops in Steps 1–4 (memref) skip
bufferization, while any path that lowers a tensor op runs a `--one-shot-bufferize`
step first: the inspect `linalg.generic` forms in Steps 1–2, the fused kernel in
Step 5, and the tiling in Step 6 all do. (Step 5 also returns a tensor, so it adds
`-buffer-results-to-out-params` to turn the result into an in-place buffer.) Same
ops, different data model.

---

## Chapter layout

| Step | Directory | Topic | Runnable? |
| --- | --- | --- | --- |
| 1 | [`1_elementwise/`](1_elementwise/) | `linalg.generic`, `linalg.map`, named ops | ✅ Python |
| 2 | [`2_reduce/`](2_reduce/) | `linalg.reduce` + the `reduction` iterator | ✅ Python |
| 3 | [`3_matmul/`](3_matmul/) | `linalg.matmul` (and its generic form) | ✅ Python |
| 4 | [`4_broadcast/`](4_broadcast/) | `linalg.broadcast` + elementwise add | ✅ Python |
| 5 | [`5_fusion/`](5_fusion/) | fusing elementwise ops into one loop | ✅ Python |
| 6 | [`6_tiling/`](6_tiling/) | tiling a matmul for cache locality | 🧩 inspect |

Shared helper: [`common/np_memref.py`](common/np_memref.py) — 1-D **and** 2-D
memref descriptors + NumPy adapters (the same zero-copy ctypes bridge as
Chapters 2–3).

**Legend:** ✅ runnable · 🧩 snippet (transformed/inspected with `mlir-opt`, not run).

---

## Step 1 — Elementwise: `generic`, `map`, named ops (`1_elementwise/`) · ✅

**Goal:** meet the fundamental `linalg.generic`, then watch it collapse to a
one-liner.

`generic_add.mlir` writes an elementwise tensor add **the long way** — three
identity `indexing_maps`, one `parallel` iterator, and a region that does the
`arith.addf`. It's verbose on purpose: every other Linalg op is this op with the
boilerplate removed.

*1_elementwise/generic_add.mlir*
```mlir
#map_1d = affine_map<(d0) -> (d0)>

func.func @add_tensors(%arg0: tensor<10xf32>, %arg1: tensor<10xf32>) -> tensor<10xf32> {
  %result = tensor.empty() : tensor<10xf32>
  %0 = linalg.generic {
      indexing_maps  = [#map_1d, #map_1d, #map_1d],   // arg0, arg1, result
      iterator_types = ["parallel"]
    }
    ins(%arg0, %arg1 : tensor<10xf32>, tensor<10xf32>)
    outs(%result : tensor<10xf32>) {
  ^bb0(%a: f32, %b: f32, %out: f32):
    %1 = arith.addf %a, %b : f32
    linalg.yield %1 : f32
  } -> tensor<10xf32>
  return %0 : tensor<10xf32>
}
```

**Reading the region.** The block `^bb0(%a, %b, %out)` has **one argument per
operand** — one for each `ins()` value *and* one for each `outs()` value, in order:

| Block arg | Comes from | Role |
| --- | --- | --- |
| `%a` | `ins` `%arg0` | current input element of `arg0` |
| `%b` | `ins` `%arg1` | current input element of `arg1` |
| `%out` | `outs` `%result` | current value of the output element |

Two things trip people up here:

- **`%out` looks unused — and that's correct.** `%out` is the *current* value of
  the output element (useful when you accumulate into it, e.g. the running sum of a
  reduction — see Step 2, and the `%acc` of matmul in Step 3). A plain elementwise
  write doesn't read the old value, so `%out` is unused. It must still be declared:
  the block's argument count has to match the operand count.
- **The result is written by `linalg.yield`, not by assigning to `%out`.** You
  compute a *fresh* value `%1 = arith.addf %a, %b`, and `linalg.yield %1` is what
  stores that value into the output element (the one `%out` read). You cannot write
  `%out = arith.addf %a, %b`: `%out` is already a defined SSA value (a block arg),
  and SSA values are immutable — MLIR rejects it with *"redefinition of SSA value
  '%out'"*. So the flow is always: read args → compute a new value → `yield` it,
  and the yield is the store back to the `outs` element.

"Structured" means the loops are *implied*, not written. Ask the compiler to make
them concrete and the maps become a plain loop nest — but note the catch:
`-convert-linalg-to-loops` only lowers linalg ops on **memrefs**, not tensors. This
op works on `tensor<10xf32>`, so we must **bufferize first** (tensors → memrefs,
Chapter 2) to give the loops memory to load from and store into:

```bash
$ mlir-opt 1_elementwise/generic_add.mlir \
    -one-shot-bufferize="bufferize-function-boundaries" \
    -convert-linalg-to-loops
```

(Run `-convert-linalg-to-loops` *without* the bufferize step and nothing happens —
the tensor op round-trips unchanged, since there's no memory to loop over yet.) The
lowered form is exactly the elementwise `scf.for` you'd have written by hand — the
identity maps become the `memref.load`/`memref.store` at index `i`, and the region's
`arith.addf` sits in the loop body. `build.sh` saves it to
`build/generic_add_loops.mlir`:

*build/generic_add_loops.mlir*
```mlir
module {
  func.func @add_tensors(%arg0: memref<10xf32, strided<[?], offset: ?>>, %arg1: memref<10xf32, strided<[?], offset: ?>>) -> memref<10xf32> {
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<10xf32>
    scf.for %arg2 = %c0 to %c10 step %c1 {
      %0 = memref.load %arg0[%arg2] : memref<10xf32, strided<[?], offset: ?>>
      %1 = memref.load %arg1[%arg2] : memref<10xf32, strided<[?], offset: ?>>
      %2 = arith.addf %0, %1 : f32
      memref.store %2, %alloc[%arg2] : memref<10xf32>
    }
    return %alloc : memref<10xf32>
  }
}
```

The region maps directly onto the loop body: `%out` was the element at `%alloc[i]`,
`%1 = arith.addf` became `%2 = arith.addf`, and the `linalg.yield` became the
`memref.store` back into `%alloc[i]`.

### One rung of sugar up: `linalg.map`

For a *pure elementwise* op, all of that structure is implied — every operand is
indexed `(d0)->(d0)` and every dimension is `parallel` — so you can drop it and
give only the scalar payload. `linalg.map` is exactly that shorthand:

*1_elementwise/map_add.mlir*
```mlir
func.func @map_add(%arg0: tensor<10xf32>, %arg1: tensor<10xf32>) -> tensor<10xf32> {
  %init = tensor.empty() : tensor<10xf32>
  %mapped = linalg.map { arith.addf }
      ins(%arg0, %arg1 : tensor<10xf32>, tensor<10xf32>)
      outs(%init : tensor<10xf32>)
  return %mapped : tensor<10xf32>
}
```

No `indexing_maps`, no `iterator_types`, no explicit region: `{ arith.addf }` *is*
the payload. The op's operands are the `ins()` elements in order (`%arg0[i]`,
`%arg1[i]`), and the result is stored to the `outs()` element — the same
read-args → compute → store flow the `linalg.generic` region spelled out by hand.
It's still on tensors, so lowering to loops needs the same bufferize step, and
produces the *identical* `scf.for` nest as `generic_add.mlir` above:

```bash
$ mlir-opt 1_elementwise/map_add.mlir \
    -one-shot-bufferize="bufferize-function-boundaries" \
    -convert-linalg-to-loops
```

`build.sh` saves this to `build/map_add_loops.mlir` — `diff` it against
`build/generic_add_loops.mlir` and the only difference is the function name
(`@map_add` vs `@add_tensors`); the loop nest is byte-for-byte identical,
confirming `linalg.map` is pure shorthand for the generic form.

### Fully sugared: the named ops (`linalg.add` & friends)

The **named op** is that whole thing as a single line — payload, structure, and all.
Elementwise arithmetic gets a whole family, each just `linalg.map` with the payload
op baked into the name:

| Named op | Payload | NumPy |
| --- | --- | --- |
| `linalg.add` | `arith.addf` / `arith.addi` | `a + b` |
| `linalg.sub` | `arith.subf` / `arith.subi` | `a - b` |
| `linalg.mul` | `arith.mulf` / `arith.muli` | `a * b` |
| `linalg.div` | `arith.divf` / `arith.divsi` | `a / b` |

(The op picks the float or integer arithmetic from its operand types; there's a
separate `linalg.div_unsigned` → `arith.divui` for unsigned integers.) There are
matching families for unary math — `linalg.exp`, `linalg.log`, `linalg.abs`,
`linalg.negf`, … — all the same idea: a named elementwise op with no region to
write. They all lower through the same generic machinery to the identical loop nest.

Here `linalg.add` writes into a caller-provided memref `outs` buffer in place, so —
unlike the tensor forms above — no bufferization is needed:

*1_elementwise/add.mlir*
```mlir
func.func @addv(%a: memref<10xf32>, %b: memref<10xf32>, %c: memref<10xf32>)
    attributes {llvm.emit_c_interface} {
  linalg.add ins(%a, %b : memref<10xf32>, memref<10xf32>)
             outs(%c : memref<10xf32>)
  return
}
```

`build.sh` lowers this named version to a runnable library (and saves the
loop-lowered generic form above for comparison). The driver checks `c == a + b`.

**Run:** `cd 1_elementwise && bash build.sh`

```
linalg.add successful!
[10. 11. 12. 13. 14. 15. 16. 17. 18. 19.]
```

---

## Step 2 — Reduction: the `reduction` iterator (`2_reduce/`) · ✅

**Goal:** meet the second iterator type — the one that makes `%out` come alive.

Every op in Step 1 was `["parallel"]`: iterations independent, `%out` never read.
A **reduction** is the opposite — many inputs collapse into one output, so each
iteration reads the running total in `%out`, combines the next input, and writes it
back. This is precisely the accumulation the "unused `%out`" in Step 1 was hinting
at. We sum each row of a 3×4 matrix into a length-3 vector (NumPy `a.sum(axis=1)`):

```text
   input 3×4              collapse dim 1 (columns)        output (3,)
   [ 0  1  2  3]          Σ over each row                 [ 6]
   [ 4  5  6  7]   ───────────────────────────────►      [22]
   [ 8  9 10 11]          i = parallel, j = reduction     [38]
```

The `linalg.generic` form makes the two iterators explicit. Note the output map
drops `j` — that missing index is what marks the collapsed axis — and the region now
**reads `%out`**:

*2_reduce/reduce_generic.mlir*
```mlir
#map_in  = affine_map<(i, j) -> (i, j)>
#map_out = affine_map<(i, j) -> (i)>          // drops j -> j is the reduced axis

func.func @row_sum(%input: tensor<3x4xf32>, %init: tensor<3xf32>) -> tensor<3xf32> {
  %0 = linalg.generic {
      indexing_maps  = [#map_in, #map_out],
      iterator_types = ["parallel", "reduction"]    // i independent, j summed away
    }
    ins(%input : tensor<3x4xf32>)
    outs(%init : tensor<3xf32>) {
  ^bb0(%in: f32, %out: f32):
    %s = arith.addf %out, %in : f32               // read %out (running sum), accumulate
    linalg.yield %s : f32
  } -> tensor<3xf32>
  return %0 : tensor<3xf32>
}
```

Lowering (bufferize first, since it's on tensors) makes the accumulation concrete —
an **outer `parallel` loop over rows, an inner `reduction` loop** whose body loads
`%out`, adds, and stores it back:

```bash
$ mlir-opt 2_reduce/reduce_generic.mlir \
    -one-shot-bufferize="bufferize-function-boundaries" \
    -convert-linalg-to-loops
```

`build.sh` saves this to `build/reduce_generic_loops.mlir`:

*build/reduce_generic_loops.mlir*
```mlir
module {
  func.func @row_sum(%arg0: memref<3x4xf32, strided<[?, ?], offset: ?>>, %arg1: memref<3xf32, strided<[?], offset: ?>>) -> memref<3xf32, strided<[?], offset: ?>> {
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    scf.for %arg2 = %c0 to %c3 step %c1 {
      scf.for %arg3 = %c0 to %c4 step %c1 {
        %0 = memref.load %arg0[%arg2, %arg3] : memref<3x4xf32, strided<[?, ?], offset: ?>>
        %1 = memref.load %arg1[%arg2] : memref<3xf32, strided<[?], offset: ?>>
        %2 = arith.addf %1, %0 : f32
        memref.store %2, %arg1[%arg2] : memref<3xf32, strided<[?], offset: ?>>
      }
    }
    return %arg1 : memref<3xf32, strided<[?], offset: ?>>
  }
}
```

The outer `scf.for` is the `parallel` row loop; the inner one is the `reduction`
sweep over columns. Its body loads `%arg1[i]` (the running sum — that's `%out`),
adds the element, and stores it back — the accumulation `%out` made possible.

The **named op** is one line. `dimensions = [1]` names the axis to collapse; because
it accumulates into the caller's `%init` buffer in place, that buffer must arrive
**pre-zeroed** (the driver passes `np.zeros`), just like matmul's `%C`:

*2_reduce/reduce.mlir*
```mlir
func.func @row_sum(%input: memref<3x4xf32>, %init: memref<3xf32>)
    attributes {llvm.emit_c_interface} {
  linalg.reduce { arith.addf }
      ins(%input : memref<3x4xf32>)
      outs(%init : memref<3xf32>)
      dimensions = [1]
  return
}
```

Swap the payload for `arith.mulf` (product), `arith.maximumf` (max-pool), etc. — the
structure is identical, only the combine op changes. The driver checks against
`matrix.sum(axis=1)`.

**Run:** `cd 2_reduce && bash build.sh`

```
linalg.reduce successful!
[ 6. 22. 38.]
```

With `parallel` (Step 1) and `reduction` (here) in hand, matmul is just the two
combined — which is exactly Step 3.

---

## Step 3 — `linalg.matmul` (`3_matmul/`) · ✅

**Goal:** a whole matrix multiply as one op — contrast it with Chapter 3's
hand-written nest.

*3_matmul/matmul.mlir*
```mlir
func.func @matmul(%A: memref<8x10xf32>, %B: memref<10x16xf32>, %C: memref<8x16xf32>)
    attributes {llvm.emit_c_interface} {
  linalg.matmul ins(%A, %B : memref<8x10xf32>, memref<10x16xf32>)
                outs(%C : memref<8x16xf32>)
  return
}
```

That single line is sugar for the `linalg.generic` whose maps and iterators encode
$C_{ij} = \sum_k A_{ik}B_{kj}$ — the contraction where two indices are free and one
is summed away:

```text
        k                j                    j
      ┌────┐          ┌──────┐             ┌──────┐
   i  │ A  │    ×   k │  B   │     =     i  │  C   │
      └────┘          └──────┘             └──────┘
     (i,k)            (k,j)                (i,j)

   iterator_types: i = parallel   j = parallel   k = reduction
   C[i,j] = Σ_k A[i,k]·B[k,j]   — sweep k, accumulate; i,j independent
```

Written out, that single line is this `linalg.generic` — the same op with the
structure exposed:

*3_matmul/matmul_generic.mlir*
```mlir
#map_A = affine_map<(i, j, k) -> (i, k)>
#map_B = affine_map<(i, j, k) -> (k, j)>
#map_C = affine_map<(i, j, k) -> (i, j)>          // no k -> C's reduced axis

func.func @matmul(%A: memref<8x10xf32>, %B: memref<10x16xf32>, %C: memref<8x16xf32>)
    attributes {llvm.emit_c_interface} {
  linalg.generic {
      indexing_maps  = [#map_A, #map_B, #map_C],
      iterator_types = ["parallel", "parallel", "reduction"]   // i, j free; k summed
    }
    ins(%A, %B : memref<8x10xf32>, memref<10x16xf32>)
    outs(%C : memref<8x16xf32>) {
  ^bb0(%a: f32, %b: f32, %acc: f32):
    %p = arith.mulf %a, %b : f32
    %s = arith.addf %acc, %p : f32                 // read %acc (current C[i,j])
    linalg.yield %s : f32
  }
  return
}
```

This is Step 1's `parallel` and Step 2's `reduction` in one op: `i, j` are the
parallel output dimensions, `k` is the reduced one, and `%acc` is the read-and-write
accumulator you just saw in the reduction (here it holds the running dot product).

### `scf` loops vs. `affine` loops

Both `matmul.mlir` and `matmul_generic.mlir` lower to the same three-deep loop nest,
and there are **two passes** that generate it — they differ only in which loop
dialect they emit. `build.sh` runs the generic form through both:

```bash
$ mlir-opt 3_matmul/matmul_generic.mlir -convert-linalg-to-loops        # -> scf
$ mlir-opt 3_matmul/matmul_generic.mlir -convert-linalg-to-affine-loops # -> affine
```

*build/matmul_generic_scf.mlir*
```mlir
module {
  func.func @matmul(%arg0: memref<8x10xf32>, %arg1: memref<10x16xf32>, %arg2: memref<8x16xf32>) attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c10 = arith.constant 10 : index
    scf.for %arg3 = %c0 to %c8 step %c1 {
      scf.for %arg4 = %c0 to %c16 step %c1 {
        scf.for %arg5 = %c0 to %c10 step %c1 {
          %0 = memref.load %arg0[%arg3, %arg5] : memref<8x10xf32>
          %1 = memref.load %arg1[%arg5, %arg4] : memref<10x16xf32>
          %2 = memref.load %arg2[%arg3, %arg4] : memref<8x16xf32>
          %3 = arith.mulf %0, %1 : f32
          %4 = arith.addf %2, %3 : f32
          memref.store %4, %arg2[%arg3, %arg4] : memref<8x16xf32>
        }
      }
    }
    return
  }
}
```

*build/matmul_generic_affine.mlir*
```mlir
module {
  func.func @matmul(%arg0: memref<8x10xf32>, %arg1: memref<10x16xf32>, %arg2: memref<8x16xf32>) attributes {llvm.emit_c_interface} {
    affine.for %arg3 = 0 to 8 {
      affine.for %arg4 = 0 to 16 {
        affine.for %arg5 = 0 to 10 {
          %0 = affine.load %arg0[%arg3, %arg5] : memref<8x10xf32>
          %1 = affine.load %arg1[%arg5, %arg4] : memref<10x16xf32>
          %2 = affine.load %arg2[%arg3, %arg4] : memref<8x16xf32>
          %3 = arith.mulf %0, %1 : f32
          %4 = arith.addf %2, %3 : f32
          affine.store %4, %arg2[%arg3, %arg4] : memref<8x16xf32>
        }
      }
    }
    return
  }
}
```

Same arithmetic, two dialects — note how the `scf` form needs explicit `%c0…%c8`
index constants for its bounds, while the `affine` form writes `0 to 8` directly:

| | `-convert-linalg-to-loops` (`scf`) | `-convert-linalg-to-affine-loops` (`affine`) |
| --- | --- | --- |
| Loop op | `scf.for` (bounds are SSA values: `%c0` to `%c8`) | `affine.for %i = 0 to 8` (bounds are affine expressions) |
| Memory | `memref.load` / `memref.store` | `affine.load` / `affine.store` |
| Constraints | none — bounds/indices can be *any* values | bounds & subscripts must be **affine** in the loop vars |
| Analyzability | opaque to the polyhedral model | fully analyzable — tiling, fusion, interchange, vectorization |
| Level | **lower** (closer to raw imperative loops) | **higher** (carries structure for optimization) |

**Which is "higher"?** `affine` is the higher-level, more abstract form — it adds the
promise that every loop bound and array index is an affine function of the loop
variables, which is exactly the information Chapter 3's polyhedral transforms need.
`scf` is more general and lower-level: it can express loops `affine` can't (data-
dependent bounds, arbitrary indexing), but that generality means the optimizer can't
reason about it. The lowering ladder runs **linalg → affine → scf → cf → llvm**:
affine sits *above* scf and lowers to it via `-lower-affine`.

**When to use each:**

- **`-convert-linalg-to-affine-loops`** when you still want to *optimize the loop
  nest* — tile it, fuse it, interchange or vectorize it. This is why Step 6 (tiling)
  uses it: `-affine-loop-tile` only works on `affine.for`. It's also how you confirm
  the op *is* Chapter 3's matmul — this `affine.for` nest is identical to the one you
  wrote by hand there.
- **`-convert-linalg-to-loops`** when you're done optimizing and just want runnable
  loops on the way to LLVM (or when the op's access pattern isn't affine, so the
  affine path doesn't apply). This is why the *runnable* steps here take the `scf`
  route straight into the LLVM pipeline.

That's the whole point of the level: *one line* upstream, but by the time it's
loops it's exactly what you'd have hand-written — and the compiler was free to
choose the order. The driver checks the result against NumPy's `@` (the kernel
accumulates into `C`, so it's passed zeroed).

**Run:** `cd 3_matmul && bash build.sh`

```
linalg.matmul successful! (8x10 @ 10x16, max abs error 2.38e-07)
```

(Inputs are random each run, so the exact error varies; it's always a tiny
float-rounding value — the kernel and NumPy sum the `k` dimension in different
orders.)

---

## Step 4 — Broadcasting (`4_broadcast/`) · ✅

**Goal:** NumPy-style broadcasting — combining tensors of different shapes.

Broadcasting lets `linalg` combine tensors of *different* shapes in one operation,
following the same rules as NumPy. The smaller tensor is implicitly stretched along
the broadcast dimensions to match the larger one — so you can add a vector to every
row or column of a matrix, or fold a scalar into a whole tensor, without ever writing
out the expanded copy.

In NumPy the stretch alone is `np.broadcast_to` — replicating a length-3 vector down
three rows:

```python
arg0 = np.array([1, 2, 3])
result = np.broadcast_to(arg0, (3, 3))
# array([[1, 2, 3],
#        [1, 2, 3],
#        [1, 2, 3]])
```

That is exactly what `linalg.broadcast` does. Concretely, adding a length-4 vector
to every row of a 3×4 matrix is `matrix + vector`
in NumPy. The shapes don't match, so something has to *stretch* the vector to fill
the missing dimension:

```text
   vector (4,)        broadcast dim 0        matrix (3,4)      result (3,4)
   [1 2 3 4]   ──────────────────────►   [1 1 1 1]   +    [2 3 4 5]
                 replicate across            [1 1 1 1]   =    [2 3 4 5]
                 the 3 rows                  [1 1 1 1]        [2 3 4 5]
                                          (each row now [1,2,3,4])
```

In Linalg it's `linalg.broadcast` (stretch the vector across the rows) followed by
an elementwise add, both writing the caller's `%out` buffer in place:

*4_broadcast/add_vec_to_mat.mlir*
```mlir
func.func @add_vec_to_mat(%matrix: memref<3x4xf32>, %vector: memref<4xf32>, %out: memref<3x4xf32>)
    attributes {llvm.emit_c_interface} {
  // 1. broadcast the vector across the 3 rows of %out
  linalg.broadcast ins(%vector : memref<4xf32>)
                   outs(%out : memref<3x4xf32>)
                   dimensions = [0]
  // 2. add the matrix elementwise into %out (in place)
  linalg.add ins(%matrix, %out : memref<3x4xf32>, memref<3x4xf32>)
             outs(%out : memref<3x4xf32>)
  return
}
```

`dimensions = [0]` names the axis the input *lacks* — dimension 0, the rows — so
the vector is replicated along it. The driver checks against NumPy's own
broadcasting (`matrix + vector`).

**Run:** `cd 4_broadcast && bash build.sh`

```
linalg.broadcast + add successful!
[[2. 3. 4. 5.]
 [2. 3. 4. 5.]
 [2. 3. 4. 5.]]
```

---

## Step 5 — Kernel fusion (`5_fusion/`) · ✅

**Goal:** the optimization that makes structured ops pay off.

`separate_ops.mlir` computes `(a + b) * c` as **two** named ops on tensors. Run
as written, that's two loops: the add materializes a whole intermediate tensor to
memory, then the mul reads it back.

*5_fusion/separate_ops.mlir*
```mlir
module {
  func.func @addmul(%a: tensor<10xf32>, %b: tensor<10xf32>, %c: tensor<10xf32>) -> tensor<10xf32>
      attributes {llvm.emit_c_interface} {
    %0 = tensor.empty() : tensor<10xf32>
    %1 = linalg.add ins(%a, %b : tensor<10xf32>, tensor<10xf32>)      // t = a + b
                    outs(%0 : tensor<10xf32>) -> tensor<10xf32>
    %2 = tensor.empty() : tensor<10xf32>
    %3 = linalg.mul ins(%1, %c : tensor<10xf32>, tensor<10xf32>)      // r = t * c
                    outs(%2 : tensor<10xf32>) -> tensor<10xf32>
    return %3 : tensor<10xf32>
  }
}
```

**Fusion** merges the two into a single loop that reads each input once and writes
the result directly — no intermediate tensor, no second pass over memory:

```text
   separate                              fused
   ┌────────────────┐                    ┌──────────────────────┐
   │ for i: t[i]=a+b│  writes t          │ for i:               │
   └────────────────┘  ───► memory ───►  │   r[i] = (a[i]+b[i])  │
   ┌────────────────┐  reads t           │          * c[i]       │
   │ for i: r[i]=t*c│                    │  (t never exists)     │
   └────────────────┘                    └──────────────────────┘
   two loops + temp array                one loop, no temp
```

Run the fusion pipeline:

```bash
$ mlir-opt 5_fusion/separate_ops.mlir \
    --canonicalize \
    --linalg-fuse-elementwise-ops \
    --cse \
    --linalg-generalize-named-ops \
    --linalg-fuse-elementwise-ops
```

What each pass does:

| Pass | Role |
| --- | --- |
| `--canonicalize` | Standard cleanup — fold constants, drop dead ops, put the IR in a normal form so later passes match reliably. |
| `--linalg-fuse-elementwise-ops` | The fusion pass. It merges producer→consumer elementwise ops, **but only `linalg.generic` ops** — so on the *named* `add`/`mul` here it does nothing yet (that's why it runs a second time below). |
| `--cse` | Common-subexpression elimination — removes duplicated work (e.g. the two identical `tensor.empty()` allocations). |
| `--linalg-generalize-named-ops` | Rewrites the named `linalg.add` / `linalg.mul` into their `linalg.generic` form — the form fusion understands. |
| `--linalg-fuse-elementwise-ops` | Run **again**: now that both ops are generic, this is the pass that actually fuses them into one. |

The key ordering: fusion works on `linalg.generic`, so the named ops must be
*generalized* first. The first `--linalg-fuse-elementwise-ops` is a no-op on this
input (the ops are still named); the one after `--linalg-generalize-named-ops` is
the one that does the real merge.

The two named ops have become **one `linalg.generic`** whose body does both the
`addf` and the `mulf` before a single `linalg.yield` — note it now has four
`indexing_maps` (`a`, `b`, `c`, and the output all indexed `(d0)->(d0)`) and no
intermediate `tensor.empty` for `t`. `build.sh` writes this to
`build/fused_ops.mlir`:

*build/fused_ops.mlir*
```mlir
#map = affine_map<(d0) -> (d0)>
module {
  func.func @addmul(%arg0: tensor<10xf32>, %arg1: tensor<10xf32>, %arg2: tensor<10xf32>) -> tensor<10xf32> {
    %0 = tensor.empty() : tensor<10xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel"]} ins(%arg0, %arg1, %arg2 : tensor<10xf32>, tensor<10xf32>, tensor<10xf32>) outs(%0 : tensor<10xf32>) {
    ^bb0(%in: f32, %in_0: f32, %in_1: f32, %out: f32):
      %2 = arith.addf %in, %in_0 : f32
      %3 = arith.mulf %2, %in_1 : f32
      linalg.yield %3 : f32
    } -> tensor<10xf32>
    return %1 : tensor<10xf32>
  }
}
```

This is the bread-and-butter optimization for deep-learning graphs, which are
mostly long chains of elementwise ops — it's the `linalg`-op cousin of Chapter 3's
`-affine-loop-fusion` (which fused *loops*; this fuses *ops* before any loop
exists). Note fusion happens **on tensors, before bufferization** — that's the whole
point: the intermediate `t` is fused away while it's still an SSA value, so no buffer
is ever allocated for it.

### Running the fused kernel

Fusion is an IR-to-IR transform, but we can prove it computes the right thing by
compiling it. The fused func *returns* a tensor, so it needs bufferizing — and the
returned tensor becomes an in-place `out` buffer via `-buffer-results-to-out-params`,
giving the same `addmul(a, b, c, out)` C interface as the other steps:

```bash
$ mlir-opt build/fused_ops.mlir \
    -one-shot-bufferize="bufferize-function-boundaries" \
    -buffer-results-to-out-params \
    -convert-linalg-to-loops \
    ...                                  # -> scf -> cf -> llvm, then compile
```

The bufferized loop reads `a[i]`, `b[i]`, `c[i]` once each and writes the result —
no temp array, exactly the fused nest from the diagram. The driver checks it against
NumPy's `(a + b) * c`.

**Run:** `cd 5_fusion && bash build.sh` (also `diff separate_ops.mlir build/fused_ops.mlir`)

```
fused (a + b) * c successful!
  a   = [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]
  b   = [2. 2. 2. 2. 2. 2. 2. 2. 2. 2.]
  c   = [ 1.  2.  3.  4.  5.  6.  7.  8.  9. 10.]
  out = [  2.   6.  12.  20.  30.  42.  56.  72.  90. 110.]  == (a + b) * c
```

Reading the first column: `(a[0] + b[0]) * c[0] = (0 + 2) * 1 = 2`; the last:
`(9 + 2) * 10 = 110`. The fused single-loop kernel matches NumPy's `(a + b) * c`
element for element.

---

## Step 6 — Tiling (`6_tiling/`) · 🧩

**Goal:** restructure a matmul for cache locality.

On modern hardware, memory access — not arithmetic — is usually the bottleneck.
Fetching data from main memory is far slower than reading it from cache, so when a
computation sweeps a large matrix linearly it keeps evicting and re-fetching data,
stalling on memory the whole time.

**Tiling** fixes this by breaking a big computation into smaller **blocks** that fit
in cache. In matrix multiply, instead of computing a full row-times-column dot
product at once (which may blow past the cache), you work one small tile at a time —
so each block of data stays hot in cache and is fully reused before the next is
loaded. Fewer cache misses, better performance.

This matters even more on **GPUs**, whose memory hierarchy is explicit — global
memory, shared memory (a software-managed cache), and registers. Tiling a
computation to match the hardware — shared-memory size, warp size — is *the* key GPU
optimization: the classic CUDA pattern loads a tile into shared memory, synchronizes
the threads, then has each thread work within that tile. Chapter 8 uses exactly this
mechanism to target GPU shared memory; here we introduce it on the CPU.

We start from a one-line `linalg.matmul` on **tensors**:

*6_tiling/matmul_tile.mlir*
```mlir
module {
  func.func @matmul(%a: tensor<10x10xf32>, %b: tensor<10x10xf32>,
                    %c: tensor<10x10xf32>) -> tensor<10x10xf32> {
    %0 = linalg.matmul ins(%a, %b : tensor<10x10xf32>, tensor<10x10xf32>)
                       outs(%c : tensor<10x10xf32>) -> tensor<10x10xf32>
    return %0 : tensor<10x10xf32>
  }
}
```

Because these are tensors that get *returned*, the pipeline first bufferizes them
(tensors → memrefs, Chapter 2), then lowers to affine loops, then tiles:

```bash
$ mlir-opt 6_tiling/matmul_tile.mlir \
    --convert-tensor-to-linalg \
    --linalg-generalize-named-ops \
    --one-shot-bufferize="bufferize-function-boundaries" \
    --buffer-deallocation-pipeline \
    --convert-bufferization-to-memref \
    --convert-linalg-to-affine-loops \
    --affine-loop-tile="tile-size=5" \
    --canonicalize --cse
```

What each pass does:

| Pass | Role |
| --- | --- |
| `--convert-tensor-to-linalg` | Rewrites remaining `tensor` ops (e.g. `tensor.empty`/`pad`) into `linalg` form, so everything is one dialect the later passes understand. |
| `--linalg-generalize-named-ops` | Rewrites `linalg.matmul` into its `linalg.generic` form — the structure the loop-lowering and tiling passes operate on. |
| `--one-shot-bufferize="bufferize-function-boundaries"` | The core tensors → memrefs conversion; `bufferize-function-boundaries` bufferizes the function's arguments and result too (needed since this func *returns* a tensor). |
| `--buffer-deallocation-pipeline` | Inserts `memref.dealloc` for the temporary buffers bufferization allocated, so nothing leaks. |
| `--convert-bufferization-to-memref` | Lowers leftover `bufferization`-dialect ops (clones, casts) into concrete `memref` ops. |
| `--convert-linalg-to-affine-loops` | Lowers the generic op to an `affine.for` nest — the **affine** form (not `scf`), because tiling needs analyzable loops (Step 3). |
| `--affine-loop-tile="tile-size=5"` | The actual tiling: splits each loop into a *tile* loop (step 5) + a *point* loop, turning the 3-deep nest into 6-deep. |
| `--canonicalize --cse` | Cleanup — fold/simplify and drop redundant ops the earlier passes introduced. |

The order is the point: generalize → bufferize (+ dealloc + finalize) → **affine** loops
→ tile. Tiling is the last real step, and it only runs because the nest is `affine` —
`-affine-loop-tile` can't touch an `scf.for` nest.

The 3-loop matmul becomes a **6-deep nest**: three outer *tile* loops stepping by
5 over blocks, three inner *point* loops (bounds `#map = (d0)->(d0)` and
`#map1 = (d0)->(d0+5)`) walking the 5 elements inside each tile — the same
tile/point split you saw in Chapter 3's `-affine-loop-tile`, now applied to a
matmul the compiler generated from one `linalg.matmul`. `build.sh` writes it to
`build/matmul_tiled.mlir`:

```text
   before: sweep the whole            after: 5×5×5 blocks, one at a time
   10×10×10 space
   for i in 0..10                     for i0 in 0..10 step 5      ┐
     for j in 0..10                     for j0 in 0..10 step 5    │ tile loops
       for k in 0..10                     for k0 in 0..10 step 5  ┘ (step 5)
         C[i,j] += A[i,k]*B[k,j]            for i in i0..i0+5     ┐
                                              for j in j0..j0+5   │ point loops
                                                for k in k0..k0+5 ┘ (walk the tile)
                                                  C[i,j] += A[i,k]*B[k,j]
```

*build/matmul_tiled.mlir*
```mlir
#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0 + 5)>
module {
  func.func @matmul(%arg0: memref<10x10xf32, strided<[?, ?], offset: ?>>, %arg1: memref<10x10xf32, strided<[?, ?], offset: ?>>, %arg2: memref<10x10xf32, strided<[?, ?], offset: ?>>) -> memref<10x10xf32, strided<[?, ?], offset: ?>> {
    affine.for %arg3 = 0 to 10 step 5 {
      affine.for %arg4 = 0 to 10 step 5 {
        affine.for %arg5 = 0 to 10 step 5 {
          affine.for %arg6 = #map(%arg3) to #map1(%arg3) {
            affine.for %arg7 = #map(%arg4) to #map1(%arg4) {
              affine.for %arg8 = #map(%arg5) to #map1(%arg5) {
                %0 = affine.load %arg0[%arg6, %arg8] : memref<10x10xf32, strided<[?, ?], offset: ?>>
                %1 = affine.load %arg1[%arg8, %arg7] : memref<10x10xf32, strided<[?, ?], offset: ?>>
                %2 = affine.load %arg2[%arg6, %arg7] : memref<10x10xf32, strided<[?, ?], offset: ?>>
                %3 = arith.mulf %0, %1 : f32
                %4 = arith.addf %2, %3 : f32
                affine.store %4, %arg2[%arg6, %arg7] : memref<10x10xf32, strided<[?, ?], offset: ?>>
              }
            }
          }
        }
      }
    }
    %alloc = memref.alloc() : memref<10x10xf32>
    %cast = memref.cast %alloc : memref<10x10xf32> to memref<10x10xf32, strided<[?, ?], offset: ?>>
    memref.copy %arg2, %alloc : memref<10x10xf32, strided<[?, ?], offset: ?>> to memref<10x10xf32>
    return %cast : memref<10x10xf32, strided<[?, ?], offset: ?>>
  }
}
```

(The `strided<[?, ?], offset: ?>` types and the trailing `alloc`/`copy` are
bufferization's doing — it can't assume the returned buffer's layout, so it works
through a generic strided view and copies out a fresh result. The `?`s are the
runtime strides from Chapter 2's descriptor.)

Because Linalg keeps the op abstract until lowering, *the compiler* picks the tile
structure from a single `linalg.matmul` — the same abstraction that lets Chapter 8
retarget it to GPU shared memory.

**Run:** `cd 6_tiling && bash build.sh` (then read `build/matmul_tiled.mlir`).

---

## Run everything

```bash
export PATH="/opt/homebrew/opt/llvm@20/bin:$PATH"   # llvm@20 is keg-only
# Python deps for the drivers:  pip install numpy

for d in 1_elementwise 2_reduce 3_matmul 4_broadcast 5_fusion 6_tiling; do
  echo "=== $d ==="; ( cd "$d" && bash build.sh ) 2>&1 | tail -n 3
done
```

Each step writes its intermediates (`*_opt.mlir`, `*.ll`, `*.o`, `*.dylib`, and
the inspected `build/*.mlir`) into a local `build/` directory.

## Key takeaways

- **Describe the operation, not the loops.** A `linalg` op is *payload* (the
  scalar math) + *structure* (`indexing_maps` + `iterator_types`); the compiler
  generates and optimizes the loops from that structure.
- **Named ops are sugar for `linalg.generic`.** `linalg.matmul`, `linalg.add`,
  `linalg.broadcast` all lower through the same generic machinery — and to loops
  via `-convert-linalg-to-loops` / `-convert-linalg-to-affine-loops`, reproducing
  Chapter 3's affine nests exactly.
- **The three iterators** — `parallel`, `reduction`, `window` — are how an op
  declares which loops are independent, which accumulate, and which slide.
- **Tensors need bufferizing; memrefs don't.** In-place named ops on memrefs run
  straight away (Steps 1–4); ops that *return* tensors go through
  `--one-shot-bufferize` first (the fused kernel in Step 5, the tiling in Step 6).
- **Fusion and tiling are why this level exists.** Keeping ops abstract lets the
  compiler fuse elementwise chains into one loop and tile contractions for cache —
  exactly the transforms that matter for ML and GPU code.

**Next:** Part 5 — Neural networks and tensors (see [`../reference/`](../reference/)).
```
