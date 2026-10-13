# Chapter 0: A Primer on “Structured” Linalg Operations

> **Goal:** Understand *structured operations* — elementwise vector ops, reductions, contractions, and the Linalg dialect's `linalg.generic` on buffers and tensors — and the transformations their structure makes easy: lowering to smaller pieces, tiling, and producer/consumer fusion. Every listing is real IR run through stock `mlir-opt` — based on the official tutorial [Transform Ch0: A Primer on “Structured” Linalg Operations](https://mlir.llvm.org/docs/Tutorials/transform/Ch0/).

---

## 1. Overview

Before starting the tutorial on the Transform dialect, let us take a brief look at the concept of *structured operations* and its implementation in the Linalg dialect. The Transform dialect does not require structured operations, and vice versa. The two co-evolved at the beginning of the Transform dialect, which makes the subset of transformations for structured operations the most mature and the most suitable for the tutorial. If you are already familiar with this concept, skip to Chapter 1.

**Structured code generation** intends to preserve the structure of the computation for as long as necessary to enable transformations, up to and including the design of IR abstractions that support specific transformations. This chapter climbs an abstraction ladder, one rung per section:

| Section | Abstraction | What its structure tells the compiler |
|---|---|---|
| 2 | `arith.addf` on n-D vectors | the same operation applies to every element |
| 3 | `vector.reduction` | the elements of one vector are combined into a scalar |
| 4 | `vector.contract` | elements of two vectors are multiplied, then reduced, along given dimensions |
| 5–6 | `linalg.generic` on memory (memrefs) | a contraction, or any per-element computation, over buffers, with an explicit body |
| 7 | `linalg.generic` on tensors | the same, on immutable values: no aliasing to worry about |
| 8–9 | tiling and fusion | what the structure buys: loops and fusion without dependence analysis |
| 10 | named ops (`linalg.matmul`) | a shorthand for a common `linalg.generic` |

There is no C++ in this chapter and nothing to build: every command runs the installed `mlir-opt` (Homebrew LLVM/MLIR 20) on IR given on stdin through a quoted heredoc (`<<'EOF'`). The snippets are completed into valid IR — wrapped in a `func.func` that takes the operands they use as arguments — so each one parses, verifies and, where it makes a point, is transformed by a real pass. Sections 8 and 9 drive the tiling and fusion with a short Transform dialect script; Chapter 1 explains such scripts properly, and appendix A explains the handful of transform operations used here.

The appendices hold reference material: the transform operations used in sections 4, 8 and 9 (A), and the script that runs every command in this README (B).

---

## 2. Uniform Elementwise Extension

### 2.1 From scalars to n-D vectors

Consider a simple scalar arithmetic addition operation in MLIR, which maps directly to a machine instruction on most architectures that support floating point operations:

```mlir
%2 = arith.addf %0, %1 : f32
```

This operation can be easily extended to uniformly apply to elements of a 1-D vector, which is also often available as an instruction of vector machines:

```mlir
%2 = arith.addf %0, %1 : vector<8xf32>
```

Only a few modern instruction sets offer instructions for two- or more-dimensional vectors. In MLIR, however, it is possible to transparently extend the uniform elementwise application to vectors of arbitrary rank:

```mlir
%2 = arith.addf %0, %1 : vector<8x4xf32>
%5 = arith.addf %3, %4 : vector<2x2x2x2x2x2x2xf32>
```

The four forms are one operation, `arith.addf`, with four operand types. Put into one function, they parse and verify as they are:

```bash
cd /Users/roy/study/mlir/transform/Ch0
mlir-opt <<'EOF'
func.func @add(%0: f32, %1: f32, %v0: vector<8xf32>, %v1: vector<8xf32>,
               %m0: vector<8x4xf32>, %m1: vector<8x4xf32>,
               %3: vector<2x2x2x2x2x2x2xf32>, %4: vector<2x2x2x2x2x2x2xf32>)
    -> (f32, vector<8xf32>, vector<8x4xf32>, vector<2x2x2x2x2x2x2xf32>) {
  %2 = arith.addf %0, %1 : f32
  %v2 = arith.addf %v0, %v1 : vector<8xf32>
  %m2 = arith.addf %m0, %m1 : vector<8x4xf32>
  %5 = arith.addf %3, %4 : vector<2x2x2x2x2x2x2xf32>
  return %2, %v2, %m2, %5 : f32, vector<8xf32>, vector<8x4xf32>, vector<2x2x2x2x2x2x2xf32>
}
EOF
```

Real output **[1]**:

```text
module {
  func.func @add(%arg0: f32, %arg1: f32, %arg2: vector<8xf32>, %arg3: vector<8xf32>, %arg4: vector<8x4xf32>, %arg5: vector<8x4xf32>, %arg6: vector<2x2x2x2x2x2x2xf32>, %arg7: vector<2x2x2x2x2x2x2xf32>) -> (f32, vector<8xf32>, vector<8x4xf32>, vector<2x2x2x2x2x2x2xf32>) {
    %0 = arith.addf %arg0, %arg1 : f32
    %1 = arith.addf %arg2, %arg3 : vector<8xf32>
    %2 = arith.addf %arg4, %arg5 : vector<8x4xf32>
    %3 = arith.addf %arg6, %arg7 : vector<2x2x2x2x2x2x2xf32>
    return %0, %1, %2, %3 : f32, vector<8xf32>, vector<8x4xf32>, vector<2x2x2x2x2x2x2xf32>
  }
}
```

- The printer renames the values (`%arg0`, `%0`, ...): names in the textual IR are only for reading, so the SSA names of the snippets above are free to choose. A name can't be defined twice, though, which is why each function argument here has its own.
- The seven-dimensional vector is as valid an operand as `f32`: rank is part of the type, not a limit of the operation.

### 2.2 What the structure is good for

MLIR's arithmetic operations on vectors preserve the structure of uniform elementwise application. This structure can be leveraged by the compiler, for example, to produce smaller-rank operations available on the target, or to fuse multiplication and addition when such a fused instruction is available (which becomes complicated when there are a hundred multiplications followed by a hundred additions).

#### Smaller-rank operations

LLVM IR has only 1-D vectors. Converting a 2-D `arith.addf` to the LLVM dialect therefore splits it into one 1-D addition per row: the `vector<8x4xf32>` becomes an array of eight `vector<4xf32>` (`!llvm.array<8 x vector<4xf32>>`), and each row is extracted, added and inserted back:

```bash
cd /Users/roy/study/mlir/transform/Ch0
mlir-opt --convert-arith-to-llvm <<'EOF'
func.func @add2d(%0: vector<8x4xf32>, %1: vector<8x4xf32>) -> vector<8x4xf32> {
  %2 = arith.addf %0, %1 : vector<8x4xf32>
  return %2 : vector<8x4xf32>
}
EOF
```

Real output **[2]** (abridged: rows 2 to 6 elided):

```text
module {
  func.func @add2d(%arg0: vector<8x4xf32>, %arg1: vector<8x4xf32>) -> vector<8x4xf32> {
    %0 = builtin.unrealized_conversion_cast %arg1 : vector<8x4xf32> to !llvm.array<8 x vector<4xf32>>
    %1 = builtin.unrealized_conversion_cast %arg0 : vector<8x4xf32> to !llvm.array<8 x vector<4xf32>>
    %2 = llvm.mlir.undef : !llvm.array<8 x vector<4xf32>>
    %3 = llvm.extractvalue %1[0] : !llvm.array<8 x vector<4xf32>> 
    %4 = llvm.extractvalue %0[0] : !llvm.array<8 x vector<4xf32>> 
    %5 = llvm.fadd %3, %4 : vector<4xf32>
    %6 = llvm.insertvalue %5, %2[0] : !llvm.array<8 x vector<4xf32>> 
    %7 = llvm.extractvalue %1[1] : !llvm.array<8 x vector<4xf32>> 
    %8 = llvm.extractvalue %0[1] : !llvm.array<8 x vector<4xf32>> 
    %9 = llvm.fadd %7, %8 : vector<4xf32>
    %10 = llvm.insertvalue %9, %6[1] : !llvm.array<8 x vector<4xf32>> 
    ...
    %31 = llvm.extractvalue %1[7] : !llvm.array<8 x vector<4xf32>> 
    %32 = llvm.extractvalue %0[7] : !llvm.array<8 x vector<4xf32>> 
    %33 = llvm.fadd %31, %32 : vector<4xf32>
    %34 = llvm.insertvalue %33, %30[7] : !llvm.array<8 x vector<4xf32>> 
    %35 = builtin.unrealized_conversion_cast %34 : !llvm.array<8 x vector<4xf32>> to vector<8x4xf32>
    return %35 : vector<8x4xf32>
  }
}
```

- The pass knows the operation applies the same addition to every element, so it can pick any decomposition that matches the target: here one `llvm.fadd` on `vector<4xf32>` per row (indices `[0]` to `[7]`).
- The `builtin.unrealized_conversion_cast` ops bridge between the converted and not-yet-converted types (`func.func` still takes `vector<8x4xf32>`); converting the function too, which a full pipeline does, removes them.

#### Fusing multiplication and addition

A multiplication followed by an addition of the same shape can become one fused multiply-add when floating-point contraction is allowed (`fastmath<contract>`). Because both ops are whole-vector ops, recognizing the pair is a single local pattern, whatever the rank:

```bash
cd /Users/roy/study/mlir/transform/Ch0
mlir-opt --math-uplift-to-fma <<'EOF'
func.func @mul_add(%a: vector<8x4xf32>, %b: vector<8x4xf32>,
                   %c: vector<8x4xf32>) -> vector<8x4xf32> {
  %0 = arith.mulf %a, %b fastmath<contract> : vector<8x4xf32>
  %1 = arith.addf %0, %c fastmath<contract> : vector<8x4xf32>
  return %1 : vector<8x4xf32>
}
EOF
```

Real output **[3]**:

```text
module {
  func.func @mul_add(%arg0: vector<8x4xf32>, %arg1: vector<8x4xf32>, %arg2: vector<8x4xf32>) -> vector<8x4xf32> {
    %0 = math.fma %arg0, %arg1, %arg2 fastmath<contract> : vector<8x4xf32>
    return %0 : vector<8x4xf32>
  }
}
```

Written as 32 scalar multiplications followed by 32 scalar additions, the same rewrite would first have to pair each addition with its multiplication. **Keeping the computation as one whole-vector op turns a matching problem into a one-op pattern.**

---

## 3. Reduction

### 3.1 `vector.reduction`

Sometimes it is necessary to add the elements of a vector to obtain a scalar. Some platforms provide specific instructions for this operation; others provide ones that can be combined to achieve the desired effect, such as addition of adjacent elements and element shuffle.

The Vector dialect in MLIR defines an operation to explicitly denote a within-vector reduction:

```mlir
%1 = vector.reduction <add>, %0 : vector<8xf32> into f32
```

`<add>` is the combining kind (others: `mul`, `minimumf`, `maximumf`, `and`, `or`, `xor`, ...), `%0` the vector, and `f32` the scalar result. Lowered to the LLVM dialect, the op maps one-to-one to LLVM's reduction intrinsic, which each LLVM backend expands to the best instruction sequence its target has:

```bash
cd /Users/roy/study/mlir/transform/Ch0
mlir-opt --convert-vector-to-llvm <<'EOF'
func.func @reduce(%0: vector<8xf32>) -> f32 {
  %1 = vector.reduction <add>, %0 : vector<8xf32> into f32
  return %1 : f32
}
EOF
```

Real output **[4]**:

```text
module {
  func.func @reduce(%arg0: vector<8xf32>) -> f32 {
    %0 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %1 = "llvm.intr.vector.reduce.fadd"(%0, %arg0) <{fastmathFlags = #llvm.fastmath<none>}> : (f32, vector<8xf32>) -> f32
    return %1 : f32
  }
}
```

The `0.0` start value (`llvm.mlir.constant(0.000000e+00 : f32)`) is the neutral element of addition: the intrinsic computes `0.0 + %0[0] + ... + %0[7]`.

### 3.2 The loop form

When no support is available, such an operation can be transformed into a loop:

```bash
cd /Users/roy/study/mlir/transform/Ch0
mlir-opt <<'EOF'
func.func @reduce_loop(%0: vector<8xf32>) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %init = arith.constant 0.0 : f32
  %result = scf.for %i = %c0 to %c8 step %c1 iter_args(%partial = %init) -> (f32) {
    %element = vector.extractelement %0[%i : index] : vector<8xf32>
    %updated = arith.addf %partial, %element : f32
    scf.yield %updated : f32
  }
  return %result : f32
}
EOF
```

Real output **[5]**:

```text
module {
  func.func @reduce_loop(%arg0: vector<8xf32>) -> f32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = scf.for %arg1 = %c0 to %c8 step %c1 iter_args(%arg2 = %cst) -> (f32) {
      %1 = vector.extractelement %arg0[%arg1 : index] : vector<8xf32>
      %2 = arith.addf %arg2, %1 : f32
      scf.yield %2 : f32
    }
    return %0 : f32
  }
}
```

- `scf.for ... iter_args(%partial = %init)` carries a loop-carried value: `%partial` starts as `%init` and takes, in each next iteration, the value of `scf.yield`; the loop's result `%result` is the last yielded value.
- `vector.extractelement` reads one element at a dynamic position `%i`.

Even when special instructions are available, it may still be desirable to use the loop form (with unrolling), depending on instruction latency and register pressure. Preserving the structure of the operation as a single reduction gives the compiler an understanding that a within-vector reduction is performed and, therefore, a choice in implementation. **From the loop, the compiler would have to rediscover that it is a reduction before it could pick an instruction for it; from `vector.reduction`, both forms are one rewrite away.**

---

## 4. Contraction

### 4.1 A reduction as a contraction

Contraction is a generalization of reduction that multiplies elements from two vectors before adding them up. A simple “add” reduction can be thought of as a contraction where one of the vectors contains `1.0`, the neutral element of multiplication. Contractions offer even more flexibility to the compiler, and are represented by a dedicated operation in MLIR, `vector.contract`:

```bash
cd /Users/roy/study/mlir/transform/Ch0
mlir-opt --convert-vector-to-llvm <<'EOF'
func.func @dot_with_ones(%0: vector<8xf32>) -> f32 {
  // Neutral initializer for the addition.
  %init  = arith.constant 0.0 : f32
  // Neutral element of multiplication.
  %ones = arith.constant dense<1.0> : vector<8xf32>
  // Actual contraction.
  %result = vector.contract {
    indexing_maps = [affine_map<(i) -> (i)>,
                     affine_map<(i) -> (i)>,
                     affine_map<(i) -> ()>],
    iterator_types = ["reduction"]
  } %0, %ones, %init : vector<8xf32>, vector<8xf32> into f32
  return %result : f32
}
EOF
```

Real output **[6]**:

```text
module {
  func.func @dot_with_ones(%arg0: vector<8xf32>) -> f32 {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = "llvm.intr.vector.reduce.fadd"(%cst, %arg0) <{fastmathFlags = #llvm.fastmath<none>}> : (f32, vector<8xf32>) -> f32
    return %0 : f32
  }
}
```

The lowering produced exactly the reduction intrinsic of section 3.1: it saw that one operand is all ones, dropped the multiplication, and was left with an add reduction. The contraction and the reduction are the same computation, and the compiler can tell because both are stated as what they compute rather than as loops.

Note the `affine_map` expressions indicating how vector elements are indexed. Their meaning is perhaps most evident when writing the loop form in pseudo-code equivalent to this contraction (pseudo-code, not MLIR):

```text
for i in 0 to 8:
  init += p0[i] * ones[i]
```

Both `%0` (here `p0`) and `%ones` use the loop induction variable `i`, as noted on the right-hand side of the corresponding affine map, `(i) -> (i)`, and `%init` does not, as reflected on the right-hand side of its affine map, `(i) -> ()`. The left-hand side `(i)` lists the dimensions of the implicit iteration space, one per loop; each map says which of them index its operand. A dimension missing from the result's map is one that gets reduced, which is what `iterator_types = ["reduction"]` states explicitly. The op also has a combining `kind`, `#vector.kind<add>` by default, so `mlir-opt` prints `kind = #vector.kind<add>` when it echoes a contraction back.

### 4.2 Matrix multiplication as a contraction

Similarly to uniform elementwise extension, MLIR vector contractions are not limited to 1-D cases. In the 2-D+ case, one can additionally specify which of the vector dimensions are being reduced and which ones are being preserved. This is the job of the `iterator_types` attribute, which specifies, for each dimension, whether it is being reduced (`"reduction"`) or preserved (`"parallel"`). Consider the following 3-D contraction that encodes a matrix-matrix multiplication:

```mlir
%result = vector.contract {
  indexing_maps = [affine_map<(i, j, k) -> (i, k)>,
                   affine_map<(i, j, k) -> (k, j)>,
                   affine_map<(i, j, k) -> (i, j)>],
  iterator_types = ["parallel", "parallel", "reduction"]
} %lhs, %rhs, %init: vector<8x10xf32>, vector<10x16xf32> into vector<8x16xf32>
```

Looking at the indexing maps, it is easy to recognize the loop form (pseudo-code):

```text
for i in 0 to 8:
  for j in 0 to 16:
    for k in 0 to 10:
      init[i, j] += lhs[i, k] * rhs[k, j]
```

The loop bounds come from the operand shapes: `i` indexes dimension 0 of `%lhs` (8), `j` dimension 1 of `%rhs` (16), and `k` both dimension 1 of `%lhs` and dimension 0 of `%rhs` (10, so the two must agree — the verifier checks it). `k` is absent from the result map `(i, j)`, so it is the reduced dimension.

Preserving this higher-level structure of a contraction makes it significantly easier for the compiler to recognize operations such as matrix multiplications and dot products and gives it freedom to produce lower-level operations that leverage the most advanced instructions or even pre-generated microkernels. The same `vector.contract` can be lowered in several ways. The default conversion to the LLVM dialect computes each of the 8 × 16 result elements as a dot product — a length-10 reduction:

```bash
cd /Users/roy/study/mlir/transform/Ch0
mlir-opt --convert-vector-to-llvm <<'EOF' | grep -c 'llvm.intr.vector.reduce.fadd'
func.func @matmul(%lhs: vector<8x10xf32>, %rhs: vector<10x16xf32>,
                  %init: vector<8x16xf32>) -> vector<8x16xf32> {
  %result = vector.contract {
    indexing_maps = [affine_map<(i, j, k) -> (i, k)>,
                     affine_map<(i, j, k) -> (k, j)>,
                     affine_map<(i, j, k) -> (i, j)>],
    iterator_types = ["parallel", "parallel", "reduction"]
  } %lhs, %rhs, %init: vector<8x10xf32>, vector<10x16xf32> into vector<8x16xf32>
  return %result : vector<8x16xf32>
}
EOF
```

Real output **[7]**:

```text
128
```

128 = 8 × 16 reduction intrinsics, one per result element. Asking for the *outer-product* strategy instead turns the same op into 10 rank-1 updates, one per value of `k`: column `k` of `%lhs` times row `k` of `%rhs`, added to the whole 8×16 accumulator — a shape that suits targets with outer-product or broadcast-FMA instructions. The strategy is chosen here with a Transform dialect script that applies the contraction-lowering patterns to the function (appendix A.1); the `sed` drops the script itself, which the interpreter leaves in its output unchanged, along with the module's closing brace:

```bash
cd /Users/roy/study/mlir/transform/Ch0
mlir-opt --transform-interpreter <<'EOF' | sed '/module attributes/,$d'
func.func @matmul(%lhs: vector<8x10xf32>, %rhs: vector<10x16xf32>,
                  %init: vector<8x16xf32>) -> vector<8x16xf32> {
  %result = vector.contract {
    indexing_maps = [affine_map<(i, j, k) -> (i, k)>,
                     affine_map<(i, j, k) -> (k, j)>,
                     affine_map<(i, j, k) -> (i, j)>],
    iterator_types = ["parallel", "parallel", "reduction"]
  } %lhs, %rhs, %init: vector<8x10xf32>, vector<10x16xf32> into vector<8x16xf32>
  return %result : vector<8x16xf32>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root
      : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.vector.lower_contraction
        lowering_strategy = "outerproduct"
    } : !transform.any_op
    transform.yield
  }
}
EOF
```

Real output **[8]** (abridged: updates for `k` = 2 to 8 elided):

```text
module {
  func.func @matmul(%arg0: vector<8x10xf32>, %arg1: vector<10x16xf32>, %arg2: vector<8x16xf32>) -> vector<8x16xf32> {
    %0 = vector.transpose %arg0, [1, 0] : vector<8x10xf32> to vector<10x8xf32>
    %1 = vector.extract %0[0] : vector<8xf32> from vector<10x8xf32>
    %2 = vector.extract %arg1[0] : vector<16xf32> from vector<10x16xf32>
    %3 = vector.outerproduct %1, %2, %arg2 {kind = #vector.kind<add>} : vector<8xf32>, vector<16xf32>
    %4 = vector.extract %0[1] : vector<8xf32> from vector<10x8xf32>
    %5 = vector.extract %arg1[1] : vector<16xf32> from vector<10x16xf32>
    %6 = vector.outerproduct %4, %5, %3 {kind = #vector.kind<add>} : vector<8xf32>, vector<16xf32>
    ...
    %28 = vector.extract %0[9] : vector<8xf32> from vector<10x8xf32>
    %29 = vector.extract %arg1[9] : vector<16xf32> from vector<10x16xf32>
    %30 = vector.outerproduct %28, %29, %27 {kind = #vector.kind<add>} : vector<8xf32>, vector<16xf32>
    return %30 : vector<8x16xf32>
  }
```

- `vector.transpose` makes the columns of `%lhs` into rows, so `vector.extract %0[k]` reads column `k` as a `vector<8xf32>`; `vector.extract %arg1[k]` reads row `k` of `%rhs` as a `vector<16xf32>`.
- Each `vector.outerproduct %a, %b, %acc` computes `acc[i, j] + a[i] * b[j]` for the whole 8×16 tile; the chain threads the accumulator from `%arg2` through ten updates.

**One `vector.contract`, two very different instruction mixes (128 horizontal reductions or 10 outer products): the choice stays open only as long as the IR still says "contraction".**

---

## 5. Generic Operation on Memory

Until now, we have been considering operations on vectors stored in virtual registers. A similar contraction abstraction can be defined in memory, with the Linalg dialect's `linalg.generic` on memrefs (MLIR's buffer type):

```mlir
linalg.generic {
  indexing_maps = [affine_map<(i, j, k) -> (i, k)>,
                   affine_map<(i, j, k) -> (k, j)>,
                   affine_map<(i, j, k) -> (i, j)>],
  iterator_types = ["parallel", "parallel", "reduction"]
} ins(%lhs, %rhs : memref<8x10xf32>, memref<10x16xf32>)
  outs(%init : memref<8x16xf32>) {
^bb0(%lhs_one: f32, %rhs_one: f32, %init_one: f32):
  %0 = arith.mulf %lhs_one, %rhs_one : f32
  %1 = arith.addf %init_one, %0 : f32
  linalg.yield %1 : f32
}
```

This looks more complicated, so let us unpack. The `indexing_maps` and `iterator_types` are *exactly* the same as we have seen above for vector contractions. The operands are now split into two lists:

- `ins` operands containing the buffers that are only being read by the operation;
- `outs` operands that are being read and updated by the operation.

This separation wasn't necessary on vectors because, in MLIR, vectors are read-only (SSA or functional form), and operations mutating a vector in fact produce a new one instead.

Furthermore, the operation now contains a **region** that explicitly specifies the multiplication and the addition operations that were implicit in the contraction. The block arguments of the region correspond to individual elements read from the buffers: the first two correspond to the `ins` operands and the last one corresponds to the `outs` operand. The value yielded from the region (`linalg.yield`) is “written” to the `outs` operand and is available as the last block argument for future executions of the region. Note that the order in which the region is executed for various tuples of elements read from the buffers is not specified, and the write to the `outs` buffer is written as a whole at the end of the operation.

Since the operation says everything about its loops — their number (the dimensions of the maps), their bounds (the operand shapes), and how each operand is indexed — a pass can write the loops out. `--convert-linalg-to-loops` does exactly that:

```bash
cd /Users/roy/study/mlir/transform/Ch0
mlir-opt --convert-linalg-to-loops <<'EOF'
func.func @matmul_memref(%lhs: memref<8x10xf32>, %rhs: memref<10x16xf32>,
                         %init: memref<8x16xf32>) {
  linalg.generic {
    indexing_maps = [affine_map<(i, j, k) -> (i, k)>,
                     affine_map<(i, j, k) -> (k, j)>,
                     affine_map<(i, j, k) -> (i, j)>],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%lhs, %rhs : memref<8x10xf32>, memref<10x16xf32>)
    outs(%init : memref<8x16xf32>) {
  ^bb0(%lhs_one: f32, %rhs_one: f32, %init_one: f32):
    %0 = arith.mulf %lhs_one, %rhs_one : f32
    %1 = arith.addf %init_one, %0 : f32
    linalg.yield %1 : f32
  }
  return
}
EOF
```

Real output **[9]**:

```text
module {
  func.func @matmul_memref(%arg0: memref<8x10xf32>, %arg1: memref<10x16xf32>, %arg2: memref<8x16xf32>) {
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

The result is the pseudo-code loop nest of section 4.2, made real:

- three `scf.for` loops, for `i` (`%arg3`, 0 to 8), `j` (`%arg4`, 0 to 16) and `k` (`%arg5`, 0 to 10), in the order of the map dimensions;
- three `memref.load`s whose indices are the indexing maps applied to the loop variables: `%arg0[%arg3, %arg5]` is `(i, k)`, `%arg1[%arg5, %arg4]` is `(k, j)`, `%arg2[%arg3, %arg4]` is `(i, j)`;
- the region's body, with its block arguments replaced by the loaded values, and a `memref.store` of the yielded value into the `outs` buffer.

---

## 6. “Loop” Fusion

Since the region of the `linalg.generic` operation can contain arbitrarily many operations, we can use it to express “fusion” of the implicit loops by simply having more operations chained in the region. For example, the common machine-learning rectified linear unit layer (ReLU), defined as `relu(x) = max(0, x)`, can be expressed using the “compare-and-select” idiom in one `linalg.generic` operation, without a temporary buffer for the comparison result and without repeating the outer operation:

```bash
cd /Users/roy/study/mlir/transform/Ch0
mlir-opt --convert-linalg-to-loops <<'EOF'
func.func @relu(%in: memref<?xf32>, %out: memref<?xf32>) {
  linalg.generic {
    indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
    iterator_types = ["parallel"]
  } ins(%in : memref<?xf32>) outs(%out : memref<?xf32>) {
  ^bb0(%in_one : f32, %out_one : f32):
    %c0 = arith.constant 0.0 : f32
    %0 = arith.cmpf ogt, %in_one, %c0 : f32
    %1 = arith.select %0, %in_one, %c0 : f32
    linalg.yield %1 : f32
  }
  return
}
EOF
```

Real output **[10]**:

```text
module {
  func.func @relu(%arg0: memref<?xf32>, %arg1: memref<?xf32>) {
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg0, %c0 : memref<?xf32>
    scf.for %arg2 = %c0 to %dim step %c1 {
      %0 = memref.load %arg0[%arg2] : memref<?xf32>
      %1 = arith.cmpf ogt, %0, %cst : f32
      %2 = arith.select %1, %0, %cst : f32
      memref.store %2, %arg1[%arg2] : memref<?xf32>
    }
    return
  }
}
```

- `indexing_maps = [...]` needs its `=`, and `arith.cmpf` takes its predicate (`ogt`, “ordered greater than”) followed by a comma; without either the parser stops.
- The buffers are dynamically sized (`memref<?xf32>`), so the loop's upper bound is read at run time with `memref.dim`.
- One loop holds both the comparison and the selection: the `i1` comparison result never leaves a register, and the constant `0.0` was hoisted out of the loop.
- `%out_one`, the current value of the output element, is unused: ReLU overwrites the output, so the loop never loads it.

Such operations can be converted to loops, as here, or lowered into vector forms after splitting into multiple operations, each of which maps to a Vector dialect primitive. This modeling, again, gives the compiler more choice in selecting the code generation strategy.

---

## 7. Generic Operation on Tensors

### 7.1 The same operation on tensors

Let us take one last step up on the abstraction ladder. MLIR provides a tensor abstraction that makes it easy for the compiler to reason about multidimensional yet regular data without having to solve complex problems such as alias analysis and dependency satisfaction, which would be necessary on multidimensional buffers. The tensor abstraction is very similar to the vector abstraction (major differences include the availability of unranked tensors, tensor layouts, and vectors being usable as elemental types of tensors but not of other vectors). Tensors are read-only, and operations updating a tensor produce a new tensor.

The `linalg.generic` operation from above can be lifted to operate on tensors instead of buffers:

```mlir
%result = linalg.generic {
  indexing_maps = [affine_map<(i, j, k) -> (i, k)>,
                   affine_map<(i, j, k) -> (k, j)>,
                   affine_map<(i, j, k) -> (i, j)>],
  iterator_types = ["parallel", "parallel", "reduction"]
} ins(%lhs, %rhs : tensor<8x10xf32>,tensor<10x16xf32>)
  outs(%init :tensor<8x16xf32>) {
^bb0(%lhs_one: f32, %rhs_one: f32, %init_one: f32):
  %0 = arith.mulf %lhs_one, %rhs_one : f32
  %1 = arith.addf %init_one, %0 : f32
  linalg.yield %1 : f32
} -> tensor<8x16xf32>
```

As you can notice, most components of this operation remain identical to its buffer version. It has been specifically designed this way. The main difference, besides the operand types, is that the operation now produces a new result instead of updating the `outs` buffer. The `outs` operand is used only as the initialization value. If the `linalg.generic` operation had existed on vectors, it would have had the exact same structure.

The two versions are related by **bufferization**, the step that assigns buffers to tensor values. Bufferizing the tensor version (`--one-shot-bufferize`, with function arguments and results bufferized too and plain identity layouts on them) gives back the buffer version of section 5:

```bash
cd /Users/roy/study/mlir/transform/Ch0
mlir-opt --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" <<'EOF'
func.func @matmul_tensor(%lhs: tensor<8x10xf32>, %rhs: tensor<10x16xf32>,
                         %init: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %result = linalg.generic {
    indexing_maps = [affine_map<(i, j, k) -> (i, k)>,
                     affine_map<(i, j, k) -> (k, j)>,
                     affine_map<(i, j, k) -> (i, j)>],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%lhs, %rhs : tensor<8x10xf32>,tensor<10x16xf32>)
    outs(%init :tensor<8x16xf32>) {
  ^bb0(%lhs_one: f32, %rhs_one: f32, %init_one: f32):
    %0 = arith.mulf %lhs_one, %rhs_one : f32
    %1 = arith.addf %init_one, %0 : f32
    linalg.yield %1 : f32
  } -> tensor<8x16xf32>
  return %result : tensor<8x16xf32>
}
EOF
```

Real output **[11]**:

```text
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @matmul_tensor(%arg0: memref<8x10xf32>, %arg1: memref<10x16xf32>, %arg2: memref<8x16xf32>) -> memref<8x16xf32> {
    linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : memref<8x10xf32>, memref<10x16xf32>) outs(%arg2 : memref<8x16xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %0 = arith.mulf %in, %in_0 : f32
      %1 = arith.addf %out, %0 : f32
      linalg.yield %1 : f32
    }
    return %arg2 : memref<8x16xf32>
  }
}
```

- The result `%result` is gone: the operation updates the buffer of `%init` (`%arg2`) in place, and the function returns that buffer.
- In-place reuse is allowed here because nothing reads `%init` after the operation; the analysis that decides this is exactly the aliasing reasoning the tensor form spared every earlier transformation.

### 7.2 Fusion on tensors

The tensor form is where Linalg fuses elementwise operations. Written without fusion, ReLU is two `linalg.generic`s with an `i1` tensor between them; `--linalg-fuse-elementwise-ops` merges them into the single compare-and-select operation of section 6, following the use-def chain from the second operation to the first:

```bash
cd /Users/roy/study/mlir/transform/Ch0
mlir-opt --linalg-fuse-elementwise-ops <<'EOF'
func.func @relu_unfused(%in: tensor<?xf32>, %cmp_init: tensor<?xi1>,
                        %out: tensor<?xf32>) -> tensor<?xf32> {
  %c0 = arith.constant 0.0 : f32
  %cmp = linalg.generic {
    indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
    iterator_types = ["parallel"]
  } ins(%in : tensor<?xf32>) outs(%cmp_init : tensor<?xi1>) {
  ^bb0(%in_one : f32, %out_one : i1):
    %0 = arith.cmpf ogt, %in_one, %c0 : f32
    linalg.yield %0 : i1
  } -> tensor<?xi1>
  %relu = linalg.generic {
    indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>,
                     affine_map<(i) -> (i)>],
    iterator_types = ["parallel"]
  } ins(%cmp, %in : tensor<?xi1>, tensor<?xf32>) outs(%out : tensor<?xf32>) {
  ^bb0(%cmp_one : i1, %in_one : f32, %out_one : f32):
    %1 = arith.select %cmp_one, %in_one, %c0 : f32
    linalg.yield %1 : f32
  } -> tensor<?xf32>
  return %relu : tensor<?xf32>
}
EOF
```

Real output **[12]**:

```text
#map = affine_map<(d0) -> (d0)>
module {
  func.func @relu_unfused(%arg0: tensor<?xf32>, %arg1: tensor<?xi1>, %arg2: tensor<?xf32>) -> tensor<?xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %arg2, %c0 : tensor<?xf32>
    %0 = tensor.empty(%dim) : tensor<?xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg0 : tensor<?xf32>) outs(%0 : tensor<?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %2 = arith.cmpf ogt, %in, %cst : f32
      %3 = arith.select %2, %in, %cst : f32
      linalg.yield %3 : f32
    } -> tensor<?xf32>
    return %1 : tensor<?xf32>
  }
}
```

- The `i1` tensor `%cmp` and its init `%cmp_init` are gone; the fused body computes the comparison and feeds it straight to `arith.select`.
- The output init became a `tensor.empty` of the right size: the body never reads `%out_one`, so only the shape of `%out` matters.

On tensors the fusion needs no check that the intermediate is not also written elsewhere: a tensor value is never modified. **Value semantics is what makes rewrites such as fusion local decisions; buffers come back only at bufferization, after the rewrites are done.**

---

## 8. Tiling and Loop Materialization

### 8.1 Tiling with `scf.forall`

At this level of abstraction, it becomes easy for the compiler to perform more advanced transformations usually required for high-performance code generation, such as [tiling](https://en.wikipedia.org/wiki/Loop_nest_optimization). Tiling, in general, can be seen as partitioning the iteration space into smaller parts, or tiles, so that the data required by each part fits into a level of cache, for example. The order in which tiles are executed must preserve the original data dependencies.

In the case of `linalg.generic` operations, the iteration space is implicit and is defined by the shape of the operands. Therefore, a tile can be expressed by performing the *same* operation on a subset (slice) of the original data. Since the order in which the body of `linalg.generic` is applied to different tuples of the input elements is unspecified, tiles can be executed in any order, without the need for dependence analysis. In order to control the execution of different tiles, the implementation of tiling produces loops. Thus tiling `linalg.generic` operations can also be seen as materializing the loops that have been implicit until now.

For example, tiling the matrix multiplication presented above with tile sizes `(2, 8)`, we obtain a loop nest around a `linalg.generic` expressing the same operation on a `2x8` tensor. The tiling is requested by a Transform dialect script: `transform.structured.match` finds the `linalg.generic`, and `transform.structured.tile_using_forall` tiles it (appendix A.2):

```bash
cd /Users/roy/study/mlir/transform/Ch0
mlir-opt --transform-interpreter <<'EOF' | sed '/module attributes/,$d'
func.func @matmul_tensor(%lhs: tensor<8x10xf32>, %rhs: tensor<10x16xf32>,
                         %init: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %result = linalg.generic {
    indexing_maps = [affine_map<(i, j, k) -> (i, k)>,
                     affine_map<(i, j, k) -> (k, j)>,
                     affine_map<(i, j, k) -> (i, j)>],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%lhs, %rhs : tensor<8x10xf32>,tensor<10x16xf32>)
    outs(%init :tensor<8x16xf32>) {
  ^bb0(%lhs_one: f32, %rhs_one: f32, %init_one: f32):
    %0 = arith.mulf %lhs_one, %rhs_one : f32
    %1 = arith.addf %init_one, %0 : f32
    linalg.yield %1 : f32
  } -> tensor<8x16xf32>
  return %result : tensor<8x16xf32>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op {transform.readonly}) {
    %generic = transform.structured.match ops{["linalg.generic"]} in %root
      : (!transform.any_op) -> !transform.any_op
    %tiled, %loop = transform.structured.tile_using_forall %generic
      tile_sizes [2, 8] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
EOF
```

Real output **[13]**:

```text
#map = affine_map<(d0) -> (d0 * 2)>
#map1 = affine_map<(d0) -> (d0 * 8)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @matmul_tensor(%arg0: tensor<8x10xf32>, %arg1: tensor<10x16xf32>, %arg2: tensor<8x16xf32>) -> tensor<8x16xf32> {
    %0 = scf.forall (%arg3, %arg4) in (4, 2) shared_outs(%arg5 = %arg2) -> (tensor<8x16xf32>) {
      %1 = affine.apply #map(%arg3)
      %2 = affine.apply #map1(%arg4)
      %extracted_slice = tensor.extract_slice %arg0[%1, 0] [2, 10] [1, 1] : tensor<8x10xf32> to tensor<2x10xf32>
      %extracted_slice_0 = tensor.extract_slice %arg1[0, %2] [10, 8] [1, 1] : tensor<10x16xf32> to tensor<10x8xf32>
      %extracted_slice_1 = tensor.extract_slice %arg5[%1, %2] [2, 8] [1, 1] : tensor<8x16xf32> to tensor<2x8xf32>
      %3 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction"]} ins(%extracted_slice, %extracted_slice_0 : tensor<2x10xf32>, tensor<10x8xf32>) outs(%extracted_slice_1 : tensor<2x8xf32>) {
      ^bb0(%in: f32, %in_2: f32, %out: f32):
        %4 = arith.mulf %in, %in_2 : f32
        %5 = arith.addf %out, %4 : f32
        linalg.yield %5 : f32
      } -> tensor<2x8xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %3 into %arg5[%1, %2] [2, 8] [1, 1] : tensor<2x8xf32> into tensor<8x16xf32>
      }
    }
    return %0 : tensor<8x16xf32>
  }
```

The same loop with readable value names, the affine maps written inline, and comments on each part (hand-edited from the output above, not this repo's code):

```mlir
// A special "multi-for" loop that supports tensor-insertion semantics
// as opposed to implicit updates. The resulting 8x16 tensor will be produced
// by this loop.
// The trip count of iterators is computed dividing the original tensor size,
// 8x16, by the tile size, 2x8, to obtain 4x2.
// When tensor sizes are dynamic, the trip count computation is emitted as IR
// and is being computed at runtime (section 8.2).
%0 = scf.forall (%i, %j) in (4, 2)
     shared_outs(%shared = %init) -> (tensor<8x16xf32>) {

  // Scale the loop induction variables by the tile sizes.
  %3 = affine.apply affine_map<(d0) -> (d0 * 2)>(%i)
  %4 = affine.apply affine_map<(d0) -> (d0 * 8)>(%j)

  // Take slices of inputs and outputs. Only the "i" and "j" dimensions are sliced.
  %lhs_slice = tensor.extract_slice %lhs[%3, 0] [2, 10] [1, 1]
             : tensor<8x10xf32> to tensor<2x10xf32>
  %rhs_slice = tensor.extract_slice %rhs[0, %4] [10, 8] [1, 1]
             : tensor<10x16xf32> to tensor<10x8xf32>
  %result_slice = tensor.extract_slice %shared[%3, %4] [2, 8] [1, 1]
                : tensor<8x16xf32> to tensor<2x8xf32>

  // This is exactly the same operation as before, but now operating on smaller
  // slices of data.
  %partial = linalg.generic {
    indexing_maps = [affine_map<(i, j, k) -> (i, k)>,
                     affine_map<(i, j, k) -> (k, j)>,
                     affine_map<(i, j, k) -> (i, j)>],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%lhs_slice, %rhs_slice : tensor<2x10xf32>, tensor<10x8xf32>)
    outs(%result_slice : tensor<2x8xf32>) {
  ^bb0(%lhs_one: f32, %rhs_one: f32, %init_one: f32):
    %0 = arith.mulf %lhs_one, %rhs_one : f32
    %1 = arith.addf %init_one, %0 : f32
    linalg.yield %1 : f32
  } -> tensor<2x8xf32>

  // Terminator for the loop with tensor-insertion semantics. Inserts a slice
  // into a larger tensor, potentially in parallel.
  scf.forall.in_parallel {
    tensor.parallel_insert_slice %partial into %shared[%3, %4] [2, 8] [1, 1]
        : tensor<2x8xf32> into tensor<8x16xf32>
  }
}
```

- `tensor.extract_slice %t[offsets] [sizes] [strides]` names a rectangular part of a tensor: `%lhs[%3, 0] [2, 10] [1, 1]` is rows `%3` to `%3 + 1`, all 10 columns.
- `shared_outs(%shared = %init)` gives every iteration a view of the tensor being produced; each iteration contributes only its own `2x8` slice through `tensor.parallel_insert_slice` in the `scf.forall.in_parallel` terminator, so the iterations never write the same element.
- The `k` dimension (10) is not tiled: each tile performs the full reduction over `k`, so the tile's operands are a `2x10` and a `10x8` slice.

### 8.2 Dynamic sizes

With dynamically sized tensors, the trip counts and the size of the last, partial tile are computed in the IR. The payload below is the same matrix multiplication with `tensor<?x?xf32>` operands, written in its short named form, `linalg.matmul` (section 10); `--canonicalize --cse` tidy the result:

```bash
cd /Users/roy/study/mlir/transform/Ch0
mlir-opt --transform-interpreter --canonicalize --cse <<'EOF' | sed '/module attributes/,$d'
func.func @matmul_dyn(%lhs: tensor<?x?xf32>, %rhs: tensor<?x?xf32>,
                      %init: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.matmul ins(%lhs, %rhs : tensor<?x?xf32>, tensor<?x?xf32>)
                     outs(%init : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op {transform.readonly}) {
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %root
      : (!transform.any_op) -> !transform.any_op
    %tiled, %loop = transform.structured.tile_using_forall %matmul
      tile_sizes [2, 8] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
EOF
```

Real output **[14]**:

```text
#map = affine_map<()[s0] -> (s0 ceildiv 2)>
#map1 = affine_map<()[s0] -> (s0 ceildiv 8)>
#map2 = affine_map<(d0) -> (d0 * 2)>
#map3 = affine_map<(d0) -> (d0 * 8)>
#map4 = affine_map<(d0)[s0] -> (d0 * -2 + s0, 2)>
#map5 = affine_map<(d0)[s0] -> (d0 * -8 + s0, 8)>
module {
  func.func @matmul_dyn(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
    %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
    %dim_1 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
    %0 = affine.apply #map()[%dim]
    %1 = affine.apply #map1()[%dim_1]
    %2 = scf.forall (%arg3, %arg4) in (%0, %1) shared_outs(%arg5 = %arg2) -> (tensor<?x?xf32>) {
      %3 = affine.apply #map2(%arg3)
      %4 = affine.apply #map3(%arg4)
      %5 = affine.min #map4(%arg3)[%dim]
      %6 = affine.min #map5(%arg4)[%dim_1]
      %extracted_slice = tensor.extract_slice %arg0[%3, 0] [%5, %dim_0] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
      %extracted_slice_2 = tensor.extract_slice %arg1[0, %4] [%dim_0, %6] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
      %extracted_slice_3 = tensor.extract_slice %arg5[%3, %4] [%5, %6] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
      %7 = linalg.matmul ins(%extracted_slice, %extracted_slice_2 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%extracted_slice_3 : tensor<?x?xf32>) -> tensor<?x?xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %7 into %arg5[%3, %4] [%5, %6] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
      }
    }
    return %2 : tensor<?x?xf32>
  }
```

- The trip counts are `dim ceildiv 2` and `dim ceildiv 8` (`#map`, `#map1`), computed from `tensor.dim` before the loop.
- Each tile's size is `min(tile size, remaining elements)` (`affine.min` with `#map4`, `#map5`), so the last tile shrinks when the size is not a multiple of the tile size; the slices then have dynamic sizes too.

### 8.3 Tiling the reduction dimension

The tiles of the `"parallel"` dimensions `i` and `j` are independent. The `"reduction"` dimension `k` is different: two tiles that differ only in `k` accumulate into the same result elements. Asking `tile_using_forall` to tile `k` as well (`tile_sizes [2, 8, 5]`) still produces a loop — the warning shows only on stderr, which the command keeps while discarding the IR:

```bash
cd /Users/roy/study/mlir/transform/Ch0
mlir-opt --transform-interpreter <<'EOF' 2>&1 >/dev/null
func.func @matmul(%lhs: tensor<8x10xf32>, %rhs: tensor<10x16xf32>,
                  %init: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %0 = linalg.matmul ins(%lhs, %rhs : tensor<8x10xf32>, tensor<10x16xf32>)
                     outs(%init : tensor<8x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op {transform.readonly}) {
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %root
      : (!transform.any_op) -> !transform.any_op
    %tiled, %loop = transform.structured.tile_using_forall %matmul
      tile_sizes [2, 8, 5] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
EOF
```

Real output **[15]**:

```text
<stdin>:3:8: warning: tiling is not thread safe at axis #2
  %0 = linalg.matmul ins(%lhs, %rhs : tensor<8x10xf32>, tensor<10x16xf32>)
       ^
<stdin>:3:8: note: see current operation: %0 = linalg.matmul ins(%arg0, %arg1 : tensor<8x10xf32>, tensor<10x16xf32>) outs(%arg2 : tensor<8x16xf32>) -> tensor<8x16xf32>
```

Axis #2 is `k`. The `iterator_types` attribute is what let the transformation tell: it knows `k` is a reduction without analyzing the body. Reductions are tiled with a sequential loop instead, `transform.structured.tile_using_for`, which creates one `scf.for` per tiled dimension and threads the partially computed result through `iter_args`:

```bash
cd /Users/roy/study/mlir/transform/Ch0
mlir-opt --transform-interpreter --canonicalize <<'EOF' | sed '/module attributes/,$d'
func.func @matmul(%lhs: tensor<8x10xf32>, %rhs: tensor<10x16xf32>,
                  %init: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %0 = linalg.matmul ins(%lhs, %rhs : tensor<8x10xf32>, tensor<10x16xf32>)
                     outs(%init : tensor<8x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op {transform.readonly}) {
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %root
      : (!transform.any_op) -> !transform.any_op
    %tiled, %loops:3 = transform.structured.tile_using_for %matmul
      tile_sizes [2, 8, 5]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op,
                                !transform.any_op, !transform.any_op)
    transform.yield
  }
}
EOF
```

Real output **[16]**:

```text
module {
  func.func @matmul(%arg0: tensor<8x10xf32>, %arg1: tensor<10x16xf32>, %arg2: tensor<8x16xf32>) -> tensor<8x16xf32> {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c10 = arith.constant 10 : index
    %c2 = arith.constant 2 : index
    %c5 = arith.constant 5 : index
    %0 = scf.for %arg3 = %c0 to %c8 step %c2 iter_args(%arg4 = %arg2) -> (tensor<8x16xf32>) {
      %1 = scf.for %arg5 = %c0 to %c16 step %c8 iter_args(%arg6 = %arg4) -> (tensor<8x16xf32>) {
        %2 = scf.for %arg7 = %c0 to %c10 step %c5 iter_args(%arg8 = %arg6) -> (tensor<8x16xf32>) {
          %extracted_slice = tensor.extract_slice %arg0[%arg3, %arg7] [2, 5] [1, 1] : tensor<8x10xf32> to tensor<2x5xf32>
          %extracted_slice_0 = tensor.extract_slice %arg1[%arg7, %arg5] [5, 8] [1, 1] : tensor<10x16xf32> to tensor<5x8xf32>
          %extracted_slice_1 = tensor.extract_slice %arg8[%arg3, %arg5] [2, 8] [1, 1] : tensor<8x16xf32> to tensor<2x8xf32>
          %3 = linalg.matmul ins(%extracted_slice, %extracted_slice_0 : tensor<2x5xf32>, tensor<5x8xf32>) outs(%extracted_slice_1 : tensor<2x8xf32>) -> tensor<2x8xf32>
          %inserted_slice = tensor.insert_slice %3 into %arg8[%arg3, %arg5] [2, 8] [1, 1] : tensor<2x8xf32> into tensor<8x16xf32>
          scf.yield %inserted_slice : tensor<8x16xf32>
        }
        scf.yield %2 : tensor<8x16xf32>
      }
      scf.yield %1 : tensor<8x16xf32>
    }
    return %0 : tensor<8x16xf32>
  }
```

- The innermost loop steps `k` (`%arg7`) by 5 and passes the accumulator `%arg8` from one iteration to the next: the order of the `k` tiles is now explicit and sequential.
- Each tile multiplies a `2x5` by a `5x8` slice and accumulates into the `2x8` result slice, which `tensor.insert_slice` puts back into the carried tensor.

With `tile_using_for`, every tiled dimension became an explicit loop, which is the sense in which tiling materializes the loops that `--convert-linalg-to-loops` wrote out all at once in section 5. **Because the operation declares which dimensions are parallel and which reduce, tiling needs no dependence analysis: the schedule picks the tile sizes and the loop kind, and the iterator types tell which choices are safe.**

---

## 9. Producer/Consumer Fusion and Rematerialization

After materializing loops with tiling, another key code generation transformation becomes simple: fusion. Unlike loop fusion, the structured-operations approach allows for producer/consumer fusion even when the (implicit) iteration spaces of the operations do not match. Given a high-level structured operation on tensors, such as `linalg.generic`, one can follow use-def chains to identify:

1. the subset (slice) of the operand that is used by the tile, and
2. the tensor-level structured operation producing the whole tensor that is being sliced.

By inverting the `indexing_map` and applying it to the set of elements accessed through the slice, we can compute the part of the iteration space of the operation defining the full tensor that is necessary to compute the tile. Thus fusion boils down to replacing the `tensor.extract_slice` operation with the tile of the `linalg.generic` producing the original operand.

Let us assume that the matrix multiplication operation is followed by another operation that multiplies each element of the resulting matrix with itself. This trailing elementwise operation has a 2-D iteration space, unlike the 3-D one of the matrix multiplication. Nevertheless, it is possible to tile the trailing operation and then fuse the producer of its operand, the matmul, into the loop generated by tiling. The untiled dimension will be used in its entirety.

The script does that in two steps: `tile_using_forall` tiles the consumer (the elementwise square) by `(2, 8)`, creating the loop, and `transform.structured.fuse_into_containing_op` moves the producer (the matmul) into it (appendix A.3). `transform.split_handle` first splits the handle to both `linalg.generic`s into one per operation, in program order. The command runs the IR twice, from a shell variable: once printing just the lines that show what the transformation itself leaves behind, then once more with `--canonicalize --cse` to clean that up:

```bash
cd /Users/roy/study/mlir/transform/Ch0
IR=$(cat <<'EOF'
func.func @matmul_square(%lhs: tensor<8x10xf32>, %rhs: tensor<10x16xf32>,
                         %init: tensor<8x16xf32>, %out: tensor<8x16xf32>)
    -> tensor<8x16xf32> {
  %result = linalg.generic {
    indexing_maps = [affine_map<(i, j, k) -> (i, k)>,
                     affine_map<(i, j, k) -> (k, j)>,
                     affine_map<(i, j, k) -> (i, j)>],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%lhs, %rhs : tensor<8x10xf32>, tensor<10x16xf32>)
    outs(%init : tensor<8x16xf32>) {
  ^bb0(%lhs_one: f32, %rhs_one: f32, %init_one: f32):
    %0 = arith.mulf %lhs_one, %rhs_one : f32
    %1 = arith.addf %init_one, %0 : f32
    linalg.yield %1 : f32
  } -> tensor<8x16xf32>

  %square = linalg.generic {
    indexing_maps = [affine_map<(i, j) -> (i, j)>,
                     affine_map<(i, j) -> (i, j)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%result : tensor<8x16xf32>)
    outs(%out : tensor<8x16xf32>) {
  ^bb0(%in: f32, %out_one: f32):
    %2 = arith.mulf %in, %in : f32
    linalg.yield %2 : f32
  } -> tensor<8x16xf32>
  return %square : tensor<8x16xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op {transform.readonly}) {
    %generics = transform.structured.match ops{["linalg.generic"]} in %root
      : (!transform.any_op) -> !transform.any_op
    %matmul, %square = transform.split_handle %generics
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    // Tile the consumer: this creates the loop.
    %tiled, %loop = transform.structured.tile_using_forall %square
      tile_sizes [2, 8] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    // Fuse the producer into that loop.
    %fused, %new_loop = transform.structured.fuse_into_containing_op %matmul
      into %loop : (!transform.any_op, !transform.any_op)
                -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
EOF
)
mlir-opt --transform-interpreter <<<"$IR" | grep -E '^    %0 = |affine.apply'
echo ---
mlir-opt --transform-interpreter --canonicalize --cse <<<"$IR" | sed '/module attributes/,$d'
```

Real output **[17]**:

```text
    %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<8x10xf32>, tensor<10x16xf32>) outs(%arg2 : tensor<8x16xf32>) {
      %2 = affine.apply #map3(%arg4)
      %3 = affine.apply #map4(%arg5)
      %4 = affine.apply #map3(%arg4)
      %5 = affine.apply #map4(%arg5)
      %6 = affine.apply #map3(%arg4)
      %7 = affine.apply #map4(%arg5)
---
#map = affine_map<(d0) -> (d0 * 2)>
#map1 = affine_map<(d0) -> (d0 * 8)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map5 = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @matmul_square(%arg0: tensor<8x10xf32>, %arg1: tensor<10x16xf32>, %arg2: tensor<8x16xf32>, %arg3: tensor<8x16xf32>) -> tensor<8x16xf32> {
    %0 = scf.forall (%arg4, %arg5) in (4, 2) shared_outs(%arg6 = %arg3) -> (tensor<8x16xf32>) {
      %1 = affine.apply #map(%arg4)
      %2 = affine.apply #map1(%arg5)
      %extracted_slice = tensor.extract_slice %arg0[%1, 0] [2, 10] [1, 1] : tensor<8x10xf32> to tensor<2x10xf32>
      %extracted_slice_0 = tensor.extract_slice %arg1[0, %2] [10, 8] [1, 1] : tensor<10x16xf32> to tensor<10x8xf32>
      %extracted_slice_1 = tensor.extract_slice %arg2[%1, %2] [2, 8] [1, 1] : tensor<8x16xf32> to tensor<2x8xf32>
      %3 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction"]} ins(%extracted_slice, %extracted_slice_0 : tensor<2x10xf32>, tensor<10x8xf32>) outs(%extracted_slice_1 : tensor<2x8xf32>) {
      ^bb0(%in: f32, %in_3: f32, %out: f32):
        %5 = arith.mulf %in, %in_3 : f32
        %6 = arith.addf %out, %5 : f32
        linalg.yield %6 : f32
      } -> tensor<2x8xf32>
      %extracted_slice_2 = tensor.extract_slice %arg6[%1, %2] [2, 8] [1, 1] : tensor<8x16xf32> to tensor<2x8xf32>
      %4 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel"]} ins(%3 : tensor<2x8xf32>) outs(%extracted_slice_2 : tensor<2x8xf32>) {
      ^bb0(%in: f32, %out: f32):
        %5 = arith.mulf %in, %in : f32
        linalg.yield %5 : f32
      } -> tensor<2x8xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %4 into %arg6[%1, %2] [2, 8] [1, 1] : tensor<2x8xf32> into tensor<8x16xf32>
      }
    }
    return %0 : tensor<8x16xf32>
  }
```

Before cleanup (the part above `---`), the full-size matmul `%0` is still in front of the loop and each fused slice recomputed its own offsets (six `affine.apply`s for two distinct values). The full-size matmul has no users left, and the `affine.apply`s are duplicates, so `--canonicalize --cse` delete both. With readable value names, inline maps and comments (hand-edited from the output above, not this repo's code), the cleaned loop reads:

```mlir
// Same loop as before.
%0 = scf.forall (%i, %j) in (4, 2)
     shared_outs(%shared = %out) -> (tensor<8x16xf32>) {
  // Scale the loop induction variables by the tile sizes.
  %1 = affine.apply affine_map<(d0) -> (d0 * 2)>(%i)
  %2 = affine.apply affine_map<(d0) -> (d0 * 8)>(%j)

  // Take slices of inputs and outputs. Only the "i" and "j" dimensions are sliced.
  %lhs_slice = tensor.extract_slice %lhs[%1, 0] [2, 10] [1, 1]
             : tensor<8x10xf32> to tensor<2x10xf32>
  %rhs_slice = tensor.extract_slice %rhs[0, %2] [10, 8] [1, 1]
             : tensor<10x16xf32> to tensor<10x8xf32>
  %result_slice = tensor.extract_slice %init[%1, %2] [2, 8] [1, 1]
                : tensor<8x16xf32> to tensor<2x8xf32>

  // This is exactly the same matmul slice as before. It replaces the slice
  // extraction for the generic operation below.
  %partial = linalg.generic {
    indexing_maps = [affine_map<(i, j, k) -> (i, k)>,
                     affine_map<(i, j, k) -> (k, j)>,
                     affine_map<(i, j, k) -> (i, j)>],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%lhs_slice, %rhs_slice : tensor<2x10xf32>, tensor<10x8xf32>)
   outs(%result_slice : tensor<2x8xf32>) {
  ^bb0(%lhs_one: f32, %rhs_one: f32, %init_one: f32):
    %5 = arith.mulf %lhs_one, %rhs_one : f32
    %6 = arith.addf %init_one, %5 : f32
    linalg.yield %6 : f32
  } -> tensor<2x8xf32>

  // Take the slice of the final result. Note that we don't need to take
  // the slice of the operand because the matmul operation above computes
  // it in-place.
  %shared_slice = tensor.extract_slice %shared[%1, %2] [2, 8] [1, 1]
                : tensor<8x16xf32> to tensor<2x8xf32>

  // The elementwise operation that we tiled.
  %elemwise = linalg.generic {
    indexing_maps = [affine_map<(i, j) -> (i, j)>,
                     affine_map<(i, j) -> (i, j)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%partial : tensor<2x8xf32>)
   outs(%shared_slice : tensor<2x8xf32>) {
  ^bb0(%in: f32, %out: f32):
    %5 = arith.mulf %in, %in : f32
    linalg.yield %5 : f32
  } -> tensor<2x8xf32>

  // Terminator for the loop with tensor-insertion semantics. Inserts a slice
  // into a larger tensor, potentially in parallel.
  scf.forall.in_parallel {
    tensor.parallel_insert_slice %elemwise into %shared[%1, %2] [2, 8] [1, 1]
        : tensor<2x8xf32> into tensor<8x16xf32>
  }
}
```

- The loop produces one tensor, the square's result: `shared_outs` is initialized from `%out`, the square's `outs` operand, because the loop came from tiling the square.
- The matmul tile takes its init slice from the matmul's own `outs` operand, `%init` (`%arg2`), not from the loop's shared tensor: the producer computes its `2x8` part from scratch in every iteration.
- The slice sizes of the matmul (`2x10`, `10x8`) are what inverting its indexing maps gives for a `2x8` result tile: rows `%1..%1+1` of `%lhs` for all `k`, columns `%2..%2+7` of `%rhs` for all `k`. The untiled dimension `k` is used in its entirety.

This process may result in some elements in the operand tensors being (re)computed on every iteration of the loop. This is also known as **rematerialization** and expresses the tradeoff between performing redundant computations or storing their result in (slow) memory. Here every element of the matmul is still computed exactly once, because the square reads each matmul element once; a consumer that read neighboring elements (a stencil, say) would make neighboring tiles recompute the overlap. **Fusion through slices needs only the indexing maps of the producer, not matching loops: the consumer's tile decides what to compute, and the producer's maps say what that tile needs.**

---

## 10. Shorthand “Named” Forms of Linalg Ops

Linalg provides a set of predefined operations for common cases such as matrix multiplication, dot product, convolution, etc. These operations are equivalent to the `generic` ones but spare the need to spell out the access patterns and the bodies. For example, matrix multiplication is simply:

```mlir
%matmul = linalg.matmul ins(%lhs, %rhs: tensor<8x10xf32>, tensor<10x16xf32>)
                        outs(%init: tensor<8x16xf32>) -> tensor<8x16xf32>
```

The equivalence is something `mlir-opt` can show: `--linalg-generalize-named-ops` rewrites each named op into its `linalg.generic`, and the result is the generic of section 7:

```bash
cd /Users/roy/study/mlir/transform/Ch0
mlir-opt --linalg-generalize-named-ops <<'EOF'
func.func @named(%lhs: tensor<8x10xf32>, %rhs: tensor<10x16xf32>,
                 %init: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %matmul = linalg.matmul ins(%lhs, %rhs: tensor<8x10xf32>, tensor<10x16xf32>)
                          outs(%init: tensor<8x16xf32>) -> tensor<8x16xf32>
  return %matmul : tensor<8x16xf32>
}
EOF
```

Real output **[18]**:

```text
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @named(%arg0: tensor<8x10xf32>, %arg1: tensor<10x16xf32>, %arg2: tensor<8x16xf32>) -> tensor<8x16xf32> {
    %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<8x10xf32>, tensor<10x16xf32>) outs(%arg2 : tensor<8x16xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %1 = arith.mulf %in, %in_0 : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
    } -> tensor<8x16xf32>
    return %0 : tensor<8x16xf32>
  }
}
```

The same three maps, the same iterator types, the same `mulf`/`addf` body. The init and the result must have the same type: `outs` is the initial value of the result. With an `8x10` init for an `8x16` result, the verifier rejects the op — and its generic-form dump in the note shows that `linalg.matmul` carries the `indexing_maps` and the body region itself, only hidden by the short syntax:

```bash
cd /Users/roy/study/mlir/transform/Ch0
mlir-opt <<'EOF'
func.func @named(%lhs: tensor<8x10xf32>, %rhs: tensor<10x16xf32>,
                 %init: tensor<8x10xf32>) -> tensor<8x16xf32> {
  %matmul = linalg.matmul ins(%lhs, %rhs: tensor<8x10xf32>, tensor<10x16xf32>)
                          outs(%init: tensor<8x10xf32>) -> tensor<8x16xf32>
  return %matmul : tensor<8x16xf32>
}
EOF
```

Real output **[19]** (exit status 1):

```text
<stdin>:3:13: error: 'linalg.matmul' op expected type of operand #2 ('tensor<8x10xf32>') to match type of corresponding result ('tensor<8x16xf32>')
  %matmul = linalg.matmul ins(%lhs, %rhs: tensor<8x10xf32>, tensor<10x16xf32>)
            ^
<stdin>:3:13: note: see current operation: 
%0 = "linalg.matmul"(%arg0, %arg1, %arg2) <{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], operandSegmentSizes = array<i32: 2, 1>}> ({
^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
  %1 = "arith.mulf"(%arg3, %arg4) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
  %2 = "arith.addf"(%arg5, %1) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
  "linalg.yield"(%2) : (f32) -> ()
}) : (tensor<8x10xf32>, tensor<10x16xf32>, tensor<8x10xf32>) -> tensor<8x16xf32>
```

Every transformation in this chapter that worked on `linalg.generic` also works on `linalg.matmul` (sections 8.2 and 8.3 tiled the named op directly), because both implement the same structured-op interfaces. **A named op is a `linalg.generic` with a fixed body and maps: it is easier to write and to match by name, and loses none of the structure the transformations rely on.**

---

## 11. Key Takeaways & Pitfalls

### Takeaways

1. **Structure is information the compiler would otherwise have to rediscover.** An n-D `arith.addf`, a `vector.reduction` or a `vector.contract` says *what* is computed; lowering then picks *how* (row-wise `llvm.fadd`s, a reduction intrinsic, dot products or outer products; sections 2–4).
2. **`indexing_maps` + `iterator_types` describe a loop nest without writing it.** The map dimensions are the loops, the operand shapes give the bounds, the maps give the access pattern, and `"parallel"`/`"reduction"` say which loops can run in any order (sections 4–5).
3. **`linalg.generic` is the same description over buffers or tensors**, with an explicit body region; `ins` are read, `outs` are updated (buffers) or initial values (tensors), and bufferization connects the two forms (sections 5 and 7).
4. **Tensors' value semantics make rewrites local.** Elementwise fusion, tiling and producer/consumer fusion all happen on tensors, without alias analysis (sections 7–9).
5. **Tiling materializes the implicit loops.** `scf.forall` for the parallel dimensions, `scf.for` with `iter_args` when a reduction dimension is tiled; dynamic shapes get `ceildiv` trip counts and `affine.min` tile sizes (section 8).
6. **Producer/consumer fusion follows use-def chains through slices**, even across different iteration spaces (3-D matmul into a 2-D elementwise loop), at the price of possible rematerialization (section 9).
7. **Named ops are generics with a name** (section 10).

### Pitfalls

- **Snippets are not complete IR.** An op needs its operands defined (here: function arguments), an SSA name can be defined only once, and attribute dictionaries need their `=` (`indexing_maps = [...]`); `arith.cmpf` needs a comma after the predicate (sections 2.1, 6).
- **`outs` type must match the result type** on tensors (`linalg.matmul ... outs(%init : tensor<8x16xf32>) -> tensor<8x16xf32>`); the verifier says so (section 10).
- **`tile_using_forall` on a reduction dimension only warns** ("tiling is not thread safe at axis #2") and still produces the loop; tile reductions with `tile_using_for` (section 8.3).
- **Transformations leave dead code and duplicates behind.** Fusion keeps the original producer alive until `--canonicalize`, and repeats index computations until `--cse` (section 9).
- **The interpreter keeps the transform script in the output.** The `sed '/module attributes/,$d'` in this chapter's commands removes it for readability.

---

## Appendix A: The transform operations used in this chapter

Chapter 1 introduces the Transform dialect properly. This appendix explains just enough to read the scripts of sections 4.2, 8 and 9. Each script is a `transform.named_sequence @__transform_main` in a module carrying the `transform.with_named_sequence` attribute; `--transform-interpreter` runs that sequence on the enclosing (payload) module. Its argument `%root` is a **handle** to the payload's top-level operation; transform operations take handles to payload operations and return new handles. `{transform.readonly}` promises that the sequence does not consume (invalidate) the root handle.

### A.1 `transform.structured.match` and `transform.apply_patterns`

`transform.structured.match ops{["func.func"]} in %root` returns a handle to every operation named `func.func` nested in `%root`. `transform.apply_patterns to %func { ... }` applies the listed rewrite-pattern sets to the operations inside `%func` until nothing changes; `transform.apply_patterns.vector.lower_contraction lowering_strategy = "outerproduct"` is the set that lowers `vector.contract` to `vector.outerproduct`. The pattern set has four strategies. The loop below substitutes each name into section 4.2's script (an unquoted heredoc, so `$s` expands) and counts the vector ops of each kind the lowering leaves:

```bash
cd /Users/roy/study/mlir/transform/Ch0
for s in dot matmulintrinsics outerproduct parallelarith; do
  printf '%-18s' "$s:"
  mlir-opt --transform-interpreter <<EOF | sed '/module attributes/,$d' | grep -oE '= vector\.[a-z_]+' | sed 's/^= //' | sort | uniq -c | tr -s ' \n' ' '
func.func @matmul(%lhs: vector<8x10xf32>, %rhs: vector<10x16xf32>,
                  %init: vector<8x16xf32>) -> vector<8x16xf32> {
  %result = vector.contract {
    indexing_maps = [affine_map<(i, j, k) -> (i, k)>,
                     affine_map<(i, j, k) -> (k, j)>,
                     affine_map<(i, j, k) -> (i, j)>],
    iterator_types = ["parallel", "parallel", "reduction"]
  } %lhs, %rhs, %init: vector<8x10xf32>, vector<10x16xf32> into vector<8x16xf32>
  return %result : vector<8x16xf32>
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root
      : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.vector.lower_contraction
        lowering_strategy = "$s"
    } : !transform.any_op
    transform.yield
  }
}
EOF
  echo
done
```

Real output **[20]**:

```text
dot:               136 vector.extract 128 vector.insert 128 vector.reduction 1 vector.transpose 
matmulintrinsics:  1 vector.matrix_multiply 3 vector.shape_cast 
outerproduct:      20 vector.extract 10 vector.outerproduct 1 vector.transpose 
parallelarith:    1416 vector.extract 1416 vector.insert 128 vector.reduction 
```

- `dot`: one `vector.reduction` per result element (8 × 16 = 128) over an elementwise product, after transposing `%rhs` so its columns become rows — the same choice as the default conversion of section 4.2.
- `matmulintrinsics`: the operands are flattened to 1-D (`vector.shape_cast`) and handed to `vector.matrix_multiply`, which becomes LLVM's matrix-multiply intrinsic.
- `outerproduct`: the 10 rank-1 updates of section 4.2.
- `parallelarith`: the multiplications and reductions are spelled out element by element, hence the 1416 extracts and inserts.

### A.2 `transform.structured.tile_using_forall` and `tile_using_for`

`transform.structured.tile_using_forall %op tile_sizes [2, 8]` tiles every operation behind `%op` (any op implementing the tiling interface: `linalg.generic`, `linalg.matmul`, ...) by the given sizes, one per loop dimension in order, with trailing dimensions untiled (size 0 also means "don't tile"). It returns two handles: the tiled operation inside the loop and the `scf.forall` loop. `transform.structured.tile_using_for` does the same with one `scf.for` per tiled dimension and returns the tiled op plus one handle per loop (`%loops:3` in section 8.3).

### A.3 `transform.split_handle` and `transform.structured.fuse_into_containing_op`

`transform.split_handle %h` splits a handle to N operations into N handles, one each, in the order the operations appear in the payload (the script fails if the count doesn't match the number of results). `transform.structured.fuse_into_containing_op %producer into %loop` finds the uses of `%producer`'s result inside `%loop` (the `tensor.extract_slice`s), and replaces each with a tile of the producer computed inside the loop; it returns handles to the fused op and to the (possibly updated) loop.

---

## Appendix B: Running this README's commands: `examples.sh`

[`examples.sh`](examples.sh), a symlink to the shared [`../examples.sh`](../examples.sh), runs the commands of this README and shows each one with its output and exit status. Nothing needs to be built for this chapter:

- `./examples.sh` runs every command, in README order;
- `./examples.sh 7` runs command [7] only (several numbers are fine);
- `./examples.sh -l` lists the numbered commands without running them:

```bash
cd /Users/roy/study/mlir/transform/Ch0
./examples.sh -l
```

Real output **[21]**:

```text
[ 1] 2.1   mlir-opt <<'EOF'
[ 2] 2.2   mlir-opt --convert-arith-to-llvm <<'EOF'
[ 3] 2.2   mlir-opt --math-uplift-to-fma <<'EOF'
[ 4] 3.1   mlir-opt --convert-vector-to-llvm <<'EOF'
[ 5] 3.2   mlir-opt <<'EOF'
[ 6] 4.1   mlir-opt --convert-vector-to-llvm <<'EOF'
[ 7] 4.2   mlir-opt --convert-vector-to-llvm <<'EOF' | grep -c 'llvm.intr.vector.reduce.fadd'
[ 8] 4.2   mlir-opt --transform-interpreter <<'EOF' | sed '/module attributes/,$d'
[ 9] 5     mlir-opt --convert-linalg-to-loops <<'EOF'
[10] 6     mlir-opt --convert-linalg-to-loops <<'EOF'
[11] 7.1   mlir-opt --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" <<'EOF'
[12] 7.2   mlir-opt --linalg-fuse-elementwise-ops <<'EOF'
[13] 8.1   mlir-opt --transform-interpreter <<'EOF' | sed '/module attributes/,$d'
[14] 8.2   mlir-opt --transform-interpreter --canonicalize --cse <<'EOF' | sed '/module attributes/,$d'
[15] 8.3   mlir-opt --transform-interpreter <<'EOF' 2>&1 >/dev/null
[16] 8.3   mlir-opt --transform-interpreter --canonicalize <<'EOF' | sed '/module attributes/,$d'
[17] 9     IR=$(cat <<'EOF'
[18] 10    mlir-opt --linalg-generalize-named-ops <<'EOF'
[19] 10    mlir-opt <<'EOF'
[20] A.1   for s in dot matmulintrinsics outerproduct parallelarith; do
[21] B     ./examples.sh -l
```

The script doesn't keep its own copy of the commands: it reads them from `README.md` on every run. A command is every `bash` code block in file order, except a build block (its first command after `cd` is `cmake`) and file excerpts (a `***path***` label above the fence). Number *N* is the *N*-th such block, and the README marks that block's output "Real output **[N]**". Adding, removing or reordering a command block therefore renumbers the ones after it.

---

## Links

- [Transform Ch0: A Primer on “Structured” Linalg Operations](https://mlir.llvm.org/docs/Tutorials/transform/Ch0/)
- [Linalg dialect](https://mlir.llvm.org/docs/Dialects/Linalg/) and its [rationale](https://mlir.llvm.org/docs/Rationale/RationaleLinalgDialect/)
- [Vector dialect](https://mlir.llvm.org/docs/Dialects/Vector/)
- [Bufferization](https://mlir.llvm.org/docs/Bufferization/)
- [Transform dialect](https://mlir.llvm.org/docs/Dialects/Transform/)
- Next: [Chapter 1: Combining Existing Transformations](../Ch1/README.md)
