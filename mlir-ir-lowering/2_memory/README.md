# 2 — Memory in MLIR

### Where does a number live?

Here is a question that sounds too simple to be interesting: *where does a number
live?* In mathematics, it doesn't live anywhere. The entry $A_{ij}$ of a matrix
is a pure value — it has no address, no lifetime, no notion of being "overwritten."
You can talk about "the identity matrix" without anything in the universe storing
it. In hardware, the exact opposite is true: a number is **nothing until it is
somewhere**. It's bytes at an address, in a register or a cache line or a page of
RAM, and the entire performance story of a program — copies, alignment, locality,
aliasing — is the story of *where its data lives and when*.

[Chapter 1](../1_intro/) walked computation down the compiler staircase: a loop
became branches, became phi nodes, became registers. But every value there was a
scalar that lived in a register and then evaporated. The only "memory" was a
function returning a single integer. **This chapter is about the other half of
the descent: the data.** Real programs don't pass scalars around, they pass
*arrays* — and an array is precisely the thing that has to live somewhere.

So MLIR gives you a ladder for data that runs in parallel with the one for code,
and the rungs are distinguished by exactly that question — *does this value have
an address yet?* The next section climbs it rung by rung.

> Based on Stephen Diehl's *"MLIR Part 2 — Memory in MLIR"*
> ([`../reference/`](../reference/)). The PDF is mostly **illustrative
> snippets**; this chapter turns the key ones into **runnable, verified
> examples** (Homebrew LLVM 20.1.8, Apple Silicon) and marks clearly which is
> which.

---

## MLIR's layered memory model

MLIR's memory architecture is a deliberate balance between **high-level
abstraction** (easy to write, easy to optimize) and **low-level efficiency**
(precise control over layout and access). Rather than force you to pick one, it
lets you start at whatever altitude suits the problem and *progressively lower*
toward the metal as execution demands it — the same philosophy you saw with
control flow in Chapter 1, now applied to data. There are three altitudes, and
the deciding question at each is *does this value have an address yet, and how
hardware-specific is that address?*

- **Top — `tensor`: a number that lives nowhere.** A pure mathematical value:
  immutable, with a shape and element type but **no location**. This is the world
  matrix math wants to live in, because a value with no address can't be secretly
  mutated or aliased, which makes it trivially safe for the compiler to reorder,
  fuse, and optimize.
- **Middle — `memref`: a number that's been given a home, but an abstract one.**
  A concrete buffer — pointer, offset, shape, strides — so it finally *has* an
  address and the CPU and NumPy can reach it. But it is still a high-level,
  **target-independent** description: it carries rich metadata (dynamic
  dimensions, arbitrary strided layouts) and knows nothing yet about registers,
  alignment, or a specific instruction set. This is the rung where data becomes
  *addressable* without yet becoming *hardware* — which is exactly why a memref is
  not "the bottom."
- **Bottom — `llvm` structs/arrays and `vector`: raw machine storage.** Here the
  metadata is gone. An `!llvm.struct`/`!llvm.array` is exactly a C aggregate
  (fixed size, no shape/stride bookkeeping), and a `vector<Nxf32>` is a SIMD
  register. This is what a memref descriptor is ultimately *built out of* once it
  finishes lowering, and what maps onto real addresses and AVX/NEON lanes.

```text
   more abstract  ┌──────────────────────────────────────────────┐  "lives nowhere"
        ▲         │ tensor<?x?xf32>     a value: shape, no address │  (SSA, immutable)
        │         └──────────────────────────────────────────────┘
        │            │  bufferization   ← the one big crossing (top → middle)
        │         ┌──────────────────────────────────────────────┐  "has a home,
        │         │ memref<?x?xf32>     ptr+offset+shape+strides   │   but abstract"
        │         └──────────────────────────────────────────────┘  (addressable)
        │            │  -finalize-memref-to-llvm / -convert-vector-to-llvm
        ▼         ┌──────────────────────────────────────────────┐  "raw storage"
   more concrete  │ !llvm.struct/array,  vector<8xf32>            │  (C aggregate,
                  └──────────────────────────────────────────────┘   SIMD register)
```

| Layer | What it is | Mutable? | Has an address? | Hardware-specific? | Dialect |
| --- | --- | --- | --- | --- | --- |
| **Tensor** | Pure mathematical value (a NumPy array as a *value*) | No (SSA) | No | No | `tensor` |
| **MemRef** | A concrete memory buffer: pointer + shape + strides | Yes | Yes | No (still abstract) | `memref` |
| **LLVM struct/array** | C-level aggregates, fixed size, no metadata | Yes | Yes | Yes | `llvm` |
| **Vector** | A SIMD register of N lanes | — | (registers) | Yes | `vector` |

The single most important crossing on this ladder is the one **between the top
and the middle** — turning a homeless tensor *value* into an addressable memref
*buffer*. It has a name, **bufferization**, and it is the data-world equivalent of
Chapter 1's lowering passes; everything below memref is then just further
concretization toward the metal. Two bridges tie the layers together, and each
gets its own step below:

- **Bufferization** — tensor (value) → memref (buffer): the top→middle crossing.
  Used in Step 1, dissected in Step 3.
- **`llvm.emit_c_interface`** — wraps a memref kernel so C/Python can call it.
  Used in Steps 2–3, dissected in Step 4.

The practical payoff that motivates all of it: once data reaches the memref rung,
handing real arrays back and forth with **NumPy costs zero copies** — because a
NumPy array and an MLIR memref are, underneath, *the same handful of bytes
describing a pointer* (Step 2).

> Because the bridges are *used* before they're *explained*, each step says up
> front which concept it relies on and where the deep dive lives — so you can
> read start-to-finish without getting stuck on a forward reference.

---

## Chapter layout

Each numbered directory is one step. The **Run** column is the single command
to build-and-run it; every step uses `build.sh`.

| Step | Directory | Topic | Runnable? |
| --- | --- | --- | --- |
| 1 | [`1_tensor/`](1_tensor/) | Tensors: type system + ops, then the identity matrix | 🧩 catalog + ✅ Python (2 kernels) |
| 2 | [`2_array_add/`](2_array_add/) | MemRefs + NumPy integration (AOT **and** JIT) | ✅ Python |
| 3 | [`3_low_level_llvm/`](3_low_level_llvm/) | LLVM structs/arrays, vectors, bufferization, C-iface | 🧩 snippets + ✅ one SIMD demo |
| 4 | [`4_C_compatible_wrappers/`](4_C_compatible_wrappers/) | What `llvm.emit_c_interface` actually generates | ✅ Python |

Shared helper: [`common/np_memref.py`](common/np_memref.py) — the `ctypes`
MemRef descriptor + NumPy adapter, imported by Steps 2–4 (Step 1 needs a 2-D
descriptor, so it keeps its own).

**Legend:** ✅ runnable = compiled to a native library and executed.
🧩 snippet = lowered/inspected with `mlir-opt`, not executed.

> **Why some files are snippets.** A *pure* tensor has no memory, so it can't be
> executed — code that only manipulates tensors (Step 1's `tensor_ops.mlir`, which
> catalogs `@make`/`@access`, and all of Step 3's concept files) is meant to be
> lowered/inspected with `mlir-opt`, not run. To get a runnable program you
> bufferize down to memrefs (Steps 1, 2, 4). Several PDF snippets also don't run
> as written — the PDF's `@identity` ends with a bare `return` (no output) and the
> `array_add` compile commands have typos (`arary_add.ll`) — which this repo fixes
> in the runnable versions (Step 1's `identity_return.mlir` is the corrected, executable
> form of that `@identity`).

---

## Step 1 — Tensors (`1_tensor/`) · 🧩 + ✅

**Goal:** understand the **tensor** — MLIR's highest-level data type — and why
turning one into something you can actually *run* forces it down into a memref.

### What a tensor is

A tensor is an **immutable, side-effect-free value**. It has a shape and an
element type but **no memory address** — you cannot take a pointer to it, alias
it, or mutate it. "Updating" an element produces a *new* tensor. This value
semantics is exactly what makes tensors easy for the compiler to analyze and
reorder, which is why they're the natural representation for matrix / ML /
linear-algebra math.

### The tensor type system

Shapes are written with `x` as the delimiter between dimensions and the element
type. A number is a **static** dimension (size fixed at compile time); `?` is a
**dynamic** dimension (size known only at runtime). `!Name` aliases keep long
types readable:

```mlir
!TensorStatic = tensor<10x10x10x10xf32>   // fully static 4-D
!TensorDyn    = tensor<?x?x10x10xf32>      // first two dims dynamic, last two static
```

### The core tensor ops

| Op | Meaning |
| --- | --- |
| `tensor.empty(%d0, %d1)` | Allocate an (uninitialized) tensor; one operand per `?` dim |
| `tensor.from_elements %a, %b, …` | Build a small tensor literal from SSA values |
| `tensor.extract %t[%i, %j]` | Read one element (pure read) |
| `tensor.insert %x into %t[%i, %j]` | Produce a **new** tensor with one element changed |
| `tensor.generate %m, %n { … }` | Build a tensor by running a region once per index tuple |
| `tensor.yield %v` | Inside a `generate` region: the value of the current element |

[`tensor_ops.mlir`](1_tensor/tensor_ops.mlir) (🧩 **inspect-only**) is a
heavily-commented catalog of the pure tensor-producing / tensor-accessing ops
(`@make`, `@access`). It isn't compiled to a library — there's nothing to
execute, because a pure tensor has no memory. The one op here you *can* turn into
a runnable program — `tensor.generate`, building the identity matrix — has been
extracted into its own file, [`identity_return.mlir`](1_tensor/identity_return.mlir)
(covered below). Inspect the catalog with:

```bash
mlir-opt 1_tensor/tensor_ops.mlir                 # parse + verify, print back
mlir-opt 1_tensor/tensor_ops.mlir -canonicalize   # unused tensor ops vanish (they're dead code)
```

### Why running a tensor needs bufferization

Because a tensor has no address, you can't hand one to C/Python or even execute
it directly. To run a tensor program you must **bufferize** it — rewrite the
value-semantic tensor ops into pointer-semantic `memref` ops backed by real
buffers. That's why the *runnable* files below necessarily involve memrefs:
that crossing is not incidental, it's the whole point. (Bufferization is
dissected in [Step 3](3_low_level_llvm/).)

The lowering hinges on **one-shot bufferization** with
`bufferize-function-boundaries=true`, which rewrites the function *signature*
too, so a `tensor<?x?xi32>` becomes a `memref<?x?xi32>` that C can pass:

```bash
mlir-opt identity_fill.mlir -one-shot-bufferize="bufferize-function-boundaries=true" ...
```

### The runnable kernels (✅)

The example is the **identity matrix**, chosen because it's the simplest function
that's genuinely *index-dependent* — each element is decided by comparing its two
coordinates:

$$A_{ij} = \begin{cases} 1 & \text{if } i = j \\ 0 & \text{otherwise} \end{cases}$$

This maps perfectly onto `tensor.generate`, whose region runs once per index tuple
`(i, j)` and yields that element's value — a clean demonstration of why tensors
are described as *side-effect-free*: you declare *what each element is*, not *how
to write it into memory*. Both files below do exactly this (yield `1` when
`i == j`, else `0`); they differ only in how the finished matrix crosses the C
boundary — i.e. the calling convention:

**`identity_return.mlir` — return the tensor (callee allocates).** This is the
PDF-faithful form: the function takes the two dimensions and *returns* a fresh
`tensor<?x?xi32>`.

*1_tensor/identity_return.mlir* (the function)
```mlir
func.func @identity(%m : index, %n : index) -> tensor<?x?xi32> attributes {llvm.emit_c_interface} {
  %out = tensor.generate %m, %n {
  ^bb0(%i : index, %j : index):                  // region runs for every (i, j)
    %ni = arith.index_cast %i : index to i32      // index -> i32 so we can compare
    %nj = arith.index_cast %j : index to i32
    %eq = arith.cmpi eq, %ni, %nj : i32           // i == j ?  -> i1 (1 on the diagonal)
    %v  = arith.extui %eq : i1 to i32             // widen i1 to i32 (1 or 0)
    tensor.yield %v : i32                          // value of this (i, j) element
  } : tensor<?x?xi32>
  return %out : tensor<?x?xi32>
}
```

The shape comes *in* as `%m, %n` and the matrix goes *out* as the return value.
There is no buffer in the source at all — it's pure tensor. Bufferization
(Step 3) is what later turns that returned tensor into a heap-allocated memref,
which is why the C caller ends up owning a `malloc`'d buffer it must `free`.

**`identity_fill.mlir` — fill a caller's buffer (destination-passing).** Here the
function takes the output `memref<?x?xi32>` as a parameter and returns nothing;
the same `tensor.generate` result is written into that buffer.

*1_tensor/identity_fill.mlir* (the function)
```mlir
module {
  func.func @identity(%out: memref<?x?xi32>) attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %m = memref.dim %out, %c0 : memref<?x?xi32>
    %n = memref.dim %out, %c1 : memref<?x?xi32>

    %t = tensor.generate %m, %n {
      ^bb0(%i: index, %j: index):
        %ni = arith.index_cast %i : index to i32
        %nj = arith.index_cast %j : index to i32
        %eq = arith.cmpi eq, %ni, %nj : i32
        %v  = arith.extui %eq : i1 to i32
        tensor.yield %v : i32
    } : tensor<?x?xi32>

    bufferization.materialize_in_destination %t in writable %out
      : (tensor<?x?xi32>, memref<?x?xi32>) -> ()
    return
  }
}
```

Note two consequences of taking the buffer as a parameter: the dimensions are no
longer arguments — they're recovered from the buffer with `memref.dim` — and
`bufferization.materialize_in_destination` writes the result *in place* rather
than allocating.

**Comparison.** The compute is identical — the same `tensor.generate` body, down
to the SSA value names; everything else is the calling convention:

| | `identity_return.mlir` | `identity_fill.mlir` |
| --- | --- | --- |
| Calling convention | **returns** the tensor (callee allocates) | **destination-passing** (caller's buffer) |
| Signature | `(index, index) -> tensor<?x?xi32>` | `(memref<?x?xi32>) -> ()` |
| Where do `m, n` come from? | passed in as arguments | read from the buffer via `memref.dim` |
| How the result leaves | `return` the tensor | `materialize_in_destination` into `%out` |
| Who allocates the buffer | the **callee** (after bufferization) | the **caller** (Python's `np.zeros`) |
| Who frees it | the caller must `free()` it | nobody — it's the caller's NumPy array |
| Extra allocation / copy | yes (a fresh result buffer) | none (in-place) |
| Python driver | `run_return` | `run_fill` |
| Closest to | the PDF snippet (the runnable form of the catalog's `@identity`) | the `array_add` style of Step 2 |

Neither is "better" — they're the two standard ways MLIR kernels exchange arrays
with a caller. Destination-passing (`fill`) is the more common in real pipelines
because it lets the caller control allocation and avoids copies; returning a
tensor (`return`) reads more naturally and matches how you'd write the math on
paper. Both are built and run by `build.sh`.

### The Python driver (`aot_main.py`)

A single driver loads and calls both libraries; the only real difference between
them is the **calling convention**, captured in two small helpers:

- **`run_fill(lib, m, n)`** — *destination-passing*. Python allocates the output
  `np.zeros((m, n))`, wraps it in a `MemRef2D_i32` descriptor pointing **straight
  at NumPy's buffer** (zero copy), and passes that one pointer. The kernel's
  signature is `void _mlir_ciface_identity(MemRef2D *out)`; it writes results
  directly into the array Python already holds, so there's nothing to read back
  and nothing to free.
- **`run_return(lib, m, n)`** — *callee allocates, caller frees*. The kernel
  returns a tensor, so its C signature is
  `void _mlir_ciface_identity(MemRef2D *sret, int64_t m, int64_t n)`: MLIR
  `malloc`s the buffer and writes the **descriptor** back through the `sret`
  out-pointer. Python then rebuilds a NumPy array from that descriptor's
  `aligned` pointer + `shape` + `stride` (scaling strides from *elements* to
  *bytes*), **copies it out**, and only then calls `libc.free(allocated)` — the
  copy matters because the view aliases the about-to-be-freed buffer.

Both helpers share one piece of glue, `MemRef2D_i32` — a `ctypes.Structure`
mirroring MLIR's descriptor (`allocated`, `aligned`, `offset`, `shape[2]`,
`stride[2]`). This is the same descriptor idea as `common/np_memref.py`, but 2-D;
Step 1 keeps its own because the shared helper is 1-D. The file is heavily
commented — read it alongside this table.

**Run:** `cd 1_tensor && bash build.sh`

**Output** (both kernels, one per library):

```
Identity matrix generated successfully — identity_fill.mlir   (caller's buffer)
[[1 0 0 0 0]
 [0 1 0 0 0]
 [0 0 1 0 0]
 [0 0 0 1 0]
 [0 0 0 0 1]]
... (identity_return prints the same matrix)
```

---

## Step 2 — MemRefs + NumPy, AOT & JIT (`2_array_add/`) · ✅ runnable

**Goal:** the workhorse pattern — a **MemRef** kernel called from Python over
**NumPy** arrays with **no copying**.

A **MemRef** is MLIR's primary abstraction for a **memory buffer**, and the
concrete counterpart to a tensor. Where a tensor is a value with no address, a
memref *is* the address: essentially a pointer to a region of memory carried
alongside just enough metadata to describe its structure. That is exactly what
makes it the **bridge** between high-level array concepts (a NumPy array, a
tensor) and low-level memory operations — the rung where math meets pointers.

It is **not** a raw `float*`. It's a small struct (the "descriptor") whose core
components divide into two groups: the **data** (pointers to the actual buffer)
and the **metadata** (how to interpret what those pointers reach):

| Field | Kind | Meaning | Why it exists |
| --- | --- | --- | --- |
| `allocated` ptr | data | Start of the underlying allocation | What you actually `free()`; may sit *before* the first real element |
| `aligned` ptr | data | First properly-aligned element | What loads/stores address; SIMD needs aligned data, so this can differ from `allocated` |
| `offset` | metadata | Elements from `aligned` to element 0 | Lets a memref be a *view* into a bigger buffer without copying |
| `shape[]` | metadata | Size of each dimension | The `?` dynamic dims from the type get their real values here, at runtime |
| `strides[]` | metadata | Elements to step per dimension | Encodes the layout, so a transpose or slice is just different strides — no data movement |

That last pair is the key to performance: addressing element `[i, j]` is just
`aligned + offset + i*stride[0] + j*stride[1]`, so **reshaping, slicing, and
transposing are metadata edits, not memory copies**. MLIR can even spell the
layout out in the type — these three are the same thing, increasingly explicit:

```mlir
memref<3x4xf32>                                  // implicit row-major
memref<3x4xf32, strided<[4, 1], offset: 0>>      // strides written out
memref<3x4xf32, affine_map<(i, j) -> (i, j)>>    // layout as an affine map
```

The kernel `array_add` (`c[i] = a[i] + b[i]` over three `memref<1024xf32>`)
carries the `llvm.emit_c_interface` attribute, so the compiled library exports a
C-callable `_mlir_ciface_array_add` (the *why* and the generated code are Step 4).
The one new lowering pass versus Chapter 1 is `-finalize-memref-to-llvm`, which
turns `memref.load`/`store` and the descriptor struct itself into plain LLVM
pointer arithmetic — the moment the abstraction finally becomes addresses.

*2_array_add/array_add.mlir* (the kernel)
```mlir
module {
  func.func @array_add(%arg0: memref<1024xf32>,    // input  a
                       %arg1: memref<1024xf32>,     // input  b
                       %arg2: memref<1024xf32>)     // output c (written in place)
      attributes {llvm.emit_c_interface} {
    // Loop bounds are `index`-typed (platform-sized integers).
    %c0    = arith.constant 0    : index   // start
    %c1024 = arith.constant 1024 : index   // end (exclusive)
    %c1    = arith.constant 1    : index   // step

    // One element per iteration: load a[i] and b[i], add, store into c[i].
    scf.for %arg3 = %c0 to %c1024 step %c1 {
      %0 = memref.load %arg0[%arg3] : memref<1024xf32>   // a[i]
      %1 = memref.load %arg1[%arg3] : memref<1024xf32>   // b[i]
      %2 = arith.addf %0, %1 : f32                        // a[i] + b[i]
      memref.store %2, %arg2[%arg3] : memref<1024xf32>   // c[i] = ...
    }
    return
  }
}
```

**Files:**

| File | Role |
| --- | --- |
| `array_add.mlir` | the memref kernel (annotated) |
| `aot_main.py` | **ahead-of-time**: `ctypes.CDLL` loads the pre-built `.dylib` and calls the wrapper |
| `jit_main.py` | **just-in-time**: shells to `mlir-opt`/`mlir-translate` at runtime, then **llvmlite** JITs the IR |
| `../common/np_memref.py` | builds a `MemRefDescriptor` pointing straight at the NumPy buffer |

### NumPy array memory layout

Before bridging the two worlds, look at how a NumPy array is itself laid out —
because it turns out to carry the *same* information a memref descriptor does. A
NumPy array is a thin `PyObject` wrapper around a raw memory buffer plus four
pieces of metadata:

- **Data buffer** — the contiguous block of memory holding the actual values.
- **Shape** — a tuple describing the array's dimensions.
- **Strides** — a tuple of steps **in bytes** to move one element along each
  dimension. (Watch out: NumPy strides are in *bytes*; MLIR memref strides are in
  *elements*.)
- **Dtype** — the element type (e.g. `float32`).

You can inspect all of it directly:

```python
import numpy as np
a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
a.shape        # (2, 3)
a.strides      # (12, 4)   -> 12 bytes to the next row, 4 to the next column
a.dtype        # float32
a.itemsize     # 4         (bytes per element)
a.ctypes.data  # base address of the buffer, e.g. 0x7f926d6e2b20
```

By default NumPy arrays are **C-contiguous** (row-major), so element `[i, j]`
lives at `base_address + i*strides[0] + j*strides[1]`. That is exactly the
addressing a memref performs (just counted in elements rather than bytes) — which
is precisely why the two can share one buffer with no copy.

### Converting NumPy arrays to MemRefs (`common/np_memref.py`)

To hand a NumPy array to MLIR we need a Python object whose memory layout matches
MLIR's memref descriptor exactly. That bridge is
[`common/np_memref.py`](common/np_memref.py): a `ctypes.Structure` mirroring the
descriptor, plus a helper that populates it from a NumPy array.

*common/np_memref.py* (the descriptor + adapter)
```python
import numpy as np
from ctypes import c_void_p, c_longlong, Structure


class MemRefDescriptor(Structure):
  """ctypes layout matching MLIR's 1-D memref descriptor.

  Must match the struct that `-finalize-memref-to-llvm` produces for a
  memref<Nxf32>:  { ptr, ptr, i64, [1 x i64], [1 x i64] }.
  """
  _fields_ = [
    ("allocated", c_void_p),        # base pointer (the one you'd free())
    ("aligned",   c_void_p),        # aligned data pointer (often same as allocated)
    ("offset",    c_longlong),      # offset into data, in ELEMENTS (not bytes)
    ("shape",     c_longlong * 1),
    ("stride",    c_longlong * 1),
  ]


def numpy_to_memref(arr):
  """Wrap a 1-D contiguous NumPy array in a MemRefDescriptor (no copy).

  The descriptor points straight at the NumPy buffer, so MLIR and NumPy
  share the same memory. `allocated` and `aligned` are set to the same
  pointer because NumPy gives us a single buffer with no separate
  free-handle (and nothing here ever free()s it).
  """
  if not arr.flags["C_CONTIGUOUS"]:
    arr = np.ascontiguousarray(arr)

  desc = MemRefDescriptor()
  desc.allocated = arr.ctypes.data_as(c_void_p)
  desc.aligned   = desc.allocated
  desc.offset    = 0
  desc.shape[0]  = arr.shape[0]
  desc.stride[0] = 1            # contiguous 1-D: stride is 1 element
  return desc
```

The five fields mirror what MLIR expects: **allocated** points to the start of
the allocated memory; **aligned** points to the aligned data (often the same
pointer); **offset** is the number of elements to skip from the start; **shape**
holds the dimensions; and **stride** holds the steps (in elements) between
consecutive elements in each dimension. The `numpy_to_memref` helper then does
four things — ensure the array is contiguous, create the descriptor, point it at
the NumPy buffer, and configure shape/stride — **without copying** the underlying
data.

```text
   NumPy ndarray            ONE buffer in RAM            MLIR memref descriptor
   ┌──────────────┐        ┌───────────────────┐        ┌──────────────────┐
   │ .ctypes.data ├───────►│ f32 f32 f32 … f32  │◄───────┤ allocated/aligned│
   │ .shape (N,)  │        └───────────────────┘        │ shape[0] = N     │
   │ .strides     │         (no copy — both              │ stride[0] = 1    │
   └──────────────┘          descriptors point here)     │ offset = 0       │
                                                         └──────────────────┘
```

**Why this is zero-copy.** Both sides describe the same thing: `numpy_to_memref`
reads `arr.ctypes.data` (NumPy's base pointer) and writes it into a descriptor
that points back at NumPy's own buffer. Because NumPy arrays are C-contiguous by
default (row-major, last-axis `stride = 1`), the layouts already agree, and MLIR
writes its results *directly into the array Python is holding*. The two languages
look at one block of memory through two descriptors — which is the whole reason
MLIR is a practical backend for the NumPy/ML ecosystem.

> **AOT vs JIT — a real trade-off, not a style choice.** Both drivers call the
> identical compiled kernel; they differ only in *when* compilation happens.
>
> - **AOT** (`aot_main.py`) compiles once at build time and ships a `.dylib`.
>   Simpler, no toolchain needed at runtime, ideal for production — you just
>   distribute the library with your package and `ctypes.CDLL` it.
> - **JIT** (`jit_main.py`) shells out to `mlir-opt`/`mlir-translate` and has
>   **llvmlite** compile the IR *at runtime*. It offers several advantages:
>   - the MLIR can be **generated or modified at runtime**;
>   - there is **no separate compiled library** to build, ship, and manage;
>   - it can **optimize for the specific host CPU architecture** at runtime.
>
>   It also has drawbacks: **compilation overhead at startup**, a **more complex
>   setup** that depends on the MLIR tools being installed, and the need to
>   **handle compilation errors at runtime** rather than at build time.
>
> (This step also fixes two PDF-era papercuts: the deprecated `llvm.initialize()`
> call, and the filename typos — `arary_add.ll` — in the build commands.)

**Run:** `cd 2_array_add && bash build.sh`  *(builds the dylib, then runs both drivers)*

**Output:**

```
python aot_main.py
Array addition successful!
First few elements: [3. 3. 3. 3. 3.]
python jit_main.py
Array addition successful!
First few elements: [3. 3. 3. 3. 3.]
```

(`a = ones(1024)`, `b = 2*ones(1024)` ⇒ `c = 3` everywhere.)

---

## Step 3 — Low-level LLVM, vectors, bufferization (`3_low_level_llvm/`) · 🧩 + ✅

**Goal:** the C-level building blocks *beneath* `memref`/`tensor`. This is the
**bottom rung** of the memory ladder — the point where the comfortable
abstractions run out and you see the raw material everything above is built from.
Most files here are **inspect-only snippets** you lower with `mlir-opt` to watch
the IR change; one (`array_add_vec.mlir`) is compiled and run end-to-end.

| File | Kind | Shows |
| --- | --- | --- |
| `structs_arrays.mlir` | 🧩 | `!llvm.struct<(i32, f32)>`, `!llvm.array<N x T>`, `insert`/`extractvalue` |
| `vectors.mlir` | 🧩 | `vector<4xf32>` SIMD: `arith` elementwise, `vector.extract`/`shuffle`/`reduction`/`broadcast` |
| `bufferization.mlir` | 🧩 | tensor→memref three ways (manual, `-one-shot-bufferize`, `materialize_in_destination`) |
| `c_interface.mlir` | 🧩 | how `emit_c_interface` yields `@fn` (unrolled) **and** `@_mlir_ciface_fn` (the Step 4 deep dive) |
| `array_add_vec.mlir` | ✅ | Step-2's kernel processing **8 floats/iteration** with `vector.load`/`store` |

### LLVM structs & arrays — the C floor (`structs_arrays.mlir` 🧩)

When you drop into the `llvm` dialect you get exactly C's aggregate types: structs
and fixed-size arrays. Both are **rigid** — fixed size known at compile time, with
**no** shape/offset/stride metadata (unlike a memref, which carries all that) —
and both are value-typed. Because the LLVM dialect is SSA, you never mutate one in
place; you build a *new* value with `llvm.insertvalue` and read out of it with
`llvm.extractvalue`.

#### Structs

`!llvm.struct<(i32, f32)>` *is* C's `struct { int a; float b; }`. You can give it
a type alias (`!Pair`), zero-initialize it, then fill each field:

```mlir
!Pair = !llvm.struct<(i32, f32)>             // struct Pair { int a; float b; };
%z  = llvm.mlir.zero        : !Pair
%a  = llvm.mlir.constant(42 : i32) : i32
%s0 = llvm.insertvalue %a, %z[0] : !Pair      // build a new struct value (field 0 = 42)
%x  = llvm.extractvalue %s0[0]   : !Pair      // read field 0 back out
```

#### Arrays

`!llvm.array<N x T>` is fixed-size sequential storage — the rigid sibling of
`memref`, closely mirroring a C array. It has a size fixed at compile time and
carries no metadata about its structure. Element types compose, so you can make an
array of the struct above:

```mlir
!IntArray  = !llvm.array<10 x i32>            // like int[10]
!PairArray = !llvm.array<4 x !Pair>           // array of 4 Pairs

// create and initialize a constant array {1, 2, 3, 4}
%arr = llvm.mlir.constant(dense<[1, 2, 3, 4]> : tensor<4xi32>) : !llvm.array<4 x i32>

// access element index 2 (== 3)
%element = llvm.extractvalue %arr[2] : !llvm.array<4 x i32>

// "update" element 0 — produces a NEW array value (SSA), nothing mutated in place
%new_arr = llvm.insertvalue %element, %arr[0] : !llvm.array<4 x i32>
```

Structs and arrays nest just like in C, and a **nested index path** reaches a
field inside a struct inside an array — `llvm.insertvalue %k, %arr[0, 0]` writes
element 0, field 0.

This pair of types is precisely the raw material a memref descriptor is *built out
of*: recall from Step 2 that the descriptor is itself an
`!llvm.struct<(ptr, ptr, i64, array<N x i64>, array<N x i64>)>` — a struct whose
last two fields are arrays. Inspect the lowered IR with:

```bash
mlir-opt structs_arrays.mlir -reconcile-unrealized-casts
```

Since this file is already entirely in the `llvm` dialect, you can also translate
it straight to textual LLVM IR and watch `!llvm.struct<(i32, f32)>` become the
C-style `{ i32, float }`:

```bash
mlir-translate structs_arrays.mlir --mlir-to-llvmir
```

```llvm
define { i32, float } @make_pair() {
  ret { i32, float } { i32 42, float 0x40091EB860000000 }
}
```

### Vectors — SIMD without intrinsics (`vectors.mlir` 🧩)

A `vector<4xf32>` is MLIR's SIMD type: a bundle of lanes operated on by one
instruction. The point of the `vector` dialect is **portability** — you write the
operation once and let LLVM map it to whatever the target has: a single **AVX**
instruction on x86, **two NEON** ops on ARM, or a **scalar loop** on hardware
with no SIMD at all. The same `.mlir` runs everywhere; it's just faster where the
silicon allows.

Elementwise math is plain `arith.*` applied to a vector operand type — there is no
special `vector.add`. The lane-level manipulation ops live in the `vector` dialect:

| Op | What it does |
| --- | --- |
| `arith.addf` / `arith.mulf` | Elementwise add / multiply across all lanes |
| `vector.extract` | Pull one lane out as a scalar |
| `vector.shuffle` | Permute lanes from two vectors |
| `vector.reduction` | Horizontally reduce a vector to a scalar |
| `vector.broadcast` | Splat a scalar across every lane |

```mlir
%c   = arith.addf %a, %b : vector<4xf32>                          // elementwise
%lane = vector.extract %c[0] : f32 from vector<4xf32>             // one lane out
%p   = vector.shuffle %a, %b [0, 1, 4, 5] : vector<4xf32>, vector<4xf32>  // a0,a1,b0,b1
%sum = vector.reduction <add>, %c : vector<4xf32> into f32        // horizontal sum
%spl = vector.broadcast %x : f32 to vector<4xf32>                 // splat
```

```bash
mlir-opt vectors.mlir -convert-vector-to-llvm -convert-arith-to-llvm \
  -convert-func-to-llvm -reconcile-unrealized-casts
```

### Bufferization — the tensor→memref bridge, dissected (`bufferization.mlir` 🧩)

Bufferization is the crossing from the **value world** (tensors) to the
**address world** (memrefs) — the same step Step 1 relied on, now shown three
ways:

1. **Manual primitives** — `bufferization.to_memref` (tensor → memref) and
   `bufferization.to_tensor` (memref → read-only tensor view). The `restrict`
   keyword promises the analyzer no other tensor aliases the buffer.
2. **Automatic, program-wide** — the `-one-shot-bufferize` pass. With
   `bufferize-function-boundaries`, even the function *signature* is converted, so
   `tensor<8xf32>` in the arguments/return becomes `memref<8xf32>`.
3. **In-place destination** — `bufferization.materialize_in_destination`, the
   typical *final* step of a kernel that writes into a buffer the caller already
   owns (exactly what `identity_fill.mlir` did in Step 1).

```mlir
%r = tensor.insert %f into %arg0[%idx] : tensor<8xf32>   // before: value semantics
// after -one-shot-bufferize="bufferize-function-boundaries":
memref.store %f, %arg0[%idx] : memref<8xf32>             // in place, on the same buffer
```

```bash
mlir-opt bufferization.mlir -canonicalize
mlir-opt bufferization.mlir -one-shot-bufferize="bufferize-function-boundaries"
```

The payoff to watch for in the one-shot output: a `tensor.insert` (which
*logically* produces a brand-new tensor) becomes a `memref.store` that **reuses
the input buffer** — no copy, no allocation. The math-level immutability costs
nothing at runtime.

### The C interface, previewed (`c_interface.mlir` 🧩)

Without `llvm.emit_c_interface`, lowering a function that takes `memref<3xf32>`
unrolls the descriptor into **five scalar arguments**
(`allocated, aligned, offset, sizes[], strides[]`) — awkward to call from C.
Adding the attribute keeps that unrolled function *and* emits a wrapper
`_mlir_ciface_<name>` that takes descriptor *pointers* instead. Lowering
`c_interface.mlir` shows both `@identity` and `@_mlir_ciface_identity`:

```bash
mlir-opt c_interface.mlir -finalize-memref-to-llvm \
  -convert-func-to-llvm -reconcile-unrealized-casts
```

This is the mechanism **Step 4 dissects line by line**, so we only flag it here.

### Runnable — vectorized `array_add` (`array_add_vec.mlir` ✅)

The one runnable file is Step 2's kernel, vectorized: the loop steps by 8 and uses
`vector.load`/`vector.store` with a `vector<8xf32>` add, so each iteration
processes 8 floats. On x86 this lowers to a single AVX `vaddps`; on ARM, two NEON
adds.

*3_low_level_llvm/array_add_vec.mlir*
```mlir
module {
  func.func @array_add(%arg0: memref<1024xf32>,
                       %arg1: memref<1024xf32>,
                       %arg2: memref<1024xf32>)
      attributes {llvm.emit_c_interface} {
    %c0    = arith.constant 0    : index
    %c1024 = arith.constant 1024 : index
    %c8    = arith.constant 8    : index

    scf.for %i = %c0 to %c1024 step %c8 {
      %va = vector.load %arg0[%i] : memref<1024xf32>, vector<8xf32>
      %vb = vector.load %arg1[%i] : memref<1024xf32>, vector<8xf32>
      %vc = arith.addf %va, %vb : vector<8xf32>
      vector.store %vc, %arg2[%i] : memref<1024xf32>, vector<8xf32>
    }
    return
  }
}
```

`attributes {llvm.emit_c_interface}` makes the `.dylib` export
`_mlir_ciface_array_add`, which `aot_main.py` calls via `ctypes` (using the shared
`common/np_memref.py` descriptor). `build_array_add_vec.sh` runs the now-familiar
pipeline (`mlir-opt` → `mlir-translate` → `llc` → `clang -shared`) then the driver.

This directory has two build scripts:

- **`build.sh`** — builds *every* example into `build/`: it lowers each concept
  file with `mlir-opt` (and `mlir-translate` where applicable) so you can read the
  resulting IR, then delegates the runnable kernel to `build_array_add_vec.sh`.
- **`build_array_add_vec.sh`** — builds and runs *only* the SIMD kernel.

**Run everything:** `cd 3_low_level_llvm && bash build.sh`  ·  **Just the demo:**
`bash build_array_add_vec.sh`  ·  **Clean:** `bash clean.sh` (removes `build/` and
`__pycache__`).

```
Array addition successful!
First few elements: [3. 3. 3. 3. 3.]
```

(`a = ones(1024)`, `b = 2*ones(1024)` ⇒ `c = 3` everywhere.)

---

## Step 4 — C-compatible wrappers (`4_C_compatible_wrappers/`) · ✅ runnable

**Goal:** dissect what `llvm.emit_c_interface` *generates* and **why it has to
exist** — the ABI mismatch between how MLIR passes memrefs and how C expects to.

**The problem it solves.** When MLIR lowers a memref function, it doesn't pass the
descriptor as one struct — it **unrolls** every memref into its five scalar fields
as *separate* arguments (`ptr, ptr, i64, i64, i64` for a 1-D memref). That's
efficient for MLIR-to-MLIR calls, but it's a nightmare to call from C or `ctypes`,
where you'd have to hand-pack five positional arguments per array and somehow
receive an unrolled struct back. C wants to pass *one pointer to a struct* per
array. These two calling conventions are simply incompatible.

`llvm.emit_c_interface` bridges them by emitting **two** functions: the original
(`@…`) keeps the fast unrolled signature, and a generated wrapper
(`@_mlir_ciface_…`) exposes the C-friendly one — it takes *descriptor pointers*,
`llvm.load`s the structs, `extractvalue`s the five fields back out, calls the real
function, and `llvm.store`s the result struct back through an out-pointer.

### The kernel

The kernel `add_vector_to_matrix` is deliberately the **identity** on
`memref<3xf32>` — the lesson is the calling convention, not the math:

*4_C_compatible_wrappers/add_vector_to_matrix.mlir* (the function)
```mlir
module {
  func.func @add_vector_to_matrix(%A: memref<3xf32>)
    -> memref<3xf32> attributes {llvm.emit_c_interface} {
    return %A : memref<3xf32>   // identity: hand the input buffer straight back
  }
}
```

### What `emit_c_interface` generates

Lowering produces the two functions below. The inner `@add_vector_to_matrix` takes
the memref **unrolled** into five scalars and re-packs them into a struct; the
wrapper `@_mlir_ciface_add_vector_to_matrix` is what C actually calls. Note its two
arguments: `%arg0` is the **output** (a caller-allocated descriptor the wrapper
writes the returned memref into) and `%arg1` is the **input** descriptor pointer.

```mlir
llvm.func @add_vector_to_matrix(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64,
    %arg3: i64, %arg4: i64) -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    attributes {llvm.emit_c_interface} {
  %0 = llvm.mlir.undef  : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  %1 = llvm.insertvalue %arg0, %0[0] : ...   // allocated ptr
  %2 = llvm.insertvalue %arg1, %1[1] : ...   // aligned ptr
  %3 = llvm.insertvalue %arg2, %2[2] : ...   // offset
  %4 = llvm.insertvalue %arg3, %3[3, 0] : ... // shape[0]
  %5 = llvm.insertvalue %arg4, %4[4, 0] : ... // stride[0]
  llvm.return %5 : !llvm.struct<(...)>
}

llvm.func @_mlir_ciface_add_vector_to_matrix(%arg0: !llvm.ptr, %arg1: !llvm.ptr)
    attributes {llvm.emit_c_interface} {
  %0 = llvm.load %arg1 : !llvm.ptr -> !llvm.struct<(...)>   // load the INPUT struct
  %1 = llvm.extractvalue %0[0] : ...   // unpack its five fields ...
  %2 = llvm.extractvalue %0[1] : ...
  %3 = llvm.extractvalue %0[2] : ...
  %4 = llvm.extractvalue %0[3, 0] : ...
  %5 = llvm.extractvalue %0[4, 0] : ...
  %6 = llvm.call @add_vector_to_matrix(%1, %2, %3, %4, %5) : (...) -> !llvm.struct<(...)>
  llvm.store %6, %arg0 : !llvm.struct<(...)>, !llvm.ptr     // write result via OUT ptr
  llvm.return
}
```

So from the caller's side you hand it two pointers to `MemRefDescriptor` structs
and let the wrapper do all the marshalling.

### The descriptor on the Python side

The struct the wrapper expects must be mirrored on the caller's side. That's what
`common/np_memref.py` is (Step 2) — a `ctypes.Structure` that matches the LLVM
struct `-finalize-memref-to-llvm` produces, field for field. Three details bite if
you get them wrong:

- **`allocated` vs `aligned`.** MLIR separates the pointer the runtime would
  `free` from the pointer used for accesses (an aligned allocator may hand back a
  different address). NumPy gives us one buffer pointer with no separate
  free-handle, so we set both to the same value — safe here because nothing
  `free`s the descriptor.
- **`offset` and `stride` are in elements, not bytes.** A contiguous 1-D float
  array has `stride[0] == 1`, not `4`.
- **The struct's shape depends on the memref rank.** A 2-D memref needs
  `shape[2]`/`stride[2]`. If the MLIR signature's rank changes, the ctypes struct
  must change with it, or the wrapper reads garbage past the end of your struct.

### Reading the result back

Because the kernel **returns** a memref, the wrapper writes the returned
descriptor into the result pointer you pass. The data you want is whatever that
descriptor's `aligned` points at — *not* some buffer you prepared:

```python
result_desc = MemRefDescriptor()                 # uninitialized; filled by the call
fn(ctypes.byref(result_desc), ctypes.byref(a_desc))

size    = result_desc.shape[0]
out_ptr = ctypes.cast(result_desc.aligned, ctypes.POINTER(ctypes.c_float))
out     = np.ctypeslib.as_array(out_ptr, shape=(size,)).copy()
```

For this identity kernel, `result_desc.aligned` ends up equal to `a_desc.aligned`
— the kernel literally returned its input. (A kernel that instead *fills a
caller-provided buffer* would have no return value, and you'd read the NumPy array
you allocated — the `fill` convention from Step 1.)

### Build-pipeline gotchas

`build.sh` runs `mlir-opt` → `mlir-translate` → `llc` → `clang -shared`. Two
non-obvious points:

- `--finalize-memref-to-llvm` is what unrolls the descriptor into its five scalar
  fields, and it must run **before** `--reconcile-unrealized-casts`, because the
  unrolling emits `unrealized_conversion_cast` ops the latter pass then erases.
- `--relocation-model=pic` / `-fPIC` are required because the `.so` is loaded by
  `dlopen` (via `ctypes.CDLL`); without them the linker rejects the object.

On macOS, `nm -gU build/*.so | grep mlir` shows the wrapper with an **extra leading
underscore** (`__mlir_ciface_add_vector_to_matrix`) — that's just the Mach-O ABI
convention and doesn't change how `ctypes` looks the symbol up.

**Run:** `cd 4_C_compatible_wrappers && bash build.sh`

**Output:**

```
add_vector_to_matrix call successful!
input : [1. 2. 3.]
output: [1. 2. 3.]
```

---

## Run everything

```bash
export PATH="/opt/homebrew/opt/llvm@20/bin:$PATH"   # llvm@20 is keg-only
# Python deps for the drivers:  pip install numpy llvmlite

for d in 1_tensor 2_array_add 3_low_level_llvm 4_C_compatible_wrappers; do
  echo "=== $d ==="; ( cd "$d" && bash build.sh ) 2>&1 | tail -n 4
done
```

Every step writes its intermediates (`*_opt.mlir`, `*.ll`, `*.o`, `*.so`/
`.dylib`) into a local `build/` directory; `bash 3_low_level_llvm/clean.sh`
shows the pattern for removing them.

## Key takeaways

- **The whole chapter is one question: does this value have an address yet?** A
  tensor is a number that lives nowhere (a value); a memref is one that's been
  given a home (an address). Crossing from the first to the second is
  *bufferization* — the data-world counterpart of Chapter 1's lowering.
- **Tensor → MemRef → LLVM** is a ladder of decreasing abstraction; you descend
  it with lowering passes, and **bufferization** is the tensor→memref rung.
- A **MemRef is just a struct** (ptr, ptr, offset, shape, strides). Once you
  know that, sharing data with NumPy is free — point the struct at NumPy's
  buffer (`common/np_memref.py`).
- `llvm.emit_c_interface` is what makes any of this callable from C/Python.
- The `vector` dialect is how you get SIMD without writing intrinsics.

**Next:** Part 3 — the Affine dialect and OpenMP (see [`../reference/`](../reference/)).
