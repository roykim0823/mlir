# 1 — Introduction: From LLVM IR to MLIR

### A compiler is a staircase, not a black box

Most people picture a compiler as a single magic step: source code goes in, a
mysterious box hums, an executable falls out. That picture is wrong in a way that
matters. A modern compiler is really a **staircase of small, meaning-preserving
translations**. At the top you have a representation that's comfortable for
*humans* — loops, functions, types, tensors. At the bottom you have one that's
comfortable for *silicon* — registers, branches, fixed-width integers, memory
addresses. Each step down throws away a little abstraction in exchange for a
little concreteness, and crucially, **every step preserves what the program
means**. Compilation is the disciplined act of walking down that staircase
without ever changing the answer.

The deep idea behind MLIR is hidden in plain sight in its name: **Multi-Level**
Intermediate Representation. Older compilers had essentially *one* landing on the
staircase — LLVM has LLVM IR, and everything had to be expressed there, no matter
how high-level it really was. Want to represent a matrix multiply, a parallel
loop, or a GPU kernel? You had to shred it down into low-level IR immediately and
hope the optimizer could reconstruct your intent. MLIR's bet is that the
staircase should have **as many landings as you need** — each one a *dialect*
pitched at exactly the right altitude for the thing you're describing — and that
lowering should be a series of short, well-understood hops between adjacent
landings rather than one terrifying leap.

### Why this chapter does the same thing twice

The fastest way to feel that idea is to watch the *same trivial program* descend
the staircase from two different starting heights. So this chapter walks the
**complete compilation pipeline twice**, on two tiny programs, narrating exactly
what each tool does and showing you what it emits at every stage:

- [`1_llvm_modules/`](1_llvm_modules/) — a minimal program written in **raw LLVM
  IR**. We start a few steps from the bottom of the staircase and just walk the
  rest of the way down: assemble it, link it into a shared library, and call it
  from Python. This establishes the destination — *native code you can actually
  run* — and the tools (`llc`, `clang`, `ctypes`) that get you there.
- [`2_mlir/`](2_mlir/) — the *same kind of program* (now with an actual loop),
  but written in **high-level MLIR** using the `scf` ("structured control flow")
  dialect. We start much higher up and let `mlir-opt` mechanically lower it,
  one pass per step, down to the `llvm` dialect — and then re-join the exact same
  back-end path from Part A. You'll literally watch a clean `scf.for` loop
  decompose into basic blocks, then into phi nodes, then into ARM64 registers.

Put side by side, the two halves answer the question that motivates the entire
series: ***what does MLIR actually buy us over plain LLVM IR?*** The short answer
— *the ability to start higher up the staircase and have the descent handled for
us* — only becomes convincing once you've seen both climbs with your own eyes.
Everything in the later chapters (`memref`, `linalg`, `tensor`, `gpu`) is just
**adding taller landings at the top of this same staircase**; the bottom steps
never change, which is exactly why we nail them down first, here.

> All commands assume `llvm@20` is on your `PATH`:
> `export PATH="/opt/homebrew/opt/llvm@20/bin:$PATH"`. Outputs shown below were
> produced with **Homebrew LLVM 20.1.8** on Apple Silicon — your hex addresses
> and exact assembly may differ.

---

## Part A — `1_llvm_modules/`: raw LLVM IR

**Why start here?** Before reaching for MLIR it's worth seeing the layer
*underneath* it. MLIR's whole job is to eventually produce LLVM IR, so if you
understand how a tiny piece of LLVM IR becomes a callable native function, the
MLIR pipeline in Part B is just the same path with extra lowering stages bolted
on the front. We also establish the end goal up front: **a shared library Python
can call**, which is the bridge we use to drive compiled code from the ML
ecosystem.

> **What's a "module"?** Every LLVM IR file is a *module* — the top-level
> container holding functions, globals, and metadata, roughly the IR equivalent
> of one translation unit (one `.c` file). MLIR borrows the same idea: its
> top-level container is also a `module`. Compilation, at every level, is a
> series of transformations from one module to another.

### The source: [`simple.ll`](1_llvm_modules/simple.ll)

*1_llvm_modules/simple.ll*
```llvm
define i32 @main() {
	ret i32 42
}
```

This is hand-written **LLVM IR**. It is equivalent to the C program:

```c
int main() {
  return 42;
}
```

`define i32 @main()` declares a function named `main` returning a 32-bit
integer; `ret i32 42` returns the constant `42`.

A few conventions worth naming, because they recur in MLIR too:

- **`@` is a symbol (global) name.** `@main` is a globally visible function
  symbol — the same name the linker and `ctypes` look up later. (Local SSA values
  use `%` instead; we'll see those in Part B.)
- **Types are explicit everywhere.** `i32` is an arbitrary-width integer type —
  LLVM has `i1`, `i8`, `i64`, etc. There is no implicit `int`; the width is
  always written out, which is what lets the same IR target many architectures.
- **It's already SSA.** Even this trivial function is in *Static Single
  Assignment* form (every value is assigned exactly once). That constraint is
  invisible here but becomes the central plot point in Part B.

### The build: [`build.sh`](1_llvm_modules/build.sh)

```bash
mkdir -p build
llc -filetype=obj --relocation-model=pic simple.ll -o ./build/simple.o   # 1
clang -shared -fPIC ./build/simple.o -o ./build/libsimple.so             # 2
clang ./build/simple.o -o ./build/simple   # optional executable          # 3
./build/simple; echo $?                                                  # 4
python3 simple.py                                                        # 5
```

**Step 1 — `llc`: LLVM IR → object code.** `llc` is LLVM's static compiler — the
back end that does instruction selection and register allocation and emits a
native object file (`.o`) full of machine code. `--relocation-model=pic` emits
*position-independent code*: code that works no matter what address it's loaded
at. That's required for a shared library, because the OS may map it anywhere in a
process's address space.

**Step 2 — `clang -shared`: object → shared library.** A bare `.o` isn't directly
loadable; the linker has to package it. This step links the object into
`libsimple.so` (a `.dylib`-style shared object on macOS) — a self-contained,
dynamically loadable unit that other programs, including the Python interpreter,
can `dlopen` at runtime and call into. This is the artifact that makes Step 5
possible.

**Step 3 — `clang`: object → executable.** Optionally links the same object into
a standalone runnable binary `simple`.

**Step 4 — run it.** The program does nothing but return `42`, which becomes the
process **exit code**. `echo $?` prints the exit code of the last command:

```
42
```

**Step 5 — call it from Python.** [`simple.py`](1_llvm_modules/simple.py) uses
the standard-library `ctypes` module to load the shared library and call
`main()` directly:

*1_llvm_modules/simple.py*
```python
import ctypes

# Load the shared library (the .so we just built) into the process.
module = ctypes.CDLL("./build/libsimple.so")

# Declare the C signature of `main` so ctypes marshals values correctly:
module.main.argtypes = []            # main takes no arguments
module.main.restype = ctypes.c_int   # main returns a C int (i32)

# Call the native function and print its return value.
print(module.main())
```

Line by line:

- `ctypes.CDLL(path)` `dlopen`s the shared library and returns a handle whose
  attributes are its exported symbols (`module.main` is the C `main`).
- `argtypes` / `restype` tell ctypes how to convert Python values to/from C.
  Without `restype`, ctypes assumes the function returns a C `int` — correct
  here, but setting it explicitly is good practice and essential once return
  types aren't `int`.
- `module.main()` actually invokes the compiled native code and returns `42` as
  a Python `int`.

Output:

```
42
```

> Unlike Step 4, the value `42` here is a **return value** we print, not a
> process exit code.

This is the punchline of Part A: **anything you compile through LLVM becomes a
plain shared library you can call from Python** — the bridge into the ML
ecosystem we'll rely on throughout the series.

---

## Part B — `2_mlir/`: high-level MLIR

Returning `42` is trivial in LLVM IR. But real programs have loops, and writing
a loop in raw LLVM IR means manually managing basic blocks and **phi nodes**
(the SSA construct that merges values across control-flow paths). MLIR lets us
write the loop at a high level and *lower* it for us.

> **Why loops are awkward in SSA.** SSA says every value is assigned exactly
> once — but a loop counter is, by definition, a thing that *changes*. LLVM
> reconciles this with a **phi node**: a pseudo-instruction at the top of a block
> that says "my value is whichever incoming value corresponds to the edge we
> arrived on" (e.g. `0` on the first entry, `previous + 1` on each loop back).
> It's correct but fiddly bookkeeping, and you must write it by hand in raw LLVM
> IR. The motivation for Part B is watching MLIR generate all of it for us from a
> plain `scf.for`.

### The source: [`example.mlir`](2_mlir/example.mlir)

*2_mlir/example.mlir*
```mlir
// loop_add: sum the integers 0..9 using a structured (scf) for-loop.
// Returns `index` (a platform-sized integer, like size_t).
func.func @loop_add() -> (index) {
  // `index` dialect: loop bounds and the running total are index-typed.
  %init = index.constant 0   // initial accumulator value
  %lb = index.constant 0     // loop lower bound (inclusive)
  %ub = index.constant 10    // loop upper bound (exclusive)
  %step = index.constant 1   // loop step

  // scf.for is a *structured* loop. `iter_args` threads a loop-carried value
  // (%acc) through each iteration; whatever we scf.yield becomes %acc next
  // time, and the final value is bound to %sum.
  %sum = scf.for %iv = %lb to %ub step %step iter_args(%acc = %init) -> (index) {
    %sum_next = arith.addi %acc, %iv : index   // arith dialect: acc + iv
    scf.yield %sum_next : index                // carry sum_next into next iter
  }
  return %sum : index
}

// main: call loop_add and return the result as a C-style i32 exit code.
func.func @main() -> i32 {
  %out = call @loop_add() : () -> index
  // index is i64 here; narrow it to i32 so main can return a normal exit code.
  %out_i32 = arith.index_cast %out : index to i32
  return %out_i32 : i32
}
```

In C this is:

```c
int loop_add() {
  int sum = 0;
  for (int iv = 0; iv < 10; iv += 1)
    sum = sum + iv;
  return sum;          // 0+1+2+...+9 = 45
}
int main() { return loop_add(); }
```

The single file touches **four dialects** — `func` (functions & `call`), `index`
(loop bounds / induction var), `scf` (the structured `scf.for` loop), and `arith`
(`addi`, `index_cast`). See the comments above for what each line does.

This mixing is the point of MLIR, not an accident: dialects are designed to
**coexist in one module**, each modelling the part of the program it's best
suited to. There's no separate "control-flow IR" and "arithmetic IR" file — the
loop (`scf`), the math inside it (`arith`), and the addressing types (`index`)
all live together and get lowered independently. Two type details are worth
calling out:

- **`index` is platform-sized.** Like C's `size_t`, an `index` is whatever the
  target's natural pointer-width integer is (64-bit on Apple Silicon). It's the
  right type for loop counters and array offsets because it always matches the
  machine. That's also why `main` needs `arith.index_cast` to narrow it to a
  fixed-width `i32` before returning a normal exit code.
- **`scf.for` *returns* a value.** Unlike a C `for` loop (a statement that
  mutates a variable), `scf.for` with `iter_args` is an expression that yields
  its final loop-carried value — a more functional framing that keeps the IR in
  SSA form without any phi nodes at this level.

> **Result preview:** the loop sums `0..9`, so the program returns **45**, not
> 42. (Different from Part A on purpose.)

### The build: [`build.sh`](2_mlir/build.sh)

The script runs the full pipeline. We'll go through it stage by stage.

#### Step 1 — `mlir-opt`: lower high-level dialects to the `llvm` dialect

```bash
mlir-opt example.mlir \
  --convert-func-to-llvm \
  --convert-math-to-llvm \
  --convert-index-to-llvm \
  --convert-scf-to-cf \
  --convert-cf-to-llvm \
  --convert-arith-to-llvm \
  --reconcile-unrealized-casts \
  -o ./build/example_opt.mlir
```

**What "lowering" actually means.** Lowering is translating an operation from a
higher-level dialect into one or more operations from a lower-level dialect that
have the same runtime behaviour but less abstraction. You never lower the whole
program in one leap; you apply a *sequence* of small, focused passes, each
responsible for one dialect, until everything has bottomed out in the `llvm`
dialect. That incremental, pass-by-pass design is exactly what makes the
machinery reusable — every dialect only has to know how to take one step down.

Each `--convert-X-to-Y` flag is one such **pass**, rewriting one dialect into a
lower one. The order matters because passes form a *dependency chain*:
`convert-scf-to-cf` turns the structured loop into unstructured branches (`cf`
dialect), and **only then** can `convert-cf-to-llvm` turn those branches into the
`llvm` dialect — run them in the wrong order and the second pass sees ops it
doesn't recognise and does nothing. While a conversion is in progress, the module
can be temporarily "mixed" (some ops converted, some not); MLIR bridges the two
worlds with `unrealized_conversion_cast` placeholders, and
`reconcile-unrealized-casts` removes them once every dialect has been lowered and
the casts cancel out.

The result, `build/example_opt.mlir`, is entirely in the `llvm` dialect — note
how the `scf.for` loop has become explicit basic blocks and branches:

*build/example_opt.mlir*
```mlir
module {
  llvm.func @loop_add() -> i64 {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(0 : i64) : i64
    %2 = llvm.mlir.constant(10 : i64) : i64
    %3 = llvm.mlir.constant(1 : i64) : i64
    llvm.br ^bb1(%1, %0 : i64, i64)
  ^bb1(%4: i64, %5: i64):  // 2 preds: ^bb0, ^bb2
    %6 = llvm.icmp "slt" %4, %2 : i64
    llvm.cond_br %6, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %7 = llvm.add %5, %4 : i64
    %8 = llvm.add %4, %3 : i64
    llvm.br ^bb1(%8, %7 : i64, i64)
  ^bb3:  // pred: ^bb1
    llvm.return %5 : i64
  }
  llvm.func @main() -> i32 {
    %0 = llvm.call @loop_add() : () -> i64
    %1 = llvm.trunc %0 : i64 to i32
    llvm.return %1 : i32
  }
}
```

The loop's carried value (`iter_args`) became the **block arguments** of `^bb1`
(`%4` = induction var, `%5` = accumulator) — MLIR's alternative to phi nodes:
`%6 = llvm.icmp "slt" %4, %2` is the `i < 10` test, `%7 = llvm.add %5, %4` is
`sum += iv`, and `%8 = llvm.add %4, %3` is `iv += 1`.

#### Step 2 — `mlir-translate`: `llvm` dialect → LLVM IR

```bash
mlir-translate ./build/example_opt.mlir -mlir-to-llvmir -o ./build/example.ll
```

This crosses out of MLIR into textual **LLVM IR**.

> **The `llvm` dialect is not LLVM IR.** This catches everyone at first. After
> Step 1 the module is in the *`llvm` dialect* — still MLIR data structures,
> still parsed and printed by MLIR tools, just using ops that mirror LLVM IR
> one-to-one. It is *not yet* LLVM IR. `mlir-translate` is the bridge that
> finally leaves the MLIR universe and emits actual LLVM IR text (`.ll`) that the
> rest of the LLVM toolchain (`llc`, `opt`, `clang`) understands. The split
> exists so MLIR can keep a faithful in-memory model of LLVM IR — and run its own
> passes on it — without depending on LLVM's own data structures until the very
> last moment.

Notice that here the loop is expressed with real **phi nodes** (`%2`, `%3`) —
exactly the bookkeeping MLIR saved us from writing by hand. This is the same loop
as the `^bb1(%4, %5)` block arguments from Step 1: `mlir-translate` converts
MLIR's block-argument form into LLVM's phi-node form, since LLVM IR has no notion
of block arguments. The two `phi i64` nodes (`%2` the induction variable, `%3`
the accumulator) are that bookkeeping made explicit:

*build/example.ll*
```llvm
; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

define i64 @loop_add() {
  br label %1

1:                                                ; preds = %5, %0
  %2 = phi i64 [ %7, %5 ], [ 0, %0 ]
  %3 = phi i64 [ %6, %5 ], [ 0, %0 ]
  %4 = icmp slt i64 %2, 10
  br i1 %4, label %5, label %8

5:                                                ; preds = %1
  %6 = add i64 %3, %2
  %7 = add i64 %2, 1
  br label %1

8:                                                ; preds = %1
  ret i64 %3
}

define i32 @main() {
  %1 = call i64 @loop_add()
  %2 = trunc i64 %1 to i32
  ret i32 %2
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
```

#### Step 3 — `llc` + `clang`: LLVM IR → object → shared library + executable

```bash
llc -filetype=obj --relocation-model=pic ./build/example.ll -o ./build/example.o
clang -shared -fPIC ./build/example.o -o ./build/libexample.so   # shared library
clang ./build/example.o -o ./build/example                       # executable
./build/example; echo $?
```

Exactly the same final stages as Part A. `llc` produces the native object, then
`clang` links it two ways:

- `-shared` → `libexample.so`, loadable from Python with `ctypes` just like
  `simple.py` (`ctypes.CDLL("./build/libexample.so").main()` returns `45`).
- no `-shared` → a standalone executable `example`. Running it makes `main`'s
  return value the **process exit code**:

```
45
```

This is the binary/executable equivalent of Part A's `simple` — the high-level
MLIR loop has become an ordinary native program.

#### Step 4 — inspect the native assembly

```bash
llc -filetype=asm --relocation-model=pic ./build/example.ll -o ./build/example.s
```

Inspecting the assembly is the final sanity check that the whole chain produced
*good* code, not just *correct* code — it's where you confirm the optimizer did
its job. On Apple Silicon this emits ARM64. Notice the payoff of running through
the real LLVM back end: the loop bound `10` from our MLIR has become the
immediate `#9` in the `cmp`, the induction variable and accumulator live entirely
in registers (`x8`, `x0`) with no memory traffic, and the body is a tight inner
loop with a single conditional branch:

*build/example.s* (excerpt — the `loop_add` core; `.cfi`/`%bb.0` directives elided, comments added)
```asm
_loop_add:                              ; @loop_add
	mov	x8, xzr            ; iv  = 0
	mov	x0, xzr            ; sum = 0
	cmp	x8, #9
	b.gt	LBB0_2
LBB0_1:                                 ; inner loop, depth 1
	add	x0, x0, x8         ; sum += iv
	add	x8, x8, #1         ; iv  += 1
	cmp	x8, #9
	b.le	LBB0_1
LBB0_2:
	ret                        ; return sum (in x0)
```

The `build.sh` also disassembles the object/dylib with `objdump` if you want to
see resolved addresses:

```bash
objdump -d --no-show-raw-insn ./build/example.o    > ./build/example.dis
objdump -d --no-show-raw-insn ./build/libexample.so > ./build/libexample.dis
```

#### Step 5 — (alternative) `mlir-runner`: JIT-execute directly

There are two ways to actually *run* compiled code, and it's worth having the
names straight because later chapters use both:

- **AOT (ahead-of-time)** — Steps 2–4: produce a `.so`/executable on disk now,
  run it later. This is what you ship.
- **JIT (just-in-time)** — this step: compile to machine code *in memory* at the
  moment you run, execute immediately, produce no files. This is what you reach
  for while iterating.

Everything above goes through full AOT code generation to a native binary. For
quick iteration you can instead **skip codegen entirely** and JIT-run the lowered
MLIR:

```bash
mlir-runner -e main -entry-point-result=i32 ./build/example_opt.mlir
```

`mlir-runner` JIT-compiles and runs the module in-process without producing any
files. `-e main` names the entry function and `-entry-point-result=i32` tells it
the return type. Output:

```
45
```

(The script also shows the variant with
`-shared-libs=/opt/homebrew/opt/llvm@20/lib/libmlir_runner_utils.dylib`, needed
only when your MLIR calls runtime helpers like `printMemref` — not required
here.)

This is a shortcut for *running* the code, not part of the build-to-native path:
it consumes `example_opt.mlir` (the output of Step 1), parallel to Steps 2–4
rather than after them.

---

## The big picture

The two halves are the *same staircase* entered at different heights — Part A
starts near the bottom, Part B a few landings up and descends through them:

```text
   Part B (2_mlir)                          Part A (1_llvm_modules)
   high-level MLIR    example.mlir  ─┐
     scf / index / arith / func      │  mlir-opt (lowering passes)
   llvm dialect       example_opt.mlir│
     │  mlir-translate                ▼
   LLVM IR            example.ll  ◄────────  simple.ll   ← Part A starts here
     │  llc                                   │  llc
   object (.o)        example.o               simple.o
     │  clang                                 │  clang
   native .so / executable  ◄────────────────┘   → callable from Python (ctypes)
```

Both descents share the bottom steps (`llc` → `clang`); MLIR just adds taller
landings at the top and walks them down for you.

| | `1_llvm_modules` | `2_mlir` |
| --- | --- | --- |
| You write | raw LLVM IR | high-level MLIR (`scf.for`) |
| Loops / phi nodes | manual | generated by lowering |
| Lowering tool | — (already LLVM IR) | `mlir-opt` passes |
| To LLVM IR | (it *is* LLVM IR) | `mlir-translate` |
| To native (object) | `llc` | `llc` |
| Shared library + executable | `clang` | `clang` |
| JIT option | — | `mlir-runner` |

Both programs end up as ordinary native code — a shared library *and* a
standalone executable — but MLIR let us express the loop at a human level and
mechanically lower it, with the phi-node bookkeeping handled for us. That
lowering machinery is the whole point of MLIR, and every later chapter adds
*higher* dialects (`memref`, `linalg`, `tensor`, `gpu`) on top of this same
pipeline.

---

## Run it yourself

```bash
export PATH="/opt/homebrew/opt/llvm@20/bin:$PATH"

cd 1_llvm_modules && bash build.sh && cd ..   # exit code 42, then 42 (from Python)
cd 2_mlir         && bash build.sh && cd ..   # exit code 45 (executable), then 45 (JIT)
```

Inspect the generated files under each `build/` directory:
`example_opt.mlir` (lowered MLIR), `example.ll` (LLVM IR), `example` (executable),
`libexample.so` (shared library), `example.s` (assembly), and `*.dis`
(disassembly).

**Next:** [`../2_memory/`](../2_memory/) — how MLIR represents memory with the
`tensor` and `memref` dialects.
