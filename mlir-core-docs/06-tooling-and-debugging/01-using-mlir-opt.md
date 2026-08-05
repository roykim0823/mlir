# Using `mlir-opt`

> **Section:** Tooling and Debugging · document 1 of 7  
> **Upstream:** [https://mlir.llvm.org/docs/Tutorials/MlirOpt/](https://mlir.llvm.org/docs/Tutorials/MlirOpt/) · source [`mlir/docs/Tutorials/MlirOpt.md`](https://github.com/llvm/llvm-project/blob/main/mlir/docs/Tutorials/MlirOpt.md)  
> **License:** upstream text is Apache-2.0 WITH LLVM-exception.

## Orientation

Filed upstream as a tutorial and placed here because it is the practical entry point to all MLIR
tooling. `mlir-opt` reads `.mlir`, runs a pass pipeline, writes `.mlir`. Every upstream test is an
invocation of it, every dialect gets its own equivalent binary, and it is where you will do almost
all debugging.

Read it early — earlier than its position here suggests if you are hands-on. Fluency with the
pipeline syntax and the printing flags is the difference between debugging by inspection and
debugging by guesswork.

**Read first**

- [Pass Infrastructure](../03-passes-and-rewriting/01-pass-infrastructure.md)

**What you should be able to do after this page**

- Compose and run a nested pass pipeline from the command line.
- Inspect the IR between passes and isolate the pass that broke it.
- Write and run a `lit`/`FileCheck` test.

---

## Upstream documentation

`mlir-opt` is a command-line entry point for running passes and lowerings on MLIR code.
This tutorial will explain how to use `mlir-opt`, show some examples of its usage,
and mention some useful tips for working with it.

Prerequisites:

- [Building MLIR from source](https://github.com/llvm/llvm-project/blob/main//getting_started)
- [MLIR Language Reference](https://github.com/llvm/llvm-project/blob/main//docs/LangRef)

## Table of contents


## `mlir-opt` basics

The `mlir-opt` tool loads a textual IR or bytecode into an in-memory structure,
and optionally executes a sequence of passes
before serializing back the IR (textual form by default).
It is intended as a testing and debugging utility.

After building the MLIR project,
the `mlir-opt` binary (located in `build/bin`)
is the entry point for running passes and lowerings,
as well as emitting debug and diagnostic data.

Running `mlir-opt` with no flags will consume textual or bytecode IR
from the standard input, parse and run verifiers on it,
and write the textual format back to the standard output.
This is a good way to test if an input MLIR is well-formed.

`mlir-opt --help` shows a complete list of flags
(there are nearly 1000).
Each pass has its own flag,
though it is recommended to use `--pass-pipeline`
to run passes rather than bare flags.

## Running a pass

Next we run [`convert-to-llvm`](https://github.com/llvm/llvm-project/blob/main//docs/Passes#-convert-to-llvm),
which converts all supported dialects to the `llvm` dialect,
on the following IR:

```mlir
// mlir/test/Examples/mlir-opt/ctlz.mlir
module {
  func.func @main(%arg0: i32) -> i32 {
    %0 = math.ctlz %arg0 : i32
    func.return %0 : i32
  }
}
```

After building MLIR, and from the `llvm-project` base directory, run

```bash
build/bin/mlir-opt --pass-pipeline="builtin.module(convert-math-to-llvm)" mlir/test/Examples/mlir-opt/ctlz.mlir
```

which produces

```mlir
module {
  func.func @main(%arg0: i32) -> i32 {
    %0 = "llvm.intr.ctlz"(%arg0) <{is_zero_poison = false}> : (i32) -> i32
    return %0 : i32
  }
}
```

Note that `llvm` here is MLIR's `llvm` dialect,
which would still need to be processed through `mlir-translate`
to generate LLVM-IR.

## Running a pass with options

Next we will show how to run a pass that takes configuration options.
Consider the following IR containing loops with poor cache locality.

```mlir
// mlir/test/Examples/mlir-opt/loop_fusion.mlir
module {
  func.func @producer_consumer_fusion(%arg0: memref<10xf32>, %arg1: memref<10xf32>) {
    %0 = memref.alloc() : memref<10xf32>
    %1 = memref.alloc() : memref<10xf32>
    %cst = arith.constant 0.000000e+00 : f32
    affine.for %arg2 = 0 to 10 {
      affine.store %cst, %0[%arg2] : memref<10xf32>
      affine.store %cst, %1[%arg2] : memref<10xf32>
    }
    affine.for %arg2 = 0 to 10 {
      %2 = affine.load %0[%arg2] : memref<10xf32>
      %3 = arith.addf %2, %2 : f32
      affine.store %3, %arg0[%arg2] : memref<10xf32>
    }
    affine.for %arg2 = 0 to 10 {
      %2 = affine.load %1[%arg2] : memref<10xf32>
      %3 = arith.mulf %2, %2 : f32
      affine.store %3, %arg1[%arg2] : memref<10xf32>
    }
    return
  }
}
```

Running this with the [`affine-loop-fusion`](https://github.com/llvm/llvm-project/blob/main//docs/Passes#-affine-loop-fusion) pass
produces a fused loop.

```bash
build/bin/mlir-opt --pass-pipeline="builtin.module(affine-loop-fusion)" mlir/test/Examples/mlir-opt/loop_fusion.mlir
```

```mlir
module {
  func.func @producer_consumer_fusion(%arg0: memref<10xf32>, %arg1: memref<10xf32>) {
    %alloc = memref.alloc() : memref<1xf32>
    %alloc_0 = memref.alloc() : memref<1xf32>
    %cst = arith.constant 0.000000e+00 : f32
    affine.for %arg2 = 0 to 10 {
      affine.store %cst, %alloc[0] : memref<1xf32>
      affine.store %cst, %alloc_0[0] : memref<1xf32>
      %0 = affine.load %alloc_0[0] : memref<1xf32>
      %1 = arith.mulf %0, %0 : f32
      affine.store %1, %arg1[%arg2] : memref<10xf32>
      %2 = affine.load %alloc[0] : memref<1xf32>
      %3 = arith.addf %2, %2 : f32
      affine.store %3, %arg0[%arg2] : memref<10xf32>
    }
    return
  }
}
```

This pass has options that allow the user to configure its behavior.
For example, the `compute-tolerance` option
is described as the "fractional increase in additional computation tolerated while fusing."
If this value is set to zero on the command line,
the pass will not fuse the loops.

```bash
build/bin/mlir-opt --pass-pipeline="builtin.module(affine-loop-fusion{compute-tolerance=0})" \
mlir/test/Examples/mlir-opt/loop_fusion.mlir
```

```mlir
module {
  func.func @producer_consumer_fusion(%arg0: memref<10xf32>, %arg1: memref<10xf32>) {
    %alloc = memref.alloc() : memref<10xf32>
    %alloc_0 = memref.alloc() : memref<10xf32>
    %cst = arith.constant 0.000000e+00 : f32
    affine.for %arg2 = 0 to 10 {
      affine.store %cst, %alloc[%arg2] : memref<10xf32>
      affine.store %cst, %alloc_0[%arg2] : memref<10xf32>
    }
    affine.for %arg2 = 0 to 10 {
      %0 = affine.load %alloc[%arg2] : memref<10xf32>
      %1 = arith.addf %0, %0 : f32
      affine.store %1, %arg0[%arg2] : memref<10xf32>
    }
    affine.for %arg2 = 0 to 10 {
      %0 = affine.load %alloc_0[%arg2] : memref<10xf32>
      %1 = arith.mulf %0, %0 : f32
      affine.store %1, %arg1[%arg2] : memref<10xf32>
    }
    return
  }
}
```

Options passed to a pass
are specified via the syntax `{option1=value1 option2=value2 ...}`,
i.e., use space-separated `key=value` pairs for each option.

## Building a pass pipeline on the command line

The `--pass-pipeline` flag supports combining multiple passes into a pipeline.
So far we have used the trivial pipeline with a single pass
that is "anchored" on the top-level `builtin.module` op.
[Pass anchoring](https://github.com/llvm/llvm-project/blob/main//docs/PassManagement#oppassmanager)
is a way for passes to specify
that they only run on particular ops.
While many passes are anchored on `builtin.module`,
if you try to run a pass that is anchored on some other op
inside `--pass-pipeline="builtin.module(pass-name)"`,
it will not run.

Multiple passes can be chained together
by providing the pass names in a comma-separated list
in the `--pass-pipeline` string,
e.g.,
`--pass-pipeline="builtin.module(pass1,pass2)"`.
The passes will be run sequentially.

To use passes that have nontrivial anchoring,
the appropriate level of nesting must be specified
in the pass pipeline.
For example, consider the following IR which has the same redundant code,
but in two different levels of nesting.

```mlir
module {
  module {
    func.func @func1(%arg0: i32) -> i32 {
      %0 = arith.addi %arg0, %arg0 : i32
      %1 = arith.addi %arg0, %arg0 : i32
      %2 = arith.addi %0, %1 : i32
      func.return %2 : i32
    }
  }

  gpu.module @gpu_module {
    gpu.func @func2(%arg0: i32) -> i32 {
      %0 = arith.addi %arg0, %arg0 : i32
      %1 = arith.addi %arg0, %arg0 : i32
      %2 = arith.addi %0, %1 : i32
      gpu.return %2 : i32
    }
  }
}
```

The following pipeline runs `cse` (common subexpression elimination)
but only on the `func.func` inside the two `builtin.module` ops.

```bash
build/bin/mlir-opt mlir/test/Examples/mlir-opt/ctlz.mlir --pass-pipeline='
    builtin.module(
        builtin.module(
            func.func(cse,canonicalize),
            convert-to-llvm
        )
    )'
```

The output leaves the `gpu.module` alone

```mlir
module {
  module {
    llvm.func @func1(%arg0: i32) -> i32 {
      %0 = llvm.add %arg0, %arg0 : i32
      %1 = llvm.add %0, %0 : i32
      llvm.return %1 : i32
    }
  }
  gpu.module @gpu_module {
    gpu.func @func2(%arg0: i32) -> i32 {
      %0 = arith.addi %arg0, %arg0 : i32
      %1 = arith.addi %arg0, %arg0 : i32
      %2 = arith.addi %0, %1 : i32
      gpu.return %2 : i32
    }
  }
}
```

Specifying a pass pipeline with nested anchoring
is also beneficial for performance reasons:
passes with anchoring can run on IR subsets in parallel,
which provides better threaded runtime and cache locality
within threads.
For example,
even if a pass is not restricted to anchor on `func.func`,
running `builtin.module(func.func(cse, canonicalize))`
is more efficient than `builtin.module(cse, canonicalize)`.

For a spec of the pass-pipeline textual description language,
see [the docs](https://github.com/llvm/llvm-project/blob/main//docs/PassManagement#textual-pass-pipeline-specification).
For more general information on pass management, see [Pass Infrastructure](https://github.com/llvm/llvm-project/blob/main//docs/PassManagement#).

## Useful CLI flags

- `--help` and `--help-hidden` show a list of flags.
- `--debug` prints all debug information produced by `LLVM_DEBUG` calls.
- `--debug-only="my-tag"` prints only the debug information produced by `LLVM_DEBUG`
  in files that have the macro `#define DEBUG_TYPE "my-tag"`.
  This often allows you to print only debug information associated with a specific pass.
    - `"greedy-rewriter"` only prints debug information
      for patterns applied with the greedy rewriter engine.
    - `"dialect-conversion"` only prints debug information
      for the dialect conversion framework.
 - `--dump-pass-pipeline` dumps the pass pipeline that will be run to standard output.
   This output can be directly passed to `--pass-pipeline` and is useful to
   identify exactly which passes and options are executed.
 - `--emit-bytecode` emits MLIR in the bytecode format.
 - `--mlir-pass-statistics` print statistics about the passes run.
    These are generated via [pass statistics](https://github.com/llvm/llvm-project/blob/main//docs/PassManagement#pass-statistics).
 - `--mlir-print-ir-after-all` prints the IR after each pass.
    - See also `--mlir-print-ir-after-change`, `--mlir-print-ir-after-failure`,
      and analogous versions of these flags with `before` instead of `after`.
    - When using `print-ir` flags, adding `--mlir-print-ir-tree-dir` writes the
      IRs to files in a directory tree, making them easier to inspect versus a
      large dump to the terminal.
 - `--mlir-timing` displays execution times of each pass.
 - `--view-op-graph` runs a pass that generates a Graphviz DOT file representing
   the module at the given step of a pipeline.

## Further reading

- [List of passes](https://github.com/llvm/llvm-project/blob/main//docs/Passes)
- [List of dialects](https://github.com/llvm/llvm-project/blob/main//docs/Dialects)

---

## Deeper notes

### The flags worth memorising

| Flag | What it does |
|------|--------------|
| `--pass-pipeline='builtin.module(func.func(cse,canonicalize))'` | explicit nested pipeline |
| `--mlir-print-ir-after-all` | IR after every pass |
| `--mlir-print-ir-after-change` | only when a pass changed something |
| `--mlir-print-ir-after-failure` | only on failure |
| `--mlir-print-ir-before=my-pass` | targeted |
| `--mlir-print-op-generic` | generic syntax; the truth about your custom format |
| `--mlir-print-debuginfo` | locations, for tracing where an operation came from |
| `--debug-only=X` | debug output from one component |
| `--mlir-disable-threading` | deterministic ordering while debugging |
| `--mlir-timing` | per-pass timing |
| `--split-input-file` | many independent cases in one file |
| `--verify-diagnostics` | assert on expected diagnostics |
| `--allow-unregistered-dialect` | parse operations from dialects that are not linked in |
| `--emit-bytecode` | write the binary format |

`--debug-only` values you will use constantly: `greedy-rewriter`, `dialect-conversion`,
`pattern-application`, `pass-manager`.

### The debugging loop

1. Reduce the input to the smallest file that reproduces the problem.
2. `--mlir-print-ir-after-all` and find the first pass whose output is wrong.
3. Re-run only that pass on the IR from step 2's dump.
4. `--debug-only=` for that pass's driver to see which patterns fired.
5. If it crashes, use the crash reproducer
   ([Pass Infrastructure](../03-passes-and-rewriting/01-pass-infrastructure.md)) and
   [`mlir-reduce`](06-mlir-reduce.md).

Step 3 is the one people skip. Isolating a single pass on a single input removes almost all the
variables, and re-running that one invocation is fast enough to iterate on.

### `--allow-unregistered-dialect` is a trap as well as a convenience

It lets you write tests using operations you have not defined, which is genuinely useful for testing
generic infrastructure. It also silently accepts typos in operation names, so a mistyped
`arith.adf` parses as an unregistered operation and your pattern mysteriously never fires. If a
pattern is not firing, check this flag first.

### Your own tool

A downstream dialect gets its own `my-opt` binary registering its dialects and passes. `mlir-opt`'s
`main` is a few lines around `MlirOptMain`; copy it. Do this early — it is the prerequisite for
testing everything else.

### The neighbouring tools

| Tool | Purpose |
|------|---------|
| `mlir-opt` | run passes on `.mlir` |
| `mlir-translate` | in and out of MLIR — `--mlir-to-llvmir` and importers |
| `mlir-tblgen` | generate C++ and docs from `.td` |
| `mlir-pdll` | compile PDLL patterns |
| `mlir-reduce` | shrink a failing test case |
| `mlir-lsp-server` | editor support |
| `mlir-cpu-runner` | JIT and execute a lowered module |

`mlir-cpu-runner` deserves more attention than it usually gets: being able to actually *run* the
output of your pipeline and compare numbers against a reference is a much stronger check on a
lowering than reading the generated IR.


---

[← MLIR Bytecode Format](../05-data-representation/03-bytecode-format.md) · [Index](../README.md) · [Diagnostic Infrastructure →](02-diagnostic-infrastructure.md)
