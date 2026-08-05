# `mlir-reduce`

> **Section:** Tooling and Debugging · document 6 of 7  
> **Upstream:** [https://mlir.llvm.org/docs/Tools/mlir-reduce/](https://mlir.llvm.org/docs/Tools/mlir-reduce/) · source [`mlir/docs/Tools/mlir-reduce.md`](https://github.com/llvm/llvm-project/blob/main/mlir/docs/Tools/mlir-reduce.md)  
> **License:** upstream text is Apache-2.0 WITH LLVM-exception.

## Orientation

Given an input that triggers a failure and a script that tests for that failure, `mlir-reduce`
repeatedly removes parts of the input while the failure persists — the MLIR equivalent of
`llvm-reduce` or `creduce`.

The technique to internalise is that this turns "a two-thousand-operation model crashes somewhere"
into "these four operations crash", usually in minutes and without your attention. Combined with the
pass-manager crash reproducer, it is the standard opening move for any nontrivial bug report.

**Read first**

- [Using `mlir-opt`](01-using-mlir-opt.md)

**What you should be able to do after this page**

- Write an interestingness test and drive a reduction with it.
- Combine reduction with the crash reproducer.

---

## Upstream documentation

An MLIR input may trigger bugs after series of transformations. To root cause
the problem or help verification after fixes, developers want to be able to
reduce the size of a reproducer for a bug. This document describes
`mlir-reduce`, which is similar to
[bugpoint](https://llvm.org/docs/CommandGuide/bugpoint.html), a tool that can
reduce the size of the input needed to trigger the error.

`mlir-reduce` supports reducing the input in several ways, including simply
deleting code not required to reproduce an error, applying the reducer
patterns heuristically or run with optimization passes to reduce the input. To
use it, the first thing you need to do is, provide a command which tells if an
input is interesting, e.g., exhibits the characteristics that you would like to
focus on. For example, you may want to see if `mlir-opt` invocation fails after
it runs on the certain MLIR input. Afterwards, select your reduction strategy
then `mlir-reduce` will do the remaining works for you.

## How to Use it

`mlir-reduce` adopts the reduction-tree algorithm to reduce the input. It
generates several reduced outputs and further reduces in between them according
to the tree traversal strategy. The different strategies may lead to different
results and different time complexity. You can run as
`-reduction-tree='traversal-mode=0'` to select the mode for example.

### Example MLIR input

```mlir
// query-test.mlir
func.func @func1() {
  // A func can be pruned if it's not relevant to the error.
  return
}

func.func @func2(%arg0: i1) -> f32 {
  %0 = arith.constant 1 : i32
  %1 = arith.constant 2 : i32
  %2 = arith.constant 2.2 : f32
  %3 = arith.constant 5.3 : f32
  %4 = arith.addi %0, %1 : i32
  %5 = arith.addf %2, %3 : f32
  %6 = arith.muli %4, %4 : i32
  %7 = arith.subi %6, %4 : i32
  %8 = arith.select %arg0, %5, %2 : f32
  %9 = arith.addf %2, %8 : f32
  return %9 : f32
}
```

### Write the script for testing interestingness

You need to provide a command to `mlir-reduce` which identifies cases you're
interested in. For each intermediate output generated during reduction,
`mlir-reduce` will run the command over the it, the script should returns 1 for
interesting case, 0 otherwise. For the IR above, a sample script might simply
look for the presence of the `arith.fptosi` operation. A more realistic script
would check for the presence of a particular kind of error message in stderr.

```shell
#!/bin/bash
#
# query-test.sh
# `2>&1` redirects stderr (where errors and diagnostics are printed) to stdout
# so it can be piped to grep.
#
# Note mlir-opt needs to be on your path, or else replace mlir-opt with the
# absolute path to the binary. If mlir-opt cannot be found, this interestingness
# test will report "uninteresting on the first run, and mlir-reduce will
# report that the input is not marked as interesting.
#
mlir-opt $1 2>&1 | grep "arith.select"

# grep's exit code is 0 if the queried string is found
if [[ $? -eq 0 ]]; then
  exit 1
else
  exit 0
fi
```

### Running the example

The sample usage will be as follows, noting that the `test` argument is part of
the mode argument.

```shell
mlir-reduce query-test.mlir -reduction-tree='traversal-mode=0 test=query-test.sh'
```

The output:

```
module {
  func.func @func2(%arg0: i1) -> f32 {
    %cst = arith.constant 2.200000e+00 : f32
    %cst_0 = arith.constant 7.500000e+00 : f32
    %0 = arith.select %arg0, %cst_0, %cst : f32
    %1 = arith.addf %0, %cst : f32
    return %1 : f32
  }
}
```

## Available reduction strategies

### Operation elimination

`mlir-reduce` will try to remove the operations directly. This is the most
aggressive reduction as it may result in an invalid output as long as it ends up
retaining the error message that the test script is interesting. To avoid that,
`mlir-reduce` always checks the validity and it expects the user will provide a
valid input as well.

### Rewrite patterns into simpler forms

In some cases, rewrite an operation into a simpler or smaller form can still
retain the interestingness. For example, `mlir-reduce` will try to rewrite a
`tensor<?xindex>` with unknown rank into a constant rank one like
`tensor<1xi32>`. Not only produce a simpler operation, it may introduce further
reduction chances because of precise type information.

MLIR supports dialects and `mlir-reduce` supports rewrite patterns for every
dialect as well. Which means you can have the dialect specific rewrite patterns.
To do that, you need to implement the `DialectReductionPatternInterface`. For
example,

```c++
#include "mlir/Reducer/ReductionPatternInterface.h"

struct MyReductionPatternInterface : public DialectReductionPatternInterface {
  MyReductionPatternInterface(Dialect *dialect)
      : DialectReductionPatternInterface(dialect) {};

  virtual void
  populateReductionPatterns(RewritePatternSet &patterns) const final {
    populateMyReductionPatterns(patterns);
  }
}
```

`mlir-reduce` will call `populateReductionPatterns` to collect the reduction
rewrite patterns provided by each dialect. Here's a hint, if you use
[DRR](../03-passes-and-rewriting/04-declarative-rewrite-rules-drr.md) to write the reduction patterns, you can
leverage the method `populateWithGenerated` generated by `mlir-tblgen`.

### Reduce with built-in optimization passes

MLIR provides amount of transformation passes and some of them are useful for
reducing the input size, e.g., Symbol-DCE. `mlir-reduce` will schedule them
along with above two strategies.

## Build a custom mlir-reduce

In the cases of, 1. have defined a custom syntax, 2. the failure is specific to
certain dialects or 3. there's a dialect specific reducer patterns, you need to
build your own `mlir-reduce`. Link it with `MLIRReduceLib` and implement it
like,

```c++
#include "mlir/Tools/mlir-reduce/MlirReduceMain.h"
using namespace mlir;

int main(int argc, char **argv) {
  DialectRegistry registry;
  registerMyDialects(registry);
  // Register the DialectReductionPatternInterface if any.
  MLIRContext context(registry);
  return failed(mlirReduceMain(argc, argv, context));
}

```

## Future works

`mlir-reduce` is missing several features,

*   `-reduction-tree` now only supports `Single-Path` traversal mode, extends it
with different traversal strategies may reduce the input better.
*   Produce the optimal result when interrupted. The reduction process may take
a quite long time, it'll be better to get an optimal result so far while an
interrupt is triggered.

---

## Deeper notes

### The workflow

1. Get a crash reproducer:
   `mlir-opt --mlir-pass-pipeline-crash-reproducer=repro.mlir …`
2. Write an interestingness test — a script that exits 0 when the input still exhibits the problem:

   ```bash
   #!/bin/bash
   mlir-opt --my-pass "$1" 2>&1 | grep -q "the specific error"
   ```

3. `mlir-reduce repro.mlir --test=./interesting.sh --test-arg=...`

### Getting the interestingness test right

This is the whole game, and the failure mode is a test that is too loose. If your script checks only
for a non-zero exit code, the reducer will happily produce input that fails for an entirely different
reason — a parse error, say — and you will spend time on a bug that does not exist. Grep for the
specific message, or for a specific assertion.

Conversely a test that is too strict, matching an exact line number, prevents reduction from making
progress.

### Reduce before reporting, and before reasoning

A reduced case is easier for a maintainer to act on, but the larger benefit is to you: most bugs
become obvious once the input is four operations. It is common to reduce a case and see the problem
without needing to debug at all.

### How it reduces

The tool applies reduction strategies — removing operations, replacing operands with simpler values,
running passes that shrink the IR — and keeps whatever still satisfies the interestingness test. You
can extend it with your own reduction patterns for dialect-specific structure, which is worth doing
if your dialect has large constructs that the generic strategies cannot break down.


---

[← MLIR Language Server Protocol](05-language-server-protocol.md) · [Index](../README.md) · [`mlir-rewrite` →](07-mlir-rewrite.md)
