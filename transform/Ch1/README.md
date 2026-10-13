# Chapter 1: Combining Existing Transformations

> **Goal:** Drive a sequence of existing MLIR transformations — tiling, fusion, outlining — from a script written in the Transform dialect itself, run it with the stock `mlir-opt` transform interpreter, and learn the rules that keep such scripts safe: handles, consumption, invalidation and failure propagation — based on the official tutorial [Transform Ch1: Combining Existing Transformations](https://mlir.llvm.org/docs/Tutorials/transform/Ch1/).

---

## 1. Overview

Chapter 0 introduced Linalg's *structured* operations: operations such as `linalg.matmul` that keep the shape of a computation (its iteration space and access patterns) visible, so that tiling and fusion become simple. This chapter applies those transformations — not by writing a pass in C++, but by writing a **transform script**: MLIR operations from the `transform` dialect that say *which* payload operations to transform and *how*, step by step.

We will:

1. Meet the vocabulary: **payload IR** (the IR being transformed), **transform IR** (the script), and **handles** (transform-IR values that refer to payload operations) — section 2.
2. Write the entry point of a script, `transform.named_sequence @__transform_main`, and run it — sections 3 and 5.
3. See how failures propagate (or are suppressed) and how to print what a handle points to — section 4.
4. Tile a matrix multiplication, then tile and fuse a whole "fully connected + bias + ReLU" layer, and outline the innermost tile into a function — sections 6, 8 and 9.
5. Break the rules on purpose: use a handle after a transformation consumed it, and watch the interpreter's *expensive checks* catch it — sections 7 and 9.
6. Watch the dialect keep handles up to date when patterns rewrite the payload — section 10.

Everything in this chapter runs with the stock `mlir-opt` from Homebrew LLVM/MLIR 20 (`/opt/homebrew/opt/llvm@20/bin/mlir-opt`); there is no C++ to build. The inputs are the chapter's test files in `test_Example/transform/Ch1/` (`sequence.mlir`, `invalidation-1.mlir`, `invalidation-2.mlir`) and the stage files in `stages/`, which build the final script up one step at a time: each `stages/N-*.mlir` holds the same payload function followed by the script of one section.

Sections 2–10 follow the chapter's path; section 11 shows how to arrive at the complete script without having it — from the payload's operations, the interpreter's options, the operations' definitions, the expensive checks and the microkernel's signature — and section 12 collects the takeaways. The appendices hold a script kept in a file of its own, applied with `--transform-preload-library` and finding its handles with `transform.structured.match` (A), how the stage files relate to each other and to the test files (B), and the script that runs every command in this README (C).

---

## 2. Introduction: Transform IR and Payload IR

The Transform dialect lets you target transformations precisely at specific operations in the IR and **chain** them, that is, apply a transformation to the operations produced by the previous one. To do that, the transformations are themselves expressed as operations in the IR:

- the IR containing these transform operations is the **transform IR**;
- the IR being transformed is the **payload IR**.

Transform IR operations operate on values that may be associated with payload IR **operations**, payload IR **values**, or **attributes**. The first two kinds of transform values are called **operation handles** and **value handles**; the last kind are called **parameters**. A handle is not a single pointer: it is associated with a *list* of payload entities (possibly empty, possibly with several elements).

Applying transform IR always starts from **one top-level transform operation**. In the C++ API, this operation is passed to the `applyTransforms` function. The top-level operation decides whether and how the other transformations are performed. The most common one, `transform.named_sequence`, simply applies the transform operations in its body one after the other, like a function or a macro.

The running example is the common "fully connected + bias + ReLU" machine-learning layer (ReLU, the rectified linear unit, is `relu(x) = max(0, x)`). It boils down to a matrix multiplication, an elementwise matrix addition, and an elementwise maximum with 0:

***stages/payload.mlir***
```mlir
// Original function to optimize.
func.func @fc_relu(%lhs: tensor<512x512xf32>, %rhs: tensor<512x512xf32>,
                   %bias: tensor<512x512xf32>, %output: tensor<512x512xf32>)
                   -> tensor<512x512xf32> {
  // Matrix-matrix multiplication.
  %matmul = linalg.matmul ins(%lhs, %rhs: tensor<512x512xf32>, tensor<512x512xf32>)
                          outs(%output: tensor<512x512xf32>) -> tensor<512x512xf32>

  // Elementwise addition.
  %biased = linalg.elemwise_binary { fun = #linalg.binary_fn<add> }
    ins(%matmul, %bias : tensor<512x512xf32>, tensor<512x512xf32>)
    outs(%output : tensor<512x512xf32>) -> tensor<512x512xf32>

  // Elementwise max with 0 (ReLU).
  %c0f = arith.constant 0.0 : f32
  %relued = linalg.elemwise_binary { fun = #linalg.binary_fn<max_signed> }
    ins(%biased, %c0f : tensor<512x512xf32>, f32)
    outs(%output : tensor<512x512xf32>) -> tensor<512x512xf32>
  func.return %relued : tensor<512x512xf32>
}
```

The same 20 lines open every `stages/N-*.mlir` file and `test_Example/transform/Ch1/sequence.mlir` (appendix B checks this). Three things to notice before transforming it:

- All three operations are structured Linalg operations on **tensors**: each produces a new 512x512 tensor; `outs(%output ...)` only provides the initial value (chapter 0).
- The two elementwise operations are **the same kind of operation**, `linalg.elemwise_binary`, differing only in their `fun` attribute (`add` vs `max_signed`). A pass that works by operation name cannot tell them apart; a transform script can, because it points at individual operations (section 8).
- The constant `%c0f` is a scalar `f32`: `linalg.elemwise_binary` broadcasts it to every element.

The payload parses and verifies on its own, with no transform script involved:

```bash
cd /Users/roy/study/mlir/transform/Ch1
mlir-opt stages/payload.mlir
```

Real output **[1]**:

```text
module {
  func.func @fc_relu(%arg0: tensor<512x512xf32>, %arg1: tensor<512x512xf32>, %arg2: tensor<512x512xf32>, %arg3: tensor<512x512xf32>) -> tensor<512x512xf32> {
    %0 = linalg.matmul ins(%arg0, %arg1 : tensor<512x512xf32>, tensor<512x512xf32>) outs(%arg3 : tensor<512x512xf32>) -> tensor<512x512xf32>
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%0, %arg2 : tensor<512x512xf32>, tensor<512x512xf32>) outs(%arg3 : tensor<512x512xf32>) -> tensor<512x512xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<max_signed>} ins(%1, %cst : tensor<512x512xf32>, f32) outs(%arg3 : tensor<512x512xf32>) -> tensor<512x512xf32>
    return %2 : tensor<512x512xf32>
  }
}

```

`mlir-opt` prints the function in its canonical form: arguments are renumbered (`%lhs` → `%arg0`, ...), the attribute dictionary of `linalg.elemwise_binary` loses its spaces, and `0.0` is printed as `0.000000e+00`. Later outputs use these names.

---

## 3. Top-Level Sequence Operation

For performance, we want to **tile and fuse** these operations to exploit cache locality. That is a sequence of transformations performed one after another, so we start with the corresponding top-level transform operation — an empty one for now:

***stages/1-empty.mlir***
```mlir
...
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %arg0: !transform.any_op,
      %arg1: !transform.op<"linalg.matmul">,
      %arg2: !transform.op<"linalg.elemwise_binary">) {
    transform.yield
  }
}
```

Several aspects of this operation are worth noticing:

- **The name `@__transform_main` and the first argument are mandated by the interpreter pass**, much like a C program's entry point must be called `main` and may have the `int (int argc, char **argv)` signature. The first argument is associated with the top-level payload operation, most often the operation the pass runs on (here, the whole `builtin.module`). None of this is required when the script is applied *programmatically*, through `applyTransforms` or `applyNamedSequence`.
- **The remaining entry-block arguments are optional.** They can be associated with payload attributes, operations or values that are useful in the sequence; when calling `applyTransforms`, they are passed in too. Here we want the matrix multiplication and the elementwise operations, which we are going to tile and fuse. With the interpreter pass, a pass option binds them (section 5).
- **Every handle has a Transform dialect type**, which states properties of the payload entities it may be associated with. `!transform.any_op` accepts arbitrary payload operations; `!transform.op<"X">` accepts *only* payload operations named `X`. These constraints are verified when the handle-to-payload association is created — for the entry-block arguments of the top-level operation, early in `applyTransforms`. If they are not satisfied, the application fails with a diagnostic (section 5.3 shows one).
- **The sequence lives in a module carrying the `transform.with_named_sequence` attribute**, which triggers the verifications needed when several named sequences exist (for example, that a call to a named sequence refers to one with a matching signature).
- **The body ends with `transform.yield`**, the terminator of transform regions.

Running the empty script leaves the payload untouched — the transform module itself is printed too, since it is part of the input module:

```bash
cd /Users/roy/study/mlir/transform/Ch1
mlir-opt stages/1-empty.mlir \
  --pass-pipeline="builtin.module(transform-interpreter{
      debug-bind-trailing-args=linalg.matmul,linalg.elemwise_binary})"
```

Real output **[2]**:

```text
module {
  func.func @fc_relu(%arg0: tensor<512x512xf32>, %arg1: tensor<512x512xf32>, %arg2: tensor<512x512xf32>, %arg3: tensor<512x512xf32>) -> tensor<512x512xf32> {
    %0 = linalg.matmul ins(%arg0, %arg1 : tensor<512x512xf32>, tensor<512x512xf32>) outs(%arg3 : tensor<512x512xf32>) -> tensor<512x512xf32>
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%0, %arg2 : tensor<512x512xf32>, tensor<512x512xf32>) outs(%arg3 : tensor<512x512xf32>) -> tensor<512x512xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<max_signed>} ins(%1, %cst : tensor<512x512xf32>, f32) outs(%arg3 : tensor<512x512xf32>) -> tensor<512x512xf32>
    return %2 : tensor<512x512xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op, %arg1: !transform.op<"linalg.matmul">, %arg2: !transform.op<"linalg.elemwise_binary">) {
      transform.yield 
    }
  }
}

```

The transform script and its payload share one module, but they are different worlds: the interpreter executes the operations of `@__transform_main`, and the payload is what those operations act upon. Section 5 explains the command.

---

## 4. Failure Propagation

The Transform dialect has a dedicated mechanism for diagnostics that supports **recoverable errors**. A transform operation can end in one of three ways: success, a **silenceable failure** (the transformation did not apply, but the payload is still in a valid state, so the script may decide to carry on), or a **definite failure** (something irreparable happened, and the whole application stops).

The mechanism is easiest to understand with the (unnamed) `transform.sequence` operation, whose mandatory `failures(...)` clause specifies the failure propagation mode. There are two options:

- **`propagate`** makes the sequence fail if any of the nested transformations fails;
- **`suppress`** makes the sequence succeed even if one of the nested transformations fails, but without performing the transformations that follow the failed one in the sequence.

The latter lets the script around the sequence continue despite errors within it, assuming they are recoverable. While we are building the script, it is preferable to propagate failures, so we know when something did not apply.

To check or debug a transform sequence, you can print various entities associated with transform IR values. For example, this prints the operations associated with the handles, by emitting a remark at each of them (not this repo's code — section 5.4 runs it, and shows why it cannot be the top-level operation for the interpreter pass):

```mlir
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op,
     %arg1: !transform.op<"linalg.matmul">,
     %arg2: !transform.op<"linalg.elemwise_binary">):
  transform.debug.emit_remark_at %arg1, "matmul"
      : !transform.op<"linalg.matmul">
  transform.debug.emit_remark_at %arg2, "elemwise_binaries"
      : !transform.op<"linalg.elemwise_binary">
  transform.yield
}
```

#### The two modes side by side

`transform.sequence` is also usable *inside* a named sequence, applied to a handle (here the payload root `%arg0`, whose ops become the block argument `%root`). The following script runs the same failing step twice, once under each mode. The step is `transform.split_handle`, which splits a handle associated with N payload operations into N handles; asking for three handles from `%arg2`, which holds the two elementwise operations, is a silenceable failure:

***stages/2-failures.mlir***
```mlir
...
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %arg0: !transform.any_op,
      %arg1: !transform.op<"linalg.matmul">,
      %arg2: !transform.op<"linalg.elemwise_binary">) {
    // "suppress": the failing split_handle stops this sequence, the remark
    // after it is skipped, but the enclosing script carries on.
    transform.sequence %arg0 : !transform.any_op failures(suppress) {
    ^bb0(%root: !transform.any_op):
      // %arg2 holds two payload ops, so splitting it into three fails.
      %a, %b, %c = transform.split_handle %arg2
          : (!transform.op<"linalg.elemwise_binary">)
          -> (!transform.any_op, !transform.any_op, !transform.any_op)
      transform.debug.emit_remark_at %a, "not reached (suppress)" : !transform.any_op
      transform.yield
    }
    transform.debug.emit_remark_at %arg1, "after the suppressed sequence"
        : !transform.op<"linalg.matmul">

    // "propagate": the same failure makes the whole script fail.
    transform.sequence %arg0 : !transform.any_op failures(propagate) {
    ^bb0(%root: !transform.any_op):
      %a, %b, %c = transform.split_handle %arg2
          : (!transform.op<"linalg.elemwise_binary">)
          -> (!transform.any_op, !transform.any_op, !transform.any_op)
      transform.debug.emit_remark_at %a, "not reached (propagate)" : !transform.any_op
      transform.yield
    }
    transform.debug.emit_remark_at %arg1, "not reached either"
        : !transform.op<"linalg.matmul">
    transform.yield
  }
}
```

```bash
cd /Users/roy/study/mlir/transform/Ch1
mlir-opt stages/2-failures.mlir \
  --pass-pipeline="builtin.module(transform-interpreter{
      debug-bind-trailing-args=linalg.matmul,linalg.elemwise_binary})" \
  -o /dev/null
```

Real output **[3]** (exit status 1):

```text
stages/2-failures.mlir:6:13: remark: after the suppressed sequence
  %matmul = linalg.matmul ins(%lhs, %rhs: tensor<512x512xf32>, tensor<512x512xf32>)
            ^
stages/2-failures.mlir:6:13: note: see current operation: %0 = linalg.matmul ins(%arg0, %arg1 : tensor<512x512xf32>, tensor<512x512xf32>) outs(%arg3 : tensor<512x512xf32>) -> tensor<512x512xf32>
stages/2-failures.mlir:44:20: error: <block argument> of type '!transform.op<"linalg.elemwise_binary">' at index: 2 expected to contain 3 payloads but it contains 2 payloads
      %a, %b, %c = transform.split_handle %arg2
                   ^
```

Reading the output top to bottom:

- **The suppressed failure left no trace at all**: no error, and the remark `"not reached (suppress)"` after it is skipped. The sequence then counts as a success, so the remark `"after the suppressed sequence"` is emitted.
- **The propagated failure is reported as an error** (`expected to contain 3 payloads but it contains 2 payloads`) pointing at the failing `split_handle`. It ends the second sequence, then the named sequence, then the interpreter pass — `"not reached either"` never runs, and `mlir-opt` exits with status 1.

`suppress` swallows the diagnostic entirely, which is why **a script under development should propagate failures: a suppressed step that never applied looks exactly like one that was not needed.**

---

## 5. Transform Dialect Interpreter

Since we don't want to recompile the compiler every time the transformation changes, we use the **transform dialect interpreter pass**, `transform-interpreter`, to apply the sequence to the payload IR. (The next chapters show that you can also define custom passes, or integrate the interpreter into a larger pass.)

### 5.1 Running the sequence

Here is the remark-printing sequence of section 4, as a named sequence:

***stages/3-remarks.mlir***
```mlir
...
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %arg0: !transform.any_op,
      %arg1: !transform.op<"linalg.matmul">,
      %arg2: !transform.op<"linalg.elemwise_binary">) {
    transform.debug.emit_remark_at %arg1, "matmul"
        : !transform.op<"linalg.matmul">
    transform.debug.emit_remark_at %arg2, "elemwise_binaries"
        : !transform.op<"linalg.elemwise_binary">
    transform.yield
  }
}
```

```bash
cd /Users/roy/study/mlir/transform/Ch1
mlir-opt stages/3-remarks.mlir \
  --pass-pipeline="builtin.module(transform-interpreter{
      debug-bind-trailing-args=linalg.matmul,linalg.elemwise_binary})" \
  -o /dev/null
```

Real output **[4]**:

```text
stages/3-remarks.mlir:6:13: remark: matmul
  %matmul = linalg.matmul ins(%lhs, %rhs: tensor<512x512xf32>, tensor<512x512xf32>)
            ^
stages/3-remarks.mlir:6:13: note: see current operation: %0 = linalg.matmul ins(%arg0, %arg1 : tensor<512x512xf32>, tensor<512x512xf32>) outs(%arg3 : tensor<512x512xf32>) -> tensor<512x512xf32>
stages/3-remarks.mlir:10:13: remark: elemwise_binaries
  %biased = linalg.elemwise_binary { fun = #linalg.binary_fn<add> }
            ^
stages/3-remarks.mlir:10:13: note: see current operation: %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%0, %arg2 : tensor<512x512xf32>, tensor<512x512xf32>) outs(%arg3 : tensor<512x512xf32>) -> tensor<512x512xf32>
stages/3-remarks.mlir:16:13: remark: elemwise_binaries
  %relued = linalg.elemwise_binary { fun = #linalg.binary_fn<max_signed> }
            ^
stages/3-remarks.mlir:16:13: note: see current operation: %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<max_signed>} ins(%1, %cst : tensor<512x512xf32>, f32) outs(%arg3 : tensor<512x512xf32>) -> tensor<512x512xf32>
```

The file contains *both* the payload function *and* the transform sequence, nested in the same module. The pieces of the command:

- `--pass-pipeline="builtin.module(transform-interpreter{...})"` runs the interpreter on the top-level `builtin.module`, the pass's **anchor** operation. The interpreter looks up `@__transform_main` and applies it to the anchor: `%arg0` is bound to the module.
- `debug-bind-trailing-args=linalg.matmul,linalg.elemwise_binary` binds the extra (trailing) entry-block arguments, in order: `%arg1` to *all* `linalg.matmul` operations in the payload, `%arg2` to *all* `linalg.elemwise_binary` operations. As its name says, this option is a debugging aid; a real script finds its operations itself (appendix A).
- `-o /dev/null` discards the printed module; the remarks go to the error stream and remain visible.

The remarks land as expected. Note that **`%arg2` is associated with both elementwise operations**, so its single `emit_remark_at` produced two remarks. Any handle is associated with a list of entities; individual transformations may or may not care about the order of the elements in that list (here it is the order in which they appear in the payload).

The interpreter pass has these options (from `mlir-opt --help`, MLIR 20):

| Option | Effect |
|---|---|
| `entry-point=<name>` | Name of the named sequence to start from (default `__transform_main`). |
| `debug-bind-trailing-args=<op,...>` | Bind each trailing entry argument to all payload ops with the given name (sections 5.1–5.3). |
| `debug-payload-root-tag=<tag>` | Use the op whose `transform.target_tag` attribute equals `<tag>` as the payload root instead of the pass anchor (section 5.5). |
| `disable-expensive-checks` | Skip the handle-invalidation checks (section 7). |

### 5.2 Forgetting the trailing arguments

The entry point declares two extra arguments; without the option, nothing provides them:

```bash
cd /Users/roy/study/mlir/transform/Ch1
mlir-opt stages/3-remarks.mlir --transform-interpreter -o /dev/null
```

Real output **[5]** (exit status 1):

```text
stages/3-remarks.mlir:23:3: error: operation expects 2 extra value bindings, but 0 were provided to the interpreter
  transform.named_sequence @__transform_main(
  ^
```

`--transform-interpreter` is the shorthand for running the pass with default options on the module; the check that every entry argument is bound happens before any transform runs.

### 5.3 Handle types are checked when the handle is bound

Swapping the two names binds `%arg1`, typed `!transform.op<"linalg.matmul">`, to the `linalg.elemwise_binary` operations:

```bash
cd /Users/roy/study/mlir/transform/Ch1
mlir-opt stages/3-remarks.mlir \
  --pass-pipeline="builtin.module(transform-interpreter{
      debug-bind-trailing-args=linalg.elemwise_binary,linalg.matmul})" \
  -o /dev/null
```

Real output **[6]** (exit status 1):

```text
stages/3-remarks.mlir:25:7: error: incompatible payload operation name expected linalg.matmul vs linalg.elemwise_binary -> 
%1 = "linalg.elemwise_binary"(%0, %arg2, %arg3) <{fun = #linalg.binary_fn<add>, operandSegmentSizes = array<i32: 2, 1>}> ({
^bb0(%arg7: f32, %arg8: f32, %arg9: f32):
  %5 = "arith.addf"(%arg7, %arg8) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
  "linalg.yield"(%5) : (f32) -> ()
}) : (tensor<512x512xf32>, tensor<512x512xf32>, tensor<512x512xf32>) -> tensor<512x512xf32>
      %arg1: !transform.op<"linalg.matmul">,
      ^
stages/3-remarks.mlir:10:13: note: payload operation
  %biased = linalg.elemwise_binary { fun = #linalg.binary_fn<add> }
            ^
```

The error points at the argument declaration (`%arg1: !transform.op<"linalg.matmul">`) and the note at the first offending payload operation. No remark was emitted: the association is rejected before the first transform operation runs. **A handle type is a contract checked at the boundary**, so a script can rely on `%arg1` holding only matmuls without testing it.

### 5.4 The top-level operation must be a named sequence

The unnamed `transform.sequence` listing of section 4 is a valid transform operation, but the interpreter pass only starts from a *named* sequence:

```bash
cd /Users/roy/study/mlir/transform/Ch1
{ cat stages/payload.mlir; cat <<'EOF'

transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op,
     %arg1: !transform.op<"linalg.matmul">,
     %arg2: !transform.op<"linalg.elemwise_binary">):
  transform.debug.emit_remark_at %arg1, "matmul"
      : !transform.op<"linalg.matmul">
  transform.debug.emit_remark_at %arg2, "elemwise_binaries"
      : !transform.op<"linalg.elemwise_binary">
  transform.yield
}
EOF
} | mlir-opt --pass-pipeline="builtin.module(transform-interpreter{
      debug-bind-trailing-args=linalg.matmul,linalg.elemwise_binary})" \
  -o /dev/null
```

Real output **[7]** (exit status 1; the operation dump is abridged):

```text
<stdin>:0:0: error: could not find a nested named sequence with name: __transform_main
<stdin>:0:0: note: see current operation: 
"builtin.module"() ({
...
  "transform.sequence"() <{failure_propagation_mode = 1 : i32, operandSegmentSizes = array<i32: 0, 0>}> ({
  ^bb0(%arg0: !transform.any_op, %arg1: !transform.op<"linalg.matmul">, %arg2: !transform.op<"linalg.elemwise_binary">):
    "transform.debug.emit_remark_at"(%arg1) <{message = "matmul"}> : (!transform.op<"linalg.matmul">) -> ()
    "transform.debug.emit_remark_at"(%arg2) <{message = "elemwise_binaries"}> : (!transform.op<"linalg.elemwise_binary">) -> ()
    "transform.yield"() : () -> ()
  }) : () -> ()
}) : () -> ()
```

The unnamed form is how sequences are nested (section 4); as an entry point, it only works when C++ code calls `applyTransforms` on it directly.

### 5.5 Choosing the payload root

`%arg0` is bound to the payload root — by default the pass anchor. Adding a remark at `%arg0` to the empty script shows which operation that is:

```bash
cd /Users/roy/study/mlir/transform/Ch1
sed '/transform.yield/i\
    transform.debug.emit_remark_at %arg0, "payload root" : !transform.any_op' \
  stages/1-empty.mlir |
mlir-opt --pass-pipeline="builtin.module(transform-interpreter{
      debug-bind-trailing-args=linalg.matmul,linalg.elemwise_binary})" \
  -o /dev/null
```

Real output **[8]**:

```text
<stdin>:0:0: remark: payload root
<stdin>:0:0: note: see current operation: 
module {
  func.func @fc_relu(%arg0: tensor<512x512xf32>, %arg1: tensor<512x512xf32>, %arg2: tensor<512x512xf32>, %arg3: tensor<512x512xf32>) -> tensor<512x512xf32> {
    %0 = linalg.matmul ins(%arg0, %arg1 : tensor<512x512xf32>, tensor<512x512xf32>) outs(%arg3 : tensor<512x512xf32>) -> tensor<512x512xf32>
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%0, %arg2 : tensor<512x512xf32>, tensor<512x512xf32>) outs(%arg3 : tensor<512x512xf32>) -> tensor<512x512xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<max_signed>} ins(%1, %cst : tensor<512x512xf32>, f32) outs(%arg3 : tensor<512x512xf32>) -> tensor<512x512xf32>
    return %2 : tensor<512x512xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op, %arg1: !transform.op<"linalg.matmul">, %arg2: !transform.op<"linalg.elemwise_binary">) {
      transform.debug.emit_remark_at %arg0, "payload root" : !transform.any_op
      transform.yield 
    }
  }
}
```

The root is the whole input module (`<stdin>:0:0`, a module has no source location of its own) — **including the transform script itself**. `debug-payload-root-tag` narrows it: it selects the operation whose `transform.target_tag` attribute has the given value. Tagging the function:

```bash
cd /Users/roy/study/mlir/transform/Ch1
sed -e '4s/-> tensor<512x512xf32> {/-> tensor<512x512xf32> attributes {transform.target_tag = "payload"} {/' \
    -e '/transform.yield/i\
    transform.debug.emit_remark_at %arg0, "payload root" : !transform.any_op' \
  stages/1-empty.mlir |
mlir-opt --pass-pipeline="builtin.module(transform-interpreter{
      debug-payload-root-tag=payload
      debug-bind-trailing-args=linalg.matmul,linalg.elemwise_binary})" \
  -o /dev/null
```

Real output **[9]**:

```text
<stdin>:2:1: remark: payload root
func.func @fc_relu(%lhs: tensor<512x512xf32>, %rhs: tensor<512x512xf32>,
^
<stdin>:2:1: note: see current operation: 
func.func @fc_relu(%arg0: tensor<512x512xf32>, %arg1: tensor<512x512xf32>, %arg2: tensor<512x512xf32>, %arg3: tensor<512x512xf32>) -> tensor<512x512xf32> attributes {transform.target_tag = "payload"} {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<512x512xf32>, tensor<512x512xf32>) outs(%arg3 : tensor<512x512xf32>) -> tensor<512x512xf32>
  %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%0, %arg2 : tensor<512x512xf32>, tensor<512x512xf32>) outs(%arg3 : tensor<512x512xf32>) -> tensor<512x512xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<max_signed>} ins(%1, %cst : tensor<512x512xf32>, f32) outs(%arg3 : tensor<512x512xf32>) -> tensor<512x512xf32>
  return %2 : tensor<512x512xf32>
}
```

Now `%arg0` is `@fc_relu` (`<stdin>:2:1`), and `debug-bind-trailing-args` searches for its operations inside that root only. Appendix A shows the other way to keep script and payload apart: separate files.

---

## 6. Specifying Transformations

With handles to the operations we want to transform, we can apply transformations. Let us first try tiling the matmul operation itself:

***stages/4-tile-matmul.mlir***
```mlir
...
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
       %arg0: !transform.any_op,
       %arg1: !transform.op<"linalg.matmul">,
       %arg2: !transform.op<"linalg.elemwise_binary">) {
    // The actual tiling transformation takes tile sizes as attributes.
    %tiled, %loop = transform.structured.tile_using_forall %arg1
                    tile_sizes [4, 32]
      : (!transform.op<"linalg.matmul">)
     -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
```

`transform.structured.tile_using_forall` ([documentation](https://mlir.llvm.org/docs/Dialects/Transform/#transformstructuredtile_using_forall-transformtileusingforallop)) takes the tile sizes as an attribute and returns two handles, in this order:

- a handle to the tiled operation — the same `linalg` operation, now working on a subset (slice) of the original data;
- a handle to the `scf.forall` "multi-for" loop around it, which iterates over the tiles and assembles the result tensor.

Running it with the same command as before produces the tiled code:

```bash
cd /Users/roy/study/mlir/transform/Ch1
mlir-opt stages/4-tile-matmul.mlir \
  --pass-pipeline="builtin.module(transform-interpreter{
      debug-bind-trailing-args=linalg.matmul,linalg.elemwise_binary})"
```

Real output **[10]**:

```text
#map = affine_map<(d0) -> (d0 * 4)>
#map1 = affine_map<(d0) -> (d0 * 32)>
module {
  func.func @fc_relu(%arg0: tensor<512x512xf32>, %arg1: tensor<512x512xf32>, %arg2: tensor<512x512xf32>, %arg3: tensor<512x512xf32>) -> tensor<512x512xf32> {
    %0 = scf.forall (%arg4, %arg5) in (128, 16) shared_outs(%arg6 = %arg3) -> (tensor<512x512xf32>) {
      %3 = affine.apply #map(%arg4)
      %4 = affine.apply #map1(%arg5)
      %extracted_slice = tensor.extract_slice %arg0[%3, 0] [4, 512] [1, 1] : tensor<512x512xf32> to tensor<4x512xf32>
      %extracted_slice_0 = tensor.extract_slice %arg1[0, %4] [512, 32] [1, 1] : tensor<512x512xf32> to tensor<512x32xf32>
      %extracted_slice_1 = tensor.extract_slice %arg6[%3, %4] [4, 32] [1, 1] : tensor<512x512xf32> to tensor<4x32xf32>
      %5 = linalg.matmul ins(%extracted_slice, %extracted_slice_0 : tensor<4x512xf32>, tensor<512x32xf32>) outs(%extracted_slice_1 : tensor<4x32xf32>) -> tensor<4x32xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %5 into %arg6[%3, %4] [4, 32] [1, 1] : tensor<4x32xf32> into tensor<512x512xf32>
      }
    }
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%0, %arg2 : tensor<512x512xf32>, tensor<512x512xf32>) outs(%arg3 : tensor<512x512xf32>) -> tensor<512x512xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<max_signed>} ins(%1, %cst : tensor<512x512xf32>, f32) outs(%arg3 : tensor<512x512xf32>) -> tensor<512x512xf32>
    return %2 : tensor<512x512xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op, %arg1: !transform.op<"linalg.matmul">, %arg2: !transform.op<"linalg.elemwise_binary">) {
      %tiled_op, %forall_op = transform.structured.tile_using_forall %arg1 tile_sizes [4, 32] : (!transform.op<"linalg.matmul">) -> (!transform.any_op, !transform.any_op)
      transform.yield 
    }
  }
}

```

How the tiled code maps back to chapter 0's tiling:

- **`scf.forall (%arg4, %arg5) in (128, 16)`**: the trip counts are the tensor size divided by the tile size, 512/4 = 128 and 512/32 = 16. The tiles are independent, so they may run in parallel.
- **`affine.apply #map(%arg4)` / `#map1(%arg5)`** scale the induction variables by the tile sizes (`d0 * 4`, `d0 * 32`) to get the tile's offsets.
- **Three `tensor.extract_slice`s**: a 4x512 row block of the left-hand side, a 512x32 column block of the right-hand side, and the 4x32 block of the output. The reduction dimension (512) is not tiled, so every tile sees it whole.
- **`linalg.matmul` on `tensor<4x512xf32>` × `tensor<512x32xf32>`**: the same operation on the slices.
- **`scf.forall.in_parallel { tensor.parallel_insert_slice ... }`** writes each 4x32 result back into the shared output `%arg6`; the loop's result `%0` is the full 512x512 tensor, which feeds the two elementwise operations, still untiled.
- The printer names the results `%tiled_op, %forall_op` in the script — the default names `tile_using_forall` gives them, confirming the order above.

Besides producing new handles, the tiling operation **consumes** its operand handle: `%arg1` is **invalidated** after this operation and must not be used again. Transform operations are required to mark each operand as either *consumed* or *read-only*. They usually consume an operand when the associated payload operations are erased or recreated (erased and created anew with a similar structure), because handles are essentially references to payload operations, and they would dangle once the payload is gone. Here the original 512x512 `linalg.matmul` no longer exists: the matmul in the output is a new operation inside the loop.

---

## 7. Handle Invalidation and Expensive Checks Mode

Undefined behavior is difficult to grapple with when it does happen, so the transform interpreter by default performs a set of additional, potentially expensive, checks that detect most undefined behavior in the transform IR. For example, using `%arg1` after it was consumed is undefined behavior: in a debug build it manifests as an assertion, in a release build likely as a segmentation fault — or, as section 7.3 shows, as silence.

### 7.1 Using a consumed handle

The first test case of `invalidation-1.mlir` does exactly that:

***test_Example/transform/Ch1/invalidation-1.mlir***
```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %arg0: !transform.any_op,
      // expected-note @below {{handle to invalidated ops}}
      %arg1: !transform.op<"linalg.matmul">,
      %arg2: !transform.op<"linalg.elemwise_binary">) {
    // The actual tiling transformation takes tile sizes as attributes.
    // expected-note @below {{invalidated by this transform op that consumes its operand #0 and invalidates all handles to payload IR entities associated with this operand and entities nested in them}}
    %tiled, %loop = transform.structured.tile_using_forall %arg1 tile_sizes [4, 32]
        : (!transform.op<"linalg.matmul">) -> (!transform.any_op, !transform.any_op)

    // This is trying to use an invalidated handle leading to undefined behavior.
    // expected-error @below {{uses a handle invalidated by a previously executed transform op}}
    transform.debug.emit_remark_at %arg1, "remark" : !transform.op<"linalg.matmul">
    transform.yield
  }
}
...
```

(The `expected-...` comments are for `--verify-diagnostics`, section 7.3; the payload `@fc_relu` follows the script in this file.)

### 7.2 Aliasing handles

Some operations, such as `transform.cast`, do *not* consume their operand, because they don't erase the payload operation. So what happens if we use the handle `transform.cast` produced, after the original was consumed? That is the second test case:

***test_Example/transform/Ch1/invalidation-1.mlir***
```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %arg0: !transform.any_op,
      %arg1: !transform.op<"linalg.matmul">,
      %arg2: !transform.op<"linalg.elemwise_binary">) {
    // We can cast one type to another as long as operations are compatible
    // with both types. This creates "aliasing" handles.
    // expected-note @below {{handle to invalidated ops}}
    %casted = transform.cast %arg1 : !transform.op<"linalg.matmul"> to
        !transform.any_op

    // The actual tiling transformation takes tile sizes as attributes.
    // expected-note @below {{invalidated by this transform op that consumes its operand #0 and invalidates all handles to payload IR entities associated with this operand and entities nested in them}}
    %tiled, %loop = transform.structured.tile_using_forall %arg1 tile_sizes [4, 32]
      : (!transform.op<"linalg.matmul">) -> (!transform.any_op, !transform.any_op)

    // Consuming an operand invalidates the consumed handle and any other handle that is
    // associated with the same payload operations, or payload operations nested in them.
    // expected-error @below {{uses a handle invalidated by a previously executed transform op}}
    transform.debug.emit_remark_at %casted, "remark"
      : !transform.any_op
    transform.yield
  }
}
...
```

Both `%arg1` and `%casted` reference the same payload operation. Extending the reference analogy, these references **alias**. Naturally, when the payload operation is erased, all references to it dangle. The same holds for handles: consuming an operand invalidates the operand handle *and any other handle associated with any of the same payload operations*. The rule is also **recursive in the payload**: a handle associated with a payload operation *nested* in the erased one is invalidated too, because erasing an operation erases its regions and everything they contain. The expensive-checks mode handles this case as well.

The file holds the two cases separated by `// -----`; `--split-input-file` runs each as its own module:

```bash
cd /Users/roy/study/mlir/transform/Ch1
mlir-opt ../../test_Example/transform/Ch1/invalidation-1.mlir \
  --pass-pipeline="builtin.module(transform-interpreter{
      debug-bind-trailing-args=linalg.matmul,linalg.elemwise_binary},
      canonicalize,cse,symbol-dce)" \
  --split-input-file -o /dev/null
```

Real output **[11]** (exit status 1):

```text
within split at ../../test_Example/transform/Ch1/invalidation-1.mlir:1 offset :27:5: error: op uses a handle invalidated by a previously executed transform op
    transform.debug.emit_remark_at %arg1, "remark" : !transform.op<"linalg.matmul">
    ^
within split at ../../test_Example/transform/Ch1/invalidation-1.mlir:1 offset :18:7: note: handle to invalidated ops
      %arg1: !transform.op<"linalg.matmul">,
      ^
within split at ../../test_Example/transform/Ch1/invalidation-1.mlir:1 offset :22:21: note: invalidated by this transform op that consumes its operand #0 and invalidates all handles to payload IR entities associated with this operand and entities nested in them
    %tiled, %loop = transform.structured.tile_using_forall %arg1 tile_sizes [4, 32]
                    ^
within split at ../../test_Example/transform/Ch1/invalidation-1.mlir:1 offset :38:13: note: ancestor payload op
  %matmul = linalg.matmul ins(%lhs, %rhs: tensor<512x512xf32>, tensor<512x512xf32>)
            ^
within split at ../../test_Example/transform/Ch1/invalidation-1.mlir:1 offset :38:13: note: nested payload op
within split at ../../test_Example/transform/Ch1/invalidation-1.mlir:54 offset :22:5: error: op uses a handle invalidated by a previously executed transform op
    transform.debug.emit_remark_at %casted, "remark"
    ^
within split at ../../test_Example/transform/Ch1/invalidation-1.mlir:54 offset :11:15: note: handle to invalidated ops
    %casted = transform.cast %arg1 : !transform.op<"linalg.matmul"> to
              ^
within split at ../../test_Example/transform/Ch1/invalidation-1.mlir:54 offset :16:21: note: invalidated by this transform op that consumes its operand #0 and invalidates all handles to payload IR entities associated with this operand and entities nested in them
    %tiled, %loop = transform.structured.tile_using_forall %arg1 tile_sizes [4, 32]
                    ^
within split at ../../test_Example/transform/Ch1/invalidation-1.mlir:54 offset :34:13: note: ancestor payload op
  %matmul = linalg.matmul ins(%lhs, %rhs: tensor<512x512xf32>, tensor<512x512xf32>)
            ^
within split at ../../test_Example/transform/Ch1/invalidation-1.mlir:54 offset :34:13: note: nested payload op
```

Each case gets one error and four notes (line numbers are relative to the split, which starts at the file line given after `within split at`):

- **error** at the use of the invalidated handle (`emit_remark_at %arg1`, then `emit_remark_at %casted`);
- **note "handle to invalidated ops"** at the definition of that handle — the block argument `%arg1`, or the `transform.cast` that created the alias `%casted`;
- **note "invalidated by this transform op ..."** at the consuming `tile_using_forall`, spelling out the rule: it consumes operand #0 and invalidates all handles to the associated payload entities *and the entities nested in them*;
- **notes "ancestor payload op" / "nested payload op"** at the payload operation involved — here both are the matmul itself (the consumed op is the ancestor; the op the invalidated handle pointed to is nested in it, or, as here, is the same op).

### 7.3 Turning the checks off, and checking the diagnostics

When compile-time performance matters and the script is stable, the expensive checks can be disabled with the pass option `disable-expensive-checks` (or the corresponding flag of the `TransformOptions` passed to `applyTransforms`). On the same file:

```bash
cd /Users/roy/study/mlir/transform/Ch1
mlir-opt ../../test_Example/transform/Ch1/invalidation-1.mlir \
  --pass-pipeline="builtin.module(transform-interpreter{
      debug-bind-trailing-args=linalg.matmul,linalg.elemwise_binary
      disable-expensive-checks})" \
  --split-input-file -o /dev/null
```

Real output **[12]**: none (exit status 0).

This release build of MLIR did not crash — it did nothing visible. The interpreter had already dropped the consumed handle's payload, so `emit_remark_at` found nothing to remark on, and the run "succeeded". That is one possible face of undefined behavior; it can as well be a crash or a wrong transformation. **Without the checks, a use-after-consume bug can pass silently, so keep them on until the script is known to be correct.**

The test file states the expected diagnostics in `expected-error` / `expected-note` comments; `--verify-diagnostics` turns the run into a test that succeeds only if exactly those diagnostics appear (this is the file's `RUN` line):

```bash
cd /Users/roy/study/mlir/transform/Ch1
mlir-opt ../../test_Example/transform/Ch1/invalidation-1.mlir \
  --pass-pipeline="builtin.module(transform-interpreter{
      debug-bind-trailing-args=linalg.matmul,linalg.elemwise_binary},
      canonicalize,cse,symbol-dce)" \
  --split-input-file --verify-diagnostics -o /dev/null
echo "mlir-opt exit status: $?"
```

Real output **[13]**:

```text
mlir-opt exit status: 0
```

Every expected diagnostic was produced, and nothing else, so the errors are consumed by the verifier and the exit status is 0.

---

## 8. Chaining Transformations with Handles

Back to the script: we have tiled the matrix multiplication, but we also want to tile and fuse the elementwise operations. The typical way to do this in the structured-operations paradigm is to **tile the last operation** of an acyclic dataflow graph, then **progressively fuse the operations that produce its operands**. This removes the need to tile every operation explicitly, because fusion adapts the producers' sizes to what each tile needs and can inject recomputation if desired (chapter 0, "Producer/Consumer Fusion"). So instead of tiling the matmul, we tile the last operation in the chain — the ReLU `max` — and fuse the preceding operations into the loop that tiling produced:

***stages/5-tile-fuse.mlir***
```mlir
...
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
       %arg0: !transform.any_op,
       %arg1: !transform.op<"linalg.matmul">,
       %arg2: !transform.op<"linalg.elemwise_binary">) {
    // Since the %arg2 handle is associated with both elementwise operations,
    // we need to split it into two handles so we can target only the second
    // elementwise operation.
    %add, %max = transform.split_handle %arg2
        : (!transform.op<"linalg.elemwise_binary">)
        -> (!transform.any_op, !transform.any_op)

    // The actual tiling transformation takes tile sizes as attributes. It
    // produces a handle to the loop generated during tiling.
    %tiled_max, %loop =
        transform.structured.tile_using_forall %max tile_sizes [8, 32]
          : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // We can now fuse the other operations into the loop. Here, we fuse
    // operations one by one. This requires the operation that is being fused to
    // define the value used within the loop, so the order of such fusions is
    // important. We could also use "transform.merge_handles" to obtain a single
    // handle to all operations and give it to `fuse_into_containing_op` that
    // would take care of the ordering in this case.
    %add_fused, %loop_0 =
        transform.structured.fuse_into_containing_op %add into %loop
          : (!transform.any_op, !transform.any_op)
            -> (!transform.any_op, !transform.any_op)
    %matmul_fused, %loop_1 =
        transform.structured.fuse_into_containing_op %arg1 into %loop_0
          : (!transform.op<"linalg.matmul">, !transform.any_op)
            -> (!transform.any_op, !transform.any_op)

    transform.yield
  }
}
```

Step by step:

- **`transform.split_handle %arg2`** turns the two-element handle into two one-element handles, in payload order: `%add` and `%max`. This is how the script tells two operations of the same kind apart.
- **`tile_using_forall %max tile_sizes [8, 32]`** tiles the ReLU into 8x32 tiles, producing `%loop`.
- **`fuse_into_containing_op %add into %loop`** finds the uses of the add's result inside `%loop` (a `tensor.extract_slice` of it), computes which part of the add is needed for one tile, and puts a tile of the add inside the loop instead. It returns a handle to the fused op and to the (possibly new) loop.
- **`fuse_into_containing_op %arg1 into %loop_0`** does the same for the matmul, which now feeds the fused add inside the loop.

Running it:

```bash
cd /Users/roy/study/mlir/transform/Ch1
mlir-opt stages/5-tile-fuse.mlir \
  --pass-pipeline="builtin.module(transform-interpreter{
      debug-bind-trailing-args=linalg.matmul,linalg.elemwise_binary})"
```

Real output **[14]**:

```text
#map = affine_map<(d0) -> (d0 * 8)>
#map1 = affine_map<(d0) -> (d0 * 32)>
module {
  func.func @fc_relu(%arg0: tensor<512x512xf32>, %arg1: tensor<512x512xf32>, %arg2: tensor<512x512xf32>, %arg3: tensor<512x512xf32>) -> tensor<512x512xf32> {
    %0 = linalg.matmul ins(%arg0, %arg1 : tensor<512x512xf32>, tensor<512x512xf32>) outs(%arg3 : tensor<512x512xf32>) -> tensor<512x512xf32>
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%0, %arg2 : tensor<512x512xf32>, tensor<512x512xf32>) outs(%arg3 : tensor<512x512xf32>) -> tensor<512x512xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %2 = scf.forall (%arg4, %arg5) in (64, 16) shared_outs(%arg6 = %arg3) -> (tensor<512x512xf32>) {
      %3 = affine.apply #map(%arg4)
      %4 = affine.apply #map1(%arg5)
      %5 = affine.apply #map(%arg4)
      %6 = affine.apply #map1(%arg5)
      %7 = affine.apply #map(%arg4)
      %8 = affine.apply #map1(%arg5)
      %9 = affine.apply #map(%arg4)
      %10 = affine.apply #map1(%arg5)
      %11 = affine.apply #map(%arg4)
      %12 = affine.apply #map1(%arg5)
      %13 = affine.apply #map(%arg4)
      %14 = affine.apply #map1(%arg5)
      %extracted_slice = tensor.extract_slice %arg0[%11, 0] [8, 512] [1, 1] : tensor<512x512xf32> to tensor<8x512xf32>
      %extracted_slice_0 = tensor.extract_slice %arg1[0, %12] [512, 32] [1, 1] : tensor<512x512xf32> to tensor<512x32xf32>
      %extracted_slice_1 = tensor.extract_slice %arg3[%13, %14] [8, 32] [1, 1] : tensor<512x512xf32> to tensor<8x32xf32>
      %15 = linalg.matmul ins(%extracted_slice, %extracted_slice_0 : tensor<8x512xf32>, tensor<512x32xf32>) outs(%extracted_slice_1 : tensor<8x32xf32>) -> tensor<8x32xf32>
      %extracted_slice_2 = tensor.extract_slice %arg2[%7, %8] [8, 32] [1, 1] : tensor<512x512xf32> to tensor<8x32xf32>
      %extracted_slice_3 = tensor.extract_slice %arg3[%9, %10] [8, 32] [1, 1] : tensor<512x512xf32> to tensor<8x32xf32>
      %16 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%15, %extracted_slice_2 : tensor<8x32xf32>, tensor<8x32xf32>) outs(%extracted_slice_3 : tensor<8x32xf32>) -> tensor<8x32xf32>
      %extracted_slice_4 = tensor.extract_slice %arg6[%3, %4] [8, 32] [1, 1] : tensor<512x512xf32> to tensor<8x32xf32>
      %17 = linalg.elemwise_binary {fun = #linalg.binary_fn<max_signed>} ins(%16, %cst : tensor<8x32xf32>, f32) outs(%extracted_slice_4 : tensor<8x32xf32>) -> tensor<8x32xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %17 into %arg6[%3, %4] [8, 32] [1, 1] : tensor<8x32xf32> into tensor<512x512xf32>
      }
    }
    return %2 : tensor<512x512xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op, %arg1: !transform.op<"linalg.matmul">, %arg2: !transform.op<"linalg.elemwise_binary">) {
      %0:2 = transform.split_handle %arg2 : (!transform.op<"linalg.elemwise_binary">) -> (!transform.any_op, !transform.any_op)
      %tiled_op, %forall_op = transform.structured.tile_using_forall %0#1 tile_sizes [8, 32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
      %fused_op, %new_containing_op = transform.structured.fuse_into_containing_op %0#0 into %forall_op : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
      %fused_op_0, %new_containing_op_1 = transform.structured.fuse_into_containing_op %arg1 into %new_containing_op : (!transform.op<"linalg.matmul">, !transform.any_op) -> (!transform.any_op, !transform.any_op)
      transform.yield 
    }
  }
}

```

This achieves the desired tiling and fusion: inside the `scf.forall` over (64, 16) tiles, an 8x32 tile of the matmul feeds an 8x32 tile of the add, which feeds the 8x32 tile of the max. Two things look untidy:

- **The original 512x512 `linalg.matmul` and the add are still there**, at the top of the function. Fusion does not erase the producer; after the max was tiled, nothing uses the original add's result any more, so both are dead code.
- **There are six pairs of identical `affine.apply`s**: each slice created by tiling and fusion computes its own offsets.

Transformations leave such cleanup to the usual passes. Appending `canonicalize,cse` to the pipeline removes both:

```bash
cd /Users/roy/study/mlir/transform/Ch1
mlir-opt stages/5-tile-fuse.mlir \
  --pass-pipeline="builtin.module(transform-interpreter{
      debug-bind-trailing-args=linalg.matmul,linalg.elemwise_binary},
      canonicalize,cse)"
```

Real output **[15]**:

```text
#map = affine_map<(d0) -> (d0 * 8)>
#map1 = affine_map<(d0) -> (d0 * 32)>
module {
  func.func @fc_relu(%arg0: tensor<512x512xf32>, %arg1: tensor<512x512xf32>, %arg2: tensor<512x512xf32>, %arg3: tensor<512x512xf32>) -> tensor<512x512xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = scf.forall (%arg4, %arg5) in (64, 16) shared_outs(%arg6 = %arg3) -> (tensor<512x512xf32>) {
      %1 = affine.apply #map(%arg4)
      %2 = affine.apply #map1(%arg5)
      %extracted_slice = tensor.extract_slice %arg0[%1, 0] [8, 512] [1, 1] : tensor<512x512xf32> to tensor<8x512xf32>
      %extracted_slice_0 = tensor.extract_slice %arg1[0, %2] [512, 32] [1, 1] : tensor<512x512xf32> to tensor<512x32xf32>
      %extracted_slice_1 = tensor.extract_slice %arg3[%1, %2] [8, 32] [1, 1] : tensor<512x512xf32> to tensor<8x32xf32>
      %3 = linalg.matmul ins(%extracted_slice, %extracted_slice_0 : tensor<8x512xf32>, tensor<512x32xf32>) outs(%extracted_slice_1 : tensor<8x32xf32>) -> tensor<8x32xf32>
      %extracted_slice_2 = tensor.extract_slice %arg2[%1, %2] [8, 32] [1, 1] : tensor<512x512xf32> to tensor<8x32xf32>
      %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%3, %extracted_slice_2 : tensor<8x32xf32>, tensor<8x32xf32>) outs(%extracted_slice_1 : tensor<8x32xf32>) -> tensor<8x32xf32>
      %extracted_slice_3 = tensor.extract_slice %arg6[%1, %2] [8, 32] [1, 1] : tensor<512x512xf32> to tensor<8x32xf32>
      %5 = linalg.elemwise_binary {fun = #linalg.binary_fn<max_signed>} ins(%4, %cst : tensor<8x32xf32>, f32) outs(%extracted_slice_3 : tensor<8x32xf32>) -> tensor<8x32xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %5 into %arg6[%1, %2] [8, 32] [1, 1] : tensor<8x32xf32> into tensor<512x512xf32>
      }
    }
    return %0 : tensor<512x512xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op, %arg1: !transform.op<"linalg.matmul">, %arg2: !transform.op<"linalg.elemwise_binary">) {
      %0:2 = transform.split_handle %arg2 : (!transform.op<"linalg.elemwise_binary">) -> (!transform.any_op, !transform.any_op)
      %tiled_op, %forall_op = transform.structured.tile_using_forall %0#1 tile_sizes [8, 32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
      %fused_op, %new_containing_op = transform.structured.fuse_into_containing_op %0#0 into %forall_op : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
      %fused_op_0, %new_containing_op_1 = transform.structured.fuse_into_containing_op %arg1 into %new_containing_op : (!transform.op<"linalg.matmul">, !transform.any_op) -> (!transform.any_op, !transform.any_op)
      transform.yield 
    }
  }
}

```

Now the function is one loop nest: the dead 512x512 operations are gone (canonicalization erases ops without uses or side effects), and CSE (common subexpression elimination) merged the offsets into `%1` and `%2`. The slice `%extracted_slice_1` of the output serves as the init of both the matmul and the add. The ReLU's tile is the only one written back to the shared output `%arg6`.

#### Fusion order matters

The script comment says fusion requires the fused operation to define a value used within the loop. Fusing the matmul first — before the add is inside the loop, the matmul's only user is the add *outside* the loop — breaks that rule (the `sed` swaps the operands, and their types, of the two fusion steps):

```bash
cd /Users/roy/study/mlir/transform/Ch1
sed -e '47,48{s/%add/%arg1/;s/(!transform.any_op, !transform.any_op)$/(!transform.op<"linalg.matmul">, !transform.any_op)/;}' \
    -e '51,52{s/%arg1/%add/;s/(!transform.op<"linalg.matmul">, /(!transform.any_op, /;}' \
  stages/5-tile-fuse.mlir |
mlir-opt --pass-pipeline="builtin.module(transform-interpreter{
      debug-bind-trailing-args=linalg.matmul,linalg.elemwise_binary})" \
  -o /dev/null
```

Real output **[16]** (exit status 1):

```text
<stdin>:47:9: error: could not find next producer to fuse into container
        transform.structured.fuse_into_containing_op %arg1 into %loop
        ^
<stdin>:16:13: note: containing op
  %relued = linalg.elemwise_binary { fun = #linalg.binary_fn<max_signed> }
            ^
```

#### Letting `merge_handles` order the fusions

The alternative the comment mentions: merge both producers into one handle and fuse them with a single `fuse_into_containing_op`, which then picks the order itself. `merge_handles` needs operands of one type, so the matmul handle is cast to `!transform.any_op` first; the matmul is even listed first, the order that failed above. The `sed` replaces the two fusion steps (lines 46–53), and `diff` compares the result with the previous run:

```bash
cd /Users/roy/study/mlir/transform/Ch1
P="builtin.module(transform-interpreter{debug-bind-trailing-args=linalg.matmul,linalg.elemwise_binary},canonicalize,cse)"
sed -e '46,53c\
    %mm = transform.cast %arg1 : !transform.op<"linalg.matmul"> to !transform.any_op\
    %producers = transform.merge_handles %mm, %add : !transform.any_op\
    %fused, %loop_0 = transform.structured.fuse_into_containing_op %producers into %loop\
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)' \
  stages/5-tile-fuse.mlir | mlir-opt --pass-pipeline="$P" > /tmp/ch1-merged.mlir
mlir-opt stages/5-tile-fuse.mlir --pass-pipeline="$P" | diff - /tmp/ch1-merged.mlir
rm /tmp/ch1-merged.mlir
```

Real output **[17]**:

```text
27,28c27,29
<       %fused_op, %new_containing_op = transform.structured.fuse_into_containing_op %0#0 into %forall_op : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
<       %fused_op_0, %new_containing_op_1 = transform.structured.fuse_into_containing_op %arg1 into %new_containing_op : (!transform.op<"linalg.matmul">, !transform.any_op) -> (!transform.any_op, !transform.any_op)
---
>       %1 = transform.cast %arg1 : !transform.op<"linalg.matmul"> to !transform.any_op
>       %2 = transform.merge_handles %1, %0#0 : !transform.any_op
>       %fused_op, %new_containing_op = transform.structured.fuse_into_containing_op %2 into %forall_op : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
```

The only differences are in the transform script, as printed after the payload (the block's exit status is that of the final `rm`). The payload IR is identical: `fuse_into_containing_op` repeatedly picks, from its handle, a producer whose result is used inside the loop, so the order of the handle's elements does not matter. **Handles turn "which operation" into data, so a script can name one of two identical-looking ops, or hand a whole set to a transformation that orders it itself.**

---

## 9. More Handle Invalidation

Finally, assume there is an efficient **microkernel** — or a hardware instruction exposed as an intrinsic function — for a 4x4 matrix multiplication (plus the bias add). To use it, we tile the fused operations down to that size, then **outline** the 4x4 computation into a function; the resulting call can then be replaced by a call to the microkernel. This is the complete script of `sequence.mlir`, which also declares the microkernel:

***test_Example/transform/Ch1/sequence.mlir***
```mlir
// Declaration of the "microkernel" function that we will be targeting.
func.func private @microkernel(
    %lhs: tensor<4x512xf32>,
    %rhs: tensor<512x4xf32>,
    %bias: tensor<4x4xf32>,
    %init: tensor<4x4xf32>,
    %output: tensor<4x4xf32>) -> tensor<4x4xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %arg0: !transform.any_op,
      %arg1: !transform.op<"linalg.matmul">,
      %arg2: !transform.op<"linalg.elemwise_binary">) {
    // Since the %arg2 handle is associated with both elementwise operations,
    // we need to split it into two handles so we can target only the second
    // elementwise operation.
    %add, %max = transform.split_handle %arg2 : (!transform.op<"linalg.elemwise_binary">)
        -> (!transform.any_op, !transform.any_op)
  
    // The actual tiling transformation takes tile sizes as attributes. It produces a
    // handle to the loop generated during tiling.
    %tiled, %loop = transform.structured.tile_using_forall %max tile_sizes [8, 32]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  
    // We can now fuse the other operations into the loop. Here, we fuse
    // operations one-by-one. This requires the operation that is being fused
    // to define the value used within the loop, so the order of such fusions
    // is important. We could also use "transform.merge_handles" to obtain
    // a single handle to all operations and give it to `fuse_into_containing_op`
    // that would take care of the ordering in this case.
    %add_fused, %loop2 = transform.structured.fuse_into_containing_op %add into %loop
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %matmul_fused, %loop3 = transform.structured.fuse_into_containing_op %arg1 into %loop2
        : (!transform.op<"linalg.matmul">, !transform.any_op) -> (!transform.any_op, !transform.any_op)
  
    // Tile again to get the desired size. Note that this time this tiles the
    // "add" operation and fuses matmul into the loop, but doesn't affect the
    // "max" operation. This illustrates the precise targeting with the transform
    // dialect. Otherwise, it is difficult to differentiate "add" and "max", both
    // of which having the same kind.
    %tiled_second, %loop_second = transform.structured.tile_using_forall %add_fused tile_sizes [4, 4]     // (1)
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %matmul_fused_2, %loop_second_2 =
        transform.structured.fuse_into_containing_op %matmul_fused into %loop_second                      // (2)
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
  
    // Since outlining is currently only implemented for region-holding operations
    // such as loops, use tiling to size 1 to materialize the outer loop that is
    // going to be outlined.
    %_0, %loop_third = transform.structured.tile_using_forall %tiled_second tile_sizes [1]                // (3)
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %_1, %outline_target = transform.structured.fuse_into_containing_op %matmul_fused_2 into %loop_third  // (4)
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %func, %call = transform.loop.outline %outline_target {func_name = "outlined"}                        // (5)
        : (!transform.any_op) -> (!transform.any_op, !transform.op<"func.call">)
  
    transform.yield
  }
```

The first part is section 8's script. Then:

- **(1)** tiles the *fused add* — `%add_fused`, the 8x32 add inside the first loop — into 4x4 tiles. The max is not affected: this is the precise targeting the comment describes, impossible to express by operation name since both are `linalg.elemwise_binary`.
- **(2)** fuses the 8x32 matmul tile into that new 4x4 loop.
- **(3)** tiles the 4x4 add again with tile size `[1]`: outlining is currently implemented only for region-holding operations such as loops, so this materializes a loop over the 4 rows that can be outlined as a whole.
- **(4)** fuses the matmul into that loop too, so the loop computes matmul + add for a 4x4 tile.
- **(5)** `transform.loop.outline` moves the loop into a new function `@outlined` and replaces it with a call; it returns handles to the function and to the `func.call`.

Running the file's `RUN` pipeline (interpreter, then `canonicalize`, `cse` and `symbol-dce`):

```bash
cd /Users/roy/study/mlir/transform/Ch1
mlir-opt ../../test_Example/transform/Ch1/sequence.mlir \
  --pass-pipeline="builtin.module(transform-interpreter{
      debug-bind-trailing-args=linalg.matmul,linalg.elemwise_binary},
      canonicalize,cse,symbol-dce)"
```

Real output **[18]**:

```text
#map = affine_map<(d0) -> (d0 * 8)>
#map1 = affine_map<(d0) -> (d0 * 32)>
#map2 = affine_map<(d0) -> (d0 * 4)>
module {
  func.func @outlined(%arg0: tensor<4x512xf32>, %arg1: tensor<512x4xf32>, %arg2: tensor<4x4xf32>, %arg3: tensor<4x4xf32>, %arg4: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = scf.forall (%arg5) in (4) shared_outs(%arg6 = %arg4) -> (tensor<4x4xf32>) {
      %extracted_slice = tensor.extract_slice %arg0[%arg5, 0] [1, 512] [1, 1] : tensor<4x512xf32> to tensor<1x512xf32>
      %extracted_slice_0 = tensor.extract_slice %arg2[%arg5, 0] [1, 4] [1, 1] : tensor<4x4xf32> to tensor<1x4xf32>
      %1 = linalg.matmul ins(%extracted_slice, %arg1 : tensor<1x512xf32>, tensor<512x4xf32>) outs(%extracted_slice_0 : tensor<1x4xf32>) -> tensor<1x4xf32>
      %extracted_slice_1 = tensor.extract_slice %arg3[%arg5, 0] [1, 4] [1, 1] : tensor<4x4xf32> to tensor<1x4xf32>
      %extracted_slice_2 = tensor.extract_slice %arg6[%arg5, 0] [1, 4] [1, 1] : tensor<4x4xf32> to tensor<1x4xf32>
      %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%1, %extracted_slice_1 : tensor<1x4xf32>, tensor<1x4xf32>) outs(%extracted_slice_2 : tensor<1x4xf32>) -> tensor<1x4xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %2 into %arg6[%arg5, 0] [1, 4] [1, 1] : tensor<1x4xf32> into tensor<4x4xf32>
      }
    }
    return %0 : tensor<4x4xf32>
  }
  func.func @fc_relu(%arg0: tensor<512x512xf32>, %arg1: tensor<512x512xf32>, %arg2: tensor<512x512xf32>, %arg3: tensor<512x512xf32>) -> tensor<512x512xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = scf.forall (%arg4, %arg5) in (64, 16) shared_outs(%arg6 = %arg3) -> (tensor<512x512xf32>) {
      %1 = affine.apply #map(%arg4)
      %2 = affine.apply #map1(%arg5)
      %extracted_slice = tensor.extract_slice %arg0[%1, 0] [8, 512] [1, 1] : tensor<512x512xf32> to tensor<8x512xf32>
      %extracted_slice_0 = tensor.extract_slice %arg1[0, %2] [512, 32] [1, 1] : tensor<512x512xf32> to tensor<512x32xf32>
      %extracted_slice_1 = tensor.extract_slice %arg3[%1, %2] [8, 32] [1, 1] : tensor<512x512xf32> to tensor<8x32xf32>
      %extracted_slice_2 = tensor.extract_slice %arg2[%1, %2] [8, 32] [1, 1] : tensor<512x512xf32> to tensor<8x32xf32>
      %3 = scf.forall (%arg7, %arg8) in (2, 8) shared_outs(%arg9 = %extracted_slice_1) -> (tensor<8x32xf32>) {
        %5 = affine.apply #map2(%arg7)
        %6 = affine.apply #map2(%arg8)
        %extracted_slice_4 = tensor.extract_slice %extracted_slice[%5, 0] [4, 512] [1, 1] : tensor<8x512xf32> to tensor<4x512xf32>
        %extracted_slice_5 = tensor.extract_slice %extracted_slice_0[0, %6] [512, 4] [1, 1] : tensor<512x32xf32> to tensor<512x4xf32>
        %extracted_slice_6 = tensor.extract_slice %extracted_slice_1[%5, %6] [4, 4] [1, 1] : tensor<8x32xf32> to tensor<4x4xf32>
        %extracted_slice_7 = tensor.extract_slice %extracted_slice_2[%5, %6] [4, 4] [1, 1] : tensor<8x32xf32> to tensor<4x4xf32>
        %extracted_slice_8 = tensor.extract_slice %arg9[%5, %6] [4, 4] [1, 1] : tensor<8x32xf32> to tensor<4x4xf32>
        %7 = func.call @outlined(%extracted_slice_4, %extracted_slice_5, %extracted_slice_6, %extracted_slice_7, %extracted_slice_8) : (tensor<4x512xf32>, tensor<512x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %7 into %arg9[%5, %6] [4, 4] [1, 1] : tensor<4x4xf32> into tensor<8x32xf32>
        }
      }
      %extracted_slice_3 = tensor.extract_slice %arg6[%1, %2] [8, 32] [1, 1] : tensor<512x512xf32> to tensor<8x32xf32>
      %4 = linalg.elemwise_binary {fun = #linalg.binary_fn<max_signed>} ins(%3, %cst : tensor<8x32xf32>, f32) outs(%extracted_slice_3 : tensor<8x32xf32>) -> tensor<8x32xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %4 into %arg6[%1, %2] [8, 32] [1, 1] : tensor<8x32xf32> into tensor<512x512xf32>
      }
    }
    return %0 : tensor<512x512xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op, %arg1: !transform.op<"linalg.matmul">, %arg2: !transform.op<"linalg.elemwise_binary">) {
      %0:2 = transform.split_handle %arg2 : (!transform.op<"linalg.elemwise_binary">) -> (!transform.any_op, !transform.any_op)
      %tiled_op, %forall_op = transform.structured.tile_using_forall %0#1 tile_sizes [8, 32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
      %fused_op, %new_containing_op = transform.structured.fuse_into_containing_op %0#0 into %forall_op : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
      %fused_op_0, %new_containing_op_1 = transform.structured.fuse_into_containing_op %arg1 into %new_containing_op : (!transform.op<"linalg.matmul">, !transform.any_op) -> (!transform.any_op, !transform.any_op)
      %tiled_op_2, %forall_op_3 = transform.structured.tile_using_forall %fused_op tile_sizes [4, 4] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
      %fused_op_4, %new_containing_op_5 = transform.structured.fuse_into_containing_op %fused_op_0 into %forall_op_3 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
      %tiled_op_6, %forall_op_7 = transform.structured.tile_using_forall %tiled_op_2 tile_sizes [1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
      %fused_op_8, %new_containing_op_9 = transform.structured.fuse_into_containing_op %fused_op_4 into %forall_op_7 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
      %function, %call = transform.loop.outline %new_containing_op_9 {func_name = "outlined"} : (!transform.any_op) -> (!transform.any_op, !transform.op<"func.call">)
      transform.yield 
    }
  }
}

```

The final code, from the outside in:

- **`@fc_relu`**: the outer `scf.forall` over (64, 16) tiles of 8x32, as in section 8. Inside it, a second `scf.forall` over (2, 8) tiles of 4x4 computes matmul + add by calling **`@outlined`** on five slices: the 4x512 rows of the left-hand side, the 512x4 columns of the right-hand side, the 4x4 output init (for the matmul), the 4x4 bias, and the 4x4 shared output (for the add). The 8x32 **max** stays at the 8x32 level, after the inner loop.
- **`@outlined`**: the loop from step (3), over the 4 rows (`scf.forall (%arg5) in (4)`), with a 1x512 × 512x4 matmul and a 1x4 add per row. The five function arguments are the values the loop used from outside.
- **`@microkernel` is gone**: it is private and unused, so `symbol-dce` (symbol dead code elimination) removed it. Replacing the call with a call to it is the job of the next chapters.

The file's `CHECK` lines verify exactly this structure, with `FileCheck` (from Homebrew LLVM 20) reading the output:

```bash
cd /Users/roy/study/mlir/transform/Ch1
mlir-opt ../../test_Example/transform/Ch1/sequence.mlir \
  --pass-pipeline="builtin.module(transform-interpreter{
      debug-bind-trailing-args=linalg.matmul,linalg.elemwise_binary},
      canonicalize,cse,symbol-dce)" |
FileCheck ../../test_Example/transform/Ch1/sequence.mlir
echo "FileCheck exit status: $?"
```

Real output **[19]**:

```text
FileCheck exit status: 0
```

### 9.1 Invalidation of nested operations

This additional transformation also illustrates **handle invalidation for nested operations**. `transform.loop.outline` consumes the handle to the loop, which invalidates it *and all handles to operations nested in it*, such as the handle to the fused matmul inside it. Using such a handle afterwards is undefined behavior. (Strictly, this form of outlining would not need to consume its operand: the implementation only *moves* the region, without recreating the operations. But the author of the transformation chose to invalidate the handle anyway.)

`invalidation-2.mlir` is the same script with the result of step (4) named `%f` — the handle to the matmul fused into the loop — used after outlining:

***test_Example/transform/Ch1/invalidation-2.mlir***
```mlir
  %_0, %loop_third = transform.structured.tile_using_forall %tiled_second tile_sizes [1]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  // expected-note @below {{handle to invalidated ops}}
  %f, %outline_target = transform.structured.fuse_into_containing_op %matmul_fused_2 into %loop_third
      : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

  // expected-note @below {{invalidated by this transform op that consumes its operand #0 and invalidates all handles to payload IR entities associated with this operand and entities nested in them}}
  %func, %call = transform.loop.outline %outline_target {func_name = "outlined"}
      : (!transform.any_op) -> (!transform.any_op, !transform.op<"func.call">)

  // expected-error @below {{uses a handle invalidated by a previously executed transform op}}
  transform.debug.emit_remark_at %f, "fused" : !transform.any_op
```

```bash
cd /Users/roy/study/mlir/transform/Ch1
mlir-opt ../../test_Example/transform/Ch1/invalidation-2.mlir \
  --pass-pipeline="builtin.module(transform-interpreter{
      debug-bind-trailing-args=linalg.matmul,linalg.elemwise_binary},
      canonicalize,cse,symbol-dce)" \
  -o /dev/null
```

Real output **[20]** (exit status 1):

```text
../../test_Example/transform/Ch1/invalidation-2.mlir:98:3: error: op uses a handle invalidated by a previously executed transform op
  transform.debug.emit_remark_at %f, "fused" : !transform.any_op
  ^
../../test_Example/transform/Ch1/invalidation-2.mlir:90:25: note: handle to invalidated ops
  %f, %outline_target = transform.structured.fuse_into_containing_op %matmul_fused_2 into %loop_third
                        ^
../../test_Example/transform/Ch1/invalidation-2.mlir:94:18: note: invalidated by this transform op that consumes its operand #0 and invalidates all handles to payload IR entities associated with this operand and entities nested in them
  %func, %call = transform.loop.outline %outline_target {func_name = "outlined"}
                 ^
../../test_Example/transform/Ch1/invalidation-2.mlir:26:13: note: ancestor payload op
  %biased = linalg.elemwise_binary { fun = #linalg.binary_fn<add> }
            ^
../../test_Example/transform/Ch1/invalidation-2.mlir:20:13: note: nested payload op
  %matmul = linalg.matmul ins(%lhs, %rhs: tensor<512x512xf32>, tensor<512x512xf32>)
            ^
```

`%f` was never passed to `transform.loop.outline`, yet it is invalidated: the matmul it points to is **nested** in the consumed loop. The two payload notes show the relation: the **ancestor payload op** is the loop, reported at the location of the "add" elementwise operation (line 26) because the loop was produced by tiling the add and inherited its location; the **nested payload op** is the matmul (line 20). **Invalidation follows payload nesting, not transform dataflow: consuming a handle to a loop invalidates handles to everything inside it.**

Finally, we would like to replace the call to the outlined function with a call to the microkernel. Unfortunately, the Transform dialect has no operation for this (and cannot have one, if the call is to be rewritten into a custom, out-of-tree operation). We need to define new transform operations — the subject of the next chapters.

---

## 10. Tracking IR Modifications

The Transform dialect automatically tracks all IR changes made as part of transform operations. (Implementations must use the provided rewriter to modify the IR.)

- If a payload operation is **erased**, it is automatically removed from all handles it is currently associated with.
- If a payload operation is **replaced**, the dialect tries to find the replacement operation and updates all handles accordingly.
- If a multi-result operation is replaced with values defined by several operations, or if an operation is replaced with an operation of a different type, an error is produced: it is unclear whether the direct replacements actually represent the computation of the original operation.

There are ways to customize this behavior; see the documentation of `transform::TrackingListener`.

Section 8 left a dead 512x512 matmul behind. The following script appends to section 8's script a handle to *all* matmuls in the function, counts them with `transform.num_associations`, applies the canonicalization patterns with `transform.apply_patterns` (a transform operation that runs rewrite patterns on the payload, through the tracking rewriter), and counts again through the **same handle**:

***stages/6-tracking.mlir***
```mlir
...
    // Fusion leaves the original matmul and add behind, now dead. Count the
    // matmuls in the function, let canonicalization erase the dead one, and
    // count again through the same handle.
    %func = transform.structured.match ops{["func.func"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %matmuls = transform.structured.match ops{["linalg.matmul"]} in %func
        : (!transform.any_op) -> !transform.any_op
    %before = transform.num_associations %matmuls
        : (!transform.any_op) -> !transform.param<i64>
    transform.debug.emit_param_as_remark %before, "matmuls before cleanup:"
        : !transform.param<i64>
    transform.apply_patterns to %func {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    %after = transform.num_associations %matmuls
        : (!transform.any_op) -> !transform.param<i64>
    transform.debug.emit_param_as_remark %after, "matmuls after cleanup:"
        : !transform.param<i64>
    transform.debug.emit_remark_at %matmuls, "still tracked" : !transform.any_op

    transform.yield
  }
}
```

`%before` and `%after` are **parameters** (section 2): transform values associated with attributes, here an `i64` count, printed by `transform.debug.emit_param_as_remark`.

```bash
cd /Users/roy/study/mlir/transform/Ch1
mlir-opt stages/6-tracking.mlir \
  --pass-pipeline="builtin.module(transform-interpreter{
      debug-bind-trailing-args=linalg.matmul,linalg.elemwise_binary})" \
  -o /dev/null
```

Real output **[21]**:

```text
stages/6-tracking.mlir:64:5: remark: matmuls before cleanup: 2 : i64
    transform.debug.emit_param_as_remark %before, "matmuls before cleanup:"
    ^
stages/6-tracking.mlir:64:5: note: see current operation: transform.debug.emit_param_as_remark %3, "matmuls before cleanup:" : !transform.param<i64>
stages/6-tracking.mlir:71:5: remark: matmuls after cleanup: 1 : i64
    transform.debug.emit_param_as_remark %after, "matmuls after cleanup:"
    ^
stages/6-tracking.mlir:71:5: note: see current operation: transform.debug.emit_param_as_remark %4, "matmuls after cleanup:" : !transform.param<i64>
stages/6-tracking.mlir:6:13: remark: still tracked
  %matmul = linalg.matmul ins(%lhs, %rhs: tensor<512x512xf32>, tensor<512x512xf32>)
            ^
stages/6-tracking.mlir:6:13: note: see current operation: %11 = linalg.matmul ins(%extracted_slice, %extracted_slice_0 : tensor<8x512xf32>, tensor<512x32xf32>) outs(%extracted_slice_1 : tensor<8x32xf32>) -> tensor<8x32xf32>
```

`%matmuls` started with two operations — the dead original and the 8x32 tile in the loop. Canonicalization erased the dead one, and the handle shrank to one without the script doing anything; the remaining element is the tiled matmul (it carries the original's source location, line 6). `apply_patterns` reads its target handle without consuming it, so `%func` and `%matmuls` stay valid. **Handles follow the payload as transformations rewrite it, which is what makes a script that chains transformations possible at all.**

---

## 11. Deriving the Chapter 1 script yourself

Sections 2–10 explain the transform script of `sequence.mlir` as it is, built up in the stage files. This section shows how to arrive at it without having it: which handles to start from, which transform operations to use, which handles each one leaves usable, and which tile sizes to pick. Each part of the script (lines of `test_Example/transform/Ch1/sequence.mlir`) and where it comes from:

| Part of the script | Lines | Where it comes from |
|---|---|---|
| `transform.with_named_sequence` module, `@__transform_main` and its three arguments | 62–66 | The payload's operation kinds (section 11.1) and the interpreter's options (section 11.2) |
| `transform.split_handle %arg2` | 70–71 | Two `linalg.elemwise_binary` operations behind one handle (section 11.1) |
| `tile_using_forall %max` and the two `fuse_into_containing_op` | 75–87 | One transform operation per step (section 11.3), applied to handles that are still valid (section 11.4) |
| The second tiling, `[4, 4]`, and its fusion | 94–98 | The microkernel's 4x4 signature (section 11.5) |
| The `[1]` tiling, its fusion, and `transform.loop.outline` | 103–108 | The signature the outlined function must have (section 11.5) |
| The `@microkernel` declaration | 54–60 | The interface the script targets, given beforehand |

### 11.1 Step 1: count the payload's operations

A script starts from handles to payload operations, so first list what the payload contains. The custom printer writes each operation's result as `%x = dialect.op`, so `grep` can count the top-level operations of `fc_relu`:

```bash
cd /Users/roy/study/mlir/transform/Ch1
mlir-opt stages/payload.mlir | grep -oE '= [a-z_]+\.[a-z_]+' | sort | uniq -c
```

Real output **[22]**:

```text
   1 = arith.constant
   2 = linalg.elemwise_binary
   1 = linalg.matmul
```

- One `linalg.matmul` and two `linalg.elemwise_binary` (the bias add and the ReLU max): binding a handle by operation name, as `debug-bind-trailing-args` does, gives `%arg1` one operation and `%arg2` two (section 5.1).
- The two elementwise operations have the same name, so no name-based binding can tell them apart; the script splits `%arg2` by position with `split_handle` (section 8). The `arith.constant` is the ReLU's zero and needs no handle.

### 11.2 Step 2: find how the interpreter starts a script

The interpreter pass decides how a script is found and what its arguments are bound to. `mlir-opt --help` lists its options:

```bash
cd /Users/roy/study/mlir/transform/Ch1
mlir-opt --help | grep -A4 -- '--transform-interpreter '
```

Real output **[23]**:

```text
      --transform-interpreter                                -   transform dialect interpreter
        --debug-bind-trailing-args=<string>                  - Binds trailing arguments of the entry point to the payload operations with specified names.
        --debug-payload-root-tag=<string>                    - Select the operation with 'transform.target_tag' attribute having the given value as payload IR root. If empty select the pass anchor operation as the payload IR root.
        --disable-expensive-checks                           - Disable expensive checks in the interpreter for a faster run.
        --entry-point=<string>                               - Entry point of the pass pipeline.
```

- **`entry-point`** names the sequence to run; without it, the interpreter looks for `@__transform_main` in a module marked `transform.with_named_sequence` (sections 5.1 and 5.4). That fixes lines 62–63.
- **`debug-bind-trailing-args`** binds the arguments after the first, in order, to all payload operations with the given names. With step 1's counts, `linalg.matmul,linalg.elemwise_binary` gives `%arg1: !transform.op<"linalg.matmul">` and `%arg2: !transform.op<"linalg.elemwise_binary">` (lines 64–66); the first argument is always the payload root (section 5.5).
- **`disable-expensive-checks`** names the checks that step 4 relies on; leave them on (section 7.3).

### 11.3 Step 3: find a transform operation for each step

The plan is chapter 0's: tile the last operation, fuse its producers into the loop, and outline the innermost tile (section 8). Each step is one transform operation, defined in the installed `.td` files of the dialect it works on. Their operands and results, from the operation definitions:

```bash
cd /Users/roy/study/mlir/transform/Ch1
I=/opt/homebrew/opt/llvm@20/include/mlir
for op in split_handle structured.tile_using_forall structured.fuse_into_containing_op loop.outline; do
  f=$(grep -rl --include='*.td' "\"$op\"" $I)
  echo "== transform.$op  (${f#$I/})"
  awk -v op="\"$op\"" 'index($0, op) { on = 1 }
       on && /let (arguments|results)/ { p = 1 }
       p { print } p && /\);/ { p = 0; if (++n == 2) exit }' "$f"
done
```

Real output **[24]**:

```text
== transform.split_handle  (Dialect/Transform/IR/TransformOps.td)
  let arguments = (ins Transform_AnyHandleOrParamType:$handle,
                       DefaultValuedAttr<BoolAttr, "true">:$pass_through_empty_handle,
                       DefaultValuedAttr<BoolAttr, "true">:$fail_on_payload_too_small,
                       OptionalAttr<I64Attr>:$overflow_result);
  let results = (outs Variadic<Transform_AnyHandleOrParamType>:$results);
== transform.structured.tile_using_forall  (Dialect/Linalg/TransformOps/LinalgTransformOps.td)
  let arguments = (ins TransformHandleTypeInterface:$target,
                   Variadic<TransformAnyParamTypeOrAnyHandle>:$num_threads,
                   Variadic<TransformAnyParamTypeOrAnyHandle>:$tile_sizes,
                   Optional<TransformAnyParamTypeOrAnyHandle>:$packed_num_threads,
                   Optional<TransformAnyParamTypeOrAnyHandle>:$packed_tile_sizes,
                   DefaultValuedOptionalAttr<DenseI64ArrayAttr, "{}">:$static_num_threads,
                   DefaultValuedOptionalAttr<DenseI64ArrayAttr, "{}">:$static_tile_sizes,
                   OptionalAttr<DeviceMappingArrayAttr>:$mapping);
  let results = (outs TransformHandleTypeInterface:$tiled_op,
                      TransformHandleTypeInterface:$forall_op);
== transform.structured.fuse_into_containing_op  (Dialect/Linalg/TransformOps/LinalgTransformOps.td)
  let arguments = (ins TransformHandleTypeInterface:$producer_op,
                       TransformHandleTypeInterface:$containing_op);
  let results = (outs TransformHandleTypeInterface:$fused_op,
                      TransformHandleTypeInterface:$new_containing_op);
== transform.loop.outline  (Dialect/SCF/TransformOps/SCFTransformOps.td)
  let arguments = (ins TransformHandleTypeInterface:$target,
                   StrAttr:$func_name);
  let results = (outs TransformHandleTypeInterface:$function,
                      TransformHandleTypeInterface:$call);
```

- **Where they live:** the generic handle operations (`split_handle`, also `merge_handles` and `cast`) are in the Transform dialect itself; tiling and fusion belong to Linalg (`structured.*`), outlining to SCF (`loop.*`). Each dialect brings the transform operations for its own payload.
- **The results, in order:** `tile_using_forall` returns the tiled operation, then the `scf.forall` (section 6); `fuse_into_containing_op` the fused producer, then the loop; `loop.outline` the function, then the call. The printer's default names in section 9's output (`%tiled_op, %forall_op`, `%fused_op, %new_containing_op`, `%function, %call`) are these result names.
- **The operands:** `split_handle` returns a variable number of results, one per payload operation of its operand, so `%add, %max` follows step 1's count. The tile sizes are written in the `static_tile_sizes` attribute (`tile_sizes [8, 32]`), `func_name` is the base name of the outlined function.
- **The types:** every handle operand and result is `TransformHandleTypeInterface`, so `!transform.any_op` fits everywhere; the script keeps `!transform.op<"linalg.matmul">` for `%arg1` only because it is bound that way (step 2).

### 11.4 Step 4: let the expensive checks list the handles that are still valid

The argument lists don't say which operands an operation consumes: that comes from a trait (`FunctionalStyleTransformOpTrait`) or a hand-written `getEffects`, and the description doesn't always mention it. The interpreter knows it, though: with the expensive checks on, a use of a consumed handle is an error naming the operation that consumed it (section 7.1). Appending one `transform.debug.emit_remark_at` per handle to the complete script, before `transform.yield`, probes each one:

```bash
cd /Users/roy/study/mlir/transform/Ch1
T=../../test_Example/transform/Ch1/sequence.mlir
for h in '%arg1 !transform.op<"linalg.matmul">' '%arg2 !transform.op<"linalg.elemwise_binary">' \
         '%max !transform.any_op' '%add !transform.any_op' '%loop !transform.any_op' \
         '%add_fused !transform.any_op' '%outline_target !transform.any_op'; do
  set -- $h
  sed "/^    transform.yield/i\\
    transform.debug.emit_remark_at $1, \"valid\" : $2" $T |
  mlir-opt --pass-pipeline="builtin.module(transform-interpreter{
      debug-bind-trailing-args=linalg.matmul,linalg.elemwise_binary})" -o /dev/null 2>&1 |
  awk -v h="$1" -F: '/remark: valid/ { print h ": still valid at the end"; exit }
       /invalidated by this transform op/ { print h ": consumed by line " $2; exit }'
done
```

Real output **[25]**:

```text
%arg1: consumed by line 86
%arg2: consumed by line 75
%max: consumed by line 75
%add: consumed by line 84
%loop: still valid at the end
%add_fused: consumed by line 94
%outline_target: consumed by line 107
```

- **Tiling and outlining consume their target** (`%max` at line 75, `%add_fused` at 94, `%outline_target` at 107): the payload operation is replaced by a new one inside the loop, or moved into a function. So the script continues with the *results*: `%tiled`, `%loop`, `%tiled_second`, ...
- **Fusion consumes the producer but only reads the loop** (`%add` at 84, `%arg1` at 86, while `%loop` stays valid to the end): the fused operation is a new tile inside the loop, and the loop is updated in place. The script still chains the returned `%loop2`, `%loop3`, ... as section 8 does.
- **`split_handle` only reads `%arg2`**, but `%arg2` dies at line 75 anyway: it also holds the max that line 75 consumes, and consuming one payload operation invalidates every handle to it (section 7.2). Hence the split comes first and the script never uses `%arg2` again.

Line 107's `loop.outline` would not need to consume its operand, since it only moves the region (section 9.1); the probe reports what the operation declares, which is what the checks enforce.

### 11.5 Step 5: size the tiles from the microkernel

The last part of the script exists for the call: the outlined function should have the microkernel's signature, so that the call can later be redirected to it. The declaration (lines 54–60) fixes the shapes; compare it with what `@outlined` becomes with the real script and with two shorter scripts that skip the `[1]` tiling and its fusion (lines 99–106, deleted by `sed`), outlining the 4x4 loop or the 4x4 add instead:

```bash
cd /Users/roy/study/mlir/transform/Ch1
T=../../test_Example/transform/Ch1/sequence.mlir
P="builtin.module(transform-interpreter{debug-bind-trailing-args=linalg.matmul,linalg.elemwise_binary},canonicalize,cse,symbol-dce)"
mlir-opt $T | grep -o 'func.func private @microkernel.*'
mlir-opt $T --pass-pipeline="$P" | grep -o 'func.func @outlined.*{'
for target in %loop_second_2 %tiled_second; do
  sed -e '99,106d' -e "107s/%outline_target/$target/" $T |
    mlir-opt --pass-pipeline="$P" | grep -o 'func.func @outlined.*{'
done
```

Real output **[26]**:

```text
func.func private @microkernel(tensor<4x512xf32>, tensor<512x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
func.func @outlined(%arg0: tensor<4x512xf32>, %arg1: tensor<512x4xf32>, %arg2: tensor<4x4xf32>, %arg3: tensor<4x4xf32>, %arg4: tensor<4x4xf32>) -> tensor<4x4xf32> {
func.func @outlined(%arg0: tensor<8x512xf32>, %arg1: tensor<512x32xf32>, %arg2: tensor<8x32xf32>, %arg3: tensor<8x32xf32>, %arg4: tensor<8x32xf32>) -> tensor<8x32xf32> {
func.func @outlined(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>, %arg2: tensor<4x4xf32>) -> tensor<4x4xf32> {
```

- **The real script** (second line) produces exactly the microkernel's argument and result types: a 4x512 row block, a 512x4 column block, and three 4x4 tiles (the matmul's init, the bias, the add's output). The 4x512 and 512x4 shapes come from the `[4, 4]` tiling of lines 94–95 (the reduction dimension, 512, is never tiled); the first level, `[8, 32]`, only has to contain whole 4x4 tiles.
- **Outlining the 4x4 loop** (`%loop_second_2`, third line) moves the whole loop over the 2x8 tiles into the function, which then works on an 8x32 block: one call per 8x32 tile instead of one per 4x4 tile.
- **Outlining the 4x4 add** (`%tiled_second`, fourth line) is accepted (the `.td` file notes that the implementation does not require a loop), but the function holds only the add: three 4x4 tensors, no matmul.
- **Only the `[1]` tiling plus fusion** (lines 103–106) creates a region that holds the 4x4 matmul *and* the 4x4 add and nothing else, which is why the script tiles a 4x4 operation by 1 before outlining (section 9).

**Write a schedule backwards from the interface it must reach: the microkernel's signature fixes the tile sizes and what has to be fused before outlining, and comparing signatures checks each guess in one run.**

---

## 12. Key Takeaways & Pitfalls

1. **Two IRs, one module.** The transform IR (the script) and the payload IR (what it transforms) can live in the same file; the interpreter runs `@__transform_main`, binding its first argument to the payload root and, with `debug-bind-trailing-args`, the others to payload ops by name (sections 3 and 5).
2. **Handles are typed lists of payload entities.** A type such as `!transform.op<"linalg.matmul">` is checked when the handle is bound (section 5.3), and one handle can hold several ops (`%arg2`, section 5.1); `split_handle` and `merge_handles` reshape them (section 8).
3. **Consumption invalidates — transitively.** A transform that erases or recreates payload consumes its operand; that invalidates every handle aliasing the same payload *and* every handle to ops nested in it (sections 7.2 and 9.1).
4. **Keep the expensive checks on.** They turn a use-after-consume into a precise error with notes; with `disable-expensive-checks`, the same script ran silently in this build (section 7.3).
5. **Propagate failures while developing.** `failures(suppress)` swallows the diagnostic; a step that never applied looks like one that was not needed (section 4).
6. **Tile the consumer, fuse the producers.** Tile the last op of the chain and fuse its producers into the loop, consumer-first (or let `merge_handles` order them); then run `canonicalize,cse` to remove the dead originals and duplicate index computations (section 8).
7. **Handles track the payload.** Erased ops drop out of handles and replaced ops are followed, as long as the transformation uses the tracking rewriter (section 10).
8. **Pitfall — a named entry point is required.** The interpreter pass looks for a `transform.named_sequence` called `@__transform_main` (or the one given by `entry-point=`) inside a module with `transform.with_named_sequence`; an unnamed top-level `transform.sequence` is not found (section 5.4).
9. **Pitfall — the payload root includes the script.** By default `%arg0` is the whole module, transform IR included; `debug-payload-root-tag` or a separate script file narrow it (section 5.5, appendix A).
10. **A schedule is written backwards** (section 11): count the payload's operations to choose the handles, look up each transform operation's results in its `.td` file, let the expensive checks list which handles are still valid, and pick the tile sizes and the outlining target from the signature the result must have.

---

## Appendix A: A Transform Script in a Separate File

Everything so far kept the script in the payload's file. The transform dialect also supports the "scheduling language" style, where the script is a separate file: the `transform-preload-library` pass parses the files given in `transform-library-paths` and makes their named sequences available to `transform-interpreter`, which finds `@__transform_main` there when the payload does not define one. The script below also does without `debug-bind-trailing-args`: it gets its handles with **`transform.structured.match`**, which returns a handle to all ops nested in its target that satisfy the criteria (here, an operation name). And it uses **`transform.print`**, which prints the payload ops associated with a handle (or, without a handle, the whole payload root) under a title:

***stages/separate-script.mlir***
```mlir
// A transform script kept apart from the payload: no debug-bind-trailing-args,
// the handles come from matching inside the payload root instead.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %root
        : (!transform.any_op) -> !transform.op<"linalg.matmul">
    %elemwise = transform.structured.match ops{["linalg.elemwise_binary"]} in %root
        : (!transform.any_op) -> !transform.op<"linalg.elemwise_binary">
    transform.print %matmul {name = "matched matmul"} : !transform.op<"linalg.matmul">
    transform.print %elemwise {name = "matched elemwise_binary"}
        : !transform.op<"linalg.elemwise_binary">
    %tiled, %loop = transform.structured.tile_using_forall %matmul tile_sizes [4, 32]
        : (!transform.op<"linalg.matmul">) -> (!transform.any_op, !transform.any_op)
    transform.print %loop {name = "loop after tiling"} : !transform.any_op
    transform.yield
  }
}
```

```bash
cd /Users/roy/study/mlir/transform/Ch1
mlir-opt stages/payload.mlir \
  --transform-preload-library=transform-library-paths=stages/separate-script.mlir \
  --transform-interpreter
```

Real output **[27]**:

```text
[[[ IR printer: matched matmul ]]]
%0 = linalg.matmul ins(%arg0, %arg1 : tensor<512x512xf32>, tensor<512x512xf32>) outs(%arg3 : tensor<512x512xf32>) -> tensor<512x512xf32>
[[[ IR printer: matched elemwise_binary ]]]
%1 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%0, %arg2 : tensor<512x512xf32>, tensor<512x512xf32>) outs(%arg3 : tensor<512x512xf32>) -> tensor<512x512xf32>
%2 = linalg.elemwise_binary {fun = #linalg.binary_fn<max_signed>} ins(%1, %cst : tensor<512x512xf32>, f32) outs(%arg3 : tensor<512x512xf32>) -> tensor<512x512xf32>
[[[ IR printer: loop after tiling ]]]
%0 = scf.forall (%arg4, %arg5) in (128, 16) shared_outs(%arg6 = %arg3) -> (tensor<512x512xf32>) {
  %3 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg4)
  %4 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg5)
  %extracted_slice = tensor.extract_slice %arg0[%3, 0] [4, 512] [1, 1] : tensor<512x512xf32> to tensor<4x512xf32>
  %extracted_slice_0 = tensor.extract_slice %arg1[0, %4] [512, 32] [1, 1] : tensor<512x512xf32> to tensor<512x32xf32>
  %extracted_slice_1 = tensor.extract_slice %arg6[%3, %4] [4, 32] [1, 1] : tensor<512x512xf32> to tensor<4x32xf32>
  %5 = linalg.matmul ins(%extracted_slice, %extracted_slice_0 : tensor<4x512xf32>, tensor<512x32xf32>) outs(%extracted_slice_1 : tensor<4x32xf32>) -> tensor<4x32xf32>
  scf.forall.in_parallel {
    tensor.parallel_insert_slice %5 into %arg6[%3, %4] [4, 32] [1, 1] : tensor<4x32xf32> into tensor<512x512xf32>
  }
}
#map = affine_map<(d0) -> (d0 * 4)>
#map1 = affine_map<(d0) -> (d0 * 32)>
module {
  func.func @fc_relu(%arg0: tensor<512x512xf32>, %arg1: tensor<512x512xf32>, %arg2: tensor<512x512xf32>, %arg3: tensor<512x512xf32>) -> tensor<512x512xf32> {
    %0 = scf.forall (%arg4, %arg5) in (128, 16) shared_outs(%arg6 = %arg3) -> (tensor<512x512xf32>) {
      %3 = affine.apply #map(%arg4)
      %4 = affine.apply #map1(%arg5)
      %extracted_slice = tensor.extract_slice %arg0[%3, 0] [4, 512] [1, 1] : tensor<512x512xf32> to tensor<4x512xf32>
      %extracted_slice_0 = tensor.extract_slice %arg1[0, %4] [512, 32] [1, 1] : tensor<512x512xf32> to tensor<512x32xf32>
      %extracted_slice_1 = tensor.extract_slice %arg6[%3, %4] [4, 32] [1, 1] : tensor<512x512xf32> to tensor<4x32xf32>
      %5 = linalg.matmul ins(%extracted_slice, %extracted_slice_0 : tensor<4x512xf32>, tensor<512x32xf32>) outs(%extracted_slice_1 : tensor<4x32xf32>) -> tensor<4x32xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %5 into %arg6[%3, %4] [4, 32] [1, 1] : tensor<4x32xf32> into tensor<512x512xf32>
      }
    }
    %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%0, %arg2 : tensor<512x512xf32>, tensor<512x512xf32>) outs(%arg3 : tensor<512x512xf32>) -> tensor<512x512xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<max_signed>} ins(%1, %cst : tensor<512x512xf32>, f32) outs(%arg3 : tensor<512x512xf32>) -> tensor<512x512xf32>
    return %2 : tensor<512x512xf32>
  }
}

```

- The three `[[[ IR printer: ... ]]]` blocks come from `transform.print`, in script order: the matched matmul, the two matched elementwise operations (one handle, two ops), and the `scf.forall` produced by tiling — the same loop as section 6.
- The final module is the tiled payload **without any transform IR**: the script was never part of the payload, so the payload root is just the payload.
- `stages/payload.mlir` is reusable with any number of scripts, and a script with any payload containing these op kinds: **keeping the schedule in its own file separates *what* is computed from *how* it is optimized.**

## Appendix B: The Stage Files

The stage files exist so that each section's script runs on its own. Each `stages/N-*.mlir` is `stages/payload.mlir` (20 lines) followed by a blank line and that section's transform module; the payload is also lines 14–33 of the test file `sequence.mlir`:

```bash
cd /Users/roy/study/mlir/transform/Ch1
for f in stages/[1-6]-*.mlir; do
  sed -n 1,20p "$f" | diff -q - stages/payload.mlir > /dev/null && echo "$f: same payload"
done
sed -n 14,33p ../../test_Example/transform/Ch1/sequence.mlir | diff - stages/payload.mlir && echo "sequence.mlir: same payload"
```

Real output **[28]**:

```text
stages/1-empty.mlir: same payload
stages/2-failures.mlir: same payload
stages/3-remarks.mlir: same payload
stages/4-tile-matmul.mlir: same payload
stages/5-tile-fuse.mlir: same payload
stages/6-tracking.mlir: same payload
sequence.mlir: same payload
```

| File | Section | Script |
|---|---|---|
| `stages/payload.mlir` | 2, A | the payload alone |
| `stages/1-empty.mlir` | 3, 5.5 | empty `@__transform_main` |
| `stages/2-failures.mlir` | 4 | `suppress` vs `propagate` around a failing `split_handle` |
| `stages/3-remarks.mlir` | 5 | `emit_remark_at` on the bound handles |
| `stages/4-tile-matmul.mlir` | 6 | tile the matmul |
| `stages/5-tile-fuse.mlir` | 8 | tile the ReLU, fuse add and matmul |
| `test_Example/.../sequence.mlir` | 9 | the complete script (tile, fuse, tile, fuse, outline) |
| `stages/6-tracking.mlir` | 10 | section 8's script + handle tracking through canonicalization |
| `stages/separate-script.mlir` | A | a transform-only file for `--transform-preload-library` |

`stages/6-tracking.mlir` extends `stages/5-tile-fuse.mlir`; `diff` shows exactly what section 10 adds (it exits with 1 because the files differ):

```bash
cd /Users/roy/study/mlir/transform/Ch1
diff stages/5-tile-fuse.mlir stages/6-tracking.mlir
```

Real output **[29]** (exit status 1):

```text
54a55,74
>     // Fusion leaves the original matmul and add behind, now dead. Count the
>     // matmuls in the function, let canonicalization erase the dead one, and
>     // count again through the same handle.
>     %func = transform.structured.match ops{["func.func"]} in %arg0
>         : (!transform.any_op) -> !transform.any_op
>     %matmuls = transform.structured.match ops{["linalg.matmul"]} in %func
>         : (!transform.any_op) -> !transform.any_op
>     %before = transform.num_associations %matmuls
>         : (!transform.any_op) -> !transform.param<i64>
>     transform.debug.emit_param_as_remark %before, "matmuls before cleanup:"
>         : !transform.param<i64>
>     transform.apply_patterns to %func {
>       transform.apply_patterns.canonicalization
>     } : !transform.any_op
>     %after = transform.num_associations %matmuls
>         : (!transform.any_op) -> !transform.param<i64>
>     transform.debug.emit_param_as_remark %after, "matmuls after cleanup:"
>         : !transform.param<i64>
>     transform.debug.emit_remark_at %matmuls, "still tracked" : !transform.any_op
> 
```

## Appendix C: `examples.sh`

`examples.sh` (a symlink to `transform/examples.sh`) runs the commands of this README straight from the file: every `bash` block except file excerpts, numbered in order, each printed with its output and exit status.

```text
./examples.sh          # run every command, in README order
./examples.sh 7        # run command [7] only (several numbers are fine)
./examples.sh -l       # list the numbered commands without running them
```

```bash
cd /Users/roy/study/mlir/transform/Ch1
./examples.sh -l
```

Real output **[30]**:

```text
[ 1] 2     mlir-opt stages/payload.mlir
[ 2] 3     mlir-opt stages/1-empty.mlir \
[ 3] 4     mlir-opt stages/2-failures.mlir \
[ 4] 5.1   mlir-opt stages/3-remarks.mlir \
[ 5] 5.2   mlir-opt stages/3-remarks.mlir --transform-interpreter -o /dev/null
[ 6] 5.3   mlir-opt stages/3-remarks.mlir \
[ 7] 5.4   { cat stages/payload.mlir; cat <<'EOF'
[ 8] 5.5   sed '/transform.yield/i\
[ 9] 5.5   sed -e '4s/-> tensor<512x512xf32> {/-> tensor<512x512xf32> attributes {transform.target_tag = "payload"} {/' \
[10] 6     mlir-opt stages/4-tile-matmul.mlir \
[11] 7.2   mlir-opt ../../test_Example/transform/Ch1/invalidation-1.mlir \
[12] 7.3   mlir-opt ../../test_Example/transform/Ch1/invalidation-1.mlir \
[13] 7.3   mlir-opt ../../test_Example/transform/Ch1/invalidation-1.mlir \
[14] 8     mlir-opt stages/5-tile-fuse.mlir \
[15] 8     mlir-opt stages/5-tile-fuse.mlir \
[16] 8     sed -e '47,48{s/%add/%arg1/;s/(!transform.any_op, !transform.any_op)$/(!transform.op<"linalg.matmul">, !transform.any_op)/;}' \
[17] 8     P="builtin.module(transform-interpreter{debug-bind-trailing-args=linalg.matmul,linalg.elemwise_binary},canonicalize,cse)"
[18] 9     mlir-opt ../../test_Example/transform/Ch1/sequence.mlir \
[19] 9     mlir-opt ../../test_Example/transform/Ch1/sequence.mlir \
[20] 9.1   mlir-opt ../../test_Example/transform/Ch1/invalidation-2.mlir \
[21] 10    mlir-opt stages/6-tracking.mlir \
[22] 11.1  mlir-opt stages/payload.mlir | grep -oE '= [a-z_]+\.[a-z_]+' | sort | uniq -c
[23] 11.2  mlir-opt --help | grep -A4 -- '--transform-interpreter '
[24] 11.3  I=/opt/homebrew/opt/llvm@20/include/mlir
[25] 11.4  T=../../test_Example/transform/Ch1/sequence.mlir
[26] 11.5  T=../../test_Example/transform/Ch1/sequence.mlir
[27] A     mlir-opt stages/payload.mlir \
[28] B     for f in stages/[1-6]-*.mlir; do
[29] B     diff stages/5-tile-fuse.mlir stages/6-tracking.mlir
[30] C     ./examples.sh -l
```

---

## Links

- [Transform Ch1: Combining Existing Transformations](https://mlir.llvm.org/docs/Tutorials/transform/Ch1/)
- [Transform dialect reference](https://mlir.llvm.org/docs/Dialects/Transform/)
- [Linalg dialect reference](https://mlir.llvm.org/docs/Dialects/Linalg/)
