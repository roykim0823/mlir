# Chapter 4: Matching Payload with Transform Operations

> **Goal:** Let the transform script find its own targets: write *matchers* as named sequences of match operations, combine them with `transform.collect_matching` and `transform.foreach_match`, define a new match operation in C++ (`transform.match.my.has_operand_satisfying`), and match operations by *inferred* properties with the `structured` matchers — based on the official tutorial [Transform Ch4: Matching Payload with Transform Operations](https://mlir.llvm.org/docs/Tutorials/transform/Ch4/).

---

## 1. Overview

Up to Chapter 3, every transform script assumed that the caller of the transform interpreter already knew which payload operations to transform. In Chapter 1, for example, the interpreter pass was given `debug-bind-trailing-args=linalg.matmul,linalg.elemwise_binary`, and the entry point received handles to those operations as extra arguments. That works against the idea of driving transformations *from a dialect*: the targets are identified by a mechanism outside the transform IR (pass options here, C++ calls to the interpreter elsewhere), there is more C++ glue to write, and the reader has to keep two interfaces in mind at once.

This chapter removes that dependency. The Transform dialect has a family of **match operations**: transform operations with extra guarantees. They do not modify the payload IR, and they *fail* (with a silenceable failure) when their operands are associated with payload objects that lack some property, such as an operation name or a kind of operand. Small combinator operations turn such matchers into a match-and-rewrite system written entirely in the transform dialect. We will:

1. See what a match operation is and what its failure looks like (section 2).
2. Write matchers as named sequences and collect their matches with `transform.collect_matching` (section 3).
3. Match a *chain* of related operations instead of every operation of a kind, and select it with the interpreter's `entry-point` option (section 4).
4. Define a new match operation in C++, `transform.match.my.has_operand_satisfying`, and drive it with `transform.foreach_match` (section 5).
5. Match a matrix multiplication written as `linalg.generic`, through properties that no attribute spells out (section 6).

Chapter 4's code is one new operation: its ODS (Operation Definition Specification) record in `include/MyExtension.td`, its C++ in `lib/MyExtension.cpp`, and the driver `transform-opt/transform-opt.cpp` that registers it. The inputs are the three files in `test_Example/transform/Ch4/`: `sequence.mlir` (sections 3–4), `multiple.mlir` (section 5) and `features.mlir` (section 6).

Sections 1–9 follow the chapter's essential path; section 8 shows how to arrive at the new match operation and its build yourself. The appendices hold the operation's generated documentation (A), a pitfall about running the canonicalizer on a transform script (B), the build wiring (C), and the script that runs every command in this README (D).

---

## 2. Match Operations

### 2.1 Match operations are transform operations that only look

A match operation is an ordinary transform operation (it implements `TransformOpInterface` and is applied by the interpreter like any other) that also implements `MatchOpInterface`. It promises two things:

- **It does not modify the payload IR.** Its memory effects say that it only reads its operand handles and the payload.
- **It fails when the payload does not have the property it checks.** The failure is *silenceable*: the interpreter's tri-state result (`DiagnosedSilenceableFailure`: success, silenceable failure, definite failure) lets a surrounding combinator catch it and treat it as "did not match", while a definite failure (an irrecoverable error) is always propagated.

The simplest match operation is `transform.match.operation_name`, which succeeds if every payload operation associated with its handle has one of the listed names. On its own, outside any combinator, a failed match is just a transform error. Here it checks that the `func.func` of a tiny module is a `linalg.matmul`:

```bash
cd /Users/roy/study/mlir/transform/Ch4
./build/transform-opt-ch4 --transform-interpreter -o /dev/null <<'EOF'
func.func @f(%t: tensor<4xf32>) -> tensor<4xf32> {
  return %t : tensor<4xf32>
}
module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op {transform.readonly}) {
    %funcs = transform.structured.match ops{["func.func"]} in %root
      : (!transform.any_op) -> !transform.any_op
    transform.match.operation_name %funcs ["linalg.matmul"] : !transform.any_op
    transform.yield
  }
}
EOF
```

Real output **[1]** (exit status 1):

```text
<stdin>:9:5: error: wrong operation name
    transform.match.operation_name %funcs ["linalg.matmul"] : !transform.any_op
    ^
```

- The diagnostic `wrong operation name` is the *message* of the silenceable failure. Nothing caught it, so the interpreter reported it as an error at the match operation and the tool exited with status 1.
- Inside a matcher (section 3) the same failure means "this candidate does not match" and is silenced; the message is only sent to the debug stream.

Using simple combinator operations, such failures become the building block of a match-and-rewrite infrastructure inside the transform dialect itself.

---

## 3. Simple Match

### 3.1 The payload: Chapter 1's fully connected layer

The example is the "fully connected layer" from Chapter 1: a matrix multiplication, an elementwise addition of a bias, and an elementwise maximum with 0 (ReLU, the rectified linear unit). The `expected-remark` comments are for `--verify-diagnostics` (section 7.2); ignore them for now.

***test_Example/transform/Ch4/sequence.mlir***
```mlir
// Original function to optimize.
func.func @fc_relu(%lhs: tensor<512x512xf32>, %rhs: tensor<512x512xf32>,
                   %bias: tensor<512x512xf32>, %output: tensor<512x512xf32>)
                   -> tensor<512x512xf32> {
  // Matrix-matrix multiplication.
  // expected-remark @below {{matmul}}
  %matmul = linalg.matmul ins(%lhs, %rhs: tensor<512x512xf32>, tensor<512x512xf32>)
                          outs(%output: tensor<512x512xf32>) -> tensor<512x512xf32>

  // Elementwise addition.
  // expected-remark @below {{elementwise binary}}
  %biased = linalg.elemwise_binary { fun = #linalg.binary_fn<add> }
    ins(%matmul, %bias : tensor<512x512xf32>, tensor<512x512xf32>)
    outs(%output : tensor<512x512xf32>) -> tensor<512x512xf32>

  // Elementwise max with 0 (ReLU).
  %c0f = arith.constant 0.0 : f32
  // expected-remark @below {{elementwise binary}}
  %relued = linalg.elemwise_binary { fun = #linalg.binary_fn<max_signed> }
    ins(%biased, %c0f : tensor<512x512xf32>, f32)
    outs(%output : tensor<512x512xf32>) -> tensor<512x512xf32>
  func.return %relued : tensor<512x512xf32>
}
```

### 3.2 Matchers, actions and `transform.collect_matching`

In Chapter 1 the interpreter pass received `debug-bind-trailing-args=linalg.matmul,linalg.elemwise_binary` to associate the entry point's extra arguments with payload operations. Instead, match operations can *discover* the relevant operations. They combine with regular transform operations through combinators such as `transform.collect_matching`, which organizes matchers as named sequences:

***test_Example/transform/Ch4/sequence.mlir***
```mlir
// The module containing named sequences must have an attribute allowing them
// to enable verification.
module @transforms attributes { transform.with_named_sequence } {
  // Entry point. This takes as the only argument the root operation (typically
  // pass root) given to the transform interpreter.
  transform.named_sequence @__transform_main(
      %root: !transform.any_op {transform.readonly}) {
    // Collect operations that match the criteria specified in the named
    // sequence. If the named sequence fails with a silenceable failure,
    // silences it (the message is forwarded to the debug stream). If the named
    // sequence succeeds, appends its results to the results of this operation.
    %elemwise = transform.collect_matching @match_elemwise in %root            // (1)
      : (!transform.any_op) -> !transform.any_op
    %matmul = transform.collect_matching @match_matmul in %root
      : (!transform.any_op) -> !transform.any_op

    transform.include @print_elemwise failures(propagate)  (%elemwise)         // (2)
      : (!transform.any_op) -> ()
    transform.include @print_matmul failures(propagate)  (%matmul)
      : (!transform.any_op) -> ()

    transform.yield
  }
  ...
  // This is a matcher sequence. It is given an operation to match and the
  // match is considered successful unless any nested operation produces a
  // failure. The values yielded by this operation will be forwarded to the
  // rewriter sequence on success.
  transform.named_sequence @match_elemwise(
      %entry: !transform.any_op {transform.readonly}) -> !transform.any_op {   // (3)
    transform.match.operation_name %entry ["linalg.elemwise_binary"]           // (4)
      : !transform.any_op
    transform.yield %entry : !transform.any_op                                 // (5)
  }
  transform.named_sequence @match_matmul(
      %entry: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %entry ["linalg.matmul"] : !transform.any_op
    transform.yield %entry : !transform.any_op
  }

  // This is an action sequence.
  transform.named_sequence @print_elemwise(                                    // (6)
      %elemwise_binary: !transform.any_op {transform.readonly}) {
    transform.debug.emit_remark_at
      %elemwise_binary, "elementwise binary" : !transform.any_op
    transform.yield
  }
  transform.named_sequence @print_matmul(
      %matmul: !transform.any_op {transform.readonly}) {
    transform.debug.emit_remark_at %matmul, "matmul" : !transform.any_op
    transform.yield
  }
  ...
}
```

The excerpt is abridged: the `...` elide `@__transform_main_v2` and `@match_matmul_elemwise`, which section 4 explains.

- **(1)** `transform.collect_matching @match_elemwise in %root` walks every operation nested in the payload associated with `%root` (here, the whole input module) and calls the matcher `@match_elemwise` on each. A silenceable failure is silenced (its message goes to the debug stream); on success the values the matcher yields are appended to the results. `%elemwise` ends up associated with *all* `linalg.elemwise_binary` operations, in walk order.
- **(2)** `transform.include` calls a named sequence like a function. `failures(propagate)` means a silenceable failure inside it becomes a failure of the `include` itself, rather than being suppressed.
- **(3)** A **matcher** is a named sequence that takes the candidate operation. Its argument is `{transform.readonly}`: a matcher must not consume (invalidate) the handle it was given.
- **(4)** The match succeeds unless one of its nested operations fails. Here the single check is the operation name.
- **(5)** On success the yielded values become the matcher's results; `collect_matching` gathers them.
- **(6)** An **action** (also called a rewriter) sequence does the work; here it only emits a remark at every operation of its handle so we can see what was matched.

The module carries `transform.with_named_sequence`: the verifier only accepts `transform.named_sequence` operations inside a module with this attribute, which also makes the module the scope in which `@match_elemwise` and friends are looked up.

### 3.3 Running the simple matcher

The script runs with the regular (non-test) interpreter pass on the root of the input, with no extra flags: no `debug-bind-trailing-args`, nothing about which operations to target. `-o /dev/null` drops the (unchanged) output IR so that only the diagnostics remain:

```bash
cd /Users/roy/study/mlir/transform/Ch4
./build/transform-opt-ch4 ../../test_Example/transform/Ch4/sequence.mlir \
  --transform-interpreter -o /dev/null
```

Real output **[2]**:

```text
../../test_Example/transform/Ch4/sequence.mlir:25:13: remark: elementwise binary
  %biased = linalg.elemwise_binary { fun = #linalg.binary_fn<add> }
            ^
../../test_Example/transform/Ch4/sequence.mlir:25:13: note: see current operation: %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%0, %arg2 : tensor<512x512xf32>, tensor<512x512xf32>) outs(%arg3 : tensor<512x512xf32>) -> tensor<512x512xf32>
../../test_Example/transform/Ch4/sequence.mlir:32:13: remark: elementwise binary
  %relued = linalg.elemwise_binary { fun = #linalg.binary_fn<max_signed> }
            ^
../../test_Example/transform/Ch4/sequence.mlir:32:13: note: see current operation: %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<max_signed>} ins(%1, %cst : tensor<512x512xf32>, f32) outs(%arg3 : tensor<512x512xf32>) -> tensor<512x512xf32>
../../test_Example/transform/Ch4/sequence.mlir:20:13: remark: matmul
  %matmul = linalg.matmul ins(%lhs, %rhs: tensor<512x512xf32>, tensor<512x512xf32>)
            ^
../../test_Example/transform/Ch4/sequence.mlir:20:13: note: see current operation: %0 = linalg.matmul ins(%arg0, %arg1 : tensor<512x512xf32>, tensor<512x512xf32>) outs(%arg3 : tensor<512x512xf32>) -> tensor<512x512xf32>
```

- Two `elementwise binary` remarks, at lines 25 and 32, then one `matmul` remark at line 20: the order of the two `transform.include`s, not the payload's order.
- Each remark comes with a `note: see current operation`: `transform.debug.emit_remark_at` attaches the operation as it is now (in generic-name form, `%1 = ...`) so you can tell which operation the remark is about after transformations have changed it.
- The exit status is 0: a match that doesn't apply to a candidate isn't an error.

Nothing in this script needs Chapter 4's extension: `collect_matching`, `match.operation_name` and the debug operations are all upstream, so the stock `mlir-opt` runs the same file and prints the same three remarks:

```bash
cd /Users/roy/study/mlir/transform/Ch4
mlir-opt ../../test_Example/transform/Ch4/sequence.mlir \
  --transform-interpreter -o /dev/null 2>&1 | grep remark
```

Real output **[3]**:

```text
../../test_Example/transform/Ch4/sequence.mlir:25:13: remark: elementwise binary
../../test_Example/transform/Ch4/sequence.mlir:32:13: remark: elementwise binary
../../test_Example/transform/Ch4/sequence.mlir:20:13: remark: matmul
```

### 3.4 Seeing why a candidate did not match

In a debug build of MLIR, `-debug-only=transform-matcher` prints the silenceable failure messages that the matchers produce into the debug stream, for example (not this repo's output):

```
<...>
[transform-matcher] matching %0 = linalg.matmul ins(%arg0, %arg1 : tensor<512x512xf32>, tensor<512x512xf32>) outs(%arg3 : tensor<512x512xf32>) -> tensor<512x512xf32> @0x5622eee08410
[transform-matcher] matcher match_elemwise failed: wrong operation name
<...>
```

The Homebrew LLVM 20 build is a Release build: the `-debug-only` option, and the `LLVM_DEBUG` output behind it, are compiled out:

```bash
cd /Users/roy/study/mlir/transform/Ch4
./build/transform-opt-ch4 ../../test_Example/transform/Ch4/sequence.mlir \
  --transform-interpreter -debug-only=transform-matcher -o /dev/null
```

Real output **[4]** (exit status 1):

```text
transform-opt-ch4: Unknown command line argument '-debug-only=transform-matcher'.  Try: './build/transform-opt-ch4 --help'
transform-opt-ch4: Did you mean '--debug-pass=transform-matcher'?
```

The same information is available without a debug build: call the matcher yourself on the candidate with `transform.include ... failures(propagate)`. Then nothing silences its failure, and the message is reported as an error. The command below appends a new entry point `@why` to the script module (`sed '$d'` drops the file's final `}`, the heredoc adds the sequence and the `}` back) and selects it with `entry-point=why` (section 4.2):

```bash
cd /Users/roy/study/mlir/transform/Ch4
{ sed '$d' ../../test_Example/transform/Ch4/sequence.mlir; cat <<'EOF'
  transform.named_sequence @why(%root: !transform.any_op {transform.readonly}) {
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %root
      : (!transform.any_op) -> !transform.any_op
    transform.include @match_elemwise failures(propagate) (%matmul)
      : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
EOF
} | ./build/transform-opt-ch4 --transform-interpreter=entry-point=why -o /dev/null
```

Real output **[5]** (exit status 1):

```text
<stdin>:85:5: error: wrong operation name
    transform.match.operation_name %entry ["linalg.elemwise_binary"] 
    ^
```

This is the `matcher match_elemwise failed: wrong operation name` line of the debug log above, as an error. **A matcher's verdict is an ordinary transform result, so the same sequence that filters candidates silently can be called directly to explain a rejection.**

### 3.5 The rest of the Chapter 1 schedule, without pass options

The matched handles are sufficient to run the rest of Chapter 1's transform script, substituting the extra arguments (`%arg1`, `%arg2` there) with handles found by matchers. Below, the payload of section 3.1 is followed by a script whose entry point takes only `%root`: a matcher (the chain matcher of section 4.1) returns the matmul, the add and the max, and the first part of Chapter 1's schedule (tile the max by `[8, 32]`, then fuse the add and the matmul into the loop) runs on them unchanged:

```bash
cd /Users/roy/study/mlir/transform/Ch4
{ sed -n '14,36p' ../../test_Example/transform/Ch4/sequence.mlir; cat <<'EOF'

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op {transform.readonly}) {
    // No handles from the command line: find the matmul -> add -> max chain.
    %matmul, %add, %max = transform.collect_matching @match_matmul_elemwise
      in %root : (!transform.any_op)
              -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // From here on, the Chapter 1 schedule, unchanged.
    %tiled, %loop = transform.structured.tile_using_forall %max
        tile_sizes [8, 32]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %add_fused, %loop2 = transform.structured.fuse_into_containing_op %add
        into %loop
        : (!transform.any_op, !transform.any_op)
       -> (!transform.any_op, !transform.any_op)
    %matmul_fused, %loop3 = transform.structured.fuse_into_containing_op
        %matmul into %loop2
        : (!transform.any_op, !transform.any_op)
       -> (!transform.any_op, !transform.any_op)
    transform.yield
  }

  transform.named_sequence @match_matmul_elemwise(
      %last: !transform.any_op {transform.readonly})
      -> (!transform.any_op, !transform.any_op, !transform.any_op) {
    transform.match.operation_name %last ["linalg.elemwise_binary"]
      : !transform.any_op
    %middle = transform.get_producer_of_operand %last[0]
      : (!transform.any_op) -> !transform.any_op
    transform.match.operation_name %middle ["linalg.elemwise_binary"]
      : !transform.any_op
    %matmul = transform.get_producer_of_operand %middle[0]
      : (!transform.any_op) -> !transform.any_op
    transform.match.operation_name %matmul ["linalg.matmul"] : !transform.any_op
    transform.yield %matmul, %middle, %last
      : !transform.any_op, !transform.any_op, !transform.any_op
  }
}
EOF
} | mlir-opt --pass-pipeline='builtin.module(transform-interpreter,canonicalize,cse)' \
  | sed -n '1,22p'
```

Real output **[6]**:

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
```

The `sed -n '1,22p'` keeps the payload function and cuts the transform module that follows it. The result is the tiled and fused loop nest of Chapter 1: one `scf.forall` over 64×16 tiles of 8×32 elements, with the matmul, the add and the max all computed per tile. The command line names no operations: **once matching lives in the script, the script is the complete description of what to transform and how.**

---

## 4. Matching Chains of Operations

### 4.1 A matcher for `max(add(matmul(...)))`

The matchers of section 3 are naive: they match *all* operations of a kind under the payload root. Those operations may or may not be related; they may belong to different functions. Even in one function, if there are several groups of such operations, collecting them by name cannot tell the groups apart. What we really want is a specific group: a `matmul` whose result is used by an elementwise operation, which in turn feeds another elementwise operation.

The following matcher sequence does that:

***test_Example/transform/Ch4/sequence.mlir***
```mlir
  // This is also a matcher sequence. It is similarly given an operation to
  // match and nested operations must succeed in order for a match to be deemed
  // successful. It starts matching from the last operation in the use-def chain
  // and goes back because each operand (use) has exactly one definition.
  transform.named_sequence @match_matmul_elemwise(
      %last: !transform.any_op {transform.readonly}) 
      -> (!transform.any_op, !transform.any_op, !transform.any_op) {
    // The last operation must be an elementwise binary.
    transform.match.operation_name %last ["linalg.elemwise_binary"]
      : !transform.any_op
    // Its first operand must be defined by another operation, to which we
    // will get a handle here. We are guaranteed that the first operand exists
    // because we know the operation is binary, but even in absence of such a
    // guarantee, this operation would have produced a silenceable failure when
    // `%last` does not have enough operands.
    %middle = transform.get_producer_of_operand %last[0]
      : (!transform.any_op) -> !transform.any_op
    // The defining operation must itself be an elementwise binary.
    transform.match.operation_name %middle ["linalg.elemwise_binary"]
      : !transform.any_op
    // And the first operand of that operation must be defined by yet another
    // operation.
    %matmul = transform.get_producer_of_operand %middle[0]
      : (!transform.any_op) -> !transform.any_op
    // And that operation is a matmul.
    transform.match.operation_name %matmul ["linalg.matmul"] : !transform.any_op
    // We will yield the handles to the matmul and the two elementwise
    // operations separately. 
    transform.yield %matmul, %middle, %last
      : !transform.any_op, !transform.any_op, !transform.any_op
  }
```

- The matcher starts from the *last* operation of the chain and walks the use-def chain backwards: each operand has exactly one definition, while a result may have many uses, so going backwards never branches.
- `transform.get_producer_of_operand %last[0]` returns a handle to the operation defining operand #0. It is itself a match-friendly operation: it fails silenceably if the operand does not exist or is a block argument (a function argument has no producer).
- The matcher yields three handles. Applied to a candidate, it returns the triple of *related* operations, rather than operations in the order in which a walk finds them, and it still works when unrelated `elemwise_binary` and `matmul` operations sit next to the chain.

### 4.2 An alternative entry point and the `entry-point` option

The same file holds a second entry point that uses this matcher:

***test_Example/transform/Ch4/sequence.mlir***
```mlir
  // Alternative entry point.
  transform.named_sequence @__transform_main_v2(
      %root: !transform.any_op {transform.readonly}) {
    // Collect groups of operations that match the criteria specified in the
    // named sequence.
    %matmul, %el1, %el2 = transform.collect_matching @match_matmul_elemwise in %root 
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %elemwise = transform.merge_handles %el1, %el2 : !transform.any_op

    transform.include @print_elemwise failures(propagate)  (%elemwise)
      : (!transform.any_op) -> ()
    transform.include @print_matmul failures(propagate)  (%matmul)
      : (!transform.any_op) -> ()

    transform.yield
  }
```

`collect_matching` now has three results, one per value the matcher yields; each match appends to all three. `transform.merge_handles` concatenates the two elementwise handles into one, so the same `@print_elemwise` action can be reused.

The interpreter looks for `@__transform_main` by default. Its `entry-point` option selects another named sequence; the pass also has options to bind trailing arguments (Chapter 1) and to choose the payload root:

```bash
cd /Users/roy/study/mlir/transform/Ch4
./build/transform-opt-ch4 --help | grep -A5 -- '--transform-interpreter '
```

Real output **[7]**:

```text
      --transform-interpreter                                -   transform dialect interpreter
        --debug-bind-trailing-args=<string>                  - Binds trailing arguments of the entry point to the payload operations with specified names.
        --debug-payload-root-tag=<string>                    - Select the operation with 'transform.target_tag' attribute having the given value as payload IR root. If empty select the pass anchor operation as the payload IR root.
        --disable-expensive-checks                           - Disable expensive checks in the interpreter for a faster run.
        --entry-point=<string>                               - Entry point of the pass pipeline.
  --translator-compatibility-mode                            - SPIR-V Translator compatibility mode
```

Running the alternative entry point:

```bash
cd /Users/roy/study/mlir/transform/Ch4
./build/transform-opt-ch4 ../../test_Example/transform/Ch4/sequence.mlir \
  --transform-interpreter=entry-point=__transform_main_v2 -o /dev/null
```

Real output **[8]**:

```text
../../test_Example/transform/Ch4/sequence.mlir:25:13: remark: elementwise binary
  %biased = linalg.elemwise_binary { fun = #linalg.binary_fn<add> }
            ^
../../test_Example/transform/Ch4/sequence.mlir:25:13: note: see current operation: %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%0, %arg2 : tensor<512x512xf32>, tensor<512x512xf32>) outs(%arg3 : tensor<512x512xf32>) -> tensor<512x512xf32>
../../test_Example/transform/Ch4/sequence.mlir:32:13: remark: elementwise binary
  %relued = linalg.elemwise_binary { fun = #linalg.binary_fn<max_signed> }
            ^
../../test_Example/transform/Ch4/sequence.mlir:32:13: note: see current operation: %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<max_signed>} ins(%1, %cst : tensor<512x512xf32>, f32) outs(%arg3 : tensor<512x512xf32>) -> tensor<512x512xf32>
../../test_Example/transform/Ch4/sequence.mlir:20:13: remark: matmul
  %matmul = linalg.matmul ins(%lhs, %rhs: tensor<512x512xf32>, tensor<512x512xf32>)
            ^
../../test_Example/transform/Ch4/sequence.mlir:20:13: note: see current operation: %0 = linalg.matmul ins(%arg0, %arg1 : tensor<512x512xf32>, tensor<512x512xf32>) outs(%arg3 : tensor<512x512xf32>) -> tensor<512x512xf32>
```

On this payload the output is identical to section 3.3: there is only one group, so "all operations of the kind" and "the related triple" coincide. The candidates the chain matcher rejected (the matmul, the add, the constant, the function, the module...) were silenced, as before.

### 4.3 Two groups: where each matcher stops working

The difference shows with two groups. `multiple.mlir` (section 5.4) has two copies of the layer: `@fc_relu_operands_00`, identical to section 3.1, and `@fc_relu_operands_01`, where the max takes the constant first, `max(0, add(...))`. Below, its two functions (lines 1–51) are fed to `sequence.mlir`'s script module (lines 38–139). First, the naive matchers of section 3:

```bash
cd /Users/roy/study/mlir/transform/Ch4
{ sed -n '1,51p' ../../test_Example/transform/Ch4/multiple.mlir
  sed -n '38,139p' ../../test_Example/transform/Ch4/sequence.mlir
} | ./build/transform-opt-ch4 --transform-interpreter -o /dev/null 2>&1 | grep remark
```

Real output **[9]**:

```text
<stdin>:15:13: remark: elementwise binary
<stdin>:22:13: remark: elementwise binary
<stdin>:40:13: remark: elementwise binary
<stdin>:47:13: remark: elementwise binary
<stdin>:10:13: remark: matmul
<stdin>:35:13: remark: matmul
```

All four elementwise operations, then both matmuls: everything of the right kind, with no way to tell which add belongs with which matmul. Now the chain matcher:

```bash
cd /Users/roy/study/mlir/transform/Ch4
{ sed -n '1,51p' ../../test_Example/transform/Ch4/multiple.mlir
  sed -n '38,139p' ../../test_Example/transform/Ch4/sequence.mlir
} | ./build/transform-opt-ch4 --transform-interpreter=entry-point=__transform_main_v2 \
    -o /dev/null 2>&1 | grep remark
```

Real output **[10]**:

```text
<stdin>:15:13: remark: elementwise binary
<stdin>:22:13: remark: elementwise binary
<stdin>:10:13: remark: matmul
```

Only the first function's group (lines 10, 15, 22) matches. In `@fc_relu_operands_01` the max's operand #0 is the constant `%c0f`: `get_producer_of_operand %last[0]` returns the `arith.constant`, whose name is not `linalg.elemwise_binary`, so the match fails, although the computation is the same (max is commutative). **A matcher written with fixed operand positions encodes an accident of how the IR was written, not the property we care about.** Section 5 fixes it with a new match operation.

---

## 5. Defining Match Operations

### 5.1 `has_operand_satisfying`: the ODS definition

The chain matcher is correct in presence of other operations but not robust: `get_producer_of_operand %last[0]` requires the *first* operand of the elementwise operation to be produced by the previous operation, while the same transformation strategy applies regardless of the operand position, since many binary operations are commutative. We want a new match operation that succeeds if *any* of the operands satisfies some conditions, themselves expressed as other match operations, and returns the position of the matched operand plus some state.

Match operations are defined like other transform operations (Chapter 2 introduced the pattern), with one difference: they also implement `MatchOpInterface`. The interface has *no methods* (it may gain some later); it is a verification contract saying that the operation is intended for matching and will not transform the payload. The operation's minimal definition:

***include/MyExtension.td***
```tablegen
include "mlir/Dialect/Transform/Interfaces/MatchInterfaces.td"
include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Define the new operation. By convention, prefix its name with `match`
// followed by the name of the dialect extension.
def HasOperandSatisfyingOp : TransformDialectOp<"match.my.has_operand_satisfying",  // (1)
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,                            // (2)
     DeclareOpInterfaceMethods<TransformOpInterface>,
     // Indicate that the operation implements MatchOpInterface in addition to
     // the TransformOpInterface. This interface is only used as a tag at this
     // point and has no methods that are mandatory to implement.
     MatchOpInterface,                                                               // (3)
     SingleBlockImplicitTerminator<"::mlir::transform::YieldOp">]> {                 // (4)
  let summary = "Succeed if any of the operands matches all nested criteria";
  let arguments = (ins TransformHandleTypeInterface:$op);                            // (5)
  let results = (outs TransformParamTypeInterface:$position,                         // (6)
                      Variadic<Transform_AnyHandleOrParamType>:$results);

  // Match operations can be arbitrarily complex, e.g., containing regions.
  let regions = (region SizedRegion<1>:$body);                                       // (7)
  let hasVerifier = 1;                                                               // (8)
  let assemblyFormat = [{
    $op `:` functional-type($op, results) attr-dict-with-keyword $body
  }];
}
```

- **(1)** `TransformDialectOp` places the operation in the `transform` dialect: its full name is `transform.match.my.has_operand_satisfying`. The `match.` prefix plus the extension name (`my`) is the naming convention for match operations.
- **(2)** `DeclareOpInterfaceMethods<...>` makes ODS declare the interface methods we must write: `getEffects` for `MemoryEffectsOpInterface`, `apply` for `TransformOpInterface`.
- **(3)** `MatchOpInterface`, the tag. Being a plain interface (not `DeclareOpInterfaceMethods`), it adds no method to write.
- **(4)** The body is one block ended by `transform.yield`. The "implicit" terminator means the parser accepts a body without an explicit `transform.yield` (when there is nothing to yield) and inserts one.
- **(5)** The operand: a handle to the payload operation(s) whose operands are examined.
- **(6)** Two results: `$position`, a transform *parameter* holding the matched operand's index, and `$results`, a variadic list of whatever the body yields on success.
- **(7)** One region with exactly one block (`SizedRegion<1>`) that holds the nested match criteria. Its block argument is a handle to the operand being tried.
- **(8)** A hand-written verifier (section 5.3) checks what ODS cannot express: the block argument's type, the number and kinds of the yielded values, and that the body contains only match operations.

So the operation takes the handle to the payload operation whose operands it matches, has a single-block region holding the match criteria, and returns the position of the matched operand together with any transform value yielded from the body on a successful match. The members ODS generates for it are in section 8.4, and its documentation in appendix A.

### 5.2 `apply`: the matching loop

The matching logic is the `apply` method of `TransformOpInterface`, so it composes with other transform operations, and all the facilities for managing the interpreter state and recursively entering blocks are available exactly as for "regular" transform operations. Match operations must return a silenceable failure to signal a failed match and propagate definite failures immediately. Operations that contain nested operations are expected to handle, and in most cases silence, the silenceable failures those produce. For our operation, matching is a loop over all operands of the (single) payload operation that applies the nested transform operations until they all succeed for one operand:

***lib/MyExtension.cpp***
```cpp
// Matcher ops implement `apply` similarly to other transform ops. They are not
// expected to modify payload, but use the tri-state result to signal failure or
// success to match, as well as potential irrecoverable errors.
mlir::DiagnosedSilenceableFailure
mlir::transform::HasOperandSatisfyingOp::apply(
    mlir::transform::TransformRewriter &rewriter,
    mlir::transform::TransformResults &results,
    mlir::transform::TransformState &state) {
  // For simplicity, only handle a single payload op. Actual implementations
  // can use `SingleOpMatcher` trait to simplify implementation and document
  // this expectation.
  auto payloadOps = state.getPayloadOps(getOp());
  if (!llvm::hasSingleElement(payloadOps))                                     // (1)
    return emitSilenceableError() << "expected single payload";

  // Iterate over all operands of the payload op to see if they can be matched
  // using the body of this op.
  Operation *payload = *payloadOps.begin();
  for (OpOperand &operand : payload->getOpOperands()) {
    // Create a scope for transform values defined in the body. This corresponds
    // to the syntactic scope of the region attached to this op. Any values
    // associated with payloads from now on will be automatically dissociated
    // when this object is destroyed, i.e. at the end of the iteration.
    // Associate the block argument handle with the operand.
    auto matchScope = state.make_region_scope(getBody());                      // (2)
    if (failed(state.mapBlockArgument(getBody().getArgument(0),                // (3)
                                      {operand.get()}))) {
      return DiagnosedSilenceableFailure::definiteFailure();
    }

    // Iterate over all nested matchers with the current mapping and see if they
    // succeed.
    bool matchSucceeded = true;
    for (Operation &matcher : getBody().front().without_terminator()) {
      // Matcher ops are applied similarly to any other transform op.
      DiagnosedSilenceableFailure diag =
          state.applyTransform(cast<TransformOpInterface>(matcher));           // (4)

      // Definite failures are immediately propagated as they are irrecoverable.
      if (diag.isDefiniteFailure())                                            // (5)
        return diag;

      // On success, keep checking the remaining conditions.
      if (diag.succeeded())
        continue;

      // Report failure-to-match for debugging purposes and stop matching this
      // operand.
      assert(diag.isSilenceableFailure());
      DEBUG_MATCHER(DBGS_MATCHER()                                             // (6)
                    << "failed to match operand #" << operand.getOperandNumber()
                    << ": " << diag.getMessage());
      (void)diag.silence();
      matchSucceeded = false;
      break;
    }
    // If failed to match this operand, try other operands.
    if (!matchSucceeded)
      continue;

    // If we reached this point, the matching succeeded for the current operand.
    // Remap the values associated with terminator operands to be associated
    // with op results, and also map the parameter result to the operand's
    // position. Note that it is safe to do here despite the end of the scope
    // as `results` are integrated into `state` by the interpreter after `apply`
    // returns rather than immediately.
    SmallVector<SmallVector<MappedValue>> yieldedMappings;
    transform::detail::prepareValueMappings(                                   // (7)
        yieldedMappings, getBody().front().getTerminator()->getOperands(),
        state);
    results.setParams(cast<OpResult>(getPosition()),                           // (8)
                      {rewriter.getI32IntegerAttr(operand.getOperandNumber())});
    for (auto &&[result, mapping] : llvm::zip(getResults(), yieldedMappings))
      results.setMappedValues(result, mapping);
    return DiagnosedSilenceableFailure::success();
  }

  // If we reached this point, none of the operands succeeded the match.
  return emitSilenceableError()                                                // (9)
         << "none of the operands satisfied the conditions";
}
```

- **(1)** The operation handles exactly one payload operation; anything else is "no match", a silenceable error. (The `SingleOpMatcher` trait from `MatchInterfaces.td` would perform this check and pass `apply` the single operation directly; section 8.3.)
- **(2)** `make_region_scope` opens a scope matching the region's syntactic scope. Every transform value associated inside the body is dissociated when `matchScope` is destroyed at the end of the iteration, so each operand starts from a clean state.
- **(3)** The block argument `%operand` (a value handle, `!transform.any_value`) is associated with the payload *value* of the operand being tried.
- **(4)** Nested match operations are applied with `state.applyTransform`, just as the interpreter applies any transform operation.
- **(5)** The tri-state result: a definite failure aborts everything; success moves on to the next condition; a silenceable failure means "this operand does not match".
- **(6)** The failure message goes to the debug stream under the `transform-matcher` debug type (`DEBUG_MATCHER` wraps `DEBUG_WITH_TYPE`, defined at the top of the file), is silenced, and the loop tries the next operand. In the Release build this line compiles to nothing (section 3.4).
- **(7)** On success, `prepareValueMappings` reads what the body's `transform.yield` operands are associated with, and `setMappedValues` forwards those to `$results`.
- **(8)** `$position` is set to the operand number, as an `i32` attribute. Setting results here is safe even though the scope ends right after: the interpreter integrates `results` into the state after `apply` returns.
- **(9)** No operand matched: a silenceable failure, which a combinator such as `foreach_match` silences.

### 5.3 Effects and verification

By convention, operations implementing `MatchOpInterface` must not modify the payload, so their effects say they only read the operand handle and the payload. In this repo:

***lib/MyExtension.cpp***
```cpp
// By convention, operations implementing MatchOpInterface must not modify
// payload IR and must therefore specify that they only read operand handles and
// payload as their effects.
void mlir::transform::HasOperandSatisfyingOp::getEffects(
    llvm::SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  onlyReadsPayload(effects);
  onlyReadsHandle(getOpMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
}
```

The combinator `transform.collect_matching` follows the same convention: it reads its root handle, produces its results and only reads the payload (not this repo's code):

```cpp
void transform::CollectMatchingOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getRoot(), effects);
  producesHandle(getResults(), effects);
  onlyReadsPayload(effects);
}
```

`producesHandle` covers both results, the parameter and the variadic handles. Because nothing is *consumed*, handles passed to a matcher remain valid afterwards, which is what lets `foreach_match` hand the same operations to the action sequence (appendix B shows a side effect of "only reads").

The verifier, requested by `hasVerifier = 1`, checks the body:

***lib/MyExtension.cpp***
```cpp
// Verify well-formedness of the operation and emit diagnostics if it is
// ill-formed.
llvm::LogicalResult mlir::transform::HasOperandSatisfyingOp::verify() {
  mlir::Block &bodyBlock = getBody().front();
  if (bodyBlock.getNumArguments() != 1 ||                                      // (1)
      !isa<TransformValueHandleTypeInterface>(
          bodyBlock.getArgument(0).getType())) {
    return emitOpError()
           << "expects the body to have one value handle argument";
  }
  if (bodyBlock.getTerminator()->getNumOperands() != getNumResults() - 1) {    // (2)
    return emitOpError() << "expects the body to yield "
                         << (getNumResults() - 1) << " values, got "
                         << bodyBlock.getTerminator()->getNumOperands();
  }
  for (auto &&[i, operand, result] :                                           // (3)
       llvm::enumerate(bodyBlock.getTerminator()->getOperands().getTypes(),
                       getResults().getTypes())) {
    if (implementSameTransformInterface(operand, result))
      continue;
    return emitOpError() << "expects terminator operand #" << i
                         << " and result #" << (i + 1)
                         << " to implement the same transform interface";
  }

  for (Operation &op : bodyBlock.without_terminator()) {                       // (4)
    if (!isa<TransformOpInterface>(op) || !isa<MatchOpInterface>(op)) {
      InFlightDiagnostic diag = emitOpError()
                                << "expects body to contain match ops";
      diag.attachNote(op.getLoc()) << "non-match operation";
      return diag;
    }
  }

  return success();
}
```

- **(1)** One block argument, of a value-handle type: it is mapped to an operand *value* in `apply`.
- **(2)** The body yields one value per result except `$position`, which `apply` fills itself.
- **(3)** Each yielded value and its result are both handles, both parameters, or both value handles (`implementSameTransformInterface`, a small helper above `apply`).
- **(4)** Only match operations may appear in the body: this is where the `MatchOpInterface` tag pays off. A transforming operation inside a matcher would change the payload while `apply` is still iterating over its operands.

Two bodies that break rules (4) and (2), in one block. The first puts `transform.structured.generalize`, which rewrites the payload, in the body; the second yields nothing although the operation has one result after `$position`:

```bash
cd /Users/roy/study/mlir/transform/Ch4
for body in 'transform.structured.generalize %def : (!transform.any_op) -> !transform.any_op
      transform.yield %def : !transform.any_op' 'transform.yield'; do
./build/transform-opt-ch4 -o /dev/null <<EOF
module attributes { transform.with_named_sequence } {
  transform.named_sequence @m(%last: !transform.any_op {transform.readonly}) {
    %pos, %d = transform.match.my.has_operand_satisfying %last
        : (!transform.any_op) -> (!transform.param<i32>, !transform.any_op) {
    ^bb0(%operand: !transform.any_value):
      %def = transform.get_defining_op %operand
        : (!transform.any_value) -> !transform.any_op
      $body
    }
    transform.yield
  }
}
EOF
echo "transform-opt-ch4 exit status: $?"
done
```

Real output **[11]**:

```text
<stdin>:3:16: error: 'transform.match.my.has_operand_satisfying' op expects body to contain match ops
    %pos, %d = transform.match.my.has_operand_satisfying %last
               ^
<stdin>:3:16: note: see current operation: 
%0:2 = "transform.match.my.has_operand_satisfying"(%arg0) ({
^bb0(%arg1: !transform.any_value):
  %1 = "transform.get_defining_op"(%arg1) : (!transform.any_value) -> !transform.any_op
  %2 = "transform.structured.generalize"(%1) : (!transform.any_op) -> !transform.any_op
  "transform.yield"(%1) : (!transform.any_op) -> ()
}) : (!transform.any_op) -> (!transform.param<i32>, !transform.any_op)
<stdin>:8:7: note: non-match operation
      transform.structured.generalize %def : (!transform.any_op) -> !transform.any_op
      ^
transform-opt-ch4 exit status: 1
<stdin>:3:16: error: 'transform.match.my.has_operand_satisfying' op expects the body to yield 1 values, got 0
    %pos, %d = transform.match.my.has_operand_satisfying %last
               ^
<stdin>:3:16: note: see current operation: 
%0:2 = "transform.match.my.has_operand_satisfying"(%arg0) ({
^bb0(%arg1: !transform.any_value):
  %1 = "transform.get_defining_op"(%arg1) : (!transform.any_value) -> !transform.any_op
  "transform.yield"() : () -> ()
}) : (!transform.any_op) -> (!transform.param<i32>, !transform.any_op)
transform-opt-ch4 exit status: 1
```

Both are rejected while parsing, before any interpreter runs. The `note: see current operation` shows the operation in generic form; in the second one the implicit terminator is visible as an explicit `"transform.yield"() : () -> ()`. Note that `transform.debug.emit_remark_at` would *not* be rejected: the debug operations implement `MatchOpInterface` too, since they only read, which is why matchers may use them for tracing.

### 5.4 Registering the operation

The operation joins the `transform` dialect through a dialect extension, as in Chapter 2. `init` lists the operations generated from ODS:

***lib/MyExtension.cpp***
```cpp
void MyExtension::init() {
  // Register the additional match operations with the dialect similarly to
  // other transform operations. List all operations generated from ODS. This
  // call will perform additional checks that the operations implement the
  // transform and memory effect interfaces required by the dialect interpreter
  // and assert if they do not.
  registerTransformOps<
#define GET_OP_LIST
#include "MyExtension.cpp.inc"
      >();
}
...
void registerMyExtension(::mlir::DialectRegistry &registry) {
  registry.addExtensions<MyExtension>();
}
```

The driver registers all upstream dialects and extensions, then this one, plus the interpreter pass and a few cleanup passes:

***transform-opt/transform-opt.cpp***
```cpp
int main(int argc, char **argv) {
  // Register all "core" dialects and our transform dialect extension.
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  registerMyExtension(registry);

  // Register a handful of cleanup passes that we can run to make the output IR
  // look nicer.
  mlir::registerCanonicalizerPass();
  mlir::registerCSEPass();
  mlir::registerSymbolDCEPass();
  mlir::transform::registerInterpreterPass();

  // Delegate to the MLIR utility for parsing and pass management.
  return mlir::MlirOptMain(argc, argv, "transform-opt-ch4", registry)
                 .succeeded()
             ? EXIT_SUCCESS
             : EXIT_FAILURE;
}
```

Without the extension, the operation does not exist. The stock `mlir-opt`, which ran `sequence.mlir` in section 3.3, cannot even parse `multiple.mlir`:

```bash
cd /Users/roy/study/mlir/transform/Ch4
mlir-opt ../../test_Example/transform/Ch4/multiple.mlir --transform-interpreter -o /dev/null
```

Real output **[12]** (exit status 1):

```text
../../test_Example/transform/Ch4/multiple.mlir:107:21: error: custom op 'transform.match.my.has_operand_satisfying' is unknown
    %pos, %middle = transform.match.my.has_operand_satisfying %last
                    ^
```

### 5.5 The commutative matcher and `transform.foreach_match`

With the operation registered, the matcher can say that *either* operand of the "max" may be produced by the previous elementwise operation. For simplicity, the previous operation still requires the matmul to produce its first operand. This is the updated matcher sequence:

***test_Example/transform/Ch4/multiple.mlir***
```mlir
  transform.named_sequence @match_matmul_elemwise(
      %last: !transform.any_op {transform.readonly}) 
      -> (!transform.any_op, !transform.any_op, !transform.any_op,
          !transform.param<i32>) {
    // The last operation must be an elementwise binary.
    transform.match.operation_name %last ["linalg.elemwise_binary"]
      : !transform.any_op

    // One of its operands must be defined by another operation, to which we
    // will get a handle here. This is achieved thanks to a newly defined
    // operation that tries to match operands one by one using the match
    // operations nested in its region.
    %pos, %middle = transform.match.my.has_operand_satisfying %last
        : (!transform.any_op) -> (!transform.param<i32>, !transform.any_op) {
    ^bb0(%operand: !transform.any_value):
      // The operand must be defined by an operation.
      %def = transform.get_defining_op %operand 
        : (!transform.any_value) -> !transform.any_op
      // The defining operation must itself be an elementwise binary.
      transform.match.operation_name %def ["linalg.elemwise_binary"]
        : !transform.any_op
      transform.yield %def : !transform.any_op
    }
    
    // And the first operand of that operation must be defined by yet another
    // operation.
    %matmul = transform.get_producer_of_operand %middle[0]
      : (!transform.any_op) -> !transform.any_op
    // And that operation is a matmul.
    transform.match.operation_name %matmul ["linalg.matmul"] : !transform.any_op
    // We will yield the handles to the matmul and the two elementwise
    // operations separately. 
    transform.yield %matmul, %middle, %last, %pos
      : !transform.any_op, !transform.any_op, !transform.any_op,
        !transform.param<i32>
  }
```

Inside the body, `%operand` is a *value* handle; `transform.get_defining_op` turns it into a handle to its defining operation (and fails silenceably for a block argument). This matches both `max(add(matmul(...), bias), 0)` and `max(0, add(matmul(...), bias))` with the same sequence. `%pos` is a transform dialect **parameter** (`!transform.param<i32>`): parameters hold lists of entities known to be constant throughout the transform application, most often numbers, but generally any MLIR attribute. Unlike handles, they point to nothing in the payload and are never invalidated.

To show that groups are matched independently of each other, the entry point uses `transform.foreach_match`. It implements a simple high-level pattern rewriting approach inside the transform dialect (for advanced or lower-level pattern rewriting, consider PDL (Pattern Descriptor Language), PDLL, or the C++ rewriting APIs): it maps a matcher named sequence to an action named sequence and invokes the action whenever the matcher succeeds:

***test_Example/transform/Ch4/multiple.mlir***
```mlir
  transform.named_sequence @__transform_main(
      %root: !transform.any_op {transform.consumed}) {

    // Traverses the payload IR associated with the operand handle, invoking
    // @match_matmul_elemwise on each of the operations. If the named sequence
    // succeeds, i.e., if none of the nested match (transform) operations
    // produced a silenceable failure, invokes @print_matmul_elemwise and
    // forwards the values yielded as arguments of the new invocation. If the
    // named sequence fails with a silenceable failure, silences it (the message
    // is forwarded to the debug stream). Definite failures are propagated
    // immediately and unconditionally, as usual.
    transform.foreach_match in %root
      @match_matmul_elemwise -> @print_matmul_elemwise
      : (!transform.any_op) -> !transform.any_op

    transform.yield
  }

  // This is an action sequence.
  transform.named_sequence @print_matmul_elemwise(
      %matmul: !transform.any_op {transform.readonly},
      %add: !transform.any_op {transform.readonly},
      %max: !transform.any_op {transform.readonly},
      %pos: !transform.param<i32> {transform.readonly}) {
    transform.debug.emit_param_as_remark %pos, "matmul #" at %matmul
      : !transform.param<i32>, !transform.any_op
    transform.debug.emit_param_as_remark %pos, "add #" at %add
      : !transform.param<i32>, !transform.any_op
    transform.debug.emit_param_as_remark %pos, "max #" at %max
      : !transform.param<i32>, !transform.any_op
    transform.yield
  }
```

- The action's arguments are exactly the matcher's results, in order: three operation handles and the parameter. `foreach_match` calls the action **once per match**, so inside it the handles refer to one group only, unlike `collect_matching`, which concatenates all matches.
- `foreach_match` *consumes* `%root` (actions may rewrite the payload, so the old handle could become stale) and returns a fresh handle to the updated root; hence `{transform.consumed}` on the entry point's argument.
- `transform.debug.emit_param_as_remark` prints the parameter's value after the message, at the given operations.

The payload is two copies of the layer; the second swaps the max's operands:

***test_Example/transform/Ch4/multiple.mlir***
```mlir
// Matmul+ReLU with swapped operands.
func.func @fc_relu_operands_01(
    %lhs: tensor<512x512xf32>, %rhs: tensor<512x512xf32>,
    %bias: tensor<512x512xf32>, %output: tensor<512x512xf32>)
    -> tensor<512x512xf32> {
  // Matrix-matrix multiplication.
  // expected-remark @below {{matmul # 1}}
  %matmul = linalg.matmul ins(%lhs, %rhs: tensor<512x512xf32>, tensor<512x512xf32>)
                          outs(%output: tensor<512x512xf32>) -> tensor<512x512xf32>

  // Elementwise addition.
  // expected-remark @below {{add # 1}}
  %biased = linalg.elemwise_binary { fun = #linalg.binary_fn<add> }
    ins(%matmul, %bias : tensor<512x512xf32>, tensor<512x512xf32>)
    outs(%output : tensor<512x512xf32>) -> tensor<512x512xf32>

  // Elementwise max with 0 (ReLU).
  %c0f = arith.constant 0.0 : f32
  // expected-remark @below {{max # 1}}
  %relued = linalg.elemwise_binary { fun = #linalg.binary_fn<max_signed> }
    ins(%c0f, %biased : f32, tensor<512x512xf32>)
    outs(%output : tensor<512x512xf32>) -> tensor<512x512xf32>
  func.return %relued : tensor<512x512xf32>
}
```

`@fc_relu_operands_00`, above it in the file, is section 3.1's function with the remarks `matmul # 0`, `add # 0`, `max # 0`. The action uses the operand position to tell the two groups apart:

```bash
cd /Users/roy/study/mlir/transform/Ch4
./build/transform-opt-ch4 ../../test_Example/transform/Ch4/multiple.mlir \
  --transform-interpreter -o /dev/null
```

Real output **[13]**:

```text
../../test_Example/transform/Ch4/multiple.mlir:10:13: remark: matmul # 0 : i32
  %matmul = linalg.matmul ins(%lhs, %rhs: tensor<512x512xf32>, tensor<512x512xf32>)
            ^
../../test_Example/transform/Ch4/multiple.mlir:15:13: remark: add # 0 : i32
  %biased = linalg.elemwise_binary { fun = #linalg.binary_fn<add> }
            ^
../../test_Example/transform/Ch4/multiple.mlir:22:13: remark: max # 0 : i32
  %relued = linalg.elemwise_binary { fun = #linalg.binary_fn<max_signed> }
            ^
../../test_Example/transform/Ch4/multiple.mlir:35:13: remark: matmul # 1 : i32
  %matmul = linalg.matmul ins(%lhs, %rhs: tensor<512x512xf32>, tensor<512x512xf32>)
            ^
../../test_Example/transform/Ch4/multiple.mlir:40:13: remark: add # 1 : i32
  %biased = linalg.elemwise_binary { fun = #linalg.binary_fn<add> }
            ^
../../test_Example/transform/Ch4/multiple.mlir:47:13: remark: max # 1 : i32
  %relued = linalg.elemwise_binary { fun = #linalg.binary_fn<max_signed> }
            ^
```

- Both groups match now, including the swapped one that section 4.3's chain matcher missed.
- Each group's three remarks carry *its own* position: `0` for the first function (the add is operand #0 of the max), `1` for the second. The parameter is printed with its type, `0 : i32`.
- The remarks come group by group (matmul, add, max of the first function, then of the second), because the action runs once per match.
- `emit_param_as_remark` adds no `see current operation` note, unlike `emit_remark_at`.

### 5.6 Calling the new operation directly

As in section 3.4, applying the operation outside a matcher makes its silenceable failures visible. Below, a new entry point `@direct` is appended to `multiple.mlir`'s script. It first finds the second function's max (`structured.match` with an attribute filter, then `split_handle` to take the second of the two), runs the whole matcher on it with `failures(propagate)`, and emits the returned position:

```bash
cd /Users/roy/study/mlir/transform/Ch4
{ sed '$d' ../../test_Example/transform/Ch4/multiple.mlir; cat <<'EOF'
  transform.named_sequence @direct(
      %root: !transform.any_op {transform.readonly}) {
    %maxes = transform.structured.match ops{["linalg.elemwise_binary"]}
      attributes{fun = #linalg.binary_fn<max_signed>} in %root
      : (!transform.any_op) -> !transform.any_op
    %m0, %m1 = transform.split_handle %maxes
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %matmul, %add, %max, %pos = transform.include @match_matmul_elemwise
      failures(propagate) (%m1)
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op,
                                !transform.any_op, !transform.param<i32>)
    transform.debug.emit_param_as_remark %pos, "operand position" at %max
      : !transform.param<i32>, !transform.any_op
    transform.yield
  }
}
EOF
} | ./build/transform-opt-ch4 --transform-interpreter=entry-point=direct -o /dev/null
```

Real output **[14]**:

```text
<stdin>:47:13: remark: operand position 1 : i32
  %relued = linalg.elemwise_binary { fun = #linalg.binary_fn<max_signed> }
            ^
```

The match succeeds on operand #1. Applied to an operation none of whose operands has a producer, the same operation fails with the message of section 5.2's last `return`. Here it runs on the first matmul, whose operands are all function arguments, so `get_defining_op` fails for each of them:

```bash
cd /Users/roy/study/mlir/transform/Ch4
{ sed '$d' ../../test_Example/transform/Ch4/multiple.mlir; cat <<'EOF'
  transform.named_sequence @direct(
      %root: !transform.any_op {transform.readonly}) {
    %matmuls = transform.structured.match ops{["linalg.matmul"]} in %root
      : (!transform.any_op) -> !transform.any_op
    %first, %second = transform.split_handle %matmuls
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %pos, %def = transform.match.my.has_operand_satisfying %first
        : (!transform.any_op) -> (!transform.param<i32>, !transform.any_op) {
    ^bb0(%operand: !transform.any_value):
      %d = transform.get_defining_op %operand
        : (!transform.any_value) -> !transform.any_op
      transform.yield %d : !transform.any_op
    }
    transform.yield
  }
}
EOF
} | ./build/transform-opt-ch4 --transform-interpreter=entry-point=direct -o /dev/null
```

Real output **[15]** (exit status 1):

```text
<stdin>:137:18: error: none of the operands satisfied the conditions
    %pos, %def = transform.match.my.has_operand_satisfying %first
                 ^
```

**The new operation is a few dozen lines of C++, yet it plugs into the language: nested matchers, parameters, `foreach_match` and the interpreter's failure handling all work with it unchanged, because it implements the same interfaces as every upstream operation.**

---

## 6. Matchers for Inferred Features

### 6.1 Why operation names are not enough

The matchers so far, although useful to drive transformations from within the interpreter, are basic: they rely on operation names and use-def chains, and an implementation through C++ APIs or declarative rewrite rules would be barely less expressive and sometimes more concise. The real power of transform dialect match operations is that they can match **inferred properties** of the payload: properties that are not directly available as an attribute of an operation, nor as a straightforward relation between IR components.

A small change to the example shows why. If the matrix multiplication is written as a special case of tensor contraction with `linalg.generic` instead of `linalg.matmul`, a name-based matcher no longer applies. Yet this form is very common, both in the original input and during transformation, for example when a higher-dimensional contraction is decomposed into loops around a matrix multiplication. To be a (potentially transposed) matrix multiplication, a `linalg.generic` must have these features:

- Total rank of 3.
- Two inputs accessed as projected permutation of iteration dimensions.
- One output accessed as projected permutation of iteration dimensions.
- Iteration dimensions can be subdivided into LHS (left-hand side) parallel, RHS (right-hand side) parallel and reduction dimensions.
- The body block consists of a multiplication and an addition.

A *projected permutation* is an indexing map whose results are distinct loop dimensions, with some dimensions possibly dropped: `(d0, d1, d2) -> (d0, d2)` is one, `(d0, d1, d2) -> (d0 + d1)` or `-> (d0, d0)` are not.

Most of these features can be derived from the operation's properties (the total rank is the number of entries in `iterator_types`), but almost none of them can be read directly from the IR or checked in a declarative form, which is usually limited to the presence or exact value of an attribute or a type. The transform dialect lets such features be implemented in the `apply` method of a match operation and reused across matchers. For structured linear algebra payload operations, many of these match operations are ready-made in the `structured` extension.

### 6.2 A matmul matcher built from `structured` match operations

They are enough to write the matrix multiplication matcher almost verbatim from the list above:

***test_Example/transform/Ch4/features.mlir***
```mlir
  transform.named_sequence @match_generic_matmul(
      %candidate: !transform.any_op {transform.readonly}) -> !transform.any_op {
    // Match a structured linear algebra operation.
    transform.match.structured %candidate : !transform.any_op {                  // (1)
    ^bb0(%c: !transform.any_op):
      // With a rank equal to 3.
      %rank = transform.match.structured.rank %c                                 // (2)
        : (!transform.any_op) -> !transform.param<i64>
      %c3 = transform.param.constant 3 : i64 -> !transform.param<i64>
      transform.match.param.cmpi eq %rank, %c3 : !transform.param<i64>

      // With 2 inputs.
      %n_ins = transform.match.structured.num_inputs %c
        : (!transform.any_op) -> !transform.param<i64>
      %c2 = transform.param.constant 2 : i64 -> !transform.param<i64>
      transform.match.param.cmpi eq %n_ins, %c2 : !transform.param<i64>

      // With 1 output (note that structured ops in destination passing style
      // has as many inits as outputs).
      %n_inits = transform.match.structured.num_inits %c
        : (!transform.any_op) -> !transform.param<i64>
      %c1 = transform.param.constant 1 : i64 -> !transform.param<i64>
      transform.match.param.cmpi eq %n_inits, %c1 : !transform.param<i64>

      // All inputs and inits are accessed with a projected permutation.
      transform.match.structured.input %c[all] {projected_permutation}           // (3)
        : !transform.any_op
      transform.match.structured.init %c[0] {projected_permutation}
        : !transform.any_op

      // The body is a mulf/addf contraction with appropriate dimensions.
      transform.match.structured.body %c                                         // (4)
        { contraction = ["arith.mulf", "arith.addf"] } : !transform.any_op
      %batch, %lhs, %rhs, %reduction =
      transform.match.structured.classify_contraction_dims %c                    // (5)
        : (!transform.any_op)
        -> (!transform.param<i64>, !transform.param<i64>, !transform.param<i64>,
            !transform.param<i64>)

      // There is one of lhs, rhs and reduction dimensions and zero batch
      // dimensions.
      %n_batch = transform.num_associations %batch                               // (6)
        : (!transform.param<i64>) -> !transform.param<i64>
      %n_lhs = transform.num_associations %lhs
        : (!transform.param<i64>) -> !transform.param<i64>
      %n_rhs = transform.num_associations %rhs
        : (!transform.param<i64>) -> !transform.param<i64>
      %n_reduction = transform.num_associations %reduction
        : (!transform.param<i64>) -> !transform.param<i64>
      %c0 = transform.param.constant 0 : i64 -> !transform.param<i64>
      transform.match.param.cmpi eq %n_batch, %c0 : !transform.param<i64>
      transform.match.param.cmpi eq %n_lhs, %c1 : !transform.param<i64>
      transform.match.param.cmpi eq %n_rhs, %c1 : !transform.param<i64>
      transform.match.param.cmpi eq %n_reduction, %c1 : !transform.param<i64>
    }
    transform.yield %candidate : !transform.any_op
  }
```

- **(1)** `transform.match.structured` succeeds only on operations implementing Linalg's structured-operation interface (`LinalgOp`), named ones like `linalg.matmul` as well as `linalg.generic`, and applies the nested `match.structured.*` operations to it, much like `has_operand_satisfying` applies its body to an operand.
- **(2)** The inferred features come back as *parameters*: `structured.rank` returns the number of loops, which is compared with a constant through `transform.param.constant` and `transform.match.param.cmpi`, a match operation on parameters.
- **(3)** `structured.input %c[all]` checks every input operand (`init %c[0]`, the first init) and succeeds only if its indexing map is a projected permutation.
- **(4)** `structured.body {contraction = [...]}` checks that the body multiplies two block arguments with `arith.mulf` and accumulates the product with `arith.addf`.
- **(5)** `classify_contraction_dims` splits the loop dimensions into batch, LHS-parallel (M), RHS-parallel (N) and reduction (K) dimensions; each result is a parameter associated with one value per dimension of its class, and the operation fails if the operation is not a contraction at all.
- **(6)** `transform.num_associations` counts how many values a parameter holds: a plain matmul has no batch dimension and exactly one dimension in each of the other three classes.

### 6.3 Running it: named and generic matmuls

The payload holds the same matrix multiplication twice, as `linalg.matmul` and as `linalg.generic`:

***test_Example/transform/Ch4/features.mlir***
```mlir
// Matmul as a generic operation.
func.func @generic(
    %lhs: tensor<512x512xf32>, %rhs: tensor<512x512xf32>,
    %bias: tensor<512x512xf32>, %output: tensor<512x512xf32>)
    -> tensor<512x512xf32> {
  // expected-remark @below {{matmul}}
  %matmul = linalg.generic {
    iterator_types = ["parallel", "parallel", "reduction"],
    indexing_maps = [
      affine_map<(d0, d1, d2) -> (d0, d2)>,
      affine_map<(d0, d1, d2) -> (d2, d1)>,
      affine_map<(d0, d1, d2) -> (d0, d1)>]
  } ins(%lhs, %rhs: tensor<512x512xf32>, tensor<512x512xf32>)
    outs(%output: tensor<512x512xf32>) {
  ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
    %0 = arith.mulf %arg0, %arg1 : f32
    %1 = arith.addf %0, %arg2 : f32
    linalg.yield %1 : f32
  } -> tensor<512x512xf32>
  return %matmul : tensor<512x512xf32>
}
```

The `@named` function above it holds the `linalg.matmul` form. The entry point is the same `foreach_match` as in section 5.5, now mapping `@match_generic_matmul` to an action `@print_generic_matmul` that emits the remark `matmul`:

```bash
cd /Users/roy/study/mlir/transform/Ch4
./build/transform-opt-ch4 ../../test_Example/transform/Ch4/features.mlir \
  --transform-interpreter -o /dev/null
```

Real output **[16]**:

```text
../../test_Example/transform/Ch4/features.mlir:9:13: remark: matmul
  %matmul = linalg.matmul ins(%lhs, %rhs: tensor<512x512xf32>, tensor<512x512xf32>)
            ^
../../test_Example/transform/Ch4/features.mlir:9:13: note: see current operation: %0 = linalg.matmul ins(%arg0, %arg1 : tensor<512x512xf32>, tensor<512x512xf32>) outs(%arg3 : tensor<512x512xf32>) -> tensor<512x512xf32>
../../test_Example/transform/Ch4/features.mlir:20:13: remark: matmul
  %matmul = linalg.generic {
            ^
../../test_Example/transform/Ch4/features.mlir:20:13: note: see current operation: 
%0 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<512x512xf32>, tensor<512x512xf32>) outs(%arg3 : tensor<512x512xf32>) {
^bb0(%in: f32, %in_0: f32, %out: f32):
  %1 = arith.mulf %in, %in_0 : f32
  %2 = arith.addf %1, %out : f32
  linalg.yield %2 : f32
} -> tensor<512x512xf32>
```

Both forms match: `linalg.matmul` is a structured operation with exactly these features, so it passes the same checks as the generic one. The `see current operation` note prints the generic in full, region included.

### 6.4 What the matcher accepts and rejects, and why

Three more payloads test the edges. `@transposed_b` reads the RHS transposed (`(d1, d2)` instead of `(d2, d1)`), which is still "a potentially transposed matrix multiplication". `@batch` is a `linalg.batch_matmul`, with rank 4. `@all_parallel` is the generic matmul with one change, the third iterator is `"parallel"` instead of `"reduction"`. The shell function `input` prints the payloads, `features.mlir`'s matchers (lines 36–122) and two extra entry points; the command feeds it to the default entry point and to the two explaining ones, which `include` the matcher with `failures(propagate)` (section 3.4):

```bash
cd /Users/roy/study/mlir/transform/Ch4
input() {
  cat <<'EOF'
// C = A * B^T: the RHS is read transposed, (d1, d2) instead of (d2, d1).
func.func @transposed_b(%a: tensor<8x10xf32>, %b: tensor<16x10xf32>,
                        %c: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %0 = linalg.generic {
    iterator_types = ["parallel", "parallel", "reduction"],
    indexing_maps = [
      affine_map<(d0, d1, d2) -> (d0, d2)>,
      affine_map<(d0, d1, d2) -> (d1, d2)>,
      affine_map<(d0, d1, d2) -> (d0, d1)>]
  } ins(%a, %b : tensor<8x10xf32>, tensor<16x10xf32>)
    outs(%c : tensor<8x16xf32>) {
  ^bb0(%x: f32, %y: f32, %acc: f32):
    %m = arith.mulf %x, %y : f32
    %s = arith.addf %m, %acc : f32
    linalg.yield %s : f32
  } -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// Rank 4: a batched matmul.
func.func @batch(%a: tensor<2x8x10xf32>, %b: tensor<2x10x16xf32>,
                 %c: tensor<2x8x16xf32>) -> tensor<2x8x16xf32> {
  %0 = linalg.batch_matmul ins(%a, %b : tensor<2x8x10xf32>, tensor<2x10x16xf32>)
                           outs(%c : tensor<2x8x16xf32>) -> tensor<2x8x16xf32>
  return %0 : tensor<2x8x16xf32>
}

// The generic matmul with d2 marked "parallel" instead of "reduction".
func.func @all_parallel(%a: tensor<8x10xf32>, %b: tensor<10x16xf32>,
                        %c: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %0 = linalg.generic {
    iterator_types = ["parallel", "parallel", "parallel"],
    indexing_maps = [
      affine_map<(d0, d1, d2) -> (d0, d2)>,
      affine_map<(d0, d1, d2) -> (d2, d1)>,
      affine_map<(d0, d1, d2) -> (d0, d1)>]
  } ins(%a, %b : tensor<8x10xf32>, tensor<10x16xf32>)
    outs(%c : tensor<8x16xf32>) {
  ^bb0(%x: f32, %y: f32, %acc: f32):
    %m = arith.mulf %x, %y : f32
    %s = arith.addf %m, %acc : f32
    linalg.yield %s : f32
  } -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}
EOF
  sed -n '36,122p' ../../test_Example/transform/Ch4/features.mlir
  cat <<'EOF'
  transform.named_sequence @why_batch(%root: !transform.any_op {transform.readonly}) {
    %op = transform.structured.match ops{["linalg.batch_matmul"]} in %root
      : (!transform.any_op) -> !transform.any_op
    transform.include @match_generic_matmul failures(propagate) (%op)
      : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
  transform.named_sequence @why_all_parallel(%root: !transform.any_op {transform.readonly}) {
    %funcs = transform.structured.match ops{["func.func"]} in %root
      : (!transform.any_op) -> !transform.any_op
    %f0, %f1, %f2 = transform.split_handle %funcs
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %op = transform.structured.match ops{["linalg.generic"]} in %f2
      : (!transform.any_op) -> !transform.any_op
    transform.include @match_generic_matmul failures(propagate) (%op)
      : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
EOF
}
input | ./build/transform-opt-ch4 --transform-interpreter -o /dev/null 2>&1 | grep -A2 remark
echo '--- why_batch:'
input | ./build/transform-opt-ch4 --transform-interpreter=entry-point=why_batch -o /dev/null
echo '--- why_all_parallel:'
input | ./build/transform-opt-ch4 --transform-interpreter=entry-point=why_all_parallel -o /dev/null
```

Real output **[17]** (exit status 1):

```text
<stdin>:4:8: remark: matmul
  %0 = linalg.generic {
       ^
--- why_batch:
<stdin>:85:7: error: expected parameter to be equal to 3, got 4
      transform.match.param.cmpi eq %rank, %c3 : !transform.param<i64>
      ^
<stdin>:82:15: note: value # 0 associated with the parameter defined here
      %rank = transform.match.structured.rank %c
              ^
--- why_all_parallel:
<stdin>:129:7: error: expected parameter to be equal to 1, got 0
      transform.match.param.cmpi eq %n_reduction, %c1 : !transform.param<i64>
      ^
<stdin>:123:22: note: value # 0 associated with the parameter defined here
      %n_reduction = transform.num_associations %reduction
                     ^
```

- Only `@transposed_b` (line 4) gets the `matmul` remark: a transposed operand is still a projected permutation, and the dimensions still classify as one M, one N and one K.
- `@batch` fails the first check: the rank parameter holds 4, not 3. The error is at the `cmpi` that compared it (line 85), and the note points at line 82, the `structured.rank` that produced the parameter.
- `@all_parallel` has the same operation name, the same operands, the same indexing maps and the same `mulf`/`addf` body as the matmul; only one iterator type differs. It passes every check up to the last one: line 129 is `transform.match.param.cmpi eq %n_reduction, %c1`, and the parameter defined at line 123 (`%n_reduction`) holds 0: with no reduction iterator there is no reduction dimension.
- The block's exit status, 1, is the last run's.

**A name or an attribute can be checked declaratively; "is this a matmul" is a computed property, and a match operation is the place where that computation is written once and reused by every script.**

While this example uses contraction-specific matchers with a rather non-trivial C++ implementation, the transform dialect is flexible enough to express this reasoning directly if desired. One could, for example, obtain the access map of each input as a parameter and extract the accessed dimensions as other parameters, which can then be compared with each other to ensure the subscripts are `m,k` for the LHS, `k,n` for the RHS and `m,n` for the init/result, given the `m,n,k` notation for loops.

---

## 7. Build and Run

The top-level [README](../README.md) describes the repository and its other chapters; this chapter builds on its own.

### 7.1 Building

```bash
cd /Users/roy/study/mlir/transform/Ch4
cmake -S . -B build -G Ninja
cmake --build build          # → ./build/transform-opt-ch4
```

No `-DMLIR_DIR` or compiler flags are needed: the shell puts `/opt/homebrew/opt/llvm@20/bin` on `PATH` and sets `CC`/`CXX` to Homebrew's llvm@20 `clang`/`clang++`, so `find_package(MLIR)` finds `/opt/homebrew/opt/llvm@20/lib/cmake/mlir`. Otherwise, add `-DMLIR_DIR=/opt/homebrew/opt/llvm@20/lib/cmake/mlir -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm@20/bin/clang++`. The build produces `./build/transform-opt-ch4` (not `build/bin/`). Appendix C explains what the chapter's CMake files do.

`transform-opt-ch4` is an `mlir-opt`-style driver (`MlirOptMain`): it takes an input file (or stdin), pass flags such as `--transform-interpreter[=entry-point=NAME]`, and the usual `-o`, `--verify-diagnostics`, `--split-input-file` options.

### 7.2 Running the three tests

Each test file declares its expected remarks with `// expected-remark @below {{...}}` comments, and `--verify-diagnostics` checks that the run emits exactly those diagnostics on exactly those lines (then the remarks are not printed, and the exit status says whether they matched). These are the four `RUN` lines of the three files:

```bash
cd /Users/roy/study/mlir/transform/Ch4
T=../../test_Example/transform/Ch4
./build/transform-opt-ch4 $T/sequence.mlir --transform-interpreter \
  --verify-diagnostics -o /dev/null && echo "sequence.mlir: ok"
./build/transform-opt-ch4 $T/sequence.mlir \
  --transform-interpreter='entry-point=__transform_main_v2' \
  --verify-diagnostics -o /dev/null && echo "sequence.mlir (v2): ok"
./build/transform-opt-ch4 $T/multiple.mlir --transform-interpreter \
  --verify-diagnostics -o /dev/null && echo "multiple.mlir: ok"
./build/transform-opt-ch4 $T/features.mlir --transform-interpreter \
  --verify-diagnostics -o /dev/null && echo "features.mlir: ok"
```

Real output **[18]**:

```text
sequence.mlir: ok
sequence.mlir (v2): ok
multiple.mlir: ok
features.mlir: ok
```

The test files' `RUN` lines omit `-o /dev/null` and print the IR to stdout as well; the transforms of this chapter only emit remarks, so that IR is the input unchanged. Sections 3.3, 4.2, 5.5 and 6.3 show the same runs without `--verify-diagnostics`, with their remarks.

---

## 8. Deriving the match operation and its build yourself

How would you arrive at Chapter 4's code without having it in front of you? The chapter replaces Chapter 3's two operations and its handle type with one match operation; the driver and the build stay Chapter 3's, minus the type. Each part has a source you can query:

| Part | Lines | Where it comes from |
|---|---|---|
| What is new, what is dropped from Chapter 3 | — | A `diff` against Chapter 3 (section 8.1) |
| What the body of `has_operand_satisfying` contains and yields | — | The stock-operation chain matcher, prototyped with `mlir-opt` for each operand position (section 8.2) |
| `include/MyExtension.td`: the `include`s and the trait list (`TransformDialectOp`, `MatchOpInterface`, `SingleBlockImplicitTerminator`) | 17–32 | `MatchInterfaces.td` and `TransformDialect.td` in the installed headers (section 8.3) |
| `include/MyExtension.td`: operand, results, region, `hasVerifier`, assembly format | 33–43 | The prototype: one handle in, the operand position and the yielded handles out (sections 5.1, 8.2) |
| `lib/MyExtension.cpp`: the methods to define, `apply`, `getEffects`, `verify` | 78–205 | `mlir-tblgen -gen-op-decls` (section 8.4), and the interpreter and effect helpers of `TransformInterfaces.h` (section 8.5) |
| The checks in `verify` | 172–205 | The bodies that must be rejected (section 5.3) |
| `lib/MyExtension.cpp`: the debug macros | 16–20 | `DEBUG_WITH_TYPE` from `llvm/Support/Debug.h` (sections 3.4, 5.2) |
| `include/MyExtension.h`: `#include "mlir/Dialect/Transform/IR/TransformOps.h"` | 16 | The compiler error without it (section 8.6) |
| `lib/MyExtension.cpp`: `init` and `registerMyExtension`; `transform-opt/transform-opt.cpp`; the CMake files | 46–56, 207–209 | Chapter 3's, without the type (sections 8.1, 5.4, appendix C) |

### 8.1 Step 1: see what changes from Chapter 3

Chapter 3's files are the starting point. A `diff` of the two chapters, of the driver, and of the definitions each file declares or registers:

```bash
cd /Users/roy/study/mlir/transform/Ch4
diff -rq ../Ch3 . -x build -x README.md -x examples.sh
diff ../Ch3/transform-opt/transform-opt.cpp transform-opt/transform-opt.cpp
grep -nE '^def |registerTypes<|declareGeneratedDialect<|mlir_tablegen' \
  ../Ch3/include/*.td ../Ch3/include/CMakeLists.txt ../Ch3/lib/MyExtension.cpp \
  include/*.td include/CMakeLists.txt lib/MyExtension.cpp
```

Real output **[19]**:

```text
Files ../Ch3/CMakeLists.txt and ./CMakeLists.txt differ
Files ../Ch3/include/CMakeLists.txt and ./include/CMakeLists.txt differ
Files ../Ch3/include/MyExtension.h and ./include/MyExtension.h differ
Files ../Ch3/include/MyExtension.td and ./include/MyExtension.td differ
Only in ../Ch3/include: MyExtensionTypes.td
Files ../Ch3/lib/CMakeLists.txt and ./lib/CMakeLists.txt differ
Files ../Ch3/lib/MyExtension.cpp and ./lib/MyExtension.cpp differ
Files ../Ch3/transform-opt/transform-opt.cpp and ./transform-opt/transform-opt.cpp differ
9c9
< // This is the top-level file for the Transform dialect tutorial chapter 3.
---
> // This is the top-level file for the Transform dialect tutorial chapter 4.
31,33d30
<   // Register the interpreter pass.
<   mlir::transform::registerInterpreterPass();
< 
38a36
>   mlir::transform::registerInterpreterPass();
41c39
<   return mlir::MlirOptMain(argc, argv, "transform-opt-ch3", registry)
---
>   return mlir::MlirOptMain(argc, argv, "transform-opt-ch4", registry)
../Ch3/include/MyExtension.td:26:def ChangeCallTargetOp : Op<Transform_Dialect, "my.change_call_target",
../Ch3/include/MyExtension.td:71:def CallToOp : Op<Transform_Dialect, "my.call_to_op",
../Ch3/include/MyExtensionTypes.td:22:def CallOpInterfaceHandle
../Ch3/include/CMakeLists.txt:5:mlir_tablegen(MyExtension.h.inc -gen-op-decls)
../Ch3/include/CMakeLists.txt:6:mlir_tablegen(MyExtension.cpp.inc -gen-op-defs)
../Ch3/include/CMakeLists.txt:12:mlir_tablegen(MyExtensionTypes.h.inc -gen-typedef-decls)
../Ch3/include/CMakeLists.txt:13:mlir_tablegen(MyExtensionTypes.cpp.inc -gen-typedef-defs)
../Ch3/lib/MyExtension.cpp:70:  declareGeneratedDialect<::mlir::scf::SCFDialect>();
../Ch3/lib/MyExtension.cpp:71:  declareGeneratedDialect<::mlir::func::FuncDialect>();
../Ch3/lib/MyExtension.cpp:75:  registerTypes<
include/MyExtension.td:25:def HasOperandSatisfyingOp : TransformDialectOp<"match.my.has_operand_satisfying",
include/CMakeLists.txt:5:mlir_tablegen(MyExtension.h.inc -gen-op-decls)
include/CMakeLists.txt:6:mlir_tablegen(MyExtension.cpp.inc -gen-op-defs)
```

- **Every file changes, one disappears:** Chapter 3's two operations and its `CallOpInterfaceHandle` type (with `MyExtensionTypes.td`) give way to one operation, `HasOperandSatisfyingOp`. Everything that served the type goes too: the two `-gen-typedef-*` rules in `include/CMakeLists.txt`, `registerTypes<...>` in `init`, and the `GET_TYPEDEF_CLASSES` includes.
- **No generated dialects:** a match operation creates nothing, so `init` drops Chapter 3's `declareGeneratedDialect<scf>`/`<func>` and only registers the operation (section 5.4).
- **The driver** is Chapter 3's: the only change besides the name is where `registerInterpreterPass()` sits among the registrations, which doesn't change what is registered.
- **The rest of the CMake changes** are the chapter number in target names (appendix C).

### 8.2 Step 2: prototype with stock operations first

Before writing C++, write the matcher with stock operations and run it with the stock `mlir-opt`. Section 4.1's chain matcher is that prototype; section 4.3 showed that it matches only the first function of `multiple.mlir`, whose max takes the add's result as operand #0. The same run with the stock tool, once with the matcher as written and once with the max's operand position changed to #1 by `sed`:

```bash
cd /Users/roy/study/mlir/transform/Ch4
for i in 0 1; do
  echo "operand #$i:"
  { sed -n '1,51p' ../../test_Example/transform/Ch4/multiple.mlir
    sed -n '38,139p' ../../test_Example/transform/Ch4/sequence.mlir \
      | sed "s/get_producer_of_operand %last\[0\]/get_producer_of_operand %last[$i]/"
  } | mlir-opt --transform-interpreter=entry-point=__transform_main_v2 -o /dev/null 2>&1 | grep remark
done
```

Real output **[20]**:

```text
operand #0:
<stdin>:15:13: remark: elementwise binary
<stdin>:22:13: remark: elementwise binary
<stdin>:10:13: remark: matmul
operand #1:
<stdin>:40:13: remark: elementwise binary
<stdin>:47:13: remark: elementwise binary
<stdin>:35:13: remark: matmul
```

- Each position catches exactly one of the two groups: `[0]` the first function (lines 10, 15, 22), `[1]` the commuted one (lines 35, 40, 47). With stock operations, "either operand" takes one copy of the matcher per position.
- The new operation is that set of copies folded into a loop: its body is the part of the prototype that follows `get_producer_of_operand %last[i]` (`get_defining_op` of the operand being tried, then `match.operation_name`), its `$position` result reports which `i` matched, and its `$results` carry what the body yields, here the middle operation (section 5.5).
- The stock tool runs the whole prototype, so the operations the body will contain are already known to work; only the loop needs C++.

**Prototype the matcher with stock operations first; the C++ operation is only the part the stock ones cannot say, here "any operand".**

### 8.3 Step 3: find the match interface and its traits

The interface is in the installed headers, next to traits that document and enforce how many payload objects a matcher expects; the base class the record uses is in the dialect's own `.td` file:

```bash
cd /Users/roy/study/mlir/transform/Ch4
grep -n -A1 '^def ' /opt/homebrew/opt/llvm@20/include/mlir/Dialect/Transform/Interfaces/MatchInterfaces.td
grep -n -A1 '^class TransformDialectOp' /opt/homebrew/opt/llvm@20/include/mlir/Dialect/Transform/IR/TransformDialect.td
```

Real output **[21]**:

```text
12:def MatchOpInterface
13-    : OpInterface<"MatchOpInterface", [TransformOpInterface]> {
--
23:def AtMostOneOpMatcher : NativeOpTrait<"AtMostOneOpMatcherOpTrait"> {
24-  let cppNamespace = "::mlir::transform";
--
40:def SingleOpMatcher : NativeOpTrait<"SingleOpMatcherOpTrait"> {
41-  let cppNamespace = "::mlir::transform";
--
57:def SingleValueMatcher : NativeOpTrait<"SingleValueMatcherOpTrait"> {
58-  let cppNamespace = "::mlir::transform";
158:class TransformDialectOp<string mnemonic, list<Trait> traits = []>
159-    : Op<Transform_Dialect, mnemonic, traits>;
```

`MatchOpInterface` derives from `TransformOpInterface`, so a match operation must also be a transform operation. `SingleOpMatcher` (and `SingleValueMatcher`) would replace the "expected single payload" check of section 5.2 by a trait that calls a `matchOperation(Operation *, ...)` method instead of `apply`; this chapter writes `apply` by hand to show the whole mechanism. `TransformDialectOp<"...">` is shorthand for Chapter 3's `Op<Transform_Dialect, "...">`. Chapter 3's `TransformEachOpTrait` doesn't fit either: it calls `applyToOne` once per payload operation, while this operation has to drive its own region once per operand, so it declares `TransformOpInterface`'s `apply` itself (`DeclareOpInterfaceMethods`).

### 8.4 Step 4: let `mlir-tblgen` list the methods you owe

With the ODS record written (section 5.1), the generated class declaration shows what is left to implement in C++. The `sed` keeps the class and `grep` its members that have no inline body:

```bash
cd /Users/roy/study/mlir/transform/Ch4
mlir-tblgen -gen-op-decls -I /opt/homebrew/opt/llvm@20/include include/MyExtension.td \
  | sed -n '/^class HasOperandSatisfyingOp :/,/^};/p' \
  | grep -E '^  (::llvm::LogicalResult verify\(|void getEffects|::mlir::DiagnosedSilenceableFailure apply|::mlir::TypedValue.*getPosition|::mlir::Operation::result_range getResults|::mlir::Region &getBody)'
```

Real output **[22]**:

```text
  ::mlir::TypedValue<::mlir::transform::TransformParamTypeInterface> getPosition() {
  ::mlir::Operation::result_range getResults() {
  ::mlir::Region &getBody() {
  ::llvm::LogicalResult verify();
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
  ::mlir::DiagnosedSilenceableFailure apply(::mlir::transform::TransformRewriter &rewriter, ::mlir::transform::TransformResults &transformResults, ::mlir::transform::TransformState &state);
```

- `verify()` comes from `hasVerifier = 1`, `getEffects` from `DeclareOpInterfaceMethods<MemoryEffectsOpInterface>`, `apply` from `DeclareOpInterfaceMethods<TransformOpInterface>`: these three are the definitions `lib/MyExtension.cpp` must provide (leaving one out compiles, then fails to link, section 8.6).
- `getPosition()`, `getResults()` and `getBody()` are the accessors `apply` uses, named after `$position`, `$results` and `$body`; `getPosition()` returns a typed value, hence the `cast<OpResult>` in section 5.2's marker (8).

### 8.5 Step 5: find the interpreter helpers

`apply` needs to scope the body's values, bind its block argument, apply nested operations and forward the yielded values; `getEffects` needs the standard effect helpers. All are declared in one installed header:

```bash
cd /Users/roy/study/mlir/transform/Ch4
grep -n -E 'make_region_scope\(|LogicalResult mapBlockArgument\(|^void prepareValueMappings|applyTransform\(TransformOpInterface|void setParams\(|void setMappedValues\(|^void (onlyReadsPayload|onlyReadsHandle|producesHandle)\(' \
  /opt/homebrew/opt/llvm@20/include/mlir/Dialect/Transform/Interfaces/TransformInterfaces.h
```

Real output **[23]**:

```text
67:void prepareValueMappings(
317:  DiagnosedSilenceableFailure applyTransform(TransformOpInterface transform);
332:  LogicalResult mapBlockArgument(BlockArgument argument,
344:  inline RegionScope make_region_scope(Region &region);
848:  void setParams(OpResult value, ArrayRef<TransformState::Param> params);
882:  void setMappedValues(OpResult handle, ArrayRef<MappedValue> values);
937:TransformState::RegionScope TransformState::make_region_scope(Region &region) {
1269:void producesHandle(ResultRange handles,
1271:void producesHandle(MutableArrayRef<BlockArgument> handles,
1273:void onlyReadsHandle(MutableArrayRef<OpOperand> handles,
1282:void onlyReadsPayload(SmallVectorImpl<MemoryEffects::EffectInstance> &effects);
```

These declarations are the calls of section 5.2's markers (2), (3), (4), (7), (8) and of `getEffects` in section 5.3.

### 8.6 Step 6: let the compiler and the linker find what is missing

Two details are hard to guess and easy to let the tools find: a header the generated code needs, and a method body that is still owed. A scratch copy of the chapter (in a temporary directory; each build takes about 15 seconds) first without `MyExtension.h`'s `TransformOps.h` include, then with the include back but without the `verify` definition, both removed by `sed`; then the installed header that declares the missing class:

```bash
cd /Users/roy/study/mlir/transform/Ch4
S=$(mktemp -d)
cp -R CMakeLists.txt include lib transform-opt "$S"/
sed -i '' '/#include "mlir\/Dialect\/Transform\/IR\/TransformOps.h"/d' "$S"/include/MyExtension.h
cmake -S "$S" -B "$S"/build -G Ninja >/dev/null
cmake --build "$S"/build 2>&1 | grep -m1 'error:' | sed "s|$S/||"
cp include/MyExtension.h "$S"/include/
sed -i '' '/^llvm::LogicalResult mlir::transform::HasOperandSatisfyingOp::verify/,/^}/d' "$S"/lib/MyExtension.cpp
cmake --build "$S"/build 2>&1 | grep -E '^Undefined|^  "|^ld:'
rm -rf "$S"
I=/opt/homebrew/opt/llvm@20/include/mlir/Dialect/Transform/IR
grep -n '^class YieldOp ' $I/TransformOps.h.inc | cut -c1-60
grep -n 'TransformOps.h.inc' $I/TransformOps.h
```

Real output **[24]**:

```text
build/include/MyExtension.h.inc:99:303: error: no member named 'YieldOp' in namespace 'mlir::transform'
Undefined symbols for architecture arm64:
  "mlir::transform::HasOperandSatisfyingOp::verify()", referenced from:
ld: symbol(s) not found for architecture arm64
8007:class YieldOp : public ::mlir::Op<YieldOp, ::mlir::OpTr
45:#include "mlir/Dialect/Transform/IR/TransformOps.h.inc"
```

- **The include:** `SingleBlockImplicitTerminator<"::mlir::transform::YieldOp">` makes the generated class refer to `transform::YieldOp`, which nothing in Chapter 3 needed. That class is generated into the installed `TransformOps.h.inc`, which `TransformOps.h` includes; that is the header `MyExtension.h` adds. A trait that names a C++ class needs that class's declaration in scope wherever the generated header is included.
- **The missing body:** without `verify`, every file compiles, because the generated class only *declares* it (section 8.4); the linker is the first to notice, and it names the method. Leaving out `apply` or `getEffects` fails the same way.

---

## 9. Key Takeaways & Pitfalls

**Takeaways**

1. **Match operations are transform operations that only read**: they implement `MatchOpInterface`, never modify the payload, and report "no match" as a *silenceable* failure that combinators catch (section 2).
2. **Matchers and actions are named sequences**: `collect_matching` gathers the matches of a matcher into handles; `foreach_match` invokes an action once per match with the matcher's results as arguments (sections 3 and 5.5).
3. **The script is self-contained**: with matchers, no pass option or C++ call has to tell the interpreter what to transform, and Chapter 1's schedule runs unchanged on matched handles (section 3.5).
4. **Match related groups, not kinds**: walk use-def chains back from the last operation; but fixed operand positions encode accidents of the IR, which is why the chain matcher missed the commutated `max` (section 4.3).
5. **A new match operation is an ordinary transform operation plus a tag**: ODS with `MatchOpInterface`, an `apply` that loops and silences, read-only effects and a verifier; it composes with nested matchers and parameters with no further glue (section 5).
6. **Parameters carry constant facts**: `!transform.param<...>` values such as the operand position or a rank flow between matchers and actions and can be compared with `match.param.cmpi` (sections 5.5 and 6.2).
7. **Inferred properties are where match operations shine**: "is a possibly transposed matmul" is computed from ranks, indexing maps, the body and the dimension classes, not read from a name (section 6).
8. **Design the matcher with stock operations first**: prototyped with stock `mlir-opt`, the matcher shows that the only missing piece is "any operand"; the installed headers, `mlir-tblgen`, and the compiler and linker errors of a scratch build then give everything needed to write the new operation (section 8).

**Pitfalls**

- **No `-debug-only` in a Release build**: the Homebrew MLIR rejects the option. Call the matcher yourself with `transform.include ... failures(propagate)` to see why it rejects a candidate (section 3.4).
- **A matcher argument must be `{transform.readonly}`**: matchers are called on every candidate; consuming the handle would invalidate it for the action.
- **`foreach_match` consumes its root**: the entry point's argument must be `{transform.consumed}`, and later operations must use the handle `foreach_match` returns (section 5.5).
- **Operand positions are part of a stock matcher**: `get_producer_of_operand %op[0]` silently skips groups whose operands are commuted (section 4.3).
- **Non-match operations in a match body are rejected**: `has_operand_satisfying`'s verifier only accepts match operations, so a transforming operation must go in the action, not the matcher (section 5.3).
- **Don't canonicalize the transform script before interpreting it**: match and debug operations only read and have no results, so the canonicalizer deletes them as dead code, and the matchers then match everything (appendix B).
- **The custom operation needs the custom tool**: the stock `mlir-opt` fails to parse `multiple.mlir` (section 5.4).

---

## Appendix A: Autogenerated documentation

`mlir-tblgen -gen-op-doc` turns the ODS record of section 5.1 into the operation's reference documentation, the same Markdown the build's `add_mlir_doc` rule produces (appendix C.1):

```bash
cd /Users/roy/study/mlir/transform/Ch4
mlir-tblgen -gen-op-doc -I /opt/homebrew/opt/llvm@20/include include/MyExtension.td
```

Real output **[25]**:

~~~text
<!-- Autogenerated by mlir-tblgen; don't manually edit -->
### `transform.match.my.has_operand_satisfying` (transform::HasOperandSatisfyingOp)

_Succeed if any of the operands matches all nested criteria_


Syntax:

```
operation ::= `transform.match.my.has_operand_satisfying` $op `:` functional-type($op, results) attr-dict-with-keyword $body
```


Traits: `SingleBlockImplicitTerminator<::mlir::transform::YieldOp>`, `SingleBlock`

Interfaces: `MatchOpInterface`, `MemoryEffectOpInterface`, `TransformOpInterface`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `op` | TransformHandleTypeInterface instance

#### Results:

| Result | Description |
| :----: | ----------- |
| `position` | TransformParamTypeInterface instance
| `results` | variadic of any transform handle or parameter

~~~

The syntax line is the `assemblyFormat`; the traits and interfaces come from the trait list (the `MatchOpInterface` tag among them); the operand and result tables come from `arguments` and `results`, described by their type constraints.

---

## Appendix B: Match operations look dead to the canonicalizer

Match and debug operations declare only read effects and, like `transform.match.operation_name`, often have no results. To the canonicalizer, an operation without results whose only effects are reads is trivially dead, and it erases it. Running `--canonicalize` on `sequence.mlir` (without the interpreter) empties the matchers and the actions:

```bash
cd /Users/roy/study/mlir/transform/Ch4
./build/transform-opt-ch4 ../../test_Example/transform/Ch4/sequence.mlir --canonicalize \
  | sed -n '/^  module @transforms/,$p'
```

Real output **[26]**:

```text
  module @transforms attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.collect_matching @match_elemwise in %arg0 : (!transform.any_op) -> !transform.any_op
      %1 = transform.collect_matching @match_matmul in %arg0 : (!transform.any_op) -> !transform.any_op
      transform.include @print_elemwise failures(propagate) (%0) : (!transform.any_op) -> ()
      transform.include @print_matmul failures(propagate) (%1) : (!transform.any_op) -> ()
      transform.yield 
    }
    transform.named_sequence @__transform_main_v2(%arg0: !transform.any_op {transform.readonly}) {
      %0:3 = transform.collect_matching @match_matmul_elemwise in %arg0 : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
      %1 = transform.merge_handles %0#1, %0#2 : !transform.any_op
      transform.include @print_elemwise failures(propagate) (%1) : (!transform.any_op) -> ()
      transform.include @print_matmul failures(propagate) (%0#0) : (!transform.any_op) -> ()
      transform.yield 
    }
    transform.named_sequence @match_elemwise(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op {
      transform.yield %arg0 : !transform.any_op
    }
    transform.named_sequence @match_matmul(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op {
      transform.yield %arg0 : !transform.any_op
    }
    transform.named_sequence @print_elemwise(%arg0: !transform.any_op {transform.readonly}) {
      transform.yield 
    }
    transform.named_sequence @print_matmul(%arg0: !transform.any_op {transform.readonly}) {
      transform.yield 
    }
    transform.named_sequence @match_matmul_elemwise(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_op, !transform.any_op) {
      %0 = transform.get_producer_of_operand %arg0[0] : (!transform.any_op) -> !transform.any_op
      %1 = transform.get_producer_of_operand %0[0] : (!transform.any_op) -> !transform.any_op
      transform.yield %1, %0, %arg0 : !transform.any_op, !transform.any_op, !transform.any_op
    }
  }
}

```

Compare with the input of section 3.2: `@match_elemwise` and `@match_matmul` lost their `transform.match.operation_name` (they now match *every* operation), `@print_elemwise` and `@print_matmul` lost their `emit_remark_at`, and `@match_matmul_elemwise` kept only the `get_producer_of_operand` operations, which have results that are used. Interpreting this module would match the wrong operations and print nothing. This is why section 3.5 runs `canonicalize,cse` *after* `transform-interpreter` in the pipeline: there the cleanups apply to the transformed payload, and the transform module they also touch is no longer needed.

---

## Appendix C: What Chapter 4 adds to the build

### C.1 TableGen and the library

***include/CMakeLists.txt***
```cmake
# Tell Tablegen to use MyExtension.td as input.
set(LLVM_TARGET_DEFINITIONS MyExtension.td)

# Ask Tablegen to generate op declarations and definitions from ODS.
mlir_tablegen(MyExtension.h.inc -gen-op-decls)
mlir_tablegen(MyExtension.cpp.inc -gen-op-defs)

# Add a CMakeTarget we can depend on to ensure the generation happens before the
# compilation.
add_public_tablegen_target(MyExtensionCh4IncGen)

# Don't forget to generate the documentation, this will produce a
# MyExtensionCh4.md under Tutorials/transform
add_mlir_doc(MyExtension MyExtensionCh4 Tutorials/transform/ -gen-op-doc)
```

`mlir_tablegen` runs `mlir-tblgen` on `MyExtension.td` into `build/include/MyExtension.h.inc` (the class declaration, included by `MyExtension.h`) and `MyExtension.cpp.inc` (the definitions, plus the `GET_OP_LIST` used by `MyExtension::init`). `add_mlir_doc` registers the `-gen-op-doc` output of appendix A with the `mlir-doc` target, which is not part of the default build. Chapter 4 has no types, so unlike Chapter 3 there is no `MyExtensionTypes.td` rule.

***lib/CMakeLists.txt***
```cmake
# Outside examples, this should be `add_mlir_library`.
add_mlir_example_library(
  # Library called MyExtension.
  MyExtensionCh4

  # Built from the following source files.
  MyExtension.cpp

  # Make includes visible without top-level path.
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/examples/transform/Ch4/include

  # Make sure ODS declaration and definitions are generated before compiling this.
  DEPENDS
  MyExtensionCh4IncGen

  # Link in the transform dialect, an all generated dialects.
  LINK_LIBS PRIVATE
  MLIR                      # libMLIR.dylib
)
```

`DEPENDS MyExtensionCh4IncGen` makes sure the `.inc` files exist before `MyExtension.cpp` is compiled. The library links only `libMLIR.dylib`, which contains the transform dialect and everything else; linking static MLIR archives as well would put two copies of MLIR's type identifiers in one process.

### C.2 The tool and the standalone guard

***CMakeLists.txt***
```cmake
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_subdirectory(include)
add_subdirectory(lib)

add_dependencies(TransformExample transform-opt-ch4)
add_llvm_example(transform-opt-ch4
  transform-opt/transform-opt.cpp)

target_link_libraries(transform-opt-ch4
  PRIVATE
  MLIR                      # libMLIR.dylib (all dialects, passes, extensions)
  MyExtensionCh4
)
```

The include directories make both the sources (`include/MyExtension.h`) and the generated `build/include/MyExtension.h.inc` reachable. The block above this excerpt runs only when the chapter is configured on its own (`cmake -S .` from `Ch4/`): it calls `find_package(MLIR)` and includes the LLVM/MLIR CMake modules, turns `LLVM_BUILD_EXAMPLES` on (otherwise `add_llvm_example` leaves the tool out of the default build), defines the umbrella target `TransformExample`, and defaults the build type to Release. The last point matters: `libMLIR.dylib` is built with `NDEBUG`, and MLIR's headers reference debug-only helpers (such as the check that registered operations implement `TransformOpInterface`) under `#ifndef NDEBUG`, so a non-Release chapter build fails to link.

---

## Appendix D: Running this README's commands: `examples.sh`

[`examples.sh`](examples.sh), a symlink to the shared [`../examples.sh`](../examples.sh), runs the commands of this README and shows each one with its output and exit status. Build the chapter first (section 7.1), then:

- `./examples.sh` runs every command, in README order;
- `./examples.sh 7` runs command [7] only (several numbers are fine);
- `./examples.sh -l` lists the numbered commands without running them:

```bash
cd /Users/roy/study/mlir/transform/Ch4
./examples.sh -l
```

Real output **[27]**:

```text
[ 1] 2.1   ./build/transform-opt-ch4 --transform-interpreter -o /dev/null <<'EOF'
[ 2] 3.3   ./build/transform-opt-ch4 ../../test_Example/transform/Ch4/sequence.mlir \
[ 3] 3.3   mlir-opt ../../test_Example/transform/Ch4/sequence.mlir \
[ 4] 3.4   ./build/transform-opt-ch4 ../../test_Example/transform/Ch4/sequence.mlir \
[ 5] 3.4   { sed '$d' ../../test_Example/transform/Ch4/sequence.mlir; cat <<'EOF'
[ 6] 3.5   { sed -n '14,36p' ../../test_Example/transform/Ch4/sequence.mlir; cat <<'EOF'
[ 7] 4.2   ./build/transform-opt-ch4 --help | grep -A5 -- '--transform-interpreter '
[ 8] 4.2   ./build/transform-opt-ch4 ../../test_Example/transform/Ch4/sequence.mlir \
[ 9] 4.3   { sed -n '1,51p' ../../test_Example/transform/Ch4/multiple.mlir
[10] 4.3   { sed -n '1,51p' ../../test_Example/transform/Ch4/multiple.mlir
[11] 5.3   for body in 'transform.structured.generalize %def : (!transform.any_op) -> !transform.any_op
[12] 5.4   mlir-opt ../../test_Example/transform/Ch4/multiple.mlir --transform-interpreter -o /dev/null
[13] 5.5   ./build/transform-opt-ch4 ../../test_Example/transform/Ch4/multiple.mlir \
[14] 5.6   { sed '$d' ../../test_Example/transform/Ch4/multiple.mlir; cat <<'EOF'
[15] 5.6   { sed '$d' ../../test_Example/transform/Ch4/multiple.mlir; cat <<'EOF'
[16] 6.3   ./build/transform-opt-ch4 ../../test_Example/transform/Ch4/features.mlir \
[17] 6.4   input() {
[18] 7.2   T=../../test_Example/transform/Ch4
[19] 8.1   diff -rq ../Ch3 . -x build -x README.md -x examples.sh
[20] 8.2   for i in 0 1; do
[21] 8.3   grep -n -A1 '^def ' /opt/homebrew/opt/llvm@20/include/mlir/Dialect/Transform/Interfaces/MatchInterfaces.td
[22] 8.4   mlir-tblgen -gen-op-decls -I /opt/homebrew/opt/llvm@20/include include/MyExtension.td \
[23] 8.5   grep -n -E 'make_region_scope\(|LogicalResult mapBlockArgument\(|^void prepareValueMappings|applyTransform\(TransformOpInterface|void setParams\(|void setMappedValues\(|^void (onlyReadsPayload|onlyReadsHandle|producesHandle)\(' \
[24] 8.6   S=$(mktemp -d)
[25] A     mlir-tblgen -gen-op-doc -I /opt/homebrew/opt/llvm@20/include include/MyExtension.td
[26] B     ./build/transform-opt-ch4 ../../test_Example/transform/Ch4/sequence.mlir --canonicalize \
[27] D     ./examples.sh -l
```

The script doesn't keep its own copy of the commands: it reads them from `README.md` on every run. A command is every `bash` code block in file order, except the build block (its first command after `cd` is `cmake`) and file excerpts (a `***path***` label above the fence). Number *N* is the *N*-th such block, and the README marks that block's output "Real output **[N]**". Adding, removing or reordering a command block therefore renumbers the ones after it.

---

## Links

- Official doc: [Transform Dialect Tutorial Chapter 4 — Matching Payload with Transform Operations](https://mlir.llvm.org/docs/Tutorials/transform/Ch4/)
- Related MLIR docs: [Transform Dialect](https://mlir.llvm.org/docs/Dialects/Transform/) · [Operation Definition Specification (ODS)](https://mlir.llvm.org/docs/DefiningDialects/Operations/)
- Previous: [Chapter 3 — More than Simple Transform Operations](../Ch3/README.md)
- Next: [Chapter H — Reproducing Halide Schedule](../ChH/README.md)
- Back to [README](../README.md)
