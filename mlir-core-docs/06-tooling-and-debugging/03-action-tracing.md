# Action: Tracing and Debugging MLIR-based Compilers

> **Section:** Tooling and Debugging · document 3 of 7  
> **Upstream:** [https://mlir.llvm.org/docs/ActionTracing/](https://mlir.llvm.org/docs/ActionTracing/) · source [`mlir/docs/ActionTracing.md`](https://github.com/llvm/llvm-project/blob/main/mlir/docs/ActionTracing.md)  
> **License:** upstream text is Apache-2.0 WITH LLVM-exception.

## Orientation

An abstraction over "things the compiler does": a pass running, a pattern applying, a fold happening.
Because actions are a first-class concept, external code can observe them, log them, count them, or
*intervene* — stopping execution after N actions.

That last capability is the reason to care. Bisecting a miscompilation by action count localises a
bug to a single rewrite, which is a far sharper tool than bisecting by pass.

**Read first**

- [Pass Infrastructure](../03-passes-and-rewriting/01-pass-infrastructure.md)
- [Pattern Rewriting](../03-passes-and-rewriting/03-pattern-rewriting.md)

**What you should be able to do after this page**

- Trace what the compiler is doing at a finer grain than passes.
- Bisect a miscompilation down to a single rewrite.
- Wrap your own transformations as actions.

---

## Upstream documentation

See also the [slides](https://mlir.llvm.org/OpenMeetings/2023-02-23-Actions.pdf)
and the [recording](https://youtu.be/ayQSyekVa3c) from the MLIR Open Meeting
where this feature was demoed.

## Overview

`Action` are means to encapsulate any transformation of any granularity in a way
that can be intercepted by the framework for debugging or tracing purposes,
including skipping a transformation programmatically (think about "compiler
fuel" or "debug counters" in LLVM). As such, "executing a pass" is an Action, so
is "try to apply one canonicalization pattern", or "tile this loop".

In MLIR, passes and patterns are the main abstractions to encapsulate general IR
transformations. The primary way of observing transformations along the way is
to enable “debug printing” of the IR (e.g. -mlir-print-ir-after-all to print
after each pass execution). On top of this, finer grain tracing may be available
with -debug which enables more detailed logs from the transformations
themselves. However, this method has some scaling issues: it is limited to a
single stream of text that can be gigantic and requires tedious crawling through
this log a posteriori. Iterating through multiple runs of collecting such logs
and analyzing it can be very time consuming and often not very practical beyond
small input programs.

The `Action` framework doesn't make any assumptions about how the higher level
driver is controlling the execution, it merely provides a framework for
connecting the two together. A high level overview of the workflow surrounding
`Action` execution is shown below:

- Compiler developer defines an `Action` class, that is representing the
  transformation or utility that they are developing.
- Depending on the needs, the developer identifies single unit of
  transformations, and dispatch them to the `MLIRContext` for execution.
- An external entity registers an "action handler" with the action manager, and
  provides the logic surrounding the transformation execution.

The exact definition of an `external entity` is left opaque, to allow for more
interesting handlers.

## Wrapping a Transformation in an Action

There are two parts for getting started with enabling tracing through Action in
existing or new code: 1) defining an actual `Action` class, and 2) encapsulating
the transformation in a lambda function.

There are no constraints on the granularity of an “action”, it can be as simple
as “perform this fold” and as complex as “run this pass pipeline”. An action is
comprised of the following:

```c++
/// A custom Action can be defined minimally by deriving from
/// `tracing::ActionImpl`.
class MyCustomAction : public tracing::ActionImpl<MyCustomAction> {
public:
  using Base = tracing::ActionImpl<MyCustomAction>;
  /// Actions are initialized with an array of IRUnit (that is either Operation,
  /// Block, or Region) that provide context for the IR affected by a transformation.
  MyCustomAction(ArrayRef<IRUnit> irUnits)
      : Base(irUnits) {}
  /// This tag should uniquely identify this action, it can be matched for filtering
  /// during processing.
  static constexpr StringLiteral tag = "unique-tag-for-my-action";
  static constexpr StringLiteral desc =
      "This action will encapsulate a some very specific transformation";
};
```

Any transformation can then be dispatched with this `Action` through the
`MLIRContext`:

```c++
context->executeAction<ApplyPatternAction>(
    [&]() {
      rewriter.setInsertionPoint(op);

      ...
    },
    /*IRUnits=*/{op, region});
```

An action can also carry arbitrary payload, for example we can extend the
`MyCustomAction` class above with the following member:

```c++
/// A custom Action can be defined minimally by deriving from
/// `tracing::ActionImpl`. It can have any members!
class MyCustomAction : public tracing::ActionImpl<MyCustomAction> {
public:
  using Base = tracing::ActionImpl<MyCustomAction>;
  /// Actions are initialized with an array of IRUnit (that is either Operation,
  /// Block, or Region) that provide context for the IR affected by a transformation.
  /// Other constructor arguments can also be required here.
  MyCustomAction(ArrayRef<IRUnit> irUnits, int count, PaddingStyle padding)
      : Base(irUnits), count(count), padding(padding) {}
  /// This tag should uniquely identify this action, it can be matched for filtering
  /// during processing.
  static constexpr StringLiteral tag = "unique-tag-for-my-action";
  static constexpr StringLiteral desc =
      "This action will encapsulate a some very specific transformation";
  /// Extra members can be carried by the Action
  int count;
  PaddingStyle padding;
};
```

These new members must then be passed as arguments when dispatching an `Action`:

```c++
context->executeAction<ApplyPatternAction>(
    [&]() {
      rewriter.setInsertionPoint(op);

      ...
    },
    /*IRUnits=*/{op, region},
    /*count=*/count,
    /*padding=*/padding);
```

## Intercepting Actions

When a transformation is executed through an `Action`, it can be directly
intercepted via a handler that can be set on the `MLIRContext`:

```c++
  /// Signatures for the action handler that can be registered with the context.
  using HandlerTy =
      std::function<void(function_ref<void()>, const tracing::Action &)>;

  /// Register a handler for handling actions that are dispatched through this
  /// context. A nullptr handler can be set to disable a previously set handler.
  void registerActionHandler(HandlerTy handler);
```

This handler takes two arguments: the first on is the transformation wrapped in
a callback, and the second is a reference to the associated action object. The
handler has full control of the execution, as such it can also decide to return
without executing the callback, skipping the transformation entirely!

## MLIR-provided Handlers

MLIR provides some predefined action handlers for immediate use that are
believed to be useful for most projects built with MLIR.

### Debug Counters

When debugging a compiler issue,
["bisection"](https://github.com/llvm/llvm-project/blob/main/mlir/docs/<https:/en.wikipedia.org/wiki/Bisection_(software_engineering)>)
is a useful technique for locating the root cause of the issue. `Debug Counters`
enable using this technique for debug actions by attaching a counter value to a
specific action and enabling/disabling execution of this action based on the
value of the counter. The counter controls the execution of the action with a
"skip" and "count" value. The "skip" value is used to skip a certain number of
initial executions of a debug action. The "count" value is used to prevent a
debug action from executing after it has executed for a set number of times (not
including any executions that have been skipped). If the "skip" value is
negative, the action will always execute. If the "count" value is negative, the
action will always execute after the "skip" value has been reached. For example,
a counter for a debug action with `skip=47` and `count=2`, would skip the first
47 executions, then execute twice, and finally prevent any further executions.
With a bit of tooling, the values to use for the counter can be automatically
selected; allowing for finding the exact execution of a debug action that
potentially causes the bug being investigated.

Note: The DebugCounter action handler does not support multi-threaded execution,
and should only be used in MLIRContexts where multi-threading is disabled (e.g.
via `-mlir-disable-threading`).

#### CommandLine Configuration

The `DebugCounter` handler provides several that allow for configuring counters.
The main option is `mlir-debug-counter`, which accepts a comma separated list of
`<count-name>=<counter-value>`. A `<counter-name>` is the debug action tag to
attach the counter, suffixed with either `-skip` or `-count`. A `-skip` suffix
will set the "skip" value of the counter. A `-count` suffix will set the "count"
value of the counter. The `<counter-value>` component is a numeric value to use
for the counter. An example is shown below using `MyCustomAction` defined above:

```shell
$ mlir-opt foo.mlir -mlir-debug-counter=unique-tag-for-my-action-skip=47,unique-tag-for-my-action-count=2
```

The above configuration would skip the first 47 executions of
`ApplyPatternAction`, then execute twice, and finally prevent any further
executions.

Note: Each counter currently only has one `skip` and one `count` value, meaning
that sequences of `skip`/`count` will not be chained.

The `mlir-print-debug-counter` option may be used to print out debug counter
information after all counters have been accumulated. The information is printed
in the following format:

```shell
DebugCounter counters:
<action-tag>                   : {<current-count>,<skip>,<count>}
```

For example, using the options above we can see how many times an action is
executed:

```shell
$ mlir-opt foo.mlir -mlir-debug-counter=unique-tag-for-my-action-skip=-1 -mlir-print-debug-counter --pass-pipeline="builtin.module(func.func(my-pass))" --mlir-disable-threading

DebugCounter counters:
unique-tag-for-my-action         : {370,-1,-1}
```

### ExecutionContext

The `ExecutionContext` is a component that provides facility to unify the kind
of functionalities that most compiler debuggers tool would need, exposed in a
composable way.

![IMG](/actions/ActionTracing_ExecutionContext.png)

The `ExecutionContext` is itself registered as a handler with the MLIRContext
and tracks all executed actions, keeping a per-thread stack of action execution.
It acts as a middleware that handles the flow of action execution while allowing
injection and control from a debugger.

- Multiple `Observers` can be registered with the `ExecutionContext`. When an
  action is dispatched for execution, it is passed to each of the `Observers`
  before and after executing the transformation.
- Multiple `BreakpointManager` can be registered with the `ExecutionContext`.
  When an action is dispatched for execution, it is passed to each of the
  registered `BreakpointManager` until one matches the action and return a valid
  `Breakpoint` object. In this case, the "callback" set by the client on the
  `ExecutionContext` is invoked, otherwise the transformation is directly
  executed.
- A single callback:
  `using CallbackTy = function_ref<Control(const ActionActiveStack *)>;` can be
  registered with the `ExecutionContext`, it is invoked when a `BreakPoint` is
  hit by an `Action`. The returned value of type `Control` is an enum
  instructing the `ExecutionContext` of how to proceed next:
  ```c++
  /// Enum that allows the client of the context to control the execution of the
  /// action.
  /// - Apply: The action is executed.
  /// - Skip: The action is skipped.
  /// - Step: The action is executed and the execution is paused before the next
  ///         action, including for nested actions encountered before the
  ///         current action finishes.
  /// - Next: The action is executed and the execution is paused after the
  ///         current action finishes before the next action.
  /// - Finish: The action is executed and the execution is paused only when we
  ///           reach the parent/enclosing operation. If there are no enclosing
  ///           operation, the execution continues without stopping.
  enum Control { Apply = 1, Skip = 2, Step = 3, Next = 4, Finish = 5 };
  ```
  Since the callback actually controls the execution, there can be only one
  registered at any given time.

#### Debugger ExecutionContext Hook

MLIR provides a callback for the `ExecutionContext` that implements a small
runtime suitable for debuggers like `gdb` or `lldb` to interactively control the
execution. It can be setup with
`mlir::setupDebuggerExecutionContextHook(executionContext);` or using `mlir-opt`
with the `--mlir-enable-debugger-hook` flag. This runtime exposes a set of C API
function that can be called from a debugger to:

- set breakpoints matching either action tags, or the `FileLineCol` locations of
  the IR associated with the action.
- set the `Control` flag to be returned to the `ExecutionContext`.
- control a "cursor" allowing to navigate through the IR and inspect it from the
  IR context associated with the action.

The implementation of this runtime can serve as an example for other
implementation of programmatic control of the execution.

#### Logging Observer

One observer is provided that allows to log action execution on a provided
stream. It can be exercised with `mlir-opt` using `--log-actions-to=<filename>`,
and optionally filtering the output with
`--log-mlir-actions-filter=<FileLineCol>`. This observer is not thread-safe at
the moment.

---

## Deeper notes

### The bisection technique

If a pipeline produces wrong output but every pass looks plausible, run with an action limit and
binary-search it: at N actions the output is correct, at N+1 it is not, so action N+1 is the culprit.
The tooling will tell you exactly which rewrite that was, on which operation.

This is the single most effective technique in this section for a certain class of bug — the ones
where the IR is valid at every stage and merely wrong. Those are otherwise very expensive to find,
because the verifier cannot help you.

### Wrapping your own work

Wrap a transformation in an action and it becomes observable and controllable by the same machinery.
For a downstream compiler with substantial custom transformations, doing this uniformly means the
bisection technique above works on your code too, which is worth the small amount of boilerplate.

### Relationship to other debugging tools

| Tool | Granularity | Use for |
|------|-------------|---------|
| `--mlir-print-ir-after-all` | pass | which pass broke it |
| `--debug-only=greedy-rewriter` | pattern application | which pattern fired |
| actions with a limit | individual action | which single rewrite broke it |
| [`mlir-reduce`](06-mlir-reduce.md) | input size | smallest reproducing input |

They compose: reduce the input first, then bisect actions on the reduced case.

### Beyond debugging

Because actions are observable, they are also a natural hook for profiling and for building
compilation traces — answering "where did the time go in this pipeline" at a finer grain than
`--mlir-timing`. If you maintain a compiler whose compile time matters, this is worth wiring up once.


---

[← Diagnostic Infrastructure](02-diagnostic-infrastructure.md) · [Index](../README.md) · [Remark Infrastructure →](04-remark-infrastructure.md)
