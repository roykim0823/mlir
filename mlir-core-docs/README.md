# MLIR Documentation, Organized

A re-sequenced, annotated edition of the MLIR code documentation from <https://mlir.llvm.org/docs/>.

**47 documents in 9 sections**, plus the orientation material on this page. Every page carries the upstream text with its internal links rewritten to resolve inside this tree, wrapped in an orientation header explaining where the document sits and what it is for, and — where the upstream text is terse or has known sharp edges — a section of additional notes.

New to MLIR? Read [The MLIR Mental Model](#the-mlir-mental-model) below, then start with the [MLIR Language Reference](01-core-ir/01-language-reference.md). When a page uses an unfamiliar term, check the [Vocabulary Quick Reference](#vocabulary-quick-reference).

---

## How this collection is organized

The upstream `mlir.llvm.org/docs` landing page lists its documents alphabetically. That is fine as a lookup table and poor as a curriculum: `Bufferization` lands before `MLIR Language Reference`, the tutorial that teaches you to read the IR sits at the bottom under `Tutorials`, and design rationale documents are interleaved with API references as if they served the same purpose.

This collection re-sequences the same material along the axis that actually matters, which is **dependency order**. You cannot understand pattern rewriting without operations, you cannot understand operations without the type and attribute system, and you cannot understand dialect conversion without both. Reading front to back — the sections in numeric order, the documents in numeric order within each section — never requires a concept that has not yet been introduced.

### Scope

This edition covers the **core infrastructure** documentation: the IR itself, how to extend it, how to transform it, how to lower it, and the tooling around it. Dialect reference material and the two large tutorials are out of scope by request, which makes the remaining set unusually coherent — what is left is the framework, not the library built on top of it.

### Contents

| # | Section | Docs | Answers the question |
|---|---------|------|----------------------|
| 01 | [Core IR](01-core-ir/README.md) | 7 | What *is* an MLIR program, structurally? |
| 02 | [Defining Dialects](02-defining-dialects/README.md) | 7 | How do I add my own operations, types and attributes? |
| 03 | [Passes and Rewriting](03-passes-and-rewriting/README.md) | 8 | How do I transform the IR? |
| 04 | [Memory and Lowering](04-memory-and-lowering/README.md) | 3 | How do I get to buffers, and then out to LLVM? |
| 05 | [Data Representation](05-data-representation/README.md) | 3 | How do types, numbers and IR map onto bytes? |
| 06 | [Tooling and Debugging](06-tooling-and-debugging/README.md) | 7 | Why is my pass doing that, and what tool tells me? |
| 07 | [Bindings and Embedding](07-bindings-and-embedding/README.md) | 2 | How do I drive MLIR from C or Python? |
| 08 | [Rationale and Design History](08-rationale/README.md) | 7 | Why is it designed this way? |
| 09 | [Appendix](09-appendix/README.md) | 3 | Release notes, external material, and what was excluded. |

Each section page lists its documents with a one-line description of what is in them.

### Ordering rules used

1. **Structure before manipulation.** Sections 01 and 02 describe what the IR *is* before section
   03 describes how to change it.
2. **Concepts before specialisations.** `Traits` precedes `The Broadcastable Trait`; the pass
   manager precedes everything that runs inside it.
3. **Producer before consumer.** ODS comes before DRR, because DRR patterns are written against
   ODS-declared operations. The pattern rewriter comes before dialect conversion, because the
   conversion framework is a driver built on top of patterns.
4. **Rationale last, cross-linked throughout.** The design documents are valuable but they are
   historical arguments, not references. Reading them first is a common way to get confused by
   decisions that have since been revised. The one exception — *Side Effects and Speculation*,
   which is still normative — is placed first within that section and flagged as such.
5. **Tutorials filed by topic, not by being tutorials.** *Understanding the IR Structure* is a
   core-IR document that happens to be written as a walkthrough, so it sits in section 01 next to
   the Language Reference. *Using `mlir-opt`* is a tooling document. *Creating a Dialect* is the
   build-system half of section 02. Each is marked in its orientation header.

### What is in each page

```
# Title
> Section / position / upstream URL / source file / license
## Orientation          <- written for this collection
   why the document exists, what to read first, what you get out of it
## Upstream documentation
   the upstream text, verbatim, with internal links rewritten to point
   into this tree where the target is included here
## Deeper notes         <- written for this collection, where the upstream
   text is terse, assumes context, or has known sharp edges
```

Because the dialect reference is out of scope, several pages carry more in their notes than they
otherwise would — the Language Reference notes cover the builtin type catalogue, and the
bufferization and lowering pages carry the `linalg`/`memref` background they depend on.

### What was excluded, and why

| Excluded | Reason |
|----------|--------|
| **Dialects** (the whole `Dialects/` tree) | Excluded by request. Roughly forty of those pages are auto-generated from TableGen at website build time and would be stale within weeks; the hand-written ones document individual dialects rather than the framework. |
| **Passes** | Excluded by request. It is a generated catalogue; `mlir-opt --help` on your own build is authoritative anyway. |
| **Pattern Search** | Excluded by request. It is an interactive website tool, not a document. |
| **SPIR-V to LLVM Dialect conversion manual** | Excluded by request. |
| **Toy Tutorial** (Ch 1–7) | Excluded by request — to be handled separately. |
| **Transform Dialect Tutorial** (Ch 0–4, ChH) | Excluded by request — to be handled separately. |
| `getting_started/*` | Outside the `docs/` set, though the glossary and testing guide are worth bookmarking. |

Links from upstream text into any of the above resolve to `mlir.llvm.org` rather than dead-ending.
See [Material Not Mirrored Here](09-appendix/03-excluded-material.md) for the full account.

### Reading paths

Reading front to back in numeric order is the full curriculum. If you have a specific goal, these shorter paths cover it:

**"I need to read and modify IR in an existing project."**
[Language Reference](01-core-ir/01-language-reference.md) →
[Understanding the IR Structure](01-core-ir/03-understanding-the-ir-structure.md) →
[Using `mlir-opt`](06-tooling-and-debugging/01-using-mlir-opt.md) →
[Pass Infrastructure](03-passes-and-rewriting/01-pass-infrastructure.md) →
[Pattern Rewriting](03-passes-and-rewriting/03-pattern-rewriting.md).

**"I am adding a dialect for my accelerator."**
[Language Reference](01-core-ir/01-language-reference.md) →
[Traits](01-core-ir/05-traits.md) →
[Interfaces](01-core-ir/07-interfaces.md) → all of
[section 02](02-defining-dialects/README.md) →
[Pattern Rewriting](03-passes-and-rewriting/03-pattern-rewriting.md) →
[Dialect Conversion](03-passes-and-rewriting/07-dialect-conversion.md) →
[LLVM IR Target](04-memory-and-lowering/03-llvm-ir-target.md).

**"I am building the lowering pipeline."**
[Dialect Conversion](03-passes-and-rewriting/07-dialect-conversion.md) →
[Bufferization](04-memory-and-lowering/01-bufferization.md) →
[Buffer Deallocation](04-memory-and-lowering/02-ownership-based-buffer-deallocation.md) →
[LLVM IR Target](04-memory-and-lowering/03-llvm-ir-target.md) →
[Data Layout](05-data-representation/01-data-layout-modeling.md).

**"Something is broken and I need to find out why."**
[Using `mlir-opt`](06-tooling-and-debugging/01-using-mlir-opt.md) →
[Diagnostics](06-tooling-and-debugging/02-diagnostic-infrastructure.md) →
[Action Tracing](06-tooling-and-debugging/03-action-tracing.md) →
[`mlir-reduce`](06-tooling-and-debugging/06-mlir-reduce.md).

**"I want to understand the design decisions."**
[Section 08](08-rationale/README.md), in the order given.

---

## The MLIR Mental Model

Most of the upstream documentation is written for someone who already holds a specific mental model, and is confusing without it. This section states that model explicitly. It exists because the Language Reference opens with grammar productions rather than with the shape of the thing being described.

### 1. There is no instruction set

LLVM IR has a fixed, closed instruction set — `add`, `load`, `br` — defined by the LLVM project and
extended only by patching LLVM itself. Intrinsics and metadata leave some room, but a genuinely
new operation means a change to LLVM. MLIR has exactly one thing, the **operation**, and everything else
is an operation defined by some dialect. `arith.addi` has no more privileged status in the
infrastructure than an operation you define this afternoon.

The consequence people miss: *the core of MLIR contains almost no compiler.* It contains a data
structure, a pass manager, a pattern rewriter, a printer/parser, and a verification framework. The
compiler is assembled out of dialects. When you ask "how does MLIR do X", the honest answer is
usually "it doesn't; some dialect does, and you can pick a different one."

This collection documents that core. The dialects built on it are a separate subject.

### 2. An operation is a recursive container, not a line of code

```
Operation
├── name                    e.g. "scf.for"
├── operands                Values it consumes
├── results                 Values it produces
├── attributes              compile-time constants (a dictionary)
├── properties              inherent attributes, stored inline, not uniqued
├── successors              blocks it can branch to
└── regions[]               ── Block[]  ── Operation[]  ── ...
```

The last line is the whole trick. A region holds blocks, a block holds operations, and those
operations hold regions. The entire program is one tree of operations, so an `scf.for` loop is not
a control-flow-graph pattern to be recognised — it is a single operation whose body lives in its
region. A module is an operation. A function is an operation. There is no separate `Function` class
hierarchy above the instruction level, the way LLVM has `Module`/`Function`/`BasicBlock`/`Instruction`
as four distinct types.

This is why MLIR can host both a machine-learning graph and machine-level code in the same data
structure, and why "raising" — recovering structure that was destroyed — is less often necessary.

### 3. Multi-level means abstractions coexist

"Multi-Level" in the name is not marketing. A single valid module can contain high-level tensor
operations, structured loops, buffer accesses and raw pointer loads *simultaneously*, mid-pipeline.
This function is not a contrived example — it is what the middle of a lowering pipeline looks like,
and it parses and verifies:

```mlir
func.func @mixed(%t: tensor<8x8xf32>, %buf: memref<8xf32>, %p: !llvm.ptr)
    -> tensor<8x8xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index

  // tensor level: pure values, no storage, no aliasing
  %0 = linalg.matmul ins(%t, %t : tensor<8x8xf32>, tensor<8x8xf32>)
                     outs(%t : tensor<8x8xf32>) -> tensor<8x8xf32>

  // loop level: explicit control flow over a buffer
  scf.for %i = %c0 to %c8 step %c1 {
    %v = memref.load %buf[%i] : memref<8xf32>
    %w = arith.mulf %v, %v : f32
    memref.store %w, %buf[%i] : memref<8xf32>
  }

  // machine level: a raw pointer load, in the same function
  %x = llvm.load %p : !llvm.ptr -> f32
  memref.store %x, %buf[%c0] : memref<8xf32>

  return %0 : tensor<8x8xf32>
}
```

Six dialects, three levels of abstraction, one function. Lowering is therefore not one big
translation step but a sequence of local, partial rewrites, and you can stop the pipeline at any
point and print something meaningful.

Practical upshot: your lowering does not need to be complete to be useful. Partial conversion is
the normal mode of operation, not a fallback.

### 4. Semantics live in traits and interfaces, not in the operation name

Because there is no fixed instruction set, a pass cannot switch on opcodes. A loop-invariant code
motion pass written as `if (isa<scf::ForOp, affine::ForOp>(op))` would work on exactly the two loops
its author had heard of, and would silently do nothing to the loop operation you define this
afternoon.

So passes ask questions instead of matching names, and an operation declares its answers in its
definition:

- **Traits** state facts, checked statically and for free. `Commutative` tells the canonicalizer it
  may reorder operands; `Pure` tells dead-code elimination the operation can be deleted when its
  results are unused.
- **Interfaces** supply behaviour through virtual dispatch. `LoopLikeOpInterface` hands a caller the
  loop body and the induction variable, so loop-invariant code motion hoists out of any operation
  implementing it — including operations written years after the pass was. `MemoryEffectsOpInterface`
  reports what an operation reads, writes, allocates or frees, which is what CSE and alias analysis
  consult.

Constant folding works the same way from the other side: the folder calls `Operation::fold`, which
dispatches to the `fold()` hook ODS generated from your `.td` file. In none of these does the pass
mention a dialect or an operation name.

This is the single highest-leverage idea in MLIR, and it is why [Traits](01-core-ir/05-traits.md)
and [Interfaces](01-core-ir/07-interfaces.md) come so early here despite sitting far down the
alphabetical listing upstream. If you define a dialect and skip interfaces, you will reimplement
analyses that already exist. If you attach the right interfaces, a large amount of infrastructure
starts working on your operations without a single line of code being written that mentions your
dialect.

### 5. Values, not variables — and pure values before storage

MLIR is SSA. Every `Value` is defined exactly once, either as an operation result or as a block
argument. There are no phi nodes: block arguments do that job, which makes CFG manipulation
substantially less painful.

At high abstraction levels values usually have **tensor** type: a pure value with no storage and no
address, so two tensors with the same contents are indistinguishable and the compiler may freely
duplicate, reorder or eliminate them. At low levels values have **memref** type: a reference to
storage, carrying aliasing and lifetime concerns. Bufferization is the phase that crosses that
line, and it is the point where an optimizing pipeline stops being cheap to reason about. That is
why it opens [section 04](04-memory-and-lowering/README.md).

### 6. Almost everything is declarative and generated

Operations are declared in TableGen (`.td`) files, and `mlir-tblgen` generates the C++ class, the
parser, the printer, the verifier, the builders and the documentation. Rewrite patterns can be
declared in TableGen (DRR) or in PDLL. Passes are declared in TableGen. Type constraints are
declared in TableGen.

When you first read generated code this feels like indirection for its own sake. The payoff is that
adding an operation costs about ten lines rather than four hundred, and that machine-readable
declarations let the infrastructure build things like the LSP server, the bytecode format and the
operation documentation without anyone hand-maintaining them.

### Where this model shows up in the rest of the collection

| Idea | Developed in |
|------|--------------|
| Operation / region / block structure | [Language Reference](01-core-ir/01-language-reference.md), [Understanding the IR Structure](01-core-ir/03-understanding-the-ir-structure.md) |
| Types and attributes | [Language Reference](01-core-ir/01-language-reference.md), [Attributes and Types](02-defining-dialects/03-defining-attributes-and-types.md) |
| Traits and interfaces | [Traits](01-core-ir/05-traits.md), [Broadcastable](01-core-ir/06-trait-broadcastable.md), [Interfaces](01-core-ir/07-interfaces.md) |
| Declarative definition | all of [section 02](02-defining-dialects/README.md) |
| Local rewriting as the transform primitive | [Canonicalization](03-passes-and-rewriting/02-operation-canonicalization.md), [Pattern Rewriting](03-passes-and-rewriting/03-pattern-rewriting.md), [DRR](03-passes-and-rewriting/04-declarative-rewrite-rules-drr.md), [PDLL](03-passes-and-rewriting/05-pdll-pattern-language.md) |
| Partial, multi-level lowering | [Dialect Conversion](03-passes-and-rewriting/07-dialect-conversion.md), [LLVM IR Target](04-memory-and-lowering/03-llvm-ir-target.md) |
| Pure values to storage | [Bufferization](04-memory-and-lowering/01-bufferization.md) |

---

## Vocabulary Quick Reference

Terms the upstream documents use freely and define only in passing, or define in the
`getting_started/Glossary` page that most readers never open. Skim once; return when a page uses a
word as though you already knew it.

### Structure

**Block** — a list of operations ending in a terminator, plus a list of block arguments. Block
arguments replace phi nodes.

**Operation** — the single unit of computation. Has a name, operands, results, attributes,
properties, successors and regions. Everything is one, including modules and functions.

**Op** — informal shorthand for an operation, and also the name of the C++ wrapper class
(`arith::AddIOp`) giving typed accessors over a generic `Operation*`. Op classes are value-typed
handles; passing one by value is idiomatic and cheap.

**Region** — an ordered list of blocks attached to an operation. Regions are how MLIR nests. A
region with one block and no control flow is extremely common.

**SSACFG region** vs **graph region** — two region kinds. In an SSACFG region dominance holds and
operations execute in order; this is the familiar CFG. In a graph region there is a single block,
no terminator semantics, and use-before-def is permitted; this suits dataflow graphs such as an
imported machine-learning model. The kind is declared by the enclosing operation via
`RegionKindInterface`.

**Symbol** — a named entity referenced by name rather than by SSA value, e.g. a function referenced
as `@foo`. Lives in a symbol table.

**Terminator** — the last operation in a block, transferring control. An operation is a terminator
because it carries the `Terminator` trait.

**Value** — an SSA value, defined exactly once. Either an `OpResult` or a `BlockArgument`.

### Attributes and types

**Attribute** — compile-time constant data attached to an operation, e.g. `{alignment = 8 : i64}`.
Uniqued, immutable, typed. Distinct from operands, which are runtime values.

**Inherent vs discardable attribute** — inherent attributes are part of the operation's definition
and are verified. Discardable attributes are extra annotations any pass may attach or drop, and are
namespaced by dialect (`llvm.noalias`). Dropping a discardable attribute must never change
semantics.

**Property** — newer storage mechanism for an operation's inherent attributes: same information,
stored inline in the operation instead of in the uniqued attribute dictionary, which is faster and
allows non-attribute C++ types. Both mechanisms are in active use, and the ODS-generated accessors
look the same either way — the difference shows up in the generic assembly format, where properties
print inside `<{...}>`.

**Type** — the type of a `Value`. Also uniqued and immutable. `i32`, `f16`, `tensor<4x8xf32>`,
`memref<?xi8, 3>`, `!my_dialect.token`.

### Dialects and declarations

**Dialect** — a namespace grouping operations, types, attributes, and the interfaces and passes
that go with them. The prefix before the dot: `arith.addi` belongs to the `arith` dialect.

**Interface** — a virtual API an operation, type, attribute or dialect can implement, letting
generic code call into it without knowing the concrete op: `LoopLikeOpInterface`,
`MemoryEffectsOpInterface`. The mechanism that makes generic passes possible. Note the naming trap:
several interfaces have a TableGen name that differs from the C++ class you `dyn_cast` to —
`MemoryEffectsOpInterface` in a `.td` file is `MemoryEffectOpInterface`, no `s`, in C++.

**ODS** — Operation Definition Specification. The TableGen dialect used to declare operations in
`.td` files, from which `mlir-tblgen` generates C++.

**Trait** — a compile-time property attached to an operation class, mixing in verification and
behaviour: `Commutative`, `Terminator`, `SameOperandsAndResultType`, `IsolatedFromAbove`. Checking
one is a template test, with no virtual dispatch. `Pure` is written like a trait and usually called
one, but it expands to `AlwaysSpeculatable` plus `NoMemoryEffect`, and that second half is answered
through `MemoryEffectsOpInterface` — so it is not free to check.

### Rewriting

**Canonicalization** — the pass and pattern set that puts IR into a normal form so other passes have
fewer shapes to match. Not "optimization"; a canonicalization must be unconditionally desirable and
terminating.

**Driver** — the loop that applies patterns. The greedy driver applies patterns to fixpoint; the
conversion driver applies them with legality tracking and rollback.

**DRR** — Declarative Rewrite Rule. TableGen syntax for source-to-target pattern rewrites.

**Folding** — replacing an operation with an existing value or a constant attribute, in place,
without creating new operations. `fold()` is cheaper and more restricted than a rewrite pattern.

**Legalization / conversion** — moving IR from one set of dialects to another, driven by a
`ConversionTarget` declaring which operations are legal. Full conversion must eliminate all illegal
ops; partial conversion may leave some.

**PDL / PDLL** — Pattern Descriptor Language and its front-end language: an IR-based representation
of rewrite patterns that can be interpreted at run time rather than compiled in.

**Type converter** — the object mapping source types to target types during conversion, and
materializing casts when a value crosses the boundary between converted and unconverted code.

### Lowering and memory

**Bufferization** — the phase replacing tensor values with memref buffers, allocating storage and
deciding what can be written in place.

**Destination-passing style** — the convention where an operation takes its output buffer as an
operand and returns the updated value, which is what makes in-place bufferization expressible.

**Lowering** — rewriting from a higher-abstraction dialect to a lower one. Usually partial and
composed of several passes, not a single translation.

**Tensor vs memref** — a tensor is a pure value with no address; a memref is a reference to storage
with a layout and a memory space. The distinction drives the whole bufferization phase.

**Translation** — leaving MLIR entirely, e.g. emitting LLVM IR. Distinct from lowering: translation
is a one-way export implemented outside the pass infrastructure.

### Tooling

**Action** — a transformation of any granularity, wrapped so the framework can intercept it *before*
it runs: observe it, log it, or skip it outright. "Execute this pass", "apply this canonicalization
pattern" and "tile this loop" are all actions. The interception is what makes tracing and bisection
possible.

**FileCheck / lit** — the LLVM test tooling. MLIR tests are `.mlir` files with `// RUN:` and
`// CHECK:` lines; nearly all upstream examples are extracted from these tests.

**`mlir-opt`** — the standard testing tool: read `.mlir`, run a pass pipeline, print `.mlir`. Your
dialect gets its own equivalent binary.

**`mlir-tblgen`** — the generator turning `.td` declarations into C++ and documentation.

**`mlir-translate`** — the tool for translation in and out of MLIR, e.g. `--mlir-to-llvmir`.

---

## Notes on sourcing

Upstream text is taken from `mlir/docs/` in the `llvm/llvm-project` repository (main branch), which is the source the website is generated from. It is licensed Apache-2.0 WITH LLVM-exception; each page names its source file and links to both the website page and the repository file.

Orientation headers, deeper notes, section introductions, the excluded-material page and this index — including the organization, mental-model and vocabulary material above — were written for this collection.

Links to documents included here resolve locally. Links to upstream pages that were not mirrored — auto-generated dialect operation references, the Toy and Transform tutorials, `getting_started/` — point at <https://mlir.llvm.org>. See [Material Not Mirrored Here](09-appendix/03-excluded-material.md) for the full account.

MLIR moves quickly and has no API stability guarantee. Treat this as a snapshot: verify specifics against the MLIR revision you are actually building.
