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

### The nine sections

| # | Section | Answers the question |
|---|---------|----------------------|
| 01 | Core IR | What *is* an MLIR program, structurally? |
| 02 | Defining Dialects | How do I add my own operations, types and attributes? |
| 03 | Passes and Rewriting | How do I transform the IR? |
| 04 | Memory and Lowering | How do I get to buffers, and then out to LLVM? |
| 05 | Data Representation | How do types, numbers and IR map onto bytes? |
| 06 | Tooling and Debugging | Why is my pass doing that, and what tool tells me? |
| 07 | Bindings and Embedding | How do I drive MLIR from C or Python? |
| 08 | Rationale | Why is it designed this way? |
| 09 | Appendix | Release notes, external material, and what was excluded. |

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
extended only by patching LLVM. MLIR has exactly one thing, the **operation**, and everything else
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
├── properties              inherent, non-uniqued storage for the above
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
operations, structured loops, buffer accesses and LLVM intrinsics *simultaneously*, mid-pipeline.
Lowering is therefore not one big translation step but a sequence of local, partial rewrites, and
you can stop the pipeline at any point and print something meaningful.

Practical upshot: your lowering does not need to be complete to be useful. Partial conversion is
the normal mode of operation, not a fallback.

### 4. Semantics live in traits and interfaces, not in the operation name

A pass that hoists loop-invariant code does not contain a list of loop operations. It asks each
operation whether it implements `LoopLikeOpInterface`. A pass that folds constants asks for
`fold()`. A pass that reasons about aliasing asks for `MemoryEffectsOpInterface`.

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

**Operation** — the single unit of computation. Has a name, operands, results, attributes,
properties, successors and regions. Everything is one, including modules and functions.

**Op** — informal shorthand for an operation, and also the name of the C++ wrapper class
(`arith::AddIOp`) giving typed accessors over a generic `Operation*`. Op classes are value-typed
handles; passing one by value is idiomatic and cheap.

**Value** — an SSA value, defined exactly once. Either an `OpResult` or a `BlockArgument`.

**Block** — a list of operations ending in a terminator, plus a list of block arguments. Block
arguments replace phi nodes.

**Region** — an ordered list of blocks attached to an operation. Regions are how MLIR nests. A
region with one block and no control flow is extremely common.

**SSACFG region** vs **graph region** — two region kinds. In an SSACFG region dominance holds and
operations execute in order; this is the familiar CFG. In a graph region there is a single block,
no terminator semantics, and use-before-def is permitted; this suits dataflow graphs such as an
imported machine-learning model. The kind is declared by the enclosing operation via
`RegionKindInterface`.

**Terminator** — the last operation in a block, transferring control. An operation is a terminator
because it carries the `Terminator` trait.

**Attribute** — compile-time constant data attached to an operation, e.g. `{alignment = 8 : i64}`.
Uniqued, immutable, typed. Distinct from operands, which are runtime values.

**Property** — newer storage mechanism for an operation's inherent attributes: same information,
stored inline in the operation instead of in the uniqued attribute dictionary, which is faster and
allows non-attribute C++ types. You will see both spellings in real code.

**Inherent vs discardable attribute** — inherent attributes are part of the operation's definition
and are verified. Discardable attributes are extra annotations any pass may attach or drop, and are
namespaced by dialect (`llvm.noalias`). Dropping a discardable attribute must never change
semantics.

**Type** — the type of a `Value`. Also uniqued and immutable. `i32`, `f16`, `tensor<4x8xf32>`,
`memref<?xi8, 3>`, `!my_dialect.token`.

**Dialect** — a namespace grouping operations, types, attributes, and the interfaces and passes
that go with them. The prefix before the dot: `arith.addi` belongs to the `arith` dialect.

**Trait** — a compile-time property attached to an operation class, mixing in verification and
behaviour: `Commutative`, `Terminator`, `SameOperandsAndResultType`, `Pure`. Cheap to check, no
virtual dispatch.

**Interface** — a virtual API an operation, type, attribute or dialect can implement, letting
generic code call into it without knowing the concrete op: `LoopLikeOpInterface`,
`MemoryEffectsOpInterface`. The mechanism that makes generic passes possible.

**ODS** — Operation Definition Specification. The TableGen dialect used to declare operations in
`.td` files, from which `mlir-tblgen` generates C++.

**DRR** — Declarative Rewrite Rule. TableGen syntax for source-to-target pattern rewrites.

**PDL / PDLL** — Pattern Descriptor Language and its front-end language: an IR-based representation
of rewrite patterns that can be interpreted at run time rather than compiled in.

**Canonicalization** — the pass and pattern set that puts IR into a normal form so other passes have
fewer shapes to match. Not "optimization"; a canonicalization must be unconditionally desirable and
terminating.

**Folding** — replacing an operation with an existing value or a constant attribute, in place,
without creating new operations. `fold()` is cheaper and more restricted than a rewrite pattern.

**Legalization / conversion** — moving IR from one set of dialects to another, driven by a
`ConversionTarget` declaring which operations are legal. Full conversion must eliminate all illegal
ops; partial conversion may leave some.

**Type converter** — the object mapping source types to target types during conversion, and
materializing casts when a value crosses the boundary between converted and unconverted code.

**Tensor vs memref** — a tensor is a pure value with no address; a memref is a reference to storage
with a layout and a memory space. The distinction drives the whole bufferization phase.

**Bufferization** — the phase replacing tensor values with memref buffers, allocating storage and
deciding what can be written in place.

**Destination-passing style** — the convention where an operation takes its output buffer as an
operand and returns the updated value, which is what makes in-place bufferization expressible.

**Lowering** — rewriting from a higher-abstraction dialect to a lower one. Usually partial and
composed of several passes, not a single translation.

**Translation** — leaving MLIR entirely, e.g. emitting LLVM IR. Distinct from lowering: translation
is a one-way export implemented outside the pass infrastructure.

**Driver** — the loop that applies patterns. The greedy driver applies patterns to fixpoint; the
conversion driver applies them with legality tracking and rollback.

**Symbol** — a named entity referenced by name rather than by SSA value, e.g. a function referenced
as `@foo`. Lives in a symbol table.

**Action** — a first-class representation of "something the compiler did", used for tracing,
logging and bisection.

**`mlir-opt`** — the standard testing tool: read `.mlir`, run a pass pipeline, print `.mlir`. Your
dialect gets its own equivalent binary.

**`mlir-tblgen`** — the generator turning `.td` declarations into C++ and documentation.

**`mlir-translate`** — the tool for translation in and out of MLIR, e.g. `--mlir-to-llvmir`.

**FileCheck / lit** — the LLVM test tooling. MLIR tests are `.mlir` files with `// RUN:` and
`// CHECK:` lines; nearly all upstream examples are extracted from these tests.

---

## Contents


### 01 · [Core IR](01-core-ir/README.md)

What an MLIR program *is*. The Language Reference sets the ground rules, the IR-structure walkthrough shows the C++ data structures behind them, and the rest of the section covers the mechanisms that carry meaning across dialect boundaries: symbols, traits and interfaces.

| # | Document | |
|---|---|---|
| 1 | [MLIR Language Reference](01-core-ir/01-language-reference.md) | The definitive description of MLIR's structure, syntax, types and attributes. |
| 2 | [Lexical Tokens](01-core-ir/02-lexical-tokens.md) | The token kinds shared by MLIR's parser and by custom assembly formats. |
| 3 | [Understanding the IR Structure](01-core-ir/03-understanding-the-ir-structure.md) | Walking and inspecting the IR from C++ — the data structures behind the syntax. |
| 4 | [Symbols and Symbol Tables](01-core-ir/04-symbols-and-symbol-tables.md) | Named, non-SSA references: how functions and globals are found by name. |
| 5 | [Traits](01-core-ir/05-traits.md) | Compile-time operation properties, and the verification they bring with them. |
| 6 | [The `Broadcastable` Trait](01-core-ir/06-trait-broadcastable.md) | Worked example of a non-trivial trait: NumPy-style shape broadcasting rules. |
| 7 | [Interfaces](01-core-ir/07-interfaces.md) | The mechanism that lets generic transformations work on unknown dialects. |

### 02 · [Defining Dialects](02-defining-dialects/README.md)

How to extend MLIR with your own abstractions, in the order you will actually do the work: declare the dialect, declare its operations, declare its types and attributes, constrain them, give them readable syntax, wire the whole thing into a build. Shape inference closes the section because it is the first non-trivial thing most new dialects need.

| # | Document | |
|---|---|---|
| 1 | [Defining Dialects](02-defining-dialects/01-defining-dialects.md) | Declaring a dialect: namespace, hooks, dependencies and registration. |
| 2 | [Operation Definition Specification (ODS)](02-defining-dialects/02-operation-definition-specification.md) | The TableGen language for declaring operations. The core of dialect authoring. |
| 3 | [Defining Dialect Attributes and Types](02-defining-dialects/03-defining-attributes-and-types.md) | Custom types and attributes: parameters, uniquing, storage, syntax. |
| 4 | [Constraints](02-defining-dialects/04-constraints.md) | Predicates that restrict what operands, results and attributes may be. |
| 5 | [Customizing Assembly Behavior](02-defining-dialects/05-customizing-assembly-behavior.md) | Declarative assembly formats, custom directives, aliases and name hints. |
| 6 | [Creating a Dialect: Build Setup](02-defining-dialects/06-creating-a-dialect-build-setup.md) | CMake layout, TableGen invocation and directory conventions for a new dialect. |
| 7 | [Shape Inference](02-defining-dialects/07-shape-inference.md) | Propagating shapes through the IR, and the design of the shape system. |

### 03 · [Passes and Rewriting](03-passes-and-rewriting/README.md)

How IR is transformed. The pass manager first, since it is the container everything runs in; then canonicalization and the pattern rewriter, which are the primitives; then the two declarative pattern languages and a worked end-to-end example; then dialect conversion, the driver used for lowering; and finally dataflow analysis, which is how passes learn things they cannot see locally.

| # | Document | |
|---|---|---|
| 1 | [Pass Infrastructure](03-passes-and-rewriting/01-pass-infrastructure.md) | Writing passes, building pipelines, analyses, threading, instrumentation. |
| 2 | [Operation Canonicalization](03-passes-and-rewriting/02-operation-canonicalization.md) | Normal forms: what belongs in canonicalization and what does not. |
| 3 | [Pattern Rewriting: Generic DAG-to-DAG Rewriting](03-passes-and-rewriting/03-pattern-rewriting.md) | The core rewrite engine: patterns, benefits, rewriters and drivers. |
| 4 | [Table-driven Declarative Rewrite Rules (DRR)](03-passes-and-rewriting/04-declarative-rewrite-rules-drr.md) | Writing source-to-target patterns in TableGen instead of C++. |
| 5 | [PDLL — The PDL Language](03-passes-and-rewriting/05-pdll-pattern-language.md) | A dedicated language for rewrite patterns, with real editor tooling. |
| 6 | [Quickstart: Adding a Graph Rewrite](03-passes-and-rewriting/06-quickstart-adding-a-rewrite.md) | End-to-end walkthrough of adding an operation and a pattern that rewrites it. |
| 7 | [Dialect Conversion](03-passes-and-rewriting/07-dialect-conversion.md) | The legality-driven driver used for lowering between dialects. |
| 8 | [Writing Dataflow Analyses](03-passes-and-rewriting/08-writing-dataflow-analyses.md) | The sparse and dense dataflow frameworks, and how to build an analysis on them. |

### 04 · [Memory and Lowering](04-memory-and-lowering/README.md)

Going down the stack. Bufferization crosses from pure values to storage, buffer deallocation makes that storage safe, and the LLVM IR target takes what remains out of MLIR entirely. These three are grouped because they are the phases where an optimizing pipeline stops being cheap to reason about and starts being about machines.

| # | Document | |
|---|---|---|
| 1 | [Bufferization](04-memory-and-lowering/01-bufferization.md) | One-Shot Bufferize: turning tensor values into memref buffers. |
| 2 | [Ownership-based Buffer Deallocation](04-memory-and-lowering/02-ownership-based-buffer-deallocation.md) | Inserting deallocations correctly, including across control flow. |
| 3 | [LLVM IR Target](04-memory-and-lowering/03-llvm-ir-target.md) | Conversion to the LLVM dialect and translation to LLVM IR, including ABI details. |

### 05 · [Data Representation](05-data-representation/README.md)

Three documents about how abstract things become concrete bytes: how types map onto sizes and alignments, how real numbers are represented as integers, and how the IR itself is serialized. Independent of each other; read whichever applies.

| # | Document | |
|---|---|---|
| 1 | [Data Layout Modeling](05-data-representation/01-data-layout-modeling.md) | Target-dependent sizes, alignments and address spaces, queried generically. |
| 2 | [Quantization](05-data-representation/02-quantization.md) | Quantized types and the arithmetic they imply. |
| 3 | [MLIR Bytecode Format](05-data-representation/03-bytecode-format.md) | The binary serialization format: layout, versioning and upgrade paths. |

### 06 · [Tooling and Debugging](06-tooling-and-debugging/README.md)

The tools you will spend most of your time in. `mlir-opt` first, because everything else is built around it, then the diagnostic and tracing infrastructure, then the standalone tools. Read at least the `mlir-opt` page early — it pays for itself immediately.

| # | Document | |
|---|---|---|
| 1 | [Using `mlir-opt`](06-tooling-and-debugging/01-using-mlir-opt.md) | The central tool: running pipelines, testing, and the flags that matter. |
| 2 | [Diagnostic Infrastructure](06-tooling-and-debugging/02-diagnostic-infrastructure.md) | Emitting errors and warnings, source locations, and testing diagnostics. |
| 3 | [Action: Tracing and Debugging MLIR-based Compilers](06-tooling-and-debugging/03-action-tracing.md) | Observing and controlling compiler execution at a fine grain. |
| 4 | [Remark Infrastructure](06-tooling-and-debugging/04-remark-infrastructure.md) | Structured optimization reports: what the compiler did and did not do. |
| 5 | [MLIR Language Server Protocol](06-tooling-and-debugging/05-language-server-protocol.md) | Editor support for `.mlir`, `.pdll` and `.td` files. |
| 6 | [`mlir-reduce`](06-tooling-and-debugging/06-mlir-reduce.md) | Automatic test-case reduction: shrink a failing input to something minimal. |
| 7 | [`mlir-rewrite`](06-tooling-and-debugging/07-mlir-rewrite.md) | Source-to-source edits on `.mlir` files, preserving formatting. |

### 07 · [Bindings and Embedding](07-bindings-and-embedding/README.md)

Driving MLIR from outside C++. The C API is the stable foundation; the Python bindings are built on it and are what most people actually use. Read the C API page first even if you only intend to use Python — it explains the ownership model the Python layer inherits.

| # | Document | |
|---|---|---|
| 1 | [MLIR C API](07-bindings-and-embedding/01-c-api.md) | The stable C interface: conventions, ownership, and extending it. |
| 2 | [MLIR Python Bindings](07-bindings-and-embedding/02-python-bindings.md) | Building and using the Python API, and exposing your own dialect through it. |

### 08 · [Rationale and Design History](08-rationale/README.md)

Why MLIR is the way it is. These documents are arguments rather than references, and some record decisions since revised — which is why they are placed last. Reading them first is a reliable way to be confused by a design that no longer exists; reading them after the reference material is one of the fastest ways to develop judgement about the framework. The first document is an exception: it is still normative.

| # | Document | |
|---|---|---|
| 1 | [Side Effects and Speculation](08-rationale/01-side-effects-and-speculation.md) | How to model effects correctly — still normative, not history. |
| 2 | [MLIR Rationale](08-rationale/02-mlir-rationale.md) | The foundational design document: the arguments behind the core decisions. |
| 3 | [Generic DAG Rewriter Infrastructure Rationale](08-rationale/03-generic-dag-rewriter-rationale.md) | Why the pattern rewriter looks the way it does, with reference to prior systems. |
| 4 | [The Case for a Simplified Polyhedral Form](08-rationale/04-simplified-polyhedral-form.md) | Why MLIR embeds polyhedral concepts in the IR rather than using a separate representation. |
| 5 | [Linalg Dialect Rationale: The Case For Compiler-Friendly Custom Operations](08-rationale/05-structured-ops-rationale.md) | Why operations should be designed to be transformed, not merely executed. |
| 6 | [MLIR: Incremental Application to Graph Algorithms in ML Frameworks](08-rationale/06-mlir-for-graph-algorithms.md) | The case for adopting MLIR inside an existing framework, incrementally. |
| 7 | [Usage of `const` in MLIR for Core IR Types](08-rationale/07-usage-of-const.md) | Why core IR types do not use `const`, and what to do instead. |

### 09 · [Appendix](09-appendix/README.md)

Release notes, external learning material, and an account of the upstream pages that were deliberately not mirrored here.

| # | Document | |
|---|---|---|
| 1 | [MLIR Release Notes](09-appendix/01-release-notes.md) | Per-release changes; the first place to look when an upgrade breaks something. |
| 2 | [External Tutorials and Learning Resources](09-appendix/02-external-tutorials.md) | Community-maintained tutorials and courses outside the main documentation. |
| 3 | [Material Not Mirrored Here](09-appendix/03-excluded-material.md) | What was left out of this collection and where to find it. |

---

## Notes on sourcing

Upstream text is taken from `mlir/docs/` in the `llvm/llvm-project` repository (main branch), which is the source the website is generated from. It is licensed Apache-2.0 WITH LLVM-exception; each page names its source file and links to both the website page and the repository file.

Orientation headers, deeper notes, section introductions, the excluded-material page and this index — including the organization, mental-model and vocabulary material above — were written for this collection.

Links to documents included here resolve locally. Links to upstream pages that were not mirrored — auto-generated dialect operation references, the Toy and Transform tutorials, `getting_started/` — point at <https://mlir.llvm.org>. See [Material Not Mirrored Here](09-appendix/03-excluded-material.md) for the full account.

MLIR moves quickly and has no API stability guarantee. Treat this as a snapshot: verify specifics against the MLIR revision you are actually building.
