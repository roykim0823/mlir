# Passes and Rewriting

How IR is transformed. The pass manager first, since it is the container everything runs in; then canonicalization and the pattern rewriter, which are the primitives; then the two declarative pattern languages and a worked end-to-end example; then dialect conversion, the driver used for lowering; and finally dataflow analysis, which is how passes learn things they cannot see locally.

## Contents

1. [Pass Infrastructure](01-pass-infrastructure.md) — Writing passes, building pipelines, analyses, threading, instrumentation.
2. [Operation Canonicalization](02-operation-canonicalization.md) — Normal forms: what belongs in canonicalization and what does not.
3. [Pattern Rewriting: Generic DAG-to-DAG Rewriting](03-pattern-rewriting.md) — The core rewrite engine: patterns, benefits, rewriters and drivers.
4. [Table-driven Declarative Rewrite Rules (DRR)](04-declarative-rewrite-rules-drr.md) — Writing source-to-target patterns in TableGen instead of C++.
5. [PDLL — The PDL Language](05-pdll-pattern-language.md) — A dedicated language for rewrite patterns, with real editor tooling.
6. [Quickstart: Adding a Graph Rewrite](06-quickstart-adding-a-rewrite.md) — End-to-end walkthrough of adding an operation and a pattern that rewrites it.
7. [Dialect Conversion](07-dialect-conversion.md) — The legality-driven driver used for lowering between dialects.
8. [Writing Dataflow Analyses](08-writing-dataflow-analyses.md) — The sparse and dense dataflow frameworks, and how to build an analysis on them.

---

[← Back to the index](../README.md)
