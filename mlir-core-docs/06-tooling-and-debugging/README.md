# Tooling and Debugging

The tools you will spend most of your time in. `mlir-opt` first, because everything else is built around it, then the diagnostic and tracing infrastructure, then the standalone tools. Read at least the `mlir-opt` page early — it pays for itself immediately.

## Contents

1. [Using `mlir-opt`](01-using-mlir-opt.md) — The central tool: running pipelines, testing, and the flags that matter.
2. [Diagnostic Infrastructure](02-diagnostic-infrastructure.md) — Emitting errors and warnings, source locations, and testing diagnostics.
3. [Action: Tracing and Debugging MLIR-based Compilers](03-action-tracing.md) — Observing and controlling compiler execution at a fine grain.
4. [Remark Infrastructure](04-remark-infrastructure.md) — Structured optimization reports: what the compiler did and did not do.
5. [MLIR Language Server Protocol](05-language-server-protocol.md) — Editor support for `.mlir`, `.pdll` and `.td` files.
6. [`mlir-reduce`](06-mlir-reduce.md) — Automatic test-case reduction: shrink a failing input to something minimal.
7. [`mlir-rewrite`](07-mlir-rewrite.md) — Source-to-source edits on `.mlir` files, preserving formatting.

---

[← Back to the index](../README.md)
