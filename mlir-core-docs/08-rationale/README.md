# Rationale and Design History

Why MLIR is the way it is. These documents are arguments rather than references, and some record decisions since revised — which is why they are placed last. Reading them first is a reliable way to be confused by a design that no longer exists; reading them after the reference material is one of the fastest ways to develop judgement about the framework. The first document is an exception: it is still normative.

## Contents

1. [Side Effects and Speculation](01-side-effects-and-speculation.md) — How to model effects correctly — still normative, not history.
2. [MLIR Rationale](02-mlir-rationale.md) — The foundational design document: the arguments behind the core decisions.
3. [Generic DAG Rewriter Infrastructure Rationale](03-generic-dag-rewriter-rationale.md) — Why the pattern rewriter looks the way it does, with reference to prior systems.
4. [The Case for a Simplified Polyhedral Form](04-simplified-polyhedral-form.md) — Why MLIR embeds polyhedral concepts in the IR rather than using a separate representation.
5. [Linalg Dialect Rationale: The Case For Compiler-Friendly Custom Operations](05-structured-ops-rationale.md) — Why operations should be designed to be transformed, not merely executed.
6. [MLIR: Incremental Application to Graph Algorithms in ML Frameworks](06-mlir-for-graph-algorithms.md) — The case for adopting MLIR inside an existing framework, incrementally.
7. [Usage of `const` in MLIR for Core IR Types](07-usage-of-const.md) — Why core IR types do not use `const`, and what to do instead.

---

[← Back to the index](../README.md)
