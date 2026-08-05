# Defining Dialects

How to extend MLIR with your own abstractions, in the order you will actually do the work: declare the dialect, declare its operations, declare its types and attributes, constrain them, give them readable syntax, wire the whole thing into a build. Shape inference closes the section because it is the first non-trivial thing most new dialects need.

## Contents

1. [Defining Dialects](01-defining-dialects.md) — Declaring a dialect: namespace, hooks, dependencies and registration.
2. [Operation Definition Specification (ODS)](02-operation-definition-specification.md) — The TableGen language for declaring operations. The core of dialect authoring.
3. [Defining Dialect Attributes and Types](03-defining-attributes-and-types.md) — Custom types and attributes: parameters, uniquing, storage, syntax.
4. [Constraints](04-constraints.md) — Predicates that restrict what operands, results and attributes may be.
5. [Customizing Assembly Behavior](05-customizing-assembly-behavior.md) — Declarative assembly formats, custom directives, aliases and name hints.
6. [Creating a Dialect: Build Setup](06-creating-a-dialect-build-setup.md) — CMake layout, TableGen invocation and directory conventions for a new dialect.
7. [Shape Inference](07-shape-inference.md) — Propagating shapes through the IR, and the design of the shape system.

---

[← Back to the index](../README.md)
