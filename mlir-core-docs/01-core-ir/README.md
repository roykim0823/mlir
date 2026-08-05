# Core IR

What an MLIR program *is*. The Language Reference sets the ground rules, the IR-structure walkthrough shows the C++ data structures behind them, and the rest of the section covers the mechanisms that carry meaning across dialect boundaries: symbols, traits and interfaces.

## Contents

1. [MLIR Language Reference](01-language-reference.md) — The definitive description of MLIR's structure, syntax, types and attributes.
2. [Lexical Tokens](02-lexical-tokens.md) — The token kinds shared by MLIR's parser and by custom assembly formats.
3. [Understanding the IR Structure](03-understanding-the-ir-structure.md) — Walking and inspecting the IR from C++ — the data structures behind the syntax.
4. [Symbols and Symbol Tables](04-symbols-and-symbol-tables.md) — Named, non-SSA references: how functions and globals are found by name.
5. [Traits](05-traits.md) — Compile-time operation properties, and the verification they bring with them.
6. [The `Broadcastable` Trait](06-trait-broadcastable.md) — Worked example of a non-trivial trait: NumPy-style shape broadcasting rules.
7. [Interfaces](07-interfaces.md) — The mechanism that lets generic transformations work on unknown dialects.

---

[← Back to the index](../README.md)
