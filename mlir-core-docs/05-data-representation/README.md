# Data Representation

Three documents about how abstract things become concrete bytes: how types map onto sizes and alignments, how real numbers are represented as integers, and how the IR itself is serialized. Independent of each other; read whichever applies.

## Contents

1. [Data Layout Modeling](01-data-layout-modeling.md) — Target-dependent sizes, alignments and address spaces, queried generically.
2. [Quantization](02-quantization.md) — Quantized types and the arithmetic they imply.
3. [MLIR Bytecode Format](03-bytecode-format.md) — The binary serialization format: layout, versioning and upgrade paths.

---

[← Back to the index](../README.md)
