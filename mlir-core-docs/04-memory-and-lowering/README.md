# Memory and Lowering

Going down the stack. Bufferization crosses from pure values to storage, buffer deallocation makes that storage safe, and the LLVM IR target takes what remains out of MLIR entirely. These three are grouped because they are the phases where an optimizing pipeline stops being cheap to reason about and starts being about machines.

## Contents

1. [Bufferization](01-bufferization.md) — One-Shot Bufferize: turning tensor values into memref buffers.
2. [Ownership-based Buffer Deallocation](02-ownership-based-buffer-deallocation.md) — Inserting deallocations correctly, including across control flow.
3. [LLVM IR Target](03-llvm-ir-target.md) — Conversion to the LLVM dialect and translation to LLVM IR, including ABI details.

---

[← Back to the index](../README.md)
