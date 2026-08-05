# MLIR Release Notes

> **Section:** Appendix · document 1 of 3  
> **Upstream:** [https://mlir.llvm.org/docs/ReleaseNotes/](https://mlir.llvm.org/docs/ReleaseNotes/) · source [`mlir/docs/ReleaseNotes.md`](https://github.com/llvm/llvm-project/blob/main/mlir/docs/ReleaseNotes.md)  
> **License:** upstream text is Apache-2.0 WITH LLVM-exception.

## Orientation

Notable changes per LLVM release. Thin compared with the rest of the documentation, because MLIR
moves quickly and much of what changes is captured only in commit history.

Nonetheless the first place to look when an upgrade breaks your build, and worth skimming before an
upgrade rather than after.

**What you should be able to do after this page**

- Anticipate breaking changes before upgrading.
- Know where to look when an upgrade breaks something the notes do not mention.

---

## Upstream documentation

This document tries to provide some context about MLIR important changes in the
context of LLVM releases. It is updated on a best effort basis.

At the moment the MLIR community does not qualify the LLVM release branch
specifically, it is a snapshot of the MLIR development at the time of the release.


## LLVM 21

### GPU/NVVM Changes

- The default NVVM target architecture has been changed from `sm_50` to `sm_75`.
  `sm_75` is the oldest GPU variant compatible with the widest range of recent
  major CUDA Toolkit versions (11/12/13). This affects the `NVVMTargetAttr`,
  `GpuNVVMAttachTarget` pass, and the `gpu-lower-to-nvvm-pipeline`.

## LLVM 20

All the MLIR runners other than `mlir-cpu-runner` have been removed, as their functionality has been merged into it, and it has been renamed to `mlir-runner`.

## LLVM 18

### Properties: beyond attributes

See LLVM 17 notes below. The Dialect option `let usePropertiesForAttributes = 1;` is
now the default. You can set it to 0 to revert to the previous behavior. This will be
removed in LLVM 19.

## LLVM 17

See also the [deprecations and refactoring](https://mlir.llvm.org/deprecation/) doc.

### Bytecode

MLIR now support a [bytecode serialization](https://mlir.llvm.org/docs/BytecodeFormat/)
with versionning compatibility allowing 2 ways compatibility scheme, and lazy-loading
capabilities.

### Properties: beyond attributes

This is a new mechanism to implement storage for operations without having to
use attributes. You can opt-in to use Properties for ODS inherent attributes
using `let usePropertiesForAttributes = 1;` in your dialect definition (the flag
will be default in the next release). See
[slides](https://mlir.llvm.org/OpenMeetings/2023-02-09-Properties.pdf) and
[recording](https://youtu.be/7ofnlCFzlqg) of the open meeting presentation for
details.

### Action: Tracing and Debugging MLIR-based Compilers

[Action](https://mlir.llvm.org/docs/ActionTracing/) is a new mechanism to
encapsulate any transformation of any granularity in a way that can be
intercepted by the framework for debugging or tracing purposes, including
skipping a transformation programmatically (think about “compiler fuel” or
“debug counters” in LLVM). As such, “executing a pass” is an Action, so is “try
to apply one canonicalization pattern”, or “tile this loop”.

[slides](https://mlir.llvm.org/OpenMeetings/2023-02-23-Actions.pdf) and
[recording](https://youtu.be/ayQSyekVa3c) of the open meeting presentation for
details.

### Transform Dialect

See this [EuroLLVM talk](https://www.youtube.com/watch?v=P4gUj3QtH_Y&t=1s) and
[the online tutorial](https://mlir.llvm.org/docs/Tutorials/transform/).

### Others

- There is now support for
  "[distinct attributes](https://mlir.llvm.org/docs/Dialects/Builtin/#distinctattribute)".
- "Resources" (a way to store data outside the MLIR context) and "configuration"
  can now be serialized alongside the IR.

---

## Deeper notes

### The reality of upgrading MLIR

MLIR has no API stability guarantee. Upgrading across releases will break code, and the release notes
will not mention most of it. The techniques that actually work:

1. **Upgrade one release at a time.** Two at once compounds the failures and makes bisection useless.
2. **`git log --oneline` on the relevant `mlir/` subdirectories** between your two revisions. Commit
   messages for API changes are usually explicit, and this finds far more than the notes do.
3. **Watch the deprecations page** (<https://mlir.llvm.org/deprecation/>) and the LLVM Discourse
   forums, where large refactorings are announced with migration guidance before they land.
4. **Keep your test suite fast and comprehensive.** It is what turns an upgrade from an investigation
   into a mechanical exercise.

### Changes of the kind that catch people

Historically: the standard dialect being split into `arith`, `func`, `cf` and others; the migration
from `NoSideEffect` to `Pure`; the introduction of properties for inherent attributes; opaque
pointers in the LLVM dialect; member `cast`/`dyn_cast` being removed in favour of the free functions;
changes to the GPU compilation pipeline; and a number of dialect renames.

The pattern is recognisable: large mechanical renames, and consolidations that move something from
one mechanism to a better one. Both are cheap to fix once identified and expensive to identify from a
compiler error alone — which is why step 2 above is the highest-value habit.

### If you maintain a downstream project

Pin an LLVM revision, not a branch. Bump it on a schedule rather than opportunistically, so the
change set per bump stays small enough to diagnose. Keep an integration branch that only does the
bump, so the diff is never mixed with feature work.


---

[← Usage of `const` in MLIR for Core IR Types](../08-rationale/07-usage-of-const.md) · [Index](../README.md) · [External Tutorials and Learning Resources →](02-external-tutorials.md)
