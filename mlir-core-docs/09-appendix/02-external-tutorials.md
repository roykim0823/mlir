# External Tutorials and Learning Resources

> **Section:** Appendix · document 2 of 3  
> **Upstream:** [https://mlir.llvm.org/docs/Tutorials/ExternalTutorials/](https://mlir.llvm.org/docs/Tutorials/ExternalTutorials/) · source [`mlir/docs/Tutorials/ExternalTutorials.md`](https://github.com/llvm/llvm-project/blob/main/mlir/docs/Tutorials/ExternalTutorials.md)  
> **License:** upstream text is Apache-2.0 WITH LLVM-exception.

## Orientation

A short index of tutorials maintained outside the LLVM repository. Useful because the upstream
documentation is written as reference material, and several of these are written as instruction —
which is a different and complementary thing.

Quality and currency vary, and they are not maintained in step with MLIR. Check what revision a
tutorial targets before following it closely; MLIR moves fast enough that a two-year-old tutorial will
contain code that no longer compiles.

**What you should be able to do after this page**

- Find instructional material to complement the reference documentation.

---

## Upstream documentation

## Upstream tutorial

- [LLVM - Lighthouse](https://github.com/llvm/lighthouse): "In essence, this
  project should guide you through using MLIR for your own projects, showing the
  way, but not forcing you to follow a particular path. Essentially, the role of
  a lighthouse."

## Third-party tutorials

The following lists tutorials and blogs that people have created independently.

**Disclaimer**: These tutorials and blogs are maintained by third parties and
may therefore be out of sync with the upstream implementation. Please do not
report bugs upstream (e.g., on Discourse).

- MLIR for beginners —
  [Blog](https://www.jeremykun.com/2023/08/10/mlir-getting-started/) /
  [Repository](https://github.com/j2kun/mlir-tutorial): A general introduction
  to MLIR. Resembles the Toy tutorial.

- [mlir-tutor](https://github.com/Groverkss/mlir-tutor): Exercises for learning
  MLIR.

- [MLIR introduction by Stephen Diehl](https://www.stephendiehl.com/tags/mlir/):
  Focuses on explaining dialects and passes.

- [End-to-End MLIR pipeline](https://github.com/DavidGinten/ML-compiler-exercise):
  Demonstrates how deep learning models can be lowered from an ML framework to
  executable binaries.

---

## Deeper notes

### Also worth knowing about

Beyond whatever the upstream list currently contains:

- **The MLIR YouTube channel** hosts the recorded open design meetings, which are the best record of
  *why* recent changes were made — the discussion happens there before it reaches the docs.
- **LLVM Discourse** (<https://llvm.discourse.group>, MLIR category) is where design discussion
  happens and where RFCs are posted. Reading it for a few weeks gives a much better sense of the
  current state than any document.
- **The upstream test suite** (`mlir/test/`) is the largest corpus of worked examples in existence for
  MLIR, and it is always current — because it has to pass. When documentation and tests disagree, the
  tests are right. For any operation or pass you are unsure about,
  `grep -r "my.op" mlir/test/` is frequently the fastest answer available.
- **`mlir/examples/`** contains the Toy tutorial's code and the standalone out-of-tree template.
- **The Doxygen reference** (<https://mlir.llvm.org/doxygen/>) for anything the prose documentation
  does not cover, which is most of the C++ API surface.

### On using AI assistants for MLIR

Worth a caution given how fast the API moves: model training data lags MLIR by enough that generated
code frequently uses removed APIs — member `dyn_cast`, `NoSideEffect`, the pre-split standard
dialect, older GPU pipelines. Treat generated MLIR code as a starting sketch and check it against the
version you are building.


---

[← MLIR Release Notes](01-release-notes.md) · [Index](../README.md) · [Material Not Mirrored Here →](03-excluded-material.md)
