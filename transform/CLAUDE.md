# Transform dialect tutorial — chapter README conventions

The chapters are `Ch0`, `Ch1`, `Ch2`, `Ch3`, `Ch4`, `ChH`, one directory each, each
with a `README.md` and an `examples.sh` symlink (`ln -s ../examples.sh ChN/examples.sh`;
change the script in `transform/` only). These conventions are the Toy tutorial's
(`../toy/CLAUDE.md`), adapted; where this file is silent, follow that one.

## Toolchain and paths

- Homebrew LLVM/MLIR **20.1.8** (`/opt/homebrew/opt/llvm@20/bin` on `PATH`:
  `mlir-opt`, `mlir-tblgen`, `FileCheck`). Describe MLIR 20 as it is.
- Upstream source: the release branch matching the toolchain:
  `https://raw.githubusercontent.com/llvm/llvm-project/release/20.x/mlir/docs/Tutorials/transform/ChN.md`
  (code: `mlir/examples/transform/ChN`, tests: `mlir/test/Examples/transform/ChN`).
- Test inputs are the release/20.x tests, in `../test_Example/transform/ChN/*.mlir`
  (reference them from a chapter as `../../test_Example/transform/ChN/<file>.mlir`;
  never a chapter-local copy). Ch0 has none: its snippets go on stdin through quoted
  heredocs (`<<'EOF'`), completed into valid IR (wrapped in a `func.func`, with the
  operands it needs) so they parse.
- Ch0, Ch1 and ChH have no C++: their tool is stock `mlir-opt`. Ch2–Ch4 build
  `./build/transform-opt-chN` standalone (`cmake -S . -B build -G Ninja && cmake
  --build build`, from the chapter directory; the standalone guard at the top of
  `ChN/CMakeLists.txt` defaults to Release, needed because `libMLIR.dylib` is built
  with `NDEBUG`, and the tool links only `libMLIR.dylib`).
- Upstream test-only passes (`--test-transform-dialect-erase-schedule`,
  `--test-lower-to-llvm`, ...) are not in the installed `mlir-opt`: use the stock
  equivalent pipeline and say once what the test pass would have done.

## README layout

- Title `# Chapter N: <upstream title>`, then the Goal line linking the upstream
  chapter (`https://mlir.llvm.org/docs/Tutorials/transform/ChN/`). Link upstream only
  there and under Links; don't write about upstream elsewhere.
- `## 1. Overview` (ending with one paragraph on the appendices), then the concept
  sections **in upstream's order, covering every upstream topic and every upstream
  code listing** — the README is a superset of upstream: its text and listings are
  the starting point, expanded with explanation and real runs.
- Ch2–Ch4: `## N. Build and Run` after the concept sections (Building subsection
  with the `cmake` block, then the complete runs). Ch0/Ch1/ChH have no build section.
- Ch1–Ch4 and ChH: `## N. Deriving ... yourself` as the last numbered section before
  Key Takeaways (model: `../toy/Ch6/README.md` section 5): an intro and a table (each
  part of the chapter's new code → where it comes from → step), then `### N.k Step
  k: ...` subsections that each run a discovery tool with real output (`mlir-tblgen`
  output, `grep` of installed headers or `.td` files, `mlir-opt --help`, verifier,
  interpreter or linker errors from a scratch copy in `$(mktemp -d)`). In Ch2–Ch4 the
  new code is the C++ and build wiring relative to the previous chapter; in Ch1 and
  ChH, which have no C++, it is the transform schedules.
- `## N. Key Takeaways & Pitfalls`, lettered appendices (`## Appendix A: ...`,
  `### A.1`), and `## Links` last. The last appendix is `examples.sh` (usage +
  `./examples.sh -l` sample).

## Real output and numbered commands

- Every concept that has observable behavior shows: the code (a labeled excerpt
  `***lib/MyExtension.cpp***` / `***test_Example/transform/Ch1/sequence.mlir***`
  directly above the fence), the exact command in a `bash` block starting with
  `cd /Users/roy/study/mlir/transform/ChN`, and the **real output** of running it
  (abridge whole lines with `...` and say "abridged" in the label), then prose tying
  the output back to the concept. Failures show the diagnostic and the exit status.
- Each runnable `bash` block's output label carries **[N]**, its position among
  runnable blocks ("Real output **[7]** (exit status 1):"); the `cmake` build block
  and `***label***` excerpts are not counted. `ChN/examples.sh N` runs block N
  straight from the README. Run every command exactly as written and compare its
  output with the README before finishing; check `./examples.sh -l`.
- Section headings that `examples.sh` reads: `## 2. ...`, `### 2.1 ...`,
  `## Appendix A: ...`, `### A.1 ...`.
- Stage files for a step-by-step walkthrough (e.g. a schedule built up piece by
  piece) may live in the chapter directory (`ChN/stages/stepK.mlir`), each a
  complete runnable file; nothing else is added outside the README.
- Output added by the shell rather than the tool is left out; everything else is
  shown exactly as printed.

## Style

- Spell out an abbreviation at its first use in each chapter.
- Excerpts match the source text; elide with `...`; `// (N)` callout markers are
  the one allowed addition when prose explains a long block piece by piece.
- Insight sentences: one per point, right after the run that earns it, concrete and
  true for MLIR 20 (see `../toy/CLAUDE.md`, "Insight sentences").
- Number sections consecutively and keep cross-references ("section 3.2") correct.
