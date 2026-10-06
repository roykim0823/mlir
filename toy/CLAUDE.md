# Toy tutorial — chapter README conventions

`Ch2/README.md` is the reference layout for Ch2–Ch7; `Ch1/README.md` keeps its own
section layout (language, Build and Run, code walkthrough — upstream Ch-1 has no
build/run part) but follows every other rule here: the real-output scheme, command
numbering with `examples.sh`, no upstream mentions, lettered appendices (A build,
B `examples.sh`). When editing `ChN/README.md`, apply these rules and check the
result against Ch2.

## Follow and cover the upstream chapter

- Each chapter mirrors the official tutorial chapter. Fetch its source from the
  release branch matching the toolchain:
  `https://raw.githubusercontent.com/llvm/llvm-project/release/20.x/mlir/docs/Tutorials/Toy/Ch-N.md`
- **Follow upstream's section order.** Material upstream doesn't have (e.g. MLIRGen
  in Ch2) goes next to the upstream section it belongs with.
- **Cover every upstream topic**, in more detail: use this repo's real code and
  output instead of upstream's snippets, verified by running the tools, never
  assumed.
- **The README is a superset of upstream, not a copy.** Upstream's structure and
  text are the starting point; add what the repo has beyond them (an extra step, a
  field, a run) wherever it is worthwhile, in the same style.
- **Don't write about upstream itself.** Link the upstream chapter in the Goal line
  and under Links, and nowhere else. No "how this README maps to upstream" table, no
  "upstream says X / MLIR 20 does Y" or "older MLIR called it ..." asides, no "(not
  in the upstream chapter)" markers: describe MLIR 20 as it is. A snippet that isn't
  this repo's code says "(not this repo's code)" without naming its source.
- Keep cross-references few and plain ("section 6"); don't scatter links.
- Don't keep a subsection that is only a paragraph or two: fold it into its
  neighbor as a bold-lead paragraph (e.g. Ch2 folded "Locations are mandatory" into
  2.2).

## Real output inside concept and code sections (Ch2 2.3/2.4 scheme)

Wherever a concept or code section shows real output, make it reproducible on the
spot, in this order:

1. **The code behind it:** a labeled excerpt of the repo file that produces the
   behavior (`***mlir/MLIRGen.cpp***`, `***include/toy/Ops.td***`, ...) and/or the
   labeled input file (`***test_Example/Toy/ChN/x.toy***`). Generated code is not
   quoted from `build/*.inc`; it is the output of the command that prints it.
2. **The exact command**, in a `bash` block starting with
   `cd /Users/roy/study/mlir/toy/ChN`.
3. **The real output** from running that command, with the exit status when it
   matters ("Real output (exit status 3):"). If long, abridge whole lines with `...`
   and say "abridged" in the label; if a single line is shortened, the label says
   what ("the dialect list is abridged").
4. Bullets or prose tying the output back to the concept.

Group related runs under `####` headings inside one subsection (e.g. "#### A tool
only understands the dialects it has loaded"). Input that isn't a repo file (an
upstream snippet) goes on stdin via a quoted heredoc (`<<'EOF'`); a repo file that
needs a tweak is piped through `sed`. Never add test files for this — the one
exception is a step-by-step walkthrough's stage files (below). Run every command
exactly as written and compare it with the shown output (exit status included)
before finishing.

Every chapter numbers its commands (Ch1–Ch7): each runnable `bash`
block's output label carries **[N]** ("Real output **[7]** (exit status 1):"), N
being the block's position among runnable blocks (excluded: the Building block —
its first command after `cd` is `cmake` — and `***label***` excerpts), and `ChN/examples.sh [N]` reads and runs them straight from the
README. There is one script, `toy/examples.sh`; each `ChN/examples.sh` is a symlink
to it (`ln -s ../examples.sh ChN/examples.sh`), so change the script in `toy/` only. After adding,
removing or reordering a command block, renumber the labels and the count in the
Build and Run intro, regenerate appendix F's `./examples.sh -l` sample, then check
`./examples.sh -l` and every output against the README. Prose doesn't cite command
numbers (they shift); it cites sections.

## Section order

1. Overview (+ optionally a "Where Chapter N lives in this repo" table — its
   `./build.sh chN` / `./run.sh chN` rows are the only mention of the superbuild
   scripts)
2. Concept and code sections, **in upstream's order**
3. **Build and Run** — where upstream runs the complete example, normally its last
   section (upstream's "Complete Toy Example" / "running the tool" part maps here).
   If upstream runs commands in the middle of a chapter, show that run at the
   upstream spot using the real-output scheme above (code, exact command, output);
   Build and Run keeps the complete end-to-end runs.
4. Key Takeaways & Pitfalls
5. Lettered appendices, `## Appendix A: ...`, `## Appendix B: ...`, with `### A.1`
   subsections (see "Main path vs appendices" below)
6. Links

## Main path vs appendices

- Sections 1–7 carry the **essential path**: upstream's topics, in upstream's order,
  plus what this repo needs to make them concrete (e.g. MLIRGen in Ch2).
- **Reference material and extras** go to lettered appendices: per-item references
  beyond upstream's worked example (Ch2: the other eight ops), file-wide recaps (a
  map of `Ops.td`, its fields, the generated code), printing/tool details, more
  inputs, the build wiring ("What Chapter N adds to the build"), and `examples.sh`.
  The Overview says in one paragraph what each appendix holds. Ch2's layout: A other
  ops, B `Ops.td` and `Dialect.cpp` as a whole, C printing details and stock `mlir-opt`, D more
  inputs, E build, F `examples.sh`.
- The main path points to an appendix in one phrase ("appendix A.3") instead of
  repeating it; appendices point back to the main section they extend.
- **Show each listing and run once.** If two sections need the same input file or
  the same output, show it in one and point to it from the other (Ch2's section 6
  points to section 4.5 for `codegen.toy`); an abridged version plus a pointer to the
  full one is fine.
- Every concept keeps the concept → code → exact command → real output scheme, in
  the appendices too. Generated code is shown by the `mlir-tblgen ... | sed/awk`
  command that prints it, not quoted from a pre-generated `build/*.inc` file, and a
  claim like "the verifier rejects X" comes with the run that shows it.
- **Explain generated output, don't just show it.** The first time a generated
  class or file appears, explain it member by member (Ch2 4.4 step 0: a table
  for the shell's class, a bullet per part of its definitions); later steps explain what
  changed (new traits, new methods, new checks).
- When the tutorial shows hand-written C++ for something the repo does
  declaratively (Ch2 4.5: `toy.print`'s `parse`/`print`), mark it "(not this repo's
  code)" and show what ODS generates for the repo's declaration next to it.
- A step-by-step walkthrough (Ch2 4.4) shows **each step's own definition**, based
  on upstream's step text, not excerpts of the repo file. Each stage is a
  standalone file next to the real one (Ch2: `include/toy/Ops/OpsN.td` for step N;
  step 0, `Ops0.td`, is the empty shell and shows the full prelude), holding the includes, the dialect and
  base class records it needs, and that stage's definition; each contains the
  previous stage's fields plus the new ones, so `diff` between two files shows one
  step. Show the file as a labeled excerpt (the shared prelude elided as `...` after
  the first), then run `mlir-tblgen` on it; the output shows what that step's fields
  generate. The build never reads these files.
  Steps beyond upstream are fine when the repo's file has the field (Ch2's step 5,
  `Pure`). A behavior with two outcomes gets both runs (e.g. a verifier: one input
  that passes, one that fails). The real file's record and its full generated class
  come last, compared with the hand-written version.

Number sections consecutively; after moving anything, fix every cross-reference
("section X.Y", "§X", "(X.Y)", table cells) in the file. When renumbering with a
script, touch only reference patterns (never `loc(...)`/`@file:line:col` numbers)
and re-check each reference against its target heading afterwards.

## Build and Run section

- Open with one line pointing to the top README for `build.sh`, `run.sh`, and
  `CMakePresets.json`. Do **not** re-document `build.sh`/`run.sh` (no `./build.sh chN`
  block, no run-script walkthrough, no pitfalls about `--fresh` or `run.sh` paths).
- Subsection "Building": standalone build, run from the chapter directory:
  ```bash
  cd /Users/roy/study/mlir/toy/ChN
  cmake -S . -B build -G Ninja
  cmake --build build          # → ./build/toyc-chN
  ```
  No `-DMLIR_DIR`/`-DCMAKE_CXX_COMPILER` flags: the shell sets `CXX`/`CC` to
  Homebrew llvm@20 clang and has `/opt/homebrew/opt/llvm@20/bin` on `PATH`, so
  `find_package` finds `llvm@20/lib/cmake/{mlir,llvm}`. Explain this briefly and
  give the explicit flags only as the fallback. Point to the build appendix.
- The standalone binary is `./build/toyc-chN` (not `build/bin/` — only the
  superbuild sets `CMAKE_RUNTIME_OUTPUT_DIRECTORY`).
- Every command block starts with `cd /Users/roy/study/mlir/toy/ChN`; test inputs are
  `../../test_Example/Toy/ChN/<file>.toy` (never a chapter-local copy); binaries are
  `./build/toyc-chN`.
- Keep the explanation of the driver's arguments (input file, `-emit=...`, flags);
  that is about the driver, not the script.
- Subsections: Building, Running (the complete run with its full output and the
  line-by-line mapping back to the source), the round trip, and inspecting the
  generated code when the chapter has TableGen (Ch2 6.4). Extra inputs and other
  tools (stock `mlir-opt`) go to appendices (Ch2: D, C).

## Build appendix

The chapter's `CMakeLists.txt` targets section and its explanation ("What Chapter N
adds to the build": sources, TableGen rules, link libraries) goes to its own
appendix (Ch2: E). Tooling that upstream asks the reader to use stays in the main
path (Ch2's `run_mlir-tblgen.sh` is section 6.4, because upstream says to study the
generated C++); other chapter-specific build-tooling notes go to the build appendix.

## Code blocks

- Every code block copied from a real file gets a label line **directly above the
  fence, no blank line in between**:
  ~~~markdown
  ***include/toy/Lexer.h***
  ```cpp
  ~~~
  Paths are relative to the chapter directory (`include/toy/Ops.td`,
  `mlir/MLIRGen.cpp`, `CMakeLists.txt`) except test inputs, which use
  `test_Example/Toy/ChN/<file>.toy`. Repeat the label on every block, even
  consecutive ones from the same file. Do not put a label at the top of a section
  followed by prose.
- Blocks not from a file (grammar sketches, shell commands, program output) get
  no label.
- Merge small blocks from the same file into fewer blocks when their source lines
  are close together, eliding gaps with `...` and saying "(abridged)" in the prose.
  Rewrite the surrounding prose so nothing is lost.
- Excerpts must match the source text (comments included); don't compress code
  into one-line pseudo-declarations — elide bodies with `{ ... }` instead. Older
  READMEs often paraphrase code or add explanatory comments inside blocks; rebuild
  such blocks from the real line ranges and move the explanation into prose.
  Excerpts of class members may be dedented to top level.
- `Ops.td` `description = [{ ... }]` bodies often contain ```` ```mlir ```` fences,
  which break a surrounding markdown fence — elide the description body with `...`.
- Hypothetical code (e.g. "what it would look like in raw C++") and snippets quoted
  from the upstream docs are not from a repo file: no label.

## Content placement

- A grammar derived from the parser belongs in the parser part of the code
  walkthrough (right before the production-by-production list), not in the
  language section.
- Explain a language or tool where it is first needed. Ch2 3.2 shows the TableGen
  dialect record right after its C++ equivalent (3.1); 4.3 explains TableGen and ODS
  right before the first op walkthrough (what they are and `Toy_Op`), and 4.4's
  step 0 explains the shell's `--print-records` record, generated class and
  definitions; appendix B recaps the
  whole file (the syntax table on real lines, a map of `Ops.td`, every op field it
  sets, the generated code). Later chapters refer back to these and explain only the
  `.td` constructs and fields they add (e.g. DRR in Ch3), in the same style.
- Build and Run comes after the code sections, so its references to them point
  back ("explained in section N"). Concept sections show their own real output
  using the scheme above rather than only pointing forward to Build and Run.

## Verify, don't copy

- Regenerate every "actual output" block by running the documented commands from
  `ChN/` — locations in dumps include the path as given (`@../../test_Example/...`)
  and line numbers shift if a test file changed.
- If a test file under `test_Example/` was edited, check its `# CHECK:` lines still
  match the real output and tell the user if they don't.
- A "what if the code were different" result (Ch3 3.5: `TransposeOp` without `Pure`)
  is a runnable command too: copy the chapter's sources to `$(mktemp -d)`, edit the
  copy with `sed`, build and run it, and remove the directory — never edit the repo
  or its build tree for a demo. Since the block ends with `rm`, its own exit status is `rm`'s: end the
  tool's line with `; echo "<tool> exit status: $?"` so the real one is shown, and if
  the patched tool aborts, leave bash's `Abort trap` line out (see below).
- Output the shell adds rather than the tool (e.g. bash's `Abort trap: 6` report,
  which carries a process ID) is left out of the shown output, and the label says
  so; everything else is shown exactly as printed — trailing spaces included (a
  JIT's `printf("%f ")` lines end in a space; some MLIR printer lines do too).
