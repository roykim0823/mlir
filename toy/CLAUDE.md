# Toy tutorial — chapter README conventions

`Ch2/README.md` is the reference layout for Ch2–Ch7; `Ch1/README.md` keeps its own
layout (upstream Ch-1 has no build/run part). When editing `ChN/README.md`, apply
these rules and check the result against Ch2.

## Follow and cover the upstream chapter

- Each chapter mirrors the official tutorial chapter. Fetch its source from the
  release branch matching the toolchain:
  `https://raw.githubusercontent.com/llvm/llvm-project/release/20.x/mlir/docs/Tutorials/Toy/Ch-N.md`
- **Follow upstream's section order.** Material upstream doesn't have (e.g. MLIRGen
  in Ch2) goes next to the upstream section it belongs with.
- **Cover every upstream topic**, in more detail: use this repo's real code and
  output instead of upstream's snippets, and say where upstream is outdated for
  MLIR 20 (renamed accessors, properties `<{...}>`, removed fields, ...) — verified
  by running the tools, never assumed.
- Right after the overview, add `### How this README maps to the upstream chapter`:
  a table *upstream section → section here*, plus one line listing what goes beyond
  upstream (see Ch2).
- Keep cross-references few and plain ("section 6"); don't scatter links.
- Don't keep a subsection that is only a paragraph or two: fold it into its
  neighbor as a bold-lead paragraph (e.g. Ch2 folded "Locations are mandatory" into
  2.2).

## Real output inside concept and code sections (Ch2 2.3/2.4 scheme)

Wherever a concept or code section shows real output, make it reproducible on the
spot, in this order:

1. **The code behind it:** a labeled excerpt of the repo file that produces the
   behavior (`***mlir/MLIRGen.cpp***`, a generated `build/...inc` line, ...) and/or
   the labeled input file (`***test_Example/Toy/ChN/x.toy***`).
2. **The exact command**, in a `bash` block starting with
   `cd /Users/roy/study/mlir/toy/ChN`.
3. **The real output** from running that command, with the exit status when it
   matters ("Real output (exit status 3):"), abridged with `...` if long.
4. Bullets or prose tying the output back to the concept.

Group related runs under `####` headings inside one subsection (e.g. "#### A tool
only understands the dialects it has loaded"). Input that isn't a repo file (an
upstream snippet) goes on stdin via a quoted heredoc (`<<'EOF'`); a repo file that
needs a tweak is piped through `sed`. Never add test files for this. Run every
command exactly as written and compare it with the shown output (exit status
included) before finishing.

## Section order

1. Overview + "How this README maps to the upstream chapter" (+ optionally a
   "Where Chapter N lives in this repo" table — its `./build.sh chN` / `./run.sh chN`
   rows are the only mention of the superbuild scripts)
2. Concept and code sections, **in upstream's order**
3. **Build and Run** — where upstream runs the complete example, normally its last
   section (upstream's "Complete Toy Example" / "running the tool" part maps here).
   If upstream runs commands in the middle of a chapter, show that run at the
   upstream spot using the real-output scheme above (code, exact command, output);
   Build and Run keeps the complete end-to-end runs.
4. Key Takeaways & Pitfalls
5. `## Appendix: What Chapter N adds to the build`
6. Links

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
  give the explicit flags only as the fallback. Link to the appendix.
- The standalone binary is `./build/toyc-chN` (not `build/bin/` — only the
  superbuild sets `CMAKE_RUNTIME_OUTPUT_DIRECTORY`).
- Every command block starts with `cd /Users/roy/study/mlir/toy/ChN`; test inputs are
  `../../test_Example/Toy/ChN/<file>.toy`; binaries are `./build/toyc-chN`.
- Keep the explanation of the driver's arguments (input file, `-emit=...`, flags);
  that is about the driver, not the script.
- Running, captured output, mapping output back to source, error paths, and
  "poking at the driver" all live here as subsections.

## Appendix

The chapter's `CMakeLists.txt` targets section and its explanation ("What Chapter N
adds to the build": sources, TableGen rules, link libraries) moves to the appendix.
Chapter-specific build-tooling notes (e.g. Ch2's `run_mlir-tblgen.sh`) go there too.

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
- Build and Run comes after the code sections, so its references to them point
  back ("explained in section N"). Concept sections show their own real output
  using the scheme above rather than only pointing forward to Build and Run.

## Verify, don't copy

- Regenerate every "actual output" block by running the documented commands from
  `ChN/` — locations in dumps include the path as given (`@../../test_Example/...`)
  and line numbers shift if a test file changed.
- If a test file under `test_Example/` was edited, check its `# CHECK:` lines still
  match the real output and tell the user if they don't.
