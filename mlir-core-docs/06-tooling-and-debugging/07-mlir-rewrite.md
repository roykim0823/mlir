# `mlir-rewrite`

> **Section:** Tooling and Debugging · document 7 of 7  
> **Upstream:** [https://mlir.llvm.org/docs/Tools/mlir-rewrite/](https://mlir.llvm.org/docs/Tools/mlir-rewrite/) · source [`mlir/docs/Tools/mlir-rewrite.md`](https://github.com/llvm/llvm-project/blob/main/mlir/docs/Tools/mlir-rewrite.md)  
> **License:** upstream text is Apache-2.0 WITH LLVM-exception.

## Orientation

A small utility for making textual edits to `.mlir` files while preserving their formatting and
comments — as opposed to `mlir-opt`, which round-trips through the parser and reprints, discarding
comments and normalising layout.

Niche but occasionally exactly what you need: mass-renaming an operation across a large test suite
without producing a diff that touches every line.

**Read first**

- [Using `mlir-opt`](01-using-mlir-opt.md)

**What you should be able to do after this page**

- Make mechanical edits across many test files without reformatting them.

---

## Upstream documentation

Tool to simplify rewriting .mlir files. There are a couple of build in rewrites
discussed below along with usage.

Note: This is still in very early stage. Its so early its less a tool than a
growing collection of useful functions: to use its best to do what's needed on
a brance by just hacking it (dialects registered, rewrites etc) to say help
ease a rename, upstream useful utility functions, point to ease others
migrating, and then bin eventually. Once there are actually useful parts it
should be refactored same as mlir-opt.


## simple-rename

Rename per op given a substring to a target. The match and replace uses LLVM's
regex sub for the match and replace while the op-name is matched via regular
string comparison. E.g.,

```
mlir-rewrite input.mlir -o output.mlir --simple-rename \
   --simple-rename-op-name="test.concat" --simple-rename-match="axis" \
                                         --simple-rename-replace="bxis"
```

to replace `axis` substring in the text of the range corresponding to
`test.concat` ops with `bxis`.

---

## Deeper notes

### The problem it solves

Rename an operation in your dialect and every test mentioning it must change. Running those tests
through `mlir-opt` would fix the names and also strip every `// CHECK:` comment — which is the entire
content of the tests. `mlir-rewrite` edits the text in place.

### Scope

It is a small tool with a narrow purpose. For anything structural, write a pass. For anything textual
and simple, `sed` may be enough; `mlir-rewrite`'s advantage is that it parses, so it edits operations
rather than strings and will not corrupt a file whose text happens to match.

### When you will reach for it

Almost exclusively during dialect refactoring: renaming an operation, changing an attribute name,
adjusting a syntax. Those are rare events, which is why the tool is obscure — but when one happens
across a few hundred test files, knowing it exists saves a genuinely tedious afternoon.


---

[← `mlir-reduce`](06-mlir-reduce.md) · [Index](../README.md) · [MLIR C API →](../07-bindings-and-embedding/01-c-api.md)
