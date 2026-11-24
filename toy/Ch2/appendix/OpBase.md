# Appendix G: `OpBase.td` and the files it includes

Every ODS file starts with `include "mlir/IR/OpBase.td"`, and so does this chapter's `include/toy/Ops.td`. That one line brings in ten TableGen files, 4396 lines: `OpBase.td` itself and the nine files it includes, directly or through another file. They define the vocabulary an op definition is written in: `Dialect`, `Op`, `OpBuilder`, `ins`/`outs`, traits such as `Terminator`, and constraints such as `F64Tensor`.

This appendix shows how the ten files include each other, how their classes build on each other, what each file is for and how it is used, and which of its definitions Toy's `Ops.td` uses. It ends with a view on how much of this an MLIR compiler engineer needs to know (G.9).

The ten files are copied into `appendix/OpBase/mlir/IR/`, so they can be read and searched next to this text. The copies come from LLVM 20.1.8, the toolchain this repo builds with. The build never reads them: it reads the installed ones in `/opt/homebrew/opt/llvm@20/include`.

The commands here are run from the chapter directory, like the README's, but `examples.sh` doesn't number them: it reads `README.md` only.

- [G.1 The copies](#g1-the-copies)
- [G.2 The include hierarchy](#g2-the-include-hierarchy)
- [G.3 What each file holds](#g3-what-each-file-holds)
- [G.4 How the files relate](#g4-how-the-files-relate)
- [G.5 The files, one by one](#g5-the-files-one-by-one)
- [G.6 All ten files in one small dialect](#g6-all-ten-files-in-one-small-dialect)
- [G.7 One name, followed through the files: `F64Tensor`](#g7-one-name-followed-through-the-files-f64tensor)
- [G.8 What `OpBase.td` does not bring](#g8-what-opbasetd-does-not-bring)
- [G.9 How much of this you need to know](#g9-how-much-of-this-you-need-to-know)

---

## G.1 The copies

```text
appendix/OpBase/mlir/IR/
├── AttrTypeBase.td
├── CommonAttrConstraints.td
├── CommonTypeConstraints.td
├── Constraints.td
├── DialectBase.td
├── Interfaces.td
├── OpBase.td
├── Properties.td
├── Traits.td
└── Utils.td
```

Next to it, `appendix/OpBase/demo/Demo.td` is a small dialect written for this appendix (G.6); it is not an LLVM file.

The folder keeps the `mlir/IR/` path on purpose. The files include each other as `"mlir/IR/Constraints.td"`, so `-I appendix/OpBase` makes `mlir-tblgen` resolve those includes to the copies (G.2 uses this).

Each copy is byte-for-byte the installed file:

```bash
cd /Users/roy/study/mlir/toy/Ch2
for f in appendix/OpBase/mlir/IR/*.td; do
  cmp "$f" "/opt/homebrew/opt/llvm@20/include/mlir/IR/$(basename "$f")" && echo "identical: $(basename "$f")"
done
```

Real output:

```text
identical: AttrTypeBase.td
identical: CommonAttrConstraints.td
identical: CommonTypeConstraints.td
identical: Constraints.td
identical: DialectBase.td
identical: Interfaces.td
identical: OpBase.td
identical: Properties.td
identical: Traits.td
identical: Utils.td
```

If the toolchain is upgraded, this command reports the files that changed (`cmp` prints the first difference instead of `identical`).

---

## G.2 The include hierarchy

#### Who includes whom

Every `include` line of the ten files:

```bash
cd /Users/roy/study/mlir/toy/Ch2
cd appendix/OpBase/mlir/IR
grep '^include' *.td
```

Real output:

```text
AttrTypeBase.td:include "mlir/IR/CommonAttrConstraints.td"
AttrTypeBase.td:include "mlir/IR/CommonTypeConstraints.td"
AttrTypeBase.td:include "mlir/IR/Constraints.td"
AttrTypeBase.td:include "mlir/IR/DialectBase.td"
AttrTypeBase.td:include "mlir/IR/Traits.td"
CommonAttrConstraints.td:include "mlir/IR/Constraints.td"
CommonAttrConstraints.td:include "mlir/IR/CommonTypeConstraints.td"
CommonAttrConstraints.td:include "mlir/IR/DialectBase.td"
CommonTypeConstraints.td:include "mlir/IR/Constraints.td"
CommonTypeConstraints.td:include "mlir/IR/DialectBase.td"
Constraints.td:include "mlir/IR/Utils.td"
DialectBase.td:include "mlir/IR/Utils.td"
Interfaces.td:include "mlir/IR/AttrTypeBase.td"
Interfaces.td:include "mlir/IR/Constraints.td"
Interfaces.td:include "mlir/IR/Traits.td"
OpBase.td:include "mlir/IR/Constraints.td"
OpBase.td:include "mlir/IR/DialectBase.td"
OpBase.td:include "mlir/IR/Interfaces.td"
OpBase.td:include "mlir/IR/Properties.td"
OpBase.td:include "mlir/IR/Traits.td"
OpBase.td:include "mlir/IR/Utils.td"
OpBase.td:include "mlir/IR/AttrTypeBase.td"
Properties.td:include "mlir/IR/Constraints.td"
Properties.td:include "mlir/IR/Utils.td"
Traits.td:include "mlir/IR/Constraints.td"
```

That is 25 `include` lines among 10 files. No file includes anything outside the ten, and `Utils.td` includes nothing.

#### As a tree

The same 25 lines drawn as a tree, starting from `OpBase.td` and following each file's includes in the order they are written. A `*` marks a file that has already appeared higher up; its own includes are not repeated.

```text
OpBase.td
├── Constraints.td
│   └── Utils.td
├── DialectBase.td
│   └── Utils.td *
├── Interfaces.td
│   ├── AttrTypeBase.td
│   │   ├── CommonAttrConstraints.td
│   │   │   ├── Constraints.td *
│   │   │   ├── CommonTypeConstraints.td
│   │   │   │   ├── Constraints.td *
│   │   │   │   └── DialectBase.td *
│   │   │   └── DialectBase.td *
│   │   ├── CommonTypeConstraints.td *
│   │   ├── Constraints.td *
│   │   ├── DialectBase.td *
│   │   └── Traits.td
│   │       └── Constraints.td *
│   ├── Constraints.td *
│   └── Traits.td *
├── Properties.td
│   ├── Constraints.td *
│   └── Utils.td *
├── Traits.td *
├── Utils.td *
└── AttrTypeBase.td *
```

Two things to read off the tree:

- **`OpBase.td` includes seven files directly, and two more arrive through them.** `CommonAttrConstraints.td` and `CommonTypeConstraints.td` are never named in `OpBase.td`; they come in through `Interfaces.td` → `AttrTypeBase.td`. These two are the biggest files, and they hold the names an op definition uses most (`F64Tensor`, `F64ElementsAttr`).
- **Most includes are repeats.** Of the 25 `include` lines, only 9 bring in a new file. The other 16 (the `*` entries) name a file that is already loaded.

**Repeats cost nothing because every file has an include guard.** TableGen's `include` is textual, like C's `#include`, and each of the ten files wraps its contents the same way a C header does:

***appendix/OpBase/mlir/IR/Traits.td***
```tablegen
#ifndef TRAITS_TD
#define TRAITS_TD

include "mlir/IR/Constraints.td"
...
#endif // TRAITS_TD
```

The second time a file is included, its guard macro is already defined and the whole body is skipped. So each file is processed once, the first time the tree reaches it. Each file can list everything it needs without caring who else has already included it.

#### As layers

Sorted by what each file needs, the ten files form seven layers. A file only includes files from the layers above it in this table:

| Layer | File | Includes |
|---|---|---|
| 0 | `Utils.td` | nothing |
| 1 | `Constraints.td` | `Utils` |
| 1 | `DialectBase.td` | `Utils` |
| 2 | `Traits.td` | `Constraints` |
| 2 | `Properties.td` | `Constraints`, `Utils` |
| 2 | `CommonTypeConstraints.td` | `Constraints`, `DialectBase` |
| 3 | `CommonAttrConstraints.td` | `Constraints`, `CommonTypeConstraints`, `DialectBase` |
| 4 | `AttrTypeBase.td` | `CommonAttrConstraints`, `CommonTypeConstraints`, `Constraints`, `DialectBase`, `Traits` |
| 5 | `Interfaces.td` | `AttrTypeBase`, `Constraints`, `Traits` |
| 6 | `OpBase.td` | `Constraints`, `DialectBase`, `Interfaces`, `Properties`, `Traits`, `Utils`, `AttrTypeBase` |

G.5 walks the files in this order, so each file only uses names that have already been explained.

#### The tool's own list

`mlir-tblgen` can report which files it read: `-d` writes a dependency file, the same mechanism the build uses for its `.inc.d` files. Here the input is the copy of `OpBase.td`, the only include path is the copies' folder, and the dependency file goes to the terminal:

```bash
cd /Users/roy/study/mlir/toy/Ch2
mlir-tblgen --print-records appendix/OpBase/mlir/IR/OpBase.td -I appendix/OpBase -o /dev/null -d /dev/stdout | tr ' ' '\n'
```

Real output:

```text
/dev/null:
appendix/OpBase/mlir/IR/AttrTypeBase.td
appendix/OpBase/mlir/IR/CommonAttrConstraints.td
appendix/OpBase/mlir/IR/CommonTypeConstraints.td
appendix/OpBase/mlir/IR/Constraints.td
appendix/OpBase/mlir/IR/DialectBase.td
appendix/OpBase/mlir/IR/Interfaces.td
appendix/OpBase/mlir/IR/Properties.td
appendix/OpBase/mlir/IR/Traits.td
appendix/OpBase/mlir/IR/Utils.td
```

- The first line is the make-style target (`-o /dev/null`), and `tr` puts each dependency on its own line.
- The nine files are exactly the nine in the tree. The input file itself, `OpBase.td`, is not listed as its own dependency, which makes ten.
- The run succeeded with `-I appendix/OpBase` as the only include path, so the ten copies are complete: nothing else is needed to process `OpBase.td`.

**The copies behave like the installed files.** The chapter's `Ops0.td` (section 4.4 of the README) includes only `OpBase.td`, so it can be processed against either set, and the resolved records are the same:

```bash
cd /Users/roy/study/mlir/toy/Ch2
diff <(mlir-tblgen --print-records include/toy/Ops/Ops0.td -I appendix/OpBase) \
     <(mlir-tblgen --print-records include/toy/Ops/Ops0.td -I /opt/homebrew/opt/llvm@20/include) \
  && echo identical
```

Real output:

```text
identical
```

---

## G.3 What each file holds

A TableGen file holds two kinds of things (README appendix B.1): a `class` is a template for records, and a `def` is a concrete record. Counting both per file, in layer order:

```bash
cd /Users/roy/study/mlir/toy/Ch2
cd appendix/OpBase/mlir/IR
for f in Utils Constraints DialectBase Traits Properties CommonTypeConstraints CommonAttrConstraints AttrTypeBase Interfaces OpBase; do
  printf '%-26s %4d lines %3d classes %3d defs\n' $f.td $(wc -l < $f.td) $(grep -c '^class ' $f.td) $(grep -c '^def ' $f.td)
done
```

Real output:

```text
Utils.td                     93 lines   7 classes   2 defs
Constraints.td              194 lines  14 classes   5 defs
DialectBase.td              100 lines   1 classes   0 defs
Traits.td                    68 lines   6 classes   0 defs
Properties.td               790 lines  20 classes  10 defs
CommonTypeConstraints.td    938 lines  83 classes 116 defs
CommonAttrConstraints.td    874 lines  60 classes  73 defs
AttrTypeBase.td             472 lines  31 classes   1 defs
Interfaces.td               192 lines  12 classes   0 defs
OpBase.td                   675 lines  60 classes  33 defs
```

The columns add up to 4396 lines, 294 classes and 240 defs. What each file is for, and what Toy's `Ops.td` takes from it:

| File | What it defines | Used by Toy's `Ops.td` |
|---|---|---|
| `Utils.td` | Small helpers: the `ins`/`outs` markers, `CArg`, deprecation marks, string helpers | `ins`, `outs`, `CArg` |
| `Constraints.td` | Predicates (`Pred`, `CPred`, `And`, `Or`, `Neg`) and the `Constraint` base classes | nothing by name; every constraint below is built from it |
| `DialectBase.td` | The `Dialect` class | `Dialect` (`Toy_Dialect`) |
| `Traits.td` | The `Trait` base classes | `Trait` (the type of `Toy_Op`'s `traits` parameter) |
| `Properties.td` | `Property`: non-attribute data stored inside an op | nothing (G.6 shows one) |
| `CommonTypeConstraints.td` | The `Type` class and ready-made type constraints | `F64Tensor`, `F64`, `StaticShapeTensorOf`, `Variadic`, `FunctionType` |
| `CommonAttrConstraints.td` | The `Attr` class and ready-made attribute constraints | `F64ElementsAttr`, `FlatSymbolRefAttr`, `SymbolNameAttr`, `TypeAttrOf`, `OptionalAttr`, `DictArrayAttr` |
| `AttrTypeBase.td` | `AttrDef` / `TypeDef`: classes for defining a dialect's *own* attributes and types | nothing (Toy has no type of its own until Chapter 7) |
| `Interfaces.td` | Classes for declaring interfaces (`OpInterface`, `InterfaceMethod`) | nothing by name; `FunctionOpInterface` is built from it (G.8) |
| `OpBase.td` | The `Op` class, `OpBuilder`, op traits, regions and successors, multi-entity constraints | `Op`, `OpBuilder`, `Terminator`, `HasParent`, `IsolatedFromAbove`, `AnyRegion`, `region` |

#### Checking the totals with `--print-records`

`--print-records` prints every class and record TableGen holds after reading a file. Counting them for `OpBase.td`:

```bash
cd /Users/roy/study/mlir/toy/Ch2
mlir-tblgen --print-records appendix/OpBase/mlir/IR/OpBase.td -I appendix/OpBase \
  | awk '/^class /{c++} /^def anonymous_/{a++; next} /^def /{d++} END{printf "%d classes, %d named defs, %d anonymous defs\n", c, d, a}'
```

Real output:

```text
294 classes, 240 named defs, 344 anonymous defs
```

- **294 classes and 240 named defs** match the sums of the per-file counts above.
- **344 anonymous defs** have no `def` line in any file. TableGen creates one each time a class is instantiated inline, as in `CPred<"...">` or `TensorOf<[F64]>` used as an argument. G.7 shows two of them.

So before `Ops.td` defines anything, TableGen already holds 294 classes and 584 records. This is why the README's `--print-records` commands filter the output with `awk` (section 4.4): the records of one Toy op are a few dozen lines out of several thousand.

---

## G.4 How the files relate

The ten files are tied together in three ways. G.2 covered the first, `include`. The other two say more about the design.

#### 1. Classes derive from classes in lower files

A file includes another because its classes derive from, or take as arguments, classes defined there. Almost everything grows from two roots, `Constraint` and `Trait`:

```text
Constraint                                    Constraints.td
├── TypeConstraint                            Constraints.td
│   └── Type                                  CommonTypeConstraints.td
│       ├── F, TensorOf, ...  (F64, F64Tensor)
│       └── DialectType                       CommonTypeConstraints.td
│           └── TypeDef                       AttrTypeBase.td
├── AttrConstraint                            Constraints.td
│   └── Attr                                  CommonAttrConstraints.td
│       ├── FloatElementsAttr, ...  (F64ElementsAttr)
│       └── DialectAttr                       CommonAttrConstraints.td
│           └── AttrDef                       AttrTypeBase.td
├── RegionConstraint                          Constraints.td
│   └── Region  (AnyRegion)                   OpBase.td
└── SuccessorConstraint                       Constraints.td
    └── Successor  (AnySuccessor)             OpBase.td

Trait                                         Traits.td
├── NativeTrait                               Traits.td
│   ├── NativeOpTrait  (Terminator)           OpBase.td
│   ├── NativeAttrTrait, NativeTypeTrait      AttrTypeBase.td
│   └── InterfaceTrait                        Interfaces.td
│       └── OpInterfaceTrait                  Interfaces.td
│           └── OpInterface                   Interfaces.td
├── ParamNativeTrait                          Traits.td
│   └── ParamNativeOpTrait  (HasParent)       OpBase.td
├── GenInternalTrait                          Traits.td
│   └── GenInternalOpTrait                    OpBase.td
├── PredTrait                                 Traits.td
│   └── PredOpTrait  (AllTypesMatch)          OpBase.td
└── TraitList                                 Traits.td
```

Four classes stand outside both trees, with no base class: `Dialect` (`DialectBase.td`), `Property` (`Properties.td`), `Interface` (`Interfaces.td`) and `Op` (`OpBase.td`). `OpInterface` derives from two classes at once, `Interface` and `OpInterfaceTrait`.

The pattern is the same in every branch: **a lower file defines a general base class, and a higher file specializes it for one kind of entity.** `Traits.td` says what a native trait is; `OpBase.td` says what a native trait *of an op* is.

`--print-records` prints each record's class chain as a comment, so the trees can be checked. The input is the demo dialect of G.6, which uses something from every file:

```bash
cd /Users/roy/study/mlir/toy/Ch2
mlir-tblgen --print-records appendix/OpBase/demo/Demo.td -I appendix/OpBase \
  | grep -E '^def (Demo_Dialect|SmallTensor|PairType|CostInterface|StopOp|Terminator|I64Attr|I32Prop|AnyRegion) '
```

Real output:

```text
def AnyRegion {	// Constraint RegionConstraint Region
def CostInterface {	// Interface Trait NativeTrait InterfaceTrait OpInterfaceTrait OpInterface
def Demo_Dialect {	// Dialect
def I32Prop {	// Property IntProp
def I64Attr {	// Constraint AttrConstraint Attr TypedAttrBase SignlessIntegerAttrBase TypedSignlessIntegerAttrBase
def PairType {	// Constraint TypeConstraint Type DialectType AttrOrTypeDef TypeDef
def SmallTensor {	// Constraint TypeConstraint Type
def StopOp {	// Op
def Terminator {	// Trait NativeTrait NativeOpTrait
```

- The comment lists the classes base first. `PairType` is a `Constraint`, a `TypeConstraint`, a `Type`, a `DialectType`, and finally a `TypeDef`: a type you define is also a constraint, so it can be written in an op's operand list like `F64Tensor`.
- `CostInterface` has both parents: `Interface`, then the trait chain `Trait` → `NativeTrait` → `InterfaceTrait` → `OpInterfaceTrait`, then `OpInterface`. That is why an interface can be listed among an op's traits.
- `Demo_Dialect`, `I32Prop` and `StopOp` show the standalone roots: `Dialect`, `Property`, `Op`.

#### 2. `Op`'s fields hold records of the other files' classes

The second relation is "has a", not "is a". The `Op` class derives from nothing, but each of its fields is typed with a class from another file. This is what makes `OpBase.td` the top of the hierarchy:

| `Op` field | Holds | Class defined in |
|---|---|---|
| `opDialect` | a `Dialect` | `DialectBase.td` |
| `arguments` | a `dag` starting with `ins`, of type constraints (operands), `Attr`s (attributes) and `Property`s | `Utils.td`, `CommonTypeConstraints.td`, `CommonAttrConstraints.td`, `Properties.td` |
| `results` | a `dag` starting with `outs`, of type constraints | `Utils.td`, `CommonTypeConstraints.td` |
| `regions` | a `dag` starting with `region`, of `Region`s | `OpBase.td` |
| `successors` | a `dag` starting with `successor`, of `Successor`s | `OpBase.td` |
| `traits` | a `list<Trait>`: traits and interfaces | `Traits.td`, `Interfaces.td`, `OpBase.td` |
| `builders` | a `list<OpBuilder>`, with `CArg` parameters | `OpBase.td`, `Utils.td` |

Every element of `arguments`, `results`, `regions` and `successors` is a `Constraint` underneath (tree 1), and every `Constraint` holds a `Pred`. So an op definition is a record that points at constraints, which point at predicates, which hold C++ strings. The generator walks that structure.

#### 3. Each generator backend reads the records of one root class

The files are input for `mlir-tblgen`, and each backend looks for records that derive from one class and follows their fields from there:

| Backend | Starts from records of | Produces | Shown in |
|---|---|---|---|
| `-gen-dialect-decls`, `-gen-dialect-defs` | `Dialect` | the dialect class | README section 3.2 |
| `-gen-op-decls`, `-gen-op-defs` | `Op` | one class per op, with its verifier | README section 4.4, G.6 |
| `-gen-typedef-decls`, `-gen-typedef-defs` | `TypeDef` | one class per type | G.6 |
| `-gen-attrdef-decls`, `-gen-attrdef-defs` | `AttrDef` | one class per attribute | |
| `-gen-op-interface-decls`, `-gen-op-interface-defs` | `OpInterface` | one class per interface | G.6 |
| `-gen-op-doc`, `-gen-dialect-doc` | `Op`, `Dialect` | Markdown documentation | README section 4.4, step 2 |

Constraints, predicates and traits have no backend of their own. They are only ever reached through an op, a type or an attribute that refers to them. That is why `SmallTensor` or `F64Tensor` generates no code until some op uses it.

---

## G.5 The files, one by one

In layer order (G.2). Each file starts with the same five points: its purpose, the files it needs, the files that need it, how you use it when writing a dialect, and when its internals are worth reading. Excerpts are abridged with `...`; the full text is in the copies.

### `Utils.td`: markers and helpers

- **Purpose:** a toolbox with no MLIR concept of its own. It holds the small pieces every other file wants and that belong to none of them.
- **Needs:** nothing. It is the bottom of the hierarchy.
- **Needed by:** `Constraints.td`, `DialectBase.td`, `Properties.td` and `OpBase.td` directly, and so every file.
- **How you use it:** constantly, without noticing: `ins` and `outs` in every `arguments`/`results` list, `CArg` in a builder with a default argument, `Deprecated<"...">` mixed into a record you want to phase out.
- **Read the internals when:** almost never. The string helpers (`snakeCaseToCamelCase`) are used by other `.td` files, not by dialect authors.

***appendix/OpBase/mlir/IR/Utils.td***
```tablegen
// Marker used to identify the argument list.
def ins;

// Marker used to identify the result list.
def outs;

// This class represents a typed argument with optional default value for C
// function signatures, e.g. builders or methods.
class CArg<string ty, string value = ""> {
  string type = ty;
  string defaultValue = value;
}
```

- **`ins` and `outs` are empty records.** They carry no data. They exist only to be the first element of a `dag` value, so the generator can tell an argument list `(ins ...)` from a result list `(outs ...)`. Every Toy op uses them: `let arguments = (ins F64ElementsAttr:$value);`.
- **`CArg`** is a C++ parameter with a default value. `FuncOp`'s builder uses it: `CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs`.
- The rest of the file is `Deprecated`/`CppDeprecated` (mark a record or its generated C++ as deprecated) and string helpers (`StrFunc`, `firstCharToUpper`, `snakeCaseToCamelCase`). `StrFunc` exists because TableGen has no functions: a class whose only field is a computed string stands in for one.

### `Constraints.td`: predicates and constraints

- **Purpose:** the foundation of verification. It defines what a checkable rule is (`Pred`) and how a rule gets a name and an error message (`Constraint`).
- **Needs:** `Utils.td`.
- **Needed by:** seven of the other nine files. It is the most-included file: types, attributes, regions, successors, traits and properties are all constraints or use predicates.
- **How you use it:** directly only when no ready-made constraint fits. Then you write a `CPred` with your own C++ condition and combine it with `And`/`Or`/`Neg` (G.6 does this for `SmallTensor`). Otherwise you use it through the names of the two `Common...Constraints.td` files.
- **Read the internals when:** you write your own constraint, or a verifier error message is not what you expected. The `$_self` placeholder rules are in this file's comments, and nowhere in the generated code.

This file answers "how does ODS express a rule that the verifier can check?" in two steps.

**Step 1, predicates.** A predicate is a condition. The leaf is `CPred`, a piece of C++ kept as a string:

***appendix/OpBase/mlir/IR/Constraints.td***
```tablegen
class Pred;
...
class CPred<code pred> : Pred {
  code predExpr = "(" # pred # ")";
}
...
// A predicate that holds if all of its children hold.  Always holds for zero
// children.
class And<list<Pred> children> : CombinedPred<PredCombinerAnd, children>;

// A predicate that holds if any of its children hold.  Never holds for zero
// children.
class Or<list<Pred> children> : CombinedPred<PredCombinerOr, children>;

// A predicate that holds if its child does not.
class Neg<Pred child> : CombinedPred<PredCombinerNot, [child]>;
```

- `CPred<"$_self.isF64()">` is a predicate. `$_self` is a placeholder for the thing being checked; the generator replaces it with real C++ (for an operand's type constraint, the operand's type).
- `And`, `Or` and `Neg` combine predicates into a tree. `SubstLeaves` and `Concat` (same file) rewrite the strings in a tree; G.7 shows them at work.

**Step 2, constraints.** A constraint is a predicate plus a one-line description, which becomes the error message:

***appendix/OpBase/mlir/IR/Constraints.td***
```tablegen
class Constraint<Pred pred, string desc = ""> {
  // The predicates that this constraint requires.
  Pred predicate = pred;
  // User-readable one line summary used in error reporting messages. If empty,
  // a generic message will be used.
  string summary = desc;
}
...
// Subclass for constraints on a type.
class TypeConstraint<Pred predicate, string summary = "",
                     string cppTypeParam = "::mlir::Type",
                     string cppFunctionNameParam = ""> :
    Constraint<predicate, summary> {
  // The name of the C++ Type class if known, or Type if not.
  string cppType = cppTypeParam;
  ...
}

// Subclass for constraints on an attribute.
class AttrConstraint<Pred predicate, string summary = ""> :
    Constraint<predicate, summary>;

// Subclass for constraints on a region.
class RegionConstraint<Pred predicate, string summary = ""> :
    Constraint<predicate, summary>;

// Subclass for constraints on a successor.
class SuccessorConstraint<Pred predicate, string summary = ""> :
    Constraint<predicate, summary>;
```

The four subclasses say what the constraint applies to: an operand or result **type**, an **attribute**, a **region**, or a **successor** block. Everything an op's `arguments`, `results` and `regions` lists can contain derives from one of them. Toy's `Ops.td` never writes `CPred` or `Constraint` itself; it uses ready-made constraints from the two `Common...Constraints.td` files.

### `DialectBase.td`: the `Dialect` class

- **Purpose:** one class, `Dialect`: the record that describes a dialect as a whole and switches its optional hooks on.
- **Needs:** `Utils.td`.
- **Needed by:** `CommonTypeConstraints.td`, `CommonAttrConstraints.td`, `AttrTypeBase.td` and `OpBase.td`. They need it because a type, an attribute and an op each name the dialect they belong to (`DialectType`, `DialectAttr`, `Op`'s first argument).
- **How you use it:** once per dialect: one `def X_Dialect : Dialect { ... }`, then pass it to every `Op`, `TypeDef` and `AttrDef`.
- **Read the internals when:** you add a dialect-level feature (constant materialization, your own types or attributes, canonicalization, dependent dialects). The field comments say which C++ hook each `bit` makes you implement. The file is 100 lines; reading it whole takes a few minutes.

The file is one class. Its fields are the dialect-level settings (abridged):

***appendix/OpBase/mlir/IR/DialectBase.td***
```tablegen
class Dialect {
  // The name of the dialect.
  string name = ?;

  // Short summary of the dialect.
  string summary = ?;

  // The description of the dialect.
  string description = ?;
  ...
  string cppNamespace = name;
  ...
  // If this dialect overrides the hook for materializing constants.
  bit hasConstantMaterializer = 0;
  ...
  // If this dialect should use default generated type parser boilerplate:
  // When set, ODS will generate declarations for the type parsing and printing
  // hooks in the dialect and default implementations that dispatch to each
  // individual type directly.
  bit useDefaultTypePrinterParser = 0;
  ...
  // Whether inherent Attributes defined in ODS will be stored as Properties.
  bit usePropertiesForAttributes = 1;
}
```

- `Toy_Dialect` sets two fields, `name` and `cppNamespace` (README section 3.2). `?` means "no value": `name` must be set, and `cppNamespace` defaults to the name.
- The `bit` fields each switch on one generated hook. They default to 0, which is why Chapter 2's generated `ToyDialect` class is so small. Later chapters turn some on.
- `usePropertiesForAttributes = 1` is why Toy's inherent attributes print inside `<{...}>` in the generic form (README appendix C.1).

### `Traits.td`: the trait base classes

- **Purpose:** the four ways a trait can be implemented, defined once for any entity. It contains no actual trait.
- **Needs:** `Constraints.td`, for `Pred` (a `PredTrait` holds a predicate).
- **Needed by:** `AttrTypeBase.td` (attribute and type traits), `Interfaces.td` (an interface is a trait) and `OpBase.td` (op traits).
- **How you use it:** by listing trait records in an op's second argument: `Toy_Op<"return", [Pure, HasParent<"FuncOp">, Terminator]>`. You use this file's classes directly only to declare a trait of your own: `def MyTrait : NativeOpTrait<"MyTrait">;` names a C++ class `OpTrait::MyTrait` that you then write by hand.
- **Read the internals when:** you need to know which C++ class a TableGen trait name stands for, or you add a trait. It is 68 lines.

***appendix/OpBase/mlir/IR/Traits.td***
```tablegen
// Trait represents a trait regarding an attribute, operation, or type.
class Trait;

// Define a Trait corresponding to a list of Traits, this allows for specifying
// a list of traits as trait. Avoids needing to do `[Traits, ...] # ListOfTraits
// # [Others, ...]` while still allowing providing convenient groupings.
class TraitList<list<Trait> props> : Trait {
  list<Trait> traits = props;
}
...
class NativeTrait<string name, string entityType,
                    code extraClassDeclaration = [{}],
                    code extraClassDefinition = [{}]> : Trait {
  string trait = name;
  string cppNamespace = "::mlir::" # entityType # "Trait";
  ...
}
...
class ParamNativeTrait<string prop, string params, string entityType>
    : NativeTrait<prop # "<" # params # ">::Impl", entityType>;
...
class GenInternalTrait<string prop, string entityType> : Trait {
  string trait = "::mlir::" # entityType # "Trait::" # prop;
}

// PredTrait is a trait implemented by way of a predicate on an entity.
class PredTrait<string descr, Pred pred> : Trait {
  string summary = descr;
  Pred predicate = pred;
}
```

There are four kinds of trait, and this file defines them for any entity (op, attribute or type):

| Class | What the trait is | Example in Toy |
|---|---|---|
| `NativeTrait` | a C++ trait class, named by a string | `Terminator` → `::mlir::OpTrait::IsTerminator` |
| `ParamNativeTrait` | a C++ trait class with template parameters | `HasParent<"FuncOp">` → `::mlir::OpTrait::HasParent<FuncOp>::Impl` |
| `GenInternalTrait` | no C++ class; it changes what the generator emits | none |
| `PredTrait` | a predicate checked by the verifier | none |

`TraitList` groups several traits under one name. `Pure` is one (G.8). This file has no `def`: the concrete op traits are in `OpBase.td`.

### `Properties.td`: data stored inside the op

- **Purpose:** describes data an op keeps as a plain C++ value in its own storage, without creating an `Attribute`.
- **Needs:** `Constraints.td` (a property can carry a predicate) and `Utils.td`.
- **Needed by:** `OpBase.td` only. It is a side branch: no type, attribute, trait or interface file depends on it.
- **How you use it:** by writing a property in `arguments` next to the operands and attributes: `(ins F64Tensor:$input, I32Prop:$count)`. The generator adds a `count` member to the op's `Properties` struct and a `getCount()` accessor (G.6).
- **Read the internals when:** you define a property kind with its own storage type. Most of the file is C++ snippets for parsing, printing and bytecode, which you can skip otherwise.

***appendix/OpBase/mlir/IR/Properties.td***
```tablegen
// Base class for defining properties.
class Property<string storageTypeParam = "", string desc = ""> {
  // User-readable one line summary used in error reporting messages. If empty,
  // a generic message will be used.
  string summary = desc;
  // The full description of this property.
  string description = "";
  code storageType = storageTypeParam;
  code interfaceType = storageTypeParam;
  ...
}
```

A property is a piece of data an op stores directly as a C++ value (`storageType`), instead of as an `Attribute` allocated in the context. The file defines the base class, ready-made kinds (`I32Prop`, `I64Prop`, `StringProp`, `BoolProp`, `UnitProp`) and wrappers (`DefaultValuedProp`, `ArrayProp`, `OptionalProp`). Most of its 790 lines are the C++ snippets for parsing, printing and bytecode.

Toy's `Ops.td` declares no property. Its attributes (`value`, `callee`, `sym_name`, ...) are still *stored* as properties, but that comes from the `Dialect` field `usePropertiesForAttributes`, not from this file's classes.

### `CommonTypeConstraints.td`: the `Type` class and ready-made type constraints

- **Purpose:** the catalog of type constraints for builtin types. It is the file you look things up in most often.
- **Needs:** `Constraints.td` (it is built from `TypeConstraint` and predicates) and `DialectBase.td` (for `DialectType`).
- **Needed by:** `CommonAttrConstraints.td` (an attribute has a value type) and `AttrTypeBase.td` (`TypeDef` derives from `DialectType`).
- **How you use it:** by picking a name for each operand and result: `F64Tensor:$input`, `(outs StaticShapeTensorOf<[F64]>)`. Wrap it in `Variadic<...>` or `Optional<...>` to change how many values it stands for.
- **Read the internals when:** you need a constraint and don't know its name. Search the file for the C++ type (`grep -n TensorOf`), then read the `def`s near it. Also when you write your own `Type<...>` and want a predicate to reuse (`IsTensorTypePred`, `HasStaticShapePred`).

The biggest file. It starts with reusable predicates (`IsTensorTypePred`, `HasStaticShapePred`, ...), then the `Type` class:

***appendix/OpBase/mlir/IR/CommonTypeConstraints.td***
```tablegen
// A type, carries type constraints.
class Type<Pred condition, string descr = "",
           string cppType = "::mlir::Type"> :
    TypeConstraint<condition, descr, cppType> {
  string description = "";
  string builderCall = "";
}
```

**A `Type` record in ODS is a constraint, not a type.** `F64` does not create a type; it is a `TypeConstraint` whose predicate asks "is this type `f64`?". The rest of the file is a catalog of such constraints, in this order: integers (`I1`, `I32`, `SI8`, `UI64`, ...), `Index`, floats (`F16`, `F32`, `F64`, ...), complex, function, vectors, tensors, memrefs, tuples, and finally common combinations of these.

Three classes turn one constraint into another, and Toy uses two of them:

***appendix/OpBase/mlir/IR/CommonTypeConstraints.td***
```tablegen
class Variadic<Type type> : TypeConstraint<type.predicate,
                                           "variadic of " # type.summary,
                                           type.cppType> {
  Type baseType = type;
  int minSize = 0;
}
...
class StaticShapeTensorOf<list<Type> allowedTypes>
  : RankedTensorOf<allowedTypes, [HasStaticShapePred],
                   "statically shaped tensor">;
```

- `Variadic<F64Tensor>` means "zero or more operands, each an `F64Tensor`" (`toy.generic_call`'s inputs, `toy.return`'s operand).
- `StaticShapeTensorOf<[F64]>` is `toy.reshape`'s result: a ranked tensor of `f64` whose shape is fully known.
- `Optional<...>` (same file) means "zero or one"; Toy doesn't use it.

G.7 follows the chain behind `F64Tensor`.

### `CommonAttrConstraints.td`: the `Attr` class and ready-made attribute constraints

- **Purpose:** the catalog of attribute constraints for builtin attributes. Unlike a type constraint, an attribute constraint also tells the generator how to write the getter.
- **Needs:** `Constraints.td`, `CommonTypeConstraints.td` (an `Attr` has a `valueType`, and `TypeAttrOf<...>` takes a type constraint) and `DialectBase.td`.
- **Needed by:** `AttrTypeBase.td` (`AttrDef` derives from `DialectAttr`).
- **How you use it:** by picking a name for each attribute in `arguments`: `F64ElementsAttr:$value`, `FlatSymbolRefAttr:$callee`. Wrap it in `OptionalAttr<...>` or `DefaultValuedAttr<..., "0">` to make it optional or give it a default.
- **Read the internals when:** a generated getter returns a type you did not expect. The `def`'s `returnType` and `convertFromStorage` are the answer: `I64Attr`'s getter returns `uint64_t`, `F64ElementsAttr`'s returns the attribute itself.

Same idea for attributes. The base class adds what the generator needs to produce accessors:

***appendix/OpBase/mlir/IR/CommonAttrConstraints.td***
```tablegen
// Base class for all attributes.
class Attr<Pred condition, string summary = ""> :
    AttrConstraint<condition, summary> {
  code storageType = ?; // The backing mlir::Attribute type
  code returnType = ?;  // The underlying C++ value type
  ...
  code convertFromStorage = "$_self.getValue()";
  ...
  string constBuilderCall = ?;
  ...
}
```

- `storageType` is the C++ attribute class the op stores; `returnType` is what the generated getter returns; `convertFromStorage` is the expression that turns one into the other.

The attribute behind `toy.constant`'s `value`:

***appendix/OpBase/mlir/IR/CommonAttrConstraints.td***
```tablegen
class FloatElementsAttr<int width> : ElementsAttrBase<
  CPred<"::llvm::isa<::mlir::DenseFPElementsAttr>($_self) &&"
      "::llvm::cast<::mlir::DenseElementsAttr>($_self).getType()."
      "getElementType().isF" # width # "()">,
  width # "-bit float elements attribute"> {
  let storageType = [{ ::mlir::DenseElementsAttr }];
  let returnType = [{ ::mlir::DenseElementsAttr }];
  ...
  let convertFromStorage = "$_self";
}

def F64ElementsAttr : FloatElementsAttr<64>;
```

- The predicate is the verifier check: the attribute is a dense floating-point elements attribute whose element type is `f64`.
- `storageType` and `returnType` are both `DenseElementsAttr`, and `convertFromStorage = "$_self"` returns the attribute unchanged. That is why `ConstantOp::getValue()` returns a `DenseElementsAttr` (README section 4.4, step 1).

The other attribute constraints Toy uses are in the same file: `FlatSymbolRefAttr` (`toy.generic_call`'s `callee`; its `returnType` is `::llvm::StringRef`, so `getCallee()` returns the name), and `SymbolNameAttr`, `TypeAttrOf<FunctionType>` and `OptionalAttr<DictArrayAttr>` (the four attributes of `toy.func`). `OptionalAttr` and `DefaultValuedAttr` wrap another attribute constraint, the way `Variadic` wraps a type constraint.

### `AttrTypeBase.td`: defining your own attributes and types

- **Purpose:** the classes for adding an attribute or a type to your dialect. The two catalogs above describe what exists; this file creates something new.
- **Needs:** the two catalogs (a new type is a `DialectType`, a new attribute a `DialectAttr`), `Constraints.td`, `DialectBase.td` and `Traits.td` (types and attributes can have traits).
- **Needed by:** `Interfaces.td` and `OpBase.td`.
- **How you use it:** `def MyType : TypeDef<My_Dialect, "My"> { let mnemonic = ...; let parameters = (ins ...); }`, then run the `-gen-typedef-*` backends and include the output, as for ops (G.6).
- **Read the internals when:** your dialect needs its own type or attribute. The parameter classes (`ArrayRefParameter`, `OptionalParameter`, ...) decide how each parameter is stored, compared and printed, and their comments are the reference.

The two `Common...Constraints.td` files describe constraints on attributes and types that already exist in C++. This file is for *creating* new ones:

***appendix/OpBase/mlir/IR/AttrTypeBase.td***
```tablegen
// Define a new attribute, named `name`, belonging to `dialect` that inherits
// from the given C++ base class.
class AttrDef<Dialect dialect, string name, list<Trait> traits = [],
              string baseCppClass = "::mlir::Attribute">
    : DialectAttr<dialect, CPred<"">, /*descr*/"">,
      AttrOrTypeDef<"Attr", name, traits, baseCppClass> {
  // The name of the C++ Attribute class.
  string cppClassName = name # "Attr";
  ...
}
...
// Define a new type, named `name`, belonging to `dialect` that inherits from
// the given C++ base class.
class TypeDef<Dialect dialect, string name, list<Trait> traits = [],
              string baseCppClass = "::mlir::Type">
    : DialectType<dialect, CPred<"">, /*descr*/"", name # "Type">,
      AttrOrTypeDef<"Type", name, traits, baseCppClass> {
  // The name of the C++ Type class.
  string cppClassName = name # "Type";
  ...
}
```

- `AttrDef` and `TypeDef` are to attributes and types what `Op` is to operations: a record derived from them makes `mlir-tblgen` generate a C++ class (with `-gen-attrdef-decls`/`-gen-typedef-decls` and the matching `-defs`).
- The file also has their parameter classes (`AttrParameter`, `TypeParameter`, `ArrayRefParameter`, ...), builders (`AttrBuilder`, `TypeBuilder`), and the attribute and type versions of the trait classes (`NativeAttrTrait`, `NativeTypeTrait`).

Toy's `Ops.td` uses none of it. Chapter 2 only uses builtin types and attributes, and Chapter 7's struct type is written in C++ by hand.

### `Interfaces.td`: declaring interfaces

- **Purpose:** the classes for declaring an interface: a named set of methods that different ops, types or attributes implement.
- **Needs:** `Traits.td` (an interface is attached as a trait), `AttrTypeBase.td` (so that an interface can also act as a type or attribute constraint) and `Constraints.td`.
- **Needed by:** `OpBase.td`.
- **How you use it:** two ways. To *implement* an existing interface, list it in the op's traits, as `toy.func` does with `FunctionOpInterface`. To *declare* a new one, write `def X : OpInterface<"X"> { let methods = [InterfaceMethod<...>]; }` in its own `.td` file and run `-gen-op-interface-decls` (G.6).
- **Read the internals when:** you declare an interface. `InterfaceMethod` has six arguments (description, return type, name, parameters, body, default implementation), and the difference between the last two is explained only in this file's comments.

An interface is a set of methods that several ops (or types, or attributes) implement, so a pass can call them without knowing the concrete op. This file has the classes to declare one:

***appendix/OpBase/mlir/IR/Interfaces.td***
```tablegen
class InterfaceTrait<string name> : NativeTrait<"", ""> {
  let trait = name # "::Trait";
  let cppNamespace = "";
  ...
}
...
// OpInterface represents an interface registered to an operation.
class OpInterface<string name, list<Interface> baseInterfaces = []>
  : Interface<name, baseInterfaces>, OpInterfaceTrait<name>;
```

- **An interface is also a trait.** `OpInterface` derives from `Interface` (the description: name, namespace, methods) *and* from `OpInterfaceTrait` → `InterfaceTrait` → `NativeTrait` → `Trait`. That is why an interface name can be written in an op's trait list, as in `Toy_Op<"func", [FunctionOpInterface, IsolatedFromAbove]>`.
- `InterfaceMethod` and `StaticInterfaceMethod` describe one method each. `DeclareOpInterfaceMethods<SomeInterface>` is written in a trait list in place of the interface's own name; it attaches the same interface and also asks the generator to declare the interface's methods in the op class.
- `Interfaces.td` includes `AttrTypeBase.td` because `TypeInterface` derives from `Type`, and `AttrInterface` from `Attr`: an interface can also be used as a constraint ("any type that implements this interface").

Toy's Chapter 2 declares no interface of its own; it uses one that MLIR declares with these classes (G.8). Chapter 4 writes one (`ShapeInferenceOpInterface`).

### `OpBase.td`: the `Op` class and what an op is made of

- **Purpose:** the top file. It defines `Op`, the pieces only ops have (builders, regions, successors, op traits), and constraints that relate several operands to each other.
- **Needs:** all nine other files, seven by direct `include`.
- **Needed by:** none of the ten. It is included by every dialect's `.td` file, and by MLIR's own interface files (G.8).
- **How you use it:** one record per operation: `def MyOp : Op<My_Dialect, "my", [traits]> { let arguments = ...; let results = ...; }`. This is the file you write against all the time.
- **Read the internals when:** always, for the `Op` class. Its fields are the complete list of what an op definition can say, and the table in part 4 below is that list. The rest can be read on demand.

The top file, in six parts.

**1. Op traits (lines 24–179).** The four trait kinds of `Traits.td`, specialized for ops, then the concrete traits:

***appendix/OpBase/mlir/IR/OpBase.td***
```tablegen
class NativeOpTrait<string name, list<Trait> traits = [],
                    code extraOpDeclaration = [{}],
                    code extraOpDefinition = [{}]>
    : NativeTrait<name, "Op", extraOpDeclaration, extraOpDefinition> {
  // Specify the list of traits that need to be verified before the verification
  // of this NativeOpTrait.
  list<Trait> dependentTraits = traits;
}
...
def IsolatedFromAbove : NativeOpTrait<"IsIsolatedFromAbove">;
...
def Terminator : NativeOpTrait<"IsTerminator">;
...
// Op's parent operation is the provided one.
class HasParent<string op>
    : ParamNativeOpTrait<"HasParent", op>, StructuralOpTrait;
```

- `NativeTrait<name, "Op", ...>` sets `cppNamespace` to `"::mlir::" # "Op" # "Trait"`, so `Terminator` names the C++ class `::mlir::OpTrait::IsTerminator`. The TableGen name and the C++ name differ; the string is the C++ one.
- `HasParent` is a class, not a `def`, because it takes a parameter: `HasParent<"FuncOp">` in `toy.return`.
- These three are the op traits Toy's Chapter 2 takes from this file. The same block holds about thirty more (`Commutative`, shape traits such as `SameOperandsAndResultShape`, `SingleBlock`, `NoTerminator`, `AttrSizedOperandSegments`, ...).

**2. Successors and regions (lines 181–225), and their markers (227–235).**

***appendix/OpBase/mlir/IR/OpBase.td***
```tablegen
class Region<Pred condition, string descr = ""> :
    RegionConstraint<condition, descr>;

// Any region.
def AnyRegion : Region<CPred<"true">, "any region">;
...
// Marker used to identify the region list.
def region;

// Marker used to identify the successor list.
def successor;
```

`toy.func` uses both: `let regions = (region AnyRegion:$body);`. `region` plays the role of `ins` for the region list, and `AnyRegion` is a constraint whose predicate is always true.

**3. `OpBuilder` (lines 237–288).**

***appendix/OpBase/mlir/IR/OpBase.td***
```tablegen
class OpBuilder<dag p, code b = ""> {
  dag dagParams = p;
  code body = b;
}
```

One record per custom `build` method: a parameter list and an optional body. An empty body means "declare it only; the C++ is written by hand" (README section 4.4, step 4).

**4. The `Op` class (lines 322–443).** Every op record derives from it; Toy's go through `Toy_Op`. Its header and first fields:

***appendix/OpBase/mlir/IR/OpBase.td***
```tablegen
class Op<Dialect dialect, string mnemonic, list<Trait> props = []> {
  // The dialect of the op.
  Dialect opDialect = dialect;

  // The mnemonic of the op.
  string opName = mnemonic;

  // The C++ namespace to use for this op.
  string cppNamespace = dialect.cppNamespace;
  ...
}
```

All of its fields, with their defaults:

| Field | Default | Meaning | Set in Toy's `Ops.td` |
|---|---|---|---|
| `opDialect` | the `dialect` argument | the dialect the op belongs to | by `Toy_Op` (`Toy_Dialect`) |
| `opName` | the `mnemonic` argument | the name after `toy.` | by each op (`"constant"`) |
| `cppNamespace` | `dialect.cppNamespace` | namespace of the generated class | no |
| `summary` | `""` | one-line documentation | all 9 ops |
| `description` | `""` | long documentation | 8 ops |
| `opDocGroup` | `?` | documentation grouping | no |
| `arguments` | `(ins)` | operands and attributes | all 9 ops |
| `results` | `(outs)` | results | 6 ops |
| `regions` | `(region)` | regions | `toy.func` |
| `successors` | `(successor)` | successor blocks | no |
| `builders` | `?` | custom `build` methods | 7 ops |
| `skipDefaultBuilders` | `0` | don't generate the default builders | `toy.func` |
| `assemblyFormat` | `?` | declarative custom syntax | 5 ops |
| `hasCustomAssemblyFormat` | `0` | custom syntax written in C++ (`parse`/`print`) | 4 ops |
| `hasVerifier` | `0` | declare `verify()` for extra checks | 3 ops |
| `hasRegionVerifier` | `0` | declare `verifyRegions()` | no |
| `hasCanonicalizer` | `0` | has canonicalization patterns | no (Chapter 3) |
| `hasCanonicalizeMethod` | `0` | has a static `canonicalize` method | no |
| `hasFolder` | `0` | has a `fold` method | no |
| `useCustomPropertiesEncoding` | `0` | custom bytecode for properties | no |
| `traits` | the `props` argument | the op's traits | 3 ops, through `Toy_Op`'s second argument |
| `extraClassDeclaration` | `?` | C++ pasted into the generated class | `toy.func`, `toy.return` |
| `extraClassDefinition` | `?` | C++ pasted into the generated source | no |

The defaults explain the empty shell of README section 4.4, step 0: `def ConstantOp : Toy_Op<"constant"> {}` sets nothing, and the record that `--print-records` shows is this table's Default column.

**5. `Arguments` and `Results` (lines 445–453)** are two small classes for giving an op its `arguments`/`results` through inheritance instead of `let`. Toy doesn't use them.

**6. Multi-entity constraints (lines 455–673).** Everything so far constrains one operand, attribute or result. These constrain several at once, as traits:

***appendix/OpBase/mlir/IR/OpBase.td***
```tablegen
class AllTypesMatch<list<string> names> :
    AllMatchSameOperatorTrait<names, "$_self.getType()", "type">;
```

`AllTypesMatch<["lhs", "rhs"]>` says the named operands must have the same type, and `TypesMatchWith` relates two of them through an expression. They are `PredOpTrait`s built from the predicate combiners of `Constraints.td`. Toy's Chapter 2 doesn't use them; its cross-operand checks (e.g. `toy.return`'s operand against the function's result type) are written in C++ in `verify()`.

---

## G.6 All ten files in one small dialect

Toy's Chapter 2 uses six of the ten files by name (G.3). To see all ten at work, `appendix/OpBase/demo/Demo.td` defines a tiny `demo` dialect with one op, one type and one interface. Each record's comment names the file its pieces come from. The build never reads this file; only the commands below do.

***appendix/OpBase/demo/Demo.td***
```tablegen
// A small dialect that uses something from each of the ten files under
// OpBase.td. It is only read by the commands in appendix/OpBase.md; the build
// never reads it.

include "mlir/IR/OpBase.td"

// DialectBase.td: the Dialect class.
def Demo_Dialect : Dialect {
  let name = "demo";
  let cppNamespace = "::demo";
}

// Constraints.td (And, CPred) + CommonTypeConstraints.td (Type,
// IsTensorTypePred): a type constraint of our own.
def SmallTensor : Type<
    And<[IsTensorTypePred,
         CPred<"::llvm::cast<::mlir::ShapedType>($_self).getNumElements() <= 8">]>,
    "tensor of at most 8 elements">;

// AttrTypeBase.td: a type of our own, `!demo.pair<first, second>`.
def PairType : TypeDef<Demo_Dialect, "Pair"> {
  let mnemonic = "pair";
  let parameters = (ins "::mlir::Type":$first, "::mlir::Type":$second);
  let assemblyFormat = "`<` $first `,` $second `>`";
}

// Interfaces.td: an op interface with one method.
def CostInterface : OpInterface<"CostInterface"> {
  let cppNamespace = "::demo";
  let methods = [
    InterfaceMethod<"Returns the cost of this op.", "unsigned", "getCost">
  ];
}

// OpBase.td: the Op class, OpBuilder and the Terminator trait.
def StopOp : Op<Demo_Dialect, "stop",
    [Terminator, DeclareOpInterfaceMethods<CostInterface>]> {
  // Utils.td: ins. CommonAttrConstraints.td: I64Attr. Properties.td: I32Prop.
  let arguments = (ins SmallTensor:$input, I64Attr:$code, I32Prop:$count);

  // Utils.td: CArg.
  let builders = [
    OpBuilder<(ins "::mlir::Value":$input, CArg<"int32_t", "1">:$count)>
  ];
}
```

The runs below feed this one file to four backends. Each shows what one of the ten files contributes to the generated C++.

#### `Constraints.td` and `CommonTypeConstraints.td`: a constraint becomes a verifier check

```bash
cd /Users/roy/study/mlir/toy/Ch2
mlir-tblgen appendix/OpBase/demo/Demo.td -I appendix/OpBase -gen-op-defs \
  | sed -n '/^static ::llvm::LogicalResult __mlir_ods_local_type_constraint/,/^}/p'
```

Real output:

```cpp
static ::llvm::LogicalResult __mlir_ods_local_type_constraint_Demo1(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((::llvm::isa<::mlir::TensorType>(type))) && ((::llvm::cast<::mlir::ShapedType>(type).getNumElements() <= 8)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be tensor of at most 8 elements, but got " << type;
  }
  return ::mlir::success();
}
```

- The `if` condition is `SmallTensor`'s predicate tree turned into C++. `And<[A, B]>` became `(A) && (B)`, and `$_self` in both leaves became `type`, the parameter that holds the operand's type.
- The first leaf comes from `IsTensorTypePred` in `CommonTypeConstraints.td`; the second is the `CPred` written in `Demo.td`.
- The error text is the constraint's `summary`, `"tensor of at most 8 elements"`, placed after `must be`.

#### `Traits.td`, `Interfaces.td` and `OpBase.td`: traits become base classes

```bash
cd /Users/roy/study/mlir/toy/Ch2
mlir-tblgen appendix/OpBase/demo/Demo.td -I appendix/OpBase -gen-op-decls | grep '^class StopOp : ' | tr ',' '\n'
```

Real output:

```text
class StopOp : public ::mlir::Op<StopOp
 ::mlir::OpTrait::ZeroRegions
 ::mlir::OpTrait::ZeroResults
 ::mlir::OpTrait::ZeroSuccessors
 ::mlir::OpTrait::OneOperand
 ::mlir::OpTrait::OpInvariants
 ::mlir::BytecodeOpInterface::Trait
 ::mlir::OpTrait::IsTerminator
 ::demo::CostInterface::Trait> {
```

- The last two lines are the op's trait list. `Terminator` is a `NativeOpTrait<"IsTerminator">`, so it became `::mlir::OpTrait::IsTerminator`. `CostInterface` became `::demo::CostInterface::Trait`, from `InterfaceTrait`'s `name # "::Trait"`.
- The lines before them were added by the generator from the op's shape: no regions, no results, no successors, one operand.

#### `Utils.td`, `CommonAttrConstraints.td`, `Properties.td`: what the argument list generates

```bash
cd /Users/roy/study/mlir/toy/Ch2
mlir-tblgen appendix/OpBase/demo/Demo.td -I appendix/OpBase -gen-op-decls \
  | sed -n '/^class StopOp : /,/^};/p' \
  | grep -E 'int32_t count = 1|unsigned getCost|int32_t getCount|uint64_t getCode'
```

Real output:

```text
  int32_t getCount() {
  uint64_t getCode();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input, int32_t count = 1);
  unsigned getCost();
```

The `sed` keeps the `StopOp` class only (the generated adaptor class repeats the getters).

- `int32_t getCount()` comes from `I32Prop:$count`: the property's `interfaceType` is the getter's return type.
- `uint64_t getCode()` comes from `I64Attr:$code`: `I64Attr`'s `returnType` is `uint64_t`, not `IntegerAttr`.
- The `build` line is the `OpBuilder` record. `CArg<"int32_t", "1">:$count` became the parameter `int32_t count = 1`.
- `unsigned getCost()` is declared in the op class because of `DeclareOpInterfaceMethods<CostInterface>`; its body is left to hand-written C++.

Where the property and the attribute are stored:

```bash
cd /Users/roy/study/mlir/toy/Ch2
mlir-tblgen appendix/OpBase/demo/Demo.td -I appendix/OpBase -gen-op-decls \
  | sed -n '/^  struct Properties {/,/^    void setCount/p'
```

Real output:

```cpp
  struct Properties {
    using codeTy = ::mlir::IntegerAttr;
    codeTy code;

    auto getCode() {
      auto &propStorage = this->code;
      return ::llvm::cast<::mlir::IntegerAttr>(propStorage);
    }
    void setCode(const ::mlir::IntegerAttr &propValue) {
      this->code = propValue;
    }
    using countTy = int32_t;
    countTy count;
    int32_t getCount() const {
      auto &propStorage = this->count;
      return propStorage;
    }
    void setCount(int32_t propValue) {
```

Both `code` and `count` are members of the op's `Properties` struct. `count` is there because it is a property. `code` is an attribute, and it is there because `Dialect`'s `usePropertiesForAttributes` defaults to 1 (`DialectBase.td`). The difference is the member's type: the attribute is stored as an `IntegerAttr`, the property as a bare `int32_t`.

#### `AttrTypeBase.td`: a `TypeDef` becomes a type class

```bash
cd /Users/roy/study/mlir/toy/Ch2
mlir-tblgen appendix/OpBase/demo/Demo.td -I appendix/OpBase -gen-typedef-decls | sed -n '/^class PairType : /,/^};/p'
```

Real output:

```cpp
class PairType : public ::mlir::Type::TypeBase<PairType, ::mlir::Type, detail::PairTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "demo.pair";
  static constexpr ::llvm::StringLiteral dialectName = "demo";
  static PairType get(::mlir::MLIRContext *context, ::mlir::Type first, ::mlir::Type second);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"pair"};
  }

  static ::mlir::Type parse(::mlir::AsmParser &odsParser);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::mlir::Type getFirst() const;
  ::mlir::Type getSecond() const;
};
```

- `get(context, first, second)` and the two getters come from the `parameters` list.
- `parse` and `print` are generated from `assemblyFormat`, like an op's.
- This backend does not need `OpBase.td`: a file that only defines types can include `AttrTypeBase.td` alone.

#### `Interfaces.td`: an `OpInterface` becomes an interface class

```bash
cd /Users/roy/study/mlir/toy/Ch2
mlir-tblgen appendix/OpBase/demo/Demo.td -I appendix/OpBase -gen-op-interface-decls | sed -n '/^class CostInterface : /,/^};/p'
```

Real output:

```cpp
class CostInterface : public ::mlir::OpInterface<CostInterface, detail::CostInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<CostInterface, detail::CostInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::CostInterfaceTrait<ConcreteOp> {};
  /// Returns the cost of this op.
  unsigned getCost();
};
```

- `getCost()` is the `InterfaceMethod` record; its description became the comment.
- The nested `Trait` struct is what the op class above inherits as `::demo::CostInterface::Trait`.
- A pass that holds a `CostInterface` can call `getCost()` on any op that implements it, without knowing which op it is.

`DialectBase.td` is the tenth file: `Demo_Dialect` is passed to `PairType` and `StopOp`, and gives both their `demo.` prefix and `::demo` namespace.

---

## G.7 One name, followed through the files: `F64Tensor`

`toy.print` declares its operand as `F64Tensor:$input`. That one word is defined through three of the ten files. Reading up from the name, in `CommonTypeConstraints.td`:

***appendix/OpBase/mlir/IR/CommonTypeConstraints.td***
```tablegen
class F<int width>
    : Type<CPred<"$_self.isF" # width # "()">,
           width # "-bit float", "::mlir::FloatType">,
      BuildableType<"$_builder.getF" # width # "Type()"> {
  int bitwidth = width;
}
...
def F64 : F<64>;
...
class ShapedContainerType<list<Type> allowedTypes,
                          Pred containerPred, string descr,
                          string cppType = "::mlir::Type"> :
    Type<And<[containerPred,
              Concat<"[](::mlir::Type elementType) { return ",
                SubstLeaves<"$_self", "elementType",
                AnyTypeOf<allowedTypes>.predicate>,
                "; }(::llvm::cast<::mlir::ShapedType>($_self).getElementType())">]>,
         descr # " of " # AnyTypeOf<allowedTypes>.summary # " values", cppType>;
...
class TensorOf<
    list<Type> allowedTypes,
    list<Pred> preds = [],
    string summary = "tensor">
  : ShapedContainerType<allowedTypes,
      And<!listconcat([IsTensorTypePred], preds)>,
      summary, "::mlir::TensorType">;
...
def F64Tensor  : TensorOf<[F64]>;
```

The chain, with the file each link is defined in:

```text
F64Tensor                         def    CommonTypeConstraints.td
└── TensorOf<[F64]>               class  CommonTypeConstraints.td
    └── ShapedContainerType       class  CommonTypeConstraints.td
        └── Type                  class  CommonTypeConstraints.td
            └── TypeConstraint    class  Constraints.td
                └── Constraint    class  Constraints.td
                    └── Pred: And / Concat / SubstLeaves / CPred   Constraints.td
```

(`Utils.td` is the third file: `Constraints.td` includes it.)

`--print-records` shows the result after TableGen has resolved the whole chain:

```bash
cd /Users/roy/study/mlir/toy/Ch2
mlir-tblgen --print-records appendix/OpBase/mlir/IR/OpBase.td -I appendix/OpBase \
  | awk '/^def (F64Tensor|F64|anonymous_119|anonymous_40) /,/^}/'
```

Real output:

```text
def F64 {	// Constraint TypeConstraint Type BuildableType F
  Pred predicate = anonymous_40;
  string summary = "64-bit float";
  string cppType = "::mlir::FloatType";
  string cppFunctionName = "";
  string description = "";
  string builderCall = "$_builder.getF64Type()";
  int bitwidth = 64;
}
def F64Tensor {	// Constraint TypeConstraint Type ShapedContainerType TensorOf
  Pred predicate = anonymous_119;
  string summary = "tensor of 64-bit float values";
  string cppType = "::mlir::TensorType";
  string cppFunctionName = "";
  string description = "";
  string builderCall = "";
}
def anonymous_119 {	// Pred CombinedPred And
  PredCombinerKind kind = PredCombinerAnd;
  list<Pred> children = [anonymous_68, anonymous_118];
}
def anonymous_40 {	// Pred CPred
  string predExpr = "($_self.isF64())";
}
```

- **The comment after each `def` is its class chain**, base class first: `F64Tensor` is a `Constraint`, a `TypeConstraint`, a `Type`, a `ShapedContainerType` and a `TensorOf`. It is the tree above, read from the bottom.
- **`summary = "tensor of 64-bit float values"`** was assembled from three places: `"tensor"` (`TensorOf`'s default), `" of "` and `" values"` (`ShapedContainerType`), and `"64-bit float"` (`F<64>`). This string is the verifier's error text: the generated check for `toy.print`'s operand reports `must be tensor of 64-bit float values` (README section 4.4, step 1).
- **`cppType = "::mlir::TensorType"`** is the C++ type the generated operand accessor can assume.
- **`predicate = anonymous_119`** points at one of the 344 anonymous records (G.3). `F64Tensor`'s predicate is an `And` of two children: "is a tensor type" and "its element type is allowed". `F64`'s predicate, `anonymous_40`, is a leaf: the `CPred` holding `$_self.isF64()`.

A dialect author writes one word, `F64Tensor`, and gets a verifier check, its error message and a C++ type. **The files under `OpBase.td` are a library of pre-built checks: an op definition mostly picks names from it.**

---

## G.8 What `OpBase.td` does not bring

Toy's `Ops.td` has three more `include` lines, because three names it uses are defined outside the ten files:

***include/toy/Ops.td***
```tablegen
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
```

| Name in `Ops.td` | Defined in | Built from |
|---|---|---|
| `FunctionOpInterface` (`toy.func`) | `mlir/Interfaces/FunctionInterfaces.td` | `OpInterface` (`Interfaces.td`) |
| `Pure` (`toy.constant`, `toy.return`) | `mlir/Interfaces/SideEffectInterfaces.td` | `TraitList` (`Traits.td`) |
| symbol support behind `FunctionOpInterface` | `mlir/IR/SymbolInterfaces.td` | `OpInterface` (`Interfaces.td`) |

The ten files hold the classes; files like these three use the classes to declare specific interfaces. An ODS file includes `OpBase.td` for the vocabulary and then one file per interface it implements.

---

## G.9 How much of this you need to know

Not the internals of all ten files, but the map of them. Nobody writing a dialect needs to be able to reproduce these files, and nobody can get far without knowing what is in them and where. They are best treated like the headers of a C++ library: you learn the few central classes well and look everything else up.

What that means in practice, in three levels:

| Level | What | Why |
|---|---|---|
| **Know well** | The `Op` class's fields (table in G.5). | It is the complete list of what an op definition can say. Every `let` in an `Ops.td` sets one of them. |
| | That every operand, result and attribute name is a *constraint*, and its `summary` is the verifier's error text. | It explains what the generated verifier checks and what its messages mean (G.6, G.7). |
| | The four trait kinds, and that a trait's TableGen name is not its C++ name (`Terminator` vs `IsTerminator`). | Needed to find the C++ behind a trait and to read a generated class's base list. |
| | Which file to search for what (table in G.3). | Finding `F64Tensor` or `OptionalAttr` is a `grep` in the right file. |
| **Read when you need it** | `Pred` combinators and `$_self` (`Constraints.td`). | When you write your own constraint. |
| | `TypeDef`, `AttrDef` and their parameter classes (`AttrTypeBase.td`). | When your dialect gets its own type or attribute. |
| | `OpInterface`, `InterfaceMethod` (`Interfaces.td`). | When you declare an interface. |
| | The `Dialect` class's `bit` fields (`DialectBase.td`). | When you add a dialect-level hook. |
| | Multi-entity constraints such as `AllTypesMatch` (`OpBase.td`). | When a rule relates several operands. |
| **Skip** | The C++ snippet strings in `Properties.td`, the string helpers in `Utils.td`, the long tails of the two catalogs. | They matter only if you work on `mlir-tblgen` itself or define a new kind of property. |

Three reasons the internals are still worth opening from time to time:

1. **The comments in these files are the reference.** The placeholders (`$_self`, `$_builder`, `$_op`), the builder rules above `OpBuilder`, and each `Op` field's meaning are documented in the `.td` files, right next to the definitions.
2. **A surprising error or accessor has its answer here.** When a verifier message reads oddly or a getter returns an unexpected type, the constraint's `def` shows the `summary` or `returnType` that produced it.
3. **Your own constraints should look like the existing ones.** The fastest way to write `SmallTensor` is to copy the shape of `TensorOf`.

The habit that replaces memorizing: find the name with `grep`, read the class comment, then check what it produces with `--print-records` and the `-gen-*` backends, as this appendix and README section 4.4 do. **What a compiler engineer needs from these files is the model (constraints, traits, `Op` fields) and the ability to look up the rest.**

---

Back to the [chapter README](../README.md).
