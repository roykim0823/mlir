# Customizing Assembly Behavior

> **Section:** Defining Dialects · document 5 of 7  
> **Upstream:** [https://mlir.llvm.org/docs/DefiningDialects/Assembly/](https://mlir.llvm.org/docs/DefiningDialects/Assembly/) · source [`mlir/docs/DefiningDialects/Assembly.md`](https://github.com/llvm/llvm-project/blob/main/mlir/docs/DefiningDialects/Assembly.md)  
> **License:** upstream text is Apache-2.0 WITH LLVM-exception.

## Orientation

The difference between an operation that prints as

```mlir
%0 = "my.conv"(%in, %w) {strides = array<i64: 1, 1>} : (tensor<...>, tensor<...>) -> tensor<...>
```

and one that prints as

```mlir
%0 = my.conv %in, %w {strides = [1, 1]} : tensor<...>
```

is entirely this page. Readable syntax is not cosmetic — it is what makes `FileCheck` tests
writable, IR dumps diffable, and bug reports legible.

**Read first**

- [Operation Definition Specification (ODS)](02-operation-definition-specification.md)
- [Lexical Tokens](../01-core-ir/02-lexical-tokens.md)

**What you should be able to do after this page**

- Write an `assemblyFormat` covering optional and variadic components.
- Use `custom<>` directives for the parts the declarative form cannot express.
- Add type and attribute aliases so large modules stay readable.

---

## Upstream documentation

## Generating Aliases

`AsmPrinter` can generate aliases for frequently used types and attributes when not printing them in generic form. For example, `!my_dialect.type<a=3,b=4,c=5,d=tuple,e=another_type>` and `#my_dialect.attr<a=3>` can be aliased to `!my_dialect_type` and `#my_dialect_attr`.

There are mainly two ways to hook into the `AsmPrinter`. One is the attribute/type interface and the other is the dialect interface.

The attribute/type interface is the first hook to check. If no such hook is found, or the hook returns `OverridableAlias` (see definition below), then dialect interfaces are involved.

The dialect interface for one specific dialect could generate alias for all types/attributes, even when it does not "own" them. The `AsmPrinter` checks all dialect interfaces based on their order of registration. For example, the default alias `map` for `builtin` attribute `AffineMapAttr` could be overriden by the dialect interface for `my_dialect` as custom dialect is often registered after the `builtin` dialect.

```cpp
/// Holds the result of `OpAsm{Dialect,Attr,Type}Interface::getAlias` hook call.
enum class OpAsmAliasResult {
  /// The object (type or attribute) is not supported by the hook
  /// and an alias was not provided.
  NoAlias,
  /// An alias was provided, but it might be overriden by other hook.
  OverridableAlias,
  /// An alias was provided and it should be used
  /// (no other hooks will be checked).
  FinalAlias
};
```

If multiple types/attributes have the same alias from `getAlias` hooks, a number is appended to the alias to avoid conflicts.

### `OpAsmDialectInterface`

```cpp
#include "mlir/IR/OpImplementation.h"

struct MyDialectOpAsmDialectInterface : public OpAsmDialectInterface {
 public:
  using OpAsmDialectInterface::OpAsmDialectInterface;

  AliasResult getAlias(Type type, raw_ostream& os) const override {
    if (mlir::isa<MyType>(type)) {
      os << "my_dialect_type";
      return AliasResult::FinalAlias;
    }
    return AliasResult::NoAlias;
  }

  AliasResult getAlias(Attribute attr, raw_ostream& os) const override {
    if (mlir::isa<MyAttribute>(attr)) {
      os << "my_dialect_attr";
      return AliasResult::FinalAlias;
    }
    return AliasResult::NoAlias;
  }
};

void MyDialect::initialize() {
  // register the interface to the dialect
  addInterface<MyDialectOpAsmDialectInterface>();
}
```

### `OpAsmAttrInterface` and `OpAsmTypeInterface`

The easiest way to use these interfaces is toggling `genMnemonicAlias` in the tablegen file of the attribute/alias. It directly uses the mnemonic as alias. See [Defining Dialect Attributes and Types](https://github.com/llvm/llvm-project/blob/main//docs/DefiningDialects/AttributesAndTypes) for details.

If a more custom behavior is wanted, the following modification to the attribute/type should be made

1. Add `OpAsmAttrInterface` or `OpAsmTypeInterface` into its trait list.
2. Implement the `getAlias` method, either in tablegen or its cpp file.

```tablegen
include "mlir/IR/OpAsmInterface.td"

// Add OpAsmAttrInterface trait
def MyAttr : MyDialect_Attr<"MyAttr",
         [ OpAsmAttrInterface ] > {

  // This method could be put in the cpp file.
  let extraClassDeclaration = [{
    ::mlir::OpAsmAliasResult getAlias(::llvm::raw_ostream &os) const {
      os << "alias_name";
      return ::mlir::OpAsmAliasResult::OverridableAlias;
    }
  }];
}
```

## Suggesting SSA/Block Names

An `Operation` can suggest the SSA name prefix using `OpAsmOpInterface`.

For example, `arith.constant` will suggest a name like `%c42_i32` for its result:

```tablegen
include "mlir/IR/OpAsmInterface.td"

def Arith_ConstantOp : Op<Arith_Dialect, "constant",
    [ConstantLike, Pure,
     DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
...
}
```

And the corresponding method:

```cpp
// from https://github.com/llvm/llvm-project/blob/5ce271ef74dd3325993c827f496e460ced41af11/mlir/lib/Dialect/Arith/IR/ArithOps.cpp#L184
void arith::ConstantOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  auto type = getType();
  if (auto intCst = llvm::dyn_cast<IntegerAttr>(getValue())) {
    auto intType = llvm::dyn_cast<IntegerType>(type);

    // Sugar i1 constants with 'true' and 'false'.
    if (intType && intType.getWidth() == 1)
      return setNameFn(getResult(), (intCst.getInt() ? "true" : "false"));

    // Otherwise, build a complex name with the value and type.
    SmallString<32> specialNameBuffer;
    llvm::raw_svector_ostream specialName(specialNameBuffer);
    specialName << 'c' << intCst.getValue();
    if (intType)
      specialName << '_' << type;
    setNameFn(getResult(), specialName.str());
  } else {
    setNameFn(getResult(), "cst");
  }
}
```

Similarly, an `Operation` can suggest the name for its block arguments using `getAsmBlockArgumentNames` method in `OpAsmOpInterface`.

For custom block names, `OpAsmOpInterface` has a method `getAsmBlockNames` so that
the operation can suggest a custom prefix instead of a generic `^bb0`.

Alternatively, `OpAsmTypeInterface` provides a `getAsmName` method for scenarios where the name could be inferred from its type.

## Defining Default Dialect

An `Operation` can indicate that the nested region in it has a default dialect prefix, and the operations in the region could elide the dialect prefix.

For example, in a `func.func` op all `func` prefix could be omitted:

```tablegen
include "mlir/IR/OpAsmInterface.td"

def FuncOp : Func_Op<"func", [
  OpAsmOpInterface
  ...
]> {
  let extraClassDeclaration = [{
    /// Allow the dialect prefix to be omitted.
    static StringRef getDefaultDialect() { return "func"; }
  }];
}
```

```mlir
func.func @main() {
  // actually func.call
  call @another()
}
```

---

## Deeper notes

### The directives that do most of the work

| Directive | Purpose |
|-----------|---------|
| `attr-dict` / `attr-dict-with-keyword` | print remaining attributes; nearly always required |
| `type($x)`, `qualified(type($x))` | operand and result types |
| `functional-type($ins, $outs)` | the `(A, B) -> C` form |
| `($x^)?` | optional group, anchored on `$x` |
| `custom<Foo>($a, type($b))` | hand-written fragment, with generated code around it |
| `oilist(...)` | unordered keyword clauses |
| `$region`, `regions`, `successors` | region and successor printing |

**The anchor rule.** An optional group needs an anchor (`^`) marking the element whose presence
decides whether the group is printed. "Expected directive with anchor" means you have an optional
group whose presence the printer cannot infer.

**Type inference reduces noise.** With `SameOperandsAndResultType` or `InferTypeOpInterface`, the
format can omit types the parser can reconstruct, which is where most of the readability gain in the
example above comes from.

**`qualified()` matters more than it looks.** Without it, a custom type inside a format may print in
its short form, which is ambiguous when the same mnemonic exists in two dialects. If a type
round-trips in isolation but fails inside an operation, this is usually why.

### When to give up and write C++

Hand-written `parse`/`print` is warranted when the syntax depends on the *value* of an attribute —
printing different keywords for different enum cases beyond what `custom<>` handles cleanly — or
when you are matching an external textual format exactly. The cost is that the two directions can
drift, so pair any hand-written pair with a round-trip test:

```mlir
// RUN: mlir-opt %s | mlir-opt | FileCheck %s
```

That one line catches almost every printer/parser mismatch, and it is the test people forget to add
precisely when they most need it.

### Aliases and name hints

`OpAsmDialectInterface` lets your dialect suggest aliases, so a type repeated three hundred times
prints once as `!my_alias` and is referenced thereafter. For dialects with verbose parameterized
types — layouts, quantization parameters, target descriptors — this is the difference between a dump
you can read and one you cannot.

The same interface family provides result name hints, so values print as `%conv` rather than `%17`.
`OpAsmOpInterface::getAsmResultNames` is a handful of lines and it improves every dump and every
test you will ever write against the dialect. It is one of the highest ratios of benefit to effort
available when building a dialect, and it is almost always skipped.

### Block argument and region syntax

Regions print with their block arguments, and custom formats can control that. If your operation has
a single-block region with an implicit terminator, `SingleBlockImplicitTerminator` plus a region
directive gives you the terse form where users need not write the terminator — the convention
followed by most upstream operations with bodies.


---

[← Constraints](04-constraints.md) · [Index](../README.md) · [Creating a Dialect: Build Setup →](06-creating-a-dialect-build-setup.md)
