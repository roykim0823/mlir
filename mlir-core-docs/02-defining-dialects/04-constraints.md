# Constraints

> **Section:** Defining Dialects · document 4 of 7  
> **Upstream:** [https://mlir.llvm.org/docs/DefiningDialects/Constraints/](https://mlir.llvm.org/docs/DefiningDialects/Constraints/) · source [`mlir/docs/DefiningDialects/Constraints.md`](https://github.com/llvm/llvm-project/blob/main/mlir/docs/DefiningDialects/Constraints.md)  
> **License:** upstream text is Apache-2.0 WITH LLVM-exception.

## Orientation

Short page, high leverage. Constraints are the predicates behind `AnyRankedTensor`, `I32`,
`Confined<AnyInteger, [IntMinValue<1>]>` and everything else you write in an ODS `arguments` list.
Understanding that they compose lets you express most verification declaratively rather than in a
hand-written `verify()`.

**Read first**

- [Operation Definition Specification (ODS)](02-operation-definition-specification.md)

**What you should be able to do after this page**

- Compose existing constraints instead of hand-writing verification.
- Define a new constraint with a good error message.

---

## Upstream documentation

## Attribute / Type Constraints

When defining the arguments of an operation in TableGen, users can specify
either plain attributes/types or use attribute/type constraints to levy
additional requirements on the attribute value or operand type.

```tablegen
def My_Type1 : MyDialect_Type<"Type1", "type1"> { ... }
def My_Type2 : MyDialect_Type<"Type2", "type2"> { ... }

// Plain type
let arguments = (ins MyType1:$val);
// Type constraint
let arguments = (ins AnyTypeOf<[MyType1, MyType2]>:$val);
```

`AnyTypeOf` is an example for a type constraints. Many useful type constraints
can be found in `mlir/IR/CommonTypeConstraints.td`. Additional verification
code is generated for type/attribute constraints. Type constraints can not only
be used when defining operation arguments, but also when defining type
parameters.

Optionally, C++ functions can be generated, so that type/attribute constraints
can be checked from C++. The name of the C++ function must be specified in the
`cppFunctionName` field. If no function name is specified, no C++ function is
emitted.

```tablegen
// Example: Element type constraint for VectorType
def Builtin_VectorTypeElementType : AnyTypeOf<[AnyInteger, Index, AnyFloat]> {
  let cppFunctionName = "isValidVectorTypeElementType";
}
```

The above example tranlates into the following C++ code:
```c++
bool isValidVectorTypeElementType(::mlir::Type type) {
  return (((::llvm::isa<::mlir::IntegerType>(type))) || ((::llvm::isa<::mlir::IndexType>(type))) || ((::llvm::isa<::mlir::FloatType>(type))));
}
```

An extra TableGen rule is needed to emit C++ code for type/attribute
constraints. This will generate only the declarations/definitions of the
type/attribute constaraints that are defined in the specified `.td` file, but
not those that are in included `.td` files.

```cmake
mlir_tablegen(<Your Dialect>TypeConstraints.h.inc -gen-type-constraint-decls)
mlir_tablegen(<Your Dialect>TypeConstraints.cpp.inc -gen-type-constraint-defs)
mlir_tablegen(<Your Dialect>AttrConstraints.h.inc -gen-attr-constraint-decls)
mlir_tablegen(<Your Dialect>AttrConstraints.cpp.inc -gen-attr-constraint-defs)
```

The generated `<Your Dialect>TypeConstraints.h.inc` respectivelly
`<Your Dialect>AttrConstraints.h.inc` will need to be included whereever you are
referencing the type/attributes constraint in C++. Note that no C++ namespace
will be emitted by the code generator. The `#include` statements of the
`.h.inc`/`.cpp.inc` files should be wrapped in C++ namespaces by the user.

---

## Deeper notes

### The composition operators

`AnyTypeOf<[F32, F64]>`, `Confined<Base, [Pred...]>`, `AllOfType`, `And<[...]>`, `Or<[...]>`,
`Neg<...>`, and `CPred<"...">` as the escape hatch that drops to a raw C++ expression over `$_self`.

Most useful constraints already exist. Read `mlir/include/mlir/IR/OpBase.td` and
`CommonTypeConstraints.td` once; the time is repaid immediately, because roughly half the
constraints people write by hand are already there under a name they did not guess.

A representative sample of what is already available: `AnyType`, `AnyInteger`, `SignlessIntegerLike`,
`AnyFloat`, `Index`, `AnyShaped`, `AnyRankedTensor`, `AnyStaticShapeTensor`, `AnyMemRef`,
`AnyVectorOfAnyRank`, `TensorOf<[F32, F16]>`, `MemRefRankOf<[AnyType], [2]>`, `IntMinValue<N>`,
`ArrayMinCount<N>`, `IsNullAttr`, `NonEmptyArrayAttr`.

### Error messages are part of the constraint

A constraint carries a `summary` used to build the verifier's diagnostic. `CPred` with no
description produces messages of the form "failed to verify that ..." with the raw predicate
inlined, which is unhelpful to your users. Every constraint you define should have a description
written as the *expectation*: "operand must be a ranked tensor of signless integers", not
"constraint failed".

### Where declarative constraints stop

They apply to one entity at a time. Relationships *between* operands — "the filter's channel count
must equal the input's" — need either a trait, an op-level `verify()`, or one of the parametric
`AllMatch`-style traits. A useful heuristic: single-entity properties go in the constraint,
cross-entity invariants go in `verify()` where you can produce a diagnostic that names both entities
and prints both values.

### Constraints are reused by the pattern languages

The same constraint records are available in DRR and PDLL, so a constraint you define once for
verification can also gate a rewrite pattern. That is a good reason to name and factor them rather
than inlining `CPred` expressions at each use site.


---

[← Defining Dialect Attributes and Types](03-defining-attributes-and-types.md) · [Index](../README.md) · [Customizing Assembly Behavior →](05-customizing-assembly-behavior.md)
