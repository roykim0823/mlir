# Traits

> **Section:** Core IR · document 5 of 7  
> **Upstream:** [https://mlir.llvm.org/docs/Traits/](https://mlir.llvm.org/docs/Traits/) · source [`mlir/docs/Traits/_index.md`](https://github.com/llvm/llvm-project/blob/main/mlir/docs/Traits/_index.md)  
> **License:** upstream text is Apache-2.0 WITH LLVM-exception.

## Orientation

Traits are the cheaper half of MLIR's genericity story. A trait is a C++ mixin attached to an
operation class that declares a property — `Commutative`, `Terminator`, `SameOperandsAndResultType`,
`IsolatedFromAbove`, `Pure` — and typically contributes verification and sometimes folding
behaviour.

Traits matter far more than their placement in the alphabetical index suggests, because much of the
core infrastructure keys off them. `IsolatedFromAbove` determines what the pass manager can
parallelize. `Terminator` determines block well-formedness. `Pure` determines whether an operation
can be deleted or hoisted. Getting the traits right on your operations is most of what makes them
behave like first-class citizens.

**Read first**

- [MLIR Language Reference](01-language-reference.md)

**What you should be able to do after this page**

- Pick the correct traits for a new operation instead of hand-writing verifiers.
- Explain the difference between a trait and an interface and choose between them.
- Know which core behaviours you inherit for free from which trait.

---

## Upstream documentation

MLIR allows for a truly open ecosystem, as any dialect may define attributes,
operations, and types that suit a specific level of abstraction. `Traits` are a
mechanism which abstracts implementation details and properties that are common
across many different attributes/operations/types/etc.. `Traits` may be used to
specify special properties and constraints of the object, including whether an
operation has side effects or that its output has the same type as the input.
Some examples of operation traits are `Commutative`, `Terminator`, etc. See the
more comprehensive list of [operation traits](#operation-traits-list) below for
more examples of what is possible.

## Defining a Trait

Traits may be defined in C++ by inheriting from the `TraitBase<ConcreteType,
TraitType>` class for the specific IR type. For attributes, this is
`AttributeTrait::TraitBase`. For operations, this is `OpTrait::TraitBase`. For
types, this is `TypeTrait::TraitBase`. This base class takes as template
parameters:

*   ConcreteType
    -   The concrete class type that this trait was attached to.
*   TraitType
    -   The type of the trait class that is being defined, for use with the
        [`Curiously Recurring Template Pattern`](https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern).

A derived trait class is expected to take a single template that corresponds to
the `ConcreteType`. An example trait definition is shown below:

```c++
template <typename ConcreteType>
class MyTrait : public TraitBase<ConcreteType, MyTrait> {
};
```

Operation traits may also provide a `verifyTrait` or `verifyRegionTrait` hook
that is called when verifying the concrete operation. The difference between
these two is that whether the verifier needs to access the regions, if so, the
operations in the regions will be verified before the verification of this
trait. The [verification order](../02-defining-dialects/02-operation-definition-specification.md#verification-ordering)
determines when a verifier will be invoked.

```c++
template <typename ConcreteType>
class MyTrait : public OpTrait::TraitBase<ConcreteType, MyTrait> {
public:
  /// Override the 'verifyTrait' hook to add additional verification on the
  /// concrete operation.
  static LogicalResult verifyTrait(Operation *op) {
    // ...
  }
};
```

Note: It is generally good practice to define the implementation of the
`verifyTrait` or `verifyRegionTrait` hook out-of-line as a free function when
possible to avoid instantiating the implementation for every concrete operation
type.

Operation traits may also provide a `foldTrait` hook that is called when folding
the concrete operation. The trait folders will only be invoked if the concrete
operation fold is either not implemented, fails, or performs an in-place fold.

The following signature of fold will be called if it is implemented and the op
has a single result.

```c++
template <typename ConcreteType>
class MyTrait : public OpTrait::TraitBase<ConcreteType, MyTrait> {
public:
  /// Override the 'foldTrait' hook to support trait based folding on the
  /// concrete operation.
  static OpFoldResult foldTrait(Operation *op, ArrayRef<Attribute> operands) {
    // ...
  }
};
```

Otherwise, if the operation has a single result and the above signature is not
implemented, or the operation has multiple results, then the following signature
will be used (if implemented):

```c++
template <typename ConcreteType>
class MyTrait : public OpTrait::TraitBase<ConcreteType, MyTrait> {
public:
  /// Override the 'foldTrait' hook to support trait based folding on the
  /// concrete operation.
  static LogicalResult foldTrait(Operation *op, ArrayRef<Attribute> operands,
                                 SmallVectorImpl<OpFoldResult> &results) {
    // ...
  }
};
```

Note: It is generally good practice to define the implementation of the
`foldTrait` hook out-of-line as a free function when possible to avoid
instantiating the implementation for every concrete operation type.

### Extra Declarations and Definitions
A trait may require additional declarations and definitions directly on
the Operation, Attribute or Type instances which specify that trait.
The `extraConcreteClassDeclaration` and `extraConcreteClassDefinition`
fields under the `NativeTrait` class are mechanisms designed for injecting
code directly into generated C++ Operation, Attribute or Type classes.

Code within the `extraConcreteClassDeclaration` field will be formatted and copied
into the generated C++ Operation, Attribute or Type class. Code within
`extraConcreteClassDefinition` will be added to the generated source file inside
the class’s C++ namespace. The substitution `$cppClass` is replaced by the C++ class
name.

The intention is to group trait specific logic together and reduce
redundant extra declarations and definitions on the instances themselves.

### Parametric Traits

The above demonstrates the definition of a simple self-contained trait. It is
also often useful to provide some static parameters to the trait to control its
behavior. Given that the definition of the trait class is rigid, i.e. we must
have a single template argument for the concrete object, the templates for the
parameters will need to be split out. An example is shown below:

```c++
template <int Parameter>
class MyParametricTrait {
public:
  template <typename ConcreteType>
  class Impl : public TraitBase<ConcreteType, Impl> {
    // Inside of 'Impl' we have full access to the template parameters
    // specified above.
  };
};
```

## Attaching a Trait

Traits may be used when defining a derived object type, by simply appending the
name of the trait class to the end of the base object class operation type:

```c++
/// Here we define 'MyAttr' along with the 'MyTrait' and `MyParametric trait
/// classes we defined previously.
class MyAttr : public Attribute::AttrBase<MyAttr, ..., MyTrait, MyParametricTrait<10>::Impl> {};
/// Here we define 'MyOp' along with the 'MyTrait' and `MyParametric trait
/// classes we defined previously.
class MyOp : public Op<MyOp, MyTrait, MyParametricTrait<10>::Impl> {};
/// Here we define 'MyType' along with the 'MyTrait' and `MyParametric trait
/// classes we defined previously.
class MyType : public Type::TypeBase<MyType, ..., MyTrait, MyParametricTrait<10>::Impl> {};
```

### Attaching Operation Traits in ODS

To use an operation trait in the [ODS](../02-defining-dialects/02-operation-definition-specification.md) framework, we need to
provide a definition of the trait class. This can be done using the
`NativeOpTrait` and `ParamNativeOpTrait` classes. `ParamNativeOpTrait` provides
a mechanism in which to specify arguments to a parametric trait class with an
internal `Impl`.

```tablegen
// The argument is the c++ trait class name.
def MyTrait : NativeOpTrait<"MyTrait">;

// The first argument is the parent c++ class name. The second argument is a
// string containing the parameter list.
class MyParametricTrait<int prop>
  : NativeOpTrait<"MyParametricTrait", !cast<string>(!head(parameters))>;
```

These can then be used in the `traits` list of an op definition:

```tablegen
def OpWithInferTypeInterfaceOp : Op<...[MyTrait, MyParametricTrait<10>]> { ... }
```

See the documentation on [operation definitions](../02-defining-dialects/02-operation-definition-specification.md) for more
details.

## Using a Trait

Traits may be used to provide additional methods, static fields, or other
information directly on the concrete object. `Traits` internally become `Base`
classes of the concrete operation, so all of these are directly accessible. To
expose this information opaquely to transformations and analyses,
[`interfaces`](07-interfaces.md) may be used.

To query if a specific object contains a specific trait, the `hasTrait<>` method
may be used. This takes as a template parameter the trait class, which is the
same as the one passed when attaching the trait to an operation.

```c++
Operation *op = ..;
if (op->hasTrait<MyTrait>() || op->hasTrait<MyParametricTrait<10>::Impl>())
  ...;
```

## Operation Traits List

MLIR provides a suite of traits that provide various functionalities that are
common across many different operations. Below is a list of some key traits that
may be used directly by any dialect. The format of the header for each trait
section goes as follows:

*   `Header`
    -   (`C++ class` -- `ODS class`(if applicable))

### AffineScope

*   `OpTrait::AffineScope` -- `AffineScope`

This trait is carried by region holding operations that define a new scope for
the purposes of polyhedral optimization and the affine dialect in particular.
Any SSA values of 'index' type that either dominate such operations, or are
defined at the top-level of such operations, or appear as region arguments for
such operations automatically become valid symbols for the polyhedral scope
defined by that operation. As a result, such SSA values could be used as the
operands or index operands of various affine dialect operations like affine.for,
affine.load, and affine.store. The polyhedral scope defined by an operation with
this trait includes all operations in its region excluding operations that are
nested inside of other operations that themselves have this trait.

### AutomaticAllocationScope

*   `OpTrait::AutomaticAllocationScope` -- `AutomaticAllocationScope`

This trait is carried by region holding operations that define a new scope for
automatic allocation. Such allocations are automatically freed when control is
transferred back from the regions of such operations. As an example, allocations
performed by
[`memref.alloca`](https://mlir.llvm.org/docs/Dialects/MemRef/#memrefalloca-memrefallocaop) are
automatically freed when control leaves the region of its closest surrounding op
that has the trait AutomaticAllocationScope.

### Broadcastable

*   `OpTrait::ResultsBroadcastableShape` -- `ResultsBroadcastableShape`

This trait adds the property that the operation is known to have
[broadcast-compatible](https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
operands and that its result type is compatible with the inferred broadcast shape. 
See [The `Broadcastable` Trait](06-trait-broadcastable.md) for details.

### Commutative

*   `OpTrait::IsCommutative` -- `Commutative`

This trait adds the property that the operation is commutative, i.e. `X op Y ==
Y op X`

### ElementwiseMappable

*   `OpTrait::ElementwiseMappable` -- `ElementwiseMappable`

This trait tags scalar ops that also can be applied to vectors/tensors, with
their semantics on vectors/tensors being elementwise application. This trait
establishes a set of properties that allow reasoning about / converting between
scalar/vector/tensor code. These same properties allow blanket implementations
of various analyses/transformations for all `ElementwiseMappable` ops.

Note: Not all ops that are "elementwise" in some abstract sense satisfy this
trait. In particular, broadcasting behavior is not allowed. See the comments on
`OpTrait::ElementwiseMappable` for the precise requirements.

### HasParent

*   `OpTrait::HasParent<typename ParentOpType>` -- `HasParent<string op>` or
    `ParentOneOf<list<string> opList>`

This trait provides APIs and verifiers for operations that can only be nested
within regions that are attached to operations of `ParentOpType`.

### HasAncestor

*   `OpTrait::HasAncestor<typename AncestorOpType>` -- `HasAncestor<string op>`
    or `AncestorOneOf<list<string> opList>`

This trait provides APIs and verifiers for operations that must appear somewhere
inside a region attached to an operation of `AncestorOpType`. Unlike `HasParent`,
which checks only the immediate parent, `HasAncestor` walks the full ancestor
chain.

### IsolatedFromAbove

*   `OpTrait::IsIsolatedFromAbove` -- `IsolatedFromAbove`

This trait signals that the regions of an operations are known to be isolated
from above. This trait asserts that the regions of an operation will not
capture, or reference, SSA values defined above the region scope. This means
that the following is invalid if `foo.region_op` is defined as
`IsolatedFromAbove`:

```mlir
%result = arith.constant 10 : i32
foo.region_op {
  foo.yield %result : i32
}
```

This trait is an important structural property of the IR, and enables operations
to have [passes](https://github.com/llvm/llvm-project/blob/main/mlir/docs/PassManagement) scheduled under them.

### MemRefsNormalizable

*   `OpTrait::MemRefsNormalizable` -- `MemRefsNormalizable`

This trait is used to flag operations that consume or produce values of `MemRef`
type where those references can be 'normalized'. In cases where an associated
`MemRef` has a non-identity memory-layout specification, such normalizable
operations can be modified so that the `MemRef` has an identity layout
specification. This can be implemented by associating the operation with its own
index expression that can express the equivalent of the memory-layout
specification of the MemRef type. See [the -normalize-memrefs pass](https://mlir.llvm.org/docs/Passes/#-normalize-memrefs).

### Single Block Region

*   `OpTrait::SingleBlock` -- `SingleBlock`

This trait provides APIs and verifiers for operations with regions that have a
single block.

### Single Block with Implicit Terminator

*   `OpTrait::SingleBlockImplicitTerminator<typename TerminatorOpType>` --
    `SingleBlockImplicitTerminator<string op>`

This trait implies the `SingleBlock` above, but adds the additional requirement
that the single block must terminate with `TerminatorOpType`.

### SymbolTable

*   `OpTrait::SymbolTable` -- `SymbolTable`

This trait is used for operations that define a
[`SymbolTable`](04-symbols-and-symbol-tables.md#symbol-table).

### Terminator

*   `OpTrait::IsTerminator` -- `Terminator`

This trait provides verification and functionality for operations that are known
to be [terminators](01-language-reference.md#control-flow-and-ssacfg-regions).

*   `OpTrait::NoTerminator` -- `NoTerminator`

This trait removes the requirement on regions held by an operation to have
[terminator operations](01-language-reference.md#control-flow-and-ssacfg-regions) at the end of a block.
This requires that these regions have a single block. An example of operation
using this trait is the top-level `ModuleOp`.

### TokenProducerTrait

*   `OpTrait::TokenProducerTrait` -- `TokenProducerTrait`

This trait marks operations that are allowed to produce builtin `token` values
as operation results or as region entry block arguments.

### TokenConsumerTrait

*   `OpTrait::TokenConsumerTrait` -- `TokenConsumerTrait`

This trait marks operations that are allowed to consume builtin `token` values
as operands.

---

## Deeper notes

### Trait or interface?

The most common point of confusion in this section, and the rule is simple:

| | Trait | Interface |
|---|---|---|
| Question it answers | "Is this op X?" | "Do X to this op" |
| Dispatch | static, template mixin, zero cost | virtual, through a table |
| Carries data or behaviour? | verification, occasionally a static helper | arbitrary methods with op-specific implementations |
| Example | `Commutative` — a fact | `LoopLikeOpInterface::getLoopBody()` — an action |

Use a trait for a boolean property that generic code branches on. Use an interface when generic
code needs to *ask the operation to do something* whose implementation differs per operation.

### The traits that carry the most weight

**`Pure`** — the modern spelling combining "no side effects" with speculatability. Unlocks dead-code
elimination, CSE, hoisting and speculation. Applying it to an operation that traps or reads memory
produces miscompiles that are very hard to trace, because the resulting deletion happens in a pass
with no knowledge of your dialect. When in doubt, model effects explicitly with
`MemoryEffectsOpInterface` — see
[Side Effects and Speculation](../08-rationale/01-side-effects-and-speculation.md), which is the
normative guidance on this.

**`IsolatedFromAbove`** — the region cannot reference values defined outside it. Required for
anything the pass manager treats as an independent unit; it is what makes multithreaded pass
execution sound.

**`Terminator`** — must be last in its block. Combined with `SingleBlockImplicitTerminator`, this
lets an operation with a single-block region auto-insert the terminator so users do not have to
write it explicitly in the common case.

**`SameOperandsAndResultType`** and relatives (`SameTypeOperands`, `SameOperandsAndResultShape`,
`AllTypesMatch`) — verification you would otherwise hand-write, and also the basis for terser
assembly formats, since the printer can omit types it can infer.

**`AttrSizedOperandSegments`** — needed whenever an operation has more than one variadic operand
group, because otherwise the flat operand list is ambiguous. Forgetting it produces a verifier error
whose message rarely points at the real cause.

**`ConstantLike`**, **`Idempotent`**, **`Involution`**, **`Commutative`** — small algebraic facts
that the canonicalizer acts on directly. `Involution` alone gives you `f(f(x)) → x` for free.

**`InferTypeOpInterface`** is the interface counterpart worth mentioning here: with it, builders can
compute result types instead of requiring callers to pass them, which materially improves the
ergonomics of constructing your ops in patterns.

### Cost model

Traits are templates. They cost nothing at run time but they do cost compile time and code size, and
a trait contributing a verifier runs on every verification pass. That is rarely a problem, but on
very large modules the verifier is a real line item in profiles — `--mlir-disable-verify` exists
partly for that reason.

### A practical checklist for a new operation

1. Does it have side effects? If none and it cannot trap, `Pure`. Otherwise model effects.
2. Is it a terminator?
3. Do its operand and result types relate? Use the `Same*`/`AllTypesMatch` family.
4. More than one variadic group? `AttrSizedOperandSegments`.
5. Is it algebraically special — commutative, idempotent, an involution?
6. Does it hold a region that should be isolated?

Working through those six questions before writing a `verify()` typically removes most of the code
you were about to write.


---

[← Symbols and Symbol Tables](04-symbols-and-symbol-tables.md) · [Index](../README.md) · [The `Broadcastable` Trait →](06-trait-broadcastable.md)
