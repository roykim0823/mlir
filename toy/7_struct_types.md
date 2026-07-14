# Chapter 7: Adding a Composite Type to Toy

> Goal: extend the Toy language and dialect with a user-defined `struct` type — touching every layer of the compiler from the lexer down to constant folding — following the official tutorial: [Toy Tutorial Ch-7: Adding a Composite Type to Toy](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-7/).

**Chapter code:** `/Users/roy/study/mlir/toy/Ch7/` (an out-of-tree CMake project, *not* built inside llvm-project — built as part of the `toy/` superbuild). Built against Homebrew LLVM/MLIR 20 on macOS.

```bash
cd /Users/roy/study/mlir/toy
./build.sh ch7                # configure once + incremental build → ./build/bin/toyc-ch7
./run.sh ch7                  # runs struct-codegen.toy with -emit=mlir
```

---

## 1. Overview — Why a Struct Type Stresses Every Layer

Everything the previous chapters manipulated was a single type: an `f64` tensor. Chapter 7 asks a deceptively simple question — *"what if users can define their own composite types?"* — and answering it forces a change at **every layer** of the compiler:

| Layer | What changes |
|---|---|
| **Lexer** | New `struct` keyword (`tok_struct`) |
| **Parser** | Struct definitions, struct-typed variable declarations, `{...}` struct literals, `.` member access |
| **AST** | `StructAST`, `StructLiteralExprAST`, a `RecordAST` base class, `VarType` grows a `name` field |
| **MLIR type system** | A new `StructType` with a hand-written `StructTypeStorage` (the type-uniquing pattern) |
| **Dialect** | `addTypes<StructType>()`, custom `parseType`/`printType` for `!toy.struct<...>` |
| **Ops (ODS)** | New `toy.struct_constant` and `toy.struct_access`; `Toy_Type` constraint so `return`/`generic_call` accept structs |
| **MLIRGen** | Struct declaration tracking (`structMap`), literal codegen as `ArrayAttr`, member access as an index |
| **Optimization** | `fold()` hooks (`FoldAdaptor`), the dialect `materializeConstant` hook — so after inlining, structs *disappear entirely* |
| **Lowering / JIT** | Nothing! Once folding removes the structs, the Ch6 pipeline lowers and JITs the code unchanged |

That last row is the punchline of this chapter: we never write a lowering for `StructType`. Instead, the design leans on **inlining + constant folding** — structs only exist as compile-time aggregates in this toy language, so once functions are inlined and constants are propagated, every `struct_access` of a `struct_constant` folds to a plain tensor constant, and the Ch5/Ch6 lowering pipeline works untouched.

This is a very common pattern in real MLIR-based compilers: high-level types that carry structure through the frontend, then evaporate before lowering.

The language extension we implement:

```
# A struct is declared with named members (no shapes, no initializers).
struct Struct {
  var a;
  var b;
}

# Functions are generic over struct-typed arguments too.
def multiply_transpose(Struct value) {
  # '.' accesses a member.
  return transpose(value.a) * transpose(value.b);
}

def main() {
  # Composite initializer: one sub-literal per member, in order.
  Struct value = {[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]};
  var c = multiply_transpose(value);
  print(c);
}
```

---

## 2. Language Changes — Lexer, Parser, AST

### 2.1 Grammar additions

```
# A struct definition is a new kind of top-level record:
struct-definition ::= `struct` identifier `{` variable-declaration-list `}`

# Members may not have shapes or initializers:
variable-declaration-list ::= decl `;` variable-declaration-list?

# Declaring a struct-typed variable uses the struct name as the "type":
struct-declaration ::= identifier identifier (`=` expr)?

# Struct literal: nested tensor literals / numbers / struct literals in braces:
struct-literal ::= `{` (struct-literal | tensor-literal | number) (`,` ...)* `}`

# Member access is the '.' binary operator (highest precedence):
access ::= expr `.` identifier
```

### 2.2 Lexer: one new token

`/Users/roy/study/mlir/toy/Ch7/include/toy/Lexer.h` adds a single keyword token:

```cpp
enum Token : int {
  ...
  // commands
  tok_return = -2,
  tok_var = -3,
  tok_def = -4,
  tok_struct = -5,          // NEW
  ...
};

// In getTok():
if (identifierStr == "struct")
  return tok_struct;
```

That is the *entire* lexer change. `.`, `{`, `}` were already single-character tokens.

### 2.3 AST: records, struct definitions, struct literals

`/Users/roy/study/mlir/toy/Ch7/include/toy/AST.h` makes three structural changes:

**(a) `VarType` gains a `name`.** Previously a variable's type was just an optional shape. Now it may instead be a *named* struct type:

```cpp
/// A variable type with either name or shape information.
struct VarType {
  std::string name;              // NEW: non-empty ⇒ struct type
  std::vector<int64_t> shape;    // as before: tensor shape (possibly empty)
};
```

**(b) A `RecordAST` base class.** A module used to be a list of functions; now it is a list of *records*, each either a function or a struct definition (LLVM-style RTTI via `getKind()`):

```cpp
class RecordAST {
public:
  enum RecordASTKind { Record_Function, Record_Struct };
  ...
};

/// This class represents a struct definition.
class StructAST : public RecordAST {
  Location location;
  std::string name;
  std::vector<std::unique_ptr<VarDeclExprAST>> variables;
  ...
};

class ModuleAST {
  std::vector<std::unique_ptr<RecordAST>> records;   // was: FunctionAST list
  ...
};
```

**(c) A struct literal expression.** A new `ExprASTKind` value `Expr_StructLiteral` and:

```cpp
/// Expression class for a literal struct value.
class StructLiteralExprAST : public ExprAST {
  std::vector<std::unique_ptr<ExprAST>> values;   // one entry per member
  ...
};
```

Note there is **no dedicated AST node for member access** — `value.a` is parsed as an ordinary `BinaryExprAST` with `op == '.'` and a `VariableExprAST` (`a`) on the RHS. MLIRGen interprets it specially (section 6).

### 2.4 Parser changes

All in `/Users/roy/study/mlir/toy/Ch7/include/toy/Parser.h`.

**Top-level dispatch** (`parseModule`) now accepts both records:

```cpp
case tok_def:
  record = parseDefinition();
  break;
case tok_struct:
  record = parseStruct();
  break;
default:
  return parseError<ModuleAST>("'def' or 'struct'", ...);
```

**Struct definitions** — `definition ::= 'struct' identifier '{' decl+ '}'`:

```cpp
std::unique_ptr<StructAST> parseStruct() {
  auto loc = lexer.getLastLocation();
  lexer.consume(tok_struct);
  if (lexer.getCurToken() != tok_identifier)
    return parseError<StructAST>("name", "in struct definition");
  std::string name(lexer.getId());
  lexer.consume(tok_identifier);

  // Parse: '{'
  if (lexer.getCurToken() != '{')
    return parseError<StructAST>("{", "in struct definition");
  lexer.consume(Token('{'));

  // Parse: decl+  (each terminated by ';')
  std::vector<std::unique_ptr<VarDeclExprAST>> decls;
  do {
    auto decl = parseDeclaration(/*requiresInitializer=*/false);
    if (!decl)
      return nullptr;
    decls.push_back(std::move(decl));
    ...
  } while (lexer.getCurToken() != '}');

  lexer.consume(Token('}'));
  return std::make_unique<StructAST>(loc, name, std::move(decls));
}
```

**Struct-typed declarations.** `parseDeclaration` now handles two forms — `var x ...` (tensor) and `TypeName x = ...` (struct). Inside a function body, an identifier can begin either a call *or* a typed declaration, so the parser disambiguates on the next token:

```cpp
/// Parse either a variable declaration or a call expression.
std::unique_ptr<ExprAST> parseDeclarationOrCallExpr() {
  auto loc = lexer.getLastLocation();
  std::string id(lexer.getId());
  lexer.consume(tok_identifier);

  // Check for a call expression.
  if (lexer.getCurToken() == '(')
    return parseCallExpr(id, loc);

  // Otherwise, this is a variable declaration.
  return parseTypedDeclaration(id, /*requiresInitializer=*/true, loc);
}
```

`parseTypedDeclaration` stores the type name into `VarType::name`. Function prototypes get the same treatment: `def foo(Struct value)` — if the token after the first identifier is another identifier, the first one was a type name.

**Struct literals** — `structLiteral ::= { (structLiteral | tensorLiteral | number)+ }`:

```cpp
std::unique_ptr<ExprAST> parseStructLiteralExpr() {
  auto loc = lexer.getLastLocation();
  lexer.consume(Token('{'));

  std::vector<std::unique_ptr<ExprAST>> values;
  do {
    if (lexer.getCurToken() == '[') {            // nested tensor literal
      values.push_back(parseTensorLiteralExpr());
    } else if (lexer.getCurToken() == tok_number) {
      values.push_back(parseNumberExpr());
    } else {                                     // nested struct literal
      if (lexer.getCurToken() != '{')
        return parseError<ExprAST>("{, [, or number",
                                   "in struct literal expression");
      values.push_back(parseStructLiteralExpr());
    }
    ...
  } while (true);
  ...
  return std::make_unique<StructLiteralExprAST>(std::move(loc),
                                                std::move(values));
}
```

and `parsePrimary()` dispatches `'{' → parseStructLiteralExpr()`.

**Member access as an operator.** `.` is simply registered as the tightest-binding binary operator in `getTokPrecedence()`:

```cpp
switch (static_cast<char>(lexer.getCurToken())) {
case '-': return 20;
case '+': return 20;
case '*': return 40;
case '.': return 60;     // NEW: member access binds tightest
default:  return -1;
}
```

So `transpose(value.a) * transpose(value.b)` parses with `.` bound before `*`, no new expression machinery needed.

### 2.5 Seeing it: the AST dump

Real output from this repo (from `toy/`: `./build/bin/toyc-ch7 ../test_Example/Toy/Ch7/struct-ast.toy -emit=ast`, abbreviated):

```
Module:
  Struct: Struct @../test_Example/Toy/Ch7/struct-ast.toy:3:1
    Variables: [
      VarDecl a<> @../test_Example/Toy/Ch7/struct-ast.toy:4:3
      VarDecl b<> @../test_Example/Toy/Ch7/struct-ast.toy:5:3
    ]
  Function
    Proto 'multiply_transpose' @../test_Example/Toy/Ch7/struct-ast.toy:9:1
    Params: [value]
    Block {
      Return
        BinOp: * @...:11:31
          Call 'transpose' [ @...:11:10
            BinOp: . @...:11:26
              var: value @...:11:20
              var: a @...:11:26
          ]
          Call 'transpose' [ @...:11:31
            BinOp: . @...:11:47
              var: value @...:11:41
              var: b @...:11:47
          ]
    } // Block
  Function
    Proto 'main' @...:14:1
    Params: []
    Block {
      VarDecl value<Struct> @...:16:3
        Struct Literal:  Literal: <2, 3>[ ... ]
          Literal: <2, 3>[ ... ]
      VarDecl c<> @...:19:3
        Call 'multiply_transpose' [ var: value ]
      Print [ var: c ]
    } // Block
```

Note `VarDecl value<Struct>` (a *named* type in the angle brackets instead of a shape) and `BinOp: .` for member access.

---

## 3. Defining StructType in MLIR — the Type Storage Pattern

This is the conceptual heart of the chapter. MLIR `Type` objects are **value types**: a `mlir::Type` is just a wrapper around a pointer to an immutable, *uniqued* storage instance owned by the `MLIRContext`. Two structurally identical types are the *same pointer*, which makes type equality a pointer comparison and types trivially cheap to copy and hash. Getting a new type into this system means writing a **storage class** and hooking it into the context's `StorageUniquer`.

Builtin types like `RankedTensorType` already have storage; a struct type parameterized by an arbitrary list of element types needs its own.

### 3.1 StructTypeStorage

From `/Users/roy/study/mlir/toy/Ch7/mlir/Dialect.cpp` (namespace `mlir::toy::detail`):

```cpp
/// This class represents the internal storage of the Toy `StructType`.
struct StructTypeStorage : public mlir::TypeStorage {
  /// The `KeyTy` is a required type that provides an interface for the storage
  /// instance. This type will be used when uniquing an instance of the type
  /// storage. For our struct type, we will unique each instance structurally on
  /// the elements that it contains.
  using KeyTy = llvm::ArrayRef<mlir::Type>;

  /// A constructor for the type storage instance.
  StructTypeStorage(llvm::ArrayRef<mlir::Type> elementTypes)
      : elementTypes(elementTypes) {}

  /// Define the comparison function for the key type with the current storage
  /// instance. Used when constructing a new instance to ensure we haven't
  /// already uniqued an instance of the given key.
  bool operator==(const KeyTy &key) const { return key == elementTypes; }

  /// Define a hash function for the key type.
  /// Note: This method isn't necessary as both llvm::ArrayRef and mlir::Type
  /// have hash functions available, so we could just omit this entirely.
  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_value(key);
  }

  /// Define a construction function for the key type from a set of parameters.
  /// Note: This method isn't necessary because KeyTy can be directly
  /// constructed with the given parameters.
  static KeyTy getKey(llvm::ArrayRef<mlir::Type> elementTypes) {
    return KeyTy(elementTypes);
  }

  /// Define a construction method for creating a new instance of this storage.
  /// The given allocator must be used for *all* necessary dynamic allocations
  /// used to create the type storage and its internal.
  static StructTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                      const KeyTy &key) {
    // Copy the elements from the provided `KeyTy` into the allocator.
    llvm::ArrayRef<mlir::Type> elementTypes = allocator.copyInto(key);

    // Allocate the storage instance and construct it.
    return new (allocator.allocate<StructTypeStorage>())
        StructTypeStorage(elementTypes);
  }

  /// The following field contains the element types of the struct.
  llvm::ArrayRef<mlir::Type> elementTypes;
};
```

The pieces and what the `StorageUniquer` does with each:

| Member | Role in uniquing |
|---|---|
| `KeyTy` | The "identity" of a type instance. Here: the list of element types. Two `StructType`s with equal keys are the same type. |
| `operator==(const KeyTy&)` | Compares a candidate key against an *existing* storage instance during hash-table lookup. |
| `hashKey(KeyTy)` (optional) | Hash for the bucket lookup. Defaultable when `KeyTy` is hashable via `llvm::hash_value`, as it is here. |
| `getKey(params...)` (optional) | Builds a `KeyTy` from the arguments passed to `Base::get`. Defaultable when `KeyTy` is directly constructible from them. |
| `construct(allocator, key)` | Called **only on a miss**: allocates a permanent storage instance. Crucially, `allocator.copyInto(key)` copies the caller's (possibly stack-lived) `ArrayRef` into context-owned memory — the storage must never point at caller memory. |

### 3.2 The public StructType class

From `/Users/roy/study/mlir/toy/Ch7/include/toy/Dialect.h`:

```cpp
/// All derived types in MLIR must inherit from the CRTP class
/// 'Type::TypeBase'. It takes as template parameters the concrete type
/// (StructType), the base class to use (Type), and the storage class
/// (StructTypeStorage).
class StructType : public mlir::Type::TypeBase<StructType, mlir::Type,
                                               detail::StructTypeStorage> {
public:
  /// Inherit some necessary constructors from 'TypeBase'.
  using Base::Base;

  /// Create an instance of a `StructType` with the given element types. There
  /// *must* be at least one element type.
  static StructType get(llvm::ArrayRef<mlir::Type> elementTypes);

  /// Returns the element types of this struct type.
  llvm::ArrayRef<mlir::Type> getElementTypes();

  /// Returns the number of element type held by this struct.
  size_t getNumElementTypes() { return getElementTypes().size(); }

  /// The name of this struct type.
  static constexpr StringLiteral name = "toy.struct";
};
```

(The `name` constant is required in recent MLIR for types defined without ODS — it is used e.g. by the bytecode reader/writer and debugging utilities. The upstream tutorial predates it slightly; with LLVM/MLIR 20 you need it.)

And the implementation in `Dialect.cpp`:

```cpp
StructType StructType::get(llvm::ArrayRef<mlir::Type> elementTypes) {
  assert(!elementTypes.empty() && "expected at least 1 element type");

  // Call into a helper 'get' method in 'TypeBase' to get a uniqued instance
  // of this type. The first parameter is the context to unique in. The
  // parameters after the context are forwarded to the storage instance.
  mlir::MLIRContext *ctx = elementTypes.front().getContext();
  return Base::get(ctx, elementTypes);
}

llvm::ArrayRef<mlir::Type> StructType::getElementTypes() {
  // 'getImpl' returns a pointer to the internal storage instance.
  return getImpl()->elementTypes;
}
```

`Base::get(ctx, args...)` is where the magic lives: it asks the context's `StorageUniquer` for an instance keyed by `getKey(args...)`; on a hit it returns the existing storage pointer wrapped in a `StructType`, on a miss it calls `StructTypeStorage::construct`. This is **why MLIR uniques types**: `!toy.struct<tensor<*xf64>, tensor<*xf64>>` created in two different files/passes is literally the same object, so `type1 == type2` is a pointer compare and types can be used as map keys for free.

### 3.3 Registering with the dialect

The dialect must own the type so the context knows `!toy.…` types exist:

```cpp
void ToyDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "toy/Ops.cpp.inc"
      >();
  addInterfaces<ToyInlinerInterface>();
  addTypes<StructType>();          // NEW in Ch7
}
```

---

## 4. Custom Type Parsing & Printing

`StructType` now exists in memory, but the textual IR needs a syntax for it. Types from a dialect print as `!<dialect>.<contents>`; the dialect provides the `<contents>` part via `parseType`/`printType` hooks. In ODS (`Ops.td`) we ask TableGen to declare them:

```tablegen
def Toy_Dialect : Dialect {
  let name = "toy";
  let cppNamespace = "::mlir::toy";

  // We set this bit to generate a declaration of the `materializeConstant`
  // method so that we can materialize constants for our toy operations.
  let hasConstantMaterializer = 1;

  // We set this bit to generate the declarations for the dialect's type
  // parsing and printing hooks.
  let useDefaultTypePrinterParser = 1;
}
```

The grammar we choose:

```
struct-type ::= `struct` `<` type (`,` type)* `>`
```

so a two-member struct of unranked tensors prints as `!toy.struct<tensor<*xf64>, tensor<*xf64>>` (the `!toy.` prefix is added by MLIR; the hook only handles what follows).

### 4.1 Parsing (Dialect.cpp)

```cpp
/// Parse an instance of a type registered to the toy dialect.
mlir::Type ToyDialect::parseType(mlir::DialectAsmParser &parser) const {
  // Parse a struct type in the following form:
  //   struct-type ::= `struct` `<` type (`,` type)* `>`

  // Parse: `struct` `<`
  if (parser.parseKeyword("struct") || parser.parseLess())
    return Type();

  // Parse the element types of the struct.
  SmallVector<mlir::Type, 1> elementTypes;
  do {
    // Parse the current element type.
    SMLoc typeLoc = parser.getCurrentLocation();
    mlir::Type elementType;
    if (parser.parseType(elementType))
      return nullptr;

    // Check that the type is either a TensorType or another StructType.
    if (!llvm::isa<mlir::TensorType, StructType>(elementType)) {
      parser.emitError(typeLoc, "element type for a struct must either "
                                "be a TensorType or a StructType, got: ")
          << elementType;
      return Type();
    }
    elementTypes.push_back(elementType);

    // Parse the optional: `,`
  } while (succeeded(parser.parseOptionalComma()));

  // Parse: `>`
  if (parser.parseGreater())
    return Type();
  return StructType::get(elementTypes);
}
```

Points worth noticing:

- All parser methods return `ParseResult` (a `LogicalResult` that converts to `true` **on failure**), which is what makes the `if (a || b)` chaining idiom work.
- The parser is also a **verifier**: it rejects `!toy.struct<i32>` with a proper source-located diagnostic. Structs may nest (`!toy.struct<!toy.struct<...>, tensor<*xf64>>` — exercised by `struct-opt.mlir`).

### 4.2 Printing (Dialect.cpp)

```cpp
/// Print an instance of a type registered to the toy dialect.
void ToyDialect::printType(mlir::Type type,
                           mlir::DialectAsmPrinter &printer) const {
  // Currently the only toy type is a struct type.
  StructType structType = llvm::cast<StructType>(type);

  // Print the struct type according to the parser format.
  printer << "struct<";
  llvm::interleaveComma(structType.getElementTypes(), printer);
  printer << '>';
}
```

`llvm::interleaveComma` prints the element types separated by `", "` — and since each element is itself an `mlir::Type`, nested structs recurse naturally.

### 4.3 Exposing the type to ODS: `Toy_StructType` and `Toy_Type`

Ops defined in TableGen constrain operands/results with type-constraint definitions. To let ODS talk about our C++-defined type, `Ops.td` wraps it in a `DialectType` predicate:

```tablegen
// Provide a definition for the Toy StructType for use in ODS. This allows for
// using StructType in a similar way to Tensor or MemRef. We use `DialectType`
// to demarcate the StructType as belonging to the Toy dialect.
def Toy_StructType :
    DialectType<Toy_Dialect, CPred<"::llvm::isa<StructType>($_self)">,
                "Toy struct type">;

// Provide a definition of the types that are used within the Toy dialect.
def Toy_Type : AnyTypeOf<[F64Tensor, Toy_StructType]>;
```

`Toy_Type` = "tensor **or** struct" is the constraint we then thread through existing ops (next section).

---

## 5. New Operations

### 5.1 `toy.struct_constant`

Materializes a struct *value* from a compile-time attribute. Since a struct is an ordered collection, its payload is an `ArrayAttr` — one attribute per member (a `DenseElementsAttr` for tensor members, or a nested `ArrayAttr` for nested structs). ODS (`/Users/roy/study/mlir/toy/Ch7/include/toy/Ops.td`):

```tablegen
def StructConstantOp : Toy_Op<"struct_constant", [ConstantLike, Pure]> {
  let summary = "struct constant";
  let description = [{
    Constant operation turns a literal struct value into an SSA value. The data
    is attached to the operation as an attribute. The struct constant is encoded
    as an array of other constant values. For example:

    ```mlir
      %0 = toy.struct_constant [
        dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64>
      ] : !toy.struct<tensor<*xf64>>
    ```
  }];

  let arguments = (ins ArrayAttr:$value);
  let results = (outs Toy_StructType:$output);

  let assemblyFormat = "$value attr-dict `:` type($output)";

  let hasVerifier = 1;
  let hasFolder = 1;
}
```

The `ConstantLike` trait matters: it tells generic MLIR utilities (folding, `matchPattern(m_Constant())`, the operation folder) that this op produces a constant whose value is its single attribute.

Its verifier reuses the same recursive helper as `toy.constant` — `verifyConstantForType` in `Dialect.cpp` — which structurally checks the attribute against the result type:

```cpp
static llvm::LogicalResult verifyConstantForType(mlir::Type type,
                                                 mlir::Attribute opaqueValue,
                                                 mlir::Operation *op) {
  if (llvm::isa<mlir::TensorType>(type)) {
    // ... tensor case: DenseFPElementsAttr, rank/shape must match ...
  }
  auto resultType = llvm::cast<StructType>(type);
  llvm::ArrayRef<mlir::Type> resultElementTypes = resultType.getElementTypes();

  // Verify that the initializer is an Array with the right number of elements.
  auto attrValue = llvm::dyn_cast<ArrayAttr>(opaqueValue);
  if (!attrValue || attrValue.getValue().size() != resultElementTypes.size())
    return op->emitError("constant of StructType must be initialized by an "
                         "ArrayAttr with the same number of elements, got ")
           << opaqueValue;

  // Check that each of the elements are valid (recurses into nested structs).
  llvm::ArrayRef<mlir::Attribute> attrElementValues = attrValue.getValue();
  for (const auto it : llvm::zip(resultElementTypes, attrElementValues))
    if (failed(verifyConstantForType(std::get<0>(it), std::get<1>(it), op)))
      return mlir::failure();
  return mlir::success();
}

llvm::LogicalResult StructConstantOp::verify() {
  return verifyConstantForType(getResult().getType(), getValue(), *this);
}
```

### 5.2 `toy.struct_access`

Extracts the N-th member of a struct value:

```tablegen
def StructAccessOp : Toy_Op<"struct_access", [Pure]> {
  let summary = "struct access";
  let description = [{
    Access the Nth element of a value returning a struct type.
  }];

  let arguments = (ins Toy_StructType:$input, I64Attr:$index);
  let results = (outs Toy_Type:$output);

  let assemblyFormat = [{
    $input `[` $index `]` attr-dict `:` type($input) `->` type($output)
  }];

  // Allow building a StructAccessOp with just a struct value and an index.
  let builders = [
    OpBuilder<(ins "Value":$input, "size_t":$index)>
  ];

  let hasVerifier = 1;

  // Set the folder bit so that we can fold constant accesses.
  let hasFolder = 1;
}
```

`Pure` (no side effects, speculatable) is essential: it allows dead `struct_access` ops to be erased and lets the canonicalizer/folder move and fold them freely.

The custom builder derives the result type from the input struct, so callers pass only `(value, index)`:

```cpp
void StructAccessOp::build(mlir::OpBuilder &b, mlir::OperationState &state,
                           mlir::Value input, size_t index) {
  // Extract the result type from the input type.
  StructType structTy = llvm::cast<StructType>(input.getType());
  assert(index < structTy.getNumElementTypes());
  mlir::Type resultType = structTy.getElementTypes()[index];

  // Call into the auto-generated build method.
  build(b, state, resultType, input, b.getI64IntegerAttr(index));
}
```

And its verifier enforces both bounds and result-type consistency:

```cpp
llvm::LogicalResult StructAccessOp::verify() {
  StructType structTy = llvm::cast<StructType>(getInput().getType());
  size_t indexValue = getIndex();
  if (indexValue >= structTy.getNumElementTypes())
    return emitOpError()
           << "index should be within the range of the input struct type";
  mlir::Type resultType = getResult().getType();
  if (resultType != structTy.getElementTypes()[indexValue])
    return emitOpError() << "must have the same result type as the struct "
                            "element referred to by the index";
  return mlir::success();
}
```

(That `resultType != structTy.getElementTypes()[indexValue]` comparison is a pointer compare — type uniquing paying off.)

### 5.3 Updating existing ops to accept structs

Structs must flow through returns and calls, so their operand constraints widen from `F64Tensor` to `Toy_Type`:

```tablegen
def ReturnOp : Toy_Op<"return", [Pure, HasParent<"FuncOp">, Terminator]> {
  ...
  let arguments = (ins Variadic<Toy_Type>:$input);   // was Variadic<F64Tensor>
  ...
}

def GenericCallOp : Toy_Op<"generic_call",
    [DeclareOpInterfaceMethods<CallOpInterface>]> {
  ...
  let arguments = (ins
    FlatSymbolRefAttr:$callee,
    Variadic<Toy_Type>:$inputs,                      // was Variadic<F64Tensor>
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );

  // The generic call operation returns a single value of TensorType or
  // StructType.
  let results = (outs Toy_Type);                     // was F64Tensor
  ...
}
```

`toy.constant` also grows `let hasFolder = 1;` in this chapter (it keeps `F64ElementsAttr`/`F64Tensor` — tensor constants and struct constants remain separate ops). Compute ops (`add`, `mul`, `transpose`, `print`, ...) stay tensor-only: you cannot add two structs.

---

## 6. MLIRGen for Structs

All in `/Users/roy/study/mlir/toy/Ch7/mlir/MLIRGen.cpp`.

### 6.1 Tracking struct declarations

MLIRGen needs to map *names* (`"Struct"`, member `"a"`) to MLIR types and member indices, so it keeps the AST around:

```cpp
/// A mapping for named struct types to the underlying MLIR type and the
/// original AST node.
llvm::StringMap<std::pair<mlir::Type, StructAST *>> structMap;
```

The symbol table also changes shape — each variable now remembers its declaration so we can recover its declared *struct* type later:

```cpp
llvm::ScopedHashTable<StringRef, std::pair<mlir::Value, VarDeclExprAST *>>
    symbolTable;
```

Top-level dispatch handles both record kinds; a `StructAST` produces no IR, only a `structMap` entry:

```cpp
/// Create an MLIR type for the given struct.
llvm::LogicalResult mlirGen(StructAST &str) {
  if (structMap.count(str.getName()))
    return emitError(loc(str.loc())) << "error: struct type with name `"
                                     << str.getName() << "' already exists";

  auto variables = str.getVariables();
  std::vector<mlir::Type> elementTypes;
  elementTypes.reserve(variables.size());
  for (auto &variable : variables) {
    if (variable->getInitVal() || !variable->getType().shape.empty())
      return emitError(loc(variable->loc()))
             << "error: variables within a struct definition must not have "
                "initializers";

    mlir::Type type = getType(variable->getType(), variable->loc());
    if (!type)
      return mlir::failure();
    elementTypes.push_back(type);
  }

  structMap.try_emplace(str.getName(), StructType::get(elementTypes), &str);
  return mlir::success();
}
```

Since members have no shape info, each member's type is `tensor<*xf64>` (or a nested struct), so `struct Struct { var a; var b; }` becomes `!toy.struct<tensor<*xf64>, tensor<*xf64>>`.

### 6.2 Resolving types: `getType(VarType, ...)`

Type resolution now checks the name first:

```cpp
/// Build an MLIR type from a Toy AST variable type (forward to the generic
/// getType for non-struct types).
mlir::Type getType(const VarType &type, const Location &location) {
  if (!type.name.empty()) {
    auto it = structMap.find(type.name);
    if (it == structMap.end()) {
      emitError(loc(location))
          << "error: unknown struct type '" << type.name << "'";
      return nullptr;
    }
    return it->second.first;
  }
  return getType(type.shape);   // tensor path, as in earlier chapters
}
```

This is used for function prototypes too — which is how `!toy.struct<...>` ends up in the `toy.func` signature.

### 6.3 Struct literals → `ArrayAttr` + `StructConstantOp`

A struct literal becomes a single constant op whose attribute is built recursively (numbers/tensor literals → `DenseElementsAttr`, nested struct literals → nested `ArrayAttr`):

```cpp
/// Emit a constant for a struct literal. It will be emitted as an array of
/// other literals in an Attribute attached to a `toy.struct_constant`
/// operation. This function returns the generated constant, along with the
/// corresponding struct type.
std::pair<mlir::ArrayAttr, mlir::Type>
getConstantAttr(StructLiteralExprAST &lit) {
  std::vector<mlir::Attribute> attrElements;
  std::vector<mlir::Type> typeElements;

  for (auto &var : lit.getValues()) {
    if (auto *number = llvm::dyn_cast<NumberExprAST>(var.get())) {
      attrElements.push_back(getConstantAttr(*number));
      typeElements.push_back(getType(/*shape=*/{}));
    } else if (auto *lit = llvm::dyn_cast<LiteralExprAST>(var.get())) {
      attrElements.push_back(getConstantAttr(*lit));
      typeElements.push_back(getType(/*shape=*/{}));
    } else {
      auto *structLit = llvm::cast<StructLiteralExprAST>(var.get());
      auto attrTypePair = getConstantAttr(*structLit);
      attrElements.push_back(attrTypePair.first);
      typeElements.push_back(attrTypePair.second);
    }
  }
  mlir::ArrayAttr dataAttr = builder.getArrayAttr(attrElements);
  mlir::Type dataType = StructType::get(typeElements);
  return std::make_pair(dataAttr, dataType);
}

/// Emit a struct literal.
mlir::Value mlirGen(StructLiteralExprAST &lit) {
  mlir::ArrayAttr dataAttr;
  mlir::Type dataType;
  std::tie(dataAttr, dataType) = getConstantAttr(lit);
  return builder.create<StructConstantOp>(loc(lit.loc()), dataType, dataAttr);
}
```

`mlirGen(VarDeclExprAST&)` additionally checks that a struct-typed declaration's initializer type matches the declared struct type exactly (again, a pointer compare thanks to uniquing).

### 6.4 Member access → member index → `StructAccessOp`

Remember: the AST for `value.a` is `BinaryExprAST('.')` with a variable on each side. MLIRGen must turn the *name* `a` into a *number* (its position in the struct). Two helpers do this by walking the AST/declarations:

```cpp
/// Return the struct type that is the result of the given expression, or null
/// if it cannot be inferred.
StructAST *getStructFor(ExprAST *expr) {
  llvm::StringRef structName;
  if (auto *decl = llvm::dyn_cast<VariableExprAST>(expr)) {
    // A plain variable: look up its declaration, take its declared type name.
    auto varIt = symbolTable.lookup(decl->getName());
    if (!varIt.first)
      return nullptr;
    structName = varIt.second->getType().name;
  } else if (auto *access = llvm::dyn_cast<BinaryExprAST>(expr)) {
    if (access->getOp() != '.')
      return nullptr;
    // Nested access `x.y.z`: resolve the parent struct, then find the member.
    auto *name = llvm::dyn_cast<VariableExprAST>(access->getRHS());
    if (!name)
      return nullptr;
    StructAST *parentStruct = getStructFor(access->getLHS());
    ...
    structName = decl->getType().name;
  }
  if (structName.empty())
    return nullptr;
  auto structIt = structMap.find(structName);
  return structIt == structMap.end() ? nullptr : structIt->second.second;
}

/// Return the numeric member index of the given struct access expression.
std::optional<size_t> getMemberIndex(BinaryExprAST &accessOp) {
  assert(accessOp.getOp() == '.' && "expected access operation");
  StructAST *structAST = getStructFor(accessOp.getLHS());
  ...
  auto structVars = structAST->getVariables();
  const auto *it = llvm::find_if(structVars, [&](auto &var) {
    return var->getName() == name->getName();
  });
  if (it == structVars.end())
    return std::nullopt;
  return it - structVars.begin();        // name → position
}
```

and the binary-op codegen intercepts `.` before evaluating the RHS (the RHS is a member *name*, not a value!):

```cpp
mlir::Value mlirGen(BinaryExprAST &binop) {
  mlir::Value lhs = mlirGen(*binop.getLHS());
  if (!lhs)
    return nullptr;
  auto location = loc(binop.loc());

  // If this is an access operation, handle it immediately.
  if (binop.getOp() == '.') {
    std::optional<size_t> accessIndex = getMemberIndex(binop);
    if (!accessIndex) {
      emitError(location, "invalid access into struct expression");
      return nullptr;
    }
    return builder.create<StructAccessOp>(location, lhs, *accessIndex);
  }

  // Otherwise, this is a normal binary op ('+' / '*').
  ...
}
```

Names exist only in the frontend; the IR carries indices. `value.a` → `toy.struct_access %arg0[0]`, `value.b` → `toy.struct_access %arg0[1]`.

---

## 7. Folding & Constant Materialization

With `-opt`, the pipeline (see `toyc.cpp`) runs the **inliner**, then per-function **canonicalizer → shape inference → canonicalizer → CSE**. After inlining, `main` contains a `struct_constant` feeding `struct_access` ops — a chain the folder can collapse completely.

### 7.1 The fold hooks (`FoldAdaptor`)

Setting `let hasFolder = 1;` in ODS generates a declaration:

```cpp
OpFoldResult FooOp::fold(FoldAdaptor adaptor);
```

`FoldAdaptor` is a generated adaptor class that mirrors the op's operands, but where each operand accessor (e.g. `adaptor.getInput()`) returns an **`Attribute`** instead of a `Value`: the constant value of that operand *if* the operand is currently known to be constant, or null otherwise. The returned `OpFoldResult` is either an `Attribute` (constant result) or a `Value` (replace with an existing SSA value); returning null means "cannot fold".

The three implementations live in `/Users/roy/study/mlir/toy/Ch7/mlir/ToyCombine.cpp`:

```cpp
/// Fold constants.
OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) { return getValue(); }

/// Fold struct constants.
OpFoldResult StructConstantOp::fold(FoldAdaptor adaptor) { return getValue(); }

/// Fold simple struct access operations that access into a constant.
OpFoldResult StructAccessOp::fold(FoldAdaptor adaptor) {
  auto structAttr =
      llvm::dyn_cast_if_present<mlir::ArrayAttr>(adaptor.getInput());
  if (!structAttr)
    return nullptr;

  size_t elementIndex = getIndex();
  return structAttr[elementIndex];
}
```

- The two constant ops "fold to themselves" by returning their attribute — this is what feeds constant values into the folding framework (and into other ops' `FoldAdaptor`s).
- `StructAccessOp::fold` is the interesting one: if the input struct is constant (its `ArrayAttr` is visible through the adaptor), the access folds to the *element attribute* at `index`. For `struct_access %cst[0]` where element 0 is a `dense<...>` tensor attribute, the fold result is that `DenseElementsAttr`.

### 7.2 `materializeConstant`: turning attributes back into ops

When a fold returns an `Attribute`, the operation folder must create an op that produces that constant as an SSA value — but *which* op? That's dialect-specific, so MLIR asks the dialect via the hook enabled by `let hasConstantMaterializer = 1;`. In `Dialect.cpp`:

```cpp
mlir::Operation *ToyDialect::materializeConstant(mlir::OpBuilder &builder,
                                                 mlir::Attribute value,
                                                 mlir::Type type,
                                                 mlir::Location loc) {
  if (llvm::isa<StructType>(type))
    return builder.create<StructConstantOp>(loc, type,
                                            llvm::cast<mlir::ArrayAttr>(value));
  return builder.create<ConstantOp>(loc, type,
                                    llvm::cast<mlir::DenseElementsAttr>(value));
}
```

So when `struct_access %cst[0]` folds to a `DenseElementsAttr` with tensor type, the folder calls `materializeConstant(builder, denseAttr, tensorType, loc)` and gets a plain `toy.constant` — the struct is gone from that use. If the accessed member were itself a struct (nested case), the `ArrayAttr` + `StructType` branch materializes a smaller `toy.struct_constant` instead.

### 7.3 What `-opt` actually does to the struct program

Step by step on `struct-codegen.toy`:

1. **Inliner** inlines `multiply_transpose` into `main` (the `ToyInlinerInterface` from Ch4/Ch5 permits it; struct-typed arguments inline like any other value).
2. Inside `main`, `toy.struct_access %0[0]` and `[1]` now read directly from the `toy.struct_constant`.
3. **Canonicalizer** invokes the folders: each `struct_access` folds to the member `DenseElementsAttr`; `materializeConstant` produces `toy.constant` ops; the now-dead `struct_constant` (which is `Pure`) is erased.
4. **CSE** notices both members hold identical data — the two identical `toy.constant`s (and then the two identical `toy.transpose`s) collapse into one.
5. **Shape inference** runs as in Ch4, turning `tensor<*xf64>` into ranked shapes.

Result: **no `!toy.struct`, no `struct_constant`, no `struct_access` remain** — just the same tensor IR Chapter 6 knows how to lower to Affine → LLVM → JIT. This is why Ch7 needs zero new lowering code.

The `struct-opt.mlir` test exercises the nested-struct fold path directly at the IR level:

```mlir
// Input (hand-written IR with a struct nested inside a struct):
toy.func @main() {
  %0 = toy.struct_constant [
    [dense<4.000000e+00> : tensor<2x2xf64>], dense<4.000000e+00> : tensor<2x2xf64>
  ] : !toy.struct<!toy.struct<tensor<*xf64>>, tensor<*xf64>>
  %1 = toy.struct_access %0[0] : !toy.struct<!toy.struct<tensor<*xf64>>, tensor<*xf64>> -> !toy.struct<tensor<*xf64>>
  %2 = toy.struct_access %1[0] : !toy.struct<tensor<*xf64>> -> tensor<*xf64>
  toy.print %2 : tensor<*xf64>
  toy.return
}
```

Real output of `./build/bin/toyc-ch7 ../test_Example/Toy/Ch7/struct-opt.mlir -emit=mlir -opt` (run from `toy/`):

```mlir
module {
  toy.func @main() {
    %0 = toy.constant dense<4.000000e+00> : tensor<2x2xf64>
    toy.print %0 : tensor<2x2xf64>
    toy.return
  }
}
```

Two chained accesses through a nested struct fold all the way down to one tensor constant.

---

## 8. Building — Out-of-Tree CMake Specifics

Unlike the upstream tutorial (which builds inside `llvm-project/mlir/examples`), this repo builds Ch7 **out-of-tree** against an installed Homebrew MLIR, as one chapter of a **CMake superbuild** rooted at `/Users/roy/study/mlir/toy/`:

- Repo root: `/Users/roy/study/mlir`; superbuild: `/Users/roy/study/mlir/toy/`; chapter sources: `/Users/roy/study/mlir/toy/Ch7/`
- `MLIR_DIR=/opt/homebrew/opt/llvm@20/lib/cmake/mlir` (set by the CMake preset)
- Compiler: `/opt/homebrew/opt/llvm@20/bin/clang++` (set by the preset)
- Generator: Ninja, on macOS (Darwin)

### 8.1 The superbuild

The top-level `/Users/roy/study/mlir/toy/CMakeLists.txt` does the LLVM/MLIR boilerplate **once** for all chapters — `find_package(MLIR/LLVM)`, `include(TableGen)`, `include(AddLLVM)`, `include(AddMLIR)`, `include(HandleLLVMOptions)`, the include directories — then sets `CMAKE_RUNTIME_OUTPUT_DIRECTORY` to `build/bin/` and pulls in every chapter with `add_subdirectory(Ch1)` … `add_subdirectory(Ch7)`. All binaries land in `toy/build/bin/toyc-ch{1..7}`.

Compiler, generator, and `MLIR_DIR`/`LLVM_DIR` come from `/Users/roy/study/mlir/toy/CMakePresets.json` (preset `default`: Ninja, Release, Homebrew llvm@20), so building is just:

```bash
cd /Users/roy/study/mlir/toy
./build.sh ch7          # configure once (cmake --preset default) + build only toyc-ch7
./build.sh              # build everything (all chapters)
./build.sh ch7 --fresh  # wipe build/ first, then rebuild
```

`build.sh` is **incremental**: it configures only when `build/CMakeCache.txt` is missing (Ninja re-runs CMake automatically when a `CMakeLists.txt` changes) and otherwise just runs `cmake --build --preset default [--target toyc-ch7]`. Only `--fresh` does an `rm -rf build`.

### 8.2 The dual-mode chapter CMakeLists

`/Users/roy/study/mlir/toy/Ch7/CMakeLists.txt` still works **standalone** too. Its boilerplate is wrapped in a guard that only fires when the chapter is the top-level project:

```cmake
# Standalone-mode boilerplate.
# Runs only when this chapter is configured directly (cmake -S Ch7).
# In the superbuild (cmake -S toy/), ../CMakeLists.txt already did all this.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(toy-ch7)

  find_package(MLIR REQUIRED CONFIG)   # via MLIR_DIR from the CMake cache/env
  find_package(LLVM REQUIRED CONFIG)

  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
  include(TableGen)                    # defines mlir_tablegen()
  include(AddLLVM)
  include(AddMLIR)
  include(HandleLLVMOptions)

  include_directories(${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})
endif()
```

So a standalone configure of just this chapter is still possible:

```bash
cd /Users/roy/study/mlir/toy
cmake -S Ch7 -B Ch7/build -G Ninja \
      -DMLIR_DIR=/opt/homebrew/opt/llvm@20/lib/cmake/mlir
cmake --build Ch7/build      # → Ch7/build/toyc-ch7
```

(`run.sh` even falls back to `Ch7/build/toyc-ch7` if the superbuild binary is absent.)

The chapter targets below the guard are unchanged from a per-chapter build:

```cmake
# This chapter depends on JIT support enabled.
if(NOT MLIR_ENABLE_EXECUTION_ENGINE)
  return()
endif()

include_directories(include)
add_subdirectory(include)               # runs TableGen on Ops.td + interfaces

set(LLVM_TARGET_DEFINITIONS mlir/ToyCombine.td)
mlir_tablegen(ToyCombine.inc -gen-rewriters)
add_public_tablegen_target(ToyCh7CombineIncGen)

add_executable(toyc-ch7
  toyc.cpp
  parser/AST.cpp
  mlir/MLIRGen.cpp
  mlir/Dialect.cpp
  mlir/LowerToAffineLoops.cpp
  mlir/LowerToLLVM.cpp
  mlir/ShapeInferencePass.cpp
  mlir/ToyCombine.cpp
  )

add_dependencies(toyc-ch7 ToyCh7ShapeInferenceInterfaceIncGen)
add_dependencies(toyc-ch7 ToyCh7OpsIncGen)
add_dependencies(toyc-ch7 ToyCh7CombineIncGen)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)

# NOTE: link ONLY shared MLIR/LLVM libraries here. Mixing static .a archives
# with libMLIR.dylib causes TypeID duplication and runtime segfaults — see
# ../Ch6/MLIR_LINKING_PITFALL.md.
target_link_libraries(toyc-ch7
  PRIVATE
    MLIR                         # libMLIR.dylib (all dialects, passes, conversions)
    MLIRExecutionEngineShared    # libMLIRExecutionEngineShared.dylib (JIT support)
    )
```

Notes:

- `include/toy/CMakeLists.txt` runs the TableGen steps for `Ops.td` (`-gen-op-decls/-gen-op-defs/-gen-dialect-decls/-gen-dialect-defs`) and the shape-inference interface, producing `Ops.h.inc`, `Ops.cpp.inc`, `Dialect.h.inc`, `Dialect.cpp.inc` under the chapter's binary dir (`build/Ch7/include/toy/` in the superbuild).
- There is **no TableGen step for the struct type** — `StructType` is entirely hand-written C++ in this chapter (later MLIR practice would use ODS `TypeDef`, but the point of Ch7 is to show the raw storage mechanism).
- Linking against the monolithic `MLIR` dylib plus `MLIRExecutionEngineShared` keeps the out-of-tree link line trivial compared to enumerating dozens of static component libraries.
- The file set is Ch6's plus nothing new: struct support lives in the already-existing files.

---

## 9. Running and Testing

Test inputs live in `/Users/roy/study/mlir/test_Example/Toy/Ch7/` (`struct-ast.toy`, `struct-codegen.toy`, `struct-opt.mlir`, `jit.toy`, plus the tensor tests carried over from earlier chapters). All commands below are run from the superbuild root `/Users/roy/study/mlir/toy/` — which is also where `./run.sh ch7` runs them — so the binary is `./build/bin/toyc-ch7` and the tests sit at `../test_Example/Toy/Ch7/`.

### 9.1 The input program (`struct-codegen.toy`)

```
struct Struct {
  var a;
  var b;
}

# User defined generic function may operate on struct types as well.
def multiply_transpose(Struct value) {
  # We can access the elements of a struct via the '.' operator.
  return transpose(value.a) * transpose(value.b);
}

def main() {
  # We initialize struct values using a composite initializer.
  Struct value = {[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]};

  # We pass these arguments to functions like we do with variables.
  var c = multiply_transpose(value);
  print(c);
}
```

### 9.2 `-emit=mlir`: structs in the IR

`cd /Users/roy/study/mlir/toy && ./run.sh ch7` (equivalently, from `toy/`: `./build/bin/toyc-ch7 ../test_Example/Toy/Ch7/struct-codegen.toy -emit=mlir`) — real output:

```mlir
module {
  toy.func private @multiply_transpose(%arg0: !toy.struct<tensor<*xf64>, tensor<*xf64>>) -> tensor<*xf64> {
    %0 = toy.struct_access %arg0[0] : !toy.struct<tensor<*xf64>, tensor<*xf64>> -> tensor<*xf64>
    %1 = toy.transpose(%0 : tensor<*xf64>) to tensor<*xf64>
    %2 = toy.struct_access %arg0[1] : !toy.struct<tensor<*xf64>, tensor<*xf64>> -> tensor<*xf64>
    %3 = toy.transpose(%2 : tensor<*xf64>) to tensor<*xf64>
    %4 = toy.mul %1, %3 : tensor<*xf64>
    toy.return %4 : tensor<*xf64>
  }
  toy.func @main() {
    %0 = toy.struct_constant [dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>, dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>] : !toy.struct<tensor<*xf64>, tensor<*xf64>>
    %1 = toy.generic_call @multiply_transpose(%0) : (!toy.struct<tensor<*xf64>, tensor<*xf64>>) -> tensor<*xf64>
    toy.print %1 : tensor<*xf64>
    toy.return
  }
}
```

Reading it against the sections above:

- **Function signature** — `%arg0: !toy.struct<tensor<*xf64>, tensor<*xf64>>`: `getType(VarType)` resolved the name `Struct` via `structMap` (§6.2), and the custom `printType` produced the `!toy.struct<...>` syntax (§4.2). Members declared without shapes print as unranked `tensor<*xf64>`.
- **`toy.struct_access %arg0[0]` / `[1]`** — `value.a`/`value.b` lowered from names to indices by `getMemberIndex` (§6.4); the `-> tensor<*xf64>` result type came from the custom builder (§5.2).
- **`toy.struct_constant [dense<...>, dense<...>]`** — the struct literal packed into one `ArrayAttr` with two `DenseElementsAttr` members (§6.3).
- **`toy.generic_call`** with a struct operand — legal only because `GenericCallOp`'s inputs are `Variadic<Toy_Type>` (§5.3).

### 9.3 `-emit=mlir -opt`: the struct disappears

`./build/bin/toyc-ch7 ../test_Example/Toy/Ch7/struct-codegen.toy -emit=mlir -opt` — real output:

```mlir
module {
  toy.func @main() {
    %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
    %1 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
    %2 = toy.mul %1, %1 : tensor<3x2xf64>
    toy.print %2 : tensor<3x2xf64>
    toy.return
  }
}
```

Compare the shapes of the two dumps:

| Before `-opt` | After `-opt` |
|---|---|
| 2 functions | 1 function (`multiply_transpose` inlined and removed) |
| 1 `struct_constant`, 2 `struct_access` | **zero struct ops, zero `!toy.struct` types** |
| 2 loads of identical data | 1 `toy.constant` (CSE merged them) |
| 2 transposes | 1 `toy.transpose` (CSE), used twice by `toy.mul %1, %1` |
| everything `tensor<*xf64>` | ranked `tensor<2x3xf64>` / `tensor<3x2xf64>` (shape inference) |

This is exactly the inline → fold(`StructAccessOp::fold` + `materializeConstant`) → CSE → shape-inference story from §7.3 — and it matches the final example in the official docs.

### 9.4 `-emit=jit`: the unchanged backend still works

Because the optimized module is pure tensor IR, the whole Ch6 pipeline (Affine → LLVM dialect → LLVM IR → ExecutionEngine) runs unmodified. Real output of `./build/bin/toyc-ch7 ../test_Example/Toy/Ch7/jit.toy -emit=jit`:

```
1.000000 2.000000
3.000000 4.000000
```

(`jit.toy` is a struct-free smoke test; `struct-codegen.toy` also JITs once `-opt` has eliminated the structs, since `print` of a struct-derived tensor becomes an ordinary tensor print.)

### 9.5 Other test files

- **`struct-ast.toy`** + `-emit=ast` — verifies the parser/AST additions (`Struct:` record, `BinOp: .`, `VarDecl value<Struct>`, `Struct Literal:`); output shown in §2.5.
- **`struct-opt.mlir`** + `-emit=mlir -opt` — round-trips the custom type syntax through `parseType` (it is hand-written `.mlir`, so the *parser* is exercised, not MLIRGen) and checks nested-struct folding; output shown in §7.3.
- The remaining files (`codegen.toy`, `ast.toy`, `affine-lowering.mlir`, `llvm-lowering.mlir`, `shape_inference.mlir`, `transpose_transpose.toy`, `trivial_reshape.toy`, `scalar.toy`, `empty.toy`, `invalid.mlir`) are the Ch1–Ch6 regression suite, proving the struct work didn't break anything.

---

## 10. Key Takeaways & Pitfalls

**Takeaways**

1. **Types are uniqued value objects.** A custom type = a `TypeStorage` subclass (`KeyTy`, `operator==`, `construct`) + a `TypeBase` wrapper + `addTypes<>()` in `initialize()`. Equality, hashing, and copying then come for free as pointer operations.
2. **The storage owns its memory.** `construct()` must copy everything through the `TypeStorageAllocator` (`allocator.copyInto(key)`) because types live as long as the `MLIRContext`, while the caller's `ArrayRef` may point at a dead stack frame.
3. **`hashKey`/`getKey` are optional** when `KeyTy` is `llvm::hash_value`-hashable and directly constructible from the `get()` arguments — the tutorial writes them anyway for pedagogy.
4. **Dialect hooks give the type syntax.** `parseType`/`printType` (declared via `useDefaultTypePrinterParser = 1`) handle everything after `!toy.`, and the parser doubles as a structural validator with real diagnostics.
5. **Bridge C++ types into ODS with `DialectType` + `CPred`,** then compose constraints (`AnyTypeOf<[F64Tensor, Toy_StructType]>`) and thread them through existing ops.
6. **`ConstantLike` + `hasFolder` + `materializeConstant` is the constant-propagation triad.** Fold hooks compute attribute results; the dialect materializer decides which op re-embodies an attribute of a given type.
7. **Design types to disappear.** No lowering was written for `StructType`; inlining plus folding erases it before the lowering pipeline runs. High-level abstractions that fold away are often cheaper than ones you must lower.

**Pitfalls**

- **Forgetting `allocator.copyInto(key)`** in `construct()` compiles fine and then reads freed memory — the classic bug in hand-written storage classes.
- **Forgetting `addTypes<StructType>()`**: `StructType::get` will assert/fail at runtime because the context has no registered storage uniquer entry for it.
- **`static constexpr StringLiteral name = "toy.struct";` is required** on hand-written types with MLIR ≈17+ (including the LLVM 20 used here); without it, `TypeBase` fails to compile or bytecode/debug utilities break. Older copies of the tutorial code omit it.
- **`FoldAdaptor` operand accessors return null** when the operand isn't constant — always `dyn_cast_if_present`, never blind `cast` (see `StructAccessOp::fold`).
- **Fold results must not create ops** — return an `Attribute` or existing `Value` and let `materializeConstant` build ops; creating ops inside `fold()` is unsupported.
- **`.` in MLIRGen must not evaluate its RHS**: the RHS of an access is a member *name*, not an expression. Evaluating it would emit a bogus "unknown variable" error — hence the early intercept in `mlirGen(BinaryExprAST&)`.
- **Verifier ordering matters**: `StructAccessOp::verify` bounds-checks the index *before* indexing `getElementTypes()`; also remember ODS constraints (`Toy_StructType` operand) are verified before your custom `verify()` runs, so `llvm::cast<StructType>` inside it is safe.
- **Out-of-tree specifics**: this chapter needs the execution engine (`MLIR_ENABLE_EXECUTION_ENGINE`) and TableGen'd `.inc` files under the chapter's binary dir (`build/Ch7/include` in the superbuild), so keep the `add_dependencies(... ToyCh7OpsIncGen ...)` lines — Ninja's parallelism will otherwise race compilation against TableGen.

---

## Links

- Official doc: [MLIR Toy Tutorial, Chapter 7 — Adding a Composite Type to Toy](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-7/)
- Previous chapter: [Chapter 6: Lowering to LLVM & JIT](6_lowering_to_llvm_jit.md)
- Back to: [README](README.md)
- Code referenced in this chapter:
  - `/Users/roy/study/mlir/toy/Ch7/include/toy/Lexer.h`, `AST.h`, `Parser.h` — language front-end
  - `/Users/roy/study/mlir/toy/Ch7/include/toy/Dialect.h`, `/Users/roy/study/mlir/toy/Ch7/mlir/Dialect.cpp` — `StructType`, storage, parse/print, verifiers, `materializeConstant`
  - `/Users/roy/study/mlir/toy/Ch7/include/toy/Ops.td` — ODS: `Toy_StructType`, `Toy_Type`, `StructConstantOp`, `StructAccessOp`
  - `/Users/roy/study/mlir/toy/Ch7/mlir/ToyCombine.cpp` — fold implementations
  - `/Users/roy/study/mlir/toy/Ch7/mlir/MLIRGen.cpp` — struct codegen
  - `/Users/roy/study/mlir/test_Example/Toy/Ch7/` — `struct-ast.toy`, `struct-codegen.toy`, `struct-opt.mlir`, `jit.toy`
