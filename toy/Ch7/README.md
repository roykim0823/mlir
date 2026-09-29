# Chapter 7: Adding a Composite Type to Toy

> Goal: extend the Toy language and dialect with a user-defined `struct` type — touching every layer of the compiler from the lexer down to constant folding — following the official tutorial: [Toy Tutorial Ch-7: Adding a Composite Type to Toy](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-7/).

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

### How this README maps to the upstream chapter

Every topic of the official [Ch-7](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-7/) is covered here, in the same order, with this repo's real code and output in place of upstream's snippets. Where upstream's code no longer matches MLIR 20 (member `isa`/`cast`, unprefixed accessors, the missing `name` member, ...), the section says so and shows the form that was checked against the llvm@20 tools and headers:

| Upstream section | Here |
|---|---|
| Introduction | 1 |
| Defining a `struct` in Toy | 2.1 (the frontend behind it: 2.2–2.5) |
| Defining a `struct` in MLIR → Defining the Type Class (storage class, type class, registration) | 3.1–3.2 |
| Exposing to ODS | 4 |
| Parsing and Printing (grammar, parsing, printing, first example) | 5, 5.1–5.2 |
| Operating on `StructType` → Updating Existing Operations | 6.1 |
| Adding New `Toy` Operations (`toy.struct_constant`, `toy.struct_access`) | 6.2, 6.3 |
| ... and the full MLIR module of the original example | 7.4 |
| Optimizing Operations on `StructType` (after inlining, folders, `materializeConstant`, final IR) | 8.1–8.4 |
| "You can build `toyc-ch7` and try yourself" | 9 |

Beyond upstream: the lexer, AST, parser and dumper changes (2.2–2.5), the verifiers and custom builder of the new ops (6.2, 6.3), the MLIRGen walkthrough (7.1–7.3), and in Build and Run the pass-by-pass view, nested structs, the JIT, error paths and the ecosystem view (9.5–9.9).

### Where Chapter 7 lives in this repo

The superbuild at `toy/`, the pinned Homebrew LLVM/MLIR 20 toolchain, and the `build.sh`/`run.sh` helpers are documented once in the top-level [README](../README.md#repository-layout). This chapter also remains configurable as a standalone project (see section 9.1). Chapter 7 specifics:

| Item | Location / value |
|---|---|
| Chapter 7 code | `/Users/roy/study/mlir/toy/Ch7/` |
| Build | `cd toy && ./build.sh ch7` → binary at `./build/bin/toyc-ch7` |
| Run | `cd toy && ./run.sh ch7` → `./build/bin/toyc-ch7 ../test_Example/Toy/Ch7/struct-codegen.toy -emit=mlir` |
| Test inputs | `/Users/roy/study/mlir/test_Example/Toy/Ch7/` (`struct-ast.toy`, `struct-codegen.toy`, `struct-opt.mlir`, `jit.toy`, plus the Ch1–Ch6 regression files) |

All struct support lives in files that already existed in Chapter 6 — there is no new source file:

```text
Ch7/
├── CMakeLists.txt              # same targets as Ch6 (see the appendix)
├── toyc.cpp                    # driver: unchanged pipeline, one extra canonicalizer
├── parser/
│   └── AST.cpp                 # dumper: prints Struct records and struct literals
├── include/toy/
│   ├── Lexer.h                 # + tok_struct
│   ├── AST.h                   # + RecordAST, StructAST, StructLiteralExprAST, VarType::name
│   ├── Parser.h                # + parseStruct, typed declarations, struct literals, '.'
│   ├── Dialect.h               # + class StructType
│   └── Ops.td                  # + Toy_StructType, Toy_Type, struct_constant, struct_access
└── mlir/
    ├── Dialect.cpp             # + StructTypeStorage, parseType/printType, materializeConstant
    ├── MLIRGen.cpp             # + structMap, struct literals, member access
    └── ToyCombine.cpp          # + fold() hooks
```

---

## 2. Defining a `struct` in Toy

The first thing to define is the interface of the type in the Toy *source* language. Upstream gives the general syntax like this (quoted from the upstream chapter):

```text
# A struct is defined by using the `struct` keyword followed by a name.
struct MyStruct {
  # Inside of the struct is a list of variable declarations without initializers
  # or shapes, which may also be other previously defined structs.
  var a;
  var b;
}
```

Structs may then be used in functions as variables or parameters by writing the struct's name instead of `var`. Members are accessed with the `.` operator, and struct values are initialized with a *composite initializer*: a comma-separated list of other initializers surrounded by `{}`.

### 2.1 The struct program used throughout this chapter

Upstream's example program is the chapter's main test input:

***test_Example/Toy/Ch7/struct-codegen.toy***
```text
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

Four new language features appear here:

- **Struct definitions** — `struct Struct { var a; var b; }` is a new kind of top-level record next to `def`. Members are declared with `var` and have no shapes and no initializers.
- **Struct-typed declarations** — `Struct value = ...` uses the struct's name where `var` used to go; function parameters can be typed the same way (`def multiply_transpose(Struct value)`).
- **Struct literals** — `{..., ...}` is a composite initializer with one sub-literal (a tensor literal, a number, or a nested struct literal) per member, in declaration order.
- **Member access** — `value.a` selects a member by name.

Structs are *compile-time aggregates*: there is no way to mutate a member or build a struct from runtime values. That restriction is exactly what lets the optimizer fold them away completely (section 8).

A member can itself be struct-typed (`Inner i;` inside `struct Outer`), which nests structs; section 9.6 compiles such a program. Upstream only states the syntax and moves on; the rest of this section shows how the frontend implements it.

### 2.2 Lexer: one new token

`include/toy/Lexer.h` adds a single keyword token, `tok_struct = -5`; the primary tokens after it shift down by one compared to Chapter 6 (`tok_identifier` was `-5`, `tok_number` was `-6`):

***include/toy/Lexer.h***
```cpp
enum Token : int {
  ...
  // commands
  tok_return = -2,
  tok_var = -3,
  tok_def = -4,
  tok_struct = -5,

  // primary
  tok_identifier = -6,
  tok_number = -7,
};
```

and `getTok()` recognizes the keyword alongside the others:

***include/toy/Lexer.h***
```cpp
      if (identifierStr == "return")
        return tok_return;
      if (identifierStr == "def")
        return tok_def;
      if (identifierStr == "struct")
        return tok_struct;
      if (identifierStr == "var")
        return tok_var;
      return tok_identifier;
```

That is the *entire* lexer change. `.`, `{`, `}` were already single-character tokens.

### 2.3 AST: records, struct definitions, struct literals

`include/toy/AST.h` makes three structural changes. The first two are near the top of the file (abridged):

***include/toy/AST.h***
```cpp
/// A variable type with either name or shape information.
struct VarType {
  std::string name;
  std::vector<int64_t> shape;
};

/// Base class for all expression nodes.
class ExprAST {
public:
  enum ExprASTKind {
    Expr_VarDecl,
    Expr_Return,
    Expr_Num,
    Expr_Literal,
    Expr_StructLiteral,
    Expr_Var,
    Expr_BinOp,
    Expr_Call,
    Expr_Print,
  };
  ...
};
...
/// Expression class for a literal struct value.
class StructLiteralExprAST : public ExprAST {
  std::vector<std::unique_ptr<ExprAST>> values;

public:
  StructLiteralExprAST(Location loc,
                       std::vector<std::unique_ptr<ExprAST>> values)
      : ExprAST(Expr_StructLiteral, std::move(loc)), values(std::move(values)) {
  }

  llvm::ArrayRef<std::unique_ptr<ExprAST>> getValues() { return values; }

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) {
    return c->getKind() == Expr_StructLiteral;
  }
};
```

- **`VarType` gains a `name`.** Previously a variable's type was just an optional shape. Now a non-empty `name` means a *named* struct type; otherwise `shape` is the tensor shape as before (possibly empty).
- **A struct literal expression.** A new `ExprASTKind` value `Expr_StructLiteral` and `StructLiteralExprAST`, holding one entry per member (a tensor literal, a number, or a nested struct literal).

The third change is at the bottom of the file: a module used to be a list of functions; now it is a list of *records*, each either a function or a struct definition, distinguished with LLVM-style RTTI via `getKind()` (abridged):

***include/toy/AST.h***
```cpp
/// This class represents a top level record in a module.
class RecordAST {
public:
  enum RecordASTKind {
    Record_Function,
    Record_Struct,
  };

  RecordAST(RecordASTKind kind) : kind(kind) {}
  virtual ~RecordAST() = default;

  RecordASTKind getKind() const { return kind; }

private:
  const RecordASTKind kind;
};

/// This class represents a function definition itself.
class FunctionAST : public RecordAST {
  ...
};

/// This class represents a struct definition.
class StructAST : public RecordAST {
  Location location;
  std::string name;
  std::vector<std::unique_ptr<VarDeclExprAST>> variables;

public:
  StructAST(Location location, const std::string &name,
            std::vector<std::unique_ptr<VarDeclExprAST>> variables)
      : RecordAST(Record_Struct), location(std::move(location)), name(name),
        variables(std::move(variables)) {}

  const Location &loc() { return location; }
  llvm::StringRef getName() const { return name; }
  llvm::ArrayRef<std::unique_ptr<VarDeclExprAST>> getVariables() {
    return variables;
  }

  /// LLVM style RTTI
  static bool classof(const RecordAST *r) {
    return r->getKind() == Record_Struct;
  }
};

/// This class represents a list of functions to be processed together
class ModuleAST {
  std::vector<std::unique_ptr<RecordAST>> records;

public:
  ModuleAST(std::vector<std::unique_ptr<RecordAST>> records)
      : records(std::move(records)) {}

  auto begin() { return records.begin(); }
  auto end() { return records.end(); }
};
```

Note there is **no dedicated AST node for member access** — `value.a` is parsed as an ordinary `BinaryExprAST` with `op == '.'` and a `VariableExprAST` (`a`) on the RHS. MLIRGen interprets it specially (section 7.3).

### 2.4 Parser changes

All in `include/toy/Parser.h`. The grammar additions, as implemented by the functions below:

```
# A struct definition is a new kind of top-level record:
struct-definition ::= `struct` identifier `{` (decl `;`)+ `}`

# Declarations: tensor-typed with `var`, or struct-typed with the struct name:
decl ::= `var` identifier [ type ] (`=` expr)?
decl ::= identifier identifier (`=` expr)?

# Prototype parameters may carry a struct type name:
param ::= identifier | identifier identifier

# Struct literal: nested tensor literals / numbers / struct literals in braces:
struct-literal ::= `{` (struct-literal | tensor-literal | number) (`,` ...)* `}`

# Member access is the '.' binary operator (highest precedence):
access ::= expr `.` identifier
```

**Top-level dispatch** (`parseModule`) now accepts both records:

***include/toy/Parser.h***
```cpp
    // Parse functions and structs one at a time and accumulate in this vector.
    std::vector<std::unique_ptr<RecordAST>> records;
    while (true) {
      std::unique_ptr<RecordAST> record;
      switch (lexer.getCurToken()) {
      case tok_eof:
        break;
      case tok_def:
        record = parseDefinition();
        break;
      case tok_struct:
        record = parseStruct();
        break;
      default:
        return parseError<ModuleAST>("'def' or 'struct'",
                                     "when parsing top level module records");
      }
      if (!record)
        break;
      records.push_back(std::move(record));
    }
```

**Struct definitions** — each member is an ordinary declaration parsed with `requiresInitializer=false` and terminated by `;`:

***include/toy/Parser.h***
```cpp
  /// Parse a struct definition, we expect a struct initiated with the
  /// `struct` keyword, followed by a block containing a list of variable
  /// declarations.
  ///
  /// definition ::= `struct` identifier `{` decl+ `}`
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

    // Parse: decl+
    std::vector<std::unique_ptr<VarDeclExprAST>> decls;
    do {
      auto decl = parseDeclaration(/*requiresInitializer=*/false);
      if (!decl)
        return nullptr;
      decls.push_back(std::move(decl));

      if (lexer.getCurToken() != ';')
        return parseError<StructAST>(";",
                                     "after variable in struct definition");
      lexer.consume(Token(';'));
    } while (lexer.getCurToken() != '}');

    // Parse: '}'
    lexer.consume(Token('}'));
    return std::make_unique<StructAST>(loc, name, std::move(decls));
  }
```

The parser itself does not stop a member from having a shape (`var a<2>;`); MLIRGen rejects it (section 7.1).

**Struct-typed declarations.** Inside a function body, an identifier at the start of a statement can begin either a call *or* a typed declaration, so `parseBlock` hands it to `parseDeclarationOrCallExpr`, which disambiguates on the next token:

***include/toy/Parser.h***
```cpp
      if (lexer.getCurToken() == tok_identifier) {
        // Variable declaration or call
        auto expr = parseDeclarationOrCallExpr();
        if (!expr)
          return nullptr;
        exprList->push_back(std::move(expr));
      } else if (lexer.getCurToken() == tok_var) {
```

The typed-declaration helpers sit together further up the file:

***include/toy/Parser.h***
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

  /// Parse a typed variable declaration.
  std::unique_ptr<VarDeclExprAST>
  parseTypedDeclaration(llvm::StringRef typeName, bool requiresInitializer,
                        const Location &loc) {
    // Parse the variable name.
    if (lexer.getCurToken() != tok_identifier)
      return parseError<VarDeclExprAST>("name", "in variable declaration");
    std::string id(lexer.getId());
    lexer.getNextToken(); // eat id

    // Parse the initializer.
    std::unique_ptr<ExprAST> expr;
    if (requiresInitializer) {
      if (lexer.getCurToken() != '=')
        return parseError<VarDeclExprAST>("initializer",
                                          "in variable declaration");
      lexer.consume(Token('='));
      expr = parseExpression();
    }

    VarType type;
    type.name = std::string(typeName);
    return std::make_unique<VarDeclExprAST>(loc, std::move(id), std::move(type),
                                            std::move(expr));
  }

  /// Parse a variable declaration, for either a tensor value or a struct value,
  /// with an optionally required initializer.
  /// decl ::= var identifier [ type ] (= expr)?
  /// decl ::= identifier identifier (= expr)?
  std::unique_ptr<VarDeclExprAST> parseDeclaration(bool requiresInitializer) {
    // Check to see if this is a 'var' declaration.
    if (lexer.getCurToken() == tok_var)
      return parseVarDeclaration(requiresInitializer);

    // Parse the type name.
    if (lexer.getCurToken() != tok_identifier)
      return parseError<VarDeclExprAST>("type name", "in variable declaration");
    auto loc = lexer.getLastLocation();
    std::string typeName(lexer.getId());
    lexer.getNextToken(); // eat id

    // Parse the rest of the declaration.
    return parseTypedDeclaration(typeName, requiresInitializer, loc);
  }
```

`parseTypedDeclaration` stores the type name into `VarType::name`; `parseDeclaration` is the entry point used both for `var` statements and for struct members.

**Typed prototype parameters.** Function prototypes get the same treatment: in `def foo(Struct value)`, if the token after the first identifier is another identifier, the first one was a type name (abridged from `parsePrototype`):

***include/toy/Parser.h***
```cpp
      do {
        VarType type;
        std::string name;

        // Parse either the name of the variable, or its type.
        std::string nameOrType(lexer.getId());
        auto loc = lexer.getLastLocation();
        lexer.consume(tok_identifier);

        // If the next token is an identifier, we just parsed the type.
        if (lexer.getCurToken() == tok_identifier) {
          type.name = std::move(nameOrType);

          // Parse the name.
          name = std::string(lexer.getId());
          lexer.consume(tok_identifier);
        } else {
          // Otherwise, we just parsed the name.
          name = std::move(nameOrType);
        }

        args.push_back(
            std::make_unique<VarDeclExprAST>(std::move(loc), name, type));
        ...
      } while (true);
```

**Struct literals** — each element is a tensor literal, a number, or (recursively) another struct literal:

***include/toy/Parser.h***
```cpp
  /// Parse a literal struct expression.
  /// structLiteral ::= { (structLiteral | tensorLiteral)+ }
  std::unique_ptr<ExprAST> parseStructLiteralExpr() {
    auto loc = lexer.getLastLocation();
    lexer.consume(Token('{'));

    // Hold the list of values.
    std::vector<std::unique_ptr<ExprAST>> values;
    do {
      // We can have either another nested array or a number literal.
      if (lexer.getCurToken() == '[') {
        values.push_back(parseTensorLiteralExpr());
        if (!values.back())
          return nullptr;
      } else if (lexer.getCurToken() == tok_number) {
        values.push_back(parseNumberExpr());
        if (!values.back())
          return nullptr;
      } else {
        if (lexer.getCurToken() != '{')
          return parseError<ExprAST>("{, [, or number",
                                     "in struct literal expression");
        values.push_back(parseStructLiteralExpr());
      }

      // End of this list on '}'
      if (lexer.getCurToken() == '}')
        break;

      // Elements are separated by a comma.
      if (lexer.getCurToken() != ',')
        return parseError<ExprAST>("} or ,", "in struct literal expression");

      lexer.getNextToken(); // eat ,
    } while (true);
    if (values.empty())
      return parseError<ExprAST>("<something>",
                                 "to fill struct literal expression");
    lexer.getNextToken(); // eat }

    return std::make_unique<StructLiteralExprAST>(std::move(loc),
                                                  std::move(values));
  }
```

(The doc comment's grammar omits the `number` alternative that the code accepts.) `parsePrimary()` dispatches `'{'` to it:

***include/toy/Parser.h***
```cpp
    case '[':
      return parseTensorLiteralExpr();
    case '{':
      return parseStructLiteralExpr();
```

**Member access as an operator.** `.` is simply registered as the tightest-binding binary operator in `getTokPrecedence()`:

***include/toy/Parser.h***
```cpp
  /// Get the precedence of the pending binary operator token.
  int getTokPrecedence() {
    if (!isascii(lexer.getCurToken()))
      return -1;

    // 1 is lowest precedence.
    switch (static_cast<char>(lexer.getCurToken())) {
    case '-':
      return 20;
    case '+':
      return 20;
    case '*':
      return 40;
    case '.':
      return 60;
    default:
      return -1;
    }
  }
```

So `transpose(value.a) * transpose(value.b)` parses with `.` bound before `*`, no new expression machinery needed.

### 2.5 The AST dumper

`parser/AST.cpp` learns to print the new nodes: `dump(ModuleAST*)` now switches on the record kind, and two new overloads print struct definitions and struct literals. The struct-literal one:

***parser/AST.cpp***
```cpp
/// Print a struct literal.
void ASTDumper::dump(StructLiteralExprAST *node) {
  INDENT();
  llvm::errs() << "Struct Literal: ";
  for (auto &value : node->getValues())
    dump(value.get());
  indent();
  llvm::errs() << " " << loc(node) << "\n";
}
```

`dump(const VarType&)` prints the struct name instead of the shape when there is one:

***parser/AST.cpp***
```cpp
/// Print type: only the shape is printed in between '<' and '>'
void ASTDumper::dump(const VarType &type) {
  llvm::errs() << "<";
  if (!type.name.empty())
    llvm::errs() << type.name;
  else
    llvm::interleaveComma(type.shape, llvm::errs());
  llvm::errs() << ">";
}
```

The test input `struct-ast.toy` is the program of section 2.1 behind a `RUN:` header, so its struct-typed declaration is on line 16:

***test_Example/Toy/Ch7/struct-ast.toy***
```text
def main() {
  # We initialize struct values using a composite initializer.
  Struct value = {[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]};
```

Dump its AST and keep only the lines for that declaration:

```bash
cd /Users/roy/study/mlir/toy/Ch7
./build/toyc-ch7 ../../test_Example/Toy/Ch7/struct-ast.toy -emit=ast 2>&1 | sed -n '/VarDecl value/,/16:18/p'
```

Real output (section 9.3 shows the whole dump):

```text
        VarDecl value<Struct> @../../test_Example/Toy/Ch7/struct-ast.toy:16:3
          Struct Literal:             Literal: <2, 3>[ <3>[ 1.000000e+00, 2.000000e+00, 3.000000e+00], <3>[ 4.000000e+00, 5.000000e+00, 6.000000e+00]] @../../test_Example/Toy/Ch7/struct-ast.toy:16:19
            Literal: <2, 3>[ <3>[ 1.000000e+00, 2.000000e+00, 3.000000e+00], <3>[ 4.000000e+00, 5.000000e+00, 6.000000e+00]] @../../test_Example/Toy/Ch7/struct-ast.toy:16:43
           @../../test_Example/Toy/Ch7/struct-ast.toy:16:18
```

- **`value<Struct>`**: `VarType::name` is set, so the dumper prints the name where a tensor declaration prints its shape.
- **The odd `Struct Literal:` line**: no newline is printed after the header, so the first member's own indentation lands on the same line.
- **The literal's location comes last**: `16:18` (the `{`) is printed after an `indent()`, on a line of its own. The file's `# CHECK` lines expect exactly this layout.

---

## 3. Defining a `struct` in MLIR: the Type Class

In MLIR we also need a representation for struct types. MLIR has no builtin type that does exactly this, so Toy defines its own: a struct is an **unnamed container of element types**. The struct's name and its member names are only useful to the AST of the Toy compiler, so they are not encoded in the IR — `struct Struct { var a; var b; }` becomes `!toy.struct<tensor<*xf64>, tensor<*xf64>>`, with neither `Struct` nor `a`/`b` in it. MLIRGen (section 7) turns the names into types and member indices.

This is the conceptual heart of the chapter. MLIR `Type` objects are **value types**: a `mlir::Type` is just a wrapper around a pointer to an immutable, *uniqued* storage instance owned by the `MLIRContext`. Constructing a `Type` really means constructing and uniquing an instance of a storage class. Two structurally identical types are the *same pointer*, which makes type equality a pointer comparison and types trivially cheap to copy and hash.

A type that carries parametric data — like the struct's list of element types — needs a derived storage class. Singleton types with no extra data (for example the builtin `index` type) use the default `TypeStorage`.

### 3.1 Defining the storage class

In namespace `mlir::toy::detail`:

***mlir/Dialect.cpp***
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
  /// instance. This is used when constructing a new instance to ensure that we
  /// haven't already uniqued an instance of the given key.
  bool operator==(const KeyTy &key) const { return key == elementTypes; }

  /// Define a hash function for the key type. This is used when uniquing
  /// instances of the storage, see the `StructType::get` method.
  /// Note: This method isn't necessary as both llvm::ArrayRef and mlir::Type
  /// have hash functions available, so we could just omit this entirely.
  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_value(key);
  }

  /// Define a construction function for the key type from a set of parameters.
  /// These parameters will be provided when constructing the storage instance
  /// itself.
  /// Note: This method isn't necessary because KeyTy can be directly
  /// constructed with the given parameters.
  static KeyTy getKey(llvm::ArrayRef<mlir::Type> elementTypes) {
    return KeyTy(elementTypes);
  }

  /// Define a construction method for creating a new instance of this storage.
  /// This method takes an instance of a storage allocator, and an instance of a
  /// `KeyTy`. The given allocator must be used for *all* necessary dynamic
  /// allocations used to create the type storage and its internal.
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

### 3.2 Defining and registering the type class

`include/toy/Dialect.h` forward-declares `detail::StructTypeStorage` before including the generated dialect header, and defines the type after the generated op classes:

***include/toy/Dialect.h***
```cpp
/// This class defines the Toy struct type. It represents a collection of
/// element types. All derived types in MLIR must inherit from the CRTP class
/// 'Type::TypeBase'. It takes as template parameters the concrete type
/// (StructType), the base class to use (Type), and the storage class
/// (StructTypeStorage).
class StructType : public mlir::Type::TypeBase<StructType, mlir::Type,
                                               detail::StructTypeStorage> {
public:
  /// Inherit some necessary constructors from 'TypeBase'.
  using Base::Base;

  /// Create an instance of a `StructType` with the given element types. There
  /// *must* be atleast one element type.
  static StructType get(llvm::ArrayRef<mlir::Type> elementTypes);

  /// Returns the element types of this struct type.
  llvm::ArrayRef<mlir::Type> getElementTypes();

  /// Returns the number of element type held by this struct.
  size_t getNumElementTypes() { return getElementTypes().size(); }

  /// The name of this struct type.
  static constexpr StringLiteral name = "toy.struct";
};
```

Two differences from upstream's listing:

- **`name` is required in MLIR 20.** Upstream's class has no `name` member. `addTypes<StructType>()` (section 3.2) registers the type through `AbstractType::get<T>()`, which passes `T::name` to the `AbstractType` constructor (`mlir/IR/TypeSupport.h`, line 54 of the llvm@20 headers), so without the member the dialect does not compile. The context keeps that name, and `AbstractType::lookup(StringRef name, MLIRContext *)` finds registered types by it.
- **The header only forward-declares the storage.** Upstream defines `get()` and `getElementTypes()` inline in the class. Here `include/toy/Dialect.h` sees only `namespace detail { struct StructTypeStorage; }`, so the two methods that touch the storage are declared in the header and defined in `mlir/Dialect.cpp`, where `StructTypeStorage` is complete.

And the implementation, right after the storage class:

***mlir/Dialect.cpp***
```cpp
/// Create an instance of a `StructType` with the given element types. There
/// *must* be at least one element type.
StructType StructType::get(llvm::ArrayRef<mlir::Type> elementTypes) {
  assert(!elementTypes.empty() && "expected at least 1 element type");

  // Call into a helper 'get' method in 'TypeBase' to get a uniqued instance
  // of this type. The first parameter is the context to unique in. The
  // parameters after the context are forwarded to the storage instance.
  mlir::MLIRContext *ctx = elementTypes.front().getContext();
  return Base::get(ctx, elementTypes);
}

/// Returns the element types of this struct type.
llvm::ArrayRef<mlir::Type> StructType::getElementTypes() {
  // 'getImpl' returns a pointer to the internal storage instance.
  return getImpl()->elementTypes;
}
```

`Base::get(ctx, args...)` is where the magic lives: it asks the context's `StorageUniquer` for an instance keyed by `getKey(args...)`; on a hit it returns the existing storage pointer wrapped in a `StructType`, on a miss it calls `StructTypeStorage::construct`. This is **why MLIR uniques types**: `!toy.struct<tensor<*xf64>, tensor<*xf64>>` created in two different files/passes is literally the same object, so `type1 == type2` is a pointer compare and types can be used as map keys for free.

**Registering the type with the dialect.** The dialect must own the type so the context knows `!toy.…` types exist; the new line in Ch7 is `addTypes<StructType>()`:

***mlir/Dialect.cpp***
```cpp
/// Dialect initialization, the instance will be owned by the context. This is
/// the point of registration of types and operations for the dialect.
void ToyDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "toy/Ops.cpp.inc"
      >();
  addInterfaces<ToyInlinerInterface>();
  addTypes<StructType>();
}
```

Upstream stresses that the storage class definition must be visible where the type is registered. In this repo that is why `initialize()` lives in `mlir/Dialect.cpp`, below the full `StructTypeStorage` definition, and not in the header, which only forward-declares the storage. With the type registered, MLIRGen can create `StructType`s (section 7).

---

## 4. Exposing the Type to ODS

Ops defined in TableGen constrain operands/results with type-constraint definitions. To let ODS talk about our C++-defined type, `Ops.td` wraps it in a `DialectType` predicate:

***include/toy/Ops.td***
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

`Toy_Type` means "tensor **or** struct". It is the constraint later threaded through existing ops (section 6.1), while `Toy_StructType` alone constrains the struct ops (sections 6.2 and 6.3).

**Upstream vs. MLIR 20.** Upstream writes the predicate as `CPred<"$_self.isa<StructType>()">`. The text of a `CPred` is pasted into the generated C++ verbatim (with `$_self` replaced by the type being checked), and MLIR 20 has deprecated the member casts: substituting upstream's predicate into a copy of `Ops.td` and running `mlir-tblgen -gen-op-defs` produces `(type.isa<StructType>())` in the generated type-constraint function, and `Type::isa` is declared `[[deprecated("Use mlir::isa<U>() instead")]]` in `mlir/IR/Types.h`, so clang emits `-Wdeprecated-declarations` warnings for it. The repo's `::llvm::isa<StructType>($_self)` generates `(::llvm::isa<StructType>(type))` instead.

---

## 5. Parsing and Printing

`StructType` now exists in memory, but the textual IR needs a syntax for it. Types from a dialect print as `!<dialect>.<contents>`; the dialect provides the `<contents>` part via `parseType`/`printType` hooks. In ODS (`Ops.td`) we ask TableGen to declare them:

***include/toy/Ops.td***
```tablegen
def Toy_Dialect : Dialect {
  let name = "toy";
  let cppNamespace = "::mlir::toy";

  // We set this bit to generate a declaration of the `materializeConstant`
  // method so that we can materialize constants for our toy operations.
  let hasConstantMaterializer = 1;

  // We set this bit to generate the declarations for the dialect's type parsing
  // and printing hooks.
  let useDefaultTypePrinterParser = 1;

}
```

**Upstream vs. MLIR 20.** Upstream says the `parseType`/`printType` declarations are provided automatically once the type is exposed to ODS, and shows them in a hand-written `class ToyDialect : public mlir::Dialect`. In MLIR 20 they come from the dialect bit `useDefaultTypePrinterParser = 1` above, not from the `Toy_StructType` definition: the bit defaults to `0` in `mlir/IR/DialectBase.td`, and `mlir-tblgen -gen-dialect-decls` on a copy of `Ops.td` without that line generates no `parseType`/`printType` at all. With it, they appear in the generated dialect class:

***build/include/toy/Dialect.h.inc***
```cpp
  /// Parse a type registered to this dialect.
  ::mlir::Type parseType(::mlir::DialectAsmParser &parser) const override;

  /// Print a type registered to this dialect.
  void printType(::mlir::Type type,
                 ::mlir::DialectAsmPrinter &os) const override;
```

As the [MLIR language reference](https://mlir.llvm.org/docs/LangRef/#dialect-types) describes, dialect types are generally written `!dialect-namespace<type-data>`, with a pretty form available in some cases. MLIR handles the `!toy` part; the two hooks only provide the `type-data` part. Both take a high-level parser or printer object that does most of the work.

The grammar we choose:

```
struct-type ::= `struct` `<` type (`,` type)* `>`
```

so a two-member struct of unranked tensors prints as `!toy.struct<tensor<*xf64>, tensor<*xf64>>` (the `!toy.` prefix is added by MLIR; the hook only handles what follows).

### 5.1 Parsing and printing hooks

The parser hook in `mlir/Dialect.cpp`:

***mlir/Dialect.cpp***
```cpp
/// Parse an instance of a type registered to the toy dialect.
mlir::Type ToyDialect::parseType(mlir::DialectAsmParser &parser) const {
  // Parse a struct type in the following form:
  //   struct-type ::= `struct` `<` type (`,` type)* `>`

  // NOTE: All MLIR parser function return a ParseResult. This is a
  // specialization of LogicalResult that auto-converts to a `true` boolean
  // value on failure to allow for chaining, but may be used with explicit
  // `mlir::failed/mlir::succeeded` as desired.

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

- All parser methods return `ParseResult`, which converts to `true` **on failure** (as the `NOTE` says) — that is what makes the `if (a || b)` chaining idiom work.
- The parser is also a **verifier**: it rejects a non-tensor, non-struct element with a source-located diagnostic (run in section 5.2).
- Structs may nest (`!toy.struct<!toy.struct<...>, tensor<*xf64>>`), because `parser.parseType` recurses into this same hook for a nested `!toy.struct`. `struct-opt.mlir` exercises that (section 9.6).
- **Upstream vs. MLIR 20:** upstream checks the element with `elementType.isa<mlir::TensorType, StructType>()`; the repo uses the free function `llvm::isa<mlir::TensorType, StructType>(elementType)`, because the member form is deprecated (section 4).

**Printing.** The printer sits right after the parser:

***mlir/Dialect.cpp***
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

- The printer relies on `llvm::interleaveComma` to print the element types separated by `", "`, and since each element is itself an `mlir::Type`, nested structs recurse naturally.
- **Upstream vs. MLIR 20:** upstream writes `type.cast<StructType>()`; the repo uses `llvm::cast<StructType>(type)`, for the same deprecation.

### 5.2 Trying the hooks on real input

#### The type in a function signature

Upstream checks the new hooks with a program that only declares a struct and takes one as a parameter. It is not a file in this repo, so pass it on stdin:

```bash
cd /Users/roy/study/mlir/toy/Ch7
./build/toyc-ch7 -emit=mlir 2>&1 <<'EOF'
struct Struct {
  var a;
  var b;
}

def multiply_transpose(Struct value) {
}
EOF
```

Real output:

```mlir
module {
  toy.func private @multiply_transpose(%arg0: !toy.struct<tensor<*xf64>, tensor<*xf64>>) {
    toy.return
  }
}
```

- `printType` wrote `struct<tensor<*xf64>, tensor<*xf64>>` after the `!toy.` prefix: one unranked tensor per member, since members have no shapes.
- The one difference from upstream's output is `private`. Upstream's text shows `toy.func @multiply_transpose`, but this repo's MLIRGen marks every function except `main` private, so the inliner can delete it after inlining (section 8.4):

***mlir/MLIRGen.cpp***
```cpp
    // If this function isn't main, then set the visibility to private.
    if (funcAST.getProto()->getName() != "main")
      function.setPrivate();
```

To run `parseType` as well, pipe the output back in as MLIR:

```bash
cd /Users/roy/study/mlir/toy/Ch7
./build/toyc-ch7 -emit=mlir 2>&1 <<'EOF' | ./build/toyc-ch7 - -x mlir -emit=mlir 2>&1
struct Struct {
  var a;
  var b;
}

def multiply_transpose(Struct value) {
}
EOF
```

Real output:

```mlir
module {
  toy.func private @multiply_transpose(%arg0: !toy.struct<tensor<*xf64>, tensor<*xf64>>) {
    toy.return
  }
}
```

The same module comes back: `printType` ran on the way out, `parseType` on the way back in, and the type survives print → parse → print.

#### The parser rejects a non-tensor element type

`parseType` checks each element with `llvm::isa<mlir::TensorType, StructType>` (section 5.1). When parsing fails, the driver gives up:

***toyc.cpp***
```cpp
  // Parse the input mlir.
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Error can't load file " << inputFilename << "\n";
    return 3;
  }
```

Feed it IR with a struct of `i32`:

```bash
cd /Users/roy/study/mlir/toy/Ch7
./build/toyc-ch7 - -x mlir -emit=mlir 2>&1 <<'EOF'
toy.func @f(%a: !toy.struct<i32>) {
  toy.return
}
EOF
```

Real output (exit status 3):

```text
loc("<stdin>":1:29): error: element type for a struct must either be a TensorType or a StructType, got: 'i32'
Error can't load file -
```

- The diagnostic comes from `parser.emitError(typeLoc, ...)`: column 29 is where `i32` starts, the location `getCurrentLocation()` recorded before parsing the element.
- The location says `<stdin>`, because this input goes through MLIR's parser, which names standard input that way; the Toy lexer calls it `-` (section 9.8).
- `loadMLIR` then prints `Error can't load file -` (the input name is `-`) and returns 3, which `main` passes on as the exit status.

---

## 6. Operating on `StructType`

The `struct` type is defined and it round-trips through the textual IR. The next step is to use it in operations: widen a few existing ops so that structs can flow through them, and add two new ops that are specific to structs.

### 6.1 Updating existing operations

Structs must flow through returns and calls, so their operand constraints widen from `F64Tensor` (Chapter 6) to `Toy_Type` (abridged):

***include/toy/Ops.td***
```tablegen
def GenericCallOp : Toy_Op<"generic_call",
    [DeclareOpInterfaceMethods<CallOpInterface>]> {
  ...
  let arguments = (ins
    FlatSymbolRefAttr:$callee,
    Variadic<Toy_Type>:$inputs,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );

  // The generic call operation returns a single value of TensorType or
  // StructType.
  let results = (outs Toy_Type);
  ...
}
...
def ReturnOp : Toy_Op<"return", [Pure, HasParent<"FuncOp">,
                                 Terminator]> {
  ...
  let arguments = (ins Variadic<Toy_Type>:$input);
  ...
}
```

Upstream's excerpt of `ReturnOp` lists only `[Terminator, HasParent<"FuncOp">]`; the repo's op also carries `Pure`, as it has since Chapter 2. The change this chapter makes is the same: `Variadic<Toy_Type>`.

`toy.constant` also grows `let hasFolder = 1;` in this chapter (it keeps `F64ElementsAttr`/`F64Tensor` — tensor constants and struct constants remain separate ops):

***include/toy/Ops.td***
```tablegen
def ConstantOp : Toy_Op<"constant",
    [ConstantLike, Pure,
     DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  ...
  // Indicate that additional verification for this operation is necessary.
  let hasVerifier = 1;

  // Set the folder bit so that we can implement constant folders.
  let hasFolder = 1;
}
```

Compute ops (`add`, `mul`, `transpose`, `print`, ...) stay tensor-only: you cannot add two structs.

### 6.2 Adding `toy.struct_constant`

Materializes a struct *value* from a compile-time attribute. Since a struct is an ordered collection, its payload is an `ArrayAttr` — one attribute per member (a `DenseElementsAttr` for tensor members, or a nested `ArrayAttr` for nested structs). The `description` body (elided below) shows an example in `mlir` syntax:

***include/toy/Ops.td***
```tablegen
def StructConstantOp : Toy_Op<"struct_constant", [ConstantLike, Pure]> {
  let summary = "struct constant";
  let description = [{
    ...
  }];

  let arguments = (ins ArrayAttr:$value);
  let results = (outs Toy_StructType:$output);

  let assemblyFormat = "$value attr-dict `:` type($output)";

  // Indicate that additional verification for this operation is necessary.
  let hasVerifier = 1;
  let hasFolder = 1;
}
```

The `ConstantLike` trait matters: it tells generic MLIR utilities (folding, `matchPattern(m_Constant())`, the operation folder) that this op produces a constant whose value is its single attribute.

Upstream writes its example across several lines:

```mlir
  %0 = toy.struct_constant [
    dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64>
  ] : !toy.struct<tensor<*xf64>>
```

The declarative `assemblyFormat` prints the `ArrayAttr` on **one line** (section 7.4), but the parser ignores the line breaks, so upstream's multi-line layout is still accepted as input. Section 8.1 parses upstream's multi-line "after inlining" module.

Its verifier reuses the same recursive helper as `toy.constant` — `verifyConstantForType` in `Dialect.cpp` — which structurally checks the attribute against the result type (abridged; the elided tensor branch checks for a `DenseFPElementsAttr` and, for a ranked result type, a matching rank and shape — which is why the unranked member types in section 7.4 accept ranked data):

***mlir/Dialect.cpp***
```cpp
/// Verify that the given attribute value is valid for the given type.
static llvm::LogicalResult verifyConstantForType(mlir::Type type,
                                                 mlir::Attribute opaqueValue,
                                                 mlir::Operation *op) {
  if (llvm::isa<mlir::TensorType>(type)) {
    ...
  }
  auto resultType = llvm::cast<StructType>(type);
  llvm::ArrayRef<mlir::Type> resultElementTypes = resultType.getElementTypes();

  // Verify that the initializer is an Array.
  auto attrValue = llvm::dyn_cast<ArrayAttr>(opaqueValue);
  if (!attrValue || attrValue.getValue().size() != resultElementTypes.size())
    return op->emitError("constant of StructType must be initialized by an "
                         "ArrayAttr with the same number of elements, got ")
           << opaqueValue;

  // Check that each of the elements are valid.
  llvm::ArrayRef<mlir::Attribute> attrElementValues = attrValue.getValue();
  for (const auto it : llvm::zip(resultElementTypes, attrElementValues))
    if (failed(verifyConstantForType(std::get<0>(it), std::get<1>(it), op)))
      return mlir::failure();
  return mlir::success();
}

/// Verifier for the constant operation. This corresponds to the `::verify(...)`
/// in the op definition.
llvm::LogicalResult ConstantOp::verify() {
  return verifyConstantForType(getResult().getType(), getValue(), *this);
}

llvm::LogicalResult StructConstantOp::verify() {
  return verifyConstantForType(getResult().getType(), getValue(), *this);
}
```

The recursive call on each `(type, attribute)` pair is what handles nested structs.

### 6.3 Adding `toy.struct_access`

Extracts the N-th member of a struct value:

***include/toy/Ops.td***
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

  // Indicate that additional verification for this operation is necessary.
  let hasVerifier = 1;

  // Set the folder bit so that we can fold constant accesses.
  let hasFolder = 1;
}
```

`Pure` (no side effects, speculatable) is essential: it allows dead `struct_access` ops to be erased and lets the canonicalizer/folder move and fold them freely.

The custom builder derives the result type from the input struct, so callers pass only `(value, index)`; the verifier right below it enforces both bounds and result-type consistency:

***mlir/Dialect.cpp***
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

---

## 7. MLIRGen for Structs

Upstream covers code generation in one sentence ("See examples/toy/Ch7/mlir/MLIRGen.cpp for more details") and then shows the resulting module. This section walks that code, all in `mlir/MLIRGen.cpp`, and 7.4 shows the module it produces.

### 7.1 Tracking struct declarations and resolving types

MLIRGen needs to map *names* (`"Struct"`, member `"a"`) to MLIR types and member indices, so it keeps the AST around. The symbol table also changes shape — each variable now remembers its declaration so we can recover its declared *struct* type later — and a new `structMap` maps each struct name to its MLIR type and AST node (abridged):

***mlir/MLIRGen.cpp***
```cpp
  llvm::ScopedHashTable<StringRef, std::pair<mlir::Value, VarDeclExprAST *>>
      symbolTable;
  using SymbolTableScopeT =
      llvm::ScopedHashTableScope<StringRef,
                                 std::pair<mlir::Value, VarDeclExprAST *>>;
  ...
  /// A mapping for named struct types to the underlying MLIR type and the
  /// original AST node.
  llvm::StringMap<std::pair<mlir::Type, StructAST *>> structMap;
```

Top-level dispatch in `mlirGen(ModuleAST&)` handles both record kinds:

***mlir/MLIRGen.cpp***
```cpp
    for (auto &record : moduleAST) {
      if (FunctionAST *funcAST = llvm::dyn_cast<FunctionAST>(record.get())) {
        mlir::toy::FuncOp func = mlirGen(*funcAST);
        if (!func)
          return nullptr;
        functionMap.insert({func.getName(), func});
      } else if (StructAST *str = llvm::dyn_cast<StructAST>(record.get())) {
        if (failed(mlirGen(*str)))
          return nullptr;
      } else {
        llvm_unreachable("unknown record type");
      }
    }
```

A `StructAST` produces no IR, only a `structMap` entry:

***mlir/MLIRGen.cpp***
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
      if (variable->getInitVal())
        return emitError(loc(variable->loc()))
               << "error: variables within a struct definition must not have "
                  "initializers";
      if (!variable->getType().shape.empty())
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

The shape check reuses the initializer message verbatim, and the messages start with `error:` although `emitError` adds that prefix itself — both visible in section 9.8. Since members have no shape info, each member's type is `tensor<*xf64>` (or a nested struct), so `struct Struct { var a; var b; }` becomes `!toy.struct<tensor<*xf64>, tensor<*xf64>>`. Because records are processed in order, a struct must be defined before the functions that use it.

**Resolving types: `getType(VarType, ...)`.** Type resolution, which the member loop above calls, now checks the name first:

***mlir/MLIRGen.cpp***
```cpp
  /// Build an MLIR type from a Toy AST variable type (forward to the generic
  /// getType above for non-struct types).
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

    return getType(type.shape);
  }
```

`mlirGen(PrototypeAST&)` calls it for every parameter — which is how `!toy.struct<...>` ends up in the `toy.func` signature in section 7.4.

### 7.2 Struct literals → `ArrayAttr` + `StructConstantOp`

A struct literal becomes a single constant op whose attribute is built recursively (numbers/tensor literals → `DenseElementsAttr`, nested struct literals → nested `ArrayAttr`) (abridged):

***mlir/MLIRGen.cpp***
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
  ...
  /// Emit a struct literal. It will be emitted as an array of
  /// other literals in an Attribute attached to a `toy.struct_constant`
  /// operation.
  mlir::Value mlirGen(StructLiteralExprAST &lit) {
    mlir::ArrayAttr dataAttr;
    mlir::Type dataType;
    std::tie(dataAttr, dataType) = getConstantAttr(lit);

    // Build the MLIR op `toy.struct_constant`. This invokes the
    // `StructConstantOp::build` method.
    return builder.create<StructConstantOp>(loc(lit.loc()), dataType, dataAttr);
  }
```

Every tensor member gets the unranked type `getType({})` regardless of its literal's shape, so the literal's type matches the unranked member types of the declared struct. `mlirGen(VarDeclExprAST&)` additionally checks that a struct-typed declaration's initializer type matches the declared struct type exactly (again, a pointer compare thanks to uniquing) — the first error in section 9.8.

### 7.3 Member access → member index → `StructAccessOp`

Remember: the AST for `value.a` is `BinaryExprAST('.')` with a variable on each side. MLIRGen must turn the *name* `a` into a *number* (its position in the struct). `getStructFor` finds the `StructAST` of the LHS — for a plain variable via its declaration in the symbol table, for a nested access `x.y.z` by recursing and looking up the member's declared type — and `getMemberIndex` searches that struct's members for the RHS name. The binary-op codegen right after them intercepts `.` before evaluating the RHS, because the RHS is a member *name*, not a value (abridged):

***mlir/MLIRGen.cpp***
```cpp
  /// Return the struct type that is the result of the given expression, or null
  /// if it cannot be inferred.
  StructAST *getStructFor(ExprAST *expr) {
    llvm::StringRef structName;
    if (auto *decl = llvm::dyn_cast<VariableExprAST>(expr)) {
      auto varIt = symbolTable.lookup(decl->getName());
      if (!varIt.first)
        return nullptr;
      structName = varIt.second->getType().name;
    } else if (auto *access = llvm::dyn_cast<BinaryExprAST>(expr)) {
      if (access->getOp() != '.')
        return nullptr;
      // The name being accessed should be in the RHS.
      auto *name = llvm::dyn_cast<VariableExprAST>(access->getRHS());
      if (!name)
        return nullptr;
      StructAST *parentStruct = getStructFor(access->getLHS());
      if (!parentStruct)
        return nullptr;

      // Get the element within the struct corresponding to the name.
      VarDeclExprAST *decl = nullptr;
      for (auto &var : parentStruct->getVariables()) {
        if (var->getName() == name->getName()) {
          decl = var.get();
          break;
        }
      }
      if (!decl)
        return nullptr;
      structName = decl->getType().name;
    }
    if (structName.empty())
      return nullptr;

    // If the struct name was valid, check for an entry in the struct map.
    auto structIt = structMap.find(structName);
    if (structIt == structMap.end())
      return nullptr;
    return structIt->second.second;
  }

  /// Return the numeric member index of the given struct access expression.
  std::optional<size_t> getMemberIndex(BinaryExprAST &accessOp) {
    assert(accessOp.getOp() == '.' && "expected access operation");

    // Lookup the struct node for the LHS.
    StructAST *structAST = getStructFor(accessOp.getLHS());
    if (!structAST)
      return std::nullopt;

    // Get the name from the RHS.
    VariableExprAST *name = llvm::dyn_cast<VariableExprAST>(accessOp.getRHS());
    if (!name)
      return std::nullopt;

    auto structVars = structAST->getVariables();
    const auto *it = llvm::find_if(structVars, [&](auto &var) {
      return var->getName() == name->getName();
    });
    if (it == structVars.end())
      return std::nullopt;
    return it - structVars.begin();
  }

  /// Emit a binary operation
  mlir::Value mlirGen(BinaryExprAST &binop) {
    ...
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

    // Otherwise, this is a normal binary op.
    mlir::Value rhs = mlirGen(*binop.getRHS());
    ...
  }
```

Names exist only in the frontend; the IR carries indices. `value.a` → `toy.struct_access %arg0[0]`, `value.b` → `toy.struct_access %arg0[1]` (section 7.4). A name that isn't a member makes `getMemberIndex` return `std::nullopt` — the "invalid access into struct expression" error in section 9.8.

### 7.4 The full MLIR module

With the new ops and this code generation, the original example compiles to a full module. The input is the chapter's test file (section 2.1 shows it whole); the lines that exercise 7.1–7.3 are the struct-typed parameter, the member accesses, and the struct literal:

***test_Example/Toy/Ch7/struct-codegen.toy***
```text
def multiply_transpose(Struct value) {
  # We can access the elements of a struct via the '.' operator.
  return transpose(value.a) * transpose(value.b);
}

def main() {
  # We initialize struct values using a composite initializer.
  Struct value = {[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]};
```

```bash
cd /Users/roy/study/mlir/toy/Ch7
./build/toyc-ch7 ../../test_Example/Toy/Ch7/struct-codegen.toy -emit=mlir 2>&1
```

Real output:

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

Reading it against the walkthrough:

- **Function signature**: `%arg0: !toy.struct<tensor<*xf64>, tensor<*xf64>>` is there because `getType(VarType)` resolved the name `Struct` via `structMap` (section 7.1), and the custom `printType` produced the `!toy.struct<...>` syntax (section 5.1). Members declared without shapes print as unranked `tensor<*xf64>`.
- **`toy.struct_access %arg0[0]` / `[1]`**: `value.a`/`value.b`, turned from names into indices by `getMemberIndex` (section 7.3). The `-> tensor<*xf64>` result type came from the custom builder (section 6.3).
- **`toy.struct_constant [dense<...>, dense<...>]`**: the struct literal packed into one `ArrayAttr` with two `DenseElementsAttr` members (section 7.2). The attribute types are ranked (`tensor<2x3xf64>`) while the struct's member types are unranked; the verifier allows that (section 6.2).
- **`toy.generic_call`** with a struct operand is legal only because `GenericCallOp`'s inputs are `Variadic<Toy_Type>` (section 6.1).

It has the same operations as upstream's module. The two textual differences are the ones already explained: `private` on `multiply_transpose` (section 5.2) and the single-line `toy.struct_constant` (section 6.2).

---

## 8. Optimizing Operations on `StructType`

With a few ops working on `StructType`, there are many new constant-folding opportunities. With `-opt` (or any `-emit` past `mlir`), the pipeline in `toyc.cpp` runs the **inliner**, then per function **canonicalizer → shape inference → canonicalizer → CSE**. (Compared with Chapter 6, the only pipeline change in `toyc.cpp` is the first of those canonicalizers.) After inlining, `main` contains a `struct_constant` feeding `struct_access` ops, a chain the folder can collapse completely.

### 8.1 After inlining

Upstream shows the module after inlining as an intermediate step: `multiply_transpose`'s body is now in `main`, and two `toy.struct_access` ops still read from the `toy.struct_constant`. **In this build that stage is never printed.** The inliner is the first pass of the `-opt` pipeline:

***toyc.cpp***
```cpp
  if (enableOpt || isLoweringToAffine) {
    // Inline all functions into main and then delete them.
    pm.addPass(mlir::createInlinerPass());

    // Now that there is only one function, we can infer the shapes of each of
    // the operations.
    mlir::OpPassManager &optPM = pm.nest<mlir::toy::FuncOp>();
    optPM.addPass(mlir::createCanonicalizerPass());
    optPM.addPass(mlir::toy::createShapeInferencePass());
    optPM.addPass(mlir::createCanonicalizerPass());
    optPM.addPass(mlir::createCSEPass());
  }
```

The MLIR 20 inliner also simplifies every callable it visits, with its `default-pipeline` option, which defaults to `"canonicalize"` (`mlir/Transforms/Passes.td`). That canonicalization runs the folders of section 8.2 on `main` right after `multiply_transpose` is inlined, so the struct ops are folded before the inliner pass finishes.

#### The inliner's own dump is already struct-free

MLIR's `-mlir-print-ir-after-all` (registered by `registerPassManagerCLOptions()` in `main`) dumps the IR after every pass; `sed` keeps only the dump after the `Inliner`:

```bash
cd /Users/roy/study/mlir/toy/Ch7
./build/toyc-ch7 ../../test_Example/Toy/Ch7/struct-codegen.toy -emit=mlir -opt -mlir-print-ir-after-all 2>&1 | sed -n '/After Inliner/,/^}/p'
```

Real output:

```mlir
// -----// IR Dump After Inliner (inline) //----- //
module {
  toy.func @main() {
    %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
    %1 = toy.transpose(%0 : tensor<*xf64>) to tensor<*xf64>
    %2 = toy.transpose(%0 : tensor<*xf64>) to tensor<*xf64>
    %3 = toy.mul %1, %2 : tensor<*xf64>
    toy.print %3 : tensor<*xf64>
    toy.return
  }
}
```

- `multiply_transpose` is gone (inlined, then deleted because it is private), and so are all struct ops: the inliner's nested canonicalization already folded them.
- The two identical tensor members already came out as one `%0` (section 8.4 explains why).
- The types are still unranked and the two transposes still separate: the nested `Canonicalizer`, `ShapeInferencePass` and `CSE` of the pipeline above have not run yet. Section 9.5 walks through all the dumps of this run.

#### Upstream's intermediate module is valid input

Upstream's "after inlining" module is not a file in this repo, so pass it on stdin, copied verbatim from the upstream chapter:

```bash
cd /Users/roy/study/mlir/toy/Ch7
./build/toyc-ch7 - -x mlir -emit=mlir 2>&1 <<'EOF'
module {
  toy.func @main() {
    %0 = toy.struct_constant [
      dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>,
      dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
    ] : !toy.struct<tensor<*xf64>, tensor<*xf64>>
    %1 = toy.struct_access %0[0] : !toy.struct<tensor<*xf64>, tensor<*xf64>> -> tensor<*xf64>
    %2 = toy.transpose(%1 : tensor<*xf64>) to tensor<*xf64>
    %3 = toy.struct_access %0[1] : !toy.struct<tensor<*xf64>, tensor<*xf64>> -> tensor<*xf64>
    %4 = toy.transpose(%3 : tensor<*xf64>) to tensor<*xf64>
    %5 = toy.mul %2, %4 : tensor<*xf64>
    toy.print %5 : tensor<*xf64>
    toy.return
  }
}
EOF
```

Real output:

```mlir
module {
  toy.func @main() {
    %0 = toy.struct_constant [dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>, dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>] : !toy.struct<tensor<*xf64>, tensor<*xf64>>
    %1 = toy.struct_access %0[0] : !toy.struct<tensor<*xf64>, tensor<*xf64>> -> tensor<*xf64>
    %2 = toy.transpose(%1 : tensor<*xf64>) to tensor<*xf64>
    %3 = toy.struct_access %0[1] : !toy.struct<tensor<*xf64>, tensor<*xf64>> -> tensor<*xf64>
    %4 = toy.transpose(%3 : tensor<*xf64>) to tensor<*xf64>
    %5 = toy.mul %2, %4 : tensor<*xf64>
    toy.print %5 : tensor<*xf64>
    toy.return
  }
}
```

- Without `-opt` no pass runs, so this is the stage upstream describes: the `toy.struct_access` ops read directly from the `toy.struct_constant`.
- The parser accepted upstream's multi-line `toy.struct_constant [ ... ]`, and the declarative `assemblyFormat` printed it back on one line (section 6.2).
- Section 8.4 runs the same module through `-opt`.

### 8.2 The fold hooks (`FoldAdaptor`)

Setting `let hasFolder = 1;` in ODS generates a declaration of the form:

```cpp
OpFoldResult FooOp::fold(FoldAdaptor adaptor);
```

`FoldAdaptor` is a generated adaptor class that mirrors the op's operands, but where each operand accessor (e.g. `adaptor.getInput()`) returns an **`Attribute`** instead of a `Value`: the constant value of that operand *if* the operand is currently known to be constant, or null otherwise. The returned `OpFoldResult` is either an `Attribute` (constant result) or a `Value` (replace with an existing SSA value); returning null means "cannot fold".

The three implementations:

***mlir/ToyCombine.cpp***
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

Upstream's chapter 3 introduced folders with `FoldConstantReshape`; here the same mechanism is switched on per op with `hasFolder` (sections 6.1–6.3).

**Upstream vs. MLIR 20.** Upstream's folders are written against an older API:

| Upstream | MLIR 20 (this repo) | Verified by |
|---|---|---|
| `return value();` | `return getValue();` | ODS emits only prefixed accessors: the generated `build/include/toy/Ops.h.inc` declares `getValue()`/`getValueAttr()` and has no `value()` |
| `index().getZExtValue()` | `getIndex()` | the generated `StructAccessOp::getIndex()` returns `uint64_t` directly; the `IntegerAttr` is `getIndexAttr()` |
| `adaptor.getInput().dyn_cast_or_null<mlir::ArrayAttr>()` | `llvm::dyn_cast_if_present<mlir::ArrayAttr>(adaptor.getInput())` | `Attribute::dyn_cast_or_null` is `[[deprecated]]` in `mlir/IR/Attributes.h`; clang warns on it |

### 8.3 `materializeConstant`: turning attributes back into ops

When a fold returns an `Attribute`, the operation folder must create an op that produces that constant as an SSA value — but *which* op? That's dialect-specific, so MLIR asks the dialect via the hook enabled by `let hasConstantMaterializer = 1;` (section 5):

***mlir/Dialect.cpp***
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

So when `struct_access %cst[0]` folds to a `DenseElementsAttr` with tensor type, the folder calls `materializeConstant(builder, denseAttr, tensorType, loc)` and gets a plain `toy.constant` — the struct is gone from that use. If the accessed member is itself a struct (the nested case in section 9.6), the `ArrayAttr` + `StructType` branch materializes a smaller `toy.struct_constant` instead.

**Upstream vs. MLIR 20:** upstream's version uses `type.isa<StructType>()` and `value.cast<...>()`; the repo uses `llvm::isa`/`llvm::cast`, for the same deprecation as in section 4.

### 8.4 What `-opt` actually does to the struct program

Step by step on `struct-codegen.toy`, matching the pass-by-pass dump in section 9.5:

1. **Inliner** inlines `multiply_transpose` into `main` (the `ToyInlinerInterface` from Ch4/Ch5 permits it; struct-typed arguments inline like any other value) and deletes the now-unused private function. Inside `main`, `toy.struct_access %0[0]` and `[1]` now read directly from the `toy.struct_constant`.
2. The inliner canonicalizes the functions it visits, and that already invokes the folders: each `struct_access` folds to its member's `DenseElementsAttr`, and `materializeConstant` produces `toy.constant` ops for them. The folder keeps constants unique per region, so the two identical member constants come out as a single `toy.constant`. The now-dead `struct_constant` (which is `Pure`) is erased. This is already the state of the `Inliner` dump in section 8.1.
3. **Shape inference** runs as in Ch4, turning `tensor<*xf64>` into ranked shapes.
4. **CSE** merges the two identical `toy.transpose`s, leaving `toy.mul %1, %1`.

Result: **no `!toy.struct`, no `struct_constant`, no `struct_access` remain** — just the same tensor IR Chapter 6 knows how to lower to Affine → LLVM → JIT. This is why Ch7 needs zero new lowering code, and why `-emit=jit` works on the struct program (section 9.7).

#### The final IR

```bash
cd /Users/roy/study/mlir/toy/Ch7
./build/toyc-ch7 ../../test_Example/Toy/Ch7/struct-codegen.toy -emit=mlir -opt 2>&1
```

Real output:

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

This is identical to the last module in upstream's chapter: one function, one `toy.constant`, ranked types, and a single `toy.transpose` used twice.

#### Upstream's intermediate module folds to the same IR

The "after inlining" module of section 8.1, on stdin again, now with `-opt`:

```bash
cd /Users/roy/study/mlir/toy/Ch7
./build/toyc-ch7 - -x mlir -emit=mlir -opt 2>&1 <<'EOF'
module {
  toy.func @main() {
    %0 = toy.struct_constant [
      dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>,
      dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
    ] : !toy.struct<tensor<*xf64>, tensor<*xf64>>
    %1 = toy.struct_access %0[0] : !toy.struct<tensor<*xf64>, tensor<*xf64>> -> tensor<*xf64>
    %2 = toy.transpose(%1 : tensor<*xf64>) to tensor<*xf64>
    %3 = toy.struct_access %0[1] : !toy.struct<tensor<*xf64>, tensor<*xf64>> -> tensor<*xf64>
    %4 = toy.transpose(%3 : tensor<*xf64>) to tensor<*xf64>
    %5 = toy.mul %2, %4 : tensor<*xf64>
    toy.print %5 : tensor<*xf64>
    toy.return
  }
}
EOF
```

Real output:

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

Exactly the IR above. There is nothing to inline here, but the inliner still canonicalizes `main` (section 8.1), so the struct ops fold at the same point as for `struct-codegen.toy`: adding `-mlir-print-ir-after-all` shows a `Canonicalizer` dump, already free of struct ops, before the `Inliner` dump.

Upstream ends by pointing to [Defining Dialect Attributes and Types](https://mlir.llvm.org/docs/DefiningDialects/AttributesAndTypes/) for more on custom types. That document covers the declarative ODS `TypeDef` way, which generates the storage class of section 3.1 for you.

---

## 9. Build and Run

The superbuild (`toy/build.sh`, `toy/run.sh`, `CMakePresets.json`) is documented once in the top-level [README](../README.md#the-build-system). This section builds and runs Chapter 7 on its own, from the chapter directory.

### 9.1 Building

```bash
cd /Users/roy/study/mlir/toy/Ch7
cmake -S . -B build -G Ninja
cmake --build build          # → ./build/toyc-ch7
```

No preset applies at the chapter level, yet no toolchain flags are needed, because the shell environment already points at Homebrew LLVM 20:

- `CXX=/opt/homebrew/opt/llvm@20/bin/clang++` (and `CC`) selects the compiler.
- `/opt/homebrew/opt/llvm@20/bin` is on `PATH`, and `find_package` also searches the prefix above each `PATH` entry, so it finds `/opt/homebrew/opt/llvm@20/lib/cmake/{mlir,llvm}` by itself.

In a shell without that setup, pass them explicitly: `-DMLIR_DIR=/opt/homebrew/opt/llvm@20/lib/cmake/mlir -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm@20/bin/clang++`. (The chapter's CMake targets are in the [appendix](#appendix-what-chapter-7-adds-to-the-build).)

Like Chapter 6, the chapter's targets are skipped silently if the MLIR installation was built without the execution engine (`MLIR_ENABLE_EXECUTION_ENGINE`); Homebrew's llvm@20 has it. A standalone build puts the binary directly in `build/`; the superbuild's `toy/build/bin/toyc-ch7` behaves identically.

### 9.2 Running

```bash
cd /Users/roy/study/mlir/toy/Ch7
./build/toyc-ch7 ../../test_Example/Toy/Ch7/struct-codegen.toy -emit=mlir 2>&1
```

The driver is Chapter 6's, unchanged in its options:

- The positional argument is the input file (`-` or no argument reads stdin). A `.toy` file goes through the Toy lexer/parser/MLIRGen; a file ending in `.mlir` (or any input with `-x mlir`) is parsed directly by MLIR — which exercises the dialect's custom *type* parser for `!toy.struct<...>` (section 5).
- `-emit=ast | mlir | mlir-affine | mlir-llvm | llvm | jit` picks how far to go. `ast` dumps the parser's output; `mlir` dumps the Toy dialect; the later values continue through the Ch5/Ch6 lowering pipeline.
- `-opt` enables the inliner + canonicalizer + shape inference + CSE pipeline for `-emit=mlir`. Every `-emit` value from `mlir-affine` on runs that pipeline unconditionally (it is guarded by `enableOpt || isLoweringToAffine` in `toyc.cpp`), because the lowering can only handle inlined, shape-inferred, struct-free IR.
- `2>&1` — AST and MLIR dumps go to **stderr**; the JIT's `printf` output goes to stdout.

**The test inputs.** Each struct test file exercises one part of the chapter:

- **`struct-ast.toy`** + `-emit=ast` — the parser/AST additions (section 9.3).
- **`struct-codegen.toy`** + `-emit=mlir` and `-emit=mlir -opt` — MLIRGen and folding (sections 9.4 and 9.5).
- **`struct-opt.mlir`** + `-emit=mlir -opt` — custom type parsing and nested-struct folding (section 9.6).
- The remaining files (`codegen.toy`, `ast.toy`, `affine-lowering.mlir`, `llvm-lowering.mlir`, `shape_inference.mlir`, `transpose_transpose.toy`, `trivial_reshape.toy`, `scalar.toy`, `empty.toy`, `invalid.mlir`, `jit.toy`) are the Ch1–Ch6 regression suite, proving the struct work didn't break anything. Every `RUN:` line in the directory passes against this build when run through `FileCheck` by hand.

### 9.3 `-emit=ast`: the AST dump

```bash
cd /Users/roy/study/mlir/toy/Ch7
./build/toyc-ch7 ../../test_Example/Toy/Ch7/struct-ast.toy -emit=ast 2>&1
```

`struct-ast.toy` is the same program as `struct-codegen.toy` (section 2.1) behind a different `RUN:` header, so struct definitions start at line 3. Actual output:

```text
  Module:
    Struct: Struct @../../test_Example/Toy/Ch7/struct-ast.toy:3:1
      Variables: [
        VarDecl a<> @../../test_Example/Toy/Ch7/struct-ast.toy:4:3
        VarDecl b<> @../../test_Example/Toy/Ch7/struct-ast.toy:5:3
      ]
    Function 
      Proto 'multiply_transpose' @../../test_Example/Toy/Ch7/struct-ast.toy:9:1
      Params: [value]
      Block {
        Return
          BinOp: * @../../test_Example/Toy/Ch7/struct-ast.toy:11:31
            Call 'transpose' [ @../../test_Example/Toy/Ch7/struct-ast.toy:11:10
              BinOp: . @../../test_Example/Toy/Ch7/struct-ast.toy:11:26
                var: value @../../test_Example/Toy/Ch7/struct-ast.toy:11:20
                var: a @../../test_Example/Toy/Ch7/struct-ast.toy:11:26
            ]
            Call 'transpose' [ @../../test_Example/Toy/Ch7/struct-ast.toy:11:31
              BinOp: . @../../test_Example/Toy/Ch7/struct-ast.toy:11:47
                var: value @../../test_Example/Toy/Ch7/struct-ast.toy:11:41
                var: b @../../test_Example/Toy/Ch7/struct-ast.toy:11:47
            ]
      } // Block
    Function 
      Proto 'main' @../../test_Example/Toy/Ch7/struct-ast.toy:14:1
      Params: []
      Block {
        VarDecl value<Struct> @../../test_Example/Toy/Ch7/struct-ast.toy:16:3
          Struct Literal:             Literal: <2, 3>[ <3>[ 1.000000e+00, 2.000000e+00, 3.000000e+00], <3>[ 4.000000e+00, 5.000000e+00, 6.000000e+00]] @../../test_Example/Toy/Ch7/struct-ast.toy:16:19
            Literal: <2, 3>[ <3>[ 1.000000e+00, 2.000000e+00, 3.000000e+00], <3>[ 4.000000e+00, 5.000000e+00, 6.000000e+00]] @../../test_Example/Toy/Ch7/struct-ast.toy:16:43
           @../../test_Example/Toy/Ch7/struct-ast.toy:16:18
        VarDecl c<> @../../test_Example/Toy/Ch7/struct-ast.toy:19:3
          Call 'multiply_transpose' [ @../../test_Example/Toy/Ch7/struct-ast.toy:19:11
            var: value @../../test_Example/Toy/Ch7/struct-ast.toy:19:30
          ]
        Print [ @../../test_Example/Toy/Ch7/struct-ast.toy:20:3
          var: c @../../test_Example/Toy/Ch7/struct-ast.toy:20:9
        ]
      } // Block
```

What to notice (the AST classes and the dumper are explained in section 2):

- The module now holds **records** of two kinds: `Struct: Struct` (with its `Variables: [...]`) and the two `Function`s.
- `VarDecl value<Struct>` prints a *named* type in the angle brackets where tensor declarations print a shape; the struct members print `a<>`/`b<>` because they have neither.
- Member access is an ordinary `BinOp: .` whose RHS is a `var: a` — there is no dedicated access node.
- The `Struct Literal:` line looks broken: its first member is printed on the same line after a run of spaces, and the literal's own location (`16:18`, the `{`) comes last on a line of its own. That is simply how the struct-literal dumper prints (section 2.5); the file's `# CHECK` lines expect exactly this.

### 9.4 `-emit=mlir`: structs in the IR

```bash
cd /Users/roy/study/mlir/toy/Ch7
./build/toyc-ch7 ../../test_Example/Toy/Ch7/struct-codegen.toy -emit=mlir 2>&1
```

Actual output:

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

Each line is read against the MLIRGen walkthrough in section 7.4.

The full module round-trips as well: `./build/toyc-ch7 ../../test_Example/Toy/Ch7/struct-codegen.toy -emit=mlir 2>&1 | ./build/toyc-ch7 - -x mlir -emit=mlir 2>&1` prints exactly the output above. On the way back in, the ops' declarative parsers call `parseType` for every `!toy.struct` in the signature and in the `struct_constant`/`struct_access`/`generic_call` types.

### 9.5 `-emit=mlir -opt`: the struct disappears

```bash
cd /Users/roy/study/mlir/toy/Ch7
./build/toyc-ch7 ../../test_Example/Toy/Ch7/struct-codegen.toy -emit=mlir -opt 2>&1
```

Actual output:

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

Compare the two dumps:

| Before `-opt` | After `-opt` |
|---|---|
| 2 functions | 1 function (`multiply_transpose` inlined and removed) |
| 1 `struct_constant`, 2 `struct_access` | **zero struct ops, zero `!toy.struct` types** |
| two identical tensor members | 1 `toy.constant` (the folder de-duplicates constants) |
| 2 transposes | 1 `toy.transpose` (CSE), used twice by `toy.mul %1, %1` |
| everything `tensor<*xf64>` | ranked `tensor<2x3xf64>` / `tensor<3x2xf64>` (shape inference) |

To watch it happen pass by pass, add MLIR's `-mlir-print-ir-after-all` (registered by `registerPassManagerCLOptions()` in `main`):

```bash
cd /Users/roy/study/mlir/toy/Ch7
./build/toyc-ch7 ../../test_Example/Toy/Ch7/struct-codegen.toy -emit=mlir -opt -mlir-print-ir-after-all 2>&1
```

Actual output, abridged. The dump has eight parts. The first three are `Canonicalizer` dumps from *inside* the inliner, which canonicalizes each callable it visits (section 8.1): `multiply_transpose` and `main` as they are in section 9.4, then `main` after `multiply_transpose` was inlined into it. That third dump is already identical to the body of the `Inliner` dump that follows, which is where this excerpt starts:

```mlir
// -----// IR Dump After Inliner (inline) //----- //
module {
  toy.func @main() {
    %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
    %1 = toy.transpose(%0 : tensor<*xf64>) to tensor<*xf64>
    %2 = toy.transpose(%0 : tensor<*xf64>) to tensor<*xf64>
    %3 = toy.mul %1, %2 : tensor<*xf64>
    toy.print %3 : tensor<*xf64>
    toy.return
  }
}
...
// -----// IR Dump After (anonymous namespace)::ShapeInferencePass (toy-shape-inference) //----- //
toy.func @main() {
  %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
  %1 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
  %2 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
  %3 = toy.mul %1, %2 : tensor<3x2xf64>
  toy.print %3 : tensor<3x2xf64>
  toy.return
}
...
// -----// IR Dump After CSE (cse) //----- //
toy.func @main() {
  %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
  %1 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
  %2 = toy.mul %1, %1 : tensor<3x2xf64>
  toy.print %2 : tensor<3x2xf64>
  toy.return
}
```

Right after the inliner, the struct is already gone, and the two member constants have already been merged into one `%0`. Shape inference then ranks the types, and only CSE merges the two transposes. The mechanics — `StructAccessOp::fold`, `materializeConstant`, and the folder's constant de-duplication — are explained in sections 8.2–8.4.

Upstream's intermediate "after inlining" module, which this pipeline never prints, folds to the same IR (section 8.4).

### 9.6 `struct-opt.mlir`: folding a nested struct from hand-written IR

The second struct test is written directly in MLIR, so it takes the `.mlir` input path and exercises the custom type *parser* (section 5.1) instead of MLIRGen. It builds a struct whose first member is itself a struct, and reads through both levels:

***test_Example/Toy/Ch7/struct-opt.mlir***
```mlir
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

```bash
cd /Users/roy/study/mlir/toy/Ch7
./build/toyc-ch7 ../../test_Example/Toy/Ch7/struct-opt.mlir -emit=mlir -opt 2>&1
```

Actual output:

```mlir
module {
  toy.func @main() {
    %0 = toy.constant dense<4.000000e+00> : tensor<2x2xf64>
    toy.print %0 : tensor<2x2xf64>
    toy.return
  }
}
```

Two chained accesses through a nested struct fold all the way down to one tensor constant: the inner access folds to an `ArrayAttr` (materialized as a smaller `toy.struct_constant`), the outer one to a `DenseElementsAttr` (materialized as a `toy.constant`) — section 8.3.

The same IR also comes out of MLIRGen for a Toy program with a struct-typed member (section 2.1):

```bash
cd /Users/roy/study/mlir/toy/Ch7
printf 'struct Inner {\n  var a;\n}\n\nstruct Outer {\n  Inner i;\n  var b;\n}\n\ndef main() {\n  Outer o = {{[1, 2]}, [3, 4]};\n  print(o.i.a);\n}\n' \
  | ./build/toyc-ch7 -emit=mlir 2>&1
```

Actual output:

```mlir
module {
  toy.func @main() {
    %0 = toy.struct_constant [[dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf64>], dense<[3.000000e+00, 4.000000e+00]> : tensor<2xf64>] : !toy.struct<!toy.struct<tensor<*xf64>>, tensor<*xf64>>
    %1 = toy.struct_access %0[0] : !toy.struct<!toy.struct<tensor<*xf64>>, tensor<*xf64>> -> !toy.struct<tensor<*xf64>>
    %2 = toy.struct_access %1[0] : !toy.struct<tensor<*xf64>> -> tensor<*xf64>
    toy.print %2 : tensor<*xf64>
    toy.return
  }
}
```

The nested literal `{{[1, 2]}, [3, 4]}` became a nested `ArrayAttr` (section 7.2), and `o.i.a` became two chained accesses: `getStructFor` recursed through `o.i` to find `Inner` (section 7.3).

### 9.7 `-emit=jit`: the unchanged backend still works

Because the optimized module is pure tensor IR, the whole Ch6 pipeline (Affine → LLVM dialect → LLVM IR → ExecutionEngine) runs unmodified. First the struct-free smoke test:

```bash
cd /Users/roy/study/mlir/toy/Ch7
./build/toyc-ch7 ../../test_Example/Toy/Ch7/jit.toy -emit=jit
```

Actual output:

```text
1.000000 2.000000 
3.000000 4.000000 
```

The struct program JITs too, with or without `-opt` — every `-emit` value past `mlir` runs the inline/fold pipeline anyway (section 9.2), so the structs are gone before lowering starts:

```bash
cd /Users/roy/study/mlir/toy/Ch7
./build/toyc-ch7 ../../test_Example/Toy/Ch7/struct-codegen.toy -emit=jit
```

Actual output — `transpose(a) * transpose(a)` for `a = [[1, 2, 3], [4, 5, 6]]`, i.e. the element-wise square of the 3x2 transpose:

```text
1.000000 16.000000 
4.000000 25.000000 
9.000000 36.000000 
```

### 9.8 Error paths

Reading a program from stdin makes it easy to try the frontend's new diagnostics. A struct initializer with the wrong number of members is rejected by MLIRGen's type check (section 7.2):

```bash
cd /Users/roy/study/mlir/toy/Ch7
printf 'struct S {\n  var a;\n  var b;\n}\n\ndef main() {\n  S s = {[1, 2]};\n  print(s.a);\n}\n' \
  | ./build/toyc-ch7 -emit=mlir 2>&1
```

Actual output (exit status 1):

```text
loc("-":7:3): error: struct type of initializer is different than the variable declaration. Got '!toy.struct<tensor<*xf64>>', but expected '!toy.struct<tensor<*xf64>, tensor<*xf64>>'
```

A struct member with a shape hits the check in `mlirGen(StructAST&)` (section 7.1). The message is shared with the initializer check, so it talks about initializers, and it prints `error:` twice because `emitError` already adds the prefix:

```bash
cd /Users/roy/study/mlir/toy/Ch7
printf 'struct S {\n  var a<2>;\n}\n' | ./build/toyc-ch7 -emit=mlir 2>&1
```

Actual output (exit status 1):

```text
loc("-":2:3): error: error: variables within a struct definition must not have initializers
```

Accessing a member that does not exist reports the error but — surprisingly — still prints a module and exits with status 0:

```bash
cd /Users/roy/study/mlir/toy/Ch7
printf 'struct S {\n  var a;\n}\n\ndef main() {\n  S s = {[1, 2]};\n  print(s.b);\n}\n' \
  | ./build/toyc-ch7 -emit=mlir 2>&1
```

Actual output (exit status 0):

```text
loc("-":7:11): error: invalid access into struct expression
module {
  toy.func @main() {
    %0 = toy.struct_constant [dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf64>] : !toy.struct<tensor<*xf64>>
    toy.return
  }
}
```

The failing access sits inside a `print`, and `mlirGen(ExprASTList&)` returns `mlir::success()` when a `print` fails to codegen (inherited from the upstream tutorial code). So codegen of the block stops there without failing: the `print` *and every statement after it* are dropped (a valid `print(s.a);` on the next line does not appear in the output either), and only the implicit `toy.return` is added. The same bad access in a declaration (`var x = s.b;`) fails compilation with exit status 1.

The custom type parser has its own diagnostic, run in section 5.2.

### 9.9 The ecosystem view: what happens to a custom *type* outside its dialect

Chapter 2 ([section 6.7](../Ch2/README.md#67-the-ecosystem-view-feeding-toy-ir-to-stock-mlir-opt)) showed that stock `mlir-opt` handles unknown *ops* in generic form. This chapter adds a custom **type** — does `!toy.struct<...>` survive too?

```bash
cd /Users/roy/study/mlir/toy/Ch7
./build/toyc-ch7 ../../test_Example/Toy/Ch7/struct-codegen.toy -emit=mlir -mlir-print-op-generic 2>&1 \
  | /opt/homebrew/opt/llvm@20/bin/mlir-opt -allow-unregistered-dialect
```

Actual output:

```mlir
module {
  "toy.func"() <{function_type = (!toy.struct<tensor<*xf64>, tensor<*xf64>>) -> tensor<*xf64>, sym_name = "multiply_transpose"}> ({
  ^bb0(%arg0: !toy.struct<tensor<*xf64>, tensor<*xf64>>):
    %0 = "toy.struct_access"(%arg0) <{index = 0 : i64}> : (!toy.struct<tensor<*xf64>, tensor<*xf64>>) -> tensor<*xf64>
    %1 = "toy.transpose"(%0) : (tensor<*xf64>) -> tensor<*xf64>
    %2 = "toy.struct_access"(%arg0) <{index = 1 : i64}> : (!toy.struct<tensor<*xf64>, tensor<*xf64>>) -> tensor<*xf64>
    %3 = "toy.transpose"(%2) : (tensor<*xf64>) -> tensor<*xf64>
    %4 = "toy.mul"(%1, %3) : (tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>
    "toy.return"(%4) : (tensor<*xf64>) -> ()
  }) {sym_visibility = "private"} : () -> ()
  "toy.func"() <{function_type = () -> (), sym_name = "main"}> ({
    %0 = "toy.struct_constant"() <{value = [dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>, dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>]}> : () -> !toy.struct<tensor<*xf64>, tensor<*xf64>>
    %1 = "toy.generic_call"(%0) <{callee = @multiply_transpose}> : (!toy.struct<tensor<*xf64>, tensor<*xf64>>) -> tensor<*xf64>
    "toy.print"(%1) : (tensor<*xf64>) -> ()
    "toy.return"() : () -> ()
  }) : () -> ()
}
```

It round-trips — but for a different reason than the ops do. Generic *op* form is dialect-free structure, whereas `!toy.struct<tensor<*xf64>, tensor<*xf64>>` is still Toy's custom syntax; under `-allow-unregistered-dialect` MLIR parses it as an **opaque type** — the body between `<...>` is kept as an uninterpreted string and printed back verbatim. Everything this chapter implemented for the type is inert in that tool: no `parseType` structural validation (the diagnostic of section 5.2), no `StructType::getElementTypes()`, no verifier checking that `struct_access`'s index is in bounds, and of course no folding. The type has become a label rather than a semantic object — which is exactly the boundary between *carrying* IR and *understanding* it. Any tool that must reason about `!toy.struct` (verify it, fold it, lower it) has to link the dialect; a tool that merely transports IR does not. That division of labor is why MLIR text files are a viable interchange format between tools that know different subsets of the dialects involved.

---

## 10. Key Takeaways & Pitfalls

**Takeaways**

1. **Types are uniqued value objects.** A custom type = a `TypeStorage` subclass (`KeyTy`, `operator==`, `construct`) + a `TypeBase` wrapper + `addTypes<>()` in `initialize()`. Equality, hashing, and copying then come for free as pointer operations.
2. **The storage owns its memory.** `construct()` must copy everything through the `TypeStorageAllocator` (`allocator.copyInto(key)`) because types live as long as the `MLIRContext`, while the caller's `ArrayRef` may point at a dead stack frame.
3. **`hashKey`/`getKey` are optional** when `KeyTy` is `llvm::hash_value`-hashable and directly constructible from the `get()` arguments — the tutorial writes them anyway for pedagogy.
4. **Dialect hooks give the type syntax.** `parseType`/`printType` (declared by the dialect bit `useDefaultTypePrinterParser = 1`, not by the ODS type definition) handle everything after `!toy.`, and the parser doubles as a structural validator with real diagnostics.
5. **Bridge C++ types into ODS with `DialectType` + `CPred`,** then compose constraints (`AnyTypeOf<[F64Tensor, Toy_StructType]>`) and thread them through existing ops.
6. **`ConstantLike` + `hasFolder` + `materializeConstant` is the constant-propagation triad.** Fold hooks compute attribute results; the dialect materializer decides which op re-embodies an attribute of a given type.
7. **Design types to disappear.** No lowering was written for `StructType`; inlining plus folding erases it before the lowering pipeline runs. High-level abstractions that fold away are often cheaper than ones you must lower.

**Pitfalls**

- **Forgetting `allocator.copyInto(key)`** in `construct()` compiles fine and then reads freed memory — the classic bug in hand-written storage classes.
- **Forgetting `addTypes<StructType>()`** compiles fine and aborts the first time a `StructType` is created. With the line removed from a scratch copy of this chapter, both `struct-codegen.toy` and `struct-opt.mlir` die with `LLVM ERROR: can't create type 'mlir::toy::StructType' because storage uniquer isn't initialized: the dialect was likely not loaded, or the type wasn't added with addTypes<...>() in the Dialect::initialize() method.`
- **`static constexpr StringLiteral name = "toy.struct";` is required** on a hand-written type in MLIR 20: `addTypes<>()` reads `T::name` (section 3.2). Upstream's listing omits it.
- **Member casts are deprecated.** Upstream's `x.isa<T>()`, `x.cast<T>()` and `x.dyn_cast_or_null<T>()` on `Type`/`Attribute` still compile in MLIR 20, with `-Wdeprecated-declarations` warnings. That includes C++ pasted from a `CPred` string (section 4). Use `llvm::isa`/`llvm::cast`/`llvm::dyn_cast_if_present`.
- **`FoldAdaptor` operand accessors return null** when the operand isn't constant — always `dyn_cast_if_present`, never blind `cast` (see `StructAccessOp::fold`).
- **Fold results must not create ops** — return an `Attribute` or existing `Value` and let `materializeConstant` build ops; creating ops inside `fold()` is unsupported.
- **`.` in MLIRGen must not evaluate its RHS**: the RHS of an access is a member *name*, not an expression. Evaluating it would emit a bogus "unknown variable" error — hence the early intercept in `mlirGen(BinaryExprAST&)`.
- **Errors inside `print(...)` are swallowed**: a failing `print` argument stops codegen of the rest of the block, but compilation "succeeds" with exit status 0 (section 9.8). Check stderr, not only the exit status.
- **Verifier ordering matters**: `StructAccessOp::verify` bounds-checks the index *before* indexing `getElementTypes()`; also remember ODS constraints (`Toy_StructType` operand) are verified before your custom `verify()` runs, so `llvm::cast<StructType>` inside it is safe.
- **TableGen ordering out of tree**: the generated `.inc` files land under the chapter's binary dir (`Ch7/build/include/toy/*.inc` and `Ch7/build/ToyCombine.inc` standalone, the same layout under `toy/build/Ch7/` in the superbuild), so keep the `add_dependencies(toyc-ch7 ToyCh7OpsIncGen ...)` lines — Ninja's parallelism will otherwise race compilation against TableGen.

---

## Appendix: What Chapter 7 adds to the build

Nothing structural. [`Ch7/CMakeLists.txt`](CMakeLists.txt) is Ch6's build with the chapter number changed: the same `MLIR_ENABLE_EXECUTION_ENGINE` guard, the same three IncGen targets (`ToyCh7OpsIncGen`, `ToyCh7ShapeInferenceInterfaceIncGen`, `ToyCh7CombineIncGen`), the same source list, and the same shared-only link line. The chapter targets, below the standalone guard:

***CMakeLists.txt***
```cmake
# This chapter depends on JIT support enabled.
if(NOT MLIR_ENABLE_EXECUTION_ENGINE)
  return()
endif()

include_directories(include)
add_subdirectory(include)

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

The linking discipline is inherited from Chapter 6 — see [Chapter 6's build appendix](../Ch6/README.md#appendix-what-chapter-6-adds-to-the-build) and [MLIR_LINKING_PITFALL.md](../Ch6/MLIR_LINKING_PITFALL.md) for the full story.

Points worth noting precisely *because* nothing changed:

- **There is no TableGen step for the struct type.** `StructType` is entirely hand-written C++ in this chapter (section 3) — the TableGen'd `.inc` files still cover only ops, dialect, the shape-inference interface, and the DRR rewriters. (Later MLIR practice would define the type declaratively with ODS `TypeDef`; Ch7 deliberately shows the raw storage mechanism underneath.)
- **All struct support lives in already-existing files** (`Ops.td`, `Dialect.h`, `Dialect.cpp`, `MLIRGen.cpp`, `ToyCombine.cpp`, the lexer/parser/AST) — extending a dialect with a new type is a *content* change, not a build change.

---

## Links

- Official doc: [MLIR Toy Tutorial, Chapter 7 — Adding a Composite Type to Toy](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-7/)
- Related MLIR docs: [Defining Dialect Attributes and Types](https://mlir.llvm.org/docs/DefiningDialects/AttributesAndTypes/) · [Language Reference: dialect types](https://mlir.llvm.org/docs/LangRef/#dialect-types)
- Previous chapter: [Chapter 6: Lowering to LLVM & JIT](../Ch6/README.md)
- Back to: [README](../README.md)
- Code referenced in this chapter:
  - `include/toy/Lexer.h`, `AST.h`, `Parser.h`, `parser/AST.cpp` — language front-end
  - `include/toy/Dialect.h`, `mlir/Dialect.cpp` — `StructType`, storage, parse/print, verifiers, `materializeConstant`
  - `include/toy/Ops.td` — ODS: `Toy_StructType`, `Toy_Type`, `StructConstantOp`, `StructAccessOp`
  - `mlir/ToyCombine.cpp` — fold implementations
  - `mlir/MLIRGen.cpp` — struct codegen
  - `test_Example/Toy/Ch7/` — `struct-ast.toy`, `struct-codegen.toy`, `struct-opt.mlir`, `jit.toy`
