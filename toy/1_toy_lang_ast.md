# Chapter 1: Toy Language and AST

> Goal: define the Toy language, build a lexer/parser for it, and dump its AST — no MLIR involved yet (except the build setup). Official doc: [Toy Tutorial Chapter 1: Toy Language and AST](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-1/)

## 1. Overview — what this chapter builds and why

The MLIR Toy tutorial walks through building a complete compiler for a tiny tensor language called **Toy**, lowering it step by step through MLIR dialects all the way to LLVM IR and JIT execution. The tutorial is deliberately modeled after the classic [LLVM Kaleidoscope tutorial](https://llvm.org/docs/tutorial/MyFirstLanguageFrontend/index.html): before you can emit any IR, you need a *frontend* — a lexer, a parser, and an in-memory Abstract Syntax Tree (AST).

Chapter 1 builds exactly that frontend and nothing more. The deliverable is a small command-line tool, `toyc-ch1`, which:

1. Reads a `.toy` source file (or stdin).
2. Lexes it into tokens.
3. Parses the tokens with a hand-written recursive-descent parser into an AST.
4. Pretty-prints (dumps) that AST when invoked with `-emit=ast`.

Nothing in this chapter touches MLIR APIs yet — the AST is plain C++ classes. The only MLIR involvement is in the *build system*: we compile against the MLIR/LLVM headers and link `MLIRSupport`, which establishes the out-of-tree CMake plumbing that every later chapter builds on. Chapter 2 will take this same AST and emit real MLIR from it.

### This repo's layout (differs from upstream llvm-project)

Unlike the upstream tutorial code, which lives inside `llvm-project/mlir/examples/toy/` and is built as part of the monorepo, this study repo builds all chapters as one **out-of-tree CMake superbuild** rooted at `toy/`, against a prebuilt Homebrew LLVM/MLIR 20 installation. The top-level `toy/CMakeLists.txt` does the `find_package(MLIR/LLVM)` setup once and pulls in every chapter with `add_subdirectory(Ch1)`..`add_subdirectory(Ch7)`; each `ChN/` additionally remains configurable as a standalone project (see section 4.5).

| Item | Location / value |
|---|---|
| Repo root | `/Users/roy/study/mlir` |
| Superbuild root | `/Users/roy/study/mlir/toy/` |
| Chapter 1 code | `/Users/roy/study/mlir/toy/Ch1/` |
| MLIR CMake package | `MLIR_DIR=/opt/homebrew/opt/llvm@20/lib/cmake/mlir` (pinned in `CMakePresets.json`) |
| Compiler | `/opt/homebrew/opt/llvm@20/bin/clang++` (Homebrew clang 20.1.8, pinned in `CMakePresets.json`) |
| Platform / generator | macOS (Darwin), Ninja |
| Build | `cd toy && ./build.sh ch1` → binary at `./build/bin/toyc-ch1` |
| Run | `cd toy && ./run.sh ch1` → `./build/bin/toyc-ch1 ../test_Example/Toy/Ch1/ast.toy -emit=ast` |
| Test inputs | `/Users/roy/study/mlir/test_Example/Toy/Ch1/` (`ast.toy`, `empty.toy`) |

Relevant files in `toy/`:

```text
toy/
├── CMakeLists.txt          # superbuild: find_package(MLIR/LLVM) once, add_subdirectory(Ch1..Ch7)
├── CMakePresets.json       # pins Ninja, Release, Homebrew llvm@20 clang, MLIR_DIR/LLVM_DIR
├── build.sh                # ./build.sh [ch1..ch7|all] [--fresh] — configure once, build incrementally
├── run.sh                  # ./run.sh <ch1..ch7|all> — each chapter's demo commands
└── Ch1/
    ├── CMakeLists.txt      # dual-mode: chapter targets + standalone-only boilerplate guard
    ├── toyc.cpp            # driver: CLI options + main()
    ├── parser/
    │   └── AST.cpp         # AST pretty-printer (dumper)
    └── include/toy/
        ├── Lexer.h         # tokens + lexer (header-only)
        ├── Parser.h        # recursive-descent parser (header-only)
        └── AST.h           # AST node class hierarchy (header-only)
```

Note that the lexer, parser, and AST are all header-only; the only two translation units are `toyc.cpp` and `parser/AST.cpp`.

## 2. The Toy Language

Toy is a **tensor-based** language: you can define functions, do some math, and print results. It is intentionally minimal so that the interesting work happens in the compiler, not the language:

- **Only one element type**: every value is a 64-bit floating point number (`double` in C parlance, `f64` in MLIR). There are no integers, booleans, or strings.
- **Values are tensors** of rank ≤ 2 (scalars, 1-D arrays, and 2-D matrices), either implicitly shaped from an initializer or explicitly shaped in a declaration. For the tutorial's codegen we limit ourselves to tensors of rank 2 or less.
- **Values are immutable** — every operation returns a *new* value (this maps naturally onto SSA form later). Memory deallocation is automatic; you never free anything.
- **Two builtins**: `transpose()` and `print()`.
- **Comments** start with `#` and run to the end of the line.
- **Statements** end with `;` and function bodies are `{ ... }` blocks.

### A first example

Here is the introductory program from the official docs:

```text
def main() {
  # Define a variable `a` with shape <2, 3>, initialized with the literal value.
  # The shape is inferred from the supplied literal.
  var a = [[1, 2, 3], [4, 5, 6]];

  # b is identical to a, the literal tensor is implicitly reshaped: defining new
  # variables is the way to reshape tensors (element count must match).
  var b<2, 3> = [1, 2, 3, 4, 5, 6];

  # transpose() and print() are the only builtin, the following will transpose
  # a and b and perform an element-wise multiplication before printing the result.
  print(transpose(a) * transpose(b));
}
```

Things to notice:

- `var a = [[1, 2, 3], [4, 5, 6]];` — the shape `<2, 3>` is **inferred** from the nesting structure of the literal.
- `var b<2, 3> = [1, 2, 3, 4, 5, 6];` — an explicit shape on the declaration **implicitly reshapes** a flat 6-element literal into a 2×3 tensor. Declaring a new variable is the *only* way to reshape in Toy; the element count of the literal must match the declared shape.
- `*` is **element-wise** multiplication, not matrix multiplication.

### Generic functions and shape specialization

Type checking in Toy is done statically through **type inference**: the language barely requires type declarations. Functions are **generic over shapes** — a function's parameters are *unranked* (we know the values are tensors, but not their dimensions). Concrete shapes are only pinned down when the function is called; conceptually, the compiler **specializes** a function for each distinct set of argument shapes it is called with. This is the second example from the docs, and it is exactly the content of this repo's test file `/Users/roy/study/mlir/test_Example/Toy/Ch1/ast.toy`:

```text
# User defined generic function that operates on unknown shaped arguments.
def multiply_transpose(a, b) {
  return transpose(a) * transpose(b);
}

def main() {
  # Define a variable `a` with shape <2, 3>, initialized with the literal value.
  # The shape is inferred from the supplied literal.
  var a = [[1, 2, 3], [4, 5, 6]];
  # b is identical to a, the literal array is implicitly reshaped: defining new
  # variables is the way to reshape arrays (element count in literal must match
  # the size of specified shape).
  var b<2, 3> = [1, 2, 3, 4, 5, 6];

  # This call will specialize `multiply_transpose` with <2, 3> for both
  # arguments and deduce a return type of <3, 2> in initialization of `c`.
  var c = multiply_transpose(a, b);
  # A second call to `multiply_transpose` with <2, 3> for both arguments will
  # reuse the previously specialized and inferred version and return `<3, 2>`
  var d = multiply_transpose(b, a);
  # A new call with `<3, 2>` for both dimension will trigger another
  # specialization of `multiply_transpose`.
  var e = multiply_transpose(c, d);
  # Finally, calling into `multiply_transpose` with incompatible shapes
  # (<2, 3> and <3, 2>) will trigger a shape inference error.
  var f = multiply_transpose(a, c);
}
```

The comments spell out the intended future semantics:

- The first call `multiply_transpose(a, b)` with two `<2, 3>` arguments would create a specialization whose return type is deduced as `<3, 2>`.
- The second call `multiply_transpose(b, a)` has the same signature, so it would *reuse* that specialization.
- `multiply_transpose(c, d)` with two `<3, 2>` arguments would trigger a *new* specialization.
- `multiply_transpose(a, c)` mixes `<2, 3>` and `<3, 2>` — element-wise `*` requires matching shapes, so shape inference would report an **error**.

Important caveat for this chapter: **none of that is checked yet**. Chapter 1's parser performs *no semantic analysis at all* — no symbol resolution, no shape checking, not even a check that called functions exist. All of the shape-inference behavior described above is implemented in later chapters (shape inference is Chapter 4). Here, even the "incompatible" call in `var f = ...` parses without complaint, as you will see in the actual AST dump in section 5. (The file also carries `# RUN:` / `# CHECK:` comment lines used by LLVM's `lit`/`FileCheck` in upstream testing; to the Toy lexer they are simply comments.)

The second test input, `/Users/roy/study/mlir/test_Example/Toy/Ch1/empty.toy`, is a negative test — it contains only comments (no `def`), and expects the compiler to emit a parse error rather than crash:

```text
# RUN: toyc-ch1 %s -emit=ast 2>&1 | FileCheck %s
# CHECK-NOT: Assert
# CHECK: Parse error
```

### Informal grammar

Reading the parser (section 3) back into a grammar, Toy in Chapter 1 is:

```text
module        ::= definition*
definition    ::= prototype block
prototype     ::= 'def' identifier '(' decl_list? ')'
decl_list     ::= identifier (',' identifier)*
block         ::= '{' (block_expr ';')* '}'
block_expr    ::= decl | return | expression
decl          ::= 'var' identifier type? '=' expression
type          ::= '<' number (',' number)* '>'
return        ::= 'return' expression?
expression    ::= primary (binop primary)*        # binop ∈ { '+', '-', '*' }
primary       ::= identifierexpr | number | parenexpr | tensorliteral
identifierexpr::= identifier | identifier '(' (expression (',' expression)*)? ')'
parenexpr     ::= '(' expression ')'
tensorliteral ::= '[' literal_list ']' | number
literal_list  ::= tensorliteral (',' tensorliteral)*
```

## 3. Code Walkthrough

The frontend is intentionally "similar to the LLVM Kaleidoscope tutorial" (as the official doc says) and is not the interesting part of the MLIR tutorial — but it is worth understanding thoroughly once, because every later chapter consumes the AST it produces.

### 3.1 `include/toy/Lexer.h` — tokens and the lexer

The lexer is a single header. It starts with a `Location` struct that every token (and later, every AST node) carries — this becomes crucial in Chapter 2, where locations are attached to MLIR operations:

```cpp
/// Structure definition a location in a file.
struct Location {
  std::shared_ptr<std::string> file; ///< filename.
  int line;                          ///< line number.
  int col;                           ///< column number.
};
```

The filename is a `shared_ptr<string>` so that thousands of tokens/nodes can share one string instead of copying it.

**Token kinds.** The `Token` enum uses a classic Kaleidoscope trick: known multi-character tokens get *negative* values, while single-character punctuation is represented by its own ASCII code (positive), so the lexer can return `Token(lastChar)` for any character it doesn't specially recognize:

```cpp
enum Token : int {
  tok_semicolon = ';',
  tok_parenthese_open = '(',
  tok_parenthese_close = ')',
  tok_bracket_open = '{',
  tok_bracket_close = '}',
  tok_sbracket_open = '[',
  tok_sbracket_close = ']',

  tok_eof = -1,

  // commands
  tok_return = -2,
  tok_var = -3,
  tok_def = -4,

  // primary
  tok_identifier = -5,
  tok_number = -6,
};
```

So Toy has exactly three keywords (`return`, `var`, `def`), identifiers, numbers, and punctuation. Operators like `+`, `-`, `*`, `<`, `>`, `=`, `,` never appear in the enum at all — they flow through as raw ASCII tokens, and the parser compares against character literals like `'('` directly.

**The `Lexer` class** is an abstract base class: it implements all tokenization logic but delegates *input* to a subclass through one pure virtual hook:

```cpp
/// Delegate to a derived class fetching the next line. Returns an empty
/// string to signal end of file (EOF). Lines are expected to always finish
/// with "\n"
virtual llvm::StringRef readNextLine() = 0;
```

The concrete `LexerBuffer final : public Lexer` at the bottom of the file walks a `[begin, end)` memory buffer and serves it one line at a time. This split means you could just as easily lex from stdin line-by-line (e.g. for a REPL) without touching the tokenizer.

The public interface is what a parser wants: one token of lookahead plus accessors for the token's payload.

```cpp
Token getCurToken() { return curTok; }

/// Move to the next token in the stream and return it.
Token getNextToken() { return curTok = getTok(); }

/// Move to the next token in the stream, asserting on the current token
/// matching the expectation.
void consume(Token tok) {
  assert(tok == curTok && "consume Token mismatch expectation");
  getNextToken();
}

llvm::StringRef getId()   // valid only when curTok == tok_identifier
double getValue()         // valid only when curTok == tok_number
Location getLastLocation() // location of the START of the current token
```

`consume(tok)` is a debugging aid: it advances like `getNextToken()` but asserts the current token is what the caller believes it is.

**How `getNextToken` works.** `getNextToken()` just caches the result of the private workhorse `getTok()`, which is a hand-rolled state machine:

1. **Skip whitespace**, then record `lastLocation` (line/col of the first non-space char) so the token's location points at its beginning.
2. **Identifier / keyword**: if the character is alphabetic, greedily consume `[a-zA-Z0-9_]*`, then compare the spelling against `"return"`, `"def"`, `"var"`; anything else is `tok_identifier` (spelling saved in `identifierStr`).
3. **Number**: if it starts with a digit or `.`, consume `[0-9.]+` and convert with `strtod` into `numVal`, returning `tok_number`. (Note this accepts malformed input like `1.2.3` — `strtod` just stops early. Toy tolerates it for simplicity.)
4. **Comment**: on `#`, consume to end of line and *recurse* (`return getTok();`) to fetch the next real token.
5. **EOF**: return `tok_eof` without consuming it.
6. **Anything else**: return the character itself as a token (`Token(lastChar)`), advancing past it.

Two implementation details worth internalizing:

- The lexer always keeps **one character of lookahead** in `lastChar` (initialized to `' '`), because deciding where an identifier or number *ends* requires reading one char too far, and there is no "putback" into the stream.
- `getNextChar()` maintains the current line buffer, pulling a new line via `readNextLine()` when it drains, and updates `curLineNum`/`curCol` when it sees `'\n'`. That is the entire location-tracking machinery.

### 3.2 `include/toy/AST.h` — the AST class hierarchy

The AST is "optimized for simplicity, not efficiency": a tree of nodes owned via `std::unique_ptr<>`. First, a helper for declared types:

```cpp
/// A variable type with shape information.
struct VarType {
  std::vector<int64_t> shape;
};
```

An empty `shape` means "no shape specified — infer it later." This models `var a = ...` (shape `<>`) versus `var b<2, 3> = ...` (shape `{2, 3}`).

**The base class and LLVM-style RTTI.** All expressions derive from `ExprAST`, which carries a *kind* discriminator and a source `Location`:

```cpp
class ExprAST {
public:
  enum ExprASTKind {
    Expr_VarDecl,
    Expr_Return,
    Expr_Num,
    Expr_Literal,
    Expr_Var,
    Expr_BinOp,
    Expr_Call,
    Expr_Print,
  };

  ExprAST(ExprASTKind kind, Location location)
      : kind(kind), location(std::move(location)) {}
  virtual ~ExprAST() = default;

  ExprASTKind getKind() const { return kind; }
  const Location &loc() { return location; }
  ...
};
```

The explicit `kind` enum exists because LLVM code is built without C++ RTTI (`-fno-rtti`); instead it uses [LLVM-style RTTI](https://llvm.org/docs/HowToSetUpLLVMStyleRTTI.html), where each subclass provides a static `classof` predicate that `llvm::isa<>`, `llvm::cast<>`, and `llvm::dyn_cast<>` consult:

```cpp
/// LLVM style RTTI
static bool classof(const ExprAST *c) { return c->getKind() == Expr_Num; }
```

This is the exact same pattern used pervasively inside MLIR itself (for `Type`, `Attribute`, `Op` casting), so it is worth getting comfortable with here.

**Every subclass and what it represents:**

| Class | Kind | Represents | Payload |
|---|---|---|---|
| `NumberExprAST` | `Expr_Num` | numeric literal like `1.0` | `double val` |
| `LiteralExprAST` | `Expr_Literal` | tensor literal like `[[1, 2], [3, 4]]` | `vector<unique_ptr<ExprAST>> values` (numbers or nested literals) + `vector<int64_t> dims` |
| `VariableExprAST` | `Expr_Var` | a variable *reference* like `a` | `std::string name` |
| `VarDeclExprAST` | `Expr_VarDecl` | a declaration `var a<2,3> = expr` | `name`, `VarType type`, `unique_ptr<ExprAST> initVal` |
| `ReturnExprAST` | `Expr_Return` | `return;` or `return expr;` | `std::optional<unique_ptr<ExprAST>> expr` (optional!) |
| `BinaryExprAST` | `Expr_BinOp` | `lhs op rhs` | `char op` + owned `lhs`, `rhs` |
| `CallExprAST` | `Expr_Call` | user function call `f(a, b)` | `std::string callee` + arg list |
| `PrintExprAST` | `Expr_Print` | the builtin `print(x)` | single owned `arg` |

A few design decisions deserve comment:

- **`print` gets its own node** instead of being a generic `CallExprAST`. That's deliberate: in Chapter 2 `print` becomes its own dedicated MLIR operation (`toy.print`), so distinguishing it in the AST makes emission trivial.
- **`LiteralExprAST` stores dims separately from values.** The values are a nested tree mirroring the source brackets; `dims` is the flattened shape (e.g. `{2, 3}`), computed by the parser while checking that nesting is uniform.
- **Declarations and `return` are "expressions"** here only in the loose sense that they live in `ExprASTList` (a block's statement list); the language has no way to nest them inside other expressions.

Above expressions sit three structural classes that do *not* derive from `ExprAST`:

```cpp
using ExprASTList = std::vector<std::unique_ptr<ExprAST>>;

class PrototypeAST { Location location; std::string name;
                     std::vector<std::unique_ptr<VariableExprAST>> args; ... };

class FunctionAST { std::unique_ptr<PrototypeAST> proto;
                    std::unique_ptr<ExprASTList> body; ... };

class ModuleAST { std::vector<FunctionAST> functions; ... };
```

- `PrototypeAST` captures the function *signature* — but since Toy parameters are untyped/unranked, that is just the name and the parameter names (implicitly, the arity).
- `FunctionAST` = prototype + body (an `ExprASTList`, i.e. a block).
- `ModuleAST` = the whole translation unit, a list of functions, iterable via `begin()`/`end()`.

Finally, the header declares the one entry point implemented in `parser/AST.cpp`:

```cpp
void dump(ModuleAST &);
```

### 3.3 `include/toy/Parser.h` — recursive descent parsing

The `Parser` holds a reference to the lexer and exposes a single public method. Its class comment is worth quoting because it states the chapter's scope precisely:

```cpp
/// This is a simple recursive parser for the Toy language. It produces a well
/// formed AST from a stream of Token supplied by the Lexer. No semantic checks
/// or symbol resolution is performed. For example, variables are referenced by
/// string and the code could reference an undeclared variable and the parsing
/// succeeds.
```

**Top level: `parseModule`.** Primes the lexer with the first token, then loops `parseDefinition()` until EOF:

```cpp
std::unique_ptr<ModuleAST> parseModule() {
  lexer.getNextToken(); // prime the lexer

  // Parse functions one at a time and accumulate in this vector.
  std::vector<FunctionAST> functions;
  while (auto f = parseDefinition()) {
    functions.push_back(std::move(*f));
    if (lexer.getCurToken() == tok_eof)
      break;
  }
  // If we didn't reach EOF, there was an error during parsing
  if (lexer.getCurToken() != tok_eof)
    return parseError<ModuleAST>("nothing", "at end of module");

  return std::make_unique<ModuleAST>(std::move(functions));
}
```

Note the error convention used throughout: every `parse*` returns `nullptr` on failure after printing a message, and failures propagate upward by early returns — there is no exception handling and no error recovery.

**How each production is parsed:**

- `parseDefinition` — `definition ::= prototype block`. Calls `parsePrototype()` then `parseBlock()` and wraps them in a `FunctionAST`.

- `parsePrototype` — `prototype ::= def id '(' decl_list ')'`. Consumes `def`, the function name, `'('`, then a comma-separated list of identifiers, each becoming a `VariableExprAST` parameter node, then `')'`. No types anywhere — Toy prototypes only carry names.

- `parseBlock` — `block ::= { expression_list }`. After `'{'`, it loops until `'}'`/EOF, dispatching on the current token:

  ```cpp
  if (lexer.getCurToken() == tok_var) {        // Variable declaration
    auto varDecl = parseDeclaration();
    ...
  } else if (lexer.getCurToken() == tok_return) { // Return statement
    auto ret = parseReturn();
    ...
  } else {                                     // General expression
    auto expr = parseExpression();
    ...
  }
  // Ensure that elements are separated by a semicolon.
  if (lexer.getCurToken() != ';')
    return parseError<ExprASTList>(";", "after expression");
  ```

  Both before the loop and after each statement it "swallows" runs of semicolons, so empty statements (`;;`) are tolerated.

- `parseDeclaration` — `decl ::= var identifier [type] = expr`. Eats `var` and the name, then *optionally* calls `parseType()` if it sees `'<'`; a missing type becomes a default-constructed (empty-shape) `VarType`. Then it consumes `'='` and parses the initializer expression.

- `parseType` — `type ::= '<' shape_list '>'` where `shape_list ::= num (',' num)*`. Just accumulates numbers into `VarType::shape`.

- `parseReturn` — `return ::= return ; | return expr ;` — the operand is optional (`std::optional`), decided by peeking for `';'`.

- `parsePrimary` — dispatches on the current token: identifier → `parseIdentifierExpr`, number → `parseNumberExpr`, `'('` → `parseParenExpr`, `'['` → `parseTensorLiteralExpr`; `';'` and `'}'` return `nullptr` (they signal "no expression here" to callers), and anything else prints `unknown token ... when expecting an expression`.

- `parseIdentifierExpr` — the classic one-token-lookahead trick: after eating the identifier, if the next token is *not* `'('` it's a plain `VariableExprAST` reference; otherwise it's a call, and the parser collects comma-separated argument expressions. Then comes the special-casing of the builtin:

  ```cpp
  // It can be a builtin call to print
  if (name == "print") {
    if (args.size() != 1)
      return parseError<ExprAST>("<single arg>", "as argument to print()");
    return std::make_unique<PrintExprAST>(std::move(loc), std::move(args[0]));
  }

  // Call to a user-defined function
  return std::make_unique<CallExprAST>(std::move(loc), name, std::move(args));
  ```

  So `print` is resolved *syntactically*, by name, at parse time — and it is arity-checked (exactly one argument), the only "semantic" check in the whole parser. `transpose`, by contrast, stays a generic `CallExprAST` until Chapter 2 special-cases it during MLIR generation.

- `parseTensorLiteralExpr` — `tensorLiteral ::= [ literalList ] | number`. This is the most interesting production. At each `[` nesting level it collects elements (either numbers or recursively parsed nested literals) into `values`, then computes the shape:

  1. `dims.push_back(values.size())` — the current level's extent.
  2. If any element is itself a `LiteralExprAST`, take the first element's `dims` and append them, then **verify every sibling has identical dims** — this rejects ragged arrays like `[[1, 2], [3]]` with the error `expected 'uniform well-nested dimensions' inside literal expression`.

  The result is a `LiteralExprAST` whose `dims` for `[[1, 2, 3], [4, 5, 6]]` is `{2, 3}`.

**Operator precedence: `parseExpression` + `parseBinOpRHS`.** Binary expressions use *operator-precedence climbing*, the same algorithm as Kaleidoscope. The precedence table is:

```cpp
int getTokPrecedence() {
  if (!isascii(lexer.getCurToken()))
    return -1;

  // 1 is lowest precedence.
  switch (static_cast<char>(lexer.getCurToken())) {
  case '-': return 20;
  case '+': return 20;
  case '*': return 40;
  default:  return -1;
  }
}
```

Only `+`, `-` (precedence 20) and `*` (precedence 40) are operators; every other token returns −1, meaning "not a binop — stop." `parseExpression` parses a primary as the LHS, then hands it to `parseBinOpRHS(0, lhs)`, which loops:

```cpp
std::unique_ptr<ExprAST> parseBinOpRHS(int exprPrec,
                                       std::unique_ptr<ExprAST> lhs) {
  while (true) {
    int tokPrec = getTokPrecedence();

    // If this is a binop that binds at least as tightly as the current binop,
    // consume it, otherwise we are done.
    if (tokPrec < exprPrec)
      return lhs;

    int binOp = lexer.getCurToken();
    lexer.consume(Token(binOp));
    auto loc = lexer.getLastLocation();

    auto rhs = parsePrimary();
    ...
    // If BinOp binds less tightly with rhs than the operator after rhs, let
    // the pending operator take rhs as its lhs.
    int nextPrec = getTokPrecedence();
    if (tokPrec < nextPrec) {
      rhs = parseBinOpRHS(tokPrec + 1, std::move(rhs));
      ...
    }

    // Merge lhs/RHS.
    lhs = std::make_unique<BinaryExprAST>(std::move(loc), binOp,
                                          std::move(lhs), std::move(rhs));
  }
}
```

Worked example, `a + b * c + d`:

1. LHS = `a`; sees `+` (20 ≥ 0), parses RHS primary `b`.
2. Peeks at `*` (40 > 20), so it recurses `parseBinOpRHS(21, b)`, which grabs `b * c` and stops at the second `+` (20 < 21).
3. Merges into `(a + (b*c))`, loops; second `+` (20 ≥ 0) grabs `d`, giving `((a + (b*c)) + d)` — correct precedence *and* left associativity.

**Error reporting** is centralized in a small template that prints the expectation, context, and the lexer's location, then returns `nullptr` typed for the caller:

```cpp
template <typename R, typename T, typename U = const char *>
std::unique_ptr<R> parseError(T &&expected, U &&context = "") {
  auto curToken = lexer.getCurToken();
  llvm::errs() << "Parse error (" << lexer.getLastLocation().line << ", "
               << lexer.getLastLocation().col << "): expected '" << expected
               << "' " << context << " but has Token " << curToken;
  if (isprint(curToken))
    llvm::errs() << " '" << (char)curToken << "'";
  llvm::errs() << "\n";
  return nullptr;
}
```

### 3.4 `parser/AST.cpp` — the AST dumper

This file implements the `toy::dump(ModuleAST&)` declared in `AST.h`. It is a straightforward tree walk with pretty indentation, and it introduces two idioms you will keep seeing in MLIR code.

**Idiom 1: RAII indentation.** The current indent level is a plain `int` on the `ASTDumper`; a tiny guard bumps it on entry to a node and restores it on scope exit:

```cpp
// RAII helper to manage increasing/decreasing the indentation as we traverse
// the AST
struct Indent {
  Indent(int &level) : level(level) { ++level; }
  ~Indent() { --level; }
  int &level;
};

#define INDENT()                                                               \
  Indent level_(curIndent);                                                    \
  indent();
```

Every `dump(SomeNode*)` overload starts with `INDENT();` — increment the level for this node's subtree, print the leading spaces (two per level), and let the destructor pop the level when the function returns. There is no manual decrement anywhere, so the indentation can never get out of sync even on early returns.

**Idiom 2: `llvm::TypeSwitch` for dispatch.** Instead of a virtual `dump()` method on each AST class (which would tangle printing into the data model), dispatch happens externally via LLVM-style RTTI:

```cpp
/// Dispatch to a generic expressions to the appropriate subclass using RTTI
void ASTDumper::dump(ExprAST *expr) {
  llvm::TypeSwitch<ExprAST *>(expr)
      .Case<BinaryExprAST, CallExprAST, LiteralExprAST, NumberExprAST,
            PrintExprAST, ReturnExprAST, VarDeclExprAST, VariableExprAST>(
          [&](auto *node) { this->dump(node); })
      .Default([&](ExprAST *) {
        // No match, fallback to a generic message
        INDENT();
        llvm::errs() << "<unknown Expr, kind " << expr->getKind() << ">\n";
      });
}
```

`TypeSwitch` tries `dyn_cast` against each listed class (using the `classof` hooks from `AST.h`) and calls the lambda with the concrete pointer type — the generic lambda (`auto *node`) then picks the right `dump` overload at compile time.

Each per-node printer is small. Two representative ones:

```cpp
void ASTDumper::dump(VarDeclExprAST *varDecl) {
  INDENT();
  llvm::errs() << "VarDecl " << varDecl->getName();
  dump(varDecl->getType());                    // prints "<2, 3>" or "<>"
  llvm::errs() << " " << loc(varDecl) << "\n";
  dump(varDecl->getInitVal());                 // recurse into initializer
}
```

Every node line ends with a location produced by a helper that formats `@file:line:col`:

```cpp
template <typename T>
static std::string loc(T *node) {
  const auto &loc = node->loc();
  return (llvm::Twine("@") + *loc.file + ":" + llvm::Twine(loc.line) + ":" +
          llvm::Twine(loc.col)).str();
}
```

Tensor literals get special treatment: a free function `printLitHelper` recurses through nested literals and prints the **dims in angle brackets at every nesting level**, so `[[1, 2], [3, 4]]` prints as `<2,2>[<2>[ 1, 2 ], <2>[ 3, 4 ] ]`. `llvm::interleaveComma` (another ubiquitous LLVM helper) handles the comma separation.

One practical detail: the dumper writes to **`llvm::errs()` — stderr, not stdout**. If you want to pipe or save the AST dump, redirect with `2>&1`.

### 3.5 `toyc.cpp` — the driver

The driver is only ~70 lines. Command-line handling uses LLVM's `cl` library, which turns declarative global option objects into a full argv parser:

```cpp
static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input toy file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));
namespace {
enum Action { None, DumpAST };
} // namespace

static cl::opt<enum Action>
    emitAction("emit", cl::desc("Select the kind of output desired"),
               cl::values(clEnumValN(DumpAST, "ast", "output the AST dump")));
```

- `inputFilename` is a positional argument defaulting to `"-"`, and `llvm::MemoryBuffer::getFileOrSTDIN` treats `-` as standard input — so `echo 'def main() {}' | ./build/bin/toyc-ch1 -emit=ast` works.
- `emitAction` defines `-emit=<value>`; in Chapter 1 the only value is `ast`. Later chapters extend this same enum with `mlir`, `mlir-affine`, `llvm`, `jit`, etc.

Parsing is wrapped in a helper that wires file → lexer → parser:

```cpp
std::unique_ptr<toy::ModuleAST> parseInputFile(llvm::StringRef filename) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(filename);
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return nullptr;
  }
  auto buffer = fileOrErr.get()->getBuffer();
  LexerBuffer lexer(buffer.begin(), buffer.end(), std::string(filename));
  Parser parser(lexer);
  return parser.parseModule();
}
```

And `main` is a parse-then-dispatch:

```cpp
int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv, "toy compiler\n");

  auto moduleAST = parseInputFile(inputFilename);
  if (!moduleAST)
    return 1;

  switch (emitAction) {
  case Action::DumpAST:
    dump(*moduleAST);
    return 0;
  default:
    llvm::errs() << "No action specified (parsing only?), use -emit=<action>\n";
  }
  return 0;
}
```

If you run without `-emit=ast`, the file is still parsed (so you get parse errors if any), but nothing is emitted except the reminder message. Also note that because LLVM's `cl` machinery links in options from every registered LLVM component, `./build/bin/toyc-ch1 --help` prints *hundreds* of inherited LLVM options (`--aarch64-neon-syntax`, etc.) around the two that actually matter here — don't be alarmed.

## 4. Building

All chapters share one **superbuild**: configure once at `toy/`, then build any chapter incrementally into the shared `toy/build/` tree. The everyday workflow is:

```bash
cd /Users/roy/study/mlir/toy
./build.sh ch1          # build only toyc-ch1  →  ./build/bin/toyc-ch1
./build.sh              # build everything (toyc-ch1 .. toyc-ch7)
./build.sh ch1 --fresh  # wipe build/ first, then rebuild
```

Or, equivalently, drive CMake directly via the presets:

```bash
cmake --preset default                          # configure (once)
cmake --build --preset default --target toyc-ch1
```

### 4.1 The superbuild: `toy/CMakeLists.txt`, line by line

The top-level CMakeLists does the **out-of-tree boilerplate once** for every chapter (upstream instead uses in-tree helpers like `add_toy_chapter` inside the monorepo build). Here is the full file with commentary:

```cmake
cmake_minimum_required(VERSION 3.20)

if(APPLE)
  set(CMAKE_OSX_DEPLOYMENT_TARGET "26.0" CACHE STRING "macOS Deployment Target" FORCE)
endif()

project(toy-tutorial)
```

- `cmake_minimum_required(3.20)` matches LLVM 20's own minimum.
- On macOS the deployment target is pinned (here to the host OS major version) to avoid the linker warning/mismatch that occurs when Homebrew's LLVM was built against a different `MACOSX_DEPLOYMENT_TARGET` than CMake's default.
- `project()` declares an ordinary C++ project — we are *not* inside the LLVM build.

```cmake
find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
```

- `CONFIG` mode makes CMake locate the *installed package config files* (`MLIRConfig.cmake`, `LLVMConfig.cmake`) rather than a `FindMLIR.cmake` module. Resolution is driven by `MLIR_DIR=/opt/homebrew/opt/llvm@20/lib/cmake/mlir` (set as a cache variable by `CMakePresets.json`; `MLIRConfig.cmake` finds its sibling LLVM automatically, and the explicit `find_package(LLVM)` makes `LLVM_CMAKE_DIR` etc. available too). These configs import every prebuilt MLIR/LLVM library as a CMake target — which is why the chapters can later write `target_link_libraries(... MLIRSupport)` without any manual `-L`/`-l` flags.

```cmake
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)
```

- The two `list(APPEND CMAKE_MODULE_PATH ...)` lines let plain `include(<name>)` find LLVM/MLIR's helper scripts inside the Homebrew install.
- `TableGen`, `AddLLVM`, and `AddMLIR` define macros such as `add_llvm_executable`, `mlir_tablegen`, and `add_mlir_dialect`; `HandleLLVMOptions` sets the compiler flags LLVM expects (e.g. `-fno-rtti`, matching the LLVM-style RTTI discussed in section 3.2). Chapter 1 doesn't strictly need the TableGen machinery (it uses plain `add_executable`), but Chapter 2+ runs TableGen for dialect definitions, and doing all of this once at the top level is exactly what makes the superbuild work.

```cmake
include_directories(${MLIR_INCLUDE_DIRS}
                    ${LLVM_INCLUDE_DIRS})
```

- Adds the installed MLIR and LLVM header directories globally, so `#include "llvm/ADT/StringRef.h"` etc. resolve. Ordering matters conceptually: packages first, module path second, includes third — each step depends on variables produced by the previous one.

```cmake
# Collect every chapter binary in build/bin/ instead of build/ChN/.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(Ch1)
add_subdirectory(Ch2)
...
add_subdirectory(Ch7)
```

- `CMAKE_RUNTIME_OUTPUT_DIRECTORY` redirects every executable into a single `build/bin/` directory — so the Chapter 1 binary lands at `toy/build/bin/toyc-ch1` rather than `toy/build/Ch1/toyc-ch1`.
- Each `add_subdirectory(ChN)` pulls in that chapter's targets; the boilerplate above is already in effect for all of them.

### 4.2 `CMakePresets.json`

The toolchain choices live in a preset instead of shell flags, so a bare `cmake --preset default` reproduces the exact same configuration every time:

```json
{
  "name": "default",
  "displayName": "Homebrew LLVM/MLIR 20 (Ninja, Release)",
  "generator": "Ninja",
  "binaryDir": "${sourceDir}/build",
  "cacheVariables": {
    "CMAKE_BUILD_TYPE": "Release",
    "CMAKE_C_COMPILER": "/opt/homebrew/opt/llvm@20/bin/clang",
    "CMAKE_CXX_COMPILER": "/opt/homebrew/opt/llvm@20/bin/clang++",
    "MLIR_DIR": "/opt/homebrew/opt/llvm@20/lib/cmake/mlir",
    "LLVM_DIR": "/opt/homebrew/opt/llvm@20/lib/cmake/llvm"
  }
}
```

This pins the **Ninja** generator, a **Release** build, the Homebrew llvm@20 `clang`/`clang++` (keeping the compiler consistent with the prebuilt libraries), and the `MLIR_DIR`/`LLVM_DIR` package locations that `find_package` needs. A matching build preset (also named `default`) lets `cmake --build --preset default [--target toyc-chN]` work without naming the build directory.

### 4.3 `Ch1/CMakeLists.txt` — dual-mode chapter file

The chapter's own CMakeLists is now **dual-mode**: the out-of-tree boilerplate from section 4.1 is repeated here, but wrapped in a guard so it runs *only* when the chapter is configured standalone. In the superbuild, `CMAKE_SOURCE_DIR` is `toy/` while `CMAKE_CURRENT_SOURCE_DIR` is `toy/Ch1/`, so the guard is false and the top level's setup is used instead:

```cmake
cmake_minimum_required(VERSION 3.20)

# ---------------------------------------------------------------------------
# Standalone-mode boilerplate.
# Runs only when this chapter is configured directly (cmake -S Ch1).
# In the superbuild (cmake -S toy/), ../CMakeLists.txt already did all this.
# ---------------------------------------------------------------------------
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  if(APPLE)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "26.0" CACHE STRING "macOS Deployment Target" FORCE)
  endif()
  project(toy-ch1)

  find_package(MLIR REQUIRED CONFIG)
  find_package(LLVM REQUIRED CONFIG)

  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

  include(TableGen)
  include(AddLLVM)
  include(AddMLIR)
  include(HandleLLVMOptions)

  include_directories(${MLIR_INCLUDE_DIRS}
                      ${LLVM_INCLUDE_DIRS})
endif()
```

The targets section below the guard runs in *both* modes and is unchanged from before the restructure:

```cmake
# ---------------------------------------------------------------------------
# Chapter targets
# ---------------------------------------------------------------------------
add_executable(toyc-ch1
  toyc.cpp
  parser/AST.cpp
)

include_directories(include/)
target_link_libraries(toyc-ch1
  PRIVATE
    MLIRSupport
)
```

- Exactly two translation units, as noted earlier — `Lexer.h`, `Parser.h`, and `AST.h` are header-only.
- `include_directories(include/)` makes `#include "toy/AST.h"` resolve to `Ch1/include/toy/AST.h`.
- **Why only `MLIRSupport`?** Chapter 1 uses no MLIR IR at all — only LLVM *support* utilities (`llvm::StringRef`, `MemoryBuffer`, `raw_ostream`, `cl::opt`, `TypeSwitch`, `Twine`, `interleaveComma`, casting). `MLIRSupport` is MLIR's small support library, and linking it transitively pulls in `LLVMSupport` (and friends) that actually provide those symbols. The upstream in-tree CMakeLists does the equivalent: it, too, links only `MLIRSupport` for Ch1. Dialect/IR libraries (`MLIRIR`, `MLIRParser`, ...) only appear starting in Chapter 2 when we build real MLIR operations.

### 4.4 `build.sh` and artifacts

`toy/build.sh` is a thin wrapper over the presets, with usage `./build.sh [ch1..ch7|all] [--fresh]`:

1. With `--fresh` it first runs `rm -rf build` — otherwise the build tree is **kept and reused incrementally** (no more per-chapter clean rebuilds).
2. If `build/CMakeCache.txt` does not exist yet, it configures via `cmake --preset default`; afterwards Ninja re-runs CMake automatically whenever a `CMakeLists.txt` changes.
3. Then it builds: `cmake --build --preset default` for `all`, or `cmake --build --preset default --target toyc-chN` for a single chapter. Verified toolchain: Homebrew clang 20.1.8, CMake 4.3.1.

Expected artifacts in `toy/build/`:

```text
CMakeCache.txt  CMakeFiles/  build.ninja  cmake_install.cmake  Ch1/ ... Ch7/  bin/
```

The ones you care about live in **`build/bin/`**: `toyc-ch1` (and, after a full build, `toyc-ch2` .. `toyc-ch7`).

```bash
cd /Users/roy/study/mlir/toy && ./build.sh ch1
```

### 4.5 Standalone chapter builds still work

Thanks to the dual-mode guard, a chapter can still be configured as its own project — useful for experimenting with one chapter in isolation. Since there is no preset at the chapter level, pass the toolchain settings explicitly:

```bash
cd /Users/roy/study/mlir/toy
cmake -S Ch1 -B Ch1/build -G Ninja \
      -DMLIR_DIR=/opt/homebrew/opt/llvm@20/lib/cmake/mlir \
      -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm@20/bin/clang++
cmake --build Ch1/build
```

This produces `Ch1/build/toyc-ch1`; `run.sh` even checks that location as a fallback when `build/bin/toyc-ch1` doesn't exist. The superbuild is the primary workflow, though — the standalone mode is secondary.

## 5. Running and Testing

### 5.1 `run.sh` and what the flags mean

The single `toy/run.sh` holds every chapter's demo commands (`./run.sh <ch1..ch7|all>`); its Chapter 1 case is:

```bash
cd /Users/roy/study/mlir/toy && ./run.sh ch1
# which runs (with build/bin/toyc-ch1 from the superbuild):
./build/bin/toyc-ch1 ../test_Example/Toy/Ch1/ast.toy -emit=ast
```

- `../test_Example/Toy/Ch1/ast.toy` — the positional `<input toy file>` (relative to `toy/`, i.e. `/Users/roy/study/mlir/test_Example/Toy/Ch1/ast.toy`). This is the same program as `mlir/test/Examples/Toy/Ch1/ast.toy` upstream.
- `-emit=ast` — selects `Action::DumpAST` in the driver: parse the file and pretty-print the resulting AST to **stderr**.

(`run.sh` looks up binaries in `build/bin/` first and falls back to `ChN/build/` for standalone chapter builds; it `cd`s to its own directory, so the relative test paths work no matter where you invoke it from.)

### 5.2 Actual captured output

Real output from `cd /Users/roy/study/mlir/toy && ./run.sh ch1` on this machine (complete, not truncated):

```text
  Module:
    Function 
      Proto 'multiply_transpose' @../test_Example/Toy/Ch1/ast.toy:4:1
      Params: [a, b]
      Block {
        Return
          BinOp: * @../test_Example/Toy/Ch1/ast.toy:5:25
            Call 'transpose' [ @../test_Example/Toy/Ch1/ast.toy:5:10
              var: a @../test_Example/Toy/Ch1/ast.toy:5:20
            ]
            Call 'transpose' [ @../test_Example/Toy/Ch1/ast.toy:5:25
              var: b @../test_Example/Toy/Ch1/ast.toy:5:35
            ]
      } // Block
    Function 
      Proto 'main' @../test_Example/Toy/Ch1/ast.toy:8:1
      Params: []
      Block {
        VarDecl a<> @../test_Example/Toy/Ch1/ast.toy:11:3
          Literal: <2, 3>[ <3>[ 1.000000e+00, 2.000000e+00, 3.000000e+00], <3>[ 4.000000e+00, 5.000000e+00, 6.000000e+00]] @../test_Example/Toy/Ch1/ast.toy:11:11
        VarDecl b<2, 3> @../test_Example/Toy/Ch1/ast.toy:15:3
          Literal: <6>[ 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00] @../test_Example/Toy/Ch1/ast.toy:15:17
        VarDecl c<> @../test_Example/Toy/Ch1/ast.toy:19:3
          Call 'multiply_transpose' [ @../test_Example/Toy/Ch1/ast.toy:19:11
            var: a @../test_Example/Toy/Ch1/ast.toy:19:30
            var: b @../test_Example/Toy/Ch1/ast.toy:19:33
          ]
        VarDecl d<> @../test_Example/Toy/Ch1/ast.toy:22:3
          Call 'multiply_transpose' [ @../test_Example/Toy/Ch1/ast.toy:22:11
            var: b @../test_Example/Toy/Ch1/ast.toy:22:30
            var: a @../test_Example/Toy/Ch1/ast.toy:22:33
          ]
        VarDecl e<> @../test_Example/Toy/Ch1/ast.toy:25:3
          Call 'multiply_transpose' [ @../test_Example/Toy/Ch1/ast.toy:25:11
            var: c @../test_Example/Toy/Ch1/ast.toy:25:30
            var: d @../test_Example/Toy/Ch1/ast.toy:25:33
          ]
        VarDecl f<> @../test_Example/Toy/Ch1/ast.toy:28:3
          Call 'multiply_transpose' [ @../test_Example/Toy/Ch1/ast.toy:28:11
            var: a @../test_Example/Toy/Ch1/ast.toy:28:30
            var: c @../test_Example/Toy/Ch1/ast.toy:28:33
          ]
      } // Block
```

### 5.3 Mapping the dump back to the source

Every `@file:line:col` in the dump is a `Location` captured by the lexer at the *start* of the corresponding token — walk them back into `ast.toy`:

- **`Module:`** — the root `ModuleAST`; each `Function` under it is one `def`.
- **`Proto 'multiply_transpose' @...:4:1` / `Params: [a, b]`** — line 4 of `ast.toy` is `def multiply_transpose(a, b) {` (lines 1–3 are comments, and line counting starts at 1). The prototype records only the name and parameter names — no types, because Toy prototypes are shape-generic.
- **`BinOp: * @...:5:25`** — line 5 is `  return transpose(a) * transpose(b);`. The `*` node is the child of `Return`. Its location (col 25) is actually where the *RHS* begins — an artifact of `parseBinOpRHS` taking `lexer.getLastLocation()` *after* consuming the operator. Its two children are the `Call 'transpose'` nodes at cols 10 and 25; note `transpose` is dumped as an ordinary `Call`, not a special node — it is not special-cased in Chapter 1.
- **`VarDecl a<> @...:11:3`** — `var a = [[1, 2, 3], [4, 5, 6]];`. The `<>` is the empty `VarType` printed by `dump(const VarType&)`: no declared shape, to be inferred. Its child `Literal: <2, 3>[ <3>[ 1.0..., ...` shows the parser-computed dims at every nesting level (`<2, 3>` outer, `<3>` per row) and every number as a `double` in scientific notation (`1.000000e+00`) — remember, Toy's only type is 64-bit float.
- **`VarDecl b<2, 3> @...:15:3`** — `var b<2, 3> = [1, 2, 3, 4, 5, 6];`. Here the declared type prints as `<2, 3>` but the initializer literal is `<6>[ ... ]` — a rank-1, 6-element tensor. The AST faithfully preserves the mismatch; the *implicit reshape* is a semantic notion handled in later chapters (a `toy.reshape` op in Chapter 2), not in the AST.
- **`VarDecl c<>` … `VarDecl f<>`** — the four calls to `multiply_transpose` from lines 19/22/25/28, each a `Call` node listing its argument `var:` references with exact source coordinates. Notice `var f = multiply_transpose(a, c);` — the shape-incompatible call — dumps identically to the others: **no error**, because Chapter 1 does zero semantic checking. The FileCheck comments at the bottom of `ast.toy` assert exactly this output shape upstream.

Two practical notes: the dump goes to **stderr** (use `2>&1` to pipe it, exactly as the `# RUN:` line in `ast.toy` does), and the indentation starts at one level (`  Module:`) because even the root `dump(ModuleAST*)` executes `INDENT()`.

### 5.4 The error path: `empty.toy`

```bash
cd /Users/roy/study/mlir/toy
./build/bin/toyc-ch1 ../test_Example/Toy/Ch1/empty.toy -emit=ast
```

Actual output:

```text
Parse error (4, 0): expected 'def' in prototype but has Token -1
  Module:
```

`empty.toy` contains only comments, so the first real token is `tok_eof` (`-1`). `parsePrototype` fails with the message above (via the `parseError` helper — note it prints the raw token integer, and `-1` maps to `tok_eof`). Interestingly, `parseModule` then observes that the current token *is* EOF, so it still returns a valid — empty — `ModuleAST`, which dumps as a bare `Module:` line, and the process exits with status 0. The upstream FileCheck test only asserts that a `Parse error` is printed and that no assertion fires (`CHECK-NOT: Assert`); it is a robustness test, not an exit-code test.

You can also feed the compiler from stdin (the default input `-`):

```bash
echo 'def main() { print([1, 2]); }' | ./build/bin/toyc-ch1 -emit=ast
```

## 6. Key Takeaways & Pitfalls

**Takeaways**

- Chapter 1 is a pure Kaleidoscope-style frontend: header-only lexer (`Lexer.h`), header-only recursive-descent parser with precedence climbing (`Parser.h`), `unique_ptr`-owned AST (`AST.h`), and a `TypeSwitch`-based dumper (`parser/AST.cpp`). MLIR appears only in the build system.
- The Toy language: rank ≤ 2 tensors of `f64` only, immutable values, `#` comments, keywords `def`/`var`/`return`, builtins `transpose`/`print`, element-wise `*`, shape inference plus per-call-signature function specialization (semantics deferred to later chapters).
- Patterns introduced here recur throughout MLIR proper: **LLVM-style RTTI** (`classof` + `isa/cast/dyn_cast/TypeSwitch`), **`Location` tracking on every node** (feeds MLIR location metadata in Chapter 2), and LLVM support utilities (`cl::opt`, `MemoryBuffer`, `raw_ostream`, `Twine`, `interleaveComma`).
- Parsing and semantics are cleanly separated: the parser only enforces *structure* (plus literal-shape uniformity and `print` arity). Undeclared variables, unknown callees, and shape mismatches all parse fine.
- Out-of-tree builds against an installed MLIR need exactly three configure-time ingredients: `find_package(MLIR/LLVM CONFIG)`, `CMAKE_MODULE_PATH` += their cmake dirs, `include(TableGen/AddLLVM/AddMLIR/HandleLLVMOptions)` — then imported targets like `MLIRSupport` just work. In this repo that boilerplate lives **once** in the top-level `toy/CMakeLists.txt`; each chapter repeats it only inside a `CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR` guard for standalone use.

**Pitfalls**

- **The AST dump goes to stderr**, not stdout. `./build/bin/toyc-ch1 file.toy -emit=ast > out.txt` produces an empty file; use `2> out.txt` or `2>&1`.
- **Forgetting `-emit=ast`** silently does parse-only and prints `No action specified (parsing only?), use -emit=<action>` — easy to misread as a failure.
- **Exit codes are not a reliable error signal** in Chapter 1: the `empty.toy` parse error still exits 0 with an empty `Module:` dump, because `parseModule` treats "stopped exactly at EOF" as success.
- **Wrong `MLIR_DIR`** is the classic out-of-tree failure: if `find_package(MLIR)` can't locate `/opt/homebrew/opt/llvm@20/lib/cmake/mlir`, configuration dies immediately. `CMakePresets.json` pins it (along with `LLVM_DIR` and the compilers) for the superbuild, but a *standalone* chapter configure must pass `-DMLIR_DIR=...` by hand. Also keep the compiler consistent with the library (`/opt/homebrew/opt/llvm@20/bin/clang++`) — mixing Apple Clang's libc++ with Homebrew LLVM 20 binaries can cause ABI-flavored link/runtime surprises, and the `CMAKE_OSX_DEPLOYMENT_TARGET` pin exists to silence version-mismatch warnings.
- **The build tree is now shared and incremental**: plain `./build.sh` never deletes `build/`; only `--fresh` runs `rm -rf build`, and that wipes *all* chapters' objects at once — don't keep anything precious in there. `run.sh`'s test paths (`../test_Example/...`) are relative to `toy/`, but the script `cd`s to its own directory first, so invoking it from elsewhere is safe.
- **Locations are 1-based lines, and a `BinaryExprAST`'s location points at its RHS** (taken after consuming the operator) — don't be surprised when `BinOp: *` reports column 25 for an operator at column 23.
- The `--help` output is flooded with generic LLVM options inherited from `cl::opt` registration; only the positional filename and `-emit` belong to `toyc-ch1`.

## Links

- Official doc: [Toy Tutorial Chapter 1: Toy Language and AST](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-1/)
- Kaleidoscope (the frontend this chapter mirrors): [LLVM Kaleidoscope Tutorial](https://llvm.org/docs/tutorial/MyFirstLanguageFrontend/index.html)
- Next chapter: [Chapter 2](2_emitting_basic_mlir.md)
- Back to [README](README.md)
