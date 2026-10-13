# Standalone: An Out-of-Tree MLIR Dialect

> **Goal:** Build a complete MLIR project *outside* the LLVM source tree: a dialect with a custom type, an operation and a pass, defined with ODS (Operation Definition Specification) and TableGen; an `mlir-opt`-style driver and an `mlir-translate`-style driver; a plugin that loads the dialect into the stock `mlir-opt`; a C API; and a lit test suite. Based on the upstream example [`mlir/examples/standalone`](https://github.com/llvm/llvm-project/tree/release/20.x/mlir/examples/standalone).

---

## 1. Overview

The Toy tutorial (`../toy/`) teaches how to *define* a dialect, one concept per chapter. This project shows how to *package* one: the directory layout, the CMake wiring and the tools a real out-of-tree MLIR project needs, all built against an installed MLIR (Homebrew `llvm@20`), with no LLVM checkout. It is small on purpose. The dialect has one type, one op and one pass, so every file can be read in full, and it is meant to be copied and renamed as the starting point of your own project (section 13 does exactly that).

What the project contains, and where each part is explained:

| Part | Files | Section |
| --- | --- | --- |
| Top-level CMake | `CMakeLists.txt` | 2 |
| The `standalone` dialect | `include/Standalone/StandaloneDialect.td`, `lib/Standalone/StandaloneDialect.cpp` | 3 |
| A custom type `!standalone.custom<"...">` | `StandaloneTypes.td`, `StandaloneTypes.cpp` | 4 |
| An operation `standalone.foo` | `StandaloneOps.td`, `StandaloneOps.cpp` | 5 |
| A pass `--standalone-switch-bar-foo` | `StandalonePasses.td`, `StandalonePasses.cpp` | 6 |
| The optimizer driver `standalone-opt` | `standalone-opt/` | 7 |
| The translation driver `standalone-translate` | `standalone-translate/` | 8 |
| A dialect and pass plugin for stock `mlir-opt` | `standalone-plugin/` | 9 |
| A C API and Python bindings | `include/Standalone-c/`, `lib/CAPI/`, `python/` | 10 |
| Tests (lit and FileCheck) | `test/` | 11 |

Every command in this README runs from `/Users/roy/study/mlir/standalone` and expects the project to be built first: run the build block in section 12.1 once (about two minutes), then any command below. Sections 2–11 explain the project part by part, each with its code, the exact command and its real output. Section 12 is the build and the test suite, section 13 shows how to derive the project (and your own dialect) yourself, and section 14 collects the takeaways. The appendices hold reference material: every `CMakeLists.txt` and the generated files (A), the generated dialect documentation (B), and the script that runs every command in this README (C).

---

## 2. The Project Layout and the Out-of-Tree CMake

### 2.1 What lives where

The layout mirrors MLIR's own source tree, which is what lets the MLIR CMake helpers work unchanged:

```text
standalone/
├── CMakeLists.txt                 top level: find MLIR, include its CMake modules
├── include/
│   ├── Standalone/                ODS (.td) files and the C++ headers that include
│   │                              what TableGen generates from them
│   └── Standalone-c/Dialects.h    C API header
├── lib/
│   ├── Standalone/                the dialect library MLIRStandalone (.cpp files)
│   └── CAPI/Dialects.cpp          the C API library StandaloneCAPI
├── standalone-opt/                optimizer driver (like mlir-opt)
├── standalone-translate/          translation driver (like mlir-translate)
├── standalone-plugin/             plugin for the stock mlir-opt
├── python/                        Python bindings (needs MLIR built with them)
└── test/                          lit configuration and tests
```

`include/` holds what other code includes (the `.td` files and headers); `lib/` holds what gets compiled into a library; each tool has its own directory. The generated `.inc` files land in `build/include/Standalone/`, outside the source tree, and `#include "Standalone/StandaloneOps.h.inc"` still works because both `include/` and `build/include/` are on the include path (section 2.2).

### 2.2 The top-level `CMakeLists.txt`

***CMakeLists.txt***
```cmake
cmake_minimum_required(VERSION 3.20.0)
project(standalone-dialect LANGUAGES CXX C)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)                 # (1)
  find_package(MLIR REQUIRED CONFIG)                                   # (2)

  message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

  set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)              # (3)
  set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
  set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")                   # (4)
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

  include(TableGen)
  include(AddLLVM)
  include(AddMLIR)
  include(HandleLLVMOptions)
else()                                                                 # (5)
  # Build via external projects mechanism
  set(MLIR_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../mlir)
  set(MLIR_INCLUDE_DIR ${MLIR_MAIN_SRC_DIR}/include)
  set(MLIR_GENERATED_INCLUDE_DIR ${LLVM_BINARY_DIR}/tools/mlir/include)
  set(MLIR_INCLUDE_DIRS "${MLIR_INCLUDE_DIR};${MLIR_GENERATED_INCLUDE_DIR}")
endif()

if(MLIR_ENABLE_BINDINGS_PYTHON)                                        # (6)
  include(MLIRDetectPythonEnv)
  mlir_configure_python_dev_packages()
endif()

set(STANDALONE_SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(STANDALONE_BINARY_DIR ${PROJECT_BINARY_DIR})
include_directories(${LLVM_INCLUDE_DIRS})                              # (7)
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${STANDALONE_SOURCE_DIR}/include)
include_directories(${STANDALONE_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(include)                                              # (8)
add_subdirectory(lib)
if(MLIR_ENABLE_BINDINGS_PYTHON)
  message(STATUS "Enabling Python API")
  add_subdirectory(python)
endif()
add_subdirectory(test)
add_subdirectory(standalone-opt)
add_subdirectory(standalone-plugin)
add_subdirectory(standalone-translate)
```

**(1)** The project supports two build modes. When `standalone/` is the top of the CMake tree (`cmake -S standalone`), it is a **component build** against an installed MLIR, which is what this README uses. **(5)** Otherwise it is being built *inside* an LLVM build through `LLVM_EXTERNAL_PROJECTS` (a **monolithic build**, section 12.1), and MLIR's paths come from the enclosing LLVM build instead.

**(2)** `find_package(MLIR REQUIRED CONFIG)` looks for `MLIRConfig.cmake`. Homebrew installs it under `/opt/homebrew/opt/llvm@20/lib/cmake/mlir`, and because this shell has `/opt/homebrew/opt/llvm@20/bin` on `PATH`, CMake finds it from there without `-DMLIR_DIR` (section 12.1 shows the message it prints). `MLIRConfig.cmake` in turn finds LLVM and defines `MLIR_INCLUDE_DIRS`, `MLIR_CMAKE_DIR`, `MLIR_ENABLE_BINDINGS_PYTHON` and the targets of every installed MLIR library.

**(3)** Executables go to `build/bin/` and libraries to `build/lib/`, the same layout as an LLVM build tree. The lit configuration relies on it (section 11.1).

**(4)** These four modules provide the functions the rest of the project calls: `TableGen` (`mlir_tablegen`, `add_public_tablegen_target`), `AddLLVM` (`add_llvm_executable`, `add_llvm_library`, `add_lit_testsuite`), `AddMLIR` (`add_mlir_dialect`, `add_mlir_dialect_library`, `add_mlir_doc`, the C API and Python helpers) and `HandleLLVMOptions` (compiler flags matching how LLVM itself was built, such as `-fno-exceptions -fno-rtti`).

**(6)** The Python bindings are built only if MLIR itself was built with them (section 10.3).

**(7)** Both `include/` (hand-written headers, `.td` files) and `build/include/` (generated `.inc` files) are on the include path, so a header and the code generated from it are included with the same `Standalone/...` prefix.

**(8)** The subdirectories, in dependency order: `include/` runs TableGen, `lib/` compiles the dialect, then the tools and tests that link it.

Because the project uses the same helpers as MLIR's in-tree dialects, a dialect written here can move into the LLVM tree (or the other way round) with almost no CMake changes. **Out-of-tree does not mean a different build system: it means the same MLIR CMake functions, reached through `find_package` instead of the LLVM source tree.**

---

## 3. The Dialect

### 3.1 `StandaloneDialect.td`

A dialect is a namespace for operations, types and attributes, plus the hooks that parse, print and verify them. ODS describes it in a TableGen record:

***include/Standalone/StandaloneDialect.td***
```tablegen
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Standalone dialect definition.
//===----------------------------------------------------------------------===//

def Standalone_Dialect : Dialect {
    let name = "standalone";                                  // (1)
    let summary = "A standalone out-of-tree MLIR dialect.";   // (2)
    let description = [{
        This dialect is an example of an out-of-tree MLIR dialect designed to
        illustrate the basic setup required to develop MLIR-based tools without
        working inside of the LLVM source tree.
    }];
    let cppNamespace = "::mlir::standalone";                  // (3)

    let useDefaultTypePrinterParser = 1;                      // (4)
    let extraClassDeclaration = [{                            // (5)
        void registerTypes();
    }];
}

//===----------------------------------------------------------------------===//
// Base standalone operation definition.
//===----------------------------------------------------------------------===//

class Standalone_Op<string mnemonic, list<Trait> traits = []> :   // (6)
        Op<Standalone_Dialect, mnemonic, traits>;
```

**(1)** `name` is the namespace that prefixes everything in the dialect: `standalone.foo`, `!standalone.custom<...>`. **(2)** `summary` and `description` end up in the generated documentation (appendix B). **(3)** `cppNamespace` is where the generated C++ classes live: `mlir::standalone::StandaloneDialect` and the rest. **(4)** `useDefaultTypePrinterParser = 1` asks ODS to generate the dialect's `parseType`/`printType`, which dispatch on the type mnemonic (section 4.2); without it you would write both by hand. **(5)** `extraClassDeclaration` pastes C++ into the generated class; here it declares `registerTypes()`, which `StandaloneTypes.cpp` defines (section 4.3). **(6)** `Standalone_Op` is the base class every op of the dialect derives from, so each op definition only states its mnemonic and traits.

### 3.2 The generated dialect class

`add_mlir_dialect` (appendix A.2) runs `mlir-tblgen -gen-dialect-decls` on the dialect. The same command, run by hand, prints the class:

```bash
cd /Users/roy/study/mlir/standalone
mlir-tblgen -gen-dialect-decls -I include -I /opt/homebrew/opt/llvm@20/include include/Standalone/StandaloneOps.td
```

Real output **[1]**:
```text
/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: StandaloneOps.td                                                     *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace standalone {

class StandaloneDialect : public ::mlir::Dialect {
  explicit StandaloneDialect(::mlir::MLIRContext *context);

  void initialize();
  friend class ::mlir::MLIRContext;
public:
  ~StandaloneDialect() override;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("standalone");
  }

  /// Parse a type registered to this dialect.
  ::mlir::Type parseType(::mlir::DialectAsmParser &parser) const override;

  /// Print a type registered to this dialect.
  void printType(::mlir::Type type,
                 ::mlir::DialectAsmPrinter &os) const override;

        void registerTypes();
    };
} // namespace standalone
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::standalone::StandaloneDialect)
```

The `-I` flags are the two include roots: `include/` for `Standalone/...` and the Homebrew include directory for `mlir/IR/OpBase.td`. What the class contains:

| Member | What it is for |
| --- | --- |
| private constructor, `friend class ::mlir::MLIRContext` | Only a context creates a dialect: `context->getOrLoadDialect<StandaloneDialect>()`. One instance per context. |
| `initialize()` | Declared, not defined: you write it (section 3.3). It adds the ops and types to the dialect. |
| `getDialectNamespace()` | Returns `"standalone"`, from `let name`. Used by the parser to route `standalone.*` to this dialect. |
| `parseType` / `printType` | Generated because of `useDefaultTypePrinterParser = 1`; their definitions come from the *type* generator (section 4.2). |
| `registerTypes()` | Pasted in verbatim from `extraClassDeclaration`. |
| `MLIR_DECLARE_EXPLICIT_TYPE_ID` | Gives the class a `TypeID`, the identity MLIR uses instead of C++ RTTI (the project is built with `-fno-rtti`). |

The definitions are short: the constructor registers the namespace and the `TypeID`, then calls your `initialize()`:

```bash
cd /Users/roy/study/mlir/standalone
mlir-tblgen -gen-dialect-defs -I include -I /opt/homebrew/opt/llvm@20/include include/Standalone/StandaloneOps.td
```

Real output **[2]**:
```text
/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: StandaloneOps.td                                                     *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::standalone::StandaloneDialect)
namespace mlir {
namespace standalone {

StandaloneDialect::StandaloneDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<StandaloneDialect>())
    
     {
  
  initialize();
}

StandaloneDialect::~StandaloneDialect() = default;

} // namespace standalone
} // namespace mlir
```

### 3.3 The hand-written half: headers and `StandaloneDialect.cpp`

Each generated `.inc` file is included by exactly one hand-written header, so the rest of the code includes ordinary headers:

***include/Standalone/StandaloneDialect.h***
```cpp
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"

#include "Standalone/StandaloneOpsDialect.h.inc"
```

***include/Standalone/StandaloneOps.h***
```cpp
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "Standalone/StandaloneOps.h.inc"
```

The generated file names follow the first argument of `add_mlir_dialect(StandaloneOps standalone)`: `StandaloneOps.h.inc` (ops), `StandaloneOpsTypes.h.inc` (types), `StandaloneOpsDialect.h.inc` (dialect). That is why the dialect header includes a file called `StandaloneOpsDialect.h.inc` rather than `StandaloneDialect.h.inc`. The `#include`s in `StandaloneOps.h` provide what the generated op class refers to: `OpDefinition.h` for `mlir::Op`, and the two interface headers for the traits `standalone.foo` uses (section 5.2).

The one function the dialect needs is `initialize()`:

***lib/Standalone/StandaloneDialect.cpp***
```cpp
#include "Standalone/StandaloneDialect.h"
#include "Standalone/StandaloneOps.h"
#include "Standalone/StandaloneTypes.h"

using namespace mlir;
using namespace mlir::standalone;

#include "Standalone/StandaloneOpsDialect.cpp.inc"           // (1)

//===----------------------------------------------------------------------===//
// Standalone dialect.
//===----------------------------------------------------------------------===//

void StandaloneDialect::initialize() {
  addOperations<
#define GET_OP_LIST                                          // (2)
#include "Standalone/StandaloneOps.cpp.inc"
      >();
  registerTypes();                                           // (3)
}
```

**(1)** The dialect definitions from section 3.2 (constructor, destructor, `TypeID`). **(2)** With `GET_OP_LIST` defined, `StandaloneOps.cpp.inc` expands to a comma-separated list of every op class, here just `::mlir::standalone::FooOp`, so `addOperations<...>` never has to be edited when an op is added to the `.td` file. **(3)** Types are registered in `StandaloneTypes.cpp`, which is the only file that includes the type definitions (section 4.3).

**A registered dialect owns its namespace.** Once the `standalone` dialect is loaded, an op name such as `standalone.impure` that it does not define is an error, even in generic form and even with `--allow-unregistered-dialect`, which only covers namespaces no loaded dialect claims:

```bash
cd /Users/roy/study/mlir/standalone
./build/bin/standalone-opt --allow-unregistered-dialect <<'EOF'
func.func @f(%x: i32) -> i32 {
  %0 = "standalone.impure"(%x) : (i32) -> i32
  return %0 : i32
}
EOF
```

Real output **[3]** (exit status 1):
```text
<stdin>:2:8: error: unregistered operation 'standalone.impure' found in dialect ('standalone') that does not allow unknown operations
  %0 = "standalone.impure"(%x) : (i32) -> i32
       ^
<stdin>:2:8: note: see current operation: %0 = "standalone.impure"(%arg0) : (i32) -> i32
```

The dialect is the authority for every name under its prefix, which is what lets the verifier trust that each `standalone.*` op it sees has a C++ class behind it.

---

## 4. A Custom Type: `!standalone.custom<"...">`

### 4.1 `StandaloneTypes.td`

Types are defined like ops, with a `TypeDef` record. A type can carry **parameters**, the data that distinguishes one instance from another (as `i32` and `i64` differ in width). `!standalone.custom` carries one string:

***include/Standalone/StandaloneTypes.td***
```tablegen
include "mlir/IR/AttrTypeBase.td"
include "Standalone/StandaloneDialect.td"

//===----------------------------------------------------------------------===//
// Standalone type definitions
//===----------------------------------------------------------------------===//

class Standalone_Type<string name, string typeMnemonic, list<Trait> traits = []>    // (1)
    : TypeDef<Standalone_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def Standalone_CustomType : Standalone_Type<"Custom", "custom"> {
    let summary = "Standalone custom type";
    let description = "Custom type in standalone dialect";
    let parameters = (ins StringRefParameter<"the custom value">:$value);           // (2)
    let assemblyFormat = "`<` $value `>`";                                          // (3)
}
```

**(1)** `Standalone_Type` plays the role `Standalone_Op` plays for ops: it ties every type to the dialect. The first template argument (`"Custom"`) names the C++ class (`CustomType`); the second (`"custom"`) is the **mnemonic**, the keyword after `!standalone.` in the IR. **(2)** `parameters` lists the type's data: one `StringRefParameter` named `value`, which becomes a `getValue()` accessor. **(3)** `assemblyFormat` declares the syntax: a literal `<`, the parameter, a literal `>`. The full spelling is `!standalone.custom<"10">`.

### 4.2 The generated type class

`StandaloneTypes.td` is included by `StandaloneOps.td`, so the type generator runs on the ops file, restricted to this dialect with `-typedefs-dialect`:

```bash
cd /Users/roy/study/mlir/standalone
mlir-tblgen -gen-typedef-decls -typedefs-dialect=standalone -I include -I /opt/homebrew/opt/llvm@20/include include/Standalone/StandaloneOps.td
```

Real output **[4]**:
```text
/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* TypeDef Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_TYPEDEF_CLASSES
#undef GET_TYPEDEF_CLASSES


namespace mlir {
class AsmParser;
class AsmPrinter;
} // namespace mlir
namespace mlir {
namespace standalone {
class CustomType;
namespace detail {
struct CustomTypeStorage;
} // namespace detail
class CustomType : public ::mlir::Type::TypeBase<CustomType, ::mlir::Type, detail::CustomTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "standalone.custom";
  static constexpr ::llvm::StringLiteral dialectName = "standalone";
  static CustomType get(::mlir::MLIRContext *context, ::llvm::StringRef value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"custom"};
  }

  static ::mlir::Type parse(::mlir::AsmParser &odsParser);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::llvm::StringRef getValue() const;
};
} // namespace standalone
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::standalone::CustomType)

#endif  // GET_TYPEDEF_CLASSES

```

- `CustomType` derives from `Type::TypeBase<CustomType, Type, detail::CustomTypeStorage>`. A `Type` is a thin value object (a pointer); the data lives in a **storage** object owned by the `MLIRContext`.
- `get(context, value)` is the only way to create one. It returns the *same* storage for the same `value`: types are **uniqued**, so two `!standalone.custom<"10">` are the same pointer and compare with `==`.
- `name`, `dialectName` and `getMnemonic()` come from the record; `parse`/`print` come from `assemblyFormat`; `getValue()` comes from the parameter.

The definitions show how uniquing works and what the assembly format turned into:

```bash
cd /Users/roy/study/mlir/standalone
mlir-tblgen -gen-typedef-defs -typedefs-dialect=standalone -I include -I /opt/homebrew/opt/llvm@20/include include/Standalone/StandaloneOps.td | sed -n '/^struct CustomTypeStorage/,/^} \/\/ namespace detail/p;/^::mlir::Type CustomType::parse/,/^}/p;/^void CustomType::print/,/^}/p'
```

Real output **[5]**:
```text
struct CustomTypeStorage : public ::mlir::TypeStorage {
  using KeyTy = std::tuple<::llvm::StringRef>;
  CustomTypeStorage(::llvm::StringRef value) : value(std::move(value)) {}

  KeyTy getAsKey() const {
    return KeyTy(value);
  }

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static CustomTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, KeyTy &&tblgenKey) {
    auto value = std::move(std::get<0>(tblgenKey));
    value = allocator.copyInto(value);
    return new (allocator.allocate<CustomTypeStorage>()) CustomTypeStorage(std::move(value));
  }

  ::llvm::StringRef value;
};
} // namespace detail
::mlir::Type CustomType::parse(::mlir::AsmParser &odsParser) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<std::string> _result_value;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'value'
  _result_value = ::mlir::FieldParser<std::string>::parse(odsParser);
  if (::mlir::failed(_result_value)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse Standalone_CustomType parameter 'value' which is to be a `::llvm::StringRef`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_value));
  return CustomType::get(odsParser.getContext(),
      ::llvm::StringRef((*_result_value)));
}
void CustomType::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter.printString(getValue());;
  odsPrinter << ">";
}
```

- `CustomTypeStorage::KeyTy` is the tuple of parameters. The context hashes the key (`hashKey`) and compares it (`operator==`) to find an existing instance; only when none exists does it call `construct`, which copies the string into the context's allocator (`allocator.copyInto`). The copy matters: a `StringRef` does not own its characters, and the type must outlive the parser's buffer.
- `parse` is the assembly format read left to right: `parseLess()` for `` `<` ``, a `FieldParser<std::string>` for `$value` (it expects a quoted string), `parseGreater()` for `` `>` ``. `print` is the same sequence in reverse.

The dialect-level `parseType` from section 3.2 is defined here too. It reads the mnemonic and dispatches to the matching type's `parse`:

```bash
cd /Users/roy/study/mlir/standalone
mlir-tblgen -gen-typedef-defs -typedefs-dialect=standalone -I include -I /opt/homebrew/opt/llvm@20/include include/Standalone/StandaloneOps.td | sed -n '/^static ::mlir::OptionalParseResult generatedTypeParser/,/^}/p;/^::mlir::Type StandaloneDialect::parseType/,/^}/p'
```

Real output **[6]**:
```text
static ::mlir::OptionalParseResult generatedTypeParser(::mlir::AsmParser &parser, ::llvm::StringRef *mnemonic, ::mlir::Type &value) {
  return ::mlir::AsmParser::KeywordSwitch<::mlir::OptionalParseResult>(parser)
    .Case(::mlir::standalone::CustomType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::standalone::CustomType::parse(parser);
      return ::mlir::success(!!value);
    })
    .Default([&](llvm::StringRef keyword, llvm::SMLoc) {
      *mnemonic = keyword;
      return std::nullopt;
    });
}
::mlir::Type StandaloneDialect::parseType(::mlir::DialectAsmParser &parser) const {
  ::llvm::SMLoc typeLoc = parser.getCurrentLocation();
  ::llvm::StringRef mnemonic;
  ::mlir::Type genType;
  auto parseResult = generatedTypeParser(parser, &mnemonic, genType);
  if (parseResult.has_value())
    return genType;
  
  parser.emitError(typeLoc) << "unknown  type `"
      << mnemonic << "` in dialect `" << getNamespace() << "`";
  return {};
}
```

### 4.3 `StandaloneTypes.cpp`

***lib/Standalone/StandaloneTypes.cpp***
```cpp
#include "Standalone/StandaloneTypes.h"

#include "Standalone/StandaloneDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir::standalone;

#define GET_TYPEDEF_CLASSES
#include "Standalone/StandaloneOpsTypes.cpp.inc"

void StandaloneDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "Standalone/StandaloneOpsTypes.cpp.inc"
      >();
}
```

The same two-pass pattern as the ops: `GET_TYPEDEF_CLASSES` pulls in the definitions (storage, `parse`, `print`, and the dialect's `parseType`/`printType`), and `GET_TYPEDEF_LIST` expands to the list of type classes for `addTypes<...>`. `TypeSwitch.h` and `DialectImplementation.h` are there because the generated `printType` and `parseType` use them. This file defines `registerTypes()` rather than `StandaloneDialect.cpp` so that the type definitions are compiled in one translation unit only.

### 4.4 The type in action

A function taking two `!standalone.custom` arguments round-trips through `standalone-opt`, which parses the IR and prints it back:

```bash
cd /Users/roy/study/mlir/standalone
./build/bin/standalone-opt <<'EOF'
func.func @f(%a: !standalone.custom<"10">, %b: !standalone.custom<"hello world">) {
  return
}
EOF
```

Real output **[7]**:
```text
module {
  func.func @f(%arg0: !standalone.custom<"10">, %arg1: !standalone.custom<"hello world">) {
    return
  }
}

```

The parameter must be a quoted string, because `StringRefParameter` parses with `FieldParser<std::string>`. A bare `10` fails inside the generated `parse`, which adds its own message after the parser's:

```bash
cd /Users/roy/study/mlir/standalone
./build/bin/standalone-opt <<'EOF'
func.func @f(%arg0: !standalone.custom<10>) { return }
EOF
```

Real output **[8]** (exit status 1):
```text
<stdin>:1:40: error: expected string
func.func @f(%arg0: !standalone.custom<10>) { return }
                                       ^
<stdin>:1:40: error: failed to parse Standalone_CustomType parameter 'value' which is to be a `::llvm::StringRef`
func.func @f(%arg0: !standalone.custom<10>) { return }
                                       ^
```

An unknown mnemonic never reaches a type's `parse`: `generatedTypeParser` falls through to `Default`, and the dialect's `parseType` reports it (the double space in `unknown  type` is in the generated code, section 4.2):

```bash
cd /Users/roy/study/mlir/standalone
./build/bin/standalone-opt <<'EOF'
func.func @f(%arg0: !standalone.other<"x">) { return }
EOF
```

Real output **[9]** (exit status 1):
```text
<stdin>:1:33: error: unknown  type `other` in dialect `standalone`
func.func @f(%arg0: !standalone.other<"x">) { return }
                                ^
```

---

## 5. An Operation: `standalone.foo`

### 5.1 `StandaloneOps.td`

***include/Standalone/StandaloneOps.td***
```tablegen
include "Standalone/StandaloneTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Standalone_FooOp : Standalone_Op<"foo", [Pure,                 // (1)
                                             SameOperandsAndResultType]> {
    let summary = "Illustrates how to define an operation.";
    let description = [{
        ...
    }];

    let arguments = (ins I32:$input);                              // (2)
    let results = (outs I32:$res);

    let assemblyFormat = [{                                        // (3)
        $input attr-dict `:` type($input)
    }];
}
```

**(1)** `Standalone_Op<"foo", ...>` makes the op `standalone.foo` (C++ class `FooOp`). Its two traits:

- **`Pure`** declares that the op has no side effects and is always safe to execute speculatively. It is defined in `SideEffectInterfaces.td`, hence the include, and it is what lets generic passes such as CSE (common subexpression elimination) and DCE (dead code elimination) delete or merge the op (section 5.5).
- **`SameOperandsAndResultType`** declares that the result has the operand's type. ODS then also attaches `InferTypeOpInterface` (from `InferTypeOpInterface.td`), so the result type can be computed instead of spelled out.

**(2)** One operand named `input` and one result named `res`, both constrained to `I32`, a 32-bit signless integer. The names become accessors (`getInput()`, `getRes()`); the constraints become verifier checks (section 5.3).

**(3)** The custom syntax: the operand, an optional attribute dictionary, a colon and the operand's type. The result type is not written; the parser fills it in from `SameOperandsAndResultType` (section 5.4). The full syntax is `%1 = standalone.foo %0 : i32`. The `description` body (elided above, because it contains an `mlir` code fence) holds that example; appendix B shows it in the generated documentation.

### 5.2 The generated op class

```bash
cd /Users/roy/study/mlir/standalone
mlir-tblgen -gen-op-decls -I include -I /opt/homebrew/opt/llvm@20/include include/Standalone/StandaloneOps.td | sed -n '/^class FooOp :/,/^};/p'
```

Real output **[10]**:
```text
class FooOp : public ::mlir::Op<FooOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::IntegerType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::SameOperandsAndResultType, ::mlir::InferTypeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = FooOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = FooOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("standalone.foo");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::operand_range getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(getOperation()->operand_begin(), valueRange.first),
             std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::IntegerType> getInput() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSOperands(0).begin());
  }

  ::mlir::OpOperand &getInputMutable() {
    auto range = getODSOperandIndexAndLength(0);
    return getOperation()->getOpOperand(range.first);
  }

  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index) {
    return {index, 1};
  }

  ::mlir::Operation::result_range getODSResults(unsigned index) {
    auto valueRange = getODSResultIndexAndLength(index);
    return {std::next(getOperation()->result_begin(), valueRange.first),
             std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
  }

  ::mlir::TypedValue<::mlir::IntegerType> getRes() {
    return ::llvm::cast<::mlir::TypedValue<::mlir::IntegerType>>(*getODSResults(0).begin());
  }

  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value input);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::llvm::LogicalResult verifyInvariantsImpl();
  ::llvm::LogicalResult verifyInvariants();
  static ::llvm::LogicalResult inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
```

- **The trait list** in `Op<FooOp, ...>` records everything ODS knows about the op's structure: `ZeroRegions`, `OneResult`, `OneTypedResult<IntegerType>`, `ZeroSuccessors`, `OneOperand` come from the argument and result lists; `OpInvariants` hooks in the generated verifier; `ConditionallySpeculatable`, `AlwaysSpeculatableImplTrait` and `MemoryEffectOpInterface` are what `Pure` expands to; `SameOperandsAndResultType` and `InferTypeOpInterface` come from the second trait.
- **Accessors**: `getInput()` and `getRes()` return `TypedValue<IntegerType>`, a `Value` already known to have an integer type; `getInputMutable()` returns the operand slot for in-place updates.
- **Five `build` methods.** Besides the usual "result type plus operand" form there are forms *without* a result type (`build(builder, state, input)`): because the op implements `InferTypeOpInterface`, they call `inferReturnTypes` to compute it.
- **Verification** is `verifyInvariantsImpl()` (section 5.3); **syntax** is `parse`/`print` (section 5.4); **side effects** are reported by `getEffects`, which for a `Pure` op adds no effect at all.
- `Adaptor` / `GenericAdaptor` give the same named accessors over a plain list of values, for code that has the operands but not the op yet (folding, conversion patterns).

### 5.3 The generated verifier

```bash
cd /Users/roy/study/mlir/standalone
mlir-tblgen -gen-op-defs -I include -I /opt/homebrew/opt/llvm@20/include include/Standalone/StandaloneOps.td | sed -n '/^static ::llvm::LogicalResult __mlir_ods_local_type_constraint/,/^}/p;/FooOp::verifyInvariantsImpl() {/,/^}/p'
```

Real output **[11]**:
```text
static ::llvm::LogicalResult __mlir_ods_local_type_constraint_StandaloneOps1(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isSignlessInteger(32)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be 32-bit signless integer, but got " << type;
  }
  return ::mlir::success();
}
::llvm::LogicalResult FooOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_StandaloneOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_StandaloneOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}
```

`I32` turned into one shared constraint function, `type.isSignlessInteger(32)`, called once for the operand and once for the result, with the message built from the value kind and index. An `i64` operand fails it:

```bash
cd /Users/roy/study/mlir/standalone
./build/bin/standalone-opt <<'EOF'
func.func @f(%x: i64) -> i64 {
  %0 = standalone.foo %x : i64
  return %0 : i64
}
EOF
```

Real output **[12]** (exit status 1):
```text
<stdin>:2:8: error: 'standalone.foo' op operand #0 must be 32-bit signless integer, but got 'i64'
  %0 = standalone.foo %x : i64
       ^
<stdin>:2:8: note: see current operation: %0 = "standalone.foo"(%arg0) : (i64) -> i64
```

The custom syntax cannot even express a wrong *result* type, since it never writes one. The generic syntax can, and the result check catches it:

```bash
cd /Users/roy/study/mlir/standalone
./build/bin/standalone-opt <<'EOF'
func.func @f(%x: i32) -> i16 {
  %0 = "standalone.foo"(%x) : (i32) -> i16
  return %0 : i16
}
EOF
```

Real output **[13]** (exit status 1):
```text
<stdin>:2:8: error: 'standalone.foo' op result #0 must be 32-bit signless integer, but got 'i16'
  %0 = "standalone.foo"(%x) : (i32) -> i16
       ^
<stdin>:2:8: note: see current operation: %0 = "standalone.foo"(%arg0) : (i32) -> i16
```

The error comes from the `I32` constraint on the result rather than from `SameOperandsAndResultType`: the invariants generated from the argument and result lists are checked first. With `I32` on both sides, `SameOperandsAndResultType` can never be the check that fails; its value in this op is the type inference it enables.

### 5.4 Custom and generic syntax

Every op has a **generic** form that any MLIR tool can print, whether or not it knows the op: the quoted name, the operands, the properties and attributes, and the full function type. `--mlir-print-op-generic` prints it for the test input `test/Standalone/dummy.mlir`:

***test/Standalone/dummy.mlir***
```mlir
// RUN: standalone-opt %s | standalone-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func.func @bar() {
        %0 = arith.constant 1 : i32
        // CHECK: %{{.*}} = standalone.foo %{{.*}} : i32
        %res = standalone.foo %0 : i32
        return
    }

    // CHECK-LABEL: func @standalone_types(%arg0: !standalone.custom<"10">)
    func.func @standalone_types(%arg0: !standalone.custom<"10">) {
        return
    }
}
```

```bash
cd /Users/roy/study/mlir/standalone
./build/bin/standalone-opt --mlir-print-op-generic test/Standalone/dummy.mlir
```

Real output **[14]**:
```text
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "bar"}> ({
    %0 = "arith.constant"() <{value = 1 : i32}> : () -> i32
    %1 = "standalone.foo"(%0) : (i32) -> i32
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = (!standalone.custom<"10">) -> (), sym_name = "standalone_types"}> ({
  ^bb0(%arg0: !standalone.custom<"10">):
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

```

`standalone.foo` becomes `"standalone.foo"(%0) : (i32) -> i32`, and the custom type prints the same way in both forms, since a type has no generic form. Going the other way, generic input is printed in the custom form, because the parser builds the same in-memory op either way:

```bash
cd /Users/roy/study/mlir/standalone
./build/bin/standalone-opt <<'EOF'
func.func @f(%x: i32) -> i32 {
  %0 = "standalone.foo"(%x) : (i32) -> i32
  return %0 : i32
}
EOF
```

Real output **[15]**:
```text
module {
  func.func @f(%arg0: i32) -> i32 {
    %0 = standalone.foo %arg0 : i32
    return %0 : i32
  }
}

```

How the custom form gets away with writing only `type($input)` is visible in the generated parser: after reading the operand and its type, it adds the *input* type as the result type. `inferReturnTypes`, used by the builders without a result type, makes the same choice:

```bash
cd /Users/roy/study/mlir/standalone
mlir-tblgen -gen-op-defs -I include -I /opt/homebrew/opt/llvm@20/include include/Standalone/StandaloneOps.td | sed -n '/FooOp::inferReturnTypes(/,/^}/p;/FooOp::parse(/,/^}/p'
```

Real output **[16]**:
```text
        if (::mlir::succeeded(FooOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      odsState.getRawProperties(),
                      odsState.regions, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::mlir::detail::reportFatalInferReturnTypesError(odsState);
        
}
  if (::mlir::succeeded(FooOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.getRawProperties(),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}
::llvm::LogicalResult FooOp::inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  if (operands.size() <= 0)
    return ::mlir::failure();
  ::mlir::Type odsInferredType0 = operands[0].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}
::mlir::ParseResult FooOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand inputRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> inputOperands(&inputRawOperand, 1);  ::llvm::SMLoc inputOperandsLoc;
  (void)inputOperandsLoc;
  ::mlir::Type inputRawType{};
  ::llvm::ArrayRef<::mlir::Type> inputTypes(&inputRawType, 1);

  inputOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(inputRawOperand))
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::IntegerType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    inputRawType = type;
  }
  result.addTypes(inputTypes[0]);
  if (parser.resolveOperands(inputOperands, inputTypes, inputOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}
```

`result.addTypes(inputTypes[0])` is `SameOperandsAndResultType` at work: **a trait is not only a check, it is information the generators use**, here to drop the result type from the syntax and from the builders.

### 5.5 What `Pure` buys: CSE and DCE

The input below computes `standalone.foo %x` three times and uses two of the results; it does the same with an op the tools know nothing about, `"mystery.op"`. Running the stock `--cse` and `--canonicalize` passes on it:

```bash
cd /Users/roy/study/mlir/standalone
./build/bin/standalone-opt --allow-unregistered-dialect --cse --canonicalize <<'EOF'
func.func @bar(%x: i32) -> i32 {
  %0 = standalone.foo %x : i32
  %1 = standalone.foo %x : i32
  %2 = standalone.foo %x : i32
  %3 = "mystery.op"(%x) : (i32) -> i32
  %4 = "mystery.op"(%x) : (i32) -> i32
  %5 = "mystery.op"(%x) : (i32) -> i32
  %6 = arith.addi %0, %1 : i32
  %7 = arith.addi %3, %4 : i32
  %8 = arith.addi %6, %7 : i32
  return %8 : i32
}
EOF
```

Real output **[17]**:
```text
module {
  func.func @bar(%arg0: i32) -> i32 {
    %0 = standalone.foo %arg0 : i32
    %1 = "mystery.op"(%arg0) : (i32) -> i32
    %2 = "mystery.op"(%arg0) : (i32) -> i32
    %3 = "mystery.op"(%arg0) : (i32) -> i32
    %4 = arith.addi %0, %0 : i32
    %5 = arith.addi %1, %2 : i32
    %6 = arith.addi %4, %5 : i32
    return %6 : i32
  }
}

```

- The three `standalone.foo` became one: CSE merged `%1` into `%0` (both are `foo %x` with no side effects), and `%2`, unused, was deleted.
- All three `mystery.op` survive, even the unused one. Nothing is known about an unregistered op, so the passes must assume it may write memory or do I/O, and they leave it alone.
- `arith.addi %0, %1` became `arith.addi %0, %0`, the visible trace of the merge.

What `Pure` adds to the generated class is three traits; dropping it from a copy of the `.td` file (through `sed`) and diffing the trait lists shows them:

```bash
cd /Users/roy/study/mlir/standalone
diff <(mlir-tblgen -gen-op-decls -I include -I /opt/homebrew/opt/llvm@20/include include/Standalone/StandaloneOps.td | grep "^class FooOp " | tr "," "\n") \
     <(sed "s/\[Pure,/[/" include/Standalone/StandaloneOps.td | mlir-tblgen -gen-op-decls -I include -I /opt/homebrew/opt/llvm@20/include | grep "^class FooOp " | tr "," "\n")
```

Real output **[18]** (exit status 1):
```text
8,10d7
<  ::mlir::ConditionallySpeculatable::Trait
<  ::mlir::OpTrait::AlwaysSpeculatableImplTrait
<  ::mlir::MemoryEffectOpInterface::Trait
```

`MemoryEffectOpInterface` (with an empty `getEffects`) is what CSE and DCE query, and the two speculation traits let passes such as loop-invariant code motion hoist the op. **Optimizations in MLIR are generic: they work on any op that answers their questions through traits and interfaces, so declaring one trait in ODS gives a brand-new op the same treatment as the built-in ones.**

---

## 6. A Pass: `--standalone-switch-bar-foo`

### 6.1 `StandalonePasses.td`

Passes are declared in ODS too, which generates their registration, command-line name and base class:

***include/Standalone/StandalonePasses.td***
```tablegen
include "mlir/Pass/PassBase.td"

def StandaloneSwitchBarFoo: Pass<"standalone-switch-bar-foo", "::mlir::ModuleOp"> {   // (1)
  let summary = "Switches the name of a FuncOp named `bar` to `foo` and folds.";        // (2)
  let description = [{
    ...
  }];
}
```

**(1)** `Pass<"standalone-switch-bar-foo", "::mlir::ModuleOp">`: the first argument is the command-line flag (`--standalone-switch-bar-foo`), the second the op the pass runs on, here the whole module. The record name `StandaloneSwitchBarFoo` becomes the C++ class name. **(2)** `summary` is the one-line help text (section 6.4). A pass record can also declare `options`, `statistics`, `dependentDialects` and a `constructor`; this one needs none.

### 6.2 The generated pass declarations

`include/Standalone/CMakeLists.txt` runs `mlir-tblgen --gen-pass-decls` on it (appendix A.2). The output is organized as blocks guarded by macros, so each includer picks only what it needs (the deprecated `GEN_PASS_CLASSES` block at the end is cut off here):

```bash
cd /Users/roy/study/mlir/standalone
mlir-tblgen -gen-pass-decls -I include -I /opt/homebrew/opt/llvm@20/include include/Standalone/StandalonePasses.td | sed '/Deprecated/,$d'
```

Real output **[19]**:
```text
/* Autogenerated by mlir-tblgen; don't manually edit */

#ifdef GEN_PASS_DECL
// Generate declarations for all passes.
#define GEN_PASS_DECL_STANDALONESWITCHBARFOO
#undef GEN_PASS_DECL
#endif // GEN_PASS_DECL

//===----------------------------------------------------------------------===//
// StandaloneSwitchBarFoo
//===----------------------------------------------------------------------===//
#ifdef GEN_PASS_DECL_STANDALONESWITCHBARFOO
std::unique_ptr<::mlir::Pass> createStandaloneSwitchBarFoo();
#undef GEN_PASS_DECL_STANDALONESWITCHBARFOO
#endif // GEN_PASS_DECL_STANDALONESWITCHBARFOO
#ifdef GEN_PASS_DEF_STANDALONESWITCHBARFOO

namespace impl {
  std::unique_ptr<::mlir::Pass> createStandaloneSwitchBarFoo();
} // namespace impl
namespace impl {

template <typename DerivedT>
class StandaloneSwitchBarFooBase : public ::mlir::OperationPass<::mlir::ModuleOp> {
public:
  using Base = StandaloneSwitchBarFooBase;

  StandaloneSwitchBarFooBase() : ::mlir::OperationPass<::mlir::ModuleOp>(::mlir::TypeID::get<DerivedT>()) {}
  StandaloneSwitchBarFooBase(const StandaloneSwitchBarFooBase &other) : ::mlir::OperationPass<::mlir::ModuleOp>(other) {}
  StandaloneSwitchBarFooBase& operator=(const StandaloneSwitchBarFooBase &) = delete;
  StandaloneSwitchBarFooBase(StandaloneSwitchBarFooBase &&) = delete;
  StandaloneSwitchBarFooBase& operator=(StandaloneSwitchBarFooBase &&) = delete;
  ~StandaloneSwitchBarFooBase() = default;

  /// Returns the command-line argument attached to this pass.
  static constexpr ::llvm::StringLiteral getArgumentName() {
    return ::llvm::StringLiteral("standalone-switch-bar-foo");
  }
  ::llvm::StringRef getArgument() const override { return "standalone-switch-bar-foo"; }

  ::llvm::StringRef getDescription() const override { return "Switches the name of a FuncOp named `bar` to `foo` and folds."; }

  /// Returns the derived pass name.
  static constexpr ::llvm::StringLiteral getPassName() {
    return ::llvm::StringLiteral("StandaloneSwitchBarFoo");
  }
  ::llvm::StringRef getName() const override { return "StandaloneSwitchBarFoo"; }

  /// Support isa/dyn_cast functionality for the derived pass class.
  static bool classof(const ::mlir::Pass *pass) {
    return pass->getTypeID() == ::mlir::TypeID::get<DerivedT>();
  }

  /// A clone method to create a copy of this pass.
  std::unique_ptr<::mlir::Pass> clonePass() const override {
    return std::make_unique<DerivedT>(*static_cast<const DerivedT *>(this));
  }

  /// Return the dialect that must be loaded in the context before this pass.
  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    
  }

  /// Explicitly declare the TypeID for this class. We declare an explicit private
  /// instantiation because Pass classes should only be visible by the current
  /// library.
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(StandaloneSwitchBarFooBase<DerivedT>)

protected:
private:

  friend std::unique_ptr<::mlir::Pass> createStandaloneSwitchBarFoo() {
    return std::make_unique<DerivedT>();
  }
};
} // namespace impl

std::unique_ptr<::mlir::Pass> createStandaloneSwitchBarFoo() {
  return impl::createStandaloneSwitchBarFoo();
}
#undef GEN_PASS_DEF_STANDALONESWITCHBARFOO
#endif // GEN_PASS_DEF_STANDALONESWITCHBARFOO
#ifdef GEN_PASS_REGISTRATION

//===----------------------------------------------------------------------===//
// StandaloneSwitchBarFoo Registration
//===----------------------------------------------------------------------===//

inline void registerStandaloneSwitchBarFoo() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createStandaloneSwitchBarFoo();
  });
}

// Old registration code, kept for temporary backwards compatibility.
inline void registerStandaloneSwitchBarFooPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createStandaloneSwitchBarFoo();
  });
}

//===----------------------------------------------------------------------===//
//  Registration
//===----------------------------------------------------------------------===//

inline void registerPasses() {
  registerStandaloneSwitchBarFoo();
}
#undef GEN_PASS_REGISTRATION
#endif // GEN_PASS_REGISTRATION
```

| Macro | What it produces | Who defines it |
| --- | --- | --- |
| `GEN_PASS_DECL` | The public declaration `createStandaloneSwitchBarFoo()` | `StandalonePasses.h` |
| `GEN_PASS_DEF_STANDALONESWITCHBARFOO` | The base class `impl::StandaloneSwitchBarFooBase<DerivedT>` (flag name, description, `clonePass`, `TypeID`) and the definition of `createStandaloneSwitchBarFoo()` | `StandalonePasses.cpp` |
| `GEN_PASS_REGISTRATION` | `registerStandaloneSwitchBarFoo()` and `registerPasses()`, which call `mlir::registerPass` | `StandalonePasses.h` |

The base class is a CRTP (curiously recurring template pattern) template: your class passes itself as `DerivedT`, and the base uses it to create, clone and identify the pass.

### 6.3 `StandalonePasses.h` and `StandalonePasses.cpp`

***include/Standalone/StandalonePasses.h***
```cpp
namespace mlir {
namespace standalone {
#define GEN_PASS_DECL
#include "Standalone/StandalonePasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "Standalone/StandalonePasses.h.inc"
} // namespace standalone
} // namespace mlir
```

The header exposes `createStandaloneSwitchBarFoo()` and `registerPasses()` in `mlir::standalone`. The tools call `registerPasses()` (sections 7 and 9).

***lib/Standalone/StandalonePasses.cpp***
```cpp
namespace mlir::standalone {
#define GEN_PASS_DEF_STANDALONESWITCHBARFOO                                    // (1)
#include "Standalone/StandalonePasses.h.inc"

namespace {
class StandaloneSwitchBarFooRewriter : public OpRewritePattern<func::FuncOp> { // (2)
public:
  using OpRewritePattern<func::FuncOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(func::FuncOp op,
                                PatternRewriter &rewriter) const final {
    if (op.getSymName() == "bar") {
      rewriter.modifyOpInPlace(op, [&op]() { op.setSymName("foo"); });         // (3)
      return success();
    }
    return failure();                                                          // (4)
  }
};

class StandaloneSwitchBarFoo
    : public impl::StandaloneSwitchBarFooBase<StandaloneSwitchBarFoo> {        // (5)
public:
  using impl::StandaloneSwitchBarFooBase<
      StandaloneSwitchBarFoo>::StandaloneSwitchBarFooBase;
  void runOnOperation() final {                                                // (6)
    RewritePatternSet patterns(&getContext());
    patterns.add<StandaloneSwitchBarFooRewriter>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsGreedily(getOperation(), patternSet)))             // (7)
      signalPassFailure();
  }
};
} // namespace
} // namespace mlir::standalone
```

**(1)** Pulls in the base class and `createStandaloneSwitchBarFoo()`. **(2)** A rewrite pattern that matches `func.func` ops. **(3)** A match renames the function in place. Every change made inside a pattern goes through the `rewriter`; `modifyOpInPlace` tells the driver the op changed, so it can revisit it. **(4)** Returning `failure()` means "no match": the IR must be unchanged. **(5)** The pass class derives from the generated base, passing itself as `DerivedT`. **(6)** `runOnOperation()` is the body of the pass; `getOperation()` is the `ModuleOp` it runs on. **(7)** `applyPatternsGreedily` applies the patterns until nothing changes. On top of the patterns, the greedy driver also **folds** ops and **erases** ops that are dead and side-effect free; that is the "and folds" in the summary.

The pattern is the whole transformation, and the pass is only the wrapper that runs it. The same pattern could be added to any other pass's pattern set, which is why MLIR code is usually organized as patterns first and passes second.

### 6.4 Running the pass

The pass-plugin test input has two functions, `@bar` and `@abar`:

***test/Standalone/standalone-pass-plugin.mlir***
```mlir
module {
  // CHECK-LABEL: func @foo()
  func.func @bar() {
    return
  }

  // CHECK-LABEL: func @abar()
  func.func @abar() {
    return
  }
}
```

```bash
cd /Users/roy/study/mlir/standalone
./build/bin/standalone-opt --standalone-switch-bar-foo test/Standalone/standalone-pass-plugin.mlir
```

Real output **[20]**:
```text
module {
  func.func @foo() {
    return
  }
  func.func @abar() {
    return
  }
}

```

Only the function named exactly `bar` is renamed; `abar` does not match `getSymName() == "bar"`. On `dummy.mlir` (section 5.4), the greedy driver's cleanup shows as well: the unused `standalone.foo` and the constant feeding it are erased, because both are `Pure` and dead:

```bash
cd /Users/roy/study/mlir/standalone
./build/bin/standalone-opt --standalone-switch-bar-foo test/Standalone/dummy.mlir
```

Real output **[21]**:
```text
module {
  func.func @foo() {
    return
  }
  func.func @standalone_types(%arg0: !standalone.custom<"10">) {
    return
  }
}

```

**The pattern does not check that the new name is free.** With both `@bar` and `@foo` in the module, the rename produces two `@foo`s. The pass itself succeeds, but `standalone-opt` verifies the IR after every pass, and the verifier of the module's symbol table rejects it:

```bash
cd /Users/roy/study/mlir/standalone
./build/bin/standalone-opt --standalone-switch-bar-foo <<'EOF'
func.func @bar() { return }
func.func @foo() { return }
EOF
```

Real output **[22]** (exit status 1):
```text
<stdin>:2:1: error: redefinition of symbol named 'foo'
func.func @foo() { return }
^
<stdin>:2:1: note: see current operation: 
"func.func"() <{function_type = () -> (), sym_name = "foo"}> ({
  "func.return"() : () -> ()
}) : () -> ()
<stdin>:1:1: note: see existing symbol definition here
func.func @bar() { return }
^
```

The verifier between passes turns a pass bug into an error at the pass that caused it, instead of a miscompile several passes later. A production version of this pattern would look up `foo` in the symbol table and return `failure()` if it exists.

Because the pass is registered by `registerPasses()`, it appears in the tool's help next to MLIR's own passes:

```bash
cd /Users/roy/study/mlir/standalone
./build/bin/standalone-opt --help | grep -A1 "standalone-switch"
```

Real output **[23]**:
```text
      --standalone-switch-bar-foo                            -   Switches the name of a FuncOp named `bar` to `foo` and folds.
      --strip-debuginfo                                      -   Strip debug info from all operations
```

---

## 7. `standalone-opt`: The Optimizer Driver

### 7.1 The driver

***standalone-opt/standalone-opt.cpp***
```cpp
int main(int argc, char **argv) {
  mlir::registerAllPasses();                                       // (1)
  mlir::standalone::registerPasses();                              // (2)
  // TODO: Register standalone passes here.

  mlir::DialectRegistry registry;                                  // (3)
  registry.insert<mlir::standalone::StandaloneDialect,
                  mlir::arith::ArithDialect, mlir::func::FuncDialect>();
  // Add the following to include *all* MLIR Core dialects, or selectively
  // include what you need like above. You only need to register dialects that
  // will be *parsed* by the tool, not the one generated
  // registerAllDialects(registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Standalone optimizer driver\n", registry));   // (4)
}
```

**(1)** Registers every upstream pass (`--cse`, `--canonicalize`, the conversions, ...), so the tool can run any of them on standalone IR. **(2)** Registers the project's own pass (section 6). The `TODO` comment is a leftover: the line above it already does it. **(3)** A `DialectRegistry` lists the dialects the tool can *parse*. It holds only factories; a dialect is loaded into the context when the parser first meets its namespace. **(4)** `MlirOptMain` is all of `mlir-opt`: command-line handling, parsing, the pass pipeline, verification, printing. A custom `-opt` tool is this function plus a registry.

### 7.2 What the tool understands

```bash
cd /Users/roy/study/mlir/standalone
./build/bin/standalone-opt --show-dialects
```

Real output **[24]**:
```text
Available Dialects: arith,builtin,func,standalone
```

`builtin` is always there (it defines `module` and the built-in types); the other three are the registry. Everything else is rejected, even though all upstream passes are registered and all upstream dialect libraries are linked in (section 7.3):

```bash
cd /Users/roy/study/mlir/standalone
./build/bin/standalone-opt <<'EOF'
func.func @f(%c: i1) {
  scf.if %c { }
  return
}
EOF
```

Real output **[25]** (exit status 1):
```text
<stdin>:2:3: error: Dialect `scf' not found for custom op 'scf.if' 
  scf.if %c { }
  ^
<stdin>:2:3: note: Registered dialects: arith, builtin, func, standalone ; for more info on dialect registration see https://mlir.llvm.org/getting_started/Faq/#registered-loaded-dependent-whats-up-with-dialects-management
```

The reverse holds for the stock `mlir-opt`: it registers every upstream dialect, but not `standalone`, so it cannot parse `dummy.mlir`:

```bash
cd /Users/roy/study/mlir/standalone
mlir-opt test/Standalone/dummy.mlir
```

Real output **[26]** (exit status 1):
```text
test/Standalone/dummy.mlir:8:16: error: Dialect `standalone' not found for custom op 'standalone.foo' 
        %res = standalone.foo %0 : i32
               ^
test/Standalone/dummy.mlir:8:16: note: Registered dialects: acc, affine, amdgpu, amx, arith, arm_neon, arm_sme, arm_sve, async, bufferization, builtin, cf, complex, dlti, emitc, func, gpu, index, irdl, linalg, llvm, math, memref, mesh, ml_program, mpi, nvgpu, nvvm, omp, pdl, pdl_interp, polynomial, ptr, quant, rocdl, scf, shape, sparse_tensor, spirv, tensor, tosa, transform, ub, vector, x86vector, xegpu ; for more info on dialect registration see https://mlir.llvm.org/getting_started/Faq/#registered-loaded-dependent-whats-up-with-dialects-management
```

**Dialect registration is per tool, not per installation**: which IR a binary can read is decided by its `main`, which is why every MLIR project ships its own `-opt` tool, or a plugin for the stock one (section 9).

### 7.3 Linking the tool

***standalone-opt/CMakeLists.txt***
```cmake
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)       # (1)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIRArithDialect
        MLIROptLib                                                 # (2)
        MLIRStandalone                                             # (3)
        )
add_llvm_executable(standalone-opt standalone-opt.cpp)

llvm_update_compile_flags(standalone-opt)
target_link_libraries(standalone-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(standalone-opt)                      # (4)
```

**(1)** `MLIR_DIALECT_LIBS` and `MLIR_CONVERSION_LIBS` are global lists that `MLIRConfig.cmake` fills with every installed dialect and conversion library; `registerAllPasses()` needs all of them. **(2)** `MLIROptLib` provides `MlirOptMain`. **(3)** The project's dialect library (section 3). **(4)** Checks at configure time that every library in the link line exists.

These libraries are static archives (`libMLIR*.a`), so everything `registerAllPasses()` reaches is copied into the executable:

```bash
cd /Users/roy/study/mlir/standalone
du -h build/bin/*
```

Real output **[27]**:
```text
 36K	build/bin/standalone-capi-test
 86M	build/bin/standalone-opt
 42M	build/bin/standalone-translate
```

A tool that only needs its own dialect and a few passes can drop `registerAllPasses()` and the two global lists and link far less.

---

## 8. `standalone-translate`: The Translation Driver

### 8.1 The driver

`mlir-translate` converts between MLIR and *other* formats (LLVM IR, SPIR-V binaries, C++), as opposed to `mlir-opt`, which goes from MLIR to MLIR. `standalone-translate` is the same driver with one extra, placeholder translation:

***standalone-translate/standalone-translate.cpp***
```cpp
int main(int argc, char **argv) {
  mlir::registerAllTranslations();                                 // (1)

  // TODO: Register standalone translations here.
  mlir::TranslateFromMLIRRegistration withdescription(             // (2)
      "option", "different from option",
      [](mlir::Operation *op, llvm::raw_ostream &output) {         // (3)
        return llvm::LogicalResult::success();
      },
      [](mlir::DialectRegistry &a) {});                            // (4)

  return failed(
      mlir::mlirTranslateMain(argc, argv, "MLIR Translation Testing Tool"));
}
```

**(1)** Registers every upstream translation. **(2)** A `TranslateFromMLIRRegistration` object registers a translation *from* MLIR: its flag (`--option`) and help text. **(3)** The translation itself: it receives the parsed top-level op and an output stream; this one writes nothing. **(4)** The dialects the translation needs to parse its input; this one registers none.

Unlike `standalone-opt`, this tool never registers the `standalone` dialect. The `#include "Standalone/StandaloneDialect.h"` at the top of the file is there for the translations you are meant to add.

### 8.2 Translations are flags

Each registered translation is a flag in the "Translations to perform" group of the help:

```bash
cd /Users/roy/study/mlir/standalone
./build/bin/standalone-translate --help | sed -n '/Translations to perform/,/serialize-spirv/p'
```

Real output **[28]**:
```text
  Translations to perform
      --deserialize-spirv                                       - deserializes the SPIR-V module
```

`--option` is ours; the rest come from `registerAllTranslations()`. Run on an empty module, `--option` parses the input, calls the lambda, and prints nothing, successfully:

```bash
cd /Users/roy/study/mlir/standalone
echo 'module {}' | ./build/bin/standalone-translate --option
echo "standalone-translate exit status: $?"
```

Real output **[29]**:
```text
standalone-translate exit status: 0
```

On `dummy.mlir` it fails while parsing, before the lambda runs. A translation's input is parsed with only the dialects that translation registered, and `--option` registered none, so even `func` is unknown:

```bash
cd /Users/roy/study/mlir/standalone
./build/bin/standalone-translate --option test/Standalone/dummy.mlir
```

Real output **[30]** (exit status 1):
```text
test/Standalone/dummy.mlir:5:5: error: Dialect `func' not found for custom op 'func.func' 
    func.func @bar() {
    ^
test/Standalone/dummy.mlir:5:5: note: Registered dialects: builtin ; for more info on dialect registration see https://mlir.llvm.org/getting_started/Faq/#registered-loaded-dependent-whats-up-with-dialects-management
```

An upstream translation such as `--mlir-to-llvmir` registers the dialects it can translate (`llvm`, `func`, the GPU dialects, ...), so input in the `llvm` dialect goes through:

```bash
cd /Users/roy/study/mlir/standalone
./build/bin/standalone-translate --mlir-to-llvmir <<'EOF'
llvm.func @main() -> i32 {
  %0 = llvm.mlir.constant(42 : i32) : i32
  llvm.return %0 : i32
}
EOF
```

Real output **[31]**:
```text
; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

define i32 @main() {
  ret i32 42
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
```

**A translation declares the dialects it accepts, and `mlir-translate` parses with exactly those**, so a translation never receives an op it does not know how to export. To translate standalone IR, a real translation would register `StandaloneDialect` in its registry lambda (4) and walk the ops in (3).

---

## 9. Plugins: Loading the Dialect into the Stock `mlir-opt`

### 9.1 The plugin

Instead of building a new tool, a project can extend the installed `mlir-opt` at run time. A **plugin** is a shared library exporting a well-known C function that `mlir-opt` looks up after loading it:

***standalone-plugin/standalone-plugin.cpp***
```cpp
/// Dialect plugin registration mechanism.
/// Observe that it also allows to register passes.
/// Necessary symbol to register the dialect plugin.
extern "C" LLVM_ATTRIBUTE_WEAK DialectPluginLibraryInfo
mlirGetDialectPluginInfo() {                                               // (1)
  return {MLIR_PLUGIN_API_VERSION, "Standalone", LLVM_VERSION_STRING,      // (2)
          [](DialectRegistry *registry) {
            registry->insert<mlir::standalone::StandaloneDialect>();       // (3)
            mlir::standalone::registerPasses();
          }};
}

/// Pass plugin registration mechanism.
/// Necessary symbol to register the pass plugin.
extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo mlirGetPassPluginInfo() {   // (4)
  return {MLIR_PLUGIN_API_VERSION, "StandalonePasses", LLVM_VERSION_STRING,
          []() { mlir::standalone::registerPasses(); }};
}
```

**(1)** `mlirGetDialectPluginInfo` is the entry point `--load-dialect-plugin` looks for. `extern "C"` keeps the name unmangled so `dlsym` can find it. **(2)** The plugin API version (checked by `mlir-opt`), the plugin's name and the LLVM version it was built against. **(3)** The callback receives `mlir-opt`'s registry and adds the dialect, exactly what `standalone-opt` did in its `main`; it also registers the pass. **(4)** `mlirGetPassPluginInfo` is the entry point for `--load-pass-plugin`, which registers passes only. One library can be both kinds of plugin.

The CMake target is a loadable module, linked against `mlir-opt`:

***standalone-plugin/CMakeLists.txt***
```cmake
add_llvm_library(StandalonePlugin
        # BUILDTREE_ONLY is only for testing purposes
        MODULE BUILDTREE_ONLY                    # (1)
        standalone-plugin.cpp

        DEPENDS
        MLIRStandalone
        PLUGIN_TOOL                              # (2)
        mlir-opt

        LINK_LIBS
        MLIRStandalone
        )
```

**(1)** `MODULE` builds a library meant for `dlopen` (`StandalonePlugin.dylib` on macOS), not for linking; `BUILDTREE_ONLY` keeps it out of `install`. **(2)** `PLUGIN_TOOL mlir-opt` marks it as a plugin of `mlir-opt`: the MLIR core symbols it uses are not copied in but resolved against the `mlir-opt` process at load time (appendix A.3 shows the link flags). The plugin depends only on `libLLVM.dylib` and the system libraries; MLIR itself comes from the host:

```bash
cd /Users/roy/study/mlir/standalone
otool -L build/lib/StandalonePlugin.dylib
```

Real output **[32]**:
```text
build/lib/StandalonePlugin.dylib:
	/opt/homebrew/opt/llvm@20/lib/libLLVM.dylib (compatibility version 1.0.0, current version 20.1.8)
	/usr/lib/libc++.1.dylib (compatibility version 1.0.0, current version 2100.43.0)
	/usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1356.0.0)
```

### 9.2 Running the plugins

The dialect-plugin test uses both the custom type and the pass:

***test/Standalone/standalone-plugin.mlir***
```mlir
// RUN: mlir-opt %s --load-dialect-plugin=%standalone_libs/StandalonePlugin%shlibext --pass-pipeline="builtin.module(standalone-switch-bar-foo)" | FileCheck %s

module {
  // CHECK-LABEL: func @foo()
  func.func @bar() {
    return
  }

  // CHECK-LABEL: func @standalone_types(%arg0: !standalone.custom<"10">)
  func.func @standalone_types(%arg0: !standalone.custom<"10">) {
    return
  }
}
```

With the dialect plugin, the *stock* `mlir-opt` parses `!standalone.custom` and runs the pass:

```bash
cd /Users/roy/study/mlir/standalone
mlir-opt test/Standalone/standalone-plugin.mlir --load-dialect-plugin=build/lib/StandalonePlugin.dylib --pass-pipeline="builtin.module(standalone-switch-bar-foo)"
```

Real output **[33]**:
```text
module {
  func.func @foo() {
    return
  }
  func.func @standalone_types(%arg0: !standalone.custom<"10">) {
    return
  }
}

```

The pass plugin is enough for an input without standalone types, the pass-plugin test of section 6.4:

```bash
cd /Users/roy/study/mlir/standalone
mlir-opt test/Standalone/standalone-pass-plugin.mlir --load-pass-plugin=build/lib/StandalonePlugin.dylib --pass-pipeline="builtin.module(standalone-switch-bar-foo)"
```

Real output **[34]**:
```text
module {
  func.func @foo() {
    return
  }
  func.func @abar() {
    return
  }
}

```

Without any plugin, the stock tool does not know the pass:

```bash
cd /Users/roy/study/mlir/standalone
mlir-opt test/Standalone/standalone-pass-plugin.mlir --pass-pipeline="builtin.module(standalone-switch-bar-foo)"
```

Real output **[35]** (exit status 1):
```text
<unknown>:0: error: MLIR Textual PassPipeline Parser:1:1: error: 'standalone-switch-bar-foo' does not refer to a registered pass or pass pipeline
standalone-switch-bar-foo
^

```

And the pass plugin alone is not enough for the dialect test, because it registers the pass but not the dialect, so the type cannot be parsed:

```bash
cd /Users/roy/study/mlir/standalone
mlir-opt test/Standalone/standalone-plugin.mlir --load-pass-plugin=build/lib/StandalonePlugin.dylib --pass-pipeline="builtin.module(standalone-switch-bar-foo)"
```

Real output **[36]** (exit status 1):
```text
test/Standalone/standalone-plugin.mlir:10:50: error: `!"standalone"<"custom<"10">">` type created with unregistered dialect. If this is intended, please call allowUnregisteredDialects() on the MLIRContext, or use -allow-unregistered-dialect with the MLIR opt tool used
  func.func @standalone_types(%arg0: !standalone.custom<"10">) {
                                                 ^
```

**A plugin trades a custom binary for a version lock**: one small library extends the installed `mlir-opt` without rebuilding it, but it must be built against that exact MLIR (the version string in (2)), since MLIR has no stable C++ ABI.

---

## 10. The C API and Python Bindings

### 10.1 The C API

MLIR's C API (`mlir-c/...`) is a C wrapper around the C++ core, used by the Python bindings and by other languages. A dialect joins it by exposing a **dialect handle**, which two macros generate:

***include/Standalone-c/Dialects.h***
```c
#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Standalone, standalone);

#ifdef __cplusplus
}
#endif
```

***lib/CAPI/Dialects.cpp***
```cpp
#include "Standalone-c/Dialects.h"

#include "Standalone/StandaloneDialect.h"
#include "mlir/CAPI/Registration.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Standalone, standalone,
                                      mlir::standalone::StandaloneDialect)
```

The declaration expands to one exported function, `mlirGetDialectHandle__standalone__()`. The definition, expanded by the preprocessor and formatted:

```bash
cd /Users/roy/study/mlir/standalone
clang++ -E -P -std=c++17 -I include -I build/include -I /opt/homebrew/opt/llvm@20/include lib/CAPI/Dialects.cpp | tail -1 | clang-format
```

Real output **[37]**:
```text
static void
mlirDialectRegistryInsertStandaloneDialect(MlirDialectRegistry registry) {
  unwrap(registry)->insert<mlir::standalone::StandaloneDialect>();
}
static MlirDialect mlirContextLoadStandaloneDialect(MlirContext context) {
  return wrap(
      unwrap(context)->getOrLoadDialect<mlir::standalone::StandaloneDialect>());
}
static MlirStringRef mlirStandaloneDialectGetNamespace() {
  return wrap(mlir::standalone::StandaloneDialect::getDialectNamespace());
}
MlirDialectHandle mlirGetDialectHandle__standalone__() {
  static MlirDialectRegistrationHooks hooks = {
      mlirDialectRegistryInsertStandaloneDialect,
      mlirContextLoadStandaloneDialect, mlirStandaloneDialectGetNamespace};
  return MlirDialectHandle{&hooks};
}
```

The handle is a pointer to a table of three hooks: insert the dialect into a registry, load it into a context, return its namespace. C code never sees `StandaloneDialect`; it passes the handle to generic functions such as `mlirDialectHandleRegisterDialect`. `lib/CAPI/CMakeLists.txt` builds this into the library `StandaloneCAPI` with `add_mlir_public_c_api_library` (appendix A.2).

### 10.2 A C program using the dialect

***test/CAPI/standalone-capi-test.c***
```c
static void registerAllUpstreamDialects(MlirContext ctx) {
  MlirDialectRegistry registry = mlirDialectRegistryCreate();
  mlirRegisterAllDialects(registry);
  mlirContextAppendDialectRegistry(ctx, registry);
  mlirDialectRegistryDestroy(registry);
}

int main(int argc, char **argv) {
  MlirContext ctx = mlirContextCreate();
  // TODO: Create the dialect handles for the builtin dialects and avoid this.
  // This adds dozens of MB of binary size over just the standalone dialect.
  registerAllUpstreamDialects(ctx);
  mlirDialectHandleRegisterDialect(mlirGetDialectHandle__standalone__(), ctx);   // (1)

  MlirModule module = mlirModuleCreateParse(                                      // (2)
      ctx, mlirStringRefCreateFromCString("%0 = arith.constant 2 : i32\n"
                                          "%1 = standalone.foo %0 : i32\n"));
  if (mlirModuleIsNull(module)) {
    printf("ERROR: Could not parse.\n");
    mlirContextDestroy(ctx);
    return 1;
  }
  MlirOperation op = mlirModuleGetOperation(module);

  // CHECK: %[[C:.*]] = arith.constant 2 : i32
  // CHECK: standalone.foo %[[C]] : i32
  mlirOperationDump(op);                                                          // (3)

  mlirModuleDestroy(module);
  mlirContextDestroy(ctx);
  return 0;
}
```

**(1)** The handle from section 10.1 registers the dialect into the context; `arith` comes from `mlirRegisterAllDialects`. **(2)** Parses a module from a string. **(3)** Prints it.

```bash
cd /Users/roy/study/mlir/standalone
./build/bin/standalone-capi-test
```

Real output **[38]**:
```text
module {
  %c2_i32 = arith.constant 2 : i32
  %0 = standalone.foo %c2_i32 : i32
}
```

The program links one shared library, `libStandaloneCAPITestLib.dylib`, built by `add_mlir_aggregate` in `test/CAPI/CMakeLists.txt` from the C API core, the "register everything" library and `StandaloneCAPI` (appendix A.2). That aggregate is self-contained, which is what a language binding wants to load, and it is large because "register everything" pulls in every upstream dialect, as the `TODO` in the test says:

```bash
cd /Users/roy/study/mlir/standalone
otool -L build/bin/standalone-capi-test
du -h build/lib/libStandaloneCAPITestLib.dylib
```

Real output **[39]**:
```text
build/bin/standalone-capi-test:
	@rpath/libStandaloneCAPITestLib.dylib (compatibility version 0.0.0, current version 0.0.0)
	/opt/homebrew/opt/llvm@20/lib/libLLVM.dylib (compatibility version 1.0.0, current version 20.1.8)
	/usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1356.0.0)
110M	build/lib/libStandaloneCAPITestLib.dylib
```

### 10.3 Python bindings

`python/` builds a Python package `mlir_standalone` on top of the C API: the full MLIR Python API plus generated Python classes for the standalone ops (`declare_mlir_dialect_python_bindings` runs TableGen on `python/mlir_standalone/dialects/StandaloneOps.td`) and a native extension exposing `register_dialect`. It comes in two flavors, nanobind and pybind11, with identical behavior:

***python/StandaloneExtensionNanobind.cpp***
```cpp
NB_MODULE(_standaloneDialectsNanobind, m) {
  //===--------------------------------------------------------------------===//
  // standalone dialect
  //===--------------------------------------------------------------------===//
  auto standaloneM = m.def_submodule("standalone");

  standaloneM.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle handle = mlirGetDialectHandle__standalone__();
        mlirDialectHandleRegisterDialect(handle, context);
        if (load) {
          mlirDialectHandleLoadDialect(handle, context);
        }
      },
      nb::arg("context").none() = nb::none(), nb::arg("load") = true);
}
```

The extension is a thin layer over the same dialect handle as the C program. The test that would use it:

***test/python/smoketest.py***
```python
with Context():
    standalone_d.register_dialect()
    module = Module.parse(
        """
    %0 = arith.constant 2 : i32
    %1 = standalone.foo %0 : i32
    """
    )
    # CHECK: %[[C:.*]] = arith.constant 2 : i32
    # CHECK: standalone.foo %[[C]] : i32
    print(str(module))
```

The bindings are built only when MLIR itself was built with Python bindings (section 2.2 (6)), and the Homebrew build was not:

```bash
cd /Users/roy/study/mlir/standalone
grep MLIR_ENABLE_BINDINGS_PYTHON /opt/homebrew/opt/llvm@20/lib/cmake/mlir/MLIRConfig.cmake
```

Real output **[40]**:
```text
set(MLIR_ENABLE_BINDINGS_PYTHON "0")
```

So `python/` is skipped at configure time and the smoke test reports *Unsupported* (section 12.2): `test/python/lit.local.cfg` marks the directory unsupported when `config.enable_bindings_python` is false. Building it requires an MLIR built from source with `-DMLIR_ENABLE_BINDINGS_PYTHON=ON`.

---

## 11. Testing with lit and FileCheck

### 11.1 The lit configuration

**lit** (LLVM Integrated Tester) finds test files, runs the shell commands in their `RUN:` lines and reports pass or fail. **FileCheck** compares a command's output with the `CHECK:` lines of the test file. The project's configuration has two parts: a template that CMake fills in with build paths, and the main configuration.

***test/lit.site.cfg.py.in***
```python
@LIT_SITE_CFG_IN_HEADER@

config.llvm_tools_dir = lit_config.substitute("@LLVM_TOOLS_DIR@")
config.mlir_obj_dir = "@MLIR_BINARY_DIR@"
config.enable_bindings_python = @MLIR_ENABLE_BINDINGS_PYTHON@
config.standalone_obj_root = "@STANDALONE_BINARY_DIR@"
config.llvm_shlib_ext = "@SHLIBEXT@"

import lit.llvm
lit.llvm.initialize(lit_config, config)

# Let the main config do the real work.
lit_config.load_config(config, "@STANDALONE_SOURCE_DIR@/test/lit.cfg.py")
```

`configure_lit_site_cfg` in `test/CMakeLists.txt` writes this to `build/test/lit.site.cfg.py` with the `@...@` values substituted, so lit is pointed at the *build* directory and finds the source tests through it.

***test/lit.cfg.py***
```python
config.name = "STANDALONE"

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".mlir"]                                                    # (1)
...
config.substitutions.append(("%PATH%", config.environment["PATH"]))
config.substitutions.append(("%shlibext", config.llvm_shlib_ext))              # (2)
...
config.standalone_tools_dir = os.path.join(config.standalone_obj_root, "bin")
config.standalone_libs_dir = os.path.join(config.standalone_obj_root, "lib")

config.substitutions.append(("%standalone_libs", config.standalone_libs_dir))  # (3)

# Tweak the PATH to include the tools dir.
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)

tool_dirs = [config.standalone_tools_dir, config.llvm_tools_dir]              # (4)
tools = [
    "mlir-opt",
    "standalone-capi-test",
    "standalone-opt",
    "standalone-translate",
]

llvm_config.add_tool_substitutions(tools, tool_dirs)
```

**(1)** `.mlir` files are tests; `test/CAPI/lit.local.cfg` adds `.c` and `test/python/lit.local.cfg` adds `.py` for their directories. **(2)** `%shlibext` is the shared-library suffix (`.dylib` here), and **(3)** `%standalone_libs` is `build/lib`; together they spell the plugin path in the plugin tests. **(4)** Each tool name in a `RUN:` line is replaced by its full path, searched first in `build/bin`, then in the LLVM tools directory. `FileCheck`, `count` and `not` are substituted by `use_default_substitutions()`.

### 11.2 `RUN:` and `CHECK:` lines

`dummy.mlir` (section 5.4) is the model test. Its `RUN:` line, `standalone-opt %s | standalone-opt | FileCheck %s`, parses and prints the file twice (a round trip through the custom syntax) and checks the result. `lit -a` shows the command after substitution, with `%s` replaced by the test file and each tool by its path:

```bash
cd /Users/roy/study/mlir/standalone
lit -a build/test/Standalone/dummy.mlir
```

Real output **[41]**:
```text
-- Testing: 1 tests, 1 workers --
PASS: STANDALONE :: Standalone/dummy.mlir (1 of 1)
Exit Code: 0

Command Output (stderr):
--
RUN: at line 1: /Users/roy/study/mlir/standalone/build/bin/standalone-opt /Users/roy/study/mlir/standalone/test/Standalone/dummy.mlir | /Users/roy/study/mlir/standalone/build/bin/standalone-opt | /opt/homebrew/opt/llvm@20/bin/FileCheck /Users/roy/study/mlir/standalone/test/Standalone/dummy.mlir
+ /Users/roy/study/mlir/standalone/build/bin/standalone-opt /Users/roy/study/mlir/standalone/test/Standalone/dummy.mlir
+ /Users/roy/study/mlir/standalone/build/bin/standalone-opt
+ /opt/homebrew/opt/llvm@20/bin/FileCheck /Users/roy/study/mlir/standalone/test/Standalone/dummy.mlir

--

********************

Testing Time: 0.12s

Total Discovered Tests: 1
  Passed: 1 (100.00%)
```

The `CHECK:` lines match the output in order, and each needs only to appear somewhere in a line; `{{.*}}` is a regular expression, so the test does not depend on value names such as `%c1_i32`. `CHECK-LABEL:` additionally splits the output into blocks, so a failure inside one function cannot be satisfied by text in another. The same check, run by hand, succeeds silently:

```bash
cd /Users/roy/study/mlir/standalone
./build/bin/standalone-opt test/Standalone/dummy.mlir | FileCheck test/Standalone/dummy.mlir
echo "FileCheck exit status: $?"
```

Real output **[42]**:
```text
FileCheck exit status: 0
```

When a check does not match, FileCheck reports where it looked, the closest candidate and an annotated dump of the input. Here the check file is `dummy.mlir` with `standalone.foo` changed to `standalone.bar` (through `sed`):

```bash
cd /Users/roy/study/mlir/standalone
./build/bin/standalone-opt test/Standalone/dummy.mlir | FileCheck <(sed "s/= standalone.foo/= standalone.bar/" test/Standalone/dummy.mlir)
echo "FileCheck exit status: $?"
```

Real output **[43]**:
```text
/dev/fd/63:7:12: error: CHECK: expected string not found in input
 // CHECK: %{{.*}} = standalone.bar %{{.*}} : i32
           ^
<stdin>:2:18: note: scanning from here
 func.func @bar() {
                 ^
<stdin>:4:2: note: possible intended match here
 %0 = standalone.foo %c1_i32 : i32
 ^

Input file: <stdin>
Check file: /dev/fd/63

-dump-input=help explains the following input dump.

Input was:
<<<<<<
           1: module { 
           2:  func.func @bar() { 
check:7'0                      X~~ error: no match found
           3:  %c1_i32 = arith.constant 1 : i32 
check:7'0     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
           4:  %0 = standalone.foo %c1_i32 : i32 
check:7'0     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
check:7'1      ?                                  possible intended match
           5:  return 
check:7'0     ~~~~~~~~
           6:  } 
check:7'0     ~~~
           7:  func.func @standalone_types(%arg0: !standalone.custom<"10">) { 
check:7'0     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
           8:  return 
           9:  } 
          10: } 
          11:  
>>>>>>
FileCheck exit status: 1
```

The other tests are one-liners, each checking one part of the project:

| Test | `RUN:` line | Checks |
| --- | --- | --- |
| `Standalone/dummy.mlir` | `standalone-opt %s \| standalone-opt \| FileCheck %s` | op and type round-trip (sections 4–5) |
| `Standalone/standalone-opt.mlir` | `standalone-opt --show-dialects \| FileCheck %s` | the dialect is registered (section 7.2) |
| `Standalone/standalone-translate.mlir` | `standalone-translate --help \| FileCheck %s` | upstream translations are registered (section 8.2) |
| `Standalone/standalone-plugin.mlir` | `mlir-opt %s --load-dialect-plugin=... --pass-pipeline=... \| FileCheck %s` | the dialect plugin (section 9.2) |
| `Standalone/standalone-pass-plugin.mlir` | `mlir-opt %s --load-pass-plugin=... --pass-pipeline=... \| FileCheck %s` | the pass plugin (section 9.2) |
| `CAPI/standalone-capi-test.c` | `standalone-capi-test 2>&1 \| FileCheck %s` | the C API (section 10.2) |
| `python/smoketest.py` | `%python %s pybind11 \| FileCheck %s` (and `nanobind`) | the Python bindings (section 10.3) |

**A lit test is the command you would type to check the feature by hand, plus the output you expect, stored in one file**, so every manual check in this README could become a test by adding a `RUN:` line and its `CHECK:` lines.

---

## 12. Build and Run

### 12.1 Building

```bash
cd /Users/roy/study/mlir/standalone
cmake -S . -B build -G Ninja -DLLVM_EXTERNAL_LIT=$(which lit) -DCMAKE_OSX_DEPLOYMENT_TARGET=26.0
cmake --build build
cmake --build build --target check-standalone
```

The shell sets `CXX`/`CC` to Homebrew `llvm@20` clang and has `/opt/homebrew/opt/llvm@20/bin` on `PATH`, so `find_package(MLIR)` finds `llvm@20/lib/cmake/mlir` without flags; the configure step reports it:

```text
-- Using MLIRConfig.cmake in: /opt/homebrew/opt/llvm@20/lib/cmake/mlir
-- Using LLVMConfig.cmake in: /opt/homebrew/opt/llvm@20/lib/cmake/llvm
```

If it picks up another MLIR, pass `-DMLIR_DIR=/opt/homebrew/opt/llvm@20/lib/cmake/mlir` and `-DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm@20/bin/clang++`. The two flags in the command:

- **`-DLLVM_EXTERNAL_LIT`**: Homebrew's `llvm@20` does not install `llvm-lit`, so the test target needs a `lit` executable; `brew install lit` provides one at `/opt/homebrew/bin/lit`. Without the flag, `check-standalone` has nothing to run the tests with.
- **`-DCMAKE_OSX_DEPLOYMENT_TARGET=26.0`**: Homebrew's MLIR archives are built for macOS 26. Without the flag, CMake targets an older macOS and the linker prints one "was built for newer 'macOS' version" warning per object file it pulls from `libMLIR*.a`. The binaries still work; the flag only silences the warnings.

`cmake --build build` builds everything (about 90 seconds): the libraries `build/lib/libMLIRStandalone.a` and `libStandaloneCAPI.a`, the plugin `build/lib/StandalonePlugin.dylib`, the aggregate `build/lib/libStandaloneCAPITestLib.dylib`, and the executables `build/bin/standalone-opt`, `standalone-translate` and `standalone-capi-test`. Unlike the Toy and Transform tutorials, which link the monolithic `libMLIR.dylib`, this project links the static `libMLIR*.a` archives, as the upstream example does; appendix A explains every `CMakeLists.txt`. `cmake --build build --target mlir-doc` additionally writes the dialect documentation of appendix B to `build/docs/Standalone/`.

**Monolithic build.** The other build mode of section 2.2 (5) builds the project inside an LLVM build, from an LLVM monorepo checkout at `$LLVM_SRC_DIR` (not used in this README):

```sh
mkdir build && cd build
cmake -G Ninja `$LLVM_SRC_DIR/llvm` \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_TARGETS_TO_BUILD=host \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_EXTERNAL_PROJECTS=standalone-dialect -DLLVM_EXTERNAL_STANDALONE_DIALECT_SOURCE_DIR=../
cmake --build . --target check-standalone
```

A component build against an MLIR you built and installed yourself works like the Homebrew one, with `-DMLIR_DIR=$PREFIX/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=$BUILD_DIR/bin/llvm-lit`; build LLVM with `-DLLVM_INSTALL_UTILS=ON` so that `FileCheck` is installed to the prefix.

### 12.2 Running the test suite

`cmake --build build --target check-standalone` (the last line of the build block) runs lit over `build/test`. Calling lit directly with `-v` lists every test:

```bash
cd /Users/roy/study/mlir/standalone
lit -v build/test
```

Real output **[44]**:
```text
-- Testing: 7 tests, 7 workers --
UNSUPPORTED: STANDALONE :: python/smoketest.py (1 of 7)
PASS: STANDALONE :: Standalone/standalone-pass-plugin.mlir (2 of 7)
PASS: STANDALONE :: Standalone/standalone-plugin.mlir (3 of 7)
PASS: STANDALONE :: Standalone/standalone-translate.mlir (4 of 7)
PASS: STANDALONE :: CAPI/standalone-capi-test.c (5 of 7)
PASS: STANDALONE :: Standalone/standalone-opt.mlir (6 of 7)
PASS: STANDALONE :: Standalone/dummy.mlir (7 of 7)

Testing Time: 0.23s

Total Discovered Tests: 7
  Unsupported: 1 (14.29%)
  Passed     : 6 (85.71%)
```

Six tests pass and the Python smoke test is unsupported, for the reason given in section 10.3. The order of the lines varies between runs, because lit runs the tests in parallel. Section 11.2 maps each test to the feature it checks.

---

## 13. Deriving the Project, and Your Own Dialect, Yourself

The project is meant to be copied, so the useful question is how to arrive at each piece without having it: what the CMake helpers generate, which C++ you still owe after TableGen, which macro unlocks which part of a generated file, which symbols a plugin must export, and what a rename has to touch. Each step below answers one of those with a tool.

| Piece | Where it comes from | Step |
| --- | --- | --- |
| The six `.inc` files and their names | `add_mlir_dialect` in the installed `AddMLIR.cmake` | 13.1 |
| `initialize()`, `registerTypes()` (hand-written) | members declared by TableGen but defined nowhere in its output | 13.2 |
| The `GET_...`/`GEN_PASS_...` macros in each `.cpp` | the `#ifdef` guards of each generator's output | 13.3 |
| `mlirGetDialectPluginInfo`, `mlirGetPassPluginInfo` | the installed plugin headers | 13.4 |
| Your own dialect | a rename of every `standalone` in names and contents | 13.5, 13.6 |

### 13.1 Step 1: read what `add_mlir_dialect` generates

`include/Standalone/CMakeLists.txt` uses one line for the whole dialect, `add_mlir_dialect(StandaloneOps standalone)`. The installed CMake module says what it expands to:

```bash
cd /Users/roy/study/mlir/standalone
grep -A10 "^function(add_mlir_dialect " /opt/homebrew/opt/llvm@20/lib/cmake/mlir/AddMLIR.cmake
```

Real output **[45]**:
```text
function(add_mlir_dialect dialect dialect_namespace)
  set(LLVM_TARGET_DEFINITIONS ${dialect}.td)
  mlir_tablegen(${dialect}.h.inc -gen-op-decls)
  mlir_tablegen(${dialect}.cpp.inc -gen-op-defs)
  mlir_tablegen(${dialect}Types.h.inc -gen-typedef-decls -typedefs-dialect=${dialect_namespace})
  mlir_tablegen(${dialect}Types.cpp.inc -gen-typedef-defs -typedefs-dialect=${dialect_namespace})
  mlir_tablegen(${dialect}Dialect.h.inc -gen-dialect-decls -dialect=${dialect_namespace})
  mlir_tablegen(${dialect}Dialect.cpp.inc -gen-dialect-defs -dialect=${dialect_namespace})
  add_public_tablegen_target(MLIR${dialect}IncGen)
  add_dependencies(mlir-headers MLIR${dialect}IncGen)
endfunction()
```

Six `mlir_tablegen` calls, each producing one file named after the first argument, plus a target `MLIRStandaloneOpsIncGen` that the dialect library must depend on (`DEPENDS MLIRStandaloneOpsIncGen` in `lib/Standalone/CMakeLists.txt`, appendix A.2) so the files exist before any `.cpp` that includes them is compiled. This is where the names in section 3.3 come from.

### 13.2 Step 2: find the C++ that TableGen leaves to you

Every member TableGen *declares* but does not *define* has to be written by hand. Running all six generators and looking for definitions of the dialect's members shows which ones are covered:

```bash
cd /Users/roy/study/mlir/standalone
for g in -gen-op-decls -gen-op-defs "-gen-typedef-decls -typedefs-dialect=standalone" "-gen-typedef-defs -typedefs-dialect=standalone" -gen-dialect-decls -gen-dialect-defs; do
  mlir-tblgen $g -I include -I /opt/homebrew/opt/llvm@20/include include/Standalone/StandaloneOps.td
done | grep -E "StandaloneDialect::(initialize|registerTypes|parseType|printType|StandaloneDialect|~StandaloneDialect)"
```

Real output **[46]**:
```text
::mlir::Type StandaloneDialect::parseType(::mlir::DialectAsmParser &parser) const {
void StandaloneDialect::printType(::mlir::Type type,
StandaloneDialect::StandaloneDialect(::mlir::MLIRContext *context)
StandaloneDialect::~StandaloneDialect() = default;
```

The constructor, destructor, `parseType` and `printType` are generated; `initialize()` (declared by every dialect, section 3.2) and `registerTypes()` (declared through `extraClassDeclaration`) are not. Those are exactly the two functions in `StandaloneDialect.cpp` and `StandaloneTypes.cpp`. If one is missing, the build fails at link time with an undefined symbol naming it.

### 13.3 Step 3: list the macros each generated file offers

A generated file does nothing unless the includer defines the right macro. The `#ifdef` guards list them:

```bash
cd /Users/roy/study/mlir/standalone
for g in -gen-op-decls -gen-op-defs "-gen-typedef-decls -typedefs-dialect=standalone" "-gen-typedef-defs -typedefs-dialect=standalone"; do
  echo "$g: $(mlir-tblgen $g -I include -I /opt/homebrew/opt/llvm@20/include include/Standalone/StandaloneOps.td | grep -oE "^#ifdef [A-Z_]+" | cut -d" " -f2 | sort -u | tr "\n" " ")"
done
echo "-gen-pass-decls: $(mlir-tblgen -gen-pass-decls -I include -I /opt/homebrew/opt/llvm@20/include include/Standalone/StandalonePasses.td | grep -oE "^#ifdef [A-Z_]+" | cut -d" " -f2 | sort -u | tr "\n" " ")"
```

Real output **[47]**:
```text
-gen-op-decls: GET_OP_CLASSES 
-gen-op-defs: GET_OP_CLASSES GET_OP_LIST 
-gen-typedef-decls -typedefs-dialect=standalone: GET_TYPEDEF_CLASSES 
-gen-typedef-defs -typedefs-dialect=standalone: GET_TYPEDEF_CLASSES GET_TYPEDEF_LIST 
-gen-pass-decls: GEN_PASS_CLASSES GEN_PASS_DECL GEN_PASS_DECL_STANDALONESWITCHBARFOO GEN_PASS_DEF_STANDALONESWITCHBARFOO GEN_PASS_REGISTRATION 
```

Each macro has its includers in the project: `GET_OP_CLASSES` in `StandaloneOps.h` (declarations) and `StandaloneOps.cpp` (definitions), `GET_OP_LIST` in `initialize()`, `GET_TYPEDEF_CLASSES` in `StandaloneTypes.h` and `StandaloneTypes.cpp`, `GET_TYPEDEF_LIST` in `registerTypes()`, `GEN_PASS_DECL` and `GEN_PASS_REGISTRATION` in `StandalonePasses.h`, `GEN_PASS_DEF_STANDALONESWITCHBARFOO` in `StandalonePasses.cpp`. The per-pass `GEN_PASS_DECL_STANDALONESWITCHBARFOO` is the fine-grained alternative to `GEN_PASS_DECL` for a file that wants one pass only, and `GEN_PASS_CLASSES` is the deprecated block cut off in section 6.2.

### 13.4 Step 4: find the plugin entry points

The names and signatures a plugin must export are documented in the installed headers:

```bash
cd /Users/roy/study/mlir/standalone
grep -n -B1 "^mlirGet.*PluginInfo();" /opt/homebrew/opt/llvm@20/include/mlir/Tools/Plugins/*.h
grep -n "define MLIR_PLUGIN_API_VERSION" /opt/homebrew/opt/llvm@20/include/mlir/Tools/Plugins/PassPlugin.h
```

Real output **[48]**:
```text
/opt/homebrew/opt/llvm@20/include/mlir/Tools/Plugins/DialectPlugin.h-103-extern "C" ::mlir::DialectPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
/opt/homebrew/opt/llvm@20/include/mlir/Tools/Plugins/DialectPlugin.h:104:mlirGetDialectPluginInfo();
--
/opt/homebrew/opt/llvm@20/include/mlir/Tools/Plugins/PassPlugin.h-109-extern "C" ::mlir::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
/opt/homebrew/opt/llvm@20/include/mlir/Tools/Plugins/PassPlugin.h:110:mlirGetPassPluginInfo();
32:#define MLIR_PLUGIN_API_VERSION 1
```

Those are the two functions in section 9.1, with the same `extern "C" LLVM_ATTRIBUTE_WEAK` declaration, and the version the plugin returns as its first field.

### 13.5 Step 5: measure what a rename has to touch

To start your own dialect, every occurrence of `standalone` must change: in file and directory names (`StandaloneOps.td`, `standalone-opt/`), in TableGen records and namespaces, in CMake target names, in the lit configuration (`config.name = "STANDALONE"`, `%standalone_libs`) and in the C API symbol `mlirGetDialectHandle__standalone__`. Counting them per file (this README and its script excluded, and the three files without the name, `.gitignore` and the two `lit.local.cfg`, filtered out):

```bash
cd /Users/roy/study/mlir/standalone
grep -rIic standalone --exclude-dir=build --exclude=README.md --exclude=examples.sh . | grep -v ":0$" | sort
```

Real output **[49]**:
```text
./CMakeLists.txt:8
./include/CMakeLists.txt:1
./include/Standalone-c/Dialects.h:4
./include/Standalone/CMakeLists.txt:6
./include/Standalone/StandaloneDialect.h:5
./include/Standalone/StandaloneDialect.td:12
./include/Standalone/StandaloneOps.h:5
./include/Standalone/StandaloneOps.td:8
./include/Standalone/StandalonePasses.h:9
./include/Standalone/StandalonePasses.td:5
./include/Standalone/StandaloneTypes.h:5
./include/Standalone/StandaloneTypes.td:11
./lib/CAPI/CMakeLists.txt:2
./lib/CAPI/Dialects.cpp:4
./lib/CMakeLists.txt:1
./lib/Standalone/CMakeLists.txt:8
./lib/Standalone/StandaloneDialect.cpp:9
./lib/Standalone/StandaloneOps.cpp:4
./lib/Standalone/StandalonePasses.cpp:12
./lib/Standalone/StandaloneTypes.cpp:7
./python/CMakeLists.txt:29
./python/StandaloneExtensionNanobind.cpp:8
./python/StandaloneExtensionPybind11.cpp:8
./python/mlir_standalone/dialects/StandaloneOps.td:4
./python/mlir_standalone/dialects/standalone_nanobind.py:2
./python/mlir_standalone/dialects/standalone_pybind11.py:2
./standalone-opt/CMakeLists.txt:5
./standalone-opt/standalone-opt.cpp:7
./standalone-plugin/CMakeLists.txt:6
./standalone-plugin/standalone-plugin.cpp:8
./standalone-translate/CMakeLists.txt:5
./standalone-translate/standalone-translate.cpp:3
./test/CAPI/CMakeLists.txt:8
./test/CAPI/standalone-capi-test.c:7
./test/CMakeLists.txt:10
./test/Standalone/dummy.mlir:5
./test/Standalone/standalone-opt.mlir:2
./test/Standalone/standalone-pass-plugin.mlir:1
./test/Standalone/standalone-plugin.mlir:3
./test/Standalone/standalone-translate.mlir:1
./test/lit.cfg.py:11
./test/lit.site.cfg.py.in:2
./test/python/smoketest.py:7
```

Forty-three files mention the name, and the three spellings (`standalone`, `Standalone`, `STANDALONE`) all occur. Nothing else is project-specific, which makes the rename mechanical.

### 13.6 Step 6: rename it and rebuild

The run below copies the project to a scratch directory, renames `standalone` to `hello` in all three spellings (paths first, deepest first, then contents), confirms nothing is left, builds it, runs its test suite, and uses the new `hello-opt` tool. It takes about two and a half minutes:

```bash
cd /Users/roy/study/mlir/standalone
T=$(mktemp -d)
rsync -a --exclude build --exclude README.md --exclude examples.sh ./ "$T/hello/"
cd "$T/hello"
find . -depth -name "*[Ss]tandalone*" | while read -r p; do
  mv "$p" "$(dirname "$p")/$(basename "$p" | sed "s/Standalone/Hello/g; s/standalone/hello/g")"
done
grep -rlI -i standalone . | xargs sed -i "" "s/Standalone/Hello/g; s/standalone/hello/g; s/STANDALONE/HELLO/g"
echo "occurrences left: $(grep -rIi standalone . | wc -l | tr -d " ")"
cmake -S . -B build -G Ninja -DLLVM_EXTERNAL_LIT=$(which lit) -DCMAKE_OSX_DEPLOYMENT_TARGET=26.0 > /dev/null
cmake --build build --target check-hello 2>&1 | tail -3
./build/bin/hello-opt --show-dialects
printf '%%0 = arith.constant 2 : i32\n%%1 = hello.foo %%0 : i32\n' | ./build/bin/hello-opt
cd / && rm -rf "$T"
```

Real output **[50]**:
```text
occurrences left: 0
Total Discovered Tests: 7
  Unsupported: 1 (14.29%)
  Passed     : 6 (85.71%)
Available Dialects: arith,builtin,func,hello
module {
  %c2_i32 = arith.constant 2 : i32
  %0 = hello.foo %c2_i32 : i32
}

```

The renamed project builds, passes the same six tests under the new suite name, and `hello-opt` parses `hello.foo`. From here, a new dialect is a matter of editing the `.td` files: add ops to `HelloOps.td`, types to `HelloTypes.td`, passes to `HelloPasses.td`, and write the C++ that sections 13.2 and 13.3 say TableGen leaves to you.

---

## 14. Key Takeaways & Pitfalls

**Takeaways**

- **Out-of-tree uses the same build system as in-tree.** `find_package(MLIR)` plus `TableGen`, `AddLLVM` and `AddMLIR` give you the functions MLIR's own dialects use (section 2.2).
- **One `add_mlir_dialect` line produces six files** named after its first argument, `StandaloneOps*.inc`; each is included by a header or `.cpp` with the right `GET_...` macro (sections 3.3, 13.1, 13.3).
- **TableGen leaves two functions to you:** `initialize()` and anything you declared in `extraClassDeclaration`, here `registerTypes()` (section 13.2).
- **Types are uniqued by their parameters** in the context; the generated storage class does the hashing, comparing and copying (section 4.2).
- **Traits are information, not just checks.** `SameOperandsAndResultType` removes the result type from the syntax and the builders (section 5.4); `Pure` lets the generic CSE and DCE passes merge and delete the op (section 5.5).
- **A pass is a wrapper around patterns**, run by the greedy driver, which also folds and erases dead pure ops (sections 6.3, 6.4).
- **Dialect registration is per tool.** `standalone-opt` reads only `arith`, `builtin`, `func` and `standalone`; the stock `mlir-opt` reads every upstream dialect but not `standalone` (section 7.2). Each translation registers its own dialects (section 8.2).
- **Plugins extend the installed `mlir-opt` without a rebuild**, at the price of being tied to that exact MLIR version (section 9).
- **The C API exposes a dialect as a handle** of three hooks, generated by two macros; the Python bindings are built on that handle (section 10).
- **A lit test is a stored manual check**: a `RUN:` command and the `CHECK:` lines its output must match (section 11).

**Pitfalls**

- **`llvm-lit` is not installed by Homebrew.** Install `lit` and pass `-DLLVM_EXTERNAL_LIT=$(which lit)`, or `check-standalone` cannot run (section 12.1).
- **Deployment-target warnings** flood the link step unless `-DCMAKE_OSX_DEPLOYMENT_TARGET=26.0` matches the Homebrew archives (section 12.1).
- **The Python bindings need an MLIR built with them.** Homebrew's is not, so `python/` is skipped and the smoke test is unsupported (section 10.3).
- **The generated file is `StandaloneOpsDialect.h.inc`, not `StandaloneDialect.h.inc`**: names follow the first argument of `add_mlir_dialect` (section 3.3).
- **A registered dialect rejects unknown ops in its namespace**, even with `--allow-unregistered-dialect` (section 3.3).
- **The rename pattern doesn't check for a symbol clash**; the verifier after the pass does (section 6.4).
- **`standalone-translate --option` registers no dialects**, so it cannot parse anything beyond `builtin` (section 8.2).
- **`standalone-opt` is large (86 MB)** because it links every upstream dialect and conversion library statically for `registerAllPasses()` (section 7.3).
- **The pass plugin alone cannot parse standalone types**; use `--load-dialect-plugin` when the input contains them (section 9.2).

---

## Appendix A: The Build Wiring

### A.1 The build tree

After `cmake --build build`, the generated `.inc` files sit under `build/include/Standalone/`, next to `.d` dependency files that tell Ninja which `.td` files each one was generated from:

```bash
cd /Users/roy/study/mlir/standalone
ls build/include/Standalone/
```

Real output **[51]**:
```text
CMakeFiles
StandaloneOps.cpp.inc
StandaloneOps.cpp.inc.d
StandaloneOps.h.inc
StandaloneOps.h.inc.d
StandaloneOpsDialect.cpp.inc
StandaloneOpsDialect.cpp.inc.d
StandaloneOpsDialect.h.inc
StandaloneOpsDialect.h.inc.d
StandaloneOpsTypes.cpp.inc
StandaloneOpsTypes.cpp.inc.d
StandaloneOpsTypes.h.inc
StandaloneOpsTypes.h.inc.d
StandalonePasses.h.inc
StandalonePasses.h.inc.d
cmake_install.cmake
```

### A.2 Every `CMakeLists.txt`

***include/Standalone/CMakeLists.txt***
```cmake
add_mlir_dialect(StandaloneOps standalone)                                          # (1)
add_mlir_doc(StandaloneDialect StandaloneDialect Standalone/ -gen-dialect-doc)      # (2)
add_mlir_doc(StandaloneOps StandaloneOps Standalone/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS StandalonePasses.td)                                    # (3)
mlir_tablegen(StandalonePasses.h.inc --gen-pass-decls)
add_public_tablegen_target(MLIRStandalonePassesIncGen)
```

**(1)** The six dialect, op and type files (section 13.1). **(2)** Markdown documentation, built by the `mlir-doc` target (appendix B). **(3)** Passes have no one-line helper: set the input file, run one generator, and wrap the output in a target, `MLIRStandalonePassesIncGen`. `include/CMakeLists.txt` and `lib/CMakeLists.txt` only descend into their subdirectories.

***lib/Standalone/CMakeLists.txt***
```cmake
add_mlir_dialect_library(MLIRStandalone                   # (1)
        StandaloneTypes.cpp
        StandaloneDialect.cpp
        StandaloneOps.cpp
        StandalonePasses.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Standalone

        DEPENDS                                           # (2)
        MLIRStandaloneOpsIncGen
        MLIRStandalonePassesIncGen

        LINK_LIBS PUBLIC                                  # (3)
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRFuncDialect
        )
```

**(1)** `add_mlir_dialect_library` is `add_mlir_library` plus adding the library to the global `MLIR_DIALECT_LIBS` list. **(2)** The two TableGen targets must run before any of the four `.cpp` files is compiled. **(3)** What the dialect's code uses: the IR core, the type-inference interface (`InferTypeOpInterface`, section 5.2) and the `func` dialect (the pass matches `func::FuncOp`, section 6.3). `PUBLIC` passes them on to everything that links `MLIRStandalone`.

***lib/CAPI/CMakeLists.txt***
```cmake
add_mlir_public_c_api_library(StandaloneCAPI
  Dialects.cpp
  LINK_LIBS PUBLIC
  MLIRStandalone
)
```

A C API library is compiled with the visibility settings the C API needs and is registered so that aggregates (below) and the Python bindings can embed it.

***test/CMakeLists.txt***
```cmake
llvm_canonicalize_cmake_booleans(
        MLIR_ENABLE_BINDINGS_PYTHON
)

configure_lit_site_cfg(                                                        # (1)
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(STANDALONE_TEST_DEPENDS                                                    # (2)
        FileCheck count not
        standalone-capi-test
        standalone-opt
        standalone-translate
        StandalonePlugin
        )
if(MLIR_ENABLE_BINDINGS_PYTHON)
  list(APPEND STANDALONE_TEST_DEPENDS StandalonePythonModules)
endif()

add_lit_testsuite(check-standalone "Running the standalone regression tests"   # (3)
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${STANDALONE_TEST_DEPENDS}
        )
set_target_properties(check-standalone PROPERTIES FOLDER "Tests")

add_lit_testsuites(STANDALONE ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${STANDALONE_TEST_DEPENDS})   # (4)

add_subdirectory(CAPI)
```

**(1)** Writes `build/test/lit.site.cfg.py` from the template (section 11.1); `llvm_canonicalize_cmake_booleans` turns `MLIR_ENABLE_BINDINGS_PYTHON` into a Python `True`/`False` first. **(2)** What must be built before the tests run. `FileCheck`, `count` and `not` are names of LLVM build targets; in a component build there are no such targets, and the installed tools are found through `config.llvm_tools_dir`. **(3)** The `check-standalone` target: lit over `build/test`. **(4)** One extra target per test directory, such as `check-standalone-standalone`, to run a subset.

***test/CAPI/CMakeLists.txt***
```cmake
add_mlir_aggregate(StandaloneCAPITestLib
  SHARED
  EMBED_LIBS
  MLIRCAPIIR
  # TODO: Remove this in favor of showing fine grained dialect registration
  # (once available).
  MLIRCAPIRegisterEverything
  StandaloneCAPI
)

add_llvm_executable(standalone-capi-test
  standalone-capi-test.c
)
llvm_update_compile_flags(standalone-capi-test)
target_link_libraries(standalone-capi-test
  PRIVATE StandaloneCAPITestLib)
```

`add_mlir_aggregate` links the listed C API libraries, and everything they depend on, into one shared library (section 10.2). The test executable is plain C and links only that.

***standalone-translate/CMakeLists.txt***
```cmake
set(LLVM_LINK_COMPONENTS
  Support
  )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

add_llvm_executable(standalone-translate
  standalone-translate.cpp
  )
llvm_update_compile_flags(standalone-translate)
target_link_libraries(standalone-translate
  PRIVATE
  ${dialect_libs}
  ${translation_libs}
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSPIRVDialect
  MLIRTranslateLib
  MLIRSupport
  )

mlir_check_link_libraries(standalone-translate)
```

The same pattern as `standalone-opt` (section 7.3), with the translation libraries that `registerAllTranslations()` needs and `MLIRTranslateLib` for `mlirTranslateMain`. It does not link `MLIRStandalone`, consistent with the tool never registering the dialect (section 8.1). The `standalone-opt` and `standalone-plugin` files are explained in sections 7.3 and 9.1, the top-level file in section 2.2, and `python/CMakeLists.txt` is summarized in section 10.3.

### A.3 How the plugin links

`PLUGIN_TOOL mlir-opt` (section 9.1) changes the link flags of the plugin:

```bash
cd /Users/roy/study/mlir/standalone
grep -A6 "^build lib/StandalonePlugin.dylib:" build/build.ninja | grep -o "LINK_FLAGS = .*"
```

Real output **[52]**:
```text
LINK_FLAGS = -bundle -Wl,-headerpad_max_install_names -L/opt/homebrew/opt/llvm@20/lib -Wl,-flat_namespace -Wl,-undefined -Wl,dynamic_lookup  -Wl,-dead_strip -Wl,-no_warn_duplicate_libraries
```

`-bundle` makes a loadable bundle. `-undefined dynamic_lookup` lets the link succeed with symbols still undefined, and `-flat_namespace` makes the dynamic loader resolve them by name against every image already in the process. At load time those images are `mlir-opt` and its `libMLIR.dylib`, so the plugin's references to the MLIR core bind to the host's copy. `-dead_strip` drops the parts of the archives the plugin does not use, which keeps it at a few megabytes.

---

## Appendix B: Generated Documentation

`add_mlir_doc` (appendix A.2) generates Markdown documentation from the `summary` and `description` fields, the same way the pages on mlir.llvm.org are produced. The `mlir-doc` build target writes them to `build/docs/Standalone/`; the generators can also be run directly:

```bash
cd /Users/roy/study/mlir/standalone
mlir-tblgen -gen-dialect-doc -I include -I /opt/homebrew/opt/llvm@20/include include/Standalone/StandaloneDialect.td
```

Real output **[53]**:
```text
<!-- Autogenerated by mlir-tblgen; don't manually edit -->
# 'standalone' Dialect

A standalone out-of-tree MLIR dialect.

This dialect is an example of an out-of-tree MLIR dialect designed to
illustrate the basic setup required to develop MLIR-based tools without
working inside of the LLVM source tree.

[TOC]

```

```bash
cd /Users/roy/study/mlir/standalone
mlir-tblgen -gen-op-doc -I include -I /opt/homebrew/opt/llvm@20/include include/Standalone/StandaloneOps.td
```

Real output **[54]**:
````text
<!-- Autogenerated by mlir-tblgen; don't manually edit -->
### `standalone.foo` (standalone::FooOp)

_Illustrates how to define an operation._


Syntax:

```
operation ::= `standalone.foo` $input attr-dict `:` type($input)
```

The `standalone.foo` operation illustrates how to define a new
operation in a dialect. It uses an operation trait to declare that it
has no side effects.

This operation takes an integer argument and returns an integer.

Example:

```mlir
%0 = arith.constant 2 : i32
// Apply the foo operation to %0
%1 = standalone.foo %0 : i32
```

Traits: `AlwaysSpeculatableImplTrait`, `SameOperandsAndResultType`

Interfaces: `ConditionallySpeculatable`, `InferTypeOpInterface`, `NoMemoryEffect (MemoryEffectOpInterface)`

Effects: `MemoryEffects::Effect{}`

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `input` | 32-bit signless integer

#### Results:

| Result | Description |
| :----: | ----------- |
| `res` | 32-bit signless integer

````

The op page is generated entirely from the record in section 5.1: the syntax from `assemblyFormat`, the description with its example, the traits and interfaces (including those `Pure` expands to, section 5.5), and a table per operand and result built from the `I32` constraints.

---

## Appendix C: `examples.sh`

`examples.sh` runs the commands of this README straight from the file: every ` ```bash ` block, in order, except the build block of section 12.1 (its first command after `cd` is `cmake`) and file excerpts. Each block's output label carries the same number, as in "Real output **[7]**". Build the project first (section 12.1).

```text
./examples.sh          # run every command, in README order
./examples.sh 7        # run command [7] only (several numbers are fine)
./examples.sh -l       # list the numbered commands without running them
```

The list:

```text
[ 1] 3.2   mlir-tblgen -gen-dialect-decls -I include -I /opt/homebrew/opt/llvm@20/include include/Standalone/StandaloneOps.td
[ 2] 3.2   mlir-tblgen -gen-dialect-defs -I include -I /opt/homebrew/opt/llvm@20/include include/Standalone/StandaloneOps.td
[ 3] 3.3   ./build/bin/standalone-opt --allow-unregistered-dialect <<'EOF'
[ 4] 4.2   mlir-tblgen -gen-typedef-decls -typedefs-dialect=standalone -I include -I /opt/homebrew/opt/llvm@20/include include/Standalone/StandaloneOps.td
[ 5] 4.2   mlir-tblgen -gen-typedef-defs -typedefs-dialect=standalone -I include -I /opt/homebrew/opt/llvm@20/include include/Standalone/StandaloneOps.td | sed -n '/^struct CustomTypeStorage/,/^} \/\/ namespace detail/p;/^::mlir::Type CustomType::parse/,/^}/p;/^void CustomType::print/,/^}/p'
[ 6] 4.2   mlir-tblgen -gen-typedef-defs -typedefs-dialect=standalone -I include -I /opt/homebrew/opt/llvm@20/include include/Standalone/StandaloneOps.td | sed -n '/^static ::mlir::OptionalParseResult generatedTypeParser/,/^}/p;/^::mlir::Type StandaloneDialect::parseType/,/^}/p'
[ 7] 4.4   ./build/bin/standalone-opt <<'EOF'
[ 8] 4.4   ./build/bin/standalone-opt <<'EOF'
[ 9] 4.4   ./build/bin/standalone-opt <<'EOF'
[10] 5.2   mlir-tblgen -gen-op-decls -I include -I /opt/homebrew/opt/llvm@20/include include/Standalone/StandaloneOps.td | sed -n '/^class FooOp :/,/^};/p'
[11] 5.3   mlir-tblgen -gen-op-defs -I include -I /opt/homebrew/opt/llvm@20/include include/Standalone/StandaloneOps.td | sed -n '/^static ::llvm::LogicalResult __mlir_ods_local_type_constraint/,/^}/p;/FooOp::verifyInvariantsImpl() {/,/^}/p'
[12] 5.3   ./build/bin/standalone-opt <<'EOF'
[13] 5.3   ./build/bin/standalone-opt <<'EOF'
[14] 5.4   ./build/bin/standalone-opt --mlir-print-op-generic test/Standalone/dummy.mlir
[15] 5.4   ./build/bin/standalone-opt <<'EOF'
[16] 5.4   mlir-tblgen -gen-op-defs -I include -I /opt/homebrew/opt/llvm@20/include include/Standalone/StandaloneOps.td | sed -n '/FooOp::inferReturnTypes(/,/^}/p;/FooOp::parse(/,/^}/p'
[17] 5.5   ./build/bin/standalone-opt --allow-unregistered-dialect --cse --canonicalize <<'EOF'
[18] 5.5   diff <(mlir-tblgen -gen-op-decls -I include -I /opt/homebrew/opt/llvm@20/include include/Standalone/StandaloneOps.td | grep "^class FooOp " | tr "," "\n") \
[19] 6.2   mlir-tblgen -gen-pass-decls -I include -I /opt/homebrew/opt/llvm@20/include include/Standalone/StandalonePasses.td | sed '/Deprecated/,$d'
[20] 6.4   ./build/bin/standalone-opt --standalone-switch-bar-foo test/Standalone/standalone-pass-plugin.mlir
[21] 6.4   ./build/bin/standalone-opt --standalone-switch-bar-foo test/Standalone/dummy.mlir
[22] 6.4   ./build/bin/standalone-opt --standalone-switch-bar-foo <<'EOF'
[23] 6.4   ./build/bin/standalone-opt --help | grep -A1 "standalone-switch"
[24] 7.2   ./build/bin/standalone-opt --show-dialects
[25] 7.2   ./build/bin/standalone-opt <<'EOF'
[26] 7.2   mlir-opt test/Standalone/dummy.mlir
[27] 7.3   du -h build/bin/*
[28] 8.2   ./build/bin/standalone-translate --help | sed -n '/Translations to perform/,/serialize-spirv/p'
[29] 8.2   echo 'module {}' | ./build/bin/standalone-translate --option
[30] 8.2   ./build/bin/standalone-translate --option test/Standalone/dummy.mlir
[31] 8.2   ./build/bin/standalone-translate --mlir-to-llvmir <<'EOF'
[32] 9.1   otool -L build/lib/StandalonePlugin.dylib
[33] 9.2   mlir-opt test/Standalone/standalone-plugin.mlir --load-dialect-plugin=build/lib/StandalonePlugin.dylib --pass-pipeline="builtin.module(standalone-switch-bar-foo)"
[34] 9.2   mlir-opt test/Standalone/standalone-pass-plugin.mlir --load-pass-plugin=build/lib/StandalonePlugin.dylib --pass-pipeline="builtin.module(standalone-switch-bar-foo)"
[35] 9.2   mlir-opt test/Standalone/standalone-pass-plugin.mlir --pass-pipeline="builtin.module(standalone-switch-bar-foo)"
[36] 9.2   mlir-opt test/Standalone/standalone-plugin.mlir --load-pass-plugin=build/lib/StandalonePlugin.dylib --pass-pipeline="builtin.module(standalone-switch-bar-foo)"
[37] 10.1  clang++ -E -P -std=c++17 -I include -I build/include -I /opt/homebrew/opt/llvm@20/include lib/CAPI/Dialects.cpp | tail -1 | clang-format
[38] 10.2  ./build/bin/standalone-capi-test
[39] 10.2  otool -L build/bin/standalone-capi-test
[40] 10.3  grep MLIR_ENABLE_BINDINGS_PYTHON /opt/homebrew/opt/llvm@20/lib/cmake/mlir/MLIRConfig.cmake
[41] 11.2  lit -a build/test/Standalone/dummy.mlir
[42] 11.2  ./build/bin/standalone-opt test/Standalone/dummy.mlir | FileCheck test/Standalone/dummy.mlir
[43] 11.2  ./build/bin/standalone-opt test/Standalone/dummy.mlir | FileCheck <(sed "s/= standalone.foo/= standalone.bar/" test/Standalone/dummy.mlir)
[44] 12.2  lit -v build/test
[45] 13.1  grep -A10 "^function(add_mlir_dialect " /opt/homebrew/opt/llvm@20/lib/cmake/mlir/AddMLIR.cmake
[46] 13.2  for g in -gen-op-decls -gen-op-defs "-gen-typedef-decls -typedefs-dialect=standalone" "-gen-typedef-defs -typedefs-dialect=standalone" -gen-dialect-decls -gen-dialect-defs; do
[47] 13.3  for g in -gen-op-decls -gen-op-defs "-gen-typedef-decls -typedefs-dialect=standalone" "-gen-typedef-defs -typedefs-dialect=standalone"; do
[48] 13.4  grep -n -B1 "^mlirGet.*PluginInfo();" /opt/homebrew/opt/llvm@20/include/mlir/Tools/Plugins/*.h
[49] 13.5  grep -rIic standalone --exclude-dir=build --exclude=README.md --exclude=examples.sh . | grep -v ":0$" | sort
[50] 13.6  T=$(mktemp -d)
[51] A.1   ls build/include/Standalone/
[52] A.3   grep -A6 "^build lib/StandalonePlugin.dylib:" build/build.ninja | grep -o "LINK_FLAGS = .*"
[53] B     mlir-tblgen -gen-dialect-doc -I include -I /opt/homebrew/opt/llvm@20/include include/Standalone/StandaloneDialect.td
[54] B     mlir-tblgen -gen-op-doc -I include -I /opt/homebrew/opt/llvm@20/include include/Standalone/StandaloneOps.td
```

---

## Links

- Upstream example: [`mlir/examples/standalone`](https://github.com/llvm/llvm-project/tree/release/20.x/mlir/examples/standalone)
- [Creating a Dialect](https://mlir.llvm.org/docs/Tutorials/CreatingADialect/)
- [Operation Definition Specification (ODS)](https://mlir.llvm.org/docs/DefiningDialects/Operations/)
- [Defining Dialect Attributes and Types](https://mlir.llvm.org/docs/DefiningDialects/AttributesAndTypes/)
- [Pass Infrastructure](https://mlir.llvm.org/docs/PassManagement/) and [Pattern Rewriting](https://mlir.llvm.org/docs/PatternRewriter/)
- [MLIR C API](https://mlir.llvm.org/docs/CAPI/) and [Python Bindings](https://mlir.llvm.org/docs/Bindings/Python/)
- [lit](https://llvm.org/docs/CommandGuide/lit.html) and [FileCheck](https://llvm.org/docs/CommandGuide/FileCheck.html)
