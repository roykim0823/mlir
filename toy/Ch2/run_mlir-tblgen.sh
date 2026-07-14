mkdir -p ./build
mlir-tblgen -gen-dialect-decls ./include/toy/Ops.td -I /opt/homebrew/opt/llvm@20/include/ -o ./build/dialect-decls.inc
mlir-tblgen -gen-dialect-defs ./include/toy/Ops.td -I /opt/homebrew/opt/llvm@20/include/ -o ./build/dialect-defs.inc
mlir-tblgen -gen-op-decls ./include/toy/Ops.td -I /opt/homebrew/opt/llvm@20/include/ -o ./build/op-decls.inc
mlir-tblgen -gen-op-defs ./include/toy/Ops.td -I /opt/homebrew/opt/llvm@20/include/ -o ./build/op-defs.inc