mkdir -p build

# 1. Lower the high-level dialects (func/scf/index/arith/...) to the llvm dialect
mlir-opt example.mlir \
--convert-func-to-llvm \
--convert-math-to-llvm \
--convert-index-to-llvm \
--convert-scf-to-cf \
--convert-cf-to-llvm \
--convert-arith-to-llvm \
--reconcile-unrealized-casts \
-o ./build/example_opt.mlir

# 2. Translate the llvm dialect into textual LLVM IR
mlir-translate ./build/example_opt.mlir -mlir-to-llvmir -o ./build/example.ll

# 3. Compile LLVM IR to a native object file
llc -filetype=obj --relocation-model=pic ./build/example.ll -o ./build/example.o

# 4. Link the object into a shared library (loadable from Python via ctypes)
clang -shared -fPIC ./build/example.o -o ./build/libexample.so

# 5. Link the object into a standalone executable and run it.
#    main() returns 45 (0+1+...+9), which becomes the process exit code.
clang ./build/example.o -o ./build/example
./build/example; echo $?

# 6. (Inspection) Emit native assembly text (.s) — symbolic form, no addresses
llc -filetype=asm --relocation-model=pic ./build/example.ll -o ./build/example.s

# 7. (Inspection) Disassemble the object — shows hex addresses
objdump -d --no-show-raw-insn ./build/example.o > ./build/example.dis
# Or disassemble the dylib (Apple Silicon symbols are prefixed with `_`)
objdump -d --no-show-raw-insn ./build/libexample.so > ./build/libexample.dis

# 8. Alternatively, JIT-execute the lowered MLIR directly — no codegen, no files.
#    Useful for quick iteration; prints the i32 result (45).
mlir-runner -e main -entry-point-result=i32 ./build/example_opt.mlir

# To use the runner utils (e.g. for debug printing) pass the runtime support lib:
mlir-runner -e main -entry-point-result=i32 -shared-libs=/opt/homebrew/opt/llvm@20/lib/libmlir_runner_utils.dylib ./build/example_opt.mlir
