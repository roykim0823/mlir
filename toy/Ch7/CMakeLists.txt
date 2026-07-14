cmake_minimum_required(VERSION 3.20)

# ---------------------------------------------------------------------------
# Standalone-mode boilerplate.
# Runs only when this chapter is configured directly (cmake -S Ch7).
# In the superbuild (cmake -S toy/), ../CMakeLists.txt already did all this.
# ---------------------------------------------------------------------------
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  if(APPLE)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "26.0" CACHE STRING "macOS Deployment Target" FORCE)
  endif()
  project(toy-ch7)

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

# ---------------------------------------------------------------------------
# Chapter targets
# ---------------------------------------------------------------------------
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
