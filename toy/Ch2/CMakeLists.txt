cmake_minimum_required(VERSION 3.20)

# ---------------------------------------------------------------------------
# Standalone-mode boilerplate.
# Runs only when this chapter is configured directly (cmake -S Ch2).
# In the superbuild (cmake -S toy/), ../CMakeLists.txt already did all this.
# ---------------------------------------------------------------------------
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  if(APPLE)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "26.0" CACHE STRING "macOS Deployment Target" FORCE)
  endif()
  project(toy-ch2)

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
add_subdirectory(include)

add_executable(toyc-ch2
  toyc.cpp
  parser/AST.cpp
  mlir/MLIRGen.cpp
  mlir/Dialect.cpp
  )

add_dependencies(toyc-ch2 ToyCh2OpsIncGen)  # Added to fix include dependency for toy/Dialect.h.inc

include_directories(include/)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)

target_link_libraries(toyc-ch2
  PRIVATE
    MLIR                      # libMLIR.dylib (all dialects, passes, conversions)
    LLVM                      # libLLVM.dylib (all targets, all components)
    )
