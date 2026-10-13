cmake_minimum_required(VERSION 3.20)

# ---------------------------------------------------------------------------
# Standalone-mode boilerplate.
# Runs only when this chapter is configured directly (cmake -S Ch4).
# ---------------------------------------------------------------------------
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  if(APPLE)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "26.0" CACHE STRING "macOS Deployment Target" FORCE)
  endif()
  # libMLIR.dylib is a Release (NDEBUG) build; MLIR headers declare debug-only
  # helpers under #ifndef NDEBUG, so the chapter must define NDEBUG too.
  if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
  endif()
  project(transform-ch4)

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

  # add_llvm_example() leaves examples out of `all` unless this is ON.
  set(LLVM_BUILD_EXAMPLES ON)
  # The umbrella target the chapter adds itself to (defined by the parent
  # CMakeLists.txt in the superbuild).
  add_custom_target(TransformExample)
endif()

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_subdirectory(include)
add_subdirectory(lib)

add_dependencies(TransformExample transform-opt-ch4)
add_llvm_example(transform-opt-ch4
  transform-opt/transform-opt.cpp)

target_link_libraries(transform-opt-ch4
  PRIVATE
  MLIR                      # libMLIR.dylib (all dialects, passes, extensions)
  MyExtensionCh4
)
