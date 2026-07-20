cmake_minimum_required(VERSION 3.20)

# Superbuild for all Toy tutorial chapters.
#
# Configure once, build any chapter incrementally:
#   cmake --preset default          (or: cmake -B build -G Ninja -S .)
#   ninja -C build toyc-ch3         (or: cmake --build build --target toyc-ch3)
#   ninja -C build                  (everything)
#
# Each ChN/ remains configurable as a standalone project as well; the
# boilerplate below is then executed by the chapter itself (see the
# CMAKE_SOURCE_DIR guard at the top of each ChN/CMakeLists.txt).
#
# The build machinery is documented in README.md ("The build system");
# each ChN/CMakeLists.txt header states what that chapter adds.

if(APPLE)
  set(CMAKE_OSX_DEPLOYMENT_TARGET "26.0" CACHE STRING "macOS Deployment Target" FORCE)
endif()

project(toy-tutorial)

find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${MLIR_INCLUDE_DIRS}
                    ${LLVM_INCLUDE_DIRS})

# Collect every chapter binary in build/bin/ instead of build/ChN/.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(Ch1)
add_subdirectory(Ch2)
add_subdirectory(Ch3)
add_subdirectory(Ch4)
add_subdirectory(Ch5)
add_subdirectory(Ch6)
add_subdirectory(Ch7)
