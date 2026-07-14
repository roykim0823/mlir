"""Driver for the manual transforms — proves the hand-written interchange,
skewing, and wavefront rewrites produce *identical* results to their originals.
Each source file is compiled to its own shared library (see build.sh)."""

import os
import sys

import ctypes
import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from np_memref2d import MemRef2DDescriptor, numpy_to_memref2d   # noqa: E402

PTR = ctypes.POINTER(MemRef2DDescriptor)


def load(lib_path, funcs):
  lib = ctypes.CDLL(lib_path)
  for name, nargs in funcs:
    fn = getattr(lib, "_mlir_ciface_" + name)
    fn.argtypes = [PTR] * nargs
    fn.restype = None
  return lib


def main():
  # --- interchange (interchange_manual.mlir) -------------------------------
  ic = load("./build/libinterchange_manual.dylib",
            [("copy_ij", 2), ("copy_ji", 2)])
  A = np.arange(20, dtype=np.float32).reshape(4, 5)
  B_ij = np.zeros((4, 5), dtype=np.float32)
  B_ji = np.zeros((4, 5), dtype=np.float32)
  ic._mlir_ciface_copy_ij(ctypes.byref(numpy_to_memref2d(A)),
                          ctypes.byref(numpy_to_memref2d(B_ij)))
  ic._mlir_ciface_copy_ji(ctypes.byref(numpy_to_memref2d(A)),
                          ctypes.byref(numpy_to_memref2d(B_ji)))
  assert np.array_equal(B_ij, B_ji) and np.array_equal(B_ij, A)
  print("interchange:  copy_ij == copy_ji == A  ✓")

  # --- skewing + wavefront (skewing_manual.mlir) ---------------------------
  sk = load("./build/libskewing_manual.dylib",
            [("stencil", 1), ("stencil_skewed", 1), ("stencil_wavefront", 1)])
  S = np.ones((8, 8), dtype=np.float32)
  S_skew = S.copy()
  S_wave = S.copy()
  sk._mlir_ciface_stencil(ctypes.byref(numpy_to_memref2d(S)))
  sk._mlir_ciface_stencil_skewed(ctypes.byref(numpy_to_memref2d(S_skew)))
  sk._mlir_ciface_stencil_wavefront(ctypes.byref(numpy_to_memref2d(S_wave)))
  assert np.array_equal(S, S_skew)
  assert np.array_equal(S, S_wave)
  print("skewing:      stencil == stencil_skewed  ✓")
  print("wavefront:    stencil == stencil_wavefront (parallel inner loop)  ✓")
  print("Manual transforms preserve results.")


if __name__ == "__main__":
  main()
