"""Driver for the fused addmul kernel — checks the fused (a + b) * c against NumPy.

The fused func returns a tensor; build.sh bufferizes it and runs
-buffer-results-to-out-params, so the C interface is the usual in-place shape:
addmul(a, b, c, out).
"""

import os
import sys

import ctypes
import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from np_memref import MemRef1D, numpy_to_memref_1d   # noqa: E402


def main():
  lib = ctypes.CDLL("./build/libfused.dylib")
  fn = lib._mlir_ciface_addmul
  fn.argtypes = [ctypes.POINTER(MemRef1D)] * 4   # a, b, c, out
  fn.restype = None

  a = np.arange(10, dtype=np.float32)
  b = np.ones(10, dtype=np.float32) * 2
  c = np.arange(10, dtype=np.float32) + 1
  out = np.zeros(10, dtype=np.float32)

  fn(ctypes.byref(numpy_to_memref_1d(a)),
     ctypes.byref(numpy_to_memref_1d(b)),
     ctypes.byref(numpy_to_memref_1d(c)),
     ctypes.byref(numpy_to_memref_1d(out)))

  np.testing.assert_allclose(out, (a + b) * c)
  print("fused (a + b) * c successful!")
  print("  a   =", a)
  print("  b   =", b)
  print("  c   =", c)
  print("  out =", out, " == (a + b) * c")


if __name__ == "__main__":
  main()
