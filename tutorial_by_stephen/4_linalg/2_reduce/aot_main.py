"""Driver for reduce.mlir — calls the named linalg.reduce kernel (row sums) and
checks it against NumPy's a.sum(axis=1)."""

import os
import sys

import ctypes
import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from np_memref import (MemRef1D, MemRef2D,                    # noqa: E402
                       numpy_to_memref_1d, numpy_to_memref_2d)


def main():
  lib = ctypes.CDLL("./build/libreduce.dylib")
  fn = lib._mlir_ciface_row_sum
  fn.argtypes = [ctypes.POINTER(MemRef2D), ctypes.POINTER(MemRef1D)]
  fn.restype = None

  matrix = np.arange(12, dtype=np.float32).reshape(3, 4)
  out = np.zeros(3, dtype=np.float32)   # reduce accumulates in place -> must be zeroed

  fn(ctypes.byref(numpy_to_memref_2d(matrix)),
     ctypes.byref(numpy_to_memref_1d(out)))

  np.testing.assert_array_equal(out, matrix.sum(axis=1))
  print("linalg.reduce successful!")
  print(out)


if __name__ == "__main__":
  main()
