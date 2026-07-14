"""Driver for add_vec_to_mat.mlir — broadcasts a vector across a matrix's rows
and adds, checking against NumPy's own broadcasting."""

import os
import sys

import ctypes
import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from np_memref import (MemRef1D, MemRef2D,                    # noqa: E402
                       numpy_to_memref_1d, numpy_to_memref_2d)


def main():
  lib = ctypes.CDLL("./build/libbcast.dylib")
  fn = lib._mlir_ciface_add_vec_to_mat
  fn.argtypes = [ctypes.POINTER(MemRef2D),
                 ctypes.POINTER(MemRef1D),
                 ctypes.POINTER(MemRef2D)]
  fn.restype = None

  matrix = np.ones((3, 4), dtype=np.float32)
  vector = np.array([1, 2, 3, 4], dtype=np.float32)
  out = np.zeros((3, 4), dtype=np.float32)

  fn(ctypes.byref(numpy_to_memref_2d(matrix)),
     ctypes.byref(numpy_to_memref_1d(vector)),
     ctypes.byref(numpy_to_memref_2d(out)))

  np.testing.assert_array_equal(out, matrix + vector)   # NumPy broadcasting
  print("linalg.broadcast + add successful!")
  print(out)


if __name__ == "__main__":
  main()
