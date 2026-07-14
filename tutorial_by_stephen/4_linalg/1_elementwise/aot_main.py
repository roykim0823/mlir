"""Driver for add.mlir — calls the named linalg.add kernel and checks a + b."""

import os
import sys

import ctypes
import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from np_memref import MemRef1D, numpy_to_memref_1d   # noqa: E402


def main():
  lib = ctypes.CDLL("./build/libadd.dylib")
  addv = lib._mlir_ciface_addv
  addv.argtypes = [ctypes.POINTER(MemRef1D)] * 3
  addv.restype = None

  a = np.arange(10, dtype=np.float32)
  b = np.ones(10, dtype=np.float32) * 10
  c = np.zeros(10, dtype=np.float32)

  addv(ctypes.byref(numpy_to_memref_1d(a)),
       ctypes.byref(numpy_to_memref_1d(b)),
       ctypes.byref(numpy_to_memref_1d(c)))

  np.testing.assert_array_equal(c, a + b)
  print("linalg.add successful!")
  print(c)


if __name__ == "__main__":
  main()
