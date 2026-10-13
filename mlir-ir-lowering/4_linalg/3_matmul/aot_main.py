"""Driver for matmul.mlir — runs the linalg.matmul kernel and checks it against
NumPy's @. The kernel accumulates into C, so we pass a zeroed C."""

import os
import sys

import ctypes
import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from np_memref import MemRef2D, numpy_to_memref_2d   # noqa: E402


def main():
  lib = ctypes.CDLL("./build/libmatmul.dylib")
  matmul = lib._mlir_ciface_matmul
  matmul.argtypes = [ctypes.POINTER(MemRef2D)] * 3
  matmul.restype = None

  A = np.random.rand(8, 10).astype(np.float32)
  B = np.random.rand(10, 16).astype(np.float32)
  C = np.zeros((8, 16), dtype=np.float32)

  matmul(ctypes.byref(numpy_to_memref_2d(A)),
         ctypes.byref(numpy_to_memref_2d(B)),
         ctypes.byref(numpy_to_memref_2d(C)))

  expected = A @ B
  np.testing.assert_allclose(C, expected, rtol=1e-5, atol=1e-5)
  print("linalg.matmul successful! "
        f"(8x10 @ 10x16, max abs error {np.abs(C - expected).max():.2e})")


if __name__ == "__main__":
  main()
