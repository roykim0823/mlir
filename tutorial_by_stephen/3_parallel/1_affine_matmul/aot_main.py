"""Driver for matmul.mlir — calls the compiled affine kernel and checks it
against NumPy's own matmul.

The kernel computes C += A·B, so we pass a zero-initialized C. All three arrays
are 2-D, so we use the shared 2-D descriptor from ../common/np_memref2d.py.
"""

import os
import sys

import ctypes
import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from np_memref2d import MemRef2DDescriptor, numpy_to_memref2d   # noqa: E402


def main():
  lib = ctypes.CDLL("./build/libmatmul.dylib")
  matmul = lib._mlir_ciface_matmul
  # _mlir_ciface_matmul(MemRef2D *A, MemRef2D *B, MemRef2D *C)
  matmul.argtypes = [ctypes.POINTER(MemRef2DDescriptor)] * 3
  matmul.restype = None

  M, K, N = 4, 5, 3
  A = np.random.rand(M, K).astype(np.float32)
  B = np.random.rand(K, N).astype(np.float32)
  C = np.zeros((M, N), dtype=np.float32)   # kernel accumulates into C, so zero it

  matmul(ctypes.byref(numpy_to_memref2d(A)),
         ctypes.byref(numpy_to_memref2d(B)),
         ctypes.byref(numpy_to_memref2d(C)))

  expected = A @ B
  np.testing.assert_allclose(C, expected, rtol=1e-5, atol=1e-5)
  print("Affine matmul successful! (max abs error "
        f"{np.abs(C - expected).max():.2e})")
  print(C)


if __name__ == "__main__":
  main()
