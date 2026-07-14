"""Driver for dense_relu.mlir — runs one neural-network dense layer
(out = relu(X @ W + b)) and checks it against a NumPy reference."""

import os
import sys

import ctypes
import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from np_memref import (MemRef1D, MemRef2D,                    # noqa: E402
                       numpy_to_memref_1d, numpy_to_memref_2d)


def main():
  lib = ctypes.CDLL("./build/libdense.dylib")
  dense = lib._mlir_ciface_dense_relu
  dense.argtypes = [ctypes.POINTER(MemRef2D),   # X
                    ctypes.POINTER(MemRef2D),   # W
                    ctypes.POINTER(MemRef1D),   # b
                    ctypes.POINTER(MemRef2D)]   # out
  dense.restype = None

  N, K, M = 4, 3, 5                              # batch=4, in=3, out=5
  X = np.random.randn(N, K).astype(np.float32)
  W = np.random.randn(K, M).astype(np.float32)
  b = np.random.randn(M).astype(np.float32)
  out = np.zeros((N, M), dtype=np.float32)       # matmul accumulates -> zero it

  dense(ctypes.byref(numpy_to_memref_2d(X)),
        ctypes.byref(numpy_to_memref_2d(W)),
        ctypes.byref(numpy_to_memref_1d(b)),
        ctypes.byref(numpy_to_memref_2d(out)))

  expected = np.maximum(X @ W + b, 0.0)          # relu(X@W + b)
  np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)
  print("MLIR dense layer  out = relu(X @ W + b)  successful! "
        f"(max abs error {np.abs(out - expected).max():.2e})")
  print(out)


if __name__ == "__main__":
  main()
