"""Driver for softmax.mlir — runs the MLIR row-wise softmax and checks it against
the NumPy softmax from the model (gpt2/model.py)."""

import os
import sys

import ctypes
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
sys.path.insert(0, os.path.join(HERE, ".."))
from np_memref import MemRef2D, numpy_to_memref_2d   # noqa: E402
from gpt2 import softmax as np_softmax               # noqa: E402


def main():
  lib = ctypes.CDLL("./build/libsoftmax.dylib")
  fn = lib._mlir_ciface_softmax
  fn.argtypes = [ctypes.POINTER(MemRef2D)] * 2
  fn.restype = None

  # Stand-in for attention scores: an N x N matrix of scaled dot products.
  x = (np.random.randn(6, 6) * 3.0).astype(np.float32)
  out = np.zeros_like(x)

  fn(ctypes.byref(numpy_to_memref_2d(x)),
     ctypes.byref(numpy_to_memref_2d(out)))

  expected = np_softmax(x)                       # the model's own softmax
  np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-6)
  np.testing.assert_allclose(out.sum(axis=1), np.ones(6), atol=1e-5)
  print("MLIR softmax successful! "
        f"(max abs error {np.abs(out - expected).max():.2e}; every row sums to 1)")


if __name__ == "__main__":
  main()
