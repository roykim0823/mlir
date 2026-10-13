"""Driver for conv2d.mlir — runs the affine 2-D convolution and checks it against
a plain NumPy reference (cross-correlation).

The output shape is (in_h - k_h + 1, in_w - k_w + 1); the caller allocates it
zero-filled. All three arrays are 2-D, so we use ../common/np_memref2d.py.
"""

import os
import sys

import ctypes
import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from np_memref2d import MemRef2DDescriptor, numpy_to_memref2d   # noqa: E402


def reference_conv2d(inp, flt):
  """NumPy cross-correlation, matching the MLIR kernel exactly."""
  kh, kw = flt.shape
  oh, ow = inp.shape[0] - kh + 1, inp.shape[1] - kw + 1
  out = np.zeros((oh, ow), dtype=np.float32)
  for i in range(oh):
    for j in range(ow):
      out[i, j] = (inp[i:i + kh, j:j + kw] * flt).sum()
  return out


def main():
  lib = ctypes.CDLL("./build/libconv2d.dylib")
  conv = lib._mlir_ciface_conv_2d
  # _mlir_ciface_conv_2d(MemRef2D *input, MemRef2D *filter, MemRef2D *output)
  conv.argtypes = [ctypes.POINTER(MemRef2DDescriptor)] * 3
  conv.restype = None

  inp = np.random.rand(10, 10).astype(np.float32)
  flt = np.arange(9, dtype=np.float32).reshape(3, 3)   # a simple 3x3 kernel
  oh, ow = inp.shape[0] - flt.shape[0] + 1, inp.shape[1] - flt.shape[1] + 1
  out = np.zeros((oh, ow), dtype=np.float32)

  conv(ctypes.byref(numpy_to_memref2d(inp)),
       ctypes.byref(numpy_to_memref2d(flt)),
       ctypes.byref(numpy_to_memref2d(out)))

  expected = reference_conv2d(inp, flt)
  np.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-4)
  print("Affine conv2d successful! "
        f"(output {oh}x{ow}, max abs error {np.abs(out - expected).max():.2e})")
  print(out)


if __name__ == "__main__":
  main()
