"""ctypes <-> MLIR MemRef bridge (2-D float32) for the MLIR attention kernel."""

import numpy as np
from ctypes import c_void_p, c_longlong, Structure


class MemRef2D(Structure):
  _fields_ = [
    ("allocated", c_void_p),
    ("aligned",   c_void_p),
    ("offset",    c_longlong),
    ("shape",     c_longlong * 2),
    ("stride",    c_longlong * 2),
  ]


def numpy_to_memref_2d(arr):
  if not arr.flags["C_CONTIGUOUS"]:
    arr = np.ascontiguousarray(arr)
  d = MemRef2D()
  d.allocated = arr.ctypes.data_as(c_void_p)
  d.aligned   = d.allocated
  d.offset    = 0
  d.shape[0],  d.shape[1]  = arr.shape
  d.stride[0] = arr.strides[0] // arr.itemsize
  d.stride[1] = arr.strides[1] // arr.itemsize
  return d
