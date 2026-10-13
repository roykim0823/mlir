"""Shared ctypes <-> MLIR MemRef bridge for the MLIR dense-layer example.

Provides 1-D and 2-D float32 descriptors + NumPy adapters (same shape as
Chapter 4's helper). `offset` and strides are in ELEMENTS, not bytes.
"""

import numpy as np
from ctypes import c_void_p, c_longlong, Structure

_FIELDS = lambda rank: [          # noqa: E731
  ("allocated", c_void_p),
  ("aligned",   c_void_p),
  ("offset",    c_longlong),
  ("shape",     c_longlong * rank),
  ("stride",    c_longlong * rank),
]


class MemRef1D(Structure):
  _fields_ = _FIELDS(1)


class MemRef2D(Structure):
  _fields_ = _FIELDS(2)


def numpy_to_memref_1d(arr):
  if not arr.flags["C_CONTIGUOUS"]:
    arr = np.ascontiguousarray(arr)
  d = MemRef1D()
  d.allocated = arr.ctypes.data_as(c_void_p)
  d.aligned   = d.allocated
  d.offset    = 0
  d.shape[0]  = arr.shape[0]
  d.stride[0] = 1
  return d


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
