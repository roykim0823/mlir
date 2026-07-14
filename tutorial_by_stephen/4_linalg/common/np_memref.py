"""Shared ctypes <-> MLIR MemRef bridge for the 4_linalg kernels.

The linalg examples mix 1-D and 2-D float32 buffers, so this module provides both
descriptor ranks and a NumPy adapter for each. Steps import it with:

    import os, sys
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "common"))
    from np_memref import (MemRef1D, MemRef2D,
                           numpy_to_memref_1d, numpy_to_memref_2d)

Each kernel carries `llvm.emit_c_interface`, so Python calls the
`_mlir_ciface_<name>` wrapper, passing one pointer per memref descriptor.
`offset` and the strides are in ELEMENTS, not bytes.
"""

import numpy as np
from ctypes import c_void_p, c_longlong, Structure

_FIELDS = lambda rank: [          # noqa: E731 - tiny helper
  ("allocated", c_void_p),
  ("aligned",   c_void_p),
  ("offset",    c_longlong),
  ("shape",     c_longlong * rank),
  ("stride",    c_longlong * rank),
]


class MemRef1D(Structure):
  """ctypes layout for memref<Nxf32>:  { ptr, ptr, i64, [1 x i64], [1 x i64] }."""
  _fields_ = _FIELDS(1)


class MemRef2D(Structure):
  """ctypes layout for memref<MxNxf32>: { ptr, ptr, i64, [2 x i64], [2 x i64] }."""
  _fields_ = _FIELDS(2)


def numpy_to_memref_1d(arr):
  """Wrap a 1-D contiguous float32 array in a MemRef1D descriptor (no copy)."""
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
  """Wrap a 2-D contiguous float32 array in a MemRef2D descriptor (no copy)."""
  if not arr.flags["C_CONTIGUOUS"]:
    arr = np.ascontiguousarray(arr)
  d = MemRef2D()
  d.allocated = arr.ctypes.data_as(c_void_p)
  d.aligned   = d.allocated
  d.offset    = 0
  d.shape[0],  d.shape[1]  = arr.shape
  d.stride[0] = arr.strides[0] // arr.itemsize   # bytes -> elements
  d.stride[1] = arr.strides[1] // arr.itemsize
  return d
