"""Shared ctypes <-> MLIR MemRef bridge for the *2-D* kernels in 3_parallel.

Chapter 2 used a 1-D descriptor (`../../2_memory/common/np_memref.py`); the affine
examples here (matmul, conv2d) all pass 2-D `memref<?x?xf32>` buffers, which need a
descriptor with `shape[2]` / `stride[2]`. Steps import this module:

    import os, sys
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "common"))
    from np_memref2d import MemRef2DDescriptor, numpy_to_memref2d

The kernels are compiled with `llvm.emit_c_interface`, so Python calls the
`_mlir_ciface_<name>` wrapper, which takes one pointer per memref descriptor.
"""

import numpy as np
from ctypes import c_void_p, c_longlong, Structure


class MemRef2DDescriptor(Structure):
  """ctypes layout matching MLIR's 2-D memref descriptor.

  Must match the struct `-finalize-memref-to-llvm` produces for a
  memref<?x?xf32>:  { ptr, ptr, i64, [2 x i64], [2 x i64] }.
  `offset` and the strides are in ELEMENTS, not bytes.
  """
  _fields_ = [
    ("allocated", c_void_p),       # base pointer (the one you'd free())
    ("aligned",   c_void_p),       # aligned data pointer (used for access)
    ("offset",    c_longlong),     # offset to element [0, 0], in elements
    ("shape",     c_longlong * 2), # (rows, cols)
    ("stride",    c_longlong * 2), # (row_stride, col_stride), in elements
  ]


def numpy_to_memref2d(arr):
  """Wrap a 2-D contiguous NumPy float32 array in a descriptor (no copy).

  The descriptor points straight at NumPy's buffer, so MLIR and NumPy share the
  same memory. NumPy reports strides in *bytes*, so we divide by `itemsize` to get
  MLIR's *element* strides.
  """
  if not arr.flags["C_CONTIGUOUS"]:
    arr = np.ascontiguousarray(arr)
  desc = MemRef2DDescriptor()
  desc.allocated = arr.ctypes.data_as(c_void_p)
  desc.aligned   = desc.allocated
  desc.offset    = 0
  desc.shape[0],  desc.shape[1]  = arr.shape
  desc.stride[0] = arr.strides[0] // arr.itemsize   # bytes -> elements
  desc.stride[1] = arr.strides[1] // arr.itemsize
  return desc
