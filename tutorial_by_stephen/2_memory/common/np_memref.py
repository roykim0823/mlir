"""Shared ctypes <-> MLIR MemRef bridge for the 2_memory examples.

Steps 2-4 (array_add, low-level LLVM, C-compatible wrappers) all pass 1-D
`memref<Nxf32>` buffers to MLIR. They import this single module instead of
keeping a copy each:

    import os, sys
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "common"))
    from np_memref import MemRefDescriptor, numpy_to_memref

(Step 1 / 1_tensor uses a *2-D* descriptor with a different calling
convention, so it keeps its own descriptor inline.)
"""

import numpy as np
from ctypes import c_void_p, c_longlong, Structure


class MemRefDescriptor(Structure):
  """ctypes layout matching MLIR's 1-D memref descriptor.

  Must match the struct that `-finalize-memref-to-llvm` produces for a
  memref<Nxf32>:  { ptr, ptr, i64, [1 x i64], [1 x i64] }.
  """
  _fields_ = [
    ("allocated", c_void_p),        # base pointer (the one you'd free())
    ("aligned",   c_void_p),        # aligned data pointer (often same as allocated)
    ("offset",    c_longlong),      # offset into data, in ELEMENTS (not bytes)
    ("shape",     c_longlong * 1),
    ("stride",    c_longlong * 1),
  ]


def numpy_to_memref(arr):
  """Wrap a 1-D contiguous NumPy array in a MemRefDescriptor (no copy).

  The descriptor points straight at the NumPy buffer, so MLIR and NumPy
  share the same memory. `allocated` and `aligned` are set to the same
  pointer because NumPy gives us a single buffer with no separate
  free-handle (and nothing here ever free()s it).
  """
  if not arr.flags["C_CONTIGUOUS"]:
    arr = np.ascontiguousarray(arr)

  desc = MemRefDescriptor()
  desc.allocated = arr.ctypes.data_as(c_void_p)
  desc.aligned   = desc.allocated
  desc.offset    = 0
  desc.shape[0]  = arr.shape[0]
  desc.stride[0] = 1            # contiguous 1-D: stride is 1 element
  return desc
