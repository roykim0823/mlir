import ctypes
import os
import sys
import numpy as np

# Shared MemRef <-> NumPy bridge lives in ../common (see common/np_memref.py).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from np_memref import MemRefDescriptor, numpy_to_memref

def main():
  # Load the shared library
  lib = ctypes.CDLL("./build/add_vector_to_matrix_opt.so")

  # The MLIR kernel returns a memref, so the C-iface wrapper takes
  # (result_descriptor_ptr, input_descriptor_ptr) and writes the
  # returned memref struct into *result_descriptor_ptr.
  fn = lib._mlir_ciface_add_vector_to_matrix
  fn.argtypes = [
    ctypes.POINTER(MemRefDescriptor),  # result (out)
    ctypes.POINTER(MemRefDescriptor),  # input
  ]
  fn.restype = None

  # The MLIR function is defined over memref<3xf32>.
  a = np.array([1.0, 2.0, 3.0], dtype=np.float32)

  a_desc      = numpy_to_memref(a)
  result_desc = MemRefDescriptor()  # populated by the call

  fn(ctypes.byref(result_desc), ctypes.byref(a_desc))

  # Read the result back through the returned descriptor.
  size = result_desc.shape[0]
  out_ptr = ctypes.cast(result_desc.aligned, ctypes.POINTER(ctypes.c_float))
  out = np.ctypeslib.as_array(out_ptr, shape=(size,)).copy()

  # The kernel is currently the identity function: it returns its input.
  np.testing.assert_array_almost_equal(out, a)
  print("add_vector_to_matrix call successful!")
  print(f"input : {a}")
  print(f"output: {out}")

if __name__ == "__main__":
  main()
