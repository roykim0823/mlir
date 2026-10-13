import ctypes

# Load the shared library (the .so we just built) into the process.
module = ctypes.CDLL("./build/libsimple.so")

# Declare the C signature of `main` so ctypes marshals values correctly:
module.main.argtypes = []            # main takes no arguments
module.main.restype = ctypes.c_int   # main returns a C int (i32)

# Call the native function and print its return value.
print(module.main())
