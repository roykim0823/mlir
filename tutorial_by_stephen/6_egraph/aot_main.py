"""Load the compiled original and optimized capstone functions and confirm they
compute the same values — i.e. the e-graph rewrite preserved meaning."""

import ctypes


def load(name):
  lib = ctypes.CDLL(f"./build/lib{name}.dylib")
  fn = lib.f                      # scalar f32 -> f32; no descriptor needed
  fn.argtypes = [ctypes.c_float]
  fn.restype = ctypes.c_float
  return fn


def main():
  orig = load("orig")
  opt = load("opt")
  for x in (-3.0, 0.0, 2.5, 7.0, 100.0):
    a, b = orig(x), opt(x)
    assert abs(a - b) < 1e-4, f"mismatch at {x}: {a} vs {b}"
    assert abs(b - x * x) < 1e-4, f"optimized wrong at {x}: {b}"
  print("Capstone OK: optimized f(a) == original f(a) == a*a for all test inputs.")


if __name__ == "__main__":
  main()
