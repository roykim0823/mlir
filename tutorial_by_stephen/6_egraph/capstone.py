"""Capstone: optimize a Python-style expression with the e-graph, then emit MLIR
for BOTH the original and the optimized form into build/.

We optimize  f(a) = ((a*2)/2) * ((a*2)/2)  which simplifies to  a * a:
each `(a*2)/2` cancels to `a`, so the whole thing is `a*a` — 1 multiply instead
of the original's 2 multiplies + 2 divides. build.sh then compiles both and
aot_main.py confirms they compute the same values.
"""

import os

from egraph import EGraph, Var
from optimize_demo import fmt
from to_mlir import to_mlir


def v(name):
  return Var(name)


def count_ops(term):
  if term[0] in ("lit", "sym"):
    return 0
  return 1 + sum(count_ops(c) for c in term[1:])


def main():
  here = os.path.dirname(os.path.abspath(__file__))
  build = os.path.join(here, "build")
  os.makedirs(build, exist_ok=True)

  # original:  ((a*2)/2) * ((a*2)/2)
  half = ("/", ("*", ("sym", "a"), ("lit", 2)), ("lit", 2))
  original = ("*", half, half)

  eg = EGraph()
  root = eg.add(original)
  rules = [(("/", ("*", v("x"), v("y")), v("y")), v("x"))]   # (x*y)/y -> x
  eg.saturate(rules)
  costs = {"lit": 0, "sym": 0, "*": 1, "/": 5}
  optimized = eg.extract(root, costs)

  print(f"original :  {fmt(original)}   ({count_ops(original)} ops)")
  print(f"optimized:  {fmt(optimized)}   ({count_ops(optimized)} ops)")

  orig_mlir, _ = to_mlir(original, "f")
  opt_mlir, _ = to_mlir(optimized, "f")
  with open(os.path.join(build, "orig.mlir"), "w") as fh:
    fh.write(orig_mlir)
  with open(os.path.join(build, "opt.mlir"), "w") as fh:
    fh.write(opt_mlir)
  print("Wrote build/orig.mlir and build/opt.mlir")


if __name__ == "__main__":
  main()
