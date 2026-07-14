"""Equality-saturation demos — the concepts from Part 6, runnable with no deps.

Each demo builds an expression, saturates a small rule set, and extracts the
cheapest equivalent form according to a cost model.
"""

from egraph import EGraph, Var


def v(name):
  return Var(name)


def fmt(t):
  """Pretty-print a term tuple."""
  if t[0] == "lit":
    return str(t[1])
  if t[0] == "sym":
    return t[1]
  if len(t) == 2:                       # unary, e.g. exp(x), T(A)
    return f"{t[0]}({fmt(t[1])})"
  return f"({fmt(t[1])} {t[0]} {fmt(t[2])})"


def demo_cancellation():
  """The classic phase-ordering example: (a*2)/2 should simplify to a.

  Apply `x*2 -> x<<1` first and you get the stuck term `(a<<1)/2`. The e-graph
  doesn't have to choose: it keeps both `a*2` and `a<<1`, AND learns `(x*y)/y=x`,
  so extraction finds plain `a`."""
  eg = EGraph()
  root = eg.add(("/", ("*", ("sym", "a"), ("lit", 2)), ("lit", 2)))
  rules = [
    (("/", ("*", v("x"), v("y")), v("y")), v("x")),          # (x*y)/y -> x
    (("*", v("x"), ("lit", 2)), ("shl", v("x"), ("lit", 1))),  # x*2 -> x<<1
  ]
  eg.saturate(rules)
  costs = {"lit": 0, "sym": 0, "shl": 1, "*": 1, "/": 5}
  print(f"  (a*2)/2           ->  {fmt(eg.extract(root, costs))}")


def demo_cost_extraction():
  """Cost model picks fewer expensive ops: e^x*e^x*e^x -> e^(x+x+x).

  `exp` is costly; collapsing three of them into one (via exp(p)*exp(q)=exp(p+q))
  is the cheaper equivalent, even though it adds cheap `+`s."""
  eg = EGraph()
  ex = ("exp", ("sym", "x"))
  root = eg.add(("*", ("*", ex, ex), ex))
  rules = [(("*", ("exp", v("p")), ("exp", v("q"))),
           ("exp", ("+", v("p"), v("q"))))]                  # e^p * e^q -> e^(p+q)
  eg.saturate(rules)
  costs = {"lit": 0, "sym": 0, "+": 1, "*": 1, "exp": 40}
  print(f"  e^x * e^x * e^x   ->  {fmt(eg.extract(root, costs))}")


def demo_linear_algebra():
  """Tensor identities: (A^T)^T -> A (a free win, removes two transposes)."""
  eg = EGraph()
  root = eg.add(("T", ("T", ("sym", "A"))))
  rules = [(("T", ("T", v("m"))), v("m"))]                   # (m^T)^T -> m
  eg.saturate(rules)
  costs = {"sym": 0, "T": 10, "*": 1}
  print(f"  (A^T)^T           ->  {fmt(eg.extract(root, costs))}")


def main():
  print("Equality saturation — extracting the cheapest equivalent form:")
  demo_cancellation()
  demo_cost_extraction()
  demo_linear_algebra()
  print("All e-graph demos succeeded.")


if __name__ == "__main__":
  main()
