"""Emit an MLIR `func.func` from an expression term.

This is the last stage of the toy compiler: once the e-graph has handed us the
cheapest equivalent expression, we lower it to MLIR `arith` ops. The function
takes each free symbol as an `f32` argument and returns the computed `f32` — the
"core language -> MLIR" step from the Chapter 1 pipeline diagram.
"""

_BINOP = {"+": "arith.addf", "-": "arith.subf", "*": "arith.mulf", "/": "arith.divf"}


def free_symbols(term, acc=None):
  """Symbol names in left-to-right order of first appearance."""
  acc = [] if acc is None else acc
  if term[0] == "sym":
    if term[1] not in acc:
      acc.append(term[1])
  elif term[0] != "lit":
    for child in term[1:]:
      free_symbols(child, acc)
  return acc


def to_mlir(term, fname="f"):
  """Return MLIR text for a function computing `term` over its free symbols."""
  syms = free_symbols(term)
  lines, counter = [], [0]
  ssa = {}

  def fresh():
    counter[0] += 1
    return f"%v{counter[0]}"

  def emit(t):
    if t[0] == "sym":
      return "%" + t[1]
    if t[0] == "lit":
      r = fresh()
      lines.append(f"  {r} = arith.constant {float(t[1]):.6e} : f32")
      return r
    op = _BINOP[t[0]]
    a, b = emit(t[1]), emit(t[2])
    r = fresh()
    lines.append(f"  {r} = {op} {a}, {b} : f32")
    return r

  result = emit(term)
  args = ", ".join(f"%{s}: f32" for s in syms)
  body = "\n".join(lines)
  return (f"func.func @{fname}({args}) -> f32 {{\n"
          f"{body}\n"
          f"  return {result} : f32\n"
          f"}}\n"), syms
