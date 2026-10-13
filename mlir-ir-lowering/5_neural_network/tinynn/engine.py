"""A tiny reverse-mode automatic-differentiation engine (micrograd-style).

Each `Value` wraps a scalar and records the op that produced it, building a
computation graph. Calling `.backward()` on the final node walks that graph in
reverse-topological order, applying the chain rule to fill in every `.grad`.

This is the conceptual core of every deep-learning framework (PyTorch, JAX, …):
the *what* of a computation is recorded as a graph so gradients can flow back
through it. Later MLIR chapters compile these same patterns; here we build the
idea from scratch so the optimizations make sense.
"""


class Value:
  """A scalar value and its gradient, plus the graph edge that created it."""

  def __init__(self, data, _children=(), _op="", label=""):
    self.data = float(data)
    self.grad = 0.0
    self.label = label
    self._backward = lambda: None      # how to push grad to inputs (set per op)
    self._prev = set(_children)        # the Values this one was computed from
    self._op = _op

  # --- forward ops, each wiring up its own local backward rule ----------------

  def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data, (self, other), "+")

    def _backward():
      # d(a+b)/da = d(a+b)/db = 1, so grad flows through unchanged
      self.grad += out.grad
      other.grad += out.grad
    out._backward = _backward
    return out

  def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other), "*")

    def _backward():
      # product rule: each input's grad is scaled by the other's value
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad
    out._backward = _backward
    return out

  def __pow__(self, p):
    assert isinstance(p, (int, float)), "only constant powers supported"
    out = Value(self.data ** p, (self,), f"**{p}")

    def _backward():
      self.grad += (p * self.data ** (p - 1)) * out.grad
    out._backward = _backward
    return out

  def relu(self):
    out = Value(0.0 if self.data < 0 else self.data, (self,), "relu")

    def _backward():
      # gradient passes through only where the input was positive
      self.grad += (out.data > 0) * out.grad
    out._backward = _backward
    return out

  # --- convenience operators, all defined in terms of the three above ---------

  def __neg__(self):       return self * -1
  def __sub__(self, o):    return self + (-o)
  def __radd__(self, o):   return self + o
  def __rsub__(self, o):   return (-self) + o
  def __rmul__(self, o):   return self * o
  def __truediv__(self, o): return self * (o ** -1 if isinstance(o, Value) else 1.0 / o)

  # --- the backward pass ------------------------------------------------------

  def backward(self):
    """Fill in .grad for every node feeding into this one."""
    topo, visited = [], set()

    def build(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          build(child)
        topo.append(v)
    build(self)

    self.grad = 1.0                    # seed: d(output)/d(output) = 1
    for v in reversed(topo):           # reverse-topological = outputs first
      v._backward()

  def __repr__(self):
    return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"
