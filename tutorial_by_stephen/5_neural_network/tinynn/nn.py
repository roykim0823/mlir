"""Neural-network building blocks on top of the autodiff engine.

Neuron -> Layer -> MLP, each made of `Value`s so gradients flow automatically.
This mirrors how PyTorch's `nn.Module` is structured, minus the performance.
"""

import random

from .engine import Value


class Module:
  """Base class: anything with trainable parameters."""

  def zero_grad(self):
    for p in self.parameters():
      p.grad = 0.0

  def parameters(self):
    return []


class Neuron(Module):
  """One neuron: a = activation( sum_i w_i * x_i + b )."""

  def __init__(self, nin, nonlin=True):
    self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
    self.b = Value(0.0)
    self.nonlin = nonlin

  def __call__(self, x):
    act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)   # w·x + b
    return act.relu() if self.nonlin else act

  def parameters(self):
    return self.w + [self.b]


class Layer(Module):
  """A row of neurons, each seeing the same inputs (a fully-connected layer)."""

  def __init__(self, nin, nout, **kwargs):
    self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

  def __call__(self, x):
    out = [n(x) for n in self.neurons]
    return out[0] if len(out) == 1 else out

  def parameters(self):
    return [p for n in self.neurons for p in n.parameters()]


class MLP(Module):
  """A multi-layer perceptron: a stack of Layers (the last one is linear)."""

  def __init__(self, nin, nouts):
    sz = [nin] + nouts
    self.layers = [
      Layer(sz[i], sz[i + 1], nonlin=(i != len(nouts) - 1))
      for i in range(len(nouts))
    ]

  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    return x

  def parameters(self):
    return [p for layer in self.layers for p in layer.parameters()]
