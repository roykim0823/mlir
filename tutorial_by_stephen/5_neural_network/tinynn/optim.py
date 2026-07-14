"""Stochastic gradient descent — the simplest optimizer.

Update rule:  p <- p - lr * p.grad   (optionally with L2 weight decay).
"""


class SGD:
  def __init__(self, parameters, lr=0.01):
    self.parameters = list(parameters)
    self.lr = lr

  def zero_grad(self):
    for p in self.parameters:
      p.grad = 0.0

  def step(self, lambda_reg=0.0):
    """Take one gradient-descent step. lambda_reg>0 adds L2 weight decay,
    which shrinks weights toward zero to discourage overfitting."""
    for p in self.parameters:
      p.data -= self.lr * (p.grad + 2.0 * lambda_reg * p.data)
