"""Train the tinynn MLP on a synthetic "two moons" dataset — the classic test of
a non-linear classifier. Pure Python + NumPy (no sklearn), so it runs anywhere.

This exercises the whole stack from scratch: forward pass through the MLP, MSE
loss, `loss.backward()` to fill gradients via the autodiff engine, then an SGD
step. It's deliberately small because scalar-`Value` autodiff is slow — that
slowness is exactly the motivation for compiling these computations (Chapters
6-8).
"""

import random

import numpy as np

from tinynn import MLP, SGD


def make_moons(n=100, noise=0.1, seed=0):
  """Two interleaving half-circles — needs a non-linear boundary to separate."""
  rng = np.random.default_rng(seed)
  n_out = n // 2
  n_in = n - n_out
  t_out = np.linspace(0, np.pi, n_out)
  t_in = np.linspace(0, np.pi, n_in)
  outer = np.c_[np.cos(t_out), np.sin(t_out)]
  inner = np.c_[1 - np.cos(t_in), 1 - np.sin(t_in) - 0.5]
  X = np.vstack([outer, inner]).astype(np.float32)
  y = np.array([0] * n_out + [1] * n_in)
  X += noise * rng.standard_normal(X.shape).astype(np.float32)
  return X, y


def main():
  random.seed(0)
  X, y = make_moons(n=100, noise=0.1)

  model = MLP(nin=2, nouts=[8, 8, 1])          # 2 -> 8 -> 8 -> 1
  opt = SGD(model.parameters(), lr=0.05)
  print(f"Training MLP(2, [8, 8, 1]) — {len(model.parameters())} parameters")

  for epoch in range(40):
    idx = list(range(len(X)))
    random.shuffle(idx)
    total_loss, correct = 0.0, 0
    for i in idx:
      target = 1.0 if y[i] == 1 else -1.0       # tanh-style ±1 targets
      pred = model([float(X[i][0]), float(X[i][1])])
      loss = (pred - target) * (pred - target)  # MSE on one example
      opt.zero_grad()
      loss.backward()
      opt.step()
      total_loss += loss.data
      correct += (pred.data > 0) == (y[i] == 1)
    if (epoch + 1) % 10 == 0:
      print(f"  epoch {epoch + 1:3d}: loss={total_loss / len(X):.4f}  "
            f"acc={correct / len(X):.2f}")

  acc = correct / len(X)
  print(f"Final training accuracy: {acc:.2f}")
  assert acc >= 0.9, "expected the MLP to fit this toy dataset"
  print("Neural network trained successfully!")


if __name__ == "__main__":
  main()
