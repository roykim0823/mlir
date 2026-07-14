# 5 — Neural Networks (from scratch, then compiled)

### A neural network is just a big differentiable function

The last four chapters built machinery: lowering (Ch 1), memory (Ch 2),
parallelism (Ch 3), structured linear algebra (Ch 4). This chapter is the
**application** the whole series is aiming at — deep learning — but before
compiling neural networks it pays to understand, from first principles, what one
*is*.

Strip away the mystique and a neural network is a large function `f(x; θ)` with
many tunable parameters `θ` (weights and biases). "Learning" is just **gradient
descent**: measure how wrong the output is with a loss `L`, compute `∇L` (how each
parameter should change), nudge every parameter a little against its gradient,
repeat. The only non-obvious part is computing `∇L` for millions of parameters
efficiently — and the answer, **automatic differentiation**, is the conceptual
heart of PyTorch, JAX, and TensorFlow alike.

Here's the connection that makes this a compiler chapter: autodiff works by
recording every operation into a **computation graph** — a DAG of ops with data
flowing between them. That is *the same idea as an MLIR module*. The frameworks
build an IR of your computation so they can differentiate it and optimize it,
exactly as MLIR builds an IR so it can lower and optimize it. We'll build that
graph from scratch in pure Python, then compile one layer of it with the `linalg`
dialect from Chapter 4 — the bridge to the GPU chapters ahead.

> Based on Stephen Diehl's *"MLIR Part 5 — Neural Networks"*
> ([`../reference/`](../reference/)). That part is a ground-up **Python**
> implementation (it has no MLIR code); this chapter reproduces it as a small
> runnable library **and adds** an MLIR forward-pass kernel to tie it back to the
> series. Verified on Homebrew LLVM 20.1.8 + NumPy.

---

## Part A — the ideas, in pure Python (`tinynn/`)

### Automatic differentiation (`tinynn/engine.py`)

Every number is wrapped in a `Value` that remembers the operation that produced
it. Building an expression therefore builds a graph; each op also stores a tiny
`_backward` closure encoding its local derivative (the chain rule, one edge at a
time). Multiplication is the clearest case — the product rule scales each input's
gradient by the *other's* value:

*tinynn/engine.py* (one op's forward + local backward)
```python
  def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other), "*")

    def _backward():
      # product rule: each input's grad is scaled by the other's value
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad
    out._backward = _backward
    return out
```

`.backward()` on the final node topologically sorts the graph, seeds the output
gradient to `1`, and runs every `_backward` in reverse — gradients flow from the
loss back to every parameter. That reverse sweep *is* **backpropagation**:

*tinynn/engine.py* (the reverse sweep)
```python
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
```

```text
   forward: build the graph          backward: chain rule, in reverse
   a ─┐                              a.grad += b·out.grad  ◄─┐
      ×──► out ─► L                                          out.grad ◄─ L.grad=1
   b ─┘                              b.grad += a·out.grad  ◄─┘
   each op stores _backward          run every _backward outputs-first
```

### Neurons, layers, the MLP (`tinynn/nn.py`)

The standard hierarchy, every value a `Value` so gradients are automatic:

- **`Neuron`** — `relu(w·x + b)`; the weighted sum plus bias, then an activation.
- **`Layer`** — a row of neurons sharing the same inputs (a fully-connected layer).
- **`MLP`** — a stack of layers; the last is linear, the rest use **ReLU**
  (`max(0, z)`), the non-linearity that lets a deep net represent more than a
  single linear map.

The whole hierarchy is tiny because each level just delegates to the one below:

*tinynn/nn.py* (the neuron — `w·x + b`, then activation)
```python
class Neuron(Module):
  """One neuron: a = activation( sum_i w_i * x_i + b )."""

  def __init__(self, nin, nonlin=True):
    self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
    self.b = Value(0.0)
    self.nonlin = nonlin

  def __call__(self, x):
    act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)   # w·x + b
    return act.relu() if self.nonlin else act
```

In matrix form each layer is `A = σ(W·A_prev + b)` — keep that equation in mind;
Part B compiles exactly it.

### Optimizer & training (`tinynn/optim.py`, `train_demo.py`)

`SGD` applies `p ← p − lr · p.grad` (with optional L2 weight decay
`p ← p − lr·(grad + 2λp)` to discourage overfitting) — the entire optimizer is one
line:

*tinynn/optim.py* (the update rule)
```python
  def step(self, lambda_reg=0.0):
    """Take one gradient-descent step. lambda_reg>0 adds L2 weight decay,
    which shrinks weights toward zero to discourage overfitting."""
    for p in self.parameters:
      p.data -= self.lr * (p.grad + 2.0 * lambda_reg * p.data)
```

`train_demo.py` puts it all together on a synthetic **two-moons** dataset — two
interleaving half-circles that *require* a non-linear boundary — and trains an
`MLP(2, [8, 8, 1])` with an MSE loss:

**Run:** `python3 train_demo.py` (needs only NumPy)

```
Training MLP(2, [8, 8, 1]) — 105 parameters
  epoch  10: loss=0.0766  acc=1.00
  epoch  20: loss=0.0243  acc=1.00
  epoch  30: loss=0.0176  acc=1.00
  epoch  40: loss=0.0096  acc=1.00
Final training accuracy: 1.00
Neural network trained successfully!
```

> **Why it's slow.** Each `Value` is one Python object and each op one Python
> call, so a forward+backward pass is thousands of interpreter operations per
> example. That sluggishness is the entire motivation for the rest of the series:
> the *same* graph, compiled to vectorized, parallel, GPU code, runs orders of
> magnitude faster.

---

## Part B — one layer, compiled (`mlir_layer/`) · ✅

A neural network is mostly the same kernel repeated: a **dense layer**
`out = relu(X @ W + b)`. That decomposes cleanly into pieces you already built in
Chapter 4:

```text
   X (N×K)     W (K×M)            b (M)             out (N×M)
   ┌────┐   ┌──────┐          ┌──────────┐       ┌──────────┐
   │batch  ×  weights   =   X@W  + bias(row) ─►  relu(·)  =  activations
   └────┘   └──────┘          └──────────┘       └──────────┘
   linalg.matmul            ── one fused linalg.generic ──
                            (add b[j] across rows, then max(·,0))
```

| Layer math | Linalg op |
| --- | --- |
| `X @ W` (the `W·A_prev`) | `linalg.matmul` |
| `+ b` broadcast over the batch | a `linalg.generic` reading `b[j]` via `(i,j)->(j)` |
| `relu(·)` | `arith.maximumf z, 0` inside that same generic |

The bias-add and the ReLU are **fused into one `linalg.generic`** — a single pass
over the output instead of three — which is exactly the kernel-fusion idea from
Chapter 4 (Step 4) applied to a real layer. The whole kernel:

*mlir_layer/dense_relu.mlir*
```mlir
func.func @dense_relu(%X: memref<?x?xf32>, %W: memref<?x?xf32>,
                      %b: memref<?xf32>, %out: memref<?x?xf32>)
    attributes {llvm.emit_c_interface} {
  // out = X @ W   (accumulates into the caller's zeroed buffer)
  linalg.matmul ins(%X, %W : memref<?x?xf32>, memref<?x?xf32>)
                outs(%out : memref<?x?xf32>)

  // out[i,j] = relu(out[i,j] + b[j])   — bias broadcast + activation, fused
  linalg.generic {
    indexing_maps  = [affine_map<(i, j) -> (j)>,        // b: depends only on column j
                      affine_map<(i, j) -> (i, j)>],    // out: full 2-D
    iterator_types = ["parallel", "parallel"]
  } ins(%b : memref<?xf32>) outs(%out : memref<?x?xf32>) {
  ^bb0(%bias: f32, %acc: f32):
    %z    = arith.addf %acc, %bias : f32
    %zero = arith.constant 0.0 : f32
    %relu = arith.maximumf %z, %zero : f32              // ReLU = max(z, 0)
    linalg.yield %relu : f32
  }
  return
}
```

The bias's indexing map `(i, j) -> (j)` is the broadcast: for every row `i` the
same `b[j]` is read, stretching the length-`M` bias across all `N` rows exactly
like Chapter 4's `linalg.broadcast` — but here folded straight into the activation
loop. `aot_main.py` feeds in NumPy `X`, `W`, `b` and checks the output against
`np.maximum(X @ W + b, 0)`.

**Run:** `cd mlir_layer && bash build.sh`

```
MLIR dense layer  out = relu(X @ W + b)  successful! (max abs error 2.38e-07)
```

(Inputs are random each run, so the exact error varies; it's always a tiny
float-rounding value. The driver also prints the resulting `4×5` activation
matrix.) This is the unit deep-learning compilers stamp out and optimize;
Chapter 7 wires many such layers into a transformer, and Chapter 8 moves them to
the GPU.

---

## Layout

```
5_neural_network/
├── tinynn/            # the pure-Python library (the PDF's content)
│   ├── engine.py      #   Value: reverse-mode autodiff
│   ├── nn.py          #   Module / Neuron / Layer / MLP
│   └── optim.py       #   SGD (+ L2 weight decay)
├── train_demo.py      # train on synthetic two-moons (NumPy only)
├── mlir_layer/        # the MLIR bridge: a dense layer in linalg
│   ├── dense_relu.mlir
│   ├── build.sh
│   └── aot_main.py
└── common/np_memref.py
```

## Run everything

```bash
export PATH="/opt/homebrew/opt/llvm@20/bin:$PATH"
# pip install numpy

python3 train_demo.py                 # Part A — train the Python MLP
( cd mlir_layer && bash build.sh )    # Part B — run the compiled layer
```

## Key takeaways

- **Training = autodiff + gradient descent.** Record the computation as a graph,
  run the chain rule backward over it (`backward()`), step the parameters. That
  graph is the same kind of IR MLIR manipulates.
- **A net is a stack of dense layers**, each `A = σ(W·A_prev + b)` — matmul, bias,
  activation.
- **That layer is pure Chapter-4 Linalg**: `linalg.matmul` + a fused
  bias/ReLU `linalg.generic`, the broadcast bias riding in via a `(i,j)->(j)`
  indexing map. Frameworks lower whole models to graphs of exactly these ops.
- **The scalar Python version is slow on purpose** — it motivates compiling the
  graph, which is what Chapters 6-8 do.

**Next:** Part 6 — e-graphs and term rewriting (see [`../reference/`](../reference/)).
```

