# 7 — Transformers

### The architecture the whole series is aiming at

This is the model the series has been building toward: a **decoder-only
transformer** (GPT-2). And here's the encouraging part — by now you've already met
every piece. A transformer is not one exotic operation; it's a *graph of the
primitives from earlier chapters*: matrix multiplies (Ch 4), elementwise
activations and bias adds (Ch 4/5), reductions and softmaxes, and residual adds —
stacked a dozen times. The "magic" is purely in how they're wired.

The one genuinely new idea is **self-attention**: when processing a token, the
model dynamically weights how much to "attend" to every other token. Processing
*"bank"* in *"deposit money at the bank"*, attention lets it lean on *"deposit"*
and *"money"* to read "bank" as a financial institution. Mechanically that's three
projections (Query, Key, Value), a scaled dot-product `QKᵀ/√d`, a **softmax** to
turn scores into weights, and a weighted sum of Values — all matmuls and a
softmax.

> Based on Stephen Diehl's *"MLIR Part 7 — Transformers"*
> ([`../reference/`](../reference/)). That part is a ~100-line **NumPy** GPT-2 (no
> MLIR); this chapter reproduces it as a runnable model **and** compiles the
> softmax — attention's defining op — to MLIR. Verified on LLVM 20.1.8 + NumPy.

---

## Part A — GPT-2 in NumPy (`gpt2/model.py`)

The full forward pass, every function mapping to ops you already know:

| Component | What it is | Built from |
| --- | --- | --- |
| token + position embeddings | look up `wte[token] + wpe[pos]` | gather + add |
| `layer_norm` | normalize across features, scale + shift | mean/var reductions |
| `linear` | `x @ w + b` | **matmul + bias** (Ch 4/5) |
| `gelu` | smooth activation | elementwise |
| `ffn` | `linear → gelu → linear` | two dense layers (Ch 5) |
| `softmax` | scores → probabilities | exp + row-sum reduction |
| `attention` | `softmax(QKᵀ/√d + mask) · V` | **matmul + softmax + matmul** |
| `mha` | run attention in parallel heads, concat, project | the above × heads |
| `transformer_block` | pre-LN → MHA → residual; pre-LN → FFN → residual | all of it |
| `gpt2` | embed → N blocks → final LN → project to vocab | the stack |

Notice how directly the code reads as that graph — `attention` is literally its
one-line formula, every operation something from Chapters 4–5:

*gpt2/model.py* (excerpt — attention and its softmax)
```python
def softmax(x):
  """Row-wise softmax (numerically stable). Turns scores into probabilities."""
  ex = np.exp(x - np.max(x, axis=-1, keepdims=True))
  return ex / np.sum(ex, axis=-1, keepdims=True)

# ... layer_norm, linear, ffn ...

def attention(q, k, v, mask):
  """Scaled dot-product attention: softmax(QKᵀ/√d + mask) · V."""
  scores = (q @ k.T) / np.sqrt(q.shape[-1]) + mask
  return softmax(scores) @ v
```

```text
   Q ─┐                                                  weighted
      ├─► Q·Kᵀ ─► ÷√d ─► + mask ─► softmax ─► weights ──►  sum   ─► out
   K ─┘   scores     scale    hide      row-wise         · V
   V ────────────────────────future    probabilities
        (matmul)                       (Part B compiles this)   (matmul)
```

Two transformer-specific details worth calling out:

- **Causal masking.** A language model must not peek at future tokens, so before
  the softmax we add a mask that sets "future" scores to `-1e10` — after softmax
  those weights become ~0. (`mha` builds it with `np.tri`.)
- **Multiple heads.** The `d=768` embedding is split into 12 heads of 64, each
  attending independently ("from different perspectives"), then concatenated back
  — `768 / 12 = 64`.

`demo.py` runs the whole thing on a **tiny random-weight** config (real GPT-2 is
hundreds of MB; we use `d=12, heads=3, layers=2, vocab=32`). With random weights
the prediction is meaningless, but the plumbing — embeddings, the block stack,
attention, FFN, residuals, layer-norm, and the vocab projection — is exactly the
real architecture.

**Run:** `python3 demo.py`

```
GPT-2-style forward pass: 5 tokens, d=12, heads=3, layers=2
logits shape: (5, 32)  (sequence_length x vocab)
next-token distribution sums to 1.0000; argmax -> token 9
Transformer forward pass successful!
```

---

## Part B — softmax, compiled (`mlir_attention/`) · ✅

Attention's defining step is the **softmax** over scores. It's also the most
interesting transformer op to compile, because it's not pure elementwise — it
needs two **reductions** per row (a max for stability, then a sum) with passes
between them:

```
out[i,j] = exp(x[i,j] - max_j x[i,:]) / Σ_j exp(x[i,:] - max_j x[i,:])
```

`softmax.mlir` implements it with three sequential `scf.for` passes per row — a
`maximumf` accumulator (the row max, subtracted for numerical stability), then an
`addf` accumulator over `math.exp`, then a divide — the same stable formula as the
NumPy `softmax` above, lowered to native code:

*mlir_attention/softmax.mlir*
```mlir
func.func @softmax(%x: memref<?x?xf32>, %out: memref<?x?xf32>)
    attributes {llvm.emit_c_interface} {
  %c0   = arith.constant 0 : index
  %c1   = arith.constant 1 : index
  %N    = memref.dim %x, %c0 : memref<?x?xf32>
  %M    = memref.dim %x, %c1 : memref<?x?xf32>
  %ninf = arith.constant -3.40282347E+38 : f32   // -FLT_MAX, the max identity
  %zero = arith.constant 0.0 : f32

  scf.for %i = %c0 to %N step %c1 {
    // 1. row maximum
    %mx = scf.for %j = %c0 to %M step %c1 iter_args(%m = %ninf) -> (f32) {
      %v  = memref.load %x[%i, %j] : memref<?x?xf32>
      %mm = arith.maximumf %m, %v : f32
      scf.yield %mm : f32
    }
    // 2. exp(x - max), stored into out, accumulating the row sum
    %sum = scf.for %j = %c0 to %M step %c1 iter_args(%s = %zero) -> (f32) {
      %v = memref.load %x[%i, %j] : memref<?x?xf32>
      %d = arith.subf %v, %mx : f32
      %e = math.exp %d : f32
      memref.store %e, %out[%i, %j] : memref<?x?xf32>
      %ss = arith.addf %s, %e : f32
      scf.yield %ss : f32
    }
    // 3. normalize the row by its sum
    scf.for %j = %c0 to %M step %c1 {
      %e = memref.load %out[%i, %j] : memref<?x?xf32>
      %r = arith.divf %e, %sum : f32
      memref.store %r, %out[%i, %j] : memref<?x?xf32>
    }
  }
  return
}
```

The two `iter_args` accumulators are Chapter 3's loop-carried values again (the
`maximumf`/`addf` reductions), and `math.exp` lowers to a libm call. The driver
checks it against the model's own `softmax` and confirms every row sums to 1.

**Run:** `cd mlir_attention && bash build.sh`

```
MLIR softmax successful! (max abs error 0.00e+00; every row sums to 1)
```

Put it together with Chapter 4's `linalg.matmul`, and you have all of
`attention = softmax(Q @ Kᵀ / √d) @ V` in MLIR — which is what a real ML compiler
fuses and ships to a GPU. That's Chapter 8.

---

## Layout

```
7_transformer/
├── gpt2/model.py        # the NumPy GPT-2 forward pass (the PDF's content)
├── demo.py              # run it on a tiny random-weight config
├── mlir_attention/      # softmax compiled to MLIR
│   ├── softmax.mlir
│   ├── build.sh
│   └── aot_main.py
└── common/np_memref.py
```

## Run everything

```bash
export PATH="/opt/homebrew/opt/llvm@20/bin:$PATH"
# pip install numpy
python3 demo.py                          # Part A — the model
( cd mlir_attention && bash build.sh )   # Part B — compiled softmax
```

## Key takeaways

- **A transformer is a graph of familiar ops** — matmul, layer-norm, GELU,
  softmax, residual adds — stacked. Nothing here is outside Chapters 4–5 plus
  softmax.
- **Self-attention** = `softmax(QKᵀ/√d + mask) · V`: project to Q/K/V, score,
  mask the future, softmax, weight the Values; run it in parallel **heads**.
- **Softmax is the interesting op to compile** because of its per-row reductions;
  `math.exp` + `scf.for` reductions (with `iter_args`, Ch 3) reproduce the stable
  formula exactly.
- **The model is a compiler workload.** Frameworks lower exactly this graph to
  `linalg`/`tensor` and optimize it — and the next chapter moves it to the GPU.

**Next:** Part 8 — GPU compilation with MLIR (see [`../reference/`](../reference/)).
```

