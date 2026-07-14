"""A decoder-only (GPT-2 style) transformer forward pass, in ~60 lines of NumPy.

Faithful to Part 7 of the reference series: token + position embeddings, then a
stack of transformer blocks (pre-LN: layer-norm -> multi-head attention ->
residual, then layer-norm -> feed-forward -> residual), then a final layer-norm
and projection back to vocabulary logits.

Every function is written the way a deep-learning compiler "sees" the model: a
graph of matmuls, elementwise ops, reductions, and softmaxes — exactly the
Chapter 4 `linalg` primitives. (`mlir_attention/` compiles the softmax piece.)
"""

from dataclasses import dataclass

import numpy as np


# ---------- elementwise / normalization building blocks ----------------------

def gelu(x):
  """Gaussian Error Linear Unit (tanh approximation) — the FFN non-linearity."""
  return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def softmax(x):
  """Row-wise softmax (numerically stable). Turns scores into probabilities."""
  ex = np.exp(x - np.max(x, axis=-1, keepdims=True))
  return ex / np.sum(ex, axis=-1, keepdims=True)


def layer_norm(x, g, b, eps=1e-5):
  """Normalize across the feature dim, then scale (g) and shift (b)."""
  mean = np.mean(x, axis=-1, keepdims=True)
  var = np.var(x, axis=-1, keepdims=True)
  return g * (x - mean) / np.sqrt(var + eps) + b


def linear(x, w, b):
  """A dense/fully-connected layer:  x @ w + b."""
  return x @ w + b


# ---------- the two sub-layers ------------------------------------------------

def ffn(x, c_fc, c_proj):
  """Position-wise feed-forward: expand -> GELU -> project back."""
  return linear(gelu(linear(x, *c_fc)), *c_proj)


def attention(q, k, v, mask):
  """Scaled dot-product attention: softmax(QKᵀ/√d + mask) · V."""
  scores = (q @ k.T) / np.sqrt(q.shape[-1]) + mask
  return softmax(scores) @ v


def mha(x, c_attn, c_proj, n_head):
  """Multi-head attention: project to Q,K,V, split into heads, attend, merge."""
  x = linear(x, *c_attn)                                   # [N, 3*d]
  qkv = np.split(x, 3, axis=-1)                            # 3 x [N, d]
  qkv_heads = [np.split(part, n_head, axis=-1) for part in qkv]  # 3 x n_head x [N, d/h]
  n = x.shape[0]
  causal = (1 - np.tri(n, dtype=x.dtype)) * -1e10          # hide future tokens
  heads = [attention(q, k, v, causal) for q, k, v in zip(*qkv_heads)]
  x = np.hstack(heads)                                     # [N, d]
  return linear(x, *c_proj)


# ---------- a block, and the whole model -------------------------------------

@dataclass
class Block:
  ln1_g: np.ndarray; ln1_b: np.ndarray
  attn_c_attn: tuple; attn_c_proj: tuple
  ln2_g: np.ndarray; ln2_b: np.ndarray
  mlp_c_fc: tuple; mlp_c_proj: tuple


def transformer_block(x, blk: Block, n_head):
  # pre-LN attention sub-layer with a residual connection
  x = x + mha(layer_norm(x, blk.ln1_g, blk.ln1_b),
              blk.attn_c_attn, blk.attn_c_proj, n_head)
  # pre-LN feed-forward sub-layer with a residual connection
  x = x + ffn(layer_norm(x, blk.ln2_g, blk.ln2_b),
              blk.mlp_c_fc, blk.mlp_c_proj)
  return x


@dataclass
class Model:
  wte: np.ndarray            # token embeddings   [vocab, d]
  wpe: np.ndarray            # position embeddings [ctx, d]
  blocks: list               # list[Block]
  lnf_g: np.ndarray; lnf_b: np.ndarray


def gpt2(inputs, model: Model, n_head):
  """Forward pass: token IDs -> logits over the vocabulary for each position."""
  x = model.wte[inputs] + model.wpe[range(len(inputs))]   # embed tokens + positions
  for blk in model.blocks:                                # the transformer stack
    x = transformer_block(x, blk, n_head)
  x = layer_norm(x, model.lnf_g, model.lnf_b)             # final layer norm
  return x @ model.wte.T                                  # project back to vocab logits
