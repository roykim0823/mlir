"""Run a forward pass of the GPT-2-style model on a TINY, random-weight config.

We don't download real GPT-2 weights (hundreds of MB); the point here is to show
the *architecture* runs end-to-end and the shapes/probabilities are correct. With
random weights the predicted token is meaningless — but the plumbing (embeddings,
12-block-style stack, attention, FFN, residuals, layer-norm, vocab projection) is
exactly the real thing, just smaller.
"""

import numpy as np

from gpt2 import gpt2, softmax, Block, Model

# A toy config (real GPT-2 small is d=768, heads=12, layers=12, vocab=50257).
VOCAB, CTX, D, N_HEAD, D_FF, N_LAYER = 32, 16, 12, 3, 48, 2
assert D % N_HEAD == 0


def randn(*shape):
  return (np.random.randn(*shape) * 0.02).astype(np.float32)


def make_block():
  return Block(
    ln1_g=np.ones(D, np.float32), ln1_b=np.zeros(D, np.float32),
    attn_c_attn=(randn(D, 3 * D), np.zeros(3 * D, np.float32)),
    attn_c_proj=(randn(D, D), np.zeros(D, np.float32)),
    ln2_g=np.ones(D, np.float32), ln2_b=np.zeros(D, np.float32),
    mlp_c_fc=(randn(D, D_FF), np.zeros(D_FF, np.float32)),
    mlp_c_proj=(randn(D_FF, D), np.zeros(D, np.float32)),
  )


def main():
  np.random.seed(0)
  model = Model(
    wte=randn(VOCAB, D), wpe=randn(CTX, D),
    blocks=[make_block() for _ in range(N_LAYER)],
    lnf_g=np.ones(D, np.float32), lnf_b=np.zeros(D, np.float32),
  )

  tokens = [3, 14, 1, 7, 9]                       # a toy input sequence
  logits = gpt2(tokens, model, n_head=N_HEAD)     # [N, VOCAB]

  print(f"GPT-2-style forward pass: {len(tokens)} tokens, "
        f"d={D}, heads={N_HEAD}, layers={N_LAYER}")
  print(f"logits shape: {logits.shape}  (sequence_length x vocab)")
  assert logits.shape == (len(tokens), VOCAB)

  # The last row predicts the next token: softmax it and take argmax.
  probs = softmax(logits[-1])
  assert abs(probs.sum() - 1.0) < 1e-4, "softmax must produce a distribution"
  next_token = int(np.argmax(probs))
  print(f"next-token distribution sums to {probs.sum():.4f}; "
        f"argmax -> token {next_token}")
  print("Transformer forward pass successful!")


if __name__ == "__main__":
  main()
