// Calls the compiled @conv through its C interface (llvm.emit_c_interface),
// checks every output element against a plain C reference, and reports the
// best of three timings.
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

enum { N = 5, H = 80, W = 100, C = 128, F = 128, KH = 3, KW = 3 };

// Memref descriptors as MLIR lowers them to LLVM (identity layout).
typedef struct {
  float *allocated, *aligned;
  int64_t offset, sizes[1], strides[1];
} Memref1D;
typedef struct {
  float *allocated, *aligned;
  int64_t offset, sizes[4], strides[4];
} Memref4D;

void _mlir_ciface_conv(Memref4D *result, Memref4D *input, Memref4D *filter,
                       Memref1D *bias, Memref4D *output);

static Memref4D wrap4(float *p, int64_t d0, int64_t d1, int64_t d2, int64_t d3) {
  Memref4D m = {p, p, 0, {d0, d1, d2, d3}, {d1 * d2 * d3, d2 * d3, d3, 1}};
  return m;
}

static double now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(void) {
  // input[N][H+2][W+2][C], filter[C][KH][KW][F], bias[F], output[N][H][W][F].
  size_t nin = (size_t)N * (H + 2) * (W + 2) * C, nfil = (size_t)C * KH * KW * F,
         nout = (size_t)N * H * W * F;
  float *in = malloc(nin * sizeof(float)), *fil = malloc(nfil * sizeof(float)),
        *bias = malloc(F * sizeof(float)), *out = malloc(nout * sizeof(float)),
        *ref = malloc(nout * sizeof(float));
  uint32_t s = 1;
#define RND() ((s = s * 1664525u + 1013904223u), (float)(s >> 8) / (1 << 24) - 0.5f)
  for (size_t i = 0; i < nin; ++i) in[i] = RND();
  for (size_t i = 0; i < nfil; ++i) fil[i] = RND();
  for (size_t i = 0; i < F; ++i) bias[i] = RND();

  Memref4D min = wrap4(in, N, H + 2, W + 2, C), mfil = wrap4(fil, C, KH, KW, F),
           mout = wrap4(out, N, H, W, F), res;
  Memref1D mbias = {bias, bias, 0, {F}, {1}};

  double best = 1e30;
  for (int rep = 0; rep < 3; ++rep) {
    double t0 = now();
    _mlir_ciface_conv(&res, &min, &mfil, &mbias, &mout);
    double t = now() - t0;
    if (t < best) best = t;
    if (res.allocated != out) free(res.allocated);
  }
  _mlir_ciface_conv(&res, &min, &mfil, &mbias, &mout);

  double t0 = now();
  // Reference: relu(bias[f] + sum_{kh,kw,c} filter[c][kh][kw][f] * input[n][y+kh][x+kw][c]),
  // with f innermost so that the C compiler can vectorize it.
  for (int n = 0; n < N; ++n)
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) {
        float acc[F];
        for (int f = 0; f < F; ++f) acc[f] = bias[f];
        for (int kh = 0; kh < KH; ++kh)
          for (int kw = 0; kw < KW; ++kw)
            for (int c = 0; c < C; ++c) {
              float v = in[((n * (H + 2) + y + kh) * (W + 2) + x + kw) * C + c];
              const float *frow = &fil[((c * KH + kh) * KW + kw) * F];
              for (int f = 0; f < F; ++f) acc[f] += frow[f] * v;
            }
        for (int f = 0; f < F; ++f)
          ref[((n * H + y) * W + x) * F + f] = acc[f] > 0 ? acc[f] : 0;
      }

  double tref = now() - t0;

  double maxerr = 0, sum = 0;
  const float *r = res.aligned + res.offset;
  for (size_t i = 0; i < nout; ++i) {
    double e = fabs((double)r[i] - ref[i]);
    if (e > maxerr) maxerr = e;
    sum += r[i];
  }
  printf("checksum %.3f, max abs error vs reference %.2e: %s\n", sum, maxerr,
         maxerr < 1e-3 ? "PASS" : "FAIL");
  double flops = (double)N * H * W * F * (2.0 * KH * KW * C + 2);
  printf("conv: best of 3 %.0f ms, %.1f GFLOP/s\n", best * 1e3, flops / best * 1e-9);
  printf("C reference (clang): %.0f ms, %.1f GFLOP/s\n", tref * 1e3, flops / tref * 1e-9);
  return maxerr < 1e-3 ? 0 : 1;
}
