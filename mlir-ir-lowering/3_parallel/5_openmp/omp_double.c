// The C counterpart of omp_double.mlir: double a 10-element array with OpenMP.
//
//   #pragma omp parallel   opens a team of threads (the region runs on each)
//   #pragma omp for        shares the loop's iterations across the team
//
// Compile with OpenMP support and run (build.sh does this for you):
//   clang -fopenmp -I$(brew --prefix libomp)/include \
//         -L$(brew --prefix libomp)/lib -Wl,-rpath,$(brew --prefix libomp)/lib \
//         omp_double.c -o omp_double_c
//
// This is the directive-based model MLIR's `omp` dialect mirrors op-for-op; the
// two produce the same output (2, 4, 6, ..., 20).
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

int kernel(float *input, float *output) {
#pragma omp parallel
  {
#pragma omp for
    for (int i = 0; i < 10; i++) {
      output[i] = input[i] * 2.0f;
    }
  }
}

int main() {
  float input[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  float output[10];
  kernel(input, output);
  for (int i = 0; i < 10; i++) {
    printf("%f ", output[i]);
  }
  return 0;
}
