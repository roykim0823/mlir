// CUDA C reference: square each element of an array, in parallel on the GPU.
//
// This is the hand-written CUDA version of what square.mlir expresses in the MLIR
// `gpu` dialect. Compile and run it on a machine with an NVIDIA GPU + nvcc:
//
//     nvcc square.cu -o square && ./square
//     nvcc -ptx square.cu -o square.ptx       # see the PTX assembly
//
// The kernel is launched as  square<<<blocks, threads>>>(...)  — a grid of thread
// blocks, each block a group of threads. Every thread computes ONE element, using
// its block/thread coordinates to find its global index:
//
//     tid = blockDim.x * blockIdx.x + threadIdx.x
//
// That index math is the essence of GPU programming, and it's exactly what the
// MLIR `gpu.block_id` / `gpu.thread_id` ops reproduce after lowering square.mlir.
#include <stdio.h>

__global__ void square(float *array, int n) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;   // this thread's global index
  if (tid < n)                                       // guard the array bounds
    array[tid] = array[tid] * array[tid];
}

int main() {
  const int N = 1024;
  float *a;
  cudaMallocManaged(&a, N * sizeof(float));
  for (int i = 0; i < N; i++) a[i] = (float)i;

  int threads = 256;
  int blocks = (N + threads - 1) / threads;          // enough blocks to cover N
  square<<<blocks, threads>>>(a, N);                 // launch the grid
  cudaDeviceSynchronize();

  printf("a[2] = %f (expected 4)\n", a[2]);
  cudaFree(a);
  return 0;
}
