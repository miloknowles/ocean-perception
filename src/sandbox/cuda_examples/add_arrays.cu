#include <iostream>
#include <math.h>


__global__
void add(int n, float *x, float *y)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  // blockDim is the # of threads in *this* thread's block.
  // gridDim is the # blocks in the grid
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride) {
    y[i] = x[i] + y[i];
  }
}


// nvprof --unified-memory-profiling off ./add_arrays
int main(int argc, char const *argv[])
{
  int N = 1<<20;
  float *x, *y;

  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;
  add<<<numBlocks, blockSize>>>(N, x, y);

  // Free memory
  cudaFree(x);
  cudaFree(y);

  std::cout << "Ran the CUDA code" << std::endl;

  return 0;
}
