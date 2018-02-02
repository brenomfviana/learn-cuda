/*
 * This code performs a simple addition of vectors.
 *
 * Code adapted from: Video tutorial series on "CUDA Programming
 * for Beginners" - Tutorial 1, link:
 * http://hpsc-mandar.blogspot.com.br/2016/08/video-tutorial-series-on-cuda.html.
 *
 * To run: ./addition_of_vectors.x SIZE_OF_VECTORS MAX_RANDOM_VALUE
 *
 * @author Breno Viana
 * @version 02/02/2018
 */
#include <ctime>
#include <cstdio>
#include <cstdlib>
#include "error_checking.cuh"

/*!
 * Apply the sum of vectors on GPU.
 *
 * @param m First vector
 * @param n Second vector
 * @param p Resulting vector
 * @param size Size of vectors
 */
__global__
void vec_add(const int* m, const int* n, int* p) {
  // Get thread id
  int i = threadIdx.x;
  // Sum each vector element
  p[i] = m[i] + n[i];
  printf("%i\n", i);
}

int main(int argc, char* argv[]) {
  // Get the size of the vectors
  int size = atoi(argv[1]);
  // Size of memory to be allocated
  size_t d_size = size * sizeof(int);
  // Max random value
  int r = atoi(argv[2]);
  // Create vectors
  int m[size], n[size], p[size], *d_m, *d_n, *d_p;
  // Initialize the vectors on CPU
  srand(time(NULL));
  for (int i = 0; i < size; i++) {
    m[i] = rand() % r;
    n[i] = rand() % r;
    p[i] = 0;
  }
  // Allocatte vectors on GPU and transfer arrays from CPU (host) memory to
  // GPU (device)
  CudaSafeCall(cudaMalloc(&d_m, d_size));
  CudaSafeCall(cudaMemcpy(d_m, m, d_size, cudaMemcpyHostToDevice));
  CudaSafeCall(cudaMalloc(&d_n, d_size));
  CudaSafeCall(cudaMemcpy(d_n, n, d_size, cudaMemcpyHostToDevice));
  CudaSafeCall(cudaMalloc(&d_p, d_size));
  // Blocks per grid
  dim3 dim_grid(1, 1);
  // Threads per block
  dim3 dim_block(size, 1);
  // Run addition of vectors
  vec_add<<<dim_grid, dim_block>>>(d_m, d_n, d_p);
  cudaThreadSynchronize();
  CudaCheckError();
  // Tranfer results from GPU (device) memory to CPU (host) memory
  CudaSafeCall(cudaMemcpy(p, d_p, d_size, cudaMemcpyDeviceToHost));
  // De-allocate GPU memory
  CudaSafeCall(cudaFree(d_m));
  CudaSafeCall(cudaFree(d_n));
  CudaSafeCall(cudaFree(d_p));
  // Print vectors
  std::cout << "Vector 1:\n";
  for (int i = 0; i < size; i++) {
    std::cout << m[i] << " ";
  }
  std::cout << "\n\n";
  std::cout << "Vector 2:\n";
  for (int i = 0; i < size; i++) {
    std::cout << n[i] << " ";
  }
  std::cout << "\n\n";
  // Print result
  std::cout << "Resulting vector:\n";
  for (int i = 0; i < size; i++) {
    std::cout << p[i] << " ";
  }
  std::cout << "\n";
  // Program successfully completed
  return EXIT_SUCCESS;
}
