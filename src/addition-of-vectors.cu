/*
 * This code performs a simple addition of vectors.
 *
 * Code adapted from a CUDA tutorial (Video tutorial series on "CUDA Programming
 * for Beginners" - Tutorial 1), link:
 * http://hpsc-mandar.blogspot.com.br/2016/08/video-tutorial-series-on-cuda.html.
 *
 * @author Breno Viana
 * @version 22/09/2017
 */
#include <vector>
#include <iostream>

#define SIZE 200

/*!
 * Apply the sum of vectors.
 *
 * @param m First vector
 * @param n Second vector
 * @param p Resulting vector
 * @param size Size of vectors
 */
__global__ void vec_add(const int* m, const int* n, int* p) {
    // Get thread id
    int myid = threadIdx.x;
    // Sum each vector element
    p[myid] = m[myid] + n[myid];
}

int main() {
    // Size of memory to be allocated
    int sa = SIZE * sizeof(int);
    // Create vectors
    int m[SIZE], n[SIZE], p[SIZE], *mgpu, *ngpu, *pgpu;
    // Initialize the vectors on CPU
    for (int i = 0; i < SIZE; i++) {
        m[i] = i;
        n[i] = i;
        p[i] = 0;
    }
    // Allocatte vectors on GPU and transfer arrays from CPU (host) memory to
    // GPU (device)
    cudaMalloc(&mgpu, sa);
    cudaMemcpy(mgpu, m, sa, cudaMemcpyHostToDevice);
    cudaMalloc(&ngpu, sa);
    cudaMemcpy(ngpu, n, sa, cudaMemcpyHostToDevice);
    cudaMalloc(&pgpu, sa);
    // Blocks per grid
    dim3 dim_grid(1, 1);
    // Threads per block
    dim3 dim_block(SIZE, 1);
    // Run addition of vectors
    vec_add<<<dim_grid, dim_block>>>(mgpu, ngpu, pgpu);
    // Tranfer results from GPU (device) memory to CPU (host) memory
    cudaMemcpy(p, pgpu, sa, cudaMemcpyDeviceToHost);
    // De-allocate GPU memory
    cudaFree(mgpu);
    cudaFree(ngpu);
    cudaFree(pgpu);
    // Print results
    for (int i = 0; i < SIZE; i++) {
        std::cout << p[i] << " ";
    }
    std::cout << std::endl;
    // Program successfully completed
    return EXIT_SUCCESS;
}
