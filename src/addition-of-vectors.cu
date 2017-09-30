/*
 * This code performs a simple addition of vectors.
 *
 * Code adapted from a CUDA tutorial (Video tutorial series on "CUDA Programming
 * for Beginners" - Tutorial 1), link:
 * http://hpsc-mandar.blogspot.com.br/2016/08/video-tutorial-series-on-cuda.html.
 *
 * To run: ./addition-of-vectors.x SIZE_OF_VECTORS MAX_RANDOM_VALUE
 *
 * @author Breno Viana
 * @version 29/09/2017
 */
#include <ctime>
#include <cstdlib>
#include "error-checking.h"

/*!
 * Apply the sum of vectors on GPU.
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

int main(int argc, char* argv[]) {
    // Get the size of the vectors
    size_t size = atoi(argv[1]);
    // Max random value
    int r = atoi(argv[2]);
    // Size of memory to be allocated
    int sa = size * sizeof(int);
    // Create vectors
    int m[size], n[size], p[size], *mgpu, *ngpu, *pgpu;
    // Initialize the vectors on CPU
    srand(time(NULL));
    for (int i = 0; i < size; i++) {
        m[i] = rand() % r;
        n[i] = rand() % r;
        p[i] = 0;
    }
    // Allocatte vectors on GPU and transfer arrays from CPU (host) memory to
    // GPU (device)
    CudaSafeCall(cudaMalloc(&mgpu, sa));
    CudaSafeCall(cudaMemcpy(mgpu, m, sa, cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMalloc(&ngpu, sa));
    CudaSafeCall(cudaMemcpy(ngpu, n, sa, cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMalloc(&pgpu, sa));
    // Blocks per grid
    dim3 dim_grid(1, 1);
    // Threads per block
    dim3 dim_block(size, 1);
    // Run addition of vectors
    vec_add<<<dim_grid, dim_block>>>(mgpu, ngpu, pgpu);
    CudaCheckError();
    // Tranfer results from GPU (device) memory to CPU (host) memory
    CudaSafeCall(cudaMemcpy(p, pgpu, sa, cudaMemcpyDeviceToHost));
    // De-allocate GPU memory
    cudaFree(mgpu);
    cudaFree(ngpu);
    cudaFree(pgpu);
    std::cout << "Vector 1:" << std::endl;
    for (int i = 0; i < size; i++) {
        std::cout << m[i] << " ";
    }
    std::cout << std::endl << std::endl;
    std::cout << "Vector 2:" << std::endl;
    for (int i = 0; i < size; i++) {
        std::cout << n[i] << " ";
    }
    std::cout << std::endl << std::endl;
    // Print results
    std::cout << "Resulting vector:" << std::endl;
    for (int i = 0; i < size; i++) {
        std::cout << p[i] << " ";
    }
    std::cout << std::endl;
    // Program successfully completed
    return EXIT_SUCCESS;
}
