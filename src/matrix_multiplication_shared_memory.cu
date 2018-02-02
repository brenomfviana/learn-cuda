/*
 * This code performs a simple matrix multiplication without memory sharing.
 *
 * Code adapted from Matrix Multiplication with CUDA — A basic introduction to
 * the CUDA programming model and CUDA tiled matrix-matrix multiplication with
 * matrix dimensions not multiple of the tile dimensions, links:
 * https://www.shodor.org/media/content/petascale/materials/UPModules/matrixMultiplication/moduleDocument.pdf
 * and http://www.orangeowlsolutions.com/archives/526
 *
 * To run: ./matrix_multiplication.x A_B_HEIGHT A_WIDTH B_WIDTH MAX_RANDOM_VALUE
 *
 * @author Breno Viana
 * @version 02/02/2018
 */
#include <iostream>
#include "error_checking.cuh"

// Thread block size
#define BLOCK_SIZE 16

/*!
 * Matrices are stored in row-major order:
 * M(row, col) = *(M.elements + row * M.width + col).
 */
typedef struct {
  int width;
  int height;
  float* elements;
} Matrix;

/*!
 * Apply matrix multiplication on GPU.
 *
 * @param A Matrix A
 * @param B Matrix B
 * @param C Resulting matrix
 */
__global__
void mmd__(Matrix A, Matrix B, Matrix C) {
  // Block row and column
  int b_row = blockIdx.y;
  int b_col = blockIdx.x;
  // Thread row and column
  int t_row = threadIdx.y;
  int t_col = threadIdx.x;
  // Indexes
  int row = b_row * blockDim.y + t_row;
  int col = b_col * blockDim.x + t_col;
  // Check if row and column are not valid
  if (row > C.height || col > C.width) {
    return;
  }
  // Element value
  float value = 0.f;
  // Shared memory used to store ds_A and ds_B respectively
  __shared__ float ds_A[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float ds_B[BLOCK_SIZE][BLOCK_SIZE];
  // Multiply each pair of sub-matrices together and accumulate the results
  for (int i = 0; i < ((BLOCK_SIZE + A.width - 1) / BLOCK_SIZE); i++) {
    // Get element of matrix A
    if ((row < A.height) && ((i * BLOCK_SIZE + t_col) < A.width)) {
      ds_A[t_row][t_col] = A.elements[(row * A.width) + (i * BLOCK_SIZE + t_col)];
    } else {
      ds_A[t_row][t_col] = 0;
    }
    // Get element of matrix B
    if ((col < B.width) && ((i * BLOCK_SIZE + t_row) < B.height)) {
      ds_B[t_row][t_col] = B.elements[((i * BLOCK_SIZE + t_row) * B.width) + col];
    } else {
      ds_B[t_row][t_col] = 0;
    }
    __syncthreads();
    for (int j = 0; j < BLOCK_SIZE; j++) {
      value += ds_A[t_row][j] * ds_B[j][t_col];
    }
    __syncthreads();
  }
  C.elements[row * C.width + col] = value;
}

/*!
 * Prepares matrix multiplication on GPU.
 *
 * @param A Matrix A
 * @param B Matrix B
 * @param C Resulting matrix
 */
void matrix_multiplication(const Matrix A, const Matrix B, Matrix C) {
  // Device matrices
  Matrix d_a, d_b, d_c;
  // Load A to device memory
  d_a.height = A.height;
  d_a.width = A.width;
  size_t size = A.width * A.height * sizeof(float);
  CudaSafeCall(cudaMalloc(&d_a.elements, size));
  CudaSafeCall(cudaMemcpy(d_a.elements, A.elements, size,
    cudaMemcpyHostToDevice));
  // Load B to device memory
  d_b.height = B.height;
  d_b.width = B.width;
  size = B.width * B.height * sizeof(float);
  CudaSafeCall(cudaMalloc(&d_b.elements, size));
  CudaSafeCall(cudaMemcpy(d_b.elements, B.elements, size,
    cudaMemcpyHostToDevice));
  // Allocate C in device memory
  d_c.height = C.height;
  d_c.width = C.width;
  size = C.width * C.height * sizeof(float);
  CudaSafeCall(cudaMalloc(&d_c.elements, size));
  // Blocks per grid
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  // Threads per block
  dim3 dimGrid((B.width + dimBlock.x - 1) / dimBlock.x,
    (A.height + dimBlock.y - 1) / dimBlock.y);
  // Performs multiplication
  mmd__<<<dimGrid, dimBlock>>>(d_a, d_b, d_c);
  CudaCheckError();
  CudaSafeCall(cudaThreadSynchronize());
  // Read C from device memory
  CudaSafeCall(cudaMemcpy(C.elements, d_c.elements, size,
    cudaMemcpyDeviceToHost));
  // Free device memory
  CudaSafeCall(cudaFree(d_a.elements));
  CudaSafeCall(cudaFree(d_b.elements));
  CudaSafeCall(cudaFree(d_c.elements));
}

int main(int argc, char* argv[]) {
  // Create matrices
  Matrix A, B, C;
  // Initialize matrix A
  A.height = atoi(argv[1]);
  A.width = atoi(argv[2]);
  A.elements = (float*) malloc(A.width * A.height * sizeof(float));
  // Initialize matrix B
  B.height = A.height;
  B.width = atoi(argv[3]);
  B.elements = (float*) malloc(B.width * B.height * sizeof(float));
  // Initialize matrix C
  C.height = A.height;
  C.width =  B.width;
  C.elements = (float*) malloc(C.width * C.height * sizeof(float));
  // Max random value
  int r = atoi(argv[4]);
  srand(time(NULL));
  // Initialize values of matrix A
  for(int i = 0; i < A.height; i++) {
    for(int j = 0; j < A.width; j++) {
      A.elements[i * A.height + j] = rand() % r;
    }
  }
  // Initialize values of matrix B
  for(int i = 0; i < B.height; i++) {
    for(int j = 0; j < B.width; j++) {
      B.elements[i * B.height + j] = rand() % r;
    }
  }

  // Performs matrix calculation (ON GPU)
  matrix_multiplication(A, B, C);

  // Print matrix A
  std::cout << "Matrix A:\n";
  for(int i = 0; i < A.height; i++) {
    for(int j = 0; j < A.width; j++) {
      std::cout << A.elements[i * A.height + j] << " ";
    }
    std::cout << "\n";
  }
  // Print matrix B
  std::cout << "Matrix B:\n";
  for(int i = 0; i < B.height; i++) {
    for(int j = 0; j < B.width; j++) {
      std::cout << B.elements[i * B.height + j] << " ";
    }
    std::cout << "\n";
  }
  // Print result (matrix C)
  std::cout << "Result of matrix multiplication (Matrix C):\n";
  for(int i = 0; i < C.height; i++) {
    for(int j = 0; j < C.width; j++) {
      std::cout << C.elements[i * C.height + j] << " ";
    }
    std::cout << "\n";
  }
  // Program successfully completed
  return EXIT_SUCCESS;
}
