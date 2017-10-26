/*
 * This code performs a simple matrix multiplication without memory sharing.
 *
 * Code adapted from: Matrix Multiplication with CUDA — A basic introduction to
 * the CUDA programming model, link:
 * https://www.shodor.org/media/content/petascale/materials/UPModules/matrixMultiplication/moduleDocument.pdf
 *
 * To run: ./matrix_multiplication.x A_B_HEIGHT A_WIDTH B_WIDTH MAX_RANDOM_VALUE
 *
 * @author Breno Viana
 * @version 29/09/2017
 */
#include <ctime>
#include <cstdlib>
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
__global__ void mmd__(Matrix A, Matrix B, Matrix C) {
    // Element of the matrix C
    float e = 0.0;
    // Get matrix row
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    // Get matrix column
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    // Check if row and column are not valid
    if (row > C.height || col > C.width) {
        return;
    }
    // Multiplication
    for (int i = 0; i < A.width; ++i) {
        e += (A.elements[row * A.width + i]) * (B.elements[i * B.width + col]);
        C.elements[row * C.width + col] = e;
    }
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
    d_a.height  = A.height;
    d_a.width   = A.width;
    size_t size = A.width * A.height * sizeof(float);
    CudaSafeCall(cudaMalloc(&d_a.elements, size));
    CudaSafeCall(cudaMemcpy(d_a.elements, A.elements, size,
                 cudaMemcpyHostToDevice));
    // Load B to device memory
    d_b.height = B.height;
    d_b.width  = B.width;
    size       = B.width * B.height * sizeof(float);
    CudaSafeCall(cudaMalloc(&d_b.elements, size));
    CudaSafeCall(cudaMemcpy(d_b.elements, B.elements, size,
                            cudaMemcpyHostToDevice));
    // Allocate C in device memory
    d_c.height = C.height;
    d_c.width  = C.width;
    size       = C.width * C.height * sizeof(float);
    CudaSafeCall(cudaMalloc(&d_c.elements, size));
    // Blocks per grid
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    // Threads per block
    dim3 dimGrid((B.width  + dimBlock.x - 1) / dimBlock.x,
                 (A.height + dimBlock.y - 1) / dimBlock.y);
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
    A.height   = atoi(argv[1]); // Height of A
    A.width    = atoi(argv[2]); // Width of A
    A.elements = (float*) malloc(A.width * A.height * sizeof(float));
    // Initialize matrix B
    B.height   = A.height;      // Height of B
    B.width    = atoi(argv[3]); // Width of B
    B.elements = (float*) malloc(B.width * B.height * sizeof(float));
    // Initialize matrix C
    C.height   = A.height;
    C.width    =  B.width;
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
    std::cout << "Matrix A" << std::endl;
    for(int i = 0; i < A.height; i++) {
        for(int j = 0; j < A.width; j++) {
            std::cout << A.elements[i * A.height + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    // Print matrix B
    std::cout << "Matrix B\n" << std::endl;
    for(int i = 0; i < B.height; i++) {
        for(int j = 0; j < B.width; j++) {
            std::cout << B.elements[i * B.height + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    // Print result (matrix C)
    std::cout << "Result of matrix multiplication" << std::endl
              << "Matrix C" << std::endl;
    for(int i = 0; i < C.height; i++) {
        for(int j = 0; j < C.width; j++) {
            std::cout << C.elements[i * C.height + j] << " ";
        }
        std::cout << std::endl;
    }
    // Program successfully completed
    return EXIT_SUCCESS;
}
