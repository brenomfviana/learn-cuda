#ifndef _ERROR_CHECKING_
#define _ERROR_CHECKING_

#include <cstdio>
#include <iostream>

// Errors checking
#define CUDA_ERROR_CHECK
#define CudaSafeCall(err) __cudaSafeCall(err, __FILE__, __LINE__)
#define CudaCheckError()  __cudaCheckError(__FILE__, __LINE__)

/*!
 * .
 *
 * @param err
 * @param file
 * @param line
 */
inline void __cudaSafeCall(cudaError err, const char* file, const int line) {
    #ifdef CUDA_ERROR_CHECK
        if (cudaSuccess != err) {
            fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n", file, line,
                    cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
    #endif
    return;
}

/*!
 * .
 *
 * @param file
 * @param line
 */
inline void __cudaCheckError(const char* file, const int line) {
    #ifdef CUDA_ERROR_CHECK
        cudaError err = cudaGetLastError();
        if (cudaSuccess != err) {
            fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n", file,
                    line, cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        // More careful checking. However, this will affect performance.
        err = cudaDeviceSynchronize();
        if(cudaSuccess != err) {
            fprintf(stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                    file, line, cudaGetErrorString( err ));
            exit(EXIT_FAILURE);
        }
    #endif
    return;
}

#endif // _ERROR_CHECKING_
