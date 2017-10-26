/*
 * This code performs error checking for CUDA code.
 *
 * Code adapted from: How to do error checking in CUDA, link:
 * https://codeyarns.com/2011/03/02/how-to-do-error-checking-in-cuda/
 *
 * @author Breno Viana
 * @version 01/10/2017
 */
#ifndef _ERROR_CHECKING_
#define _ERROR_CHECKING_

#include <iostream>

// Errors checking
#define CUDA_ERROR_CHECK
#define CudaSafeCall(err) __cudaSafeCall(err, __FILE__, __LINE__)
#define CudaCheckError()  __cudaCheckError(__FILE__, __LINE__)

/*!
 * Check if any errors occurred in the CUDA function call.
 *
 * @param err Cuda Error
 * @param file File whrer the error occurred
 * @param line Line of the file where the error occurred
 */
inline void __cudaSafeCall(cudaError err, const char* file, const int line) {
    #ifdef CUDA_ERROR_CHECK
        // Check if an error has occurred
        if (cudaSuccess != err) {
            // Print error
            std::cerr << "cudaSafeCall() failed at " << file << ":" << line
                      << " : " << cudaGetErrorString(err) << std::endl;
            exit(EXIT_FAILURE);
        }
    #endif
    return;
}

/*!
 * Check for any error.
 *
 * @param file File whrer the error occurred
 * @param line Line of the file where the error occurred
 */
inline void __cudaCheckError(const char* file, const int line) {
    #ifdef CUDA_ERROR_CHECK
        // Check if an error has occurred
        cudaError err = cudaGetLastError();
        if (cudaSuccess != err) {
            std::cerr << "cudaCheckError() with sync failed at " << file << ":"
                      << line << " : " << cudaGetErrorString(err) << std::endl;
            exit(EXIT_FAILURE);
        }
        // More careful checking. However, this will affect performance.
        err = cudaDeviceSynchronize();
        if(cudaSuccess != err) {
            std::cerr << "cudaCheckError() with sync failed at " << file << ":"
                      << line << " : " << cudaGetErrorString(err) << std::endl;
            exit(EXIT_FAILURE);
        }
    #endif
    return;
}

#endif // _ERROR_CHECKING_
