/*
 * This code performs a linear interpolation from top to bottom of two colors
 * and prints the corresponding image.
 *
 * To run: ./linear_interpolation.x WIDTH HEIGHT UPPER_COLOR LOWER_COLOR
 * Color format: "R G B"
 * For example: ./linear_interpolation.x 200 100 "127.5 178.5 255" "255 255 255"
 *
 * @author Breno Viana
 * @version 26/10/2017
 */
#include <fstream>
#include <cstdlib>
#include <cstring>
#include "error_checking.cuh"

// Thread block size
#define BLOCK_SIZE 16
#define RGB_SIZE 3

/*!
 * Image.
 */
typedef struct {
  int width;
  int height;
  char* pixels;
} Image;

/*!
 * Color.
 */
typedef struct {
  float r;
  float g;
  float b;
} Color;

/* ---------------------------- Operations ---------------------------------- */

/*!
 * Color multiplied by an intensity.
 *
 * @param color Color
 * @param t Color intensity
 */
__device__ Color operator*(const Color& color, const float t) {
  Color c = { color.r * t, color.g * t, color.b * t };
  return c;
}

/*!
 * Color multiplied by an intensity.
 *
 * @param t Color intensity
 * @param color Color
 */
__device__ Color operator*(const float t, const Color& color) {
  Color c = { color.r * t, color.g * t, color.b * t };
  return c;
}

/*!
 * Sum of colors.
 *
 * @param lhs Color 1
 * @param rhs Color 2
 */
__device__ Color operator+(const Color& lhs, const Color& rhs) {
  Color c = { lhs.r + rhs.r, lhs.g + rhs.g, lhs.b + rhs.b };
  return c;
}

/* -------------------------------------------------------------------------- */

/*!
 * Apply linear interpolation.
 *
 * @param img Image
 * @param upper_color Upper color
 * @param lower_color Lower color
 */
__global__ void li__(Image img, Color* upper_color, Color* lower_color) {
  // Get matrix row
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  // Get matrix column
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  // Check if row and column are not valid
  if (row > img.height || col > img.width) {
    return;
  }
  // Progression of the gradient
  float y = float(row) / float(img.height);
  // Get color
  Color c = (((1 - y) * (*lower_color)) + (y * (*upper_color)));
  // Set pixel color
  img.pixels[((img.height - row - 1) * img.width * RGB_SIZE) + (col * RGB_SIZE)]   = c.r;
  img.pixels[((img.height - row - 1) * img.width * RGB_SIZE) + (col * RGB_SIZE) + 1] = c.g;
  img.pixels[((img.height - row - 1) * img.width * RGB_SIZE) + (col * RGB_SIZE) + 2] = c.b;
}

/*!
 * Generate an image from a linear interpolation.
 *
 * @param img Image
 * @param upper_color Upper color
 * @param lower_color Lower color
 */
void generate_image(Image img, const Color upper_color, const Color lower_color) {
  // Load colors to devive memory
  Color* d_upper_color;
  CudaSafeCall(cudaMalloc((void**)& d_upper_color, sizeof(Color)));
  CudaSafeCall(cudaMemcpy(d_upper_color, &upper_color, sizeof(Color),
  cudaMemcpyHostToDevice));
  Color* d_lower_color;
  CudaSafeCall(cudaMalloc((void**)& d_lower_color, sizeof(Color)));
  CudaSafeCall(cudaMemcpy(d_lower_color, &lower_color, sizeof(Color),
  cudaMemcpyHostToDevice));
  // Allocate image in device memory
  Image d_img;
  d_img.height = img.height;
  d_img.width  = img.width;
  size_t size  = img.width * img.height * RGB_SIZE * sizeof(char);
  CudaSafeCall(cudaMalloc(&d_img.pixels, size));
  // Blocks per grid
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  // Threads per block
  dim3 dimGrid(((img.width  + dimBlock.x - 1) / dimBlock.x),
  ((img.height + dimBlock.y - 1) / dimBlock.y));
  li__<<<dimGrid, dimBlock>>>(d_img, d_upper_color, d_lower_color);
  CudaCheckError();
  CudaSafeCall(cudaThreadSynchronize());
  // Read image from device memory
  CudaSafeCall(cudaMemcpy(img.pixels, d_img.pixels, size, cudaMemcpyDeviceToHost));
  // Free device memory
  CudaSafeCall(cudaFree(d_img.pixels));
  CudaSafeCall(cudaFree(d_upper_color));
  CudaSafeCall(cudaFree(d_lower_color));
}

int main(int argc, char* argv[]) {
  // Initialize image
  Image img;
  img.width  = std::atoi(argv[1]);
  img.height = std::atoi(argv[2]);
  img.pixels = new char[img.height * img.width * RGB_SIZE];
  // Get upper color
  char* r, * g, * b;
  r = strtok(argv[3], " ");
  g = strtok(NULL, " ");
  b = strtok(NULL, " ");
  Color upper_color;
  upper_color.r = std::atof(r);
  upper_color.g = std::atof(g);
  upper_color.b = std::atof(b);
  // Get lower color
  r = strtok(argv[4], " ");
  g = strtok(NULL, " ");
  b = strtok(NULL, " ");
  Color lower_color;
  lower_color.r = std::atof(r);
  lower_color.g = std::atof(g);
  lower_color.b = std::atof(b);

  // Get image
  generate_image(img, upper_color, lower_color);

  // OUTPUT
  std::ofstream outfile("output.ppm");
  // PPM header
  outfile << "P6" << std::endl;
  outfile << img.width << " " << img.height << std::endl;
  outfile << "255" << std::endl;
  // Print image
  outfile.write(img.pixels, img.width * img.height * RGB_SIZE);
  outfile.close();
  // Program successfully completed
  return EXIT_SUCCESS;
}
