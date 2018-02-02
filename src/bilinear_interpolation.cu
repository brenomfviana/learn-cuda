/*
 * This code performs linear interpolation between four colors (one for each
 * corner) and generates an image.
 *
 * To run: ./bilinear_interpolation.x WIDTH HEIGHT UPPER_LEFT_CORNER_COLOR
 * LOWER_LEFT_CORNER_COLOR UPPER_RIGHT_CORNER_COLOR LOWER_RIGHT_CORNER_COLOR
 * Color format: "R G B"
 * For example: ./bilinear_interpolation.x 200 100 "0 255 0" "0 0 0" "255 255 0" "255 0 0"
 *
 * @author Breno Viana
 * @version 02/02/2018
 */
#include <fstream>
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
__device__
Color operator*(const Color& color, const float t) {
  Color c = { color.r * t, color.g * t, color.b * t };
  return c;
}

/*!
 * Color multiplied by an intensity.
 *
 * @param t Color intensity
 * @param color Color
 */
__device__
Color operator*(const float t, const Color& color) {
  Color c = { color.r * t, color.g * t, color.b * t };
  return c;
}

/*!
 * Sum of colors.
 *
 * @param lhs Color 1
 * @param rhs Color 2
 */
__device__
Color operator+(const Color& lhs, const Color& rhs) {
  Color c = { lhs.r + rhs.r, lhs.g + rhs.g, lhs.b + rhs.b };
  return c;
}

/* -------------------------------------------------------------------------- */

/*!
 * Apply bilinear interpolation.
 *
 * @param img Image
 * @param upper_left Upper left
 * @param lower_left Lower left
 * @param upper_right Upper right
 * @param lower_right Lower right
 */
__global__
void bi__(Image img, Color* upper_left, Color* lower_left,
  Color* upper_right, Color* lower_right) {
    // Get matrix row
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    // Get matrix column
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    // Check if row and column are not valid
    if (row >= img.height || col >= img.width) {
      return;
    }
    // Progression of the gradient
    float x = float(col) / float(img.width);
    float y = float(row) / float(img.height);
    // Get color
    Color c = ((1 - y) * ((1 - x) * (*lower_left) + x * (*lower_right)) +
              y  * ((1 - x) * (*upper_left) + x * (*upper_right)));
    // Set pixel color
    img.pixels[((img.height - row - 1) * img.width * RGB_SIZE) + (col * RGB_SIZE)] = c.r;
    img.pixels[((img.height - row - 1) * img.width * RGB_SIZE) + (col * RGB_SIZE) + 1] = c.g;
    img.pixels[((img.height - row - 1) * img.width * RGB_SIZE) + (col * RGB_SIZE) + 2] = c.b;
}

/*!
 * Generate an image from a bilinear interpolation.
 *
 * @param img Image
 * @param upper_left Upper left
 * @param lower_left Lower left
 * @param upper_right Upper right
 * @param lower_right Lower right
 */
void generate_image(Image img, const Color upper_left, const Color lower_left,
  const Color upper_right, const Color lower_right) {
    // Load colors to devive memory
    Color* d_upper_left;
    CudaSafeCall(cudaMalloc((void**)& d_upper_left, sizeof(Color)));
    CudaSafeCall(cudaMemcpy(d_upper_left, &upper_left, sizeof(Color),
      cudaMemcpyHostToDevice));
    Color* d_lower_left;
    CudaSafeCall(cudaMalloc((void**)& d_lower_left, sizeof(Color)));
    CudaSafeCall(cudaMemcpy(d_lower_left, &lower_left, sizeof(Color),
      cudaMemcpyHostToDevice));
    Color* d_upper_right;
    CudaSafeCall(cudaMalloc((void**)& d_upper_right, sizeof(Color)));
    CudaSafeCall(cudaMemcpy(d_upper_right, &upper_right, sizeof(Color),
      cudaMemcpyHostToDevice));
    Color* d_lower_right;
    CudaSafeCall(cudaMalloc((void**)& d_lower_right, sizeof(Color)));
    CudaSafeCall(cudaMemcpy(d_lower_right, &lower_right, sizeof(Color),
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
    bi__<<<dimGrid, dimBlock>>>(d_img, d_upper_left, d_lower_left, d_upper_right,
      d_lower_right);
    CudaCheckError();
    CudaSafeCall(cudaThreadSynchronize());
    // Read image from device memory
    CudaSafeCall(cudaMemcpy(img.pixels, d_img.pixels, size, cudaMemcpyDeviceToHost));
    // Free device memory
    CudaSafeCall(cudaFree(d_img.pixels));
    CudaSafeCall(cudaFree(d_upper_left));
    CudaSafeCall(cudaFree(d_lower_left));
    CudaSafeCall(cudaFree(d_upper_right));
    CudaSafeCall(cudaFree(d_lower_right));
}

int main(int argc, char* argv[]) {
  // Initialize image
  Image img;
  img.width  = std::atoi(argv[1]);
  img.height = std::atoi(argv[2]);
  img.pixels = new char[img.height * img.width * RGB_SIZE];
  // Get upper left color
  char* r, * g, * b;
  r = strtok(argv[3], " ");
  g = strtok(NULL, " ");
  b = strtok(NULL, " ");
  Color upper_left;
  upper_left.r = std::atof(r);
  upper_left.g = std::atof(g);
  upper_left.b = std::atof(b);
  // Get lower left color
  r = strtok(argv[4], " ");
  g = strtok(NULL, " ");
  b = strtok(NULL, " ");
  Color lower_left;
  lower_left.r = std::atof(r);
  lower_left.g = std::atof(g);
  lower_left.b = std::atof(b);
  // Get upper right color
  r = strtok(argv[5], " ");
  g = strtok(NULL, " ");
  b = strtok(NULL, " ");
  Color upper_right;
  upper_right.r = std::atof(r);
  upper_right.g = std::atof(g);
  upper_right.b = std::atof(b);
  // Get lower right color
  r = strtok(argv[6], " ");
  g = strtok(NULL, " ");
  b = strtok(NULL, " ");
  Color lower_right;
  lower_right.r = std::atof(r);
  lower_right.g = std::atof(g);
  lower_right.b = std::atof(b);

  // Get image
  generate_image(img, upper_left, lower_left, upper_right, lower_right);

  // OUTPUT
  std::ofstream outfile("output.ppm");
  // PPM header
  outfile << "P6\n";
  outfile << img.width << " " << img.height << "\n";
  outfile << "255" << "\n";
  // Print image
  outfile.write(img.pixels, img.width * img.height * RGB_SIZE);
  outfile.close();
  // Program successfully completed
  return EXIT_SUCCESS;
}
