#include <iostream>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "backend/backends.h"
#include "backend/shape.h"

using namespace blitz;

int main() {
  Shape input_shape(2);
  input_shape[0] = 128;
  input_shape[1] = 1000;
  GPUTensor<float> input(input_shape);
  input.Fill(-7);
  float* input_copy = (float*)malloc(128 * 1000 * sizeof(float));
  cudaMemcpy(input_copy, input.data(),
    input.size() * sizeof(float),
    cudaMemcpyDeviceToHost);
  for (int i = 0; i < 100; ++i) {
    std::cout << input_copy[i] << " ";
  }
  return 0;
}
