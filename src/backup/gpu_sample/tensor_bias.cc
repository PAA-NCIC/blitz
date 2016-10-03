#include <iostream>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "backend/backends.h"
#include "backend/shape.h"

using namespace blitz;

int main() {
  Shape input_shape(2);
  Shape bias_shape(1);
  input_shape[0] = 128;
  input_shape[1] = 1000;
  bias_shape[0] = 1000;
  GPUTensor<float> input(input_shape);
  GPUTensor<float> output(input_shape);
  GPUTensor<float> bias(bias_shape);
  CPUTensor<float> output_copy(input_shape);
  CPUTensor<float> bias_copy(bias_shape);
  input.Fill(1);
  Backend<GPUTensor, float>::BiasBackwardUpdateFunc(&input,
    &bias);
  cudaMemcpy(bias_copy.data(), bias.data(),
    bias.size() * sizeof(float),
    cudaMemcpyDeviceToHost);
  
  CPUTensor<float> cpu_input(input_shape);
  CPUTensor<float> cpu_output(input_shape);
  CPUTensor<float> cpu_bias(bias_shape);
  cpu_input.Fill(1);
  Backend<CPUTensor, float>::BiasBackwardUpdateFunc(&cpu_input,
    &cpu_bias);
  for (int i = 0; i < 100; ++i) {
    if (bias_copy[i] != cpu_bias[i]) {
      std::cout << " i: " << i << " : " << bias_copy[i] << std::endl;
    }
  }
  return 0;
}

