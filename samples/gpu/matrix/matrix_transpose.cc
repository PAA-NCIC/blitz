#include <iostream>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "backends/backends.h"

using namespace blitz;
// M N
Shape input_shape(2);
// N M
Shape output_shape(2);

void compare_cpu_gpu(size_t size, float* output_cpu, float* output_gpu) {
  for (size_t i = 0; i < size; ++i) {
    if (output_cpu[i] > output_gpu[i] + 1e-3 ||
      output_cpu[i] < output_gpu[i] - 1e-3) {
      std::cout << "Index: " << i << ", CPU: " << output_cpu[i] <<
        ", GPU: " << output_gpu[i] << std::endl;
    }
  }
}

void transpose(size_t m, size_t n) {
  // set up cpu
  CPUTensor<float> input_cpu(input_shape);
  CPUTensor<float> output_cpu(output_shape);
  // set up gpu
  GPUTensor<float> input_gpu(input_shape);
  GPUTensor<float> output_gpu(output_shape);
  CPUTensor<float> output_copy(output_shape);
  // init values
  Backend<CPUTensor, float>::UniformDistributionFunc(&input_cpu, 0.0, 1.0);
  cudaMemcpy(input_gpu.data(), input_cpu.data(),
    input_cpu.size() * sizeof(float), cudaMemcpyHostToDevice);
  // transpose
  Backend<CPUTensor, float>::Transpose2DFunc(&input_cpu, &output_cpu);
  Backend<GPUTensor, float>::Transpose2DFunc(&input_gpu, &output_gpu);
  // copy from gpu to cpu
  cudaMemcpy(output_copy.data(), output_gpu.data(),
    output_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
  compare_cpu_gpu(output_cpu.size(), output_cpu.data(), output_copy.data());
}

int main(int argc, char** argv) {
  const size_t NUM_ARGS = 2;
  // M N
  if (argc != NUM_ARGS + 1) {
    std::cerr << "Not enough args!" << std::endl;
    exit(1);
  }
  const size_t M = atoi(argv[1]);
  const size_t N = atoi(argv[2]);
  // set shapes
  input_shape[0] = M;
  input_shape[1] = N;
  output_shape[0] = N;
  output_shape[1] = M;
  // run
  transpose(M, N);
  return 0;
}
