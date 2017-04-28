#include <iostream>

#include <unistd.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <blitz.h>

using namespace blitz;
// M K
Shape left_shape(2);
// K N
Shape right_shape(2);
// M N
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

void multiply(size_t m, size_t n, size_t k, BLITZ_ALGORITHM algorithm, string trans) {
  bool left_trans, right_trans;
  // set shapes
  if (trans == "nn") {
    left_trans = false;
    right_trans = false;
    left_shape[0] = m;
    left_shape[1] = k;
    right_shape[0] = k;
    right_shape[1] = n;
    output_shape[0] = m;
    output_shape[1] = n;
  } else if (trans == "tn") {
    left_trans = true;
    right_trans = false;
    left_shape[0] = k;
    left_shape[1] = m;
    right_shape[0] = k;
    right_shape[1] = n;
    output_shape[0] = m;
    output_shape[1] = n;
  } else if (trans == "nt") {
    left_trans = false;
    right_trans = true;
    left_shape[0] = m;
    left_shape[1] = k;
    right_shape[0] = n;
    right_shape[1] = k;
    output_shape[0] = m;
    output_shape[1] = n;
  } else {
    std::cerr << "No such transform method!" << std::endl;
    exit(1);
  }
  // set up cpu
  CPUTensor<float> left_cpu(left_shape);
  CPUTensor<float> right_cpu(right_shape);
  CPUTensor<float> output_cpu(output_shape);
  // set up gpu
  GPUTensor<float> left_gpu(left_shape);
  GPUTensor<float> right_gpu(right_shape);
  GPUTensor<float> output_gpu(output_shape);
  CPUTensor<float> output_copy(output_shape);
  // init values
  Backend<CPUTensor, float>::UniformDistributionFunc(&left_cpu, 0.0, 1.0);
  Backend<CPUTensor, float>::UniformDistributionFunc(&right_cpu, 0.0, 1.0);
  cudaMemcpy(left_gpu.data(), left_cpu.data(),
    left_cpu.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(right_gpu.data(), right_cpu.data(),
    right_cpu.size() * sizeof(float), cudaMemcpyHostToDevice);
  // transpose
  Backend<CPUTensor, float>::MatrixMultiplyFunc(&left_cpu, &right_cpu, &output_cpu, left_trans, right_trans, 1.0, 0.0, algorithm);
  Backend<GPUTensor, float>::MatrixMultiplyFunc(&left_gpu, &right_gpu, &output_gpu, left_trans, right_trans, 1.0, 0.0, algorithm);
  // copy from gpu to cpu
  cudaMemcpy(output_copy.data(), output_gpu.data(),
    output_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
  compare_cpu_gpu(output_cpu.size(), output_cpu.data(), output_copy.data());
}

int main(int argc, char** argv) {
  const size_t NUM_ARGS = 5;
  // M N K
  if (argc != NUM_ARGS + 1) {
    std::cerr << "Not matchable args!" << std::endl;
    exit(1);
  }
  std::string kernel = std::string(argv[1]);
  std::string trans = std::string(argv[2]);
  const size_t M = atoi(argv[3]);
  const size_t N = atoi(argv[4]);
  const size_t K = atoi(argv[5]);
  // run
  multiply(M, N, K, BlitzParseAlgorithm(kernel), trans);
  return 0;
}
