#include <cuda.h>
#include <cuda_runtime_api.h>
#include <blitz.h>
#include "../../../include/kernels/sass_function.h"

using namespace blitz;

// K C R S
Shape filter_shape(4);
// K R S C
Shape filter_shuffle_shape(4);

void set_filter_shape_kcrs(size_t K, size_t C, size_t R, size_t S) {
  filter_shape[0] = K;
  filter_shape[1] = C;
  filter_shape[2] = R;
  filter_shape[3] = S;
}

void set_filter_shuffle_shape_krsc(size_t K, size_t R, size_t S, size_t C) {
  filter_shuffle_shape[0] = K;
  filter_shuffle_shape[1] = R;
  filter_shuffle_shape[2] = S;
  filter_shuffle_shape[3] = C;
}

// cpu shuffle
void cpu_shuffle(size_t K, size_t C, size_t R, size_t S,
  const float* input,
  float* output) {
  const size_t SC = S * C;
  const size_t RS = R * S;
  const size_t RSC = R * SC;
  for (size_t i = 0; i < K; ++i) {
    for (size_t j = 0; j < R; ++j) {
      for (size_t k = 0; k < S; ++k) {
        for (size_t v = 0; v < C; ++v) {
          output[i * RSC + j * SC + k * C + v] = 
            input[i * RSC + v * RS + j * S + k];
        }
      }
    }
  }
}

void compare(float* output_cpu, float* output_gpu, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    if (output_cpu[i] > output_gpu[i] + 1e-2 ||
      output_cpu[i] < output_gpu[i] - 1e-2) {
      std::cout << "Index: " << i << ", CPU: " << output_cpu[i] <<
        ", GPU: " << output_gpu[i] << std::endl;
    }
  }
}

void shuffle(size_t K, size_t C, size_t R, size_t S) {
  // set up cpu
  CPUTensor<float> filter_cpu(filter_shape);
  CPUTensor<float> filter_shuffle_cpu(filter_shuffle_shape);
  // set up gpu
  GPUTensor<float> filter_gpu(filter_shape);
  GPUTensor<float> filter_shuffle_gpu(filter_shuffle_shape);
  // set up copy
  CPUTensor<float> filter_copy(filter_shuffle_shape);
  // init values
  Backend<CPUTensor, float>::UniformDistributionFunc(&filter_cpu, 0.0, 1.0);
  cudaMemcpy(filter_gpu.data(), filter_cpu.data(),
    filter_cpu.size() * sizeof(float), cudaMemcpyHostToDevice);
  // cpu shuffle 
  cpu_shuffle(K, C, R, S,
    filter_cpu.data(),
    filter_shuffle_cpu.data());
  // gpu shuffle
  BlitzFilter2DShuffle<float>(filter_gpu.data(), filter_shuffle_gpu.data(),
    K, C, R, S);
  // copy from gpu to cpu
  cudaMemcpy(filter_copy.data(), filter_shuffle_gpu.data(),
    filter_shuffle_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
  compare(filter_shuffle_cpu.data(), filter_copy.data(), filter_shuffle_cpu.size());
}

int main(int argc, char** argv) {
  const size_t NUM_ARGS = 4;
  // K C R S
  if (argc != NUM_ARGS + 1) {
    std::cerr << "Not enough args!" << std::endl;
    exit(1);
  }
  // args
  const size_t K = atoi(argv[1]);
  const size_t C = atoi(argv[2]);
  const size_t R = atoi(argv[3]);
  const size_t S = atoi(argv[4]);
  // set shapes
  set_filter_shape_kcrs(K, C, R, S);
  set_filter_shuffle_shape_krsc(K, R, S, C);
  // run
  shuffle(K, C, R, S);
  return 0;
}
