#include <iostream>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "backend/backends.h"
#include "util/blitz_gpu_function.h"

using namespace blitz;

/*
 * 0 1 1 0
 * 0 1 0 0
 * 0 1 0 0
 */
void init_input(CPUTensor<float>& input) {
  input[0] = 1;
  input[1] = 2;
  input[2] = 3;
  input[3] = 4;
  input[4] = 5;
  input[5] = 6;
  input[6] = 7;
  input[7] = 8;
  input[8] = 9;
  input[9] = 10;
  input[10] = 11;
  input[11] = 12;
}

void transpose() {
  Shape input_shape(2);
  // batch_size
  input_shape[0] = 128;
  // input channel
  input_shape[1] = 1000;
  GPUTensor<float> input_gpu(input_shape);
  GPUTensor<float> input_gpu_trans(input_shape);
  Backend<GPUTensor, float>::ConstantDistributionFunc(2, &input_gpu);

  time_point<system_clock> start, end;
  duration<double> time = duration<double>::zero();
  start = system_clock::now();
  cudaDeviceSynchronize();

  BlitzGPUTrans(128, 100, input_gpu.data(), input_gpu_trans.data());

  cudaDeviceSynchronize();
  end = system_clock::now();
  time = end - start;
  std::cout << "GPU running time: " << time.count() << std::endl;
}

int main() {
  Shape input_shape(4);
  // batch_size
  input_shape[0] = 1;
  // input channel
  input_shape[1] = 1;
  // input height
  input_shape[2] = 3;
  // input width
  input_shape[3] = 4;
  CPUTensor<float> input(input_shape);
  GPUTensor<float> input_gpu(input_shape);
  GPUTensor<float> input_gpu_trans(input_shape);
  init_input(input);
  cudaMemcpy(input_gpu.data(), input.data(),
    input.size() * sizeof(float), cudaMemcpyHostToDevice);
  BlitzGPUTrans(3, 4, input_gpu.data(), input_gpu_trans.data());
  cudaMemcpy(input.data(), input_gpu_trans.data(),
    input.size() * sizeof(float), cudaMemcpyDeviceToHost);

/*
 * 0 0 0
 * 1 1 1
 * 1 0 0
 * 0 0 0
 */
  //for (int i = 0; i < 4; ++i) {
  //  for (int j = 0; j < 3; ++j) {
  //    std::cout << input[i * 3 + j] << " ";
  //  }
  //  std::cout << std::endl;
  //}
  for (int i = 0; i < 10; ++i) {
    transpose();
  }

  return 0;
}
