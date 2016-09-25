#include <iomanip>
#include <iostream>
#include <sstream>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "backend/backends.h"

using namespace blitz;

/*
 * 0 1 0
 * 1 1 0
 * 0 1 0
 */
void init_input(CPUTensor<float>& input) {
  input[0] = 0;
  input[1] = 1;
  input[2] = 0;
  input[3] = 1;
  input[4] = 1;
  input[5] = 0;
  input[6] = 0;
  input[7] = 1;
  input[8] = 0;
}

/*
 * 0 1 0 1
 * 1 1 0 1
 * 0 1 0 0
 * 1 0 1 1
 */
void init_pack(CPUTensor<float>& pack) {
  pack[0] = 0;
  pack[1] = 1;
  pack[2] = 0;
  pack[3] = 1;
  pack[4] = 1;
  pack[5] = 1;
  pack[6] = 0;
  pack[7] = 1;
  pack[8] = 0;
  pack[9] = 1;
  pack[10] = 0;
  pack[11] = 0;
  pack[12] = 1;
  pack[13] = 0;
  pack[14] = 1;
  pack[15] = 1;
}

/* expect unpack
 * 0 1 1 1
 * 1 0 1 0
 * 1 1 0 1
 * 1 0 1 0
 */
void unpack_input() {
  Shape input_shape(4);
  input_shape[0] = 1;
  input_shape[1] = 1;
  input_shape[2] = 3;
  input_shape[3] = 3;
  CPUTensor<float> input(input_shape);
  init_input(input);
  GPUTensor<float> input_gpu(input_shape);
  cudaMemcpy(input_gpu.data(), input.data(), input.size() * sizeof(float),
    cudaMemcpyHostToDevice);

  Shape filter_shape(4); 
  filter_shape[0] = 1;
  filter_shape[1] = 1;
  filter_shape[2] = 2;
  filter_shape[3] = 2;

  int stride_height = 1;
  int stride_width = 1;
  int padding_height = 0;
  int padding_width = 0;

  Shape unpack_shape(2);
  unpack_shape[0] = 2 * 2;
  unpack_shape[1] = 1 * 2 * 2;
  CPUTensor<float> unpack(unpack_shape);
  GPUTensor<float> unpack_gpu(unpack_shape);

  Backend<GPUTensor, float>::Unpack2DFunc(input_gpu.data(), input_shape[1],
    input_shape[2], input_shape[3], filter_shape[2], filter_shape[3], 2,
    2, padding_height, padding_width, stride_height, stride_width, unpack_gpu.data());

  cudaMemcpy(unpack.data(), unpack_gpu.data(), unpack.size() * sizeof(float),
    cudaMemcpyDeviceToHost);

  std::cout << "input: " << std::endl;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      std::cout << input[i * 3 + j];
    }
    std::cout << std::endl;
  }

  std::cout << "unpack: " << std::endl;
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      std::cout << unpack[i * 4 + j];
    }
    std::cout << std::endl;
  }
}

/*
 * 0 1 0 1
 * 1 1 0 1
 * 0 1 0 0
 * 1 0 1 1
 */

/* 
 * expect input
 * 0 2 1
 * 0 3 1
 * 0 1 1
 */
void pack_input() {
  Shape unpack_shape(2);
  unpack_shape[0] = 2 * 2;
  unpack_shape[1] = 1 * 2 * 2;
  CPUTensor<float> pack(unpack_shape);
  init_pack(pack);
  GPUTensor<float> pack_gpu(unpack_shape);
  cudaMemcpy(pack_gpu.data(), pack.data(), pack.size() * sizeof(float),
    cudaMemcpyHostToDevice);

  Shape filter_shape(4); 
  filter_shape[0] = 1;
  filter_shape[1] = 1;
  filter_shape[2] = 2;
  filter_shape[3] = 2;

  int stride_height = 1;
  int stride_width = 1;
  int padding_height = 0;
  int padding_width = 0;

  Shape input_shape(4);
  input_shape[0] = 1;
  input_shape[1] = 1;
  input_shape[2] = 3;
  input_shape[3] = 3;
  CPUTensor<float> input(input_shape);
  GPUTensor<float> input_gpu(input_shape);

  Backend<GPUTensor, float>::Pack2DFunc(pack_gpu.data(), input_shape[1],
    input_shape[2], input_shape[3], filter_shape[2], filter_shape[3], 2,
    2, padding_height, padding_width, stride_height, stride_width, input_gpu.data());
  cudaMemcpy(input.data(), input_gpu.data(), input.size() * sizeof(float),
    cudaMemcpyDeviceToHost);

  std::cout << "input: " << std::endl;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      std::cout << input[i * 3 + j];
    }
    std::cout << std::endl;
  }

  std::cout << "pack: " << std::endl;
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      std::cout << pack[i * 4 + j];
    }
    std::cout << std::endl;
  }
}

int main() {
  unpack_input();
  pack_input();
  return 0;
}
