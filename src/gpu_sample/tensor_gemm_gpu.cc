#include <iomanip>
#include <iostream>
#include <sstream>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "backend/backends.h"

using namespace blitz;

void left_transpose() {
  std::cout << "left transpose start" << std::endl;
  Shape left_shape(2);
  left_shape[0] = 4;
  left_shape[1] = 4;

  CPUTensor<float> left_cpu(left_shape);
  GPUTensor<float> left(left_shape);
  Backend<GPUTensor, float>::NormalDistributionFunc(0, 1, &left);
  //cudaMemcpy(left_cpu.data(), left.data(), left.size() * sizeof(float),
  //  cudaMemcpyDeviceToHost);

  //std::cout << "left: " << std::endl;
  //size_t shape1 = left_cpu.shape()[0];
  //size_t shape2 = left_cpu.shape()[1];
  //for (size_t i = 0; i < shape1; ++i) {
  //  for (size_t j = 0; j < shape2; ++j) {
  //    std::cout << left_cpu[i * shape2 + j] << " ";
  //  }
  //  std::cout << std::endl;
  //}

  Shape right_shape(2);
  right_shape[0] = 4;
  right_shape[1] = 4;
  CPUTensor<float> right_cpu(right_shape);
  GPUTensor<float> right(right_shape);
  Backend<GPUTensor, float>::NormalDistributionFunc(0, 1, &right);
  //cudaMemcpy(right_cpu.data(), right.data(), right.size() * sizeof(float),
  //  cudaMemcpyDeviceToHost);

  //std::cout << "right: " << std::endl;
  //shape1 = right_cpu.shape()[0];
  //shape2 = right_cpu.shape()[1];
  //for (size_t i = 0; i < shape1; ++i) {
  //  for (size_t j = 0; j < shape2; ++j) {
  //    std::cout << right_cpu[i * shape2 + j] << " ";
  //  }
  //  std::cout << std::endl;
  //}

  Shape output_shape(2);
  output_shape[0] = 4;
  output_shape[1] = 4;
  CPUTensor<float> output_cpu(output_shape);
  GPUTensor<float> output(output_shape);
  Backend<GPUTensor, float>::MatrixDotFunc(&left, &right, true, false, 1, 0, &output, "asm");
  //cudaMemcpy(output_cpu.data(), output.data(), output.size() * sizeof(float),
  //  cudaMemcpyDeviceToHost);

  //std::cout << "output: " << std::endl;
  //shape1 = output_cpu.shape()[0];
  //shape2 = output_cpu.shape()[1];
  //for (size_t i = 0; i < shape1; ++i) {
  //  for (size_t j = 0; j < shape2; ++j) {
  //    std::cout << output_cpu[i * shape2 + j] << " ";
  //  }
  //  std::cout << std::endl;
  //}
  std::cout << "left transpose end" << std::endl;
}

void right_transpose() {
  std::cout << "right transpose start" << std::endl;
  Shape left_shape(2);
  left_shape[0] = 2;
  left_shape[1] = 4;

  CPUTensor<float> left_cpu(left_shape);
  GPUTensor<float> left(left_shape);
  Backend<GPUTensor, float>::NormalDistributionFunc(0, 1, &left);
  //cudaMemcpy(left_cpu.data(), left.data(), left.size() * sizeof(float),
  //  cudaMemcpyDeviceToHost);

  //std::cout << "left: " << std::endl;
  //size_t shape1 = left_cpu.shape()[0];
  //size_t shape2 = left_cpu.shape()[1];
  //for (size_t i = 0; i < shape1; ++i) {
  //  for (size_t j = 0; j < shape2; ++j) {
  //    std::cout << left_cpu[i * shape2 + j] << " ";
  //  }
  //  std::cout << std::endl;
  //}

  Shape right_shape(2);
  right_shape[0] = 1;
  right_shape[1] = 4;
  CPUTensor<float> right_cpu(right_shape);
  GPUTensor<float> right(right_shape);
  Backend<GPUTensor, float>::NormalDistributionFunc(0, 1, &right);
  //cudaMemcpy(right_cpu.data(), right.data(), right.size() * sizeof(float),
  //  cudaMemcpyDeviceToHost);

  //std::cout << "right: " << std::endl;
  //shape1 = right_cpu.shape()[0];
  //shape2 = right_cpu.shape()[1];
  //for (size_t i = 0; i < shape1; ++i) {
  //  for (size_t j = 0; j < shape2; ++j) {
  //    std::cout << right_cpu[i * shape2 + j] << " ";
  //  }
  //  std::cout << std::endl;
  //}

  Shape output_shape(2);
  output_shape[0] = 2;
  output_shape[1] = 1;
  CPUTensor<float> output_cpu(output_shape);
  GPUTensor<float> output(right_shape);
  Backend<GPUTensor, float>::MatrixDotFunc(&left, &right, false, true, 1, 0, &output, "asm");
  //cudaMemcpy(output_cpu.data(), output.data(), right.size() * sizeof(float),
  //  cudaMemcpyDeviceToHost);

  //std::cout << "output: " << std::endl;
  //shape1 = output_cpu.shape()[0];
  //shape2 = output_cpu.shape()[1];
  //for (size_t i = 0; i < shape1; ++i) {
  //  for (size_t j = 0; j < shape2; ++j) {
  //    std::cout << output_cpu[i * shape2 + j] << " ";
  //  }
  //  std::cout << std::endl;
  //}
  std::cout << "right transpose end" << std::endl;
}

void both_transpose() {
  std::cout << "both transpose start" << std::endl;
  Shape left_shape(2);
  left_shape[0] = 1;
  left_shape[1] = 4;

  CPUTensor<float> left(left_shape);
  Backend<CPUTensor, float>::NormalDistributionFunc(0, 1, &left);

  std::cout << "left: " << std::endl;
  size_t shape1 = left.shape()[0];
  size_t shape2 = left.shape()[1];
  for (size_t i = 0; i < shape1; ++i) {
    for (size_t j = 0; j < shape2; ++j) {
      std::cout << left[i * shape2 + j] << " ";
    }
    std::cout << std::endl;
  }

  Shape right_shape(2);
  right_shape[0] = 4;
  right_shape[1] = 1;
  CPUTensor<float> right(right_shape);
  Backend<CPUTensor, float>::NormalDistributionFunc(0, 1, &right);
  std::cout << "right size " << right.shape().size() << std::endl;

  std::cout << "right: " << std::endl;
  shape1 = right.shape()[0];
  shape2 = right.shape()[1];
  for (size_t i = 0; i < shape1; ++i) {
    for (size_t j = 0; j < shape2; ++j) {
      std::cout << right[i * shape2 + j] << " ";
    }
    std::cout << std::endl;
  }

  Shape output_shape(2);
  output_shape[0] = 4;
  output_shape[1] = 4;
  CPUTensor<float> output(output_shape);
  Backend<CPUTensor, float>::MatrixDotFunc(&left, &right, true, true, 1, 0, &output);

  std::cout << "output: " << std::endl;
  shape1 = output.shape()[0];
  shape2 = output.shape()[1];
  for (size_t i = 0; i < shape1; ++i) {
    for (size_t j = 0; j < shape2; ++j) {
      std::cout << output[i * shape2 + j] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << "both transpose end" << std::endl;
}

void no_transpose() {
  std::cout << "no transpose start" << std::endl;
  Shape left_shape(2);
  left_shape[0] = 1000;
  left_shape[1] = 1000;

  CPUTensor<float> left_cpu(left_shape);
  GPUTensor<float> left_gpu(left_shape);
  Backend<GPUTensor, float>::NormalDistributionFunc(0, 1, &left_gpu);
  cudaMemcpy(left_cpu.data(), left_gpu.data(), left_cpu.size() * sizeof(float),
    cudaMemcpyDeviceToHost);

  std::cout << "left: " << std::endl;
  //size_t shape1 = left_cpu.shape()[0];
  //size_t shape2 = left_cpu.shape()[1];
  //for (size_t i = 0; i < shape1; ++i) {
  //  for (size_t j = 0; j < shape2; ++j) {
  //    std::cout << left_cpu[i * shape2 + j] << " ";
  //  }
  //  std::cout << std::endl;
  //}

  Shape right_shape(2);
  right_shape[0] = 1000;
  right_shape[1] = 1000;
  CPUTensor<float> right_cpu(right_shape);
  GPUTensor<float> right_gpu(right_shape);
  Backend<GPUTensor, float>::NormalDistributionFunc(0, 1, &right_gpu);
  std::cout << "right size " << right_cpu.shape().size() << std::endl;
  cudaMemcpy(right_cpu.data(), right_gpu.data(), right_cpu.size() * sizeof(float),
    cudaMemcpyDeviceToHost);

  std::cout << "right: " << std::endl;
  //shape1 = right_cpu.shape()[0];
  //shape2 = right_cpu.shape()[1];
  //for (size_t i = 0; i < shape1; ++i) {
  //  for (size_t j = 0; j < shape2; ++j) {
  //    std::cout << right_cpu[i * shape2 + j] << " ";
  //  }
  //  std::cout << std::endl;
  //}

  Shape output_shape(2);
  output_shape[0] = 1000;
  output_shape[1] = 1000;
  CPUTensor<float> output_cpu(output_shape);
  GPUTensor<float> output_gpu(output_shape);
  time_point<system_clock> start, end;
  duration<double> time = duration<double>::zero();
  start = system_clock::now();
  Backend<GPUTensor, float>::MatrixDotFunc(&left_gpu, &right_gpu,
    false, false, 1, 0, &output_gpu, "asm");
  end = system_clock::now();
  time += end - start;
  LOG(INFO) << "Running time: " << time.count();
  cudaMemcpy(output_cpu.data(), output_gpu.data(), output_cpu.size() * sizeof(float),
    cudaMemcpyDeviceToHost);
  cudaMemcpy(left_cpu.data(), left_gpu.data(), left_cpu.size() * sizeof(float),
    cudaMemcpyDeviceToHost);
  cudaMemcpy(right_cpu.data(), right_gpu.data(), right_cpu.size() * sizeof(float),
    cudaMemcpyDeviceToHost);

  std::cout << "left: " << std::endl;
  //shape1 = left_cpu.shape()[0];
  //shape2 = left_cpu.shape()[1];
  //for (size_t i = 0; i < shape1; ++i) {
  //  for (size_t j = 0; j < shape2; ++j) {
  //    std::cout << left_cpu[i * shape2 + j] << " ";
  //  }
  //  std::cout << std::endl;
  //}

  std::cout << "right: " << std::endl;
  //shape1 = right_cpu.shape()[0];
  //shape2 = right_cpu.shape()[1];
  //for (size_t i = 0; i < shape1; ++i) {
  //  for (size_t j = 0; j < shape2; ++j) {
  //    std::cout << right_cpu[i * shape2 + j] << " ";
  //  }
  //  std::cout << std::endl;
  //}

  std::cout << "output: " << std::endl;
  //shape1 = output_cpu.shape()[0];
  //shape2 = output_cpu.shape()[1];
  //for (size_t i = 0; i < shape1; ++i) {
  //  for (size_t j = 0; j < shape2; ++j) {
  //    std::cout << output_cpu[i * shape2 + j] << " ";
  //  }
  //  std::cout << std::endl;
  //}
  std::cout << "no transpose end" << std::endl;
}

int main()
{
  std::cout << "start" << std::endl;

  for (int i = 0; i < 10; ++i)
    no_transpose();
  //left_transpose();
  //right_transpose();

  std::cout << "end" << std::endl;
  return 0;
}
