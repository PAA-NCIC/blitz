#include <iomanip>
#include <iostream>
#include <sstream>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include "sys/time.h"

#include "backend/backends.h"

using namespace blitz;

void left_transpose(const string& kernel) {
  std::cout << "left transpose start" << std::endl;
  Shape left_shape(2);
  left_shape[0] = 2048;
  left_shape[1] = 2048;

  CPUTensor<float> left_cpu(left_shape);
  GPUTensor<float> left_gpu(left_shape);
  Backend<GPUTensor, float>::NormalDistributionFunc(0, 1, &left_gpu);
  cudaMemcpy(left_cpu.data(), left_gpu.data(), left_cpu.size() * sizeof(float),
    cudaMemcpyDeviceToHost);

  Shape right_shape(2);
  right_shape[0] = 2048;
  right_shape[1] = 2048;
  CPUTensor<float> right_cpu(right_shape);
  GPUTensor<float> right_gpu(right_shape);
  Backend<GPUTensor, float>::NormalDistributionFunc(0, 1, &right_gpu);
  cudaMemcpy(right_cpu.data(), right_gpu.data(), right_cpu.size() * sizeof(float),
    cudaMemcpyDeviceToHost);

  Shape output_shape(2);
  output_shape[0] = 2048;
  output_shape[1] = 2048;
  CPUTensor<float> output_cpu(output_shape);
  GPUTensor<float> output_gpu(output_shape);
  CPUTensor<float> output_copy(output_shape);
  timeval t1, t2; 
  double elapsed_time = 0.0f;

  cudaDeviceSynchronize();
  gettimeofday(&t1, NULL);
  Backend<GPUTensor, float>::MatrixDotFunc(&left_gpu, &right_gpu,
    true, false, 1, 0, &output_gpu, kernel);

  gettimeofday(&t2, NULL);
  elapsed_time = (t2.tv_sec - t1.tv_sec) * 1000.0;
  elapsed_time += (t2.tv_usec - t1.tv_usec) / 1000.0;
  elapsed_time /= 1000.0;
  std::cout << "GPU running time: " << elapsed_time << std::endl;

  cudaDeviceSynchronize();
  cudaMemcpy(output_copy.data(), output_gpu.data(), output_cpu.size() * sizeof(float),
    cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  gettimeofday(&t1, NULL);

  Backend<CPUTensor, float>::MatrixDotFunc(&left_cpu, &right_cpu,
    true, false, 1, 0, &output_cpu);

  gettimeofday(&t2, NULL);
  elapsed_time = (t2.tv_sec - t1.tv_sec) * 1000.0;
  elapsed_time += (t2.tv_usec - t1.tv_usec) / 1000.0;
  elapsed_time /= 1000.0;
  std::cout << "CPU running time: " << elapsed_time << std::endl;

  for (size_t i = 0; i < output_cpu.size(); ++i) {
    if (!(output_copy[i] <= output_cpu[i] + 1e-3 && output_copy[i] >= output_cpu[i] - 1e-3)) {
      std::cout << "index: " << i << " gpu: " <<
        output_copy[i] << " cpu: " << output_cpu[i] << std::endl;
    }
  }
}

void right_transpose(const string& kernel) {
  std::cout << "right transpose start" << std::endl;
  Shape left_shape(2);
  left_shape[0] = 2048;
  left_shape[1] = 2048;

  CPUTensor<float> left_cpu(left_shape);
  GPUTensor<float> left_gpu(left_shape);
  Backend<GPUTensor, float>::NormalDistributionFunc(0, 1, &left_gpu);
  cudaMemcpy(left_cpu.data(), left_gpu.data(), left_cpu.size() * sizeof(float),
    cudaMemcpyDeviceToHost);

  Shape right_shape(2);
  right_shape[0] = 2048;
  right_shape[1] = 2048;
  CPUTensor<float> right_cpu(right_shape);
  GPUTensor<float> right_gpu(right_shape);
  Backend<GPUTensor, float>::NormalDistributionFunc(0, 1, &right_gpu);
  cudaMemcpy(right_cpu.data(), right_gpu.data(), right_cpu.size() * sizeof(float),
    cudaMemcpyDeviceToHost);

  Shape output_shape(2);
  output_shape[0] = 2048;
  output_shape[1] = 2048;
  CPUTensor<float> output_cpu(output_shape);
  GPUTensor<float> output_gpu(output_shape);
  CPUTensor<float> output_copy(output_shape);
  time_point<system_clock> start, end;
  duration<double> time = duration<double>::zero();
  start = system_clock::now();
  Backend<GPUTensor, float>::MatrixDotFunc(&left_gpu, &right_gpu,
    false, true, 1, 0, &output_gpu, kernel);
  cudaDeviceSynchronize();
  end = system_clock::now();
  time = end - start;
  std::cout << "GPU running time: " << time.count() << std::endl;
  cudaMemcpy(output_copy.data(), output_gpu.data(), output_cpu.size() * sizeof(float),
    cudaMemcpyDeviceToHost);

  start = system_clock::now();
  Backend<CPUTensor, float>::MatrixDotFunc(&left_cpu, &right_cpu,
    false, true, 1, 0, &output_cpu);
  end = system_clock::now();
  time = end - start;
  std::cout << "CPU running time: " << time.count() << std::endl;

  for (size_t i = 0; i < output_cpu.size(); ++i) {
    if (!(output_copy[i] <= output_cpu[i] + 1e-3 && output_copy[i] >= output_cpu[i] - 1e-3)) {
      std::cout << "index: " << i << " gpu: " <<
        output_copy[i] << " cpu: " << output_cpu[i] << std::endl;
    }
  }
}

void both_transpose() {
  std::cout << "both transpose start" << std::endl;
  Shape left_shape(2);
  left_shape[0] = 1024;
  left_shape[1] = 1024;

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
  right_shape[0] = 1024;
  right_shape[1] = 1024;
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
  output_shape[0] = 1024;
  output_shape[1] = 1024;
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

void no_transpose(const string& kernel) {
  std::cout << "no transpose start" << std::endl;
  Shape left_shape(2);
  left_shape[0] = 4096;
  left_shape[1] = 4096;

  CPUTensor<float> left_cpu(left_shape);
  GPUTensor<float> left_gpu(left_shape);
  Backend<GPUTensor, float>::ConstantDistributionFunc(2, &left_gpu);
  cudaMemcpy(left_cpu.data(), left_gpu.data(), left_cpu.size() * sizeof(float),
    cudaMemcpyDeviceToHost);

  Shape right_shape(2);
  right_shape[0] = 4096;
  right_shape[1] = 4096;
  CPUTensor<float> right_cpu(right_shape);
  GPUTensor<float> right_gpu(right_shape);
  Backend<GPUTensor, float>::ConstantDistributionFunc(2, &right_gpu);
  cudaMemcpy(right_cpu.data(), right_gpu.data(), right_cpu.size() * sizeof(float),
    cudaMemcpyDeviceToHost);

  Shape output_shape(2);
  output_shape[0] = 4096;
  output_shape[1] = 4096;
  CPUTensor<float> output_cpu(output_shape);
  GPUTensor<float> output_gpu(output_shape);
  CPUTensor<float> output_copy(output_shape);
  time_point<system_clock> start, end;
  duration<double> time = duration<double>::zero();
  start = system_clock::now();
  Backend<GPUTensor, float>::MatrixDotFunc(&left_gpu, &right_gpu,
    false, false, 1, 0, &output_gpu, kernel);
  cudaDeviceSynchronize();
  end = system_clock::now();
  time = end - start;
  std::cout << "GPU running time: " << time.count() << std::endl;
  cudaMemcpy(output_copy.data(), output_gpu.data(), output_cpu.size() * sizeof(float),
    cudaMemcpyDeviceToHost);

  start = system_clock::now();
  Backend<CPUTensor, float>::MatrixDotFunc(&left_cpu, &right_cpu,
    false, false, 1, 0, &output_cpu);
  end = system_clock::now();
  time = end - start;
  std::cout << "CPU running time: " << time.count() << std::endl;

  for (size_t i = 0; i < output_cpu.size(); ++i) {
    if (!(output_copy[i] <= output_cpu[i] + 1e-3 && output_copy[i] >= output_cpu[i] - 1e-3)) {
      std::cout << "index: " << i << " gpu: " <<
        output_copy[i] << " cpu: " << output_cpu[i] << std::endl;
    }
  }
}

int main() {
  std::cout << "start" << std::endl;

  cudaFree(0);
  for (int i = 0; i < 10; ++i)
    no_transpose("asm");
  cudaFree(0);
  //for (int i = 0; i < 10; ++i)
  //  left_transpose("blas");
  //for (int i = 0; i < 10; ++i)
  //  right_transpose("blas");

  std::cout << "end" << std::endl;
  return 0;
}
