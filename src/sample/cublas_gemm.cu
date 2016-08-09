#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas.h>
#include <cudnn.h>
#include <curand_kernel.h>

#include "backend/backends.h"

using namespace blitz;

void no_transpose() {
  std::cout << "no transpose start" << std::endl;
  Shape left_shape(2);
  left_shape[0] = 2;
  left_shape[1] = 2;

  CPUTensor<float> left_cpu(left_shape);
  GPUTensor<float> left(left_shape);
  Backend<GPUTensor, float>::NormalDistributionFunc(0, 1, &left);
  cudaMemcpy(left_cpu.data(), left.data(), left.size() * sizeof(float),
    cudaMemcpyDeviceToHost);

  std::cout << "left: " << std::endl;
  size_t shape1 = left_cpu.shape()[0];
  size_t shape2 = left_cpu.shape()[1];
  for (size_t i = 0; i < shape1; ++i) {
    for (size_t j = 0; j < shape2; ++j) {
      std::cout << left_cpu[i * shape2 + j] << " ";
    }
    std::cout << std::endl;
  }

  Shape right_shape(2);
  right_shape[0] = 2;
  right_shape[1] = 2;
  CPUTensor<float> right_cpu(right_shape);
  GPUTensor<float> right(right_shape);
  Backend<GPUTensor, float>::NormalDistributionFunc(0, 1, &right);
  cudaMemcpy(right_cpu.data(), right.data(), right.size() * sizeof(float),
    cudaMemcpyDeviceToHost);

  std::cout << "right: " << std::endl;
  shape1 = right_cpu.shape()[0];
  shape2 = right_cpu.shape()[1];
  for (size_t i = 0; i < shape1; ++i) {
    for (size_t j = 0; j < shape2; ++j) {
      std::cout << right_cpu[i * shape2 + j] << " ";
    }
    std::cout << std::endl;
  }

  Shape output_shape(2);
  output_shape[0] = 2;
  output_shape[1] = 2;
  CPUTensor<float> output_cpu(output_shape);
  GPUTensor<float> output(output_shape);

  cublasOperation_t TransA = CUBLAS_OP_N;
  int lda = 2;
  cublasOperation_t TransB = CUBLAS_OP_N;
  int ldb = 2;
  int ldc = 2;
  float alpha = 1.0f, beta = 0.0f;
  cublasHandle_t handle;
  cublasStatus_t stat = cublasCreate_v2(&handle);
  cublasSgemm_v2(handle, TransA, TransB, 2, 2, 2, &alpha, left.data(), lda, right.data(), ldb, 
    &beta, output.data(), ldc); 

  cudaMemcpy(output_cpu.data(), output.data(), output.size() * sizeof(float),
    cudaMemcpyDeviceToHost);

  std::cout << "output: " << std::endl;
  shape1 = output_cpu.shape()[0];
  shape2 = output_cpu.shape()[1];
  for (size_t i = 0; i < shape1; ++i) {
    for (size_t j = 0; j < shape2; ++j) {
      std::cout << output_cpu[i * shape2 + j] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << "no transpose end" << std::endl;

}

int main() {
  no_transpose();

  return 0;
}
