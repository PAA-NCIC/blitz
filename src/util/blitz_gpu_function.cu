#include "util/blitz_gpu_function.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas.h>
#include <cudnn.h>
#include <curand_kernel.h>

namespace blitz {

boost::scoped_ptr<cublasHandle_t> CuBlasHandle::instance_(0);
boost::once_flag CuBlasHandle::flag_ = BOOST_ONCE_INIT;

template<>
void BlitzGPUGemm(const bool transa, const bool transb,
  const int M, const int N, const int K,
  float* A, float* B, float* C, float alpha, float beta) {
  cublasOperation_t TransA = transa ? CUBLAS_OP_T : CUBLAS_OP_N;
  int lda = transa ? M : K;
  cublasOperation_t TransB = transb ? CUBLAS_OP_T : CUBLAS_OP_N;
  int ldb = transb ? K : N;
  cublasSgemm_v2(CuBlasHandle::GetInstance(), TransB, TransA, N, M, K,
    &alpha, B, ldb, A, lda, &beta, C, N);
}

template<>
void BlitzGPUGemm(const bool transa, const bool transb,
  const int M, const int N, const int K,
  double* A, double* B, double* C, double alpha, double beta) {
  cublasOperation_t TransA = transa ? CUBLAS_OP_T : CUBLAS_OP_N;
  int lda = transa ? M : K;
  cublasOperation_t TransB = transb ? CUBLAS_OP_T : CUBLAS_OP_N;
  int ldb = transb ? K : N;
  cublasDgemm_v2(CuBlasHandle::GetInstance(), TransB, TransA, N, M, K,
    &alpha, B, ldb, A, lda, &beta, C, N);
}

template<>
void BlitzGPUTrans(const int M, const int N, float* input, float* output) {
  const float alpha = 1.0f;
  const float beta = 0.0f;
  cublasSgeam(CuBlasHandle::GetInstance(), CUBLAS_OP_T, CUBLAS_OP_N,
    M, N, &alpha, input, N, &beta, input, M, output, M);
}

template<>
void BlitzGPUTrans(const int M, const int N, double* input, double* output) {
  const double alpha = 1.0f;
  const double beta = 0.0f;
  cublasDgeam(CuBlasHandle::GetInstance(), CUBLAS_OP_T, CUBLAS_OP_N,
    M, N, &alpha, input, N, &beta, input, M, output, M);
}

template<>
float BlitzGPUASum(const int N, const float* data) {
  cublasHandle_t handle;
  cublasStatus_t stat = cublasCreate_v2(&handle);
  float sum = 0.0f;
  cublasSasum_v2(handle, N, data, 1, &sum);
  return sum;
}

template<>
double BlitzGPUASum(const int N, const double* data) {
  cublasHandle_t handle;
  cublasStatus_t stat = cublasCreate_v2(&handle);
  double sum = 0.0f;
  cublasDasum_v2(handle, N, data, 1, &sum);
  return sum;
}

template<>
void BlitzGenerateNormal(curandGenerator_t* gen, float* data,
  const int size, const float loc, const float scale) {
  curandGenerateNormal(*gen, data, size, loc, scale);
}

template<>
void BlitzGenerateNormal(curandGenerator_t* gen, double* data,
  const int size, const double loc, const double scale) {
  curandGenerateNormalDouble(*gen, data, size, loc, scale);
}

template<>
void BlitzGenerateUniform(curandGenerator_t* gen,
  float* data, const int size) {
  curandGenerateUniform(*gen, data, size);
}

template<>
void BlitzGenerateUniform(curandGenerator_t* gen,
  double* data, const int size) {
  curandGenerateUniformDouble(*gen, data, size);
}

}  // namespace blitz

