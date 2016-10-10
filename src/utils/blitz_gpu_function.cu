#include "utils/blitz_gpu_function.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas.h>
#include <cudnn.h>
#include <curand_kernel.h>

namespace blitz {

boost::scoped_ptr<cublasHandle_t> CuBlasHandle::instance_(0);
boost::once_flag CuBlasHandle::flag_ = BOOST_ONCE_INIT;

template<>
void BlitzGPUGemm(
  bool transa, bool transb,
  int M, int N, int K,
  float* A, float* B, float* C,
  float alpha, float beta) {
  cublasOperation_t TransA = transa ? CUBLAS_OP_T : CUBLAS_OP_N;
  int lda = transa ? M : K;
  cublasOperation_t TransB = transb ? CUBLAS_OP_T : CUBLAS_OP_N;
  int ldb = transb ? K : N;
  cublasSgemm_v2(CuBlasHandle::GetInstance(), TransB, TransA, N, M, K,
    &alpha, B, ldb, A, lda, &beta, C, N);
}

template<>
void BlitzGPUGemm(
  bool transa, bool transb,
  int M, int N, int K,
  double* A, double* B, double* C,
  double alpha, double beta) {
  cublasOperation_t TransA = transa ? CUBLAS_OP_T : CUBLAS_OP_N;
  int lda = transa ? M : K;
  cublasOperation_t TransB = transb ? CUBLAS_OP_T : CUBLAS_OP_N;
  int ldb = transb ? K : N;
  cublasDgemm_v2(CuBlasHandle::GetInstance(), TransB, TransA, N, M, K,
    &alpha, B, ldb, A, lda, &beta, C, N);
}

template<>
void BlitzGPUTrans(int M, int N, float* input, float* output) {
  const float alpha = 1.0;
  const float beta = 0.0;
  cublasSgeam(CuBlasHandle::GetInstance(), CUBLAS_OP_T, CUBLAS_OP_N,
    M, N, &alpha, input, N, &beta, input, M, output, M);
}

template<>
void BlitzGPUTrans(int M, int N, double* input, double* output) {
  const double alpha = 1.0;
  const double beta = 0.0;
  cublasDgeam(CuBlasHandle::GetInstance(), CUBLAS_OP_T, CUBLAS_OP_N,
    M, N, &alpha, input, N, &beta, input, M, output, M);
}

template<>
float BlitzGPUASum(int N, const float* data) {
  float sum = 0.0;
  cublasSasum_v2(CuBlasHandle::GetInstance(), N, data, 1, &sum);
  return sum;
}

template<>
double BlitzGPUASum(int N, const double* data) {
  double sum = 0.0;
  cublasDasum_v2(CuBlasHandle::GetInstance(), N, data, 1, &sum);
  return sum;
}

template<>
void BlitzGenerateNormal(curandGenerator_t* gen, float* data,
  int size, float loc, float scale) {
  curandGenerateNormal(*gen, data, size, loc, scale);
}

template<>
void BlitzGenerateNormal(curandGenerator_t* gen, double* data,
  int size, double loc, double scale) {
  curandGenerateNormalDouble(*gen, data, size, loc, scale);
}

template<>
void BlitzGenerateUniform(curandGenerator_t* gen,
  float* data, int size) {
  curandGenerateUniform(*gen, data, size);
}

template<>
void BlitzGenerateUniform(curandGenerator_t* gen,
  double* data, int size) {
  curandGenerateUniformDouble(*gen, data, size);
}

}  // namespace blitz

