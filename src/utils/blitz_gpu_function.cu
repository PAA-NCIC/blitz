#include "utils/blitz_gpu_function.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas.h>
#include <curand_kernel.h>

namespace blitz {

boost::scoped_ptr<cublasHandle_t> CuBlasHandle::instance_(0);
boost::once_flag CuBlasHandle::flag_ = BOOST_ONCE_INIT;

template<>
void BlitzGPUTrans(float* input, float* output, size_t M, size_t N) {
  float alpha = 1.0;
  float beta = 0.0;
  cublasSgeam(CuBlasHandle::GetInstance(), CUBLAS_OP_T, CUBLAS_OP_N,
    M, N, &alpha, input, N, &beta, input, M, output, M);
}

template<>
void BlitzGPUTrans(double* input, double* output, size_t M, size_t N) {
  double alpha = 1.0;
  double beta = 0.0;
  cublasDgeam(CuBlasHandle::GetInstance(), CUBLAS_OP_T, CUBLAS_OP_N,
    M, N, &alpha, input, N, &beta, input, M, output, M);
}

template<>
float BlitzGPUASum(const float* data, size_t N) {
  float sum = 0.0;
  cublasSasum_v2(CuBlasHandle::GetInstance(), N, data, 1, &sum);
  return sum;
}

template<>
double BlitzGPUASum(const double* data, size_t N) {
  double sum = 0.0;
  cublasDasum_v2(CuBlasHandle::GetInstance(), N, data, 1, &sum);
  return sum;
}

template<>
void BlitzGenerateNormal(curandGenerator_t* gen, float* data,
  float loc, float scale, size_t size) {
  curandGenerateNormal(*gen, data, size, loc, scale);
}

template<>
void BlitzGenerateNormal(curandGenerator_t* gen, double* data,
  double loc, double scale, size_t size) {
  curandGenerateNormalDouble(*gen, data, size, loc, scale);
}

template<>
void BlitzGenerateUniform(curandGenerator_t* gen, float* data, size_t size) {
  curandGenerateUniform(*gen, data, size);
}

template<>
void BlitzGenerateUniform(curandGenerator_t* gen, double* data, size_t size) {
  curandGenerateUniformDouble(*gen, data, size);
}

}  // namespace blitz

