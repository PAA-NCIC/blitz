#ifndef INCLUDE_UTIL_BLITZ_GPU_FUNCTION_H_
#define INCLUDE_UTIL_BLITZ_GPU_FUNCTION_H_

#include <cublas.h>
#include <cuda.h>
#include <curand_kernel.h>

#include <boost/scoped_ptr.hpp>
#include <boost/thread/once.hpp>
#include <boost/noncopyable.hpp>

#include "utils/common.h"

namespace blitz {

#define BLITZ_GPU_TIMER_START(elapsed_time, event_start, event_stop) \
  do { \
    elapsed_time = 0.0; \
    cuEventCreate(&event_start, CU_EVENT_BLOCKING_SYNC); \
    cuEventCreate(&event_stop, CU_EVENT_BLOCKING_SYNC); \
    cuEventRecord(event_start, NULL); \
  } while (0) 

#define BLITZ_GPU_TIMER_END(elapsed_time, event_start, event_stop) \
  do { \
    cuEventRecord(event_stop, NULL); \
    cuEventSynchronize(event_stop); \
    cuEventElapsedTime(&elapsed_time, event_start, event_stop); \
    elapsed_time /= 1000.0; \
  } while (0)

#define BLITZ_GPU_TIMER_INFO(computations, elapsed_time) \
  do { \
    LOG(INFO) << "Running time: " << elapsed_time; \
    LOG(INFO) << "Gflops: " << computations / (elapsed_time * 1e9); \
  } while (0) \

#if __CUDA_ARCH__ >= 200
  #define BLITZ_NUM_GPU_THREADS 1024
#else
  #define BLITZ_NUM_GPU_THREADS 512
#endif

// grid stride looping
// https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
#define BLITZ_CUDA_LOOP(i, n) \
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; \
      i < (n); \
      i += blockDim.x * gridDim.x)

namespace utils {

class CuBlasHandle {
 public:
  static cublasHandle_t& GetInstance() {
    // thread safe
    boost::call_once(&CuBlasHandle::Create, flag_);
    return *(CuBlasHandle::instance_);
  }

  static void Create() {
    CuBlasHandle::instance_.reset(new cublasHandle_t());
    cublasCreate_v2(CuBlasHandle::instance_.get());
  }

  virtual ~CuBlasHandle();

 private:
  CuBlasHandle();

  static boost::scoped_ptr<cublasHandle_t> instance_;
  static boost::once_flag flag_;

  DISABLE_COPY_AND_ASSIGN(CuBlasHandle);
};

inline size_t GPUGetBlocks(size_t N) {
  return (N + BLITZ_NUM_GPU_THREADS - 1) / BLITZ_NUM_GPU_THREADS;
}

inline size_t GPUGetBlocks(size_t N, size_t nthreads) {
  return (N + nthreads - 1) / nthreads;
}

template <typename DType>
inline __device__ DType GPUSafeLog(DType input) {
  return log(input > exp(-100.0) ? input : exp(-100.0));
}

template<typename DType>
void GPUTrans(DType* input, DType* output, size_t M, size_t N);

template<typename DType>
DType GPUASum(const DType* data, size_t N);

template<typename DType>
void GenerateNormal(curandGenerator_t* gen, DType* data,
  DType loc, DType scale, size_t size);

template<typename DType>
void GenerateUniform(curandGenerator_t* gen, DType* data, size_t size);

template<typename DType>
__global__ void GPURectlinApply(const DType* input, DType* output, DType compare_value, DType slope, size_t size);

template <typename DType>
__global__ void GPURectlinDerivative(const DType* input, DType* output, DType compare_value, DType slope, size_t size);

template <typename DType>
__global__ void GPUSoftmaxApply(const DType* input, DType* output, size_t batch_size, size_t dim);

template <typename DType>
__global__ void GPULogisticApply(const DType* input, DType* output, size_t size);

template <typename DType>
__global__ void GPUCrossEntropyBinaryApply(const DType* input, const DType* target, DType* sum, size_t size);

template <typename DType>
__global__ void GPUCrossEntropyMultiApply(const DType* input, const DType* target, DType* sum, size_t size);

template <typename DType>
__global__ void GPUBiasApply(const DType* input, const DType* bias, DType* output, size_t batch_size, size_t dim);

template <typename DType>
__global__ void GPUBiasDerivative(const DType* input, DType* update, size_t dim, size_t batch_size);

template <typename DType>
__global__ void GPUGradientdescent(DType* weight, DType* gradient, DType* velocity, DType momentum_coef, DType learning_rate, DType decay, size_t batch_size, size_t size);

template <typename DType>
__global__ void GPUMakeBinaryMask(DType* output, DType keep, size_t size);

template <typename DType>
__global__ void GPUUniformTransform(DType* output, DType low, DType high, size_t size);

template <typename DType>
__global__ void GPUEvaluateClass(const DType* output, const DType* target, DType* correct, size_t dim, size_t size);

}  // namespace utils

}  // namespace blitz

#endif  // INCLUDE_UTIL_BLITZ_GPU_FUNCTION_H_
