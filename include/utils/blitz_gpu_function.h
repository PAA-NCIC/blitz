#ifndef INCLUDE_UTIL_BLITZ_GPU_FUNCTION_H_
#define INCLUDE_UTIL_BLITZ_GPU_FUNCTION_H_

#include <cublas.h>
#include <cuda.h>
#ifdef BLITZ_USE_CUDNN
#include <cudnn.h>
#endif
#include <curand_kernel.h>

#include <boost/scoped_ptr.hpp>
#include <boost/thread/once.hpp>
#include <boost/noncopyable.hpp>

#include "utils/common.h"

#define CUDNN_CHECK(condition) \
  do { \
    cudnnStatus_t status = condition; \
    if (status != CUDNN_STATUS_SUCCESS) \
      LOG(INFO) << cudnnGetErrorString(status); \
  } while (0)

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

namespace blitz {

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

#ifdef BLITZ_USE_CUDNN
namespace cudnn {

template <typename DType> class DataType;

template<> class DataType<float>  {
 public:
  static const cudnnDataType_t type = CUDNN_DATA_FLOAT;
};

template<> class DataType<double> {
 public:
  static const cudnnDataType_t type = CUDNN_DATA_DOUBLE;
};

// cudnn utilities brought from Caffe
// http://caffe.berkeleyvision.org/
template <typename DType>
inline void createTensor4dDesc(cudnnTensorDescriptor_t* desc) {
  CUDNN_CHECK(cudnnCreateTensorDescriptor(desc));
}

template <typename DType>
inline void setTensor4dDesc(cudnnTensorDescriptor_t* desc,
  size_t n, size_t c, size_t h, size_t w) {
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(*desc,
    CUDNN_TENSOR_NCHW, DataType<DType>::type,
    n, c, h, w));
}

template <typename DType>
inline void createFilterDesc(cudnnFilterDescriptor_t* desc) {
  CUDNN_CHECK(cudnnCreateFilterDescriptor(desc));
}

template <typename DType>
inline void setFilterDesc(cudnnFilterDescriptor_t* desc,
  size_t n, size_t c, size_t h, size_t w) {
  CUDNN_CHECK(cudnnSetFilter4dDescriptor(*desc, DataType<DType>::type,
    CUDNN_TENSOR_NCHW, n, c, h, w));
}

template <typename DType>
inline void createConvolution2DDesc(cudnnConvolutionDescriptor_t* conv) {
  CUDNN_CHECK(cudnnCreateConvolutionDescriptor(conv));
}

template <typename DType>
inline void setConvolution2DDesc(cudnnConvolutionDescriptor_t* conv,
  size_t pad_h, size_t pad_w, size_t stride_h, size_t stride_w) {
  CUDNN_CHECK(cudnnSetConvolution2dDescriptor(*conv,
    pad_h, pad_w, stride_h, stride_w, 1, 1, CUDNN_CROSS_CORRELATION));
}

}  // namespace cudnn 
#endif

inline size_t BlitzGPUGetBlocks(size_t N) {
  return (N + BLITZ_NUM_GPU_THREADS - 1) / BLITZ_NUM_GPU_THREADS;
}

inline size_t BlitzGPUGetBlocks(size_t N, size_t nthreads) {
  return (N + nthreads - 1) / nthreads;
}

template <typename DType>
inline __device__ DType BlitzGPUSafeLog(DType input) {
  return log(input > exp(-100.0) ? input : exp(-100.0));
}

template<typename DType>
void BlitzGPUTrans(DType* input, DType* output, size_t M, size_t N);

template<typename DType>
DType BlitzGPUASum(const DType* data, size_t N);

template<typename DType>
void BlitzGenerateNormal(curandGenerator_t* gen, DType* data,
  DType loc, DType scale, size_t size);

template<typename DType>
void BlitzGenerateUniform(curandGenerator_t* gen, DType* data, size_t size);

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

}  // namespace blitz

#endif  // INCLUDE_UTIL_BLITZ_GPU_FUNCTION_H_
