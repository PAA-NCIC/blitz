#ifndef SRC_UTIL_BLITZ_GPU_FUNCTION_H_
#define SRC_UTIL_BLITZ_GPU_FUNCTION_H_

#include <cublas.h>
#include <cuda.h>
#include <cudnn.h>
#include <curand_kernel.h>
#include <iostream>

#include <boost/scoped_ptr.hpp>
#include <boost/thread/once.hpp>
#include <boost/noncopyable.hpp>

#define CUDNN_CHECK(condition) \
  do { \
    cudnnStatus_t status = condition; \
    if (status != CUDNN_STATUS_SUCCESS) \
      std::cout << cudnnGetErrorString(status); \
  } while (0)

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
    cublasStatus_t stat = cublasCreate_v2(
      CuBlasHandle::instance_.get());
  }
  
  virtual ~CuBlasHandle();

 private:
  CuBlasHandle();
  CuBlasHandle(const CuBlasHandle& cublas_handle);
  CuBlasHandle& operator=(const CuBlasHandle& rhs);

  static boost::scoped_ptr<cublasHandle_t> instance_;
  static boost::once_flag flag_;
};

#if __CUDA_ARCH__ >= 200
  #define BLITZ_NUM_GPU_THREADS 1024
#else
  #define BLITZ_NUM_GPU_THREADS 512
#endif

// grid stride looping
// https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
#define BLITZ_CUDA_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
      i < (n); \
      i += blockDim.x * gridDim.x)

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
    int n, int c, int h, int w) {
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
    int n, int c, int h, int w) {
  CUDNN_CHECK(cudnnSetFilter4dDescriptor(*desc, DataType<DType>::type,
    CUDNN_TENSOR_NCHW, n, c, h, w));
}

template <typename DType>
inline void createConvolution2DDesc(cudnnConvolutionDescriptor_t* conv) {
  CUDNN_CHECK(cudnnCreateConvolutionDescriptor(conv));
}

template <typename DType>
inline void setConvolution2DDesc(cudnnConvolutionDescriptor_t* conv,
  int pad_h, int pad_w, int stride_h, int stride_w) {
  CUDNN_CHECK(cudnnSetConvolution2dDescriptor(*conv,
    pad_h, pad_w, stride_h, stride_w, 1, 1, CUDNN_CROSS_CORRELATION));
}

}

// TODO(keren) put into header file
template<typename DType>
void BlitzGPUGemm(const bool transa, const bool transb,
  const int M, const int N, const int K,
  DType* A, DType* B, DType* C, DType alpha, DType beta);

template<typename DType>
void BlitzGPUTrans(const int M, const int N, DType* input, DType* output);

template<typename DType>
DType BlitzGPUASum(const int N, const DType* data);

template<typename DType>
void BlitzGenerateNormal(curandGenerator_t* gen, DType* data, const int size,
  const DType loc, const DType scale);

template<typename DType>
void BlitzGenerateUniform(curandGenerator_t* gen, DType* data, const int size);

inline int BlitzGPUGetBlocks(const int N) {
  return (N + BLITZ_NUM_GPU_THREADS - 1) / BLITZ_NUM_GPU_THREADS;
}

inline int BlitzGPUGetBlocks(const int N, const int nthreads) {
  return (N + nthreads - 1) / nthreads;
}

template<typename DType>
inline void AtomicAdd();

template <typename DType>
inline __device__ DType BlitzGPUSafeLog(DType input) {
  return log(input > exp(-50.0) ? input : exp(-50.0));
}

}  // namespace blitz

#endif  // SRC_UTIL_BLITZ_GPU_FUNCTION_H_
