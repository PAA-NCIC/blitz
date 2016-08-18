#ifndef SRC_UTIL_BLITZ_GPU_FUNCTION_H_
#define SRC_UTIL_BLITZ_GPU_FUNCTION_H_

#include <cublas.h>
#include <cuda.h>
#include <cudnn.h>
#include <curand_kernel.h>

#include <boost/scoped_ptr.hpp>
#include <boost/thread/once.hpp>
#include <boost/noncopyable.hpp>

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

template <typename Dtype> class CudnnDataType;

template<> class CudnnDataType<float>  {
 public:
  static const cudnnDataType_t type = CUDNN_DATA_FLOAT;
};

template<> class CudnnDataType<double> {
 public:
  static const cudnnDataType_t type = CUDNN_DATA_DOUBLE;
};

template<typename DType>
void BlitzGPUSetCudnn4dTensor(cudnnTensorDescriptor_t* tensor_descriptor,
  int N, int C, int H, int W) {
  cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW;
  cudnnSetTensor4dDescriptor(*tensor_descriptor, format,
    CudnnDataType<DType>::type, N, C, H, W);
}

template<typename DType>
void BlitzGPUSetCudnn4dFilter(cudnnFilterDescriptor_t* filter_descriptor,
  DType* data, int K, int C, int H, int W) {
  cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW;
  cudnnSetFilter4dDescriptor(*filter_descriptor, format,
    CudnnDataType<DType>::type, K, C, H, W);
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
