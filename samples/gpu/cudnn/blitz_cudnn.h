#include <cudnn.h>

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

#define CUDNN_CHECK(condition) \
  do { \
    cudnnStatus_t status = condition; \
    if (status != CUDNN_STATUS_SUCCESS) \
      LOG(INFO) << cudnnGetErrorString(status); \
  } while (0)

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
inline void setTensor4dNHWCDesc(cudnnTensorDescriptor_t* desc,
  size_t n, size_t c, size_t h, size_t w) {
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(*desc,
    CUDNN_TENSOR_NHWC, DataType<DType>::type,
    n, c, h, w));
}

template <typename DType>
inline void setTensor4dNCHWDesc(cudnnTensorDescriptor_t* desc,
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
  size_t k, size_t c, size_t h, size_t w) {
  CUDNN_CHECK(cudnnSetFilter4dDescriptor(*desc, DataType<DType>::type,
    CUDNN_TENSOR_NCHW, k, c, h, w));
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
