#include "backends/context.h"

#include "backends/gpu_tensor.h"

namespace blitz {

template<>
void ConvolutionContext<GPUTensor, float>::InitAlgorithmForUser(BLITZ_ALGORITHM algorithm) {
  Shape workspace_shape(1);
  size_t workspace_unpack_size = C_ * R_ * S_ * P_ * Q_;
  size_t input_size = N_ * C_ * H_ * W_;
  size_t output_size = N_ * KF_ * P_ * Q_;
  size_t filter_size = KF_ * CF_ * R_ * S_;
  switch (algorithm) {
    case BLITZ_CONVOLUTION_SASS_GEMM:
    case BLITZ_CONVOLUTION_BLAS_GEMM:
      workspace_shape[0] = workspace_unpack_size;
      break;
    case BLITZ_CONVOLUTION_SASS_DIRECT:
      workspace_shape[0] = input_size + output_size + filter_size;
      break;
    case BLITZ_CONVOLUTION_NAIVE_DIRECT:
      workspace_shape[0] = 0;
      break;
    default:
      LOG(FATAL) << "No such algorithm: " << algorithm;
      break;
  }
  this->workspace_ = make_shared<GPUTensor<float> >(workspace_shape);
  this->conv_algorithm_ = algorithm;
}

template<>
void ConvolutionContext<GPUTensor, float>::InitAlgorithmForMemory(size_t memory_size) {
  this->conv_algorithm_ = BLITZ_CONVOLUTION_NAIVE_DIRECT;
}

template<>
void ConvolutionContext<GPUTensor, float>::InitAlgorithmForSpeed(size_t memory_size) {
  Shape workspace_shape(1);
  size_t workspace_unpack_size = C_ * R_ * S_ * P_ * Q_;
  size_t input_size = N_ * C_ * H_ * W_;
  size_t output_size = N_ * KF_ * P_ * Q_;
  size_t filter_size = KF_ * CF_ * R_ * S_;
  if (input_size + output_size + filter_size < memory_size / sizeof(float)) {
    workspace_shape[0] = input_size + output_size + filter_size;
    this->conv_algorithm_ = BLITZ_CONVOLUTION_SASS_DIRECT;
  } else if (workspace_unpack_size < memory_size / sizeof(float)) {
    workspace_shape[0] = workspace_unpack_size;
    if ((C_ * R_ * S_) % 4 == 0 && (KF_ % 4)) {
      this->conv_algorithm_ = BLITZ_CONVOLUTION_SASS_GEMM;
    } else {
      this->conv_algorithm_ = BLITZ_CONVOLUTION_BLAS_GEMM;
    }
  } else {
    LOG(FATAL) << "Not availble memory size!";
  }
  this->workspace_ = make_shared<GPUTensor<float> >(workspace_shape);
}

INSTANTIATE_CONTEXT(Convolution, GPUTensor);

}  // namespace blitz
