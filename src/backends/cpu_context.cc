#include "backends/context.h"

#include <omp.h>

#include "backends/cpu_tensor.h"

namespace blitz {

template<>
void ConvolutionContext<CPUTensor, float>::InitAlgorithmForUser(BLITZ_ALGORITHM algorithm) {
  Shape workspace_shape(1);
  size_t workspace_unpack_size = C_ * R_ * S_ * P_ * Q_;
  size_t workspace_update_size = KF_ * C_ * R_ * S_;
  size_t I_pack_size = PQBLOCK * CBLOCK;
  size_t F_pack_size = CBLOCK * KBLOCK;
  size_t threads = omp_get_max_threads();
  switch (algorithm) {
    case BLITZ_CONVOLUTION_BLAS_GEMM:
      workspace_shape[0] = workspace_unpack_size;
      break;
    case BLITZ_CONVOLUTION_BLAS_GEMM_BATCH:
      workspace_shape[0] = threads * (workspace_unpack_size + workspace_update_size);
      break;
    case BLITZ_CONVOLUTION_NAIVE_DIRECT:
      workspace_shape[0] = 0;
      break;
    case BLITZ_CONVOLUTION_VECTOR_DIRECT:
      workspace_shape[0] = threads * (I_pack_size + F_pack_size);
      break;
    default:
      LOG(FATAL) << "No such algorithm: " << algorithm;
      break;
  }
  this->workspace_ = make_shared<CPUTensor<float> >(workspace_shape);
  this->conv_algorithm_ = algorithm;
}

template<>
void ConvolutionContext<CPUTensor, float>::InitAlgorithmForMemory(size_t memory_size) {
  this->conv_algorithm_ = BLITZ_CONVOLUTION_NAIVE_DIRECT;
}

template<>
void ConvolutionContext<CPUTensor, float>::InitAlgorithmForSpeed(size_t memory_size) {
  Shape workspace_shape(1);
  size_t workspace_unpack_size = C_ * R_ * S_ * P_ * Q_;
  size_t workspace_update_size = KF_ * C_ * R_ * S_;
  size_t threads = omp_get_max_threads();
  if (threads * (workspace_update_size + workspace_update_size) < memory_size / sizeof(float)) {
    if (threads < N_) {
      workspace_shape[0] = workspace_unpack_size;
      this->conv_algorithm_ = BLITZ_CONVOLUTION_BLAS_GEMM;
    } else {
      workspace_shape[0] = threads * (workspace_unpack_size + workspace_update_size);
      this->conv_algorithm_ = BLITZ_CONVOLUTION_BLAS_GEMM_BATCH;
    }
  } else if (workspace_update_size < memory_size / sizeof(float)) {
    workspace_shape[0] = workspace_unpack_size;
    this->conv_algorithm_ = BLITZ_CONVOLUTION_BLAS_GEMM;
  } else {
    this->conv_algorithm_ = BLITZ_CONVOLUTION_NAIVE_DIRECT;
  }
  this->workspace_ = make_shared<CPUTensor<float> >(workspace_shape);
}

INSTANTIATE_CONTEXT(Convolution, CPUTensor);

}  // namespace blitz
