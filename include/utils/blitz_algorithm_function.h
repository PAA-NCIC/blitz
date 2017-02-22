#ifndef INCLUDE_UTIL_BLITZ_ALGORITHM_FUNCTION_H_
#define INCLUDE_UTIL_BLITZ_ALGORITHM_FUNCTION_H_

#include "utils/blitz_define.h"

namespace blitz {

inline BLITZ_ALGORITHM BlitzParseAlgorithm(const string& algorithm) {
  if (algorithm == "convolution_sass_gemm") {
    return BLITZ_CONVOLUTION_SASS_GEMM;
  } else if (algorithm == "convolution_sass_direct") {
    return BLITZ_CONVOLUTION_SASS_DIRECT;
  } else if (algorithm == "convolution_cudnn") {
    return BLITZ_CONVOLUTION_CUDNN;
  } else if (algorithm == "convolution_blas_gemm") {
    return BLITZ_CONVOLUTION_BLAS_GEMM;
  } else if (algorithm == "convolution_blas_gemm_batch") {
    return BLITZ_CONVOLUTION_BLAS_GEMM_BATCH;
  } else if (algorithm == "convolution_xsmm_direct") {
    return BLITZ_CONVOLUTION_XSMM_DIRECT;
  } else if (algorithm == "convolution_naive_direct") {
    return BLITZ_CONVOLUTION_NAIVE_DIRECT;
  } else if (algorithm == "convolution_vector_direct") {
    return BLITZ_CONVOLUTION_VECTOR_DIRECT;
  } else if (algorithm == "blas_gemm") {
    return BLITZ_BLAS_GEMM;
  } else if (algorithm == "sass_gemm") {
    return BLITZ_SASS_GEMM;
  } else {
    return BLITZ_ALGORITHM_UNDEFINED;
  }
}

}  // namespace blitz

#endif  // INCLUDE_UTIL_BLITZ_ALGORITHM_FUNCTION_H_
