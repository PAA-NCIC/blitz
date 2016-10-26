#ifndef INCLUDE_UTIL_BLITZ_CPU_FUNCTION_H_
#define INCLUDE_UTIL_BLITZ_CPU_FUNCTION_H_

#ifdef USE_MKL
#include <mkl.h>
#else

#ifdef __cplusplus
  extern "C" {
#endif
    #include <cblas.h>
#ifdef __cplusplus
  }
#endif
#endif

#include <cmath>

namespace blitz {

template<typename DType>
void BlitzCPUGemm(DType* A, DType* B, DType* C,
  bool transa, bool transb,
  DType alpha, DType beta,
  size_t M, size_t N, size_t K);

template<typename DType>
void BlitzCPUCopy(const DType* X, DType* Y, size_t N);

template <typename DType>
inline DType BlitzCPUSafeLog(DType input) {
  return log(input > exp(-50.0) ? input : exp(-50.0));
}

}  // namespace blitz

#endif  // INCLUDE_UTIL_BLITZ_CPU_FUNCTION_H_
