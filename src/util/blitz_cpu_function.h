#ifndef SRC_UTIL_BLITZ_CPU_FUNCTION_H_
#define SRC_UTIL_BLITZ_CPU_FUNCTION_H_

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
void BlitzCPUGemm(const bool transa, const bool transb,
  const int M, const int N, const int K,
  DType* A, DType* B, DType* C, DType alpha, DType beta);

template<typename DType>
void BlitzCPUCopy(const DType* X, const int N, DType* Y);

template <typename DType>
inline DType BlitzCPUSafeLog(DType input) {
  return log(input > exp(-50.0) ? input : exp(-50.0));
}

}  // namespace blitz

#endif  // SRC_UTIL_BLITZ_CPU_FUNCTION_H_

