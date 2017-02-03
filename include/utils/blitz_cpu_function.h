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
#include <cstddef>

#define BLITZ_CPU_TIMER_START(elapsed_time, t1) \
  do { \
    elapsed_time = 0.0; \
    gettimeofday(&t1, NULL); \
  } while (0) 

#define BLITZ_CPU_TIMER_END(elapsed_time, t1, t2) \
  do { \
    gettimeofday(&t2, NULL); \
    elapsed_time = (t2.tv_sec - t1.tv_sec) * 1000.0; \
    elapsed_time += (t2.tv_usec - t1.tv_usec) / 1000.0; \
    elapsed_time /= 1000.0; \
  } while (0)

#define BLITZ_CPU_TIMER_INFO(computations, elapsed_time) \
  do { \
    LOG(INFO) << "Running time: " << elapsed_time; \
    LOG(INFO) << "Gflops: " << computations / (elapsed_time * 1e9); \
  } while (0) \

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
