#ifndef INCLUDE_UTIL_BLITZ_CPU_FUNCTION_H_
#define INCLUDE_UTIL_BLITZ_CPU_FUNCTION_H_

#include <cmath>
#include <cstddef>

namespace blitz {

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

namespace utils {

template<typename DType>
void CPUCopy(const DType* X, DType* Y, size_t N);

template <typename DType>
inline DType CPUSafeLog(DType input) {
  return log(input > exp(-50.0) ? input : exp(-50.0));
}

}  // namespace utils

}  // namespace blitz

#endif  // INCLUDE_UTIL_BLITZ_CPU_FUNCTION_H_
