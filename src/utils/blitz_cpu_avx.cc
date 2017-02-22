#include "utils/blitz_cpu_avx.h"

#include <immintrin.h>

namespace blitz {

namespace utils {

template <>
union BlitzAVXReg<float> {
  __m256 v;
  float d[BLITZ_AVX_WIDTH / sizeof(float)];
};

template <>
union BlitzAVXReg<double> {
  __m256d v;
  double d[BLITZ_AVX_WIDTH / sizeof(double)];
};

template<>
inline void BlitzAVXBroadcast<float>(const float* addr,
  BlitzAVXReg<float>* reg) {
  reg->v = _mm256_broadcast_ss(const_cast<float*>(addr));
}

template<>
inline void BlitzAVXBroadcast<double>(const double* addr,
  BlitzAVXReg<double>* reg) {
  reg->v = _mm256_broadcast_sd(const_cast<double*>(addr));
}

template<>
inline void BlitzAVXLoad<float>(const float* addr,
  BlitzAVXReg<float>* reg) {
  reg->v = _mm256_load_ps(const_cast<float*>(addr));
}

template<>
inline void BlitzAVXLoad<double>(const double* addr,
  BlitzAVXReg<double>* reg) {
  reg->v = _mm256_load_pd(const_cast<double*>(addr));
}

template<>
inline void BlitzAVXStore<float>(const float* addr,
  BlitzAVXReg<float>* reg) {
  _mm256_store_ps(const_cast<float*>(addr), reg->v);
}

template<>
inline void BlitzAVXStore<double>(const double* addr,
  BlitzAVXReg<double>* reg) {
  _mm256_store_pd(const_cast<double*>(addr), reg->v);
}

template<>
inline void BlitzAVXMax<float>(const BlitzAVXReg<float>* left,
  const BlitzAVXReg<float>* right, BlitzAVXReg<float>* output) {
  output->v = _mm256_max_ps(left->v, right->v);
}

template<>
inline void BlitzAVXMax<double>(const BlitzAVXReg<double>* left,
  const BlitzAVXReg<double>* right, BlitzAVXReg<double>* output) {
  output->v = _mm256_max_pd(left->v, right->v);
}

template<>
inline void BlitzAVXMin<float>(const BlitzAVXReg<float>* left,
  const BlitzAVXReg<float>* right, BlitzAVXReg<float>* output) {
  output->v = _mm256_min_ps(left->v, right->v);
}

template<>
inline void BlitzAVXMin<double>(const BlitzAVXReg<double>* left,
  const BlitzAVXReg<double>* right, BlitzAVXReg<double>* output) {
  output->v = _mm256_min_pd(left->v, right->v);
}

}  // namespace utils

}  // namespace blitz
