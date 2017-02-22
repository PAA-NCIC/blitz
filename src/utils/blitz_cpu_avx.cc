#include "utils/blitz_cpu_avx.h"

#include <immintrin.h>

namespace blitz {

namespace utils {

template <>
union AVXReg<float> {
  __m256 v;
  float d[BLITZ_AVX_WIDTH / sizeof(float)];
};

template <>
union AVXReg<double> {
  __m256d v;
  double d[BLITZ_AVX_WIDTH / sizeof(double)];
};

template<>
inline void AVXBroadcast<float>(const float* addr, AVXReg<float>* reg) {
  reg->v = _mm256_broadcast_ss(const_cast<float*>(addr));
}

template<>
inline void AVXBroadcast<double>(const double* addr, AVXReg<double>* reg) {
  reg->v = _mm256_broadcast_sd(const_cast<double*>(addr));
}

template<>
inline void AVXLoad<float>(const float* addr, AVXReg<float>* reg) {
  reg->v = _mm256_load_ps(const_cast<float*>(addr));
}

template<>
inline void AVXLoad<double>(const double* addr, AVXReg<double>* reg) {
  reg->v = _mm256_load_pd(const_cast<double*>(addr));
}

template<>
inline void AVXStore<float>(const float* addr, AVXReg<float>* reg) {
  _mm256_store_ps(const_cast<float*>(addr), reg->v);
}

template<>
inline void AVXStore<double>(const double* addr, AVXReg<double>* reg) {
  _mm256_store_pd(const_cast<double*>(addr), reg->v);
}

template<>
inline void AVXMax<float>(const AVXReg<float>* left, const AVXReg<float>* right, AVXReg<float>* output) {
  output->v = _mm256_max_ps(left->v, right->v);
}

template<>
inline void AVXMax<double>(const AVXReg<double>* left, const AVXReg<double>* right, AVXReg<double>* output) {
  output->v = _mm256_max_pd(left->v, right->v);
}

template<>
inline void AVXMin<float>(const AVXReg<float>* left, const AVXReg<float>* right, AVXReg<float>* output) {
  output->v = _mm256_min_ps(left->v, right->v);
}

template<>
inline void AVXMin<double>(const AVXReg<double>* left, const AVXReg<double>* right, AVXReg<double>* output) {
  output->v = _mm256_min_pd(left->v, right->v);
}

}  // namespace utils

}  // namespace blitz
