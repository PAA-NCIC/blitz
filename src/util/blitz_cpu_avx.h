#ifndef SRC_UTIL_BLITZ_CPU_AVX_H_
#define SRC_UTIL_BLITZ_CPU_AVX_H_

#include <immintrin.h>

// 32 bytes
#define BLITZ_AVX_WIDTH 32

namespace blitz {

template <typename DType>
union BlitzAVXReg;

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

template<typename DType>
inline void BlitzAVXBroadcast(const DType* addr, BlitzAVXReg<DType>* reg);

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

template<typename DType>
inline void BlitzAVXLoad(const DType* addr, BlitzAVXReg<DType>* reg);

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

template<typename DType>
inline void BlitzAVXStore(const DType* addr, BlitzAVXReg<DType>* reg);

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

template<typename DType>
inline void BlitzAVXMax(BlitzAVXReg<DType>* left, BlitzAVXReg<DType>* right,
  BlitzAVXReg<DType>* result);

template<>
inline void BlitzAVXMax<float>(BlitzAVXReg<float>* left, BlitzAVXReg<float>* right,
  BlitzAVXReg<float>* result) {
  result->v = _mm256_max_ps(left->v, right->v);
}

template<>
inline void BlitzAVXMax<double>(BlitzAVXReg<double>* left, BlitzAVXReg<double>* right,
  BlitzAVXReg<double>* result) {
  result->v = _mm256_max_pd(left->v, right->v);
}

template<typename DType>
inline void BlitzAVXMin(BlitzAVXReg<DType>* left, BlitzAVXReg<DType>* right,
  BlitzAVXReg<DType>* result);

template<>
inline void BlitzAVXMin<float>(BlitzAVXReg<float>* left, BlitzAVXReg<float>* right,
  BlitzAVXReg<float>* result) {
  result->v = _mm256_min_ps(left->v, right->v);
}

template<>
inline void BlitzAVXMin<double>(BlitzAVXReg<double>* left, BlitzAVXReg<double>* right,
  BlitzAVXReg<double>* result) {
  result->v = _mm256_min_pd(left->v, right->v);
}

}  // namespace blitz

#endif  // SRC_UTIL_BLITZ_CPU_AVX_H_
