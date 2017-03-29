#undef SRC_UTILS_VECTOR_CONVOLUTION_FORWARD_QBLOCK_AVX_INL_H_
#ifndef SRC_UTILS_VECTOR_CONVOLUTION_FORWARD_QBLOCK_AVX_INL_H_
#define SRC_UTILS_VECTOR_CONVOLUTION_FORWARD_QBLOCK_AVX_INL_H_

#include "qblock_pack-inl.h"
for (size_t bk = 0; bk < KBLOCK / (KREG * VEC_LEN); ++bk) {
  size_t mk = bk * (KREG * VEC_LEN);
  for (size_t bpq = 0; bpq < PQBLOCK / PQREG; ++bpq) {
    #pragma unroll
    for (size_t rpq = 0; rpq < PQREG; ++rpq) {
      #pragma unroll
      for (size_t rk = 0; rk < KREG; ++rk) {
        Ovec[rpq][rk] = _mm256_set1_ps(0);
      }
    }
    for (size_t bc = 0; bc < CBLOCK; ++bc) {
      #pragma unroll
      for (size_t rk = 0; rk < KREG; ++rk) {
        Fvec[rk] = _mm256_load_ps(F_pack + bk * (CBLOCK * KREG * VEC_LEN) + bc * (KREG * VEC_LEN) + rk * VEC_LEN);
      }
      #pragma unroll
      for (size_t rpq = 0; rpq < PQREG; ++rpq) {
        Ivec = _mm256_set1_ps(*(I_pack + (bpq * PQREG + rpq) * CBLOCK + bc));
        #pragma unroll
        for (size_t rk = 0; rk < KREG; ++rk) {
          Ovec[rpq][rk] = _mm256_add_ps(_mm256_mul_ps(Ivec, Fvec[rk]), Ovec[rpq][rk]);
        }
      }
    }
    aq = (iq + bpq * PQREG) % Q;
    ap = ip + (iq + bpq * PQREG) / Q;
    for (size_t rpq = 0; rpq < PQREG; ++rpq) {
      if (ap >= P) {
        break;
      }
      #pragma unroll
      for (size_t rk = 0; rk < KREG; ++rk) {
        _mm256_store_ps(ADDRESS_OUTPUT_NPQK(n, ap, aq, (ik + mk + rk * VEC_LEN)),
            _mm256_add_ps(_mm256_load_ps(ADDRESS_OUTPUT_NPQK(n, ap, aq, (ik + mk + rk * VEC_LEN))), Ovec[rpq][rk]));
      }
      aq += 1;
      if (aq >= Q) {
        ap += 1;
        aq = 0;
      } 
    }
    if (ap >= P) {
      break;
    }
  }
}

#endif  // SRC_UTILS_VECTOR_CONVOLUTION_FORWARD_SSE_QBLOCK_AVX_INL_H_
