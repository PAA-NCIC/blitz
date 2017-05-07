#undef SRC_UTILS_VECTOR_CONVOLUTION_FORWARD_PQBLOCK_SSE_INL_H_
#ifndef SRC_UTILS_VECTOR_CONVOLUTION_FORWARD_PQBLOCK_SSE_INL_H_
#define SRC_UTILS_VECTOR_CONVOLUTION_FORWARD_PQBLOCK_SSE_INL_H_

#include "pqblock_pack-inl.h"
for (size_t bk = 0; bk < lk / (KREG * VEC_LEN); ++bk) {
  size_t mk = bk * (KREG * VEC_LEN);
  for (size_t bpq = 0; bpq < lpq / PQREG; ++bpq) {
    #pragma unroll
    for (size_t rpq = 0; rpq < PQREG; ++rpq) {
      #pragma unroll
      for (size_t rk = 0; rk < KREG; ++rk) {
        Ovec[rpq][rk] = _mm128_setzero_ps();
      }
    }
    float *__restrict__ F_slice = F_pack + bk * (CBLOCK * KREG * VEC_LEN);
    float *__restrict__ I_slice = I_pack + bpq * PQREG * CBLOCK;
    for (size_t bc = 0; bc < lc; ++bc) {
      #pragma unroll
      for (size_t rk = 0; rk < KREG; ++rk) {
        Fvec[rk] = _mm128_load_ps(F_slice + bc * (KREG * VEC_LEN) + rk * VEC_LEN);
      }
      #pragma unroll
      for (size_t rpq = 0; rpq < PQREG; ++rpq) {
        Ivec = _mm128_set1_ps(I_slice[CBLOCK * rpq + bc]);
        #pragma unroll
        for (size_t rk = 0; rk < KREG; ++rk) {
          Ovec[rpq][rk] = _mm128_add_ps(_mm128_mul_ps(Ivec, Fvec[rk]), Ovec[rpq][rk]);
        }
      }
    }
    aq = (iq + bpq * PQREG) % Q;
    ap = ip + (iq + bpq * PQREG) / Q;
    #pragma unroll
    for (size_t rpq = 0; rpq < PQREG; ++rpq) {
      if (ap >= P) {
        break;
      }
      #pragma unroll
      for (size_t rk = 0; rk < KREG; ++rk) {
        _mm128_store_ps(ADDRESS_OUTPUT_NPQK(n, ap, aq, (ik + mk + rk * VEC_LEN)),
            _mm128_add_ps(_mm128_load_ps(ADDRESS_OUTPUT_NPQK(n, ap, aq, (ik + mk + rk * VEC_LEN))), Ovec[rpq][rk]));
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

#endif  // SRC_UTILS_VECTOR_CONVOLUTION_FORWARD_SSE_PQBLOCK_SSE_INL_H_
