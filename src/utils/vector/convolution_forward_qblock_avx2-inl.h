#undef SRC_UTILS_VECTOR_CONVOLUTION_FORWARD_QBLOCK_AVX2_INL_H_
#ifndef SRC_UTILS_VECTOR_CONVOLUTION_FORWARD_QBLOCK_AVX2_INL_H_
#define SRC_UTILS_VECTOR_CONVOLUTION_FORWARD_QBLOCK_AVX2_INL_H_

#include "qblock_pack-inl.h"
for (size_t bk = 0; bk < lk / (KREG * VEC_LEN); ++bk) {
  size_t mk = bk * (KREG * VEC_LEN);
  for (size_t bpq = 0; bpq < lpq / PQREG; ++bpq) {
    #pragma unroll
    for (size_t rpq = 0; rpq < PQREG; ++rpq) {
      #pragma unroll
      for (size_t rk = 0; rk < KREG; ++rk) {
        Ovec[rpq][rk] = _mm256_setzero_ps();
      }
    }
    float *__restrict__ F_slice = F_pack + bk * (CBLOCK * KREG * VEC_LEN);
    float *__restrict__ I_slice = I_pack + bpq * PQREG * CBLOCK;
    for (size_t bc = 0; bc < lc; ++bc) {
      // PQREG = 6
      // KREG = 2
      Fvec[0] = _mm256_load_ps(F_slice + bc * (KREG * VEC_LEN));
      Fvec[1] = _mm256_load_ps(F_slice + bc * (KREG * VEC_LEN) + VEC_LEN);
      Ivec[0] = _mm256_set1_ps(I_pack[bc]);
      Ivec[1] = _mm256_set1_ps(I_pack[CBLOCK + bc]);
      Ovec[0][0] = _mm256_fmadd_ps(Ivec[0], Fvec[0], Ovec[0][0]);
      Ovec[0][1] = _mm256_fmadd_ps(Ivec[0], Fvec[1], Ovec[0][1]);
      Ovec[1][0] = _mm256_fmadd_ps(Ivec[1], Fvec[0], Ovec[1][0]);
      Ovec[1][1] = _mm256_fmadd_ps(Ivec[1], Fvec[1], Ovec[1][1]);
      _mm_prefetch((char*)(F_slice + (bc + 1) * (KREG * VEC_LEN)), _MM_HINT_T0);
      Ivec[0] = _mm256_set1_ps(I_pack[CBLOCK * 2 + bc]);
      Ivec[1] = _mm256_set1_ps(I_pack[CBLOCK * 3 + bc]);
      Ovec[2][0] = _mm256_fmadd_ps(Ivec[0], Fvec[0], Ovec[2][0]);
      Ovec[2][1] = _mm256_fmadd_ps(Ivec[0], Fvec[1], Ovec[2][1]);
      Ovec[3][0] = _mm256_fmadd_ps(Ivec[1], Fvec[0], Ovec[3][0]);
      Ovec[3][1] = _mm256_fmadd_ps(Ivec[1], Fvec[1], Ovec[3][1]);
      _mm_prefetch((char*)(F_slice + (bc + 1) * (KREG * VEC_LEN) + VEC_LEN), _MM_HINT_T0);
      Ivec[0] = _mm256_set1_ps(I_pack[CBLOCK * 4 + bc]);
      Ivec[1] = _mm256_set1_ps(I_pack[CBLOCK * 5 + bc]);
      Ovec[4][0] = _mm256_fmadd_ps(Ivec[0], Fvec[0], Ovec[4][0]);
      Ovec[4][1] = _mm256_fmadd_ps(Ivec[0], Fvec[1], Ovec[4][1]);
      Ovec[5][0] = _mm256_fmadd_ps(Ivec[1], Fvec[0], Ovec[5][0]);
      Ovec[5][1] = _mm256_fmadd_ps(Ivec[1], Fvec[1], Ovec[5][1]);
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

#endif  // SRC_UTILS_VECTOR_CONVOLUTION_FORWARD_SSE_QBLOCK_AVX2_INL_H_
