#undef SRC_UTILS_VECTOR_CONVOLUTION_FORWARD_QBLOCK_AVX_INL_H_
#ifndef SRC_UTILS_VECTOR_CONVOLUTION_FORWARD_QBLOCK_AVX_INL_H_
#define SRC_UTILS_VECTOR_CONVOLUTION_FORWARD_QBLOCK_AVX_INL_H_

#include "qblock_pack-inl.h"
for (size_t bk = 0; bk < lk / (KREG * VEC_LEN); ++bk) {
  size_t mk = bk * (KREG * VEC_LEN);
  for (size_t bpq = 0; bpq < lpq / PQREG; ++bpq) {
    #pragma unroll
    for (size_t rpq = 0; rpq < PQREG; ++rpq) {
      #pragma unroll
      for (size_t rk = 0; rk < KREG; ++rk) {
        Ovec[rpq][rk] = _mm256_set1_ps(0);
      }
    }
    for (size_t bc = 0; bc < lc; ++bc) {
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
      // PQREG = 6
      // KREG = 2
      // F_slice = F_pack + bk * (CBLOCK * KREG * VEC_LEN) + bc * (KREG * VEC_LEN);
      // I_slice = I_pack + bpq * PQREG * CBLOCK + bc;
      // Fvec[0] = _mm256_load_ps(F_slice);
      // Ivec = _mm256_set1_ps(*I_slice);
      // Ovec[0][0] = _mm256_add_ps(_mm256_mul_ps(Ivec, Fvec[0]), Ovec[0][0]);

      // _mm_prefetch((char*)(I_slice + CBLOCK), _MM_HINT_NTA);
      // Fvec[1] = _mm256_load_ps(F_slice + VEC_LEN);
      // Ovec[0][1] = _mm256_add_ps(_mm256_mul_ps(Ivec, Fvec[1]), Ovec[0][1]);

      // Ivec = _mm256_set1_ps(*(I_slice + CBLOCK));
      // Ovec[1][0] = _mm256_add_ps(_mm256_mul_ps(Ivec, Fvec[0]), Ovec[1][0]);

      // _mm_prefetch((char*)(I_slice + 2 * CBLOCK), _MM_HINT_NTA);
      // Ovec[1][1] = _mm256_add_ps(_mm256_mul_ps(Ivec, Fvec[1]), Ovec[1][1]);

      // Ivec = _mm256_set1_ps(*(I_slice + 2 * CBLOCK));
      // Ovec[2][0] = _mm256_add_ps(_mm256_mul_ps(Ivec, Fvec[0]), Ovec[2][0]);

      // _mm_prefetch((char*)(I_slice + 3 * CBLOCK), _MM_HINT_NTA);
      // Ovec[2][1] = _mm256_add_ps(_mm256_mul_ps(Ivec, Fvec[1]), Ovec[2][1]);

      // Ivec = _mm256_set1_ps(*(I_slice + 3 * CBLOCK));
      // Ovec[3][0] = _mm256_add_ps(_mm256_mul_ps(Ivec, Fvec[0]), Ovec[3][0]);

      // _mm_prefetch((char*)(I_slice + 4 * CBLOCK), _MM_HINT_NTA);
      // Ovec[3][1] = _mm256_add_ps(_mm256_mul_ps(Ivec, Fvec[1]), Ovec[3][1]);

      // Ivec = _mm256_set1_ps(*(I_slice + 4 * CBLOCK));
      // Ovec[4][0] = _mm256_add_ps(_mm256_mul_ps(Ivec, Fvec[0]), Ovec[4][0]);

      // _mm_prefetch((char*)(I_slice + 5 * CBLOCK), _MM_HINT_NTA);
      // Ovec[4][1] = _mm256_add_ps(_mm256_mul_ps(Ivec, Fvec[1]), Ovec[4][1]);

      // Ivec = _mm256_set1_ps(*(I_slice + 5 * CBLOCK));
      // Ovec[5][0] = _mm256_add_ps(_mm256_mul_ps(Ivec, Fvec[0]), Ovec[5][0]);

      // _mm_prefetch((char*)(F_pack + bk * (CBLOCK * KREG * VEC_LEN) + (bc + 1) * (KREG * VEC_LEN)), _MM_HINT_NTA);
      // Ovec[5][1] = _mm256_add_ps(_mm256_mul_ps(Ivec, Fvec[1]), Ovec[5][1]);
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

#endif  // SRC_UTILS_VECTOR_CONVOLUTION_FORWARD_SSE_QBLOCK_AVX_INL_H_
