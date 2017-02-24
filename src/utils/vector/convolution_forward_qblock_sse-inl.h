#undef SRC_UTILS_VECTOR_CONVOLUTION_FORWARD_QBLOCK_SSE_INL_H_
#ifndef SRC_UTILS_VECTOR_CONVOLUTION_FORWARD_QBLOCK_SSE_INL_H_
#define SRC_UTILS_VECTOR_CONVOLUTION_FORWARD_QBLOCK_SSE_INL_H_

for (size_t r = 0; r < R; ++r) {
  if (ih + static_cast<int>(r) >= 0 && ih + static_cast<int>(r) < static_cast<int>(H)) {
    for (size_t s = 0; s < S; ++s) {
      for (size_t c = 0; c < C / CBLOCK; ++c) {
        size_t ic = c * CBLOCK;
        for (size_t k = 0; k < K / VEC_LEN; ++k) {
          size_t ik = k * KBLOCK;
          int ih_index = ih + r;
          if (k == 0) {
            #include "qblock_pack-inl.h"
          }
          Ovec[3] = Ovec[2] = Ovec[1] = Ovec[0] = _mm_set_ps1(0);
          for (size_t bc = 0; bc < CBLOCK; ++bc) {
            Fvec[bc] = _mm_load_ps(ADDRESS_FILTER_RSCK(r, s, (ic + bc), ik));
            for (size_t bq = 0; bq < PQBLOCK; ++bq) {
              Ivec[bq] = _mm_load_ps1(I_pack + bq * CBLOCK + bc);
              Ovec[bq] = _mm_add_ps(_mm_mul_ps(Ivec[bq], Fvec[bc]), Ovec[bq]);
            }
          }
          for (size_t bq = 0; bq < rq; ++bq) {
            _mm_store_ps(ADDRESS_OUTPUT_NPQK(n, p, (iq + bq), (k * VEC_LEN)),
              _mm_add_ps(_mm_load_ps(ADDRESS_OUTPUT_NPQK(n, p, (iq + bq), (k * VEC_LEN))), Ovec[bq]));
          }
        }
      }
      size_t ic = C / CBLOCK * CBLOCK; 
      size_t rc = C - ic;
      if (rc > 0) {
        for (size_t k = 0; k < K / KBLOCK; ++k) {
          size_t ik = k * KBLOCK;
          int ih_index = ih + r;
          if (k == 0) {
            #include "qblock_pack-inl.h"
          }
          Ovec[3] = Ovec[2] = Ovec[1] = Ovec[0] = _mm_set_ps1(0);
          for (size_t bc = 0; bc < rc; ++bc) {
            Fvec[bc] = _mm_load_ps(ADDRESS_FILTER_RSCK(r, s, (ic + bc), ik));
            for (size_t bq = 0; bq < PQBLOCK; ++bq) {
              Ivec[bq] = _mm_load_ps1(I_pack + bq * CBLOCK + bc);
              Ovec[bq] = _mm_add_ps(_mm_mul_ps(Ivec[bq], Fvec[bc]), Ovec[bq]);
            }
          }
          for (size_t bq = 0; bq < rq; ++bq) {
            _mm_store_ps(ADDRESS_OUTPUT_NPQK(n, p, (iq + bq), (k * VEC_LEN)),
              _mm_add_ps(_mm_load_ps(ADDRESS_OUTPUT_NPQK(n, p, (iq + bq), (k * VEC_LEN))), Ovec[bq]));
          }
        }
      }
    }
  }
}

#endif  // SRC_UTILS_VECTOR_CONVOLUTION_FORWARD_SSE_QBLOCK_SSE_INL_H_
