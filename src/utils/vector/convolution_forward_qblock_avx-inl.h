#undef SRC_UTILS_VECTOR_CONVOLUTION_FORWARD_QBLOCK_AVX_INL_H_
#ifndef SRC_UTILS_VECTOR_CONVOLUTION_FORWARD_QBLOCK_AVX_INL_H_
#define SRC_UTILS_VECTOR_CONVOLUTION_FORWARD_QBLOCK_AVX_INL_H_

for (size_t r = 0; r < R; ++r) {
  for (size_t s = 0; s < S; ++s) {
    for (size_t c = 0; c < C / CBLOCK; ++c) {
      size_t ic = c * CBLOCK;
      size_t rc = CBLOCK;
      #include "qblock_pack-inl.h"
      for (size_t k = 0; k < K / KBLOCK; ++k) {
        size_t ik = k * KBLOCK;
        for (size_t bpq = 0; bpq < PQBLOCK / VEC_LEN; ++bpq) {
          Ovec[7] = Ovec[6] = Ovec[5] = Ovec[4] = 
            Ovec[3] = Ovec[2] = Ovec[1] = Ovec[0] = _mm256_set1_ps(0);
          for (size_t bc = 0; bc < CBLOCK; ++bc) {
            Fvec = _mm256_load_ps(ADDRESS_FILTER_RSCK(r, s, (ic + bc), ik));
            for (size_t bv = 0; bv < VEC_LEN; ++bv) {
              Ivec = _mm256_broadcast_ss(I_pack + (bpq * VEC_LEN + bv) * CBLOCK + bc);
              Ovec[bv] = _mm256_add_ps(_mm256_mul_ps(Ivec, Fvec), Ovec[bv]);
            }
          }
          aq = (iq + bpq * VEC_LEN) % Q;
          ap = ip + (iq + bpq * VEC_LEN) / Q;
          for (size_t bv = 0; bv < VEC_LEN; ++bv) {
            if (ap >= P) {
              break;
            }
            _mm256_store_ps(ADDRESS_OUTPUT_NPQK(n, ap, aq, ik),
                _mm256_add_ps(_mm256_load_ps(ADDRESS_OUTPUT_NPQK(n, ap, aq, ik)), Ovec[bv]));
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
    }
    size_t ic = C / CBLOCK * CBLOCK; 
    size_t rc = C - ic;
    if (rc > 0) {
      #include "qblock_pack-inl.h"
      for (size_t k = 0; k < K / KBLOCK; ++k) {
        size_t ik = k * KBLOCK;
        for (size_t bpq = 0; bpq < PQBLOCK / VEC_LEN; ++bpq) {
          Ovec[7] = Ovec[6] = Ovec[5] = Ovec[4] = 
            Ovec[3] = Ovec[2] = Ovec[1] = Ovec[0] = _mm256_set1_ps(0);
          for (size_t bc = 0; bc < CBLOCK; ++bc) {
            Fvec = _mm256_load_ps(ADDRESS_FILTER_RSCK(r, s, (ic + bc), ik));
            for (size_t bv = 0; bv < VEC_LEN; ++bv) {
              Ivec = _mm256_broadcast_ss(I_pack + (bpq * VEC_LEN + bv) * CBLOCK + bc);
              Ovec[bv] = _mm256_add_ps(_mm256_mul_ps(Ivec, Fvec), Ovec[bv]);
            }
          }
          aq = (iq + bpq * VEC_LEN) % Q;
          ap = ip + (iq + bpq * VEC_LEN) / Q;
          for (size_t bv = 0; bv < VEC_LEN; ++bv) {
            if (ap >= P) {
              break;
            }
            _mm256_store_ps(ADDRESS_OUTPUT_NPQK(n, ap, aq, ik),
                _mm256_add_ps(_mm256_load_ps(ADDRESS_OUTPUT_NPQK(n, ap, aq, ik)), Ovec[bv]));
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
    }
  }
}

#endif  // SRC_UTILS_VECTOR_CONVOLUTION_FORWARD_SSE_QBLOCK_AVX_INL_H_
