#ifndef SRC_UTILS_VECTOR_CONVOLUTION_INL_H_
#define SRC_UTILS_VECTOR_CONVOLUTION_INL_H_

#ifdef BLITZ_SSE
#define CBLOCK 128
#define VEC_LEN 8  // register blocking
#define PQBLOCK 72 // divided by PQREG
#define KBLOCK 64 // divided by VEC_LEN * KREG
#define PQREG 6
#define KREG 2
#elif BLITZ_AVX
#define CBLOCK 128
#define VEC_LEN 8  // register blocking
#define PQBLOCK 72 // divided by PQREG
#define KBLOCK 128 // divided by VEC_LEN * KREG
#define PQREG 6
#define KREG 2
#elif BLITZ_AVX2
#define CBLOCK 128
#define VEC_LEN 8  // register blocking
#define PQBLOCK 72 // divided by PQREG
#define KBLOCK 128 // divided by VEC_LEN * KREG
#define PQREG 6
#define KREG 2
#elif BLITZ_AVX3
#define CBLOCK 128
#define VEC_LEN 8  // register blocking
#define PQBLOCK 72 // divided by PQREG
#define KBLOCK 128 // divided by VEC_LEN * KREG
#define PQREG 6
#define KREG 2
#elif BLITZ_AVX512
#define CBLOCK 64
#define VEC_LEN 16
#define PQBLOCK 72 // divided by PQREG
#define KBLOCK 128 // divided by VEC_LEN * KREG
#define PQREG 6
#define KREG 4
#endif

static inline void ConvolutionForwardFPack(
  const float *F,
  float *F_pack,
  size_t R, size_t S, size_t C, size_t K,
  size_t r, size_t s, size_t ic, size_t ik) {
  size_t F_index = 0;
  // F_pack contiguous
  for (size_t bk = 0; bk < KBLOCK / (KREG * VEC_LEN); ++bk) {
    for (size_t bc = 0; bc < CBLOCK; ++bc) {
      size_t mk = bk * (KREG * VEC_LEN);
      #pragma unroll
      for (size_t rk = 0; rk < KREG * VEC_LEN; ++rk) {
        F_pack[F_index + rk] = ACCESS_FILTER_RSCK(r, s, (ic + bc), (ik + mk + rk)); 
      }
      F_index += KREG * VEC_LEN;
    }
  }
}

static inline void ConvolutionForwardIPack(
  const float *I,
  float *I_pack,
  size_t N, size_t H, size_t W, size_t Q,
  size_t str_h, size_t str_w,
  size_t pad_h, size_t pad_w,
  size_t r, size_t s, size_t rc,
  size_t iq, size_t ip) {
  size_t aq = iq;
  size_t ap = ip;
  for (size_t bpq = 0; bpq < PQBLOCK; ++bpq) {
    int ih_index = static_cast<int>(ap * str_h) - static_cast<int>(pad_h) + static_cast<int>(r);
    int iw_index = static_cast<int>(aq * str_w) - static_cast<int>(pad_w) + static_cast<int>(s);
    if (ih_index >= 0 && ih_index < static_cast<int>(H)) {
      if (iw_index >= 0 && iw_index < static_cast<int>(W)) {
        #pragma unroll
        for (size_t bc = 0; bc < rc; ++bc) {
          I_pack[bpq * CBLOCK + bc] = ACCESS_INPUT_NHWC(n, ih_index, iw_index, (ic + bc));
        }
      } else {
        #pragma unroll
        for (size_t bc = 0; bc < CBLOCK; ++bc) {
          I_pack[bpq * CBLOCK + bc] = 0;
        }
      }
    } else {
      #pragma unroll
      for (size_t bc = 0; bc < CBLOCK; ++bc) {
        I_pack[bpq * CBLOCK + bc] = 0;
      }
    }
    aq += 1;
    if (aq >= Q) {
      ap += 1;
      aq = 0;
    } 
  }
}

static inline void ConvolutionForwardPQBlock(
  const float* I,
  const float* F,
  float* O,
  float* F_pack, float* I_pack,
  size_t N,
  size_t C, size_t H, size_t W,
  size_t R, size_t S,
  size_t K, size_t P, size_t Q,
  size_t str_h, size_t str_w,
  size_t pad_h, size_t pad_w,
  size_t ik, size_t ic, size_t rc,
  size_t ip, size_t iq, size_t pq) {
  #ifdef BLITZ_SSE
  __m128 Ovec[PQREG][KREG];
  __m128 Fvec[KREG];
  __m128 Ivec;
  #elif BLITZ_AVX
  __m256 Ovec[PQREG][KREG];
  __m256 Fvec[KREG];
  __m256 Ivec;
  #elif BLITZ_AVX2
  __m256 Ovec[PQREG][KREG];
  __m256 Fvec[KREG];
  __m256 Ivec;
  #elif BLITZ_AVX3
  __m512 Ovec[PQREG][KREG];
  __m512 Fvec[KREG];
  __m512 Ivec;
  #elif BLITZ_AVX512
  __m512 Ovec[PQREG][KREG];
  __m512 Fvec[KREG];
  __m512 Ivec;
  #endif
  #include "qblock_pack-inl.h"
  for (size_t bk = 0; bk < KBLOCK / (KREG * VEC_LEN); ++bk) {
    size_t mk = bk * (KREG * VEC_LEN);
    for (size_t bpq = 0; bpq < PQBLOCK / PQREG; ++bpq) {
      #pragma unroll
      for (size_t rpq = 0; rpq < PQREG; ++rpq) {
        #pragma unroll
        for (size_t rk = 0; rk < KREG; ++rk) {
          Ovec[rpq][rk] = FLOAT_SET1(0);
        }
      }
      for (size_t bc = 0; bc < rc; ++bc) {
        #pragma unroll
        for (size_t rk = 0; rk < KREG; ++rk) {
          Fvec[rk] = FLOAT_LOAD(F_pack + bk * (CBLOCK * KREG * VEC_LEN) + bc * (KREG * VEC_LEN) + rk * VEC_LEN);
        }
        #pragma unroll
        for (size_t rpq = 0; rpq < PQREG; ++rpq) {
          Ivec = FLOAT_SET1(*(I_pack + (bpq * PQREG + rpq) * CBLOCK + bc));
          #pragma unroll
          for (size_t rk = 0; rk < KREG; ++rk) {
            Ovec[rpq][rk] = FLOAT_ADD(FLOAT_MUL(Ivec, Fvec[rk]), Ovec[rpq][rk]);
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
          FLOAT_STORE(ADDRESS_OUTPUT_NPQK(n, ap, aq, (ik + mk + rk * VEC_LEN)),
              FLOAT_ADD(FLOAT_LOAD(ADDRESS_OUTPUT_NPQK(n, ap, aq, (ik + mk + rk * VEC_LEN))), Ovec[rpq][rk]));
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
}

#endif  // SRC_UTILS_VECTOR_CONVOLUTION_INL_H_
