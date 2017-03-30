#include "utils/blitz_impl_function.h"

#include <immintrin.h>

#ifdef USE_MKL
#include <mkl.h>
#else

#ifdef __cplusplus
  extern "C" {
#endif
    #include <cblas.h>
#ifdef __cplusplus
  }
#endif
#endif

#include "backends/cpu_tensor.h"

namespace blitz {

namespace utils {

#define ACCESS_INPUT_NCHW(i, j, k, v) I[((i * C + j) * H + k) * W + v]
#define ACCESS_OUTPUT_NKPQ(i, j, k, v) O[((i * K + j) * P + k) * Q + v]
#define ACCESS_FILTER_KCRS(i, j, k, v) F[((i * C + j) * R + k) * S + v]

#define ACCESS_INPUT_NHWC(i, j, k, v) I[((i * H + j) * W + k) * C + v]
#define ACCESS_OUTPUT_NPQK(i, j, k, v) O[((i * P + j) * Q + k) * K + v]
#define ACCESS_FILTER_RSCK(i, j, k, v) F[((i * S + j) * C + k) * K + v]

#define ADDRESS_OUTPUT_NPQK(i, j, k, v) (O + ((i * P + j) * Q + k) * K + v)
#define ADDRESS_FILTER_RSCK(i, j, k, v) (F + ((i * S + j) * C + k) * K + v)

template<>
void ConvolutionForwardNaiveImpl<CPUTensor, float, BLITZ_BUFFER_NCHW>(
  const float* I,
  const float* F,
  float* O,
  size_t N,
  size_t C, size_t H, size_t W,
  size_t R, size_t S,
  size_t K, size_t P, size_t Q,
  size_t pad_h, size_t pad_w,
  size_t str_h, size_t str_w) {
  if (pad_h == 0 && pad_w == 0) { //fast path
    #pragma omp parallel for
    for (size_t n = 0; n < N; ++n) {
      for (size_t k = 0; k < K; ++k) {
        for (size_t c = 0; c < C; ++c) {
          for (size_t p = 0; p < P; ++p) {
            int ih = p * str_h;
            for (size_t q = 0; q < Q; ++q) {
              int iw = q * str_w;
              for (size_t r = 0; r < R; ++r) {
                for (size_t s = 0; s < S; ++s) {
                  ACCESS_OUTPUT_NKPQ(n, k, p, q) += ACCESS_INPUT_NCHW(n, c, (ih + r), (iw + s)) *
                    ACCESS_FILTER_KCRS(k, c, r, s); 
                }
              }
            }
          }
        }
      }
    }
    return;
  }
  #pragma omp parallel for
  for (size_t n = 0; n < N; ++n) {
    for (size_t k = 0; k < K; ++k) {
      for (size_t c = 0; c < C; ++c) {
        for (size_t p = 0; p < P; ++p) {
          int ih = p * str_h - pad_h;
          for (size_t q = 0; q < Q; ++q) {
            int iw = q * str_w - pad_w;
            size_t r_end = ih + R < H ? R : H - ih;
            size_t s_end = iw + S < W ? S : W - iw;
            size_t r = ih < 0 ? -ih : 0;
            for (; r < r_end; ++r) {
              size_t s = iw < 0 ? -iw : 0;
              for (; s < s_end; ++s) {
                ACCESS_OUTPUT_NKPQ(n, k, p, q) += ACCESS_INPUT_NCHW(n, c, (ih + r), (iw + s)) *
                  ACCESS_FILTER_KCRS(k, c, r, s); 
              }
            }
          }
        }
      }
    }
  }
}

template<>
void ConvolutionForwardNaiveImpl<CPUTensor, float, BLITZ_BUFFER_NHWC>(
  const float* I,
  const float* F,
  float* O,
  size_t N,
  size_t C, size_t H, size_t W,
  size_t R, size_t S,
  size_t K, size_t P, size_t Q,
  size_t pad_h, size_t pad_w,
  size_t str_h, size_t str_w) {
  #pragma omp parallel for
  for (size_t n = 0; n < N; ++n) {
    for (size_t p = 0; p < P; ++p) {
      int ih = p * str_h - pad_h;
      for (size_t q = 0; q < Q; ++q) {
        int iw = q * str_w - pad_w;
        size_t r_end = ih + R < H ? R : H - ih;
        size_t s_end = iw + S < W ? S : W - iw;
        size_t r = ih < 0 ? -ih : 0;
        for (; r < r_end; ++r) {
          size_t s = iw < 0 ? -iw : 0;
          for (; s < s_end; ++s) {
            for (size_t k = 0; k < K; ++k) {
              for (size_t c = 0; c < C; ++c) {
                ACCESS_OUTPUT_NPQK(n, p, q, k) += ACCESS_INPUT_NHWC(n, (ih + r), (iw + s), c) *
                  ACCESS_FILTER_RSCK(r, s, c, k); 
              }
            }
          }
        }
      }
    }
  }
}

template<>
void ConvolutionBackwardNaiveImpl<CPUTensor, float, BLITZ_BUFFER_NCHW>(
  const float* O,
  const float* F,
  float* I,
  size_t N,
  size_t C, size_t H, size_t W,
  size_t R, size_t S,
  size_t K, size_t P, size_t Q,
  size_t pad_h, size_t pad_w,
  size_t str_h, size_t str_w) {
  if (pad_h == 0 && pad_w == 0) {
    #pragma omp parallel for
    for (size_t n = 0; n < N; ++n) {
      for (size_t c = 0; c < C; ++c) {
        for (size_t k = 0; k < K; ++k) {
          for (size_t p = 0; p < P; ++p) {
            int ih = p * str_h;
            for (size_t q = 0; q < Q; ++q) {
              int iw = q * str_w;
              for (size_t r = 0; r < R; ++r) {
                for (size_t s = 0; s < S; ++s) {
                  ACCESS_INPUT_NCHW(n, c, (ih + r), (iw + s)) += ACCESS_OUTPUT_NKPQ(n, k, p, q) *
                    ACCESS_FILTER_KCRS(k, c, r, s); 
                }
              }
            }
          }
        }
      }
    }
  }
  #pragma omp parallel for
  for (size_t n = 0; n < N; ++n) {
    for (size_t c = 0; c < C; ++c) {
      for (size_t k = 0; k < K; ++k) {
        for (size_t p = 0; p < P; ++p) {
          int ih = p * str_h - pad_h;
          for (size_t q = 0; q < Q; ++q) {
            int iw = q * str_w - pad_w;
            size_t r_end = ih + R < H ? R : H - ih;
            size_t s_end = iw + S < W ? S : W - iw;
            size_t r = ih < 0 ? -ih : 0;
            for (; r < r_end; ++r) {
              size_t s = iw < 0 ? -iw : 0;
              for (; s < s_end; ++s) {
                ACCESS_INPUT_NCHW(n, c, (ih + r), (iw + s)) += ACCESS_OUTPUT_NKPQ(n, k, p, q) *
                  ACCESS_FILTER_KCRS(k, c, r, s); 
              }
            }
          }
        }
      }
    }
  }
}

template<>
void ConvolutionBackwardNaiveImpl<CPUTensor, float, BLITZ_BUFFER_NHWC>(
  const float* O,
  const float* F,
  float* I,
  size_t N,
  size_t C, size_t H, size_t W,
  size_t R, size_t S,
  size_t K, size_t P, size_t Q,
  size_t pad_h, size_t pad_w,
  size_t str_h, size_t str_w) {
  #pragma omp parallel for
  for (size_t n = 0; n < N; ++n) {
    for (size_t p = 0; p < P; ++p) {
      int ih = p * str_h - pad_h;
      for (size_t q = 0; q < Q; ++q) {
        int iw = q * str_w - pad_w;
        size_t r_end = ih + R < H ? R : H - ih;
        size_t s_end = iw + S < W ? S : W - iw;
        size_t r = ih < 0 ? -ih : 0;
        for (; r < r_end; ++r) {
          size_t s = iw < 0 ? -iw : 0;
          for (; s < s_end; ++s) {
            for (size_t k = 0; k < K; ++k) {
              for (size_t c = 0; c < C; ++c) {
                ACCESS_INPUT_NHWC(n, (ih + r), (iw + s), c) += ACCESS_OUTPUT_NPQK(n, p, q, k) *
                  ACCESS_FILTER_RSCK(r, s, c, k);
              }
            }
          }
        }
      }
    }
  }
}

template<>
void ConvolutionUpdateNaiveImpl<CPUTensor, float, BLITZ_BUFFER_NCHW>(
  const float* I,
  const float* O,
  float* F,
  size_t N,
  size_t C, size_t H, size_t W,
  size_t R, size_t S,
  size_t K, size_t P, size_t Q,
  size_t pad_h, size_t pad_w,
  size_t str_h, size_t str_w) {
  if (pad_h == 0 && pad_w == 0) {
    #pragma omp parallel for
    for (size_t k = 0; k < K; ++k) {
      for (size_t c = 0; c < C; ++c) {
        for (size_t n = 0; n < N; ++n) {
          for (size_t p = 0; p < P; ++p) {
            int ih = p * str_h;
            for (size_t q = 0; q < Q; ++q) {
              int iw = q * str_w;
              for (size_t r = 0; r < R; ++r) {
                for (size_t s = 0; s < S; ++s) {
                  ACCESS_FILTER_KCRS(k, c, r, s) += ACCESS_INPUT_NCHW(n, c, (ih + r), (iw + s)) *
                    ACCESS_OUTPUT_NKPQ(n, k, p, q); 
                }
              }
            }
          }
        }
      }
    }
  }
  #pragma omp parallel for
  for (size_t k = 0; k < K; ++k) {
    for (size_t c = 0; c < C; ++c) {
      for (size_t n = 0; n < N; ++n) {
        for (size_t p = 0; p < P; ++p) {
          int ih = p * str_h - pad_h;
          for (size_t q = 0; q < Q; ++q) {
            int iw = q * str_w - pad_w;
            size_t r_end = ih + R < H ? R : H - ih;
            size_t s_end = iw + S < W ? S : W - iw;
            size_t r = ih < 0 ? -ih : 0;
            for (; r < r_end; ++r) {
              size_t s = iw < 0 ? -iw : 0;
              for (; s < s_end; ++s) {
                ACCESS_FILTER_KCRS(k, c, r, s) += ACCESS_INPUT_NCHW(n, c, (ih + r), (iw + s)) *
                  ACCESS_OUTPUT_NKPQ(n, k, p, q); 
              }
            }
          }
        }
      }
    }
  }
}

template<>
void ConvolutionUpdateNaiveImpl<CPUTensor, float, BLITZ_BUFFER_NHWC>(
  const float* I,
  const float* O,
  float* F,
  size_t N,
  size_t C, size_t H, size_t W,
  size_t R, size_t S,
  size_t K, size_t P, size_t Q,
  size_t pad_h, size_t pad_w,
  size_t str_h, size_t str_w) {
  #pragma omp parallel for
  for (size_t k = 0; k < K; ++k) {
    for (size_t n = 0; n < N; ++n) {
      for (size_t p = 0; p < P; ++p) {
        int ih = p * str_h - pad_h;
        for (size_t q = 0; q < Q; ++q) {
          int iw = q * str_w - pad_w;
          size_t r_end = ih + R < H ? R : H - ih;
          size_t s_end = iw + S < W ? S : W - iw;
          size_t r = ih < 0 ? -ih : 0;
          for (; r < r_end; ++r) {
            size_t s = iw < 0 ? -iw : 0;
            for (; s < s_end; ++s) {
              for (size_t c = 0; c < C; ++c) {
                ACCESS_FILTER_RSCK(r, s, c, k) += ACCESS_INPUT_NHWC(n, (ih + r), (iw + s), c) *
                  ACCESS_OUTPUT_NPQK(n, p, q, k); 
              }
            }
          }
        }
      }
    }
  }
}

template<>
void ConvolutionForwardVectorImpl<CPUTensor, float, BLITZ_BUFFER_NHWC>(
  const float* I,
  const float* F,
  float* O,
  size_t N,
  size_t C, size_t H, size_t W,
  size_t R, size_t S,
  size_t K, size_t P, size_t Q,
  size_t pad_h, size_t pad_w,
  size_t str_h, size_t str_w) {
  #ifdef BLITZ_SSE
  #define VEC_LEN 4
  __m128 Ovec[VEC_LEN];
  __m128 Fvec;
  __m128 Ivec;
  #elif BLITZ_AVX
  #define CBLOCK 192
  #define VEC_LEN 8  // register blocking
  #define PQBLOCK 108 // divided by PQREG
  #define KBLOCK 128 // divided by VEC_LEN * KREG
  #define PQREG 6
  #define KREG 2
  __m256 Ovec[PQREG][KREG];
  __m256 Fvec[KREG];
  __m256 Ivec;
  #elif BLITZ_AVX2
  #define CBLOCK 128
  #define VEC_LEN 8  // register blocking
  #define PQBLOCK 72 // divided by PQREG
  #define KBLOCK 128 // divided by VEC_LEN * KREG
  #define PQREG 2
  #define KREG 4
  __m256 Ovec[PQREG][KREG];
  __m256 Fvec[KREG];
  __m256 Ivec;
  #elif BLITZ_AVX3
  #define CBLOCK 128
  #define VEC_LEN 8  // register blocking
  #define PQBLOCK 72 // divided by PQREG
  #define KBLOCK 128 // divided by VEC_LEN * KREG
  #define PQREG 2
  #define KREG 4
  __m512 Ovec[PQREG][KREG];
  __m512 Fvec[KREG];
  __m512 Ivec;
  #elif BLITZ_AVX512
  #define CBLOCK 64
  #define VEC_LEN 8  // register blocking
  #define PQBLOCK 72 // divided by PQREG
  #define KBLOCK 128 // divided by VEC_LEN * KREG
  #define PQREG 2
  #define PQREG 6
  #define KREG 4
  __m512 Ovec[PQREG][KREG];
  __m512 Fvec[KREG];
  __m512 Ivec;
  #endif
  if (K % (KREG * VEC_LEN)) {
    LOG(FATAL) << "Not supported K, please set it as a multiple of: " << VEC_LEN * KREG;
  }
  float I_pack[PQBLOCK * CBLOCK];
  float F_pack[CBLOCK * KBLOCK];
  #pragma omp parallel for private(F_pack, I_pack, Ovec, Fvec, Ivec)
  for (size_t n = 0; n < N; ++n) {
    for (size_t k = 0; k < K / KBLOCK; ++k) {
      size_t ik = k * KBLOCK;
      for (size_t c = 0; c < C / CBLOCK; ++c) {
        size_t ic = c * CBLOCK;
        size_t lc = CBLOCK;
        for (size_t r = 0; r < R; ++r) {
          for (size_t s = 0; s < S; ++s) {
            size_t F_index = 0;
            // F_pack contiguous
            for (size_t bk = 0; bk < KBLOCK / (KREG * VEC_LEN); ++bk) {
              #pragma unroll
              for (size_t bc = 0; bc < CBLOCK; ++bc) {
                size_t mk = bk * (KREG * VEC_LEN);
                #pragma unroll
                for (size_t rk = 0; rk < KREG * VEC_LEN; ++rk) {
                  F_pack[F_index + rk] = ACCESS_FILTER_RSCK(r, s, (ic + bc), (ik + mk + rk)); 
                }
                F_index += KREG * VEC_LEN;
              }
            }
            for (size_t pq = 0; pq < P * Q / PQBLOCK; ++pq) {
              size_t ip = (pq * PQBLOCK) / Q;
              size_t iq = (pq * PQBLOCK) % Q;
              size_t lpq = PQBLOCK;
              #ifdef BLITZ_SSE
              #include "vector/convolution_forward_qblock_sse-inl.h"
              #elif BLITZ_AVX
              #include "vector/convolution_forward_qblock_avx-inl.h"
              #elif BLITZ_AVX2
              #include "vector/convolution_forward_qblock_avx2-inl.h"
              #elif BLITZ_AVX3
              #include "vector/convolution_forward_qblock_avx3-inl.h"
              #elif BLITZ_AVX512
              #include "vector/convolution_forward_qblock_avx512-inl.h"
              #endif
            }
            size_t ip = (P * Q / PQBLOCK) * PQBLOCK / Q;  // p remainder
            size_t iq = (P * Q / PQBLOCK) * PQBLOCK % Q;  // q remainder
            size_t lpq = P * Q - (P * Q / PQBLOCK) * PQBLOCK;
            lpq = lpq % PQREG ? ((lpq - 1) / PQREG + 1) * PQREG : lpq;
            #ifdef BLITZ_SSE
            #include "vector/convolution_forward_qblock_sse-inl.h"
            #elif BLITZ_AVX
            #include "vector/convolution_forward_qblock_avx-inl.h"
            #elif BLITZ_AVX2
            #include "vector/convolution_forward_qblock_avx2-inl.h"
            #elif BLITZ_AVX3
            #include "vector/convolution_forward_qblock_avx3-inl.h"
            #elif BLITZ_AVX512
            #include "vector/convolution_forward_qblock_avx512-inl.h"
            #endif
          }
        }
      }
    }
  }
}


template<>
void TransformBufferImpl<CPUTensor, float, BLITZ_BUFFER_NCHW, BLITZ_BUFFER_NHWC>(
  const float* nchw,
  float* nhwc,
  size_t N,
  size_t C, size_t H, size_t W) {
  #pragma omp parallel for
  for (size_t n = 0; n < N; ++n) {
    const float *chw = nchw + n * C * H * W;
    float *hwc = nhwc + n * C * H * W;
    for (size_t c = 0; c < C; ++c) {
      for (size_t h = 0; h < H; ++h) {
        for (size_t w = 0; w < W; ++w) {
          hwc[h * W * C + w * C + c] = chw[c * H * W + h * W + w];
        }
      }
    }
  }
}

template<>
void TransformBufferImpl<CPUTensor, float, BLITZ_BUFFER_NHWC, BLITZ_BUFFER_NCHW>(
  const float* nhwc,
  float* nchw,
  size_t N,
  size_t C, size_t H, size_t W) {
  #pragma omp parallel for
  for (size_t n = 0; n < N; ++n) {
    const float *hwc = nhwc + n * C * H * W;
    float *chw = nchw + n * C * H * W;
    for (size_t c = 0; c < C; ++c) {
      for (size_t h = 0; h < H; ++h) {
        for (size_t w = 0; w < W; ++w) {
          chw[c * H * W + h * W + w] = hwc[h * W * C + w * C + c];
        }
      }
    }
  }
}

template<>
void TransformFilterImpl<CPUTensor, float, BLITZ_FILTER_KCRS, BLITZ_FILTER_RSCK>(
  const float* kcrs,
  float* rsck,
  size_t K,
  size_t C, size_t R, size_t S) {
  #pragma omp parallel for
  for (size_t k = 0; k < K; ++k) {
    for (size_t c = 0; c < C; ++c) {
      for (size_t r = 0; r < R; ++r) {
        for (size_t s = 0; s < S; ++s) {
          rsck[((r * S + s) * C + c) * K + k] = kcrs[((k * C + c) * R + r) * S + s];
        }
      }
    }
  }
}

template<>
void TransformFilterImpl<CPUTensor, float, BLITZ_FILTER_RSCK, BLITZ_FILTER_KCRS>(
  const float* rsck,
  float* kcrs,
  size_t K,
  size_t C, size_t R, size_t S) {
  #pragma omp parallel for
  for (size_t k = 0; k < K; ++k) {
    for (size_t c = 0; c < C; ++c) {
      for (size_t r = 0; r < R; ++r) {
        for (size_t s = 0; s < S; ++s) {
          kcrs[((k * C + c) * R + r) * S + s] = rsck[((r * S + s) * C + c) * K + k];
        }
      }
    }
  }
}

template<>
void UnpackImpl<CPUTensor, float, BLITZ_BUFFER_NCHW>(
  const float* I,
  float* unpack,
  size_t C, size_t H, size_t W,
  size_t R, size_t S,
  size_t P, size_t Q,
  size_t pad_h, size_t pad_w,
  size_t str_h, size_t str_w) {
  size_t unpack_index = 0;
  for (size_t c = 0; c < C; ++c) {
    const size_t cHW = c * H * W;
    const float* I_slice = I + cHW;
    for (size_t r = 0; r < R; ++r) {
      for (size_t s = 0; s < S; ++s) {
        int R_offset = -pad_h + r;
        for (size_t p = 0; p < P; ++p) {
          if (R_offset < 0 || R_offset >= static_cast<int>(H)) {
            for (size_t q = 0; q < Q; ++q) {
              unpack[unpack_index++] = 0;
            }
          } else {
            int S_offset = -pad_w + s;
            for (size_t q = 0; q < Q; ++q) {
              if (S_offset < 0 || S_offset >= static_cast<int>(W)) {
                unpack[unpack_index++] = 0;
              } else {
                unpack[unpack_index++] = I_slice[R_offset * W + S_offset];
              }
              S_offset += str_w;
            }
          }
          R_offset += str_h;
        }
      }
    }
  }
}

template<>
void UnpackImpl<CPUTensor, float, BLITZ_BUFFER_NHWC>(
  const float* I,
  float* unpack,
  size_t C, size_t H, size_t W,
  size_t R, size_t S,
  size_t P, size_t Q,
  size_t pad_h, size_t pad_w,
  size_t str_h, size_t str_w) {
  // borrow from caffe2
  int R_offset = -pad_h;
  for (size_t p = 0; p < P; ++p) {
    int S_offset = -pad_w;
    for (size_t q = 0; q < Q; ++q) {
      for (int h = R_offset; h < static_cast<int>(R) + R_offset; ++h) {
        for (int w = S_offset; w < static_cast<int>(S) + S_offset; ++w) {
          if (h >= 0 && h < static_cast<int>(H) && w >= 0 && w < static_cast<int>(W)) {
            for(size_t c = 0; c < C; ++c) {
              unpack[c] = I[(h * W + w) * C + c];
            }
          } else {
            memset(unpack, 0, sizeof(float) * C);
          }
          unpack += C;
        }
      }
      S_offset += str_w;
    }
    R_offset += str_h;
  }
}

template<>
void PackImpl<CPUTensor, float, BLITZ_BUFFER_NCHW>(
  const float* unpack,
  float* I,
  size_t C, size_t H, size_t W,
  size_t R, size_t S,
  size_t P, size_t Q,
  size_t pad_h, size_t pad_w,
  size_t str_h, size_t str_w) {
  size_t unpack_index = 0;
  for (size_t c = 0; c < C; ++c) {
    const size_t cHW = c * H * W;
    float* I_slice = I + cHW;
    for (size_t r = 0; r < R; ++r) {
      for (size_t s = 0; s < S; ++s) {
        int R_offset = -pad_h + r;
        for (size_t p = 0; p < P; ++p) {
          if (R_offset < 0 || R_offset >= static_cast<int>(H)) {
            unpack_index += Q;
          } else {
            int S_offset = -pad_w + s;
            for (size_t q = 0; q < Q; ++q) {
              if (S_offset >= 0 && S_offset < static_cast<int>(W)) {
                I_slice[R_offset * W + S_offset] += unpack[unpack_index];
              }
              unpack_index++;
              S_offset += str_w;
            }
          }
          R_offset += str_h;
        }
      }
    }
  }
}

template<>
void PackImpl<CPUTensor, float, BLITZ_BUFFER_NHWC>(
  const float* unpack,
  float* I,
  size_t C, size_t H, size_t W,
  size_t R, size_t S,
  size_t P, size_t Q,
  size_t pad_h, size_t pad_w,
  size_t str_h, size_t str_w) {
  size_t unpack_index = 0;
  int R_offset = -pad_h;
  for (size_t p = 0; p < P; ++p) {
    int S_offset = -pad_w;
    for (size_t q = 0; q < Q; ++q) {
      for (int h = R_offset; h < static_cast<int>(R) + R_offset; ++h) {
        for (int w = S_offset; w < static_cast<int>(S) + S_offset; ++w) {
          if (h >= 0 && h < static_cast<int>(H) && w >= 0 && w < static_cast<int>(W)) {
            float* I_slice = I + (h * W + w) * C;
            for (size_t c = 0; c < C; ++c) {
              I_slice[c] += unpack[unpack_index + c];
            }
          }
          unpack_index += C;
        }
      }
      S_offset += str_w;
    }
    R_offset += str_h;
  }
}

template<>
void MaxPoolingForwardImpl<CPUTensor, float, BLITZ_BUFFER_NCHW>(
  const float* I,
  float* O,
  size_t* max_index,
  size_t N,
  size_t C, size_t H, size_t W,
  size_t K, size_t P, size_t Q,
  size_t R, size_t S,
  size_t str_h, size_t str_w) {
  // offset
  const size_t HW = H * W;
  const size_t CHW = C * HW;
  const size_t PQ = P * Q;
  const size_t KPQ = K * PQ;
  #pragma omp parallel for
  for (size_t n = 0; n < N; ++n) {
    for (size_t c = 0; c < C; ++c) {
      const float* input_slice = I + n * CHW + c * HW;
      float* output_slice = O + n * KPQ + c * PQ;
      size_t* max_index_slice = max_index + n * KPQ + c * PQ;
      for (size_t oh = 0; oh < P; ++oh) {
        for (size_t ow = 0; ow < Q; ++ow) {
          size_t hs = oh * str_h;
          size_t ws = ow * str_w;
          size_t he = hs + R;
          size_t we = ws + S;
          size_t pool_index = oh * Q + ow;
          max_index_slice[pool_index] = hs * W + ws;
          for (size_t h = hs; h < he; ++h) {
            for (size_t w = ws; w < we; ++w) {
              size_t index = h * W + w;
              if (input_slice[index] > input_slice[max_index_slice[pool_index]]) {
                max_index_slice[pool_index] = index;
              }
            }
          }
          output_slice[pool_index] = input_slice[max_index_slice[pool_index]];
        }
      }
    }
  }
}

template<>
void MaxPoolingForwardImpl<CPUTensor, float, BLITZ_BUFFER_NHWC>(
  const float* I,
  float* O,
  size_t* max_index,
  size_t N,
  size_t C, size_t H, size_t W,
  size_t K, size_t P, size_t Q,
  size_t R, size_t S,
  size_t str_h, size_t str_w) {
  const size_t HWC = H * W * C;
  const size_t PQK = P * Q * K;
  #pragma omp parallel for
  for (size_t n = 0; n < N; ++n) {
    const float* input_slice = I + n * HWC;
    float* output_slice = O + n * PQK;
    size_t* max_index_slice = max_index + n * PQK;
    for (size_t oh = 0; oh < P; ++oh) {
      for (size_t ow = 0; ow < Q; ++ow) {
        const size_t hs = oh * str_h;
        const size_t ws = ow * str_w;
        const size_t he = hs + R;
        const size_t we = ws + S;
        const size_t pool_index = (oh * Q + ow) * C;
        for (size_t c = 0; c < C; ++c) {
          max_index_slice[pool_index + c] = (hs * W + ws) * C + c;
        }
        for (size_t h = hs; h < he; ++h) {
          for (size_t w = ws; w < we; ++w) {
            for (size_t c = 0; c < C; ++c) {
              size_t index = (h * W + w) * C + c;
              if (input_slice[index] > input_slice[max_index_slice[pool_index + c]]) {
                max_index_slice[pool_index + c] = index;
              }
            }
          }
        }
        for (size_t c = 0; c < C; ++c) {
          output_slice[pool_index + c] = input_slice[max_index_slice[pool_index + c]];
        }
      }
    }
  }
}

template<>
void MaxPoolingBackwardImpl<CPUTensor, float, BLITZ_BUFFER_NCHW>(
  const float* O,
  float* I,
  const size_t* max_index,
  size_t N,
  size_t C, size_t H, size_t W,
  size_t K, size_t P, size_t Q) {
  const size_t HW = H * W;
  const size_t CHW = C * HW;
  const size_t PQ = P * Q;
  const size_t KPQ = K * PQ;
  #pragma omp parallel for
  for (size_t n = 0; n < N; ++n) {
    for (size_t c = 0; c < C; ++c) {
      float* input_slice = I + n * CHW + c * HW;
      const float* output_slice = O + n * KPQ + c * PQ;
      const size_t* max_index_slice = max_index + n * KPQ + c * PQ;
      for (size_t oh = 0; oh < P; ++oh) {
        for (size_t ow = 0; ow < Q; ++ow) {
          input_slice[max_index_slice[oh * Q + ow]] = output_slice[oh * Q + ow];
        }
      }
    }
  }
}

template<>
void MaxPoolingBackwardImpl<CPUTensor, float, BLITZ_BUFFER_NHWC>(
  const float* O,
  float* I,
  const size_t* max_index,
  size_t N,
  size_t C, size_t H, size_t W,
  size_t K, size_t P, size_t Q) {
  const size_t CHW = C * H * W;
  const size_t KPQ = K * P * Q;
  #pragma omp parallel for
  for (size_t n = 0; n < N; ++n) {
    float* input_slice = I + n * CHW;
    const float* output_slice = O + n * KPQ;
    const size_t* max_index_slice = max_index + n * KPQ;
    for (size_t oh = 0; oh < P; ++oh) {
      for (size_t ow = 0; ow < Q; ++ow) {
        for (size_t c = 0; c < C; ++c) {
          input_slice[max_index_slice[(oh * Q + ow) * C + c]] = output_slice[(oh * Q + ow) * C + c];
        }
      }
    }
  }
}

template<>
void Gemm<CPUTensor, float>(
  float* A, float* B, float* C,
  bool transa, bool transb,
  float alpha, float beta,
  size_t M, size_t N, size_t K) {
  CBLAS_TRANSPOSE TransA = transa ? CblasTrans : CblasNoTrans;
  size_t lda = transa ? M : K;
  CBLAS_TRANSPOSE TransB = transb ? CblasTrans : CblasNoTrans;
  size_t ldb = transb ? K : N;
  cblas_sgemm(CblasRowMajor,
    TransA, TransB, 
    M, N, K,
    alpha,
    A, lda, B, ldb,
    beta,
    C, N);
}

template<>
void Gemm<CPUTensor, double>(
  double* A, double* B, double* C,
  bool transa, bool transb,
  double alpha, double beta,
  size_t M, size_t N, size_t K) {
  CBLAS_TRANSPOSE TransA = transa ? CblasTrans : CblasNoTrans;
  size_t lda = transa ? M : K;
  CBLAS_TRANSPOSE TransB = transb ? CblasTrans : CblasNoTrans;
  size_t ldb = transb ? K : N;
  cblas_dgemm(CblasRowMajor,
    TransA, TransB,
    M, N, K,
    alpha,
    A, lda,
    B, ldb,
    beta,
    C, N);
}

}  // namespace utils

}  // namespace blitz
