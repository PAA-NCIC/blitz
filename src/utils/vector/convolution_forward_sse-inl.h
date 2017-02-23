#ifndef SRC_UTILS_VECTOR_CONVOLUTION_FORWARD_SSE_INL_H_
#define SRC_UTILS_VECTOR_CONVOLUTION_FORWARD_SSE_INL_H_

float Ipack[4 * 4];
__m128 Ovec[4];
__m128 Fvec[4];
__m128 Ivec[4];
#pragma omp parallel for private(Ipack, Ovec, Fvec, Ivec)
for (size_t n = 0; n < N; ++n) {
  for (size_t p = 0; p < P; ++p) {
    int ih = p * str_h - pad_h;
    size_t rq = 4;
    size_t iq = 0;
    for (size_t q = 0; q < Q / 4; ++q) {
      iq = q * 4;
      #include "convolution_forward_sse_qblock-inl.h"
    }
    iq = (Q / 4) * 4;  // q remainder
    rq = Q - iq;
    if (rq > 0) {
      #include "convolution_forward_sse_qblock-inl.h"
    }
  }
}

#endif  // SRC_UTILS_VECTOR_CONVOLUTION_FORWARD_SSE_INL_H_
