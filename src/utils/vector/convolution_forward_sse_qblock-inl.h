#undef SRC_UTILS_VECTOR_CONVOLUTION_FORWARD_SSE_QBLOCK_INL_H_
#ifndef SRC_UTILS_VECTOR_CONVOLUTION_FORWARD_SSE_QBLOCK_INL_H_
#define SRC_UTILS_VECTOR_CONVOLUTION_FORWARD_SSE_QBLOCK_INL_H_

for (size_t r = 0; r < R; ++r) {
  if (ih + static_cast<int>(r) >= 0 && ih + static_cast<int>(r) < static_cast<int>(H)) {
    for (size_t s = 0; s < S; ++s) {
      for (size_t c = 0; c < C / 4; ++c) {
        for (size_t k = 0; k < K / 4; ++k) {
          int ih_index = ih + r;
          if (k == 0) {
            for (size_t bq = 0; bq < 4; ++bq) {
              int iw_index = static_cast<int>((iq + bq) * str_w) - static_cast<int>(pad_w) + static_cast<int>(s);
              if (iw_index >= 0 && iw_index < static_cast<int>(W)) {
                for (size_t bc = 0; bc < 4; ++bc) {
                  Ipack[4 * bc + bq] = ACCESS_INPUT_NHWC(n, ih_index, iw_index, (c * 4 + bc));
                }
              } else {
                for (size_t bc = 0; bc < 4; ++bc) {
                  Ipack[4 * bc + bq] = 0;
                }
              }
            }
          }
          Ovec[0] = Ovec[1] = Ovec[2] = Ovec[3] = _mm_set_ps1(0);
          for (size_t bc = 0; bc < 4; ++bc) {
            Fvec[bc] = _mm_load_ps(ADDRESS_FILTER_RSCK(r, s, (c * 4 + bc), (k * 4)));
            for (size_t bq = 0; bq < 4; ++bq) {
              Ivec[bq] = _mm_load_ps1(Ipack + bc * 4 + bq);
              Ovec[bq] = _mm_add_ps(_mm_mul_ps(Ivec[bq], Fvec[bc]), Ovec[bq]);
            }
          }
          for (size_t bq = 0; bq < rq; ++bq) {
            _mm_store_ps(ADDRESS_OUTPUT_NPQK(n, p, (iq + bq), (k * 4)),
              _mm_add_ps(_mm_load_ps(ADDRESS_OUTPUT_NPQK(n, p, (iq + bq), (k * 4))), Ovec[bq]));
          }
        }
      }
      size_t ic = C / 4 * 4; 
      size_t rc = C - ic;
      if (rc > 0) {
        for (size_t k = 0; k < K / 4; ++k) {
          int ih_index = ih + r;
          if (k == 0) {
            for (size_t bq = 0; bq < 4; ++bq) {
              int iw_index = static_cast<int>((iq + bq) * str_w) - static_cast<int>(pad_w) + static_cast<int>(s);
              if (iw_index >= 0 && iw_index < static_cast<int>(W)) {
                for (size_t bc = 0; bc < rc; ++bc) {
                  Ipack[4 * bc + bq] = ACCESS_INPUT_NHWC(n, ih_index, iw_index, (ic + bc));
                }
              } else {
                for (size_t bc = 0; bc < 4; ++bc) {
                  Ipack[4 * bc + bq] = 0;
                }
              }
            }
          }
          Ovec[0] = Ovec[1] = Ovec[2] = Ovec[3] = _mm_set_ps1(0);
          for (size_t bc = 0; bc < rc; ++bc) {
            Fvec[bc] = _mm_load_ps(ADDRESS_FILTER_RSCK(r, s, (ic + bc), (k * 4)));
            for (size_t bq = 0; bq < 4; ++bq) {
              Ivec[bq] = _mm_load_ps1(Ipack + bc * 4 + bq);
              Ovec[bq] = _mm_add_ps(_mm_mul_ps(Ivec[bq], Fvec[bc]), Ovec[bq]);
            }
          }
          for (size_t bq = 0; bq < rq; ++bq) {
            _mm_store_ps(ADDRESS_OUTPUT_NPQK(n, p, (iq + bq), (k * 4)),
              _mm_add_ps(_mm_load_ps(ADDRESS_OUTPUT_NPQK(n, p, (iq + bq), (k * 4))), Ovec[bq]));
          }
        }
      }
    }
  }
}

#endif  // SRC_UTILS_VECTOR_CONVOLUTION_FORWARD_SSE_QBLOCK_INL_H_
