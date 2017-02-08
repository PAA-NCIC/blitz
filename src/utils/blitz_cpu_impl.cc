#include "utils/blitz_impl_function.h"
#include "backends/cpu_tensor.h"

namespace blitz {

#define ACCESS_INPUT(i, j, k, v) *(I + ((i * C + j) * H + k) * W + v)
#define ACCESS_OUTPUT(i, j, k, v) *(O + ((i * K + j) * P + k) * Q + v)
#define ACCESS_FILTER(i, j, k, v) *(F + ((i * C + j) * R + k) * S + v)

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
  #pragma omp parallel for
  for (size_t n = 0; n < N; ++n) {
    for (size_t k = 0; k < K; ++k) {
      for (size_t c = 0; c < C; ++c) {
        for (size_t p = 0; p < P; ++p) {
          int ih = p * str_h - pad_h;
          for (size_t q = 0; q < Q; ++q) {
            int iw = q * str_w - pad_w;
            for (size_t r = 0; r < R; ++r) {
              if (ih + static_cast<int>(r) >= 0 && ih + static_cast<int>(r) < static_cast<int>(H)) {
                for (size_t s = 0; s < S; ++s) {
                  if (iw + static_cast<int>(s) >= 0 && iw + static_cast<int>(s) < static_cast<int>(W)) {
                    ACCESS_OUTPUT(n, k, p, q) += ACCESS_INPUT(n, c, (ih + r), (iw + s)) *
                      ACCESS_FILTER(k, c, r, s); 
                  }
                }
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
  #pragma omp parallel for
  for (size_t n = 0; n < N; ++n) {
    for (size_t c = 0; c < C; ++c) {
      for (size_t k = 0; k < K; ++k) {
        for (size_t p = 0; p < P; ++p) {
          int ih = p * str_h - pad_h;
          for (size_t q = 0; q < Q; ++q) {
            int iw = q * str_w - pad_w;
            for (size_t r = 0; r < R; ++r) {
              if (ih + static_cast<int>(r) >= 0 && ih + static_cast<int>(r) < static_cast<int>(H)) {
                for (size_t s = 0; s < S; ++s) {
                  if (iw + static_cast<int>(s) >= 0 && iw + static_cast<int>(s) < static_cast<int>(W)) {
                    ACCESS_INPUT(n, c, (ih + r), (iw + s)) += ACCESS_OUTPUT(n, k, p, q) *
                      ACCESS_FILTER(k, c, r, s); 
                  }
                }
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
  #pragma omp parallel for
  for (size_t c = 0; c < C; ++c) {
    for (size_t k = 0; k < K; ++k) {
      for (size_t n = 0; n < N; ++n) {
        for (size_t p = 0; p < P; ++p) {
          int ih = p * str_h - pad_h;
          for (size_t q = 0; q < Q; ++q) {
            int iw = q * str_w - pad_w;
            for (size_t r = 0; r < R; ++r) {
              if (ih + static_cast<int>(r) >= 0 && ih + static_cast<int>(r) < static_cast<int>(H)) {
                for (size_t s = 0; s < S; ++s) {
                  if (iw + static_cast<int>(s) >= 0 && iw + static_cast<int>(s) < static_cast<int>(W)) {
                    ACCESS_FILTER(k, c, r, s) += ACCESS_INPUT(n, c, (ih + r), (iw + s)) *
                      ACCESS_OUTPUT(n, k, p, q); 
                  }
                }
              }
            }
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
void TransformFilterImpl<CPUTensor, float, BLITZ_FILTER_KCRS, BLITZ_FILTER_KRSC>(
  const float* kcrs,
  float* krsc,
  size_t K,
  size_t C, size_t R, size_t S) {
  #pragma omp parallel for
  for (size_t k = 0; k < K; ++k) {
    for (size_t c = 0; c < C; ++c) {
      for (size_t r = 0; r < R; ++r) {
        for (size_t s = 0; s < S; ++s) {
          krsc[((k * R + r) * S + s) * C + c] = kcrs[((k * C + c) * R + r) * S + s];
        }
      }
    }
  }
}

template<>
void TransformFilterImpl<CPUTensor, float, BLITZ_FILTER_KRSC, BLITZ_FILTER_KCRS>(
  const float* krsc,
  float* kcrs,
  size_t K,
  size_t C, size_t R, size_t S) {
  #pragma omp parallel for
  for (size_t k = 0; k < K; ++k) {
    for (size_t s = 0; s < S; ++s) {
      for (size_t r = 0; r < R; ++r) {
        for (size_t c = 0; c < C; ++c) {
          kcrs[((k * C + c) * R + r) * S + s] = krsc[((k * R + r) * S + s) * C + c];
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

}  // namespace blitz
