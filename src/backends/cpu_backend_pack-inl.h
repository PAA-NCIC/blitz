#ifndef SRC_BACKENDS_CPU_BACKEND_PACK_INL_H_
#define SRC_BACKENDS_CPU_BACKEND_PACK_INL_H_

template<typename DType>
void Backend<CPUTensor, DType>::UnpackCHWImpl(
  const DType* I,
  DType* unpack,
  size_t C, size_t H, size_t W,
  size_t R, size_t S,
  size_t P, size_t Q,
  size_t pad_h, size_t pad_w,
  size_t str_h, size_t str_w) {
  size_t unpack_index = 0;
  for (size_t c = 0; c < C; ++c) {
    const size_t cHW = c * H * W;
    const DType* I_slice = I + cHW;
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

template<typename DType>
void Backend<CPUTensor, DType>::UnpackHWCImpl(
  const DType* I,
  DType* unpack,
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
            memset(unpack, 0, sizeof(DType) * C);
          }
          unpack += C;
        }
      }
      S_offset += str_w;
    }
    R_offset += str_h;
  }
}

template<typename DType>
void Backend<CPUTensor, DType>::Unpack2DFunc(
  const DType* I,
  DType* unpack,
  size_t C, size_t H, size_t W,
  size_t R, size_t S,
  size_t P, size_t Q,
  size_t pad_h, size_t pad_w,
  size_t str_h, size_t str_w,
  BLITZ_DATA_LAYOUT input_data_layout) {
  if (input_data_layout == BLITZ_BUFFER_NHWC) {
    UnpackHWCImpl(
      I, unpack,
      C, H, W, R, S, P, Q,
      pad_h, pad_w, str_h, str_w);
  } else if (input_data_layout == BLITZ_BUFFER_NCHW) {
    UnpackCHWImpl(
      I, unpack,
      C, H, W, R, S, P, Q,
      pad_h, pad_w, str_h, str_w);
  } else {
    LOG(FATAL) << "Unsupported input data layout: " << input_data_layout;
  }
}

template<typename DType>
void Backend<CPUTensor, DType>::PackCHWImpl(
  const DType* unpack,
  DType* I,
  size_t C, size_t H, size_t W,
  size_t R, size_t S,
  size_t P, size_t Q,
  size_t pad_h, size_t pad_w,
  size_t str_h, size_t str_w) {
  size_t unpack_index = 0;
  for (size_t c = 0; c < C; ++c) {
    const size_t cHW = c * H * W;
    DType* I_slice = I + cHW;
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

template<typename DType>
void Backend<CPUTensor, DType>::PackHWCImpl(
  const DType* unpack,
  DType* I,
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
            DType* I_slice = I + (h * W + w) * C;
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

template<typename DType>
void Backend<CPUTensor, DType>::Pack2DFunc(
  const DType* unpack,
  DType* I,
  size_t C, size_t H, size_t W,
  size_t R, size_t S,
  size_t P, size_t Q,
  size_t pad_h, size_t pad_w,
  size_t str_h, size_t str_w,
  BLITZ_DATA_LAYOUT input_data_layout) {
  if (input_data_layout == BLITZ_BUFFER_NHWC) {
    PackHWCImpl(
      unpack, I,
      C, H, W, R, S, P, Q,
      pad_h, pad_w, str_h, str_w);
  } else if (input_data_layout == BLITZ_BUFFER_NCHW) {
    PackCHWImpl(
      unpack, I,
      C, H, W, R, S, P, Q,
      pad_h, pad_w, str_h, str_w);
  } else {
    LOG(FATAL) << "Unsupported unpack data layout: " << input_data_layout;
  }
}

#endif  // SRC_BACKENDS_CPU_BACKEND_PACK_INL_H_
