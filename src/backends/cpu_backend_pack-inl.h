#ifndef SRC_BACKENDS_CPU_BACKEND_PACK_INL_H_
#define SRC_BACKENDS_CPU_BACKEND_PACK_INL_H_

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
    UnpackImpl<CPUTensor, DType, BLITZ_BUFFER_NHWC>(
      I, unpack,
      C, H, W, R, S, P, Q,
      pad_h, pad_w, str_h, str_w);
  } else if (input_data_layout == BLITZ_BUFFER_NCHW) {
    UnpackImpl<CPUTensor, DType, BLITZ_BUFFER_NCHW>(
      I, unpack,
      C, H, W, R, S, P, Q,
      pad_h, pad_w, str_h, str_w);
  } else {
    LOG(FATAL) << "Unsupported input data layout: " << input_data_layout;
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
    PackImpl<CPUTensor, DType, BLITZ_BUFFER_NHWC>(
      unpack, I,
      C, H, W, R, S, P, Q,
      pad_h, pad_w, str_h, str_w);
  } else if (input_data_layout == BLITZ_BUFFER_NCHW) {
    PackImpl<CPUTensor, DType, BLITZ_BUFFER_NCHW>(
      unpack, I,
      C, H, W, R, S, P, Q,
      pad_h, pad_w, str_h, str_w);
  } else {
    LOG(FATAL) << "Unsupported unpack data layout: " << input_data_layout;
  }
}

#endif  // SRC_BACKENDS_CPU_BACKEND_PACK_INL_H_
