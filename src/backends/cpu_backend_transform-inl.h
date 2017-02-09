#ifndef SRC_BACKENDS_CPU_BACKEND_TRANSFORM_INL_H_
#define SRC_BACKENDS_CPU_BACKEND_TRANSFORM_INL_H_

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
  switch (input_data_layout) {
    case BLITZ_BUFFER_NHWC:
      UnpackImpl<CPUTensor, DType, BLITZ_BUFFER_NHWC>(
        I, unpack,
        C, H, W, R, S, P, Q,
        pad_h, pad_w, str_h, str_w);
      break;
    case BLITZ_BUFFER_NCHW:
      UnpackImpl<CPUTensor, DType, BLITZ_BUFFER_NCHW>(
        I, unpack,
        C, H, W, R, S, P, Q,
        pad_h, pad_w, str_h, str_w);
      break;
    default:
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
  switch (input_data_layout) {
    case BLITZ_BUFFER_NHWC:
      PackImpl<CPUTensor, DType, BLITZ_BUFFER_NHWC>(
        unpack, I,
        C, H, W, R, S, P, Q,
        pad_h, pad_w, str_h, str_w);
      break;
    case BLITZ_BUFFER_NCHW:
      PackImpl<CPUTensor, DType, BLITZ_BUFFER_NCHW>(
        unpack, I,
        C, H, W, R, S, P, Q,
        pad_h, pad_w, str_h, str_w);
      break;
    default:
      LOG(FATAL) << "Unsupported input data layout: " << input_data_layout;
  }
}

template<typename DType>
void Backend<CPUTensor, DType>::TransformCopyFunc(
  const CPUTensor<DType>* source, CPUTensor<DType>* dest) {
  if (source->size() != dest->size()) {
    LOG(FATAL) << "Tensor size do not match!";
  }
  if (source->data_layout() == BLITZ_BUFFER_NCHW) {
    if (dest->data_layout() == BLITZ_BUFFER_NHWC) {
      TransformBufferImpl<CPUTensor, DType, BLITZ_BUFFER_NCHW, BLITZ_BUFFER_NHWC>(
        source->data(), dest->data(),
        source->shape()[0], source->shape()[1], source->shape()[2], source->shape()[3]);
      return;
    }
  } else if (source->data_layout() == BLITZ_BUFFER_NHWC) {
    if (dest->data_layout() == BLITZ_BUFFER_NCHW) {
      TransformBufferImpl<CPUTensor, DType, BLITZ_BUFFER_NHWC, BLITZ_BUFFER_NCHW>(
        source->data(), dest->data(),
        dest->shape()[0], dest->shape()[1], dest->shape()[2], dest->shape()[3]);
      return;
    }
  } else if (source->data_layout() == BLITZ_FILTER_KCRS) {
    if (dest->data_layout() == BLITZ_FILTER_RSCK) {
      TransformFilterImpl<CPUTensor, DType, BLITZ_FILTER_KCRS, BLITZ_FILTER_RSCK>(
        source->data(), dest->data(),
        source->shape()[0], source->shape()[1], source->shape()[2], source->shape()[3]);
      return;
    }
  } else if (source->data_layout() == BLITZ_FILTER_RSCK) {
    if (dest->data_layout() == BLITZ_FILTER_KCRS) {
      TransformFilterImpl<CPUTensor, DType, BLITZ_FILTER_RSCK, BLITZ_FILTER_KCRS>(
        source->data(), dest->data(),
        dest->shape()[0], dest->shape()[1], dest->shape()[2], dest->shape()[3]);
      return;
    }
  }
  BlitzCPUCopy(source->data(), dest->data(), source->size());
}

#endif  // SRC_BACKENDS_CPU_BACKEND_TRANSFORM_INL_H_
