#ifndef SRC_BACKENDS_CPU_BACKEND_POOL_INL_H_
#define SRC_BACKENDS_CPU_BACKEND_POOL_INL_H_

static void MaxPooling2DForwardFunc(
  const CPUTensor<DType>* input,
  CPUTensor<DType>* output,
  CPUTensor<size_t>* max_index, 
  size_t R, size_t S,
  size_t str_w, size_t str_h) {
  // shape init
  size_t IN, C, H, W;
  size_t ON, K, P, Q;
  // shape decode
  CHECK_EQ(input->data_layout(), output->data_layout());
  Blitz2DBuffer(input->shape(), &IN, &C, &H, &W);
  Blitz2DBuffer(output->shape(), &ON, &K, &P, &Q);
  CHECK_EQ(IN, ON);
  CHECK_EQ(C, K);
  switch (input->data_layout()) {
    case BLITZ_BUFFER_NCHW:
      utils::MaxPoolingForwardImpl<CPUTensor, DType, BLITZ_BUFFER_NCHW>(
        input->data(),
        output->data(),
        max_index->data(),
        IN,
        C, H, W,
        K, P, Q,
        R, S,
        str_h, str_w);
      break;
    case BLITZ_BUFFER_NHWC:
      utils::MaxPoolingForwardImpl<CPUTensor, DType, BLITZ_BUFFER_NHWC>(
        input->data(),
        output->data(),
        max_index->data(),
        IN,
        C, H, W,
        K, P, Q,
        R, S,
        str_h, str_w);
      break;
    default:
      LOG(FATAL) << "Blitz not support pooling format: " << input->data_layout(); 
      break;
  }  
}

static void MaxPooling2DBackwardFunc(
  const CPUTensor<DType>* output,
  CPUTensor<DType>* input,
  const CPUTensor<size_t>* max_index) {
  // shape init
  size_t IN, C, H, W;
  size_t ON, K, P, Q;
  // shape decode
  CHECK_EQ(input->data_layout(), output->data_layout());
  Blitz2DBuffer(input->shape(), &IN, &C, &H, &W);
  Blitz2DBuffer(output->shape(), &ON, &K, &P, &Q);
  CHECK_EQ(IN, ON);
  CHECK_EQ(C, K);
  // set zero
  input->Fill(0);
  // no padding
  switch (input->data_layout()) {
    case BLITZ_BUFFER_NCHW:
      utils::MaxPoolingBackwardImpl<CPUTensor, DType, BLITZ_BUFFER_NCHW>(
        output->data(),
        input->data(),
        max_index->data(),
        IN,
        C, H, W,
        K, P, Q);
      break;
    case BLITZ_BUFFER_NHWC:
      utils::MaxPoolingBackwardImpl<CPUTensor, DType, BLITZ_BUFFER_NHWC>(
        output->data(),
        input->data(),
        max_index->data(),
        IN,
        C, H, W,
        K, P, Q);
      break;
    default:
      LOG(FATAL) << "Blitz not support pooling format: " << input->data_layout();
      break;
  }
}

#endif  // SRC_BACKENDS_CPU_BACKEND_POOL_INL_H_
