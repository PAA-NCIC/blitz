#ifndef SRC_BACKENDS_CPU_BACKEND_POOL_INL_H_
#define SRC_BACKENDS_CPU_BACKEND_POOL_INL_H_

template<typename DType>
void Backend<CPUTensor, DType>::MaxPooling2DForwardFunc(
  const CPUTensor<DType>* input,
  CPUTensor<DType>* output,
  CPUTensor<size_t>* max_index, 
  size_t filter_height,
  size_t filter_width,
  size_t stride_width,
  size_t stride_height) {
  // shape init
  size_t IN, C, H, W;
  size_t ON, K, P, Q;
  // shape decode
  CHECK_EQ(input->data_layout(), output->data_layout());
  Blitz2DBuffer(input->data_layout(), input->shape_ptr(), &IN, &C, &H, &W);
  Blitz2DBuffer(output->data_layout(), output->shape_ptr(), &ON, &K, &P, &Q);
  CHECK_EQ(IN, ON);
  CHECK_EQ(C, K);
  switch (input->data_layout()) {
    case BLITZ_BUFFER_NCHW:
      MaxPoolingForwardImpl<CPUTensor, DType, BLITZ_BUFFER_NCHW>(
        input->data(),
        output->data(),
        max_index->data(),
        IN,
        C, H, W,
        K, P, Q,
        filter_height, filter_width,
        stride_height, stride_width);
      break;
    case BLITZ_BUFFER_NHWC:
      MaxPoolingForwardImpl<CPUTensor, DType, BLITZ_BUFFER_NHWC>(
        input->data(),
        output->data(),
        max_index->data(),
        IN,
        C, H, W,
        K, P, Q,
        filter_height, filter_width,
        stride_height, stride_width);
      break;
    default:
      LOG(FATAL) << "Blitz not support pooling format: " << input->data_layout(); 
      break;
  }  
}

template<typename DType>
void Backend<CPUTensor, DType>::MaxPooling2DBackwardFunc(
  const CPUTensor<DType>* output,
  CPUTensor<DType>* input,
  const CPUTensor<size_t>* max_index,
  size_t filter_height,
  size_t filter_width,
  size_t stride_height,
  size_t stride_width) {
  // shape init
  size_t IN, C, H, W;
  size_t ON, K, P, Q;
  // shape decode
  CHECK_EQ(input->data_layout(), output->data_layout());
  Blitz2DBuffer(input->data_layout(), input->shape_ptr(), &IN, &C, &H, &W);
  Blitz2DBuffer(output->data_layout(), output->shape_ptr(), &ON, &K, &P, &Q);
  CHECK_EQ(IN, ON);
  CHECK_EQ(C, K);
  // set zero
  input->Fill(0);
  // no padding
  switch (input->data_layout()) {
    case BLITZ_BUFFER_NCHW:
      MaxPoolingBackwardImpl<CPUTensor, DType, BLITZ_BUFFER_NCHW>(
        output->data(),
        input->data(),
        max_index->data(),
        IN,
        C, H, W,
        K, P, Q);
      break;
    case BLITZ_BUFFER_NHWC:
      MaxPoolingBackwardImpl<CPUTensor, DType, BLITZ_BUFFER_NHWC>(
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
