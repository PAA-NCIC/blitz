#ifndef SRC_BACKEND_GPU_BACKEND_POOL_INL_H_
#define SRC_BACKEND_GPU_BACKEND_POOL_INL_H_

static void MaxPooling2DForwardFunc(
  const GPUTensor<DType>* input,
  GPUTensor<DType>* output,
  GPUTensor<size_t>* max_index, 
  size_t R, size_t S,
  size_t str_h, size_t str_w) {
  // shape init
  size_t IN, C, H, W;
  size_t ON, K, P, Q;
  // shape decode
  CHECK_EQ(input->data_layout(), output->data_layout());
  Blitz2DBuffer(input->shape(), &IN, &C, &H, &W);
  Blitz2DBuffer(output->shape(), &ON, &K, &P, &Q);
  CHECK_EQ(IN, ON);
  CHECK_EQ(C, K);
  // set min
  output->Fill(std::numeric_limits<DType>::min());
  utils::MaxPoolingForwardImpl<GPUTensor, DType, BLITZ_BUFFER_NCHW>(
    input->data(),
    output->data(),
    max_index->data(),
    IN,
    C, H, W,
    K, P, Q,
    R, S,
    str_h, str_w);
}

static void MaxPooling2DBackwardFunc(
  const GPUTensor<DType>* output,
  GPUTensor<DType>* input,
  const GPUTensor<size_t>* max_index) {
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
  utils::MaxPoolingBackwardImpl<GPUTensor, DType, BLITZ_BUFFER_NCHW>(
    output->data(),
    input->data(),
    max_index->data(),
    IN,
    C, H, W,
    K, P, Q);
}

#endif  // SRC_BACKEND_GPU_BACKEND_POOL_INL_H_
