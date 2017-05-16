#ifndef SRC_BACKENDS_GPU_BACKEND_TRANSFORM_INL_H_
#define SRC_BACKENDS_GPU_BACKEND_TRANSFORM_INL_H_

static void Unpack2DFunc(
  const GPUTensor<DType>* input,
  GPUTensor<DType>* unpack,
  size_t R, size_t S,
  size_t pad_h, size_t pad_w,
  size_t str_h, size_t str_w) {
  size_t N, C, H, W;
  size_t P, Q;
  Blitz2DBuffer(input->shape(), &N, &C, &H, &W);
  P = (H + 2 * pad_h - R) / str_h + 1;
  Q = (W + 2 * pad_w - S) / str_w + 1;
  CHECK_EQ(unpack->size(), N * C * R * S * P * Q);
  for (size_t i = 0; i < N; ++i) {
    utils::Unpack2DDispatch<GPUTensor, DType>(
      input->data(), unpack->data(),
      C, H, W,
      R, S,
      P, Q,
      pad_h, pad_w,
      str_h, str_w,
      input->data_layout());
  }
}

static void Pack2DFunc(
  const GPUTensor<DType>* unpack,
  GPUTensor<DType>* input,
  size_t R, size_t S,
  size_t pad_h, size_t pad_w,
  size_t str_h, size_t str_w) {
  size_t N, C, H, W;
  size_t P, Q;
  Blitz2DBuffer(input->shape(), &N, &C, &H, &W);
  P = (H + 2 * pad_h - R) / str_h + 1;
  Q = (W + 2 * pad_w - S) / str_w + 1;
  CHECK_EQ(unpack->size(), N * C * R * S * P * Q);
  for (size_t i = 0; i < N; ++i) {
    utils::Pack2DDispatch<GPUTensor, DType>(
      unpack->data(), input->data(),
      C, H, W,
      R, S,
      P, Q,
      pad_h, pad_w,
      str_h, str_w,
      input->data_layout());
  }
}

static void HostCopyToTensorFunc(
  const DType *host,
  GPUTensor<DType> *tensor) {
  cudaMemcpy(tensor->data(), host, tensor->size() * sizeof(DType), cudaMemcpyHostToDevice);
}

#endif  // SRC_BACKENDS_GPU_BACKEND_TRANSFORM_INL_H_
