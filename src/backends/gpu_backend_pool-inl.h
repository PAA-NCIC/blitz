#ifndef SRC_BACKEND_GPU_BACKEND_POOL_INL_H_
#define SRC_BACKEND_GPU_BACKEND_POOL_INL_H_

template<typename DType>
__global__ void GPUMaxPoolingForward(const DType* input,
  size_t size, size_t channel,
  size_t input_height, size_t input_width,
  size_t output_height, size_t output_width,
  size_t filter_height, size_t filter_width,
  size_t stride_height, size_t stride_width,
  size_t* max_index, DType* output) {
  BLITZ_CUDA_LOOP(index, size) {
    size_t output_width_index = index % output_width;
    size_t output_height_index = (index / output_width) % output_height;
    size_t channel_index = (index / (output_width * output_height)) % channel;
    size_t batch_index = index / (output_width * output_height * channel);
    size_t height_start = output_height_index * stride_height;
    size_t width_start = output_width_index * stride_width;
    size_t height_end = height_start + filter_height;
    size_t width_end = width_start + filter_width;
    size_t max_idx = height_start * input_width + width_start;
    const DType* input_slice = input +
      (batch_index * channel + channel_index) *
      input_height * input_width;
    for (size_t i = height_start; i < height_end; ++i) {
      for (size_t j = width_start; j < width_end; ++j) {
        if (input_slice[i * input_width + j] > input_slice[max_idx]) {
          max_idx = i * input_width + j;
        }
      }
    }
    output[index] = input_slice[max_idx];
    max_index[index] = max_idx;
  }
}


template<typename DType>
void Backend<GPUTensor, DType>::MaxPooling2DForwardFunc(
  const GPUTensor<DType>* input,
  size_t filter_height, size_t filter_width,
  size_t stride_height, size_t stride_width,
  GPUTensor<size_t>* max_index, GPUTensor<DType>* output) {
  // shape decode
  // input
  const Shape& input_shape = input->shape();
  size_t channel = input_shape[1];
  size_t input_height = input_shape[2];
  size_t input_width = input_shape[3];
  // output
  const Shape& output_shape = output->shape();
  size_t output_channel = output_shape[1];
  size_t output_height = output_shape[2];
  size_t output_width = output_shape[3];

  CHECK_EQ(channel, output_channel);

  output->Fill(std::numeric_limits<DType>::min());

  GPUMaxPoolingForward<DType><<<BlitzGPUGetBlocks(output->size()),
    BLITZ_NUM_GPU_THREADS>>>(input->data(), output->size(),
    channel, input_height, input_width,
    output_height, output_width,
    filter_height, filter_width,
    stride_height, stride_width,
    max_index->data(), output->data());
}

template<typename DType>
__global__ void GPUMaxPoolingBackward(const DType* output,
  size_t size, size_t channel,
  size_t input_height, size_t input_width,
  size_t output_height, size_t output_width,
  size_t filter_height, size_t filter_width,
  size_t stride_height, size_t stride_width,
  const size_t* max_index, DType* input) {
  BLITZ_CUDA_LOOP(i, size) {
    size_t channel_index = (i / (output_width * output_height)) % channel;
    size_t batch_index = i / (output_width * output_height * channel);
    DType* input_slice = input +
      (batch_index * channel + channel_index) *
      input_height * input_width;
    input_slice[max_index[i]] = output[i];
  }
}

template<typename DType>
void Backend<GPUTensor, DType>::MaxPooling2DBackwardFunc(
  const GPUTensor<DType>* output, const GPUTensor<size_t>* max_index,
  size_t filter_height, size_t filter_width,
  size_t stride_height, size_t stride_width,
  GPUTensor<DType>* input) {
  // shape decode
  // input
  const Shape& input_shape = input->shape();
  size_t channel = input_shape[1];
  size_t input_height = input_shape[2];
  size_t input_width = input_shape[3];
  // output
  const Shape& output_shape = output->shape();
  size_t output_height = output_shape[2];
  size_t output_width = output_shape[3];

  // set zero
  input->Fill(0);

  GPUMaxPoolingBackward<DType><<<BlitzGPUGetBlocks(output->size()),
    BLITZ_NUM_GPU_THREADS>>>(output->data(), output->size(),
    channel, input_height, input_width,
    output_height, output_width,
    filter_height, filter_width,
    stride_height, stride_width,
    max_index->data(), input->data());
}


#endif  // SRC_BACKEND_GPU_BACKEND_POOL_INL_H_
