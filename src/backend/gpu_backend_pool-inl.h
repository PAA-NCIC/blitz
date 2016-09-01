#ifndef SRC_BACKEND_GPU_BACKEND_POOL_INL_H_
#define SRC_BACKEND_GPU_BACKEND_POOL_INL_H_

#include <cstdio>

template<typename DType>
__global__ void GPUMaxPoolingForward(const DType* input,
  const int size, const int channel,
  const int input_height, const int input_width,
  const int output_height, const int output_width,
  const int filter_height, const int filter_width,
  const int stride_height, const int stride_width,
  int* max_index, DType* output) {
  BLITZ_CUDA_LOOP(index, size) {
    const int output_width_index = index % output_width;
    const int output_height_index = (index / output_width) % output_height;
    const int channel_index = (index / (output_width * output_height)) % channel;
    const int batch_index = index / (output_width * output_height * channel);
    const int height_start = output_height_index * stride_height;
    const int width_start = output_width_index * stride_width;
    const int height_end = height_start + filter_height;
    const int width_end = width_start + filter_width;
    int max_idx = height_start * input_width + width_start;
    const DType* input_slice = input +
      (batch_index * channel + channel_index) *
      input_height * input_width;
    for (int i = height_start; i < height_end; ++i) {
      for (int j = width_start; j < width_end; ++j) {
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
  const int filter_height, const int filter_width,
  const int stride_height, const int stride_width,
  GPUTensor<int>* max_index, GPUTensor<DType>* output) {
  // shape decode
  // input
  const Shape& input_shape = input->shape();
  const int channel = input_shape[1];
  const int input_height = input_shape[2];
  const int input_width = input_shape[3];
  // output
  const Shape& output_shape = output->shape();
  const int output_channel = output_shape[1];
  const int output_height = output_shape[2];
  const int output_width = output_shape[3];

  CHECK_EQ(channel, output_channel);

  output->Fill(std::numeric_limits<DType>::min());

  GPUMaxPoolingForward<DType><<<BlitzGPUGetBlocks(output->size()),
    BLITZ_NUM_GPU_THREADS>>>(input->data(), output->size(),
    channel, input_height, input_width,
    output_height, output_width,
    filter_height, filter_width,
    stride_height, stride_width, max_index->data(), output->data());
}

template<typename DType>
__global__ void GPUMaxPoolingBackward(const DType* output,
  const int size, const int channel,
  const int input_height, const int input_width,
  const int output_height, const int output_width,
  const int filter_height, const int filter_width,
  const int stride_height, const int stride_width,
  const int* max_index, DType* input) {
  BLITZ_CUDA_LOOP(i, size) {
    const int channel_index = (i / (output_width * output_height)) % channel;
    const int batch_index = i / (output_width * output_height * channel);
     DType* input_slice = input +
      (batch_index * channel + channel_index) *
      input_height * input_width;
    input_slice[max_index[i]] = output[i];
  }
}

template<typename DType>
void Backend<GPUTensor, DType>::MaxPooling2DBackwardFunc(
  const GPUTensor<DType>* output, const GPUTensor<int>* max_index,
  const int filter_height, const int filter_width,
  const int stride_height, const int stride_width,
  GPUTensor<DType>* input) {
  // shape decode
  // input
  const Shape& input_shape = input->shape();
  const int channel = input_shape[1];
  const int input_height = input_shape[2];
  const int input_width = input_shape[3];
  // output
  const Shape& output_shape = output->shape();
  const int output_height = output_shape[2];
  const int output_width = output_shape[3];

  // set zero
  input->Fill(0);

  GPUMaxPoolingBackward<DType><<<BlitzGPUGetBlocks(output->size()),
    BLITZ_NUM_GPU_THREADS>>>(output->data(), output->size(),
    channel, input_height, input_width,
    output_height, output_width,
    filter_height, filter_width,
    stride_height, stride_width, max_index->data(), input->data());
}


#endif  // SRC_BACKEND_GPU_BACKEND_POOL_INL_H_
