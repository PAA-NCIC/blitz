#ifndef SRC_BACKEND_CPU_BACKEND_POOL_INL_H_
#define SRC_BACKEND_CPU_BACKEND_POOL_INL_H_

template<typename DType>
void Backend<CPUTensor, DType>::MaxPooling2DForwardFunc(
  const CPUTensor<DType>* input,
  const int filter_height, const int filter_width,
  const int stride_width, const int stride_height,
  CPUTensor<int>* max_index, CPUTensor<DType>* output) {
  // shape decode
  // input
  const Shape& input_shape = input->shape();
  int batch_size = input_shape[0];
  int input_channel = input_shape[1];
  int input_height = input_shape[2];
  int input_width = input_shape[3];
  // output
  const Shape& output_shape = output->shape();
  int output_channel = output_shape[1];
  int output_height = output_shape[2];
  int output_width = output_shape[3];

  CHECK_EQ(input_channel, output_channel);

  const int input_single_offset = input_channel * input_height * input_width;
  const int input_channel_offset = input_height * input_width;
  const int output_single_offset = output_channel * output_height * output_width;
  const int output_channel_offset = output_height * output_width;
  int channel_input_offset;
  int channel_output_offset;
  int batch_input_offset;
  int batch_output_offset;

  // no padding
  #pragma omp parallel for private(batch_input_offset, batch_output_offset, \
    channel_input_offset, channel_output_offset)
  for (int batch_index = 0; batch_index < batch_size; ++batch_index) {
    batch_input_offset = batch_index * input_single_offset;
    batch_output_offset = batch_index * output_single_offset;
    for (int channel_index = 0; channel_index < input_channel; ++channel_index) {
      channel_input_offset = channel_index * input_channel_offset;
      channel_output_offset = channel_index * output_channel_offset;
      const DType* input_slice = input->Slice(batch_input_offset +
        channel_input_offset);
      DType* output_slice = output->Slice(batch_output_offset +
        channel_output_offset);
      int* max_index_slice = max_index->Slice(batch_output_offset +
        channel_output_offset);
      for (int output_height_index = 0; output_height_index < output_height;
        ++output_height_index) {
        for (int output_width_index = 0; output_width_index < output_width;
          ++output_width_index) {
          const int height_start = output_height_index * stride_height;
          const int width_start = output_width_index * stride_width;
          const int height_end = height_start + filter_height;
          const int width_end = width_start + filter_width;
          const int pool_index = output_height_index * output_width +
            output_width_index;
          int max_index_tmp = height_start * input_width + width_start;
          for (int h = height_start; h < height_end; ++h) {
            for (int w = width_start; w < width_end; ++w) {
              const int index = h * input_width + w;
              if (input_slice[index] > input_slice[max_index_tmp]) {
                max_index_tmp = index;
              }
            }
          }
          output_slice[pool_index] = input_slice[max_index_tmp];
          max_index_slice[pool_index] = max_index_tmp;
        }
      }
    }
  }
}

template<typename DType>
void Backend<CPUTensor, DType>::MaxPooling2DBackwardFunc(
  const CPUTensor<DType>* output, const CPUTensor<int>* max_index,
  const int filter_height, const int filter_width,
  const int stride_height, const int stride_width,
  CPUTensor<DType>* input) {
  // shape decode
  // input
  const Shape& input_shape = input->shape();
  int batch_size = input_shape[0];
  int channel = input_shape[1];
  int input_height = input_shape[2];
  int input_width = input_shape[3];
  // output
  const Shape& output_shape = output->shape();
  int output_height = output_shape[2];
  int output_width = output_shape[3];

  const int input_single_offset = channel * input_height * input_width;
  const int input_channel_offset = input_height * input_width;
  const int output_single_offset = channel * output_height * output_width;
  const int output_channel_offset = output_height * output_width;
  int batch_input_offset;
  int batch_output_offset;
  int channel_input_offset;
  int channel_output_offset;
  // set zero
  input->Fill(0);
  // no padding
  #pragma omp parallel for private(batch_input_offset, batch_output_offset, \
    channel_input_offset, channel_output_offset)
  for (int batch_index = 0; batch_index < batch_size; ++batch_index) {
    batch_input_offset = batch_index * input_single_offset;
    batch_output_offset = batch_index * output_single_offset;
    for (int channel_index = 0; channel_index < channel; ++channel_index) {
      channel_input_offset = channel_index * input_channel_offset;
      channel_output_offset = channel_index * output_channel_offset;
      DType* input_slice = input->Slice(batch_input_offset +
        channel_input_offset);
      const DType* output_slice = output->Slice(batch_output_offset +
        channel_output_offset);
      const int* max_index_slice = max_index->Slice(batch_output_offset +
        channel_output_offset);
      for (int output_height_index = 0; output_height_index < output_height;
        ++output_height_index) {
        for (int output_width_index = 0; output_width_index < output_width;
          ++output_width_index) {
          const int index = output_height_index * output_width +
            output_width_index;
          input_slice[max_index_slice[index]] = output_slice[index];
        }
      }
    }
  }
}

#endif  // SRC_BACKEND_CPU_BACKEND_POOL_INL_H_
