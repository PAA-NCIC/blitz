#ifndef SRC_BACKEND_CPU_BACKEND_POOL_INL_H_
#define SRC_BACKEND_CPU_BACKEND_POOL_INL_H_

template<typename DType>
void Backend<CPUTensor, DType>::MaxPooling2DForwardFunc(
  const CPUTensor<DType>* input,
  size_t filter_height, size_t filter_width,
  size_t stride_width, size_t stride_height,
  CPUTensor<size_t>* max_index, CPUTensor<DType>* output) {
  // shape decode
  // input
  const Shape& input_shape = input->shape();
  size_t batch_size = input_shape[0];
  size_t input_channel = input_shape[1];
  size_t input_height = input_shape[2];
  size_t input_width = input_shape[3];
  // output
  const Shape& output_shape = output->shape();
  size_t output_channel = output_shape[1];
  size_t output_height = output_shape[2];
  size_t output_width = output_shape[3];
  // offset
  size_t input_single_size = input_channel * input_height * input_width;
  size_t input_channel_size = input_height * input_width;
  size_t output_single_size = output_channel * output_height * output_width;
  size_t output_channel_size = output_height * output_width;
  size_t input_channel_offset;
  size_t output_channel_offset;
  size_t input_batch_offset;
  size_t output_batch_offset;
  CHECK_EQ(input_channel, output_channel);
  // no padding
  #pragma omp parallel for private(input_batch_offset, output_batch_offset, \
    input_channel_offset, output_channel_offset)
  for (size_t batch_index = 0; batch_index < batch_size; ++batch_index) {
    input_batch_offset = batch_index * input_single_size;
    output_batch_offset = batch_index * output_single_size;
    for (size_t channel_index = 0; channel_index < input_channel; ++channel_index) {
      input_channel_offset = channel_index * input_channel_size;
      output_channel_offset = channel_index * output_channel_size;
      const DType* input_slice = input->Slice(input_batch_offset +
        input_channel_offset);
      DType* output_slice = output->Slice(output_batch_offset +
        output_channel_offset);
      size_t* max_index_slice = max_index->Slice(output_batch_offset +
        output_channel_offset);
      for (size_t output_height_index = 0; output_height_index < output_height;
        ++output_height_index) {
        for (size_t output_width_index = 0; output_width_index < output_width;
          ++output_width_index) {
          size_t height_start = output_height_index * stride_height;
          size_t width_start = output_width_index * stride_width;
          size_t height_end = height_start + filter_height;
          size_t width_end = width_start + filter_width;
          size_t pool_index = output_height_index * output_width +
            output_width_index;
          size_t max_index_tmp = height_start * input_width + width_start;
          for (size_t h = height_start; h < height_end; ++h) {
            for (size_t w = width_start; w < width_end; ++w) {
              size_t index = h * input_width + w;
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
  const CPUTensor<DType>* output, const CPUTensor<size_t>* max_index,
  size_t filter_height, size_t filter_width,
  size_t stride_height, size_t stride_width,
  CPUTensor<DType>* input) {
  // shape decode
  // input
  const Shape& input_shape = input->shape();
  size_t batch_size = input_shape[0];
  size_t channel = input_shape[1];
  size_t input_height = input_shape[2];
  size_t input_width = input_shape[3];
  // output
  const Shape& output_shape = output->shape();
  size_t output_height = output_shape[2];
  size_t output_width = output_shape[3];
  // offset
  size_t input_single_size = channel * input_height * input_width;
  size_t input_channel_size = input_height * input_width;
  size_t output_single_size = channel * output_height * output_width;
  size_t output_channel_size = output_height * output_width;
  size_t input_batch_offset;
  size_t output_batch_offset;
  size_t input_channel_offset;
  size_t output_channel_offset;
  DType (&p_input)[batch_size][channel][input_height * input_width] = 
    *reinterpret_cast<DType (*) [batch_size][channel]
    [input_height * input_width]>(input->data());
  const DType (&p_output)[batch_size][channel][output_height * output_width] =
    *reinterpret_cast<const DType (*) [batch_size][channel]
    [output_height * output_width]>(output->data());
  const size_t (&p_max_index)[batch_size][channel][output_height * output_width] =
    *reinterpret_cast<const size_t (*) [batch_size][channel]
    [output_height * output_width]>(max_index->data());
  // set zero
  input->Fill(0);
  // no padding
  #pragma omp parallel for private(input_batch_offset, output_batch_offset, \
    input_channel_offset, output_channel_offset)
  for (size_t batch_index = 0; batch_index < batch_size; ++batch_index) {
    input_batch_offset = batch_index * input_single_size;
    output_batch_offset = batch_index * output_single_size;
    for (size_t channel_index = 0; channel_index < channel; ++channel_index) {
      input_channel_offset = channel_index * input_channel_size;
      output_channel_offset = channel_index * output_channel_size;
      DType* input_slice = input->Slice(input_batch_offset +
        input_channel_offset);
      const DType* output_slice = output->Slice(output_batch_offset +
        output_channel_offset);
      const size_t* max_index_slice = max_index->Slice(output_batch_offset +
        output_channel_offset);
      for (size_t output_height_index = 0; output_height_index < output_height;
        ++output_height_index) {
        for (size_t output_width_index = 0; output_width_index < output_width;
          ++output_width_index) {
          size_t index = output_height_index * output_width +
            output_width_index;
          input_slice[max_index_slice[index]] = output_slice[index];
        }
      }
    }
  }
  //#pragma omp parallel for
  //for (size_t batch_index = 0; batch_index < batch_size; ++batch_index) {
  //  for (size_t channel_index = 0; channel_index < channel; ++channel_index) {
  //    for (size_t output_height_index = 0; output_height_index < output_height;
  //      ++output_height_index) {
  //      for (size_t output_width_index = 0; output_width_index < output_width;
  //        ++output_width_index) {
  //        size_t index = p_max_index[batch_index][channel_index]
  //          [output_height_index * output_width + output_width_index];
  //        p_input[batch_index][channel_index][index] =
  //          p_output[batch_index][channel_index]
  //          [output_height_index * output_width + output_width_index];
  //      }
  //    }
  //  }
  //}
}

#endif  // SRC_BACKEND_CPU_BACKEND_POOL_INL_H_
