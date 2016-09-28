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
  const size_t batch_size = input_shape[0];
  const size_t channel = input_shape[1];
  const size_t input_height = input_shape[2];
  const size_t input_width = input_shape[3];
  // output
  const Shape& output_shape = output->shape();
  const size_t output_height = output_shape[2];
  const size_t output_width = output_shape[3];
  // multi-dimensional cast
  const DType (&p_input)[batch_size][channel][input_height * input_width] = 
    *reinterpret_cast<const DType (*)[batch_size][channel]
    [input_height * input_width]>(input->data());
  DType (&p_output)[batch_size][channel][output_height * output_width] =
    *reinterpret_cast<DType (*)[batch_size][channel]
    [output_height * output_width]>(output->data());
  size_t (&p_max_index)[batch_size][channel][output_height * output_width] =
    *reinterpret_cast<size_t (*)[batch_size][channel]
    [output_height * output_width]>(max_index->data());
  // no padding
  #pragma omp parallel for
  for (size_t batch_index = 0; batch_index < batch_size; ++batch_index) {
    for (size_t channel_index = 0; channel_index < channel; ++channel_index) {
      for (size_t output_height_index = 0; output_height_index < output_height;
        ++output_height_index) {
        for (size_t output_width_index = 0; output_width_index < output_width;
          ++output_width_index) {
          size_t height_start = output_height_index * stride_height;
          size_t width_start = output_width_index * stride_width;
          size_t height_end = height_start + filter_height;
          size_t width_end = width_start + filter_width;
          size_t max_index_tmp = height_start * input_width + width_start;
          for (size_t h = height_start; h < height_end; ++h) {
            for (size_t w = width_start; w < width_end; ++w) {
              const size_t index = h * input_width + w;
              if (p_input[batch_index][channel_index][index] >
                  p_input[batch_index][channel_index][max_index_tmp]) {
                max_index_tmp = index;
              }
            }
          }
          const size_t pool_index = output_height_index * output_width +
            output_width_index;
          p_output[batch_index][channel_index][pool_index] =
            p_input[batch_index][channel_index][max_index_tmp];
          p_max_index[batch_index][channel_index][pool_index] = max_index_tmp;
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
  const size_t batch_size = input_shape[0];
  const size_t channel = input_shape[1];
  const size_t input_height = input_shape[2];
  const size_t input_width = input_shape[3];
  // output
  const Shape& output_shape = output->shape();
  const size_t output_height = output_shape[2];
  const size_t output_width = output_shape[3];
  // multi-dimensional cast
  DType (&p_input)[batch_size][channel][input_height * input_width] = 
    *reinterpret_cast<DType (*)[batch_size][channel]
    [input_height * input_width]>(input->data());
  const DType (&p_output)[batch_size][channel][output_height * output_width] =
    *reinterpret_cast<const DType (*)[batch_size][channel]
    [output_height * output_width]>(output->data());
  const size_t (&p_max_index)[batch_size][channel][output_height * output_width] =
    *reinterpret_cast<const size_t (*)[batch_size][channel]
    [output_height * output_width]>(max_index->data());
  // set zero
  input->Fill(0);
  // no padding
  #pragma omp parallel for
  for (size_t batch_index = 0; batch_index < batch_size; ++batch_index) {
    for (size_t channel_index = 0; channel_index < channel; ++channel_index) {
      for (size_t output_height_index = 0; output_height_index < output_height;
        ++output_height_index) {
        for (size_t output_width_index = 0; output_width_index < output_width;
          ++output_width_index) {
          const size_t index = p_max_index[batch_index][channel_index]
            [output_height_index * output_width + output_width_index];
          p_input[batch_index][channel_index][index] =
            p_output[batch_index][channel_index]
            [output_height_index * output_width + output_width_index];
        }
      }
    }
  }
}

#endif  // SRC_BACKEND_CPU_BACKEND_POOL_INL_H_
