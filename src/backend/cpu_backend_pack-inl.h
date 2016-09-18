#ifndef SRC_BACKEND_CPU_BACKEND_PACK_INL_H_
#define SRC_BACKEND_CPU_BACKEND_PACK_INL_H_

template<typename DType>
void Backend<CPUTensor, DType>::Unpack2DParallelFunc(
  const DType* input, size_t channel,
  size_t input_height, size_t input_width,
  size_t filter_height, size_t filter_width,
  size_t output_height, size_t output_width,
  size_t padding_height, size_t padding_width,
  size_t stride_height, size_t stride_width,
  DType* unpack) {
  // (input_channel * filter_height * filter_width) *
  // (output_width * output_height)
  size_t unpack_index;
  size_t channel_offset;
  size_t filter_channel_offset = filter_height * filter_width;
  size_t output_channel_offset = output_height * output_width;
  size_t input_channel_offset = input_height * input_width;
  #pragma omp parallel for collapse(2) private(channel_offset, unpack_index)
  for (size_t channel_index = 0; channel_index < channel; ++channel_index) {
    for (size_t filter_height_index = 0; filter_height_index < filter_height;
        ++filter_height_index) {
      channel_offset = channel_index * input_channel_offset;
      unpack_index = (channel_index * filter_channel_offset +
        filter_height_index * filter_width) * output_channel_offset;
      const DType* input_slice = input + channel_offset;
      for (size_t filter_width_index = 0; filter_width_index < filter_width;
          ++filter_width_index) {
        size_t filter_height_offset = -padding_height + filter_height_index;
        for (size_t output_height_index = 0; output_height_index < output_height;
            ++output_height_index) {
          if (filter_height_offset < 0 || filter_height_offset >= input_height) {
            for (int output_width_index = 0; output_width_index < output_width;
                ++output_width_index) {
              unpack[unpack_index++] = 0;
            }
          } else {
            size_t filter_width_offset = -padding_width + filter_width_index;
            for (size_t output_width_index = 0; output_width_index < output_width;
                ++output_width_index) {
              if (filter_width_offset < 0 || filter_width_offset >= input_width) {
                unpack[unpack_index++] = 0;
              } else {
                unpack[unpack_index++] = input_slice[
                  filter_height_offset * input_width + filter_width_offset];
              }
              filter_width_offset += stride_width;
            }
          }
          filter_height_offset += stride_height;
        }
      }
    }
  }
}

template<typename DType>
void Backend<CPUTensor, DType>::Pack2DParallelFunc(
  const DType* pack, size_t channel,
  size_t input_height, size_t input_width,
  size_t filter_height, size_t filter_width,
  size_t output_height, size_t output_width,
  size_t padding_height, size_t padding_width,
  size_t stride_height, size_t stride_width,
  DType* input) {
  // memset
  // (input_channel * filter_height * filter_width) *
  // (output_width * output_height)
  int pack_index;
  int channel_offset;
  size_t filter_channel_offset = filter_height * filter_width;
  size_t output_channel_offset = output_height * output_width;
  size_t input_channel_offset = input_height * input_width;
  #pragma omp parallel for collapse(2) private(channel_offset, pack_index)
  for (size_t channel_index = 0; channel_index < channel; ++channel_index) {
    for (size_t filter_height_index = 0; filter_height_index < filter_height;
        ++filter_height_index) {
      channel_offset = channel_index * input_channel_offset;
      pack_index = (channel_index * filter_channel_offset +
        filter_height_index * filter_width) * output_channel_offset;
      DType* input_slice = input + channel_offset;
      for (size_t filter_width_index = 0; filter_width_index < filter_width;
          ++filter_width_index) {
        size_t filter_height_offset = -padding_height + filter_height_index;
        for (size_t output_height_index = 0; output_height_index < output_height;
            ++output_height_index) {
          if (filter_height_offset < 0 || filter_height_offset >= input_height) {
            pack_index += output_width;
          } else {
            size_t filter_width_offset = -padding_width + filter_width_index;
            for (size_t output_width_index = 0; output_width_index < output_width;
                ++output_width_index) {
              if (filter_width_offset >= 0 && filter_width_offset < input_width) {
                #pragma omp atomic
                input_slice[filter_height_offset * input_width +
                  filter_width_offset] += pack[pack_index];
              }
              pack_index++;
              filter_width_offset += stride_width;
            }
          }
          filter_height_offset += stride_height;
        }
      }
    }
  }
}

template<typename DType>
void Backend<CPUTensor, DType>::Unpack2DFunc(
  const DType* input, size_t channel,
  size_t input_height, size_t input_width,
  size_t filter_height, size_t filter_width,
  size_t output_height, size_t output_width,
  size_t padding_height, size_t padding_width,
  size_t stride_height, size_t stride_width,
  DType* unpack) {
  // (input_channel * filter_height * filter_width) *
  // (output_width * output_height)
  size_t unpack_index = 0;
  size_t channel_offset;
  size_t input_channel_offset = input_height * input_width;
  for (size_t channel_index = 0; channel_index < channel; ++channel_index) {
    channel_offset = channel_index * input_channel_offset;
    const DType* input_slice = input + channel_offset;
    for (size_t filter_height_index = 0; filter_height_index < filter_height;
        ++filter_height_index) {
      for (size_t filter_width_index = 0; filter_width_index < filter_width;
          ++filter_width_index) {
        size_t filter_height_offset = -padding_height + filter_height_index;
        for (size_t output_height_index = 0; output_height_index < output_height;
            ++output_height_index) {
          if (filter_height_offset < 0 || filter_height_offset >= input_height) {
            for (size_t output_width_index = 0; output_width_index < output_width;
                ++output_width_index) {
              unpack[unpack_index++] = 0;
            }
          } else {
            size_t filter_width_offset = -padding_width + filter_width_index;
            for (size_t output_width_index = 0; output_width_index < output_width;
                ++output_width_index) {
              if (filter_width_offset < 0 || filter_width_offset >= input_width) {
                unpack[unpack_index++] = 0;
              } else {
                unpack[unpack_index++] = input_slice[filter_height_offset *
                  input_width + filter_width_offset];
              }
              filter_width_offset += stride_width;
            }
          }
          filter_height_offset += stride_height;
        }
      }
    }
  }
}

template<typename DType>
void Backend<CPUTensor, DType>::Pack2DFunc(
  const DType* pack, size_t channel,
  size_t input_height, size_t input_width,
  size_t filter_height, size_t filter_width,
  size_t output_height, size_t output_width,
  size_t padding_height, size_t padding_width,
  size_t stride_height, size_t stride_width,
  DType* input) {
  size_t pack_index = 0;
  size_t channel_offset;
  size_t input_channel_offset = input_height * input_width;
  for (size_t channel_index = 0; channel_index < channel; ++channel_index) {
    channel_offset = channel_index * input_channel_offset;
    DType* input_slice = input + channel_offset;
    for (size_t filter_height_index = 0; filter_height_index < filter_height;
        ++filter_height_index) {
      for (size_t filter_width_index = 0; filter_width_index < filter_width;
          ++filter_width_index) {
        int filter_height_offset = -padding_height + filter_height_index;
        for (size_t output_height_index = 0; output_height_index < output_height;
            ++output_height_index) {
          if (filter_height_offset < 0 || filter_height_offset >= input_height) {
            pack_index += output_width;
          } else {
            size_t filter_width_offset = -padding_width + filter_width_index;
            for (size_t output_width_index = 0; output_width_index < output_width;
                ++output_width_index) {
              if (filter_width_offset >= 0 && filter_width_offset < input_width) {
                input_slice[filter_height_offset * input_width +
                  filter_width_offset] += pack[pack_index];
              }
              pack_index++;
              filter_width_offset += stride_width;
            }
          }
          filter_height_offset += stride_height;
        }
      }
    }
  }
}

#endif  // SRC_BACKEND_CPU_BACKEND_PACK_INL_H_
