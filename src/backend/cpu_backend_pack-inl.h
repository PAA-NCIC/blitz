#ifndef SRC_BACKEND_CPU_BACKEND_PACK_INL_H_
#define SRC_BACKEND_CPU_BACKEND_PACK_INL_H_

template<typename DType>
void Backend<CPUTensor, DType>::Unpack2DParallelFunc(
  const DType* input, const int channel,
  const int input_height, const int input_width,
  const int filter_height, const int filter_width,
  const int output_height, const int output_width,
  const int padding_height, const int padding_width,
  const int stride_height, const int stride_width,
  DType* unpack) {
  // (input_channel * filter_height * filter_width) *
  // (output_width * output_height)
  int unpack_index;
  int channel_offset;
  const int filter_channel_offset = filter_height * filter_width;
  const int output_channel_offset = output_height * output_width;
  const int input_channel_offset = input_height * input_width;
  #pragma omp parallel for collapse(2) private(channel_offset, unpack_index)
  for (int channel_index = 0; channel_index < channel; ++channel_index) {
    for (int filter_height_index = 0; filter_height_index < filter_height;
        ++filter_height_index) {
      channel_offset = channel_index * input_channel_offset;
      unpack_index = (channel_index * filter_channel_offset +
        filter_height_index * filter_width) * output_channel_offset;
      const DType* input_slice = input + channel_offset;
      for (int filter_width_index = 0; filter_width_index < filter_width;
          ++filter_width_index) {
        int filter_height_offset = -padding_height + filter_height_index;
        for (int output_height_index = 0; output_height_index < output_height;
            ++output_height_index) {
          if (filter_height_offset < 0 || filter_height_offset >= input_height) {
            for (int output_width_index = 0; output_width_index < output_width;
                ++output_width_index) {
              unpack[unpack_index++] = 0;
            }
          } else {
            int filter_width_offset = -padding_width + filter_width_index;
            for (int output_width_index = 0; output_width_index < output_width;
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
  const DType* pack, const int channel,
  const int input_height, const int input_width,
  const int filter_height, const int filter_width,
  const int output_height, const int output_width,
  const int padding_height, const int padding_width,
  const int stride_height, const int stride_width,
  DType* input) {
  // memset
  // (input_channel * filter_height * filter_width) *
  // (output_width * output_height)
  int pack_index;
  int channel_offset;
  const int filter_channel_offset = filter_height * filter_width;
  const int output_channel_offset = output_height * output_width;
  const int input_channel_offset = input_height * input_width;
  #pragma omp parallel for collapse(2) private(channel_offset, pack_index)
  for (int channel_index = 0; channel_index < channel; ++channel_index) {
    for (int filter_height_index = 0; filter_height_index < filter_height;
        ++filter_height_index) {
      channel_offset = channel_index * input_channel_offset;
      pack_index = (channel_index * filter_channel_offset +
        filter_height_index * filter_width) * output_channel_offset;
      DType* input_slice = input + channel_offset;
      for (int filter_width_index = 0; filter_width_index < filter_width;
          ++filter_width_index) {
        int filter_height_offset = -padding_height + filter_height_index;
        for (int output_height_index = 0; output_height_index < output_height;
            ++output_height_index) {
          if (filter_height_offset < 0 || filter_height_offset >= input_height) {
            pack_index += output_width;
          } else {
            int filter_width_offset = -padding_width + filter_width_index;
            for (int output_width_index = 0; output_width_index < output_width;
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
  const DType* input, const int channel,
  const int input_height, const int input_width,
  const int filter_height, const int filter_width,
  const int output_height, const int output_width,
  const int padding_height, const int padding_width,
  const int stride_height, const int stride_width,
  DType* unpack) {
  // (input_channel * filter_height * filter_width) *
  // (output_width * output_height)
  int unpack_index = 0;
  int channel_offset;
  const int input_channel_offset = input_height * input_width;
  for (int channel_index = 0; channel_index < channel; ++channel_index) {
    channel_offset = channel_index * input_channel_offset;
    const DType* input_slice = input + channel_offset;
    for (int filter_height_index = 0; filter_height_index < filter_height;
        ++filter_height_index) {
      for (int filter_width_index = 0; filter_width_index < filter_width;
          ++filter_width_index) {
        int filter_height_offset = -padding_height + filter_height_index;
        for (int output_height_index = 0; output_height_index < output_height;
            ++output_height_index) {
          if (filter_height_offset < 0 || filter_height_offset >= input_height) {
            for (int output_width_index = 0; output_width_index < output_width;
                ++output_width_index) {
              unpack[unpack_index++] = 0;
            }
          } else {
            int filter_width_offset = -padding_width + filter_width_index;
            for (int output_width_index = 0; output_width_index < output_width;
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
  const DType* pack, const int channel,
  const int input_height, const int input_width,
  const int filter_height, const int filter_width,
  const int output_height, const int output_width,
  const int padding_height, const int padding_width,
  const int stride_height, const int stride_width,
  DType* input) {
  int pack_index = 0;
  int channel_offset;
  const int input_channel_offset = input_height * input_width;
  for (int channel_index = 0; channel_index < channel; ++channel_index) {
    channel_offset = channel_index * input_channel_offset;
    DType* input_slice = input + channel_offset;
    for (int filter_height_index = 0; filter_height_index < filter_height;
        ++filter_height_index) {
      for (int filter_width_index = 0; filter_width_index < filter_width;
          ++filter_width_index) {
        int filter_height_offset = -padding_height + filter_height_index;
        for (int output_height_index = 0; output_height_index < output_height;
            ++output_height_index) {
          if (filter_height_offset < 0 || filter_height_offset >= input_height) {
            pack_index += output_width;
          } else {
            int filter_width_offset = -padding_width + filter_width_index;
            for (int output_width_index = 0; output_width_index < output_width;
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
