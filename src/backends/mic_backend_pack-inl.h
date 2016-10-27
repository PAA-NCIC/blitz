#ifndef SRC_BACKENDS_MIC_BACKEND_PACK_INL_H_
#define SRC_BACKENDS_MIC_BACKEND_PACK_INL_H_

template<typename DType>
void Backend<MICTensor, DType>::Unpack2DFunc(
  const DType* input,
  DType* unpack,
  size_t channel,
  size_t input_height,
  size_t input_width,
  size_t filter_height,
  size_t filter_width,
  size_t output_height,
  size_t output_width,
  size_t padding_height,
  size_t padding_width,
  size_t stride_height,
  size_t stride_width) {
  // (input_channel * filter_height * filter_width) *
  // (output_width * output_height)
  size_t unpack_index = 0;
  for (size_t channel_index = 0; channel_index < channel; ++channel_index) {
    const size_t channel_offset = channel_index * input_height * input_width;
    const DType* input_slice = input + channel_offset;
    for (size_t filter_height_index = 0; filter_height_index < filter_height;
        ++filter_height_index) {
      for (size_t filter_width_index = 0; filter_width_index < filter_width;
          ++filter_width_index) {
        int filter_height_offset = -padding_height + filter_height_index;
        for (size_t output_height_index = 0; output_height_index < output_height;
            ++output_height_index) {
          if (filter_height_offset < 0 || filter_height_offset >=
            static_cast<int>(input_height)) {
            for (size_t output_width_index = 0; output_width_index < output_width;
                ++output_width_index) {
              unpack[unpack_index++] = 0;
            }
          } else {
            int filter_width_offset = -padding_width + filter_width_index;
            for (size_t output_width_index = 0; output_width_index < output_width;
                ++output_width_index) {
              if (filter_width_offset < 0 || filter_width_offset >=
                static_cast<int>(input_width)) {
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
void Backend<MICTensor, DType>::Pack2DFunc(
  const DType* pack,
  DType* input,
  size_t channel,
  size_t input_height,
  size_t input_width,
  size_t filter_height,
  size_t filter_width,
  size_t output_height,
  size_t output_width,
  size_t padding_height,
  size_t padding_width,
  size_t stride_height,
  size_t stride_width) {
  // (input_channel * filter_height * filter_width) *
  // (output_width * output_height)
  size_t pack_index = 0;
  for (size_t channel_index = 0; channel_index < channel; ++channel_index) {
    const size_t channel_offset = channel_index * input_height * input_width;
    DType* input_slice = input + channel_offset;
    for (size_t filter_height_index = 0; filter_height_index < filter_height;
        ++filter_height_index) {
      for (size_t filter_width_index = 0; filter_width_index < filter_width;
          ++filter_width_index) {
        int filter_height_offset = -padding_height + filter_height_index;
        for (size_t output_height_index = 0; output_height_index < output_height;
            ++output_height_index) {
          if (filter_height_offset < 0 || filter_height_offset >=
            static_cast<int>(input_height)) {
            pack_index += output_width;
          } else {
            int filter_width_offset = -padding_width + filter_width_index;
            for (size_t output_width_index = 0; output_width_index < output_width;
                ++output_width_index) {
              if (filter_width_offset >= 0 && filter_width_offset <
                static_cast<int>(input_width)) {
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

#endif  // SRC_BACKENDS_MIC_BACKEND_PACK_INL_H_
