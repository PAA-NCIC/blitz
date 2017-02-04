#ifndef SRC_BACKENDS_CPU_BACKEND_PACK_INL_H_
#define SRC_BACKENDS_CPU_BACKEND_PACK_INL_H_

template<typename DType>
void Backend<CPUTensor, DType>::UnpackCHWImpl(
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
  for (size_t channel_index = 0; channel_index < channel; ++channel_index) {
    for (size_t output_height_index = 0; output_height_index < output_height; ++output_height_index) {
      for (size_t filter_height_index = 0; filter_height_index < filter_height; ++filter_height_index) {
        DType* unpack_slice = unpack + output_height_index * output_width * filter_height * filter_width * channel + channel_index * filter_height * filter_width + filter_height_index * filter_width;
        if (filter_height_index + output_height_index * stride_height < padding_height ||
          filter_height_index + output_height_index * stride_height >= padding_height + input_height) {
          for (size_t output_width_index = 0; output_width_index < output_width; ++output_width_index) {
            DType* unpack_slice_slice = unpack_slice + output_width_index * filter_height * filter_width * channel;
            for (size_t filter_width_index = 0; filter_width_index < filter_width; ++filter_width_index) {
              unpack_slice_slice[filter_width_index] = 0;
            }
          }
        } else {
          const DType* input_slice = input + channel_index * input_height * input_width + (output_height_index * stride_height + filter_height_index - padding_height) * input_width;
          for (size_t output_width_index = 0; output_width_index < output_width; ++output_width_index) {
            const DType* input_slice_slice = input_slice + output_width_index * stride_width;
            DType* unpack_slice_slice = unpack_slice + output_width_index * filter_height * filter_width * channel;
            size_t filter_width_index = 0; 
            for (; filter_width_index + output_width_index * stride_width < padding_width; ++filter_width_index) {
              unpack_slice_slice[filter_width_index] = 0;
            }
            size_t output_end = std::min(input_width + padding_width - output_width_index * stride_width, filter_width);
            size_t padding_end = std::min(input_width + 2 * padding_width, filter_width);
            for (; filter_width_index < output_end; ++filter_width_index) {
              unpack_slice_slice[filter_width_index] = input_slice_slice[filter_width_index - padding_width];
            }
            for (; filter_width_index < padding_end; ++filter_width_index) {
              unpack_slice_slice[filter_width_index] = 0;
            }
          }
        }
      }
    }
  }
}

template<typename DType>
void Backend<CPUTensor, DType>::UnpackHWCImpl(
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
  for (size_t output_height_index = 0; output_height_index < output_height; ++output_height_index) {
    for (size_t filter_height_index = 0; filter_height_index < filter_height; ++filter_height_index) {
      DType* unpack_slice = unpack + output_height_index * output_width * filter_height * filter_width * channel + filter_height_index * filter_width * channel;
      if (filter_height_index + output_height_index * stride_height < padding_height ||
        filter_height_index + output_height_index * stride_height >= padding_height + input_height) {
        for (size_t output_width_index = 0; output_width_index < output_width; ++output_width_index) {
          DType* unpack_slice_slice = unpack_slice + output_width_index * filter_height * filter_width * channel;
          for (size_t filter_width_index = 0; filter_width_index < filter_width * channel; ++filter_width_index) {
            unpack_slice_slice[filter_width_index] = 0;
          }
        }
      } else {
        const DType* input_slice = input + (output_height_index * stride_height + filter_height_index - padding_height) * input_width * channel;
        for (size_t output_width_index = 0; output_width_index < output_width; ++output_width_index) {
          const DType* input_slice_slice = input_slice + output_width_index * stride_width * channel;
          DType* unpack_slice_slice = unpack_slice + output_width_index * filter_height * filter_width * channel;
          size_t filter_width_index = 0; 
          for (; filter_width_index + output_width_index * stride_width < padding_width; ++filter_width_index) {
            for (size_t channel_index = 0; channel_index < channel; ++channel_index) {
              unpack_slice_slice[filter_width_index * channel + channel_index] = 0;
            }
          }
          size_t output_end = std::min(input_width + padding_width - output_width_index * stride_width, filter_width);
          size_t padding_end = std::min(input_width + 2 * padding_width, filter_width);
          for (; filter_width_index < output_end; ++filter_width_index) {
            for (size_t channel_index = 0; channel_index < channel; ++channel_index) {
              unpack_slice_slice[filter_width_index * channel + channel_index] = input_slice_slice[(filter_width_index - padding_width) * channel + channel_index];
            }
          }
          for (; filter_width_index < padding_end; ++filter_width_index) {
            for (size_t channel_index = 0; channel_index < channel; ++channel_index) {
              unpack_slice_slice[filter_width_index * channel + channel_index] = 0;
            }
          }
        }
      }
    }
  }
}

template<typename DType>
BLITZ_DATA_LAYOUT Backend<CPUTensor, DType>::Unpack2DFunc(
  const DType* input,
  DType* unpack,
  size_t C, size_t H, size_t W,
  size_t R, size_t S,
  size_t P, size_t Q,
  size_t pad_h, size_t pad_w,
  size_t str_h, size_t str_w,
  BLITZ_DATA_LAYOUT input_data_layout) {
  if (input_data_layout == BLITZ_BUFFER_NHWC) {
    UnpackHWCImpl(
      input, unpack,
      C, H, W, R, S, P, Q,
      pad_h, pad_w, str_h, str_w);
    return BLITZ_PACK_PQRSC;
  } else if (input_data_layout == BLITZ_BUFFER_NCHW) {
    UnpackCHWImpl(
      input, unpack,
      C, H, W, R, S, P, Q,
      pad_h, pad_w, str_h, str_w);
    return BLITZ_PACK_PQCRS;
  } else {
    LOG(FATAL) << "Unsupported input data layout: " << input_data_layout;
    return BLITZ_SHAPE_UNDEFINED;
  }
}

template<typename DType>
BLITZ_DATA_LAYOUT Backend<CPUTensor, DType>::Pack2DFunc(
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
  size_t stride_width,
  BLITZ_DATA_LAYOUT input_data_layout) {
  // (input_channel * filter_height * filter_width) *
  // (output_width * output_height)
  size_t pack_index = 0;
  for (size_t channel_index = 0; channel_index < channel; ++channel_index) {
    const size_t channel_offset = channel_index * input_height * input_width;
    DType* input_slice = input + channel_offset;
    for (size_t filter_height_index = 0; filter_height_index < filter_height; ++filter_height_index) {
      for (size_t filter_width_index = 0; filter_width_index < filter_width; ++filter_width_index) {
        int filter_height_offset = -padding_height + filter_height_index;
        for (size_t output_height_index = 0; output_height_index < output_height; ++output_height_index) {
          if (filter_height_offset < 0 || filter_height_offset >= static_cast<int>(input_height)) {
            pack_index += output_width;
          } else {
            int filter_width_offset = -padding_width + filter_width_index;
            for (size_t output_width_index = 0; output_width_index < output_width; ++output_width_index) {
              if (filter_width_offset >= 0 && filter_width_offset < static_cast<int>(input_width)) {
                input_slice[filter_height_offset * input_width + filter_width_offset] += pack[pack_index];
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
  return BLITZ_BUFFER_NCHW;
}

#endif  // SRC_BACKENDS_CPU_BACKEND_PACK_INL_H_
