#ifndef SRC_BACKENDS_CPU_BACKEND_PACK_INL_H_
#define SRC_BACKENDS_CPU_BACKEND_PACK_INL_H_

template<typename DType>
void Backend<CPUTensor, DType>::UnpackStrideMultiPadCHWImpl(
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
void Backend<CPUTensor, DType>::UnpackStrideMultiPadHWCImpl(
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
void Backend<CPUTensor, DType>::UnpackStrideMultiCHWImpl(
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
        const DType* input_slice = input + channel_index * input_height * input_width + (output_height_index * stride_height + filter_height_index) * input_width;
        DType* unpack_slice = unpack + output_height_index * output_width * filter_height * filter_width * channel + channel_index * filter_height * filter_width + filter_height_index * filter_width;
        for (size_t output_width_index = 0; output_width_index < output_width; ++output_width_index) {
          for (size_t filter_width_index = 0; filter_width_index < filter_width; ++filter_width_index) {
            unpack_slice[filter_width_index] = input_slice[filter_width_index];
          }
          unpack_slice += filter_height * filter_width * channel;
          input_slice += stride_width;
        }
      }
    }
  }
}

template<typename DType>
void Backend<CPUTensor, DType>::UnpackStrideMultiHWCImpl(
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
      const DType* input_slice = input + (output_height_index * stride_height + filter_height_index) * input_width * channel;
      DType* unpack_slice = unpack + output_height_index * output_width * filter_height * filter_width * channel + filter_height_index * filter_width * channel;
      for (size_t output_width_index = 0; output_width_index < output_width; ++output_width_index) {
        const DType* input_slice_slice = input_slice + output_width_index * stride_width * channel;
        DType* unpack_slice_slice = unpack_slice + output_width_index * filter_height * filter_width * channel;
        for (size_t filter_width_index = 0; filter_width_index < filter_width * channel; ++filter_width_index) {
          unpack_slice_slice[filter_width_index] = input_slice_slice[filter_width_index];
        }
      }
    }
  }
}

template<typename DType>
void Backend<CPUTensor, DType>::UnpackStrideOneCHWImpl(
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
  DType* raw_unpack = unpack;
  DType* prev_unpack = unpack;
  const DType* raw_input = input;
  const DType* prev_input = input;
  for (size_t channel_index = 0; channel_index < channel; ++channel_index) {
    prev_input = raw_input + channel_index * input_height * input_width;
    input = prev_input;
    prev_unpack = raw_unpack + channel_index * filter_height * filter_width * output_height * output_width;
    unpack = prev_unpack;
    for (size_t output_height_index = 0; output_height_index < output_height; ++output_height_index) {
      for (size_t filter_height_index = 0; filter_height_index < filter_height; ++filter_height_index) {
        const DType* input_slice = prev_input + filter_height_index * input_width;
        DType* unpack_slice = prev_unpack + filter_height_index * filter_width * output_height * output_width;
        for (size_t filter_width_index = 0; filter_width_index < filter_width; ++filter_width_index) {
          unpack = unpack_slice + filter_width_index * output_height * output_width;
          input = input_slice + filter_width_index;
          for (size_t output_width_index = 0; output_width_index < output_width; ++output_width_index) {
            unpack[output_width_index] = input[output_width_index];
          }
        }
      }
      prev_unpack += output_width;
      prev_input += input_width;
      unpack = prev_unpack;
      input = prev_input;
    }
  }
}

template<typename DType>
void Backend<CPUTensor, DType>::UnpackStrideOnePadCHWImpl(
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
  DType* raw_unpack = unpack;
  DType* prev_unpack = unpack;
  const DType* raw_input = input;
  const DType* prev_input = input;
  for (size_t channel_index = 0; channel_index < channel; ++channel_index) {
    prev_input = raw_input + channel_index * input_height * input_width;
    input = prev_input;
    prev_unpack = raw_unpack + channel_index * filter_height * filter_width * output_height * output_width;
    unpack = prev_unpack;
    for (size_t output_height_index = 0; output_height_index < output_height; ++output_height_index) {
      for (size_t filter_height_index = 0; filter_height_index < filter_height; ++filter_height_index) {
        if (filter_height_index + output_height_index < padding_height ||
          filter_height_index + output_height_index >= padding_height + input_height) {
          for (size_t filter_width_index = 0; filter_width_index < filter_width; ++filter_width_index) {
            for (size_t output_width_index = 0; output_width_index < output_width; ++output_width_index) {
              unpack[output_width_index] = 0;
            }
            unpack += output_height * output_width;
          }
        } else {
          for (size_t filter_width_index = 0; filter_width_index < filter_width; ++filter_width_index) {
            size_t output_width_index = 0;
            for (; filter_width_index + output_width_index < padding_width; ++output_width_index) {
              unpack[output_width_index] = 0;
            }
            const size_t output_end = std::min(padding_width + input_width - filter_width_index, output_width);
            const size_t padding_end = std::min(input_width + 2 * padding_width - filter_width_index, output_width);
            for (; output_width_index < output_end; ++output_width_index) {
              unpack[output_width_index] = input[output_width_index - padding_width];
            }
            for (; output_width_index < padding_end; ++output_width_index) {
              unpack[output_width_index] = 0;
            }
            unpack += output_height * output_width;
            input++;  
          }
          input += input_width - filter_width;
        }
      }
      prev_unpack += output_width;
      if (output_height_index >= padding_height) {
        prev_input += input_width;
      }
      unpack = prev_unpack;
      input = prev_input;
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
    if (pad_h == 0 && pad_w == 0) {
      UnpackStrideMultiHWCImpl(
        input, unpack,
        C, H, W, R, S, P, Q,
        pad_h, pad_w, str_h, str_w);
    } else {
      UnpackStrideMultiPadHWCImpl(
        input, unpack,
        C, H, W, R, S, P, Q,
        pad_h, pad_w, str_h, str_w);
    }
    return BLITZ_PACK_PQRSC;
  } else if (input_data_layout == BLITZ_BUFFER_NCHW) {
    if (str_h == 1 && str_w == 1) {
      if (pad_h == 0 && pad_w == 0) {
        UnpackStrideOneCHWImpl(
          input, unpack,
          C, H, W, R, S, P, Q,
          pad_h, pad_w, str_h, str_w);
      } else {
        UnpackStrideOnePadCHWImpl(
          input, unpack,
          C, H, W, R, S, P, Q,
          pad_h, pad_w, str_h, str_w);
      }
      return BLITZ_PACK_CRSPQ;
    } else {
      if (pad_h == 0 && pad_w == 0) {
        UnpackStrideMultiCHWImpl(
          input, unpack,
          C, H, W, R, S, P, Q,
          pad_h, pad_w, str_h, str_w);
      } else {
        UnpackStrideMultiPadCHWImpl(
          input, unpack,
          C, H, W, R, S, P, Q,
          pad_h, pad_w, str_h, str_w);
      }
      return BLITZ_PACK_PQCRS;
    }
  } else {
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
