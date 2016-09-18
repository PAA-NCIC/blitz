#ifndef SRC_BACKEND_GPU_BACKEND_PACK_INL_H_
#define SRC_BACKEND_GPU_BACKEND_PACK_INL_H_

// small kernel
template<typename DType>
__global__ void GPUUnpack1024Kernel(const DType* input,
  size_t input_height, size_t input_width,
  size_t filter_height, size_t filter_width,
  size_t padding_height, size_t padding_width,
  size_t stride_height, size_t stride_width,
  DType* unpack) {
  size_t output_height_index = threadIdx.x;
  size_t output_width_index = threadIdx.y;
  size_t input_channel_index = blockIdx.x;
  size_t output_width = blockDim.y;
  size_t input_channel = gridDim.x;
  size_t height_offset = output_height_index *
    stride_height - padding_height;
  size_t width_offset = output_width_index *
    stride_width - padding_width;
  const DType* p_input = input +
    (input_channel_index * input_height + height_offset) *
    input_width + width_offset;
  DType* p_unpack = unpack +
    (output_height_index * output_width + output_width_index) *
    filter_height * filter_width * input_channel +
    input_channel_index * filter_height * filter_width;

  int height_offset_index, width_offset_index;
  for (int i = 0; i < filter_height; ++i) {
    height_offset_index = height_offset + i;
    if (height_offset_index < 0 || height_offset_index >= input_height) {
      for (int j = 0; j < filter_width; ++j) {
        *p_unpack++ = 0;
      }
    } else {
      for (int j = 0; j < filter_width; ++j) {
        width_offset_index = width_offset + j;
        if (width_offset_index < 0 || width_offset_index >= input_width) {
          *p_unpack++ = 0;
        } else {
          *p_unpack++ = p_input[i * input_width + j];
        }
      }
    }
  }
}

// general kernel
template<typename DType>
__global__ void GPUUnpackKernel(const DType* input,
  size_t size, size_t input_channel,
  size_t input_height, size_t input_width,
  size_t filter_height, size_t filter_width,
  size_t output_height, size_t output_width,
  size_t padding_height, size_t padding_width,
  size_t stride_height, size_t stride_width,
  DType* unpack) {
  BLITZ_CUDA_LOOP(index, size) {
    size_t channel_output_offset = index / output_width;
    size_t output_height_index = channel_output_offset % output_height;
    size_t output_width_index = index % output_width;
    size_t input_channel_index = channel_output_offset / output_height;
    size_t height_offset = output_height_index * stride_height - padding_height;
    size_t width_offset = output_width_index * stride_width - padding_width;
    const DType* p_input = input +
      (input_channel_index * input_height + height_offset) *
      input_width + width_offset;
    DType* p_unpack = unpack +
      (output_height_index * output_width + output_width_index) *
      filter_height * filter_width * input_channel +
      input_channel_index * filter_height * filter_width;

    int height_offset_index, width_offset_index;
    for (int i = 0; i < filter_height; ++i) {
      height_offset_index = height_offset + i;
      if (height_offset_index < 0 || height_offset_index >= input_height) {
        for (int j = 0; j < filter_width; ++j) {
          *p_unpack++ = 0;
        }
      } else {
        for (int j = 0; j < filter_width; ++j) {
          width_offset_index = width_offset + j;
          if (width_offset_index < 0 || width_offset_index >= input_width) {
            *p_unpack++ = 0;
          } else {
            *p_unpack++ = p_input[i * input_width + j];
          }
        }
      }
    }
  }
}

template<typename DType>
void Backend<GPUTensor, DType>::Unpack2DParallelFunc(
  const DType* input, size_t channel,
  size_t input_height, size_t input_width,
  size_t filter_height, size_t filter_width,
  size_t output_height, size_t output_width,
  size_t padding_height, size_t padding_width,
  size_t stride_height, size_t stride_width,
  DType* unpack) {
  if (channel <= 64 && output_height * output_width <= 256) {
    dim3 thread_per_block(output_height, output_width);
    GPUUnpack1024Kernel<DType><<<channel, thread_per_block>>>(
      input, input_height, input_width,
      filter_height, filter_width,
      padding_height, padding_width,
      stride_height, stride_width, unpack);
  } else {
    size_t size = channel * output_height * output_width;
    GPUUnpackKernel<DType><<<BlitzGPUGetBlocks(size),
      BLITZ_NUM_GPU_THREADS>>>(input, size, channel,
      input_height, input_width, filter_height, filter_width,
      output_height, output_width, padding_height, padding_width,
      stride_height, stride_width, unpack);
  }
}

// small kernel
template<typename DType>
__global__ void GPUPack1024Kernel(const DType* pack,
  size_t filter_height, size_t filter_width,
  size_t output_height, size_t output_width,
  size_t padding_height, size_t padding_width,
  size_t stride_height, size_t stride_width,
  DType* input) {
  size_t input_height_index = threadIdx.x;
  size_t input_width_index = threadIdx.y;
  size_t input_channel_index = blockIdx.x;
  size_t input_height_padding = input_height_index + padding_height;
  size_t input_width_padding = input_width_index + padding_width;
  size_t input_height = blockDim.x;
  size_t input_width = blockDim.y;
  size_t input_channel = gridDim.x;
  size_t pack_width =  filter_height * filter_width * input_channel;

  size_t pack_height_start = input_height_padding < filter_height ?
    0 : (input_height_padding - filter_height) / stride_height + 1;
  size_t pack_height_end =
    min(input_height_padding / stride_height + 1, output_height);
  size_t pack_width_start = input_width_padding < filter_width ?
    0 : (input_width_padding - filter_width) / stride_width + 1;
  size_t pack_width_end =
    min(input_width_padding / stride_width + 1, output_width);

  const DType *p_pack = pack + filter_width * filter_height * input_channel_index;
  DType* p_input = input + input_channel_index * input_height * input_width +
    input_height_index * input_width + input_width_index;

  DType sum = 0.0;
  int filter_height_index, filter_width_index;
  for (int i = pack_height_start; i < pack_height_end; ++i) {
    for (int j = pack_width_start; j < pack_width_end; ++j) {
      filter_height_index = (input_height_padding - i * stride_height);
      filter_width_index = (input_width_padding - j * stride_width);
      sum += p_pack[(i * output_width + j) * pack_width +
        filter_height_index * filter_width + filter_width_index];
    }
  }
  *(p_input) = sum;
}

// general kernel
template<typename DType>
__global__ void GPUPackKernel(const DType* pack,
  size_t size, size_t input_channel,
  size_t input_height, size_t input_width,
  size_t filter_height, size_t filter_width,
  size_t output_height, size_t output_width,
  size_t padding_height, size_t padding_width,
  size_t stride_height, size_t stride_width,
  DType* input) {
  BLITZ_CUDA_LOOP(index, size) {
    size_t channel_height_offset = index / input_width;
    size_t input_height_index = channel_height_offset % input_height;
    size_t input_width_index = index % input_width;
    size_t input_channel_index = channel_height_offset % input_channel;
    size_t input_height_padding = input_height_index + padding_height;
    size_t input_width_padding = input_width_index + padding_width;
    size_t pack_width =  filter_height * filter_width * input_channel;

    size_t pack_height_start = input_height_padding < filter_height ?
      0 : (input_height_padding - filter_height) / stride_height + 1;
    size_t pack_height_end =
      min(input_height_padding / stride_height + 1, output_height);
    size_t pack_width_start = input_width_padding < filter_width ?
      0 : (input_width_padding - filter_width) / stride_width + 1;
    size_t pack_width_end =
      min(input_width_padding / stride_width + 1, output_width);

    const DType *p_pack = pack + filter_width * filter_height * input_channel_index;
    DType* p_input = input + input_channel_index * input_height * input_width +
      input_height_index * input_width + input_width_index;

    DType sum = 0.0;
    int filter_height_index, filter_width_index;
    for (int i = pack_height_start; i < pack_height_end; ++i) {
      for (int j = pack_width_start; j < pack_width_end; ++j) {
        filter_height_index = (input_height_padding - i * stride_height);
        filter_width_index = (input_width_padding - j * stride_width);
        sum += p_pack[(i * output_width + j) * pack_width +
          filter_height_index * filter_width + filter_width_index];
      }
    }
    *(p_input) = sum;
  }
}

template<typename DType>
void Backend<GPUTensor, DType>::Pack2DParallelFunc(
  const DType* pack, size_t channel,
  size_t input_height, size_t input_width,
  size_t filter_height, size_t filter_width,
  size_t output_height, size_t output_width,
  size_t padding_height, size_t padding_width,
  size_t stride_height, size_t stride_width,
  DType* input) {
  if (channel <= 64 && input_height * input_width <= 256) {
    dim3 thread_per_block(input_height, input_width);
    GPUPack1024Kernel<DType><<<channel, thread_per_block>>>(
      pack, filter_height, filter_width, output_height, output_width,
      padding_height, padding_width, stride_height, stride_width, input);
  } else {
    size_t size = channel * input_height * input_width;
    GPUPackKernel<DType><<<BlitzGPUGetBlocks(size), BLITZ_NUM_GPU_THREADS>>>(
      pack, size, channel, input_height, input_width,
      filter_height, filter_width, output_height, output_width,
      padding_height, padding_width, stride_height, stride_width, input);
  }
}

template<typename DType>
void Backend<GPUTensor, DType>::Unpack2DFunc(
  const DType* input, size_t channel,
  size_t input_height, size_t input_width,
  size_t filter_height, size_t filter_width,
  size_t output_height, size_t output_width,
  size_t padding_height, size_t padding_width,
  size_t stride_height, size_t stride_width,
  DType* unpack) {}

template<typename DType>
void Backend<GPUTensor, DType>::Pack2DFunc(
  const DType* pack, size_t channel,
  size_t input_height, size_t input_width,
  size_t filter_height, size_t filter_width,
  size_t output_height, size_t output_width,
  size_t padding_height, size_t padding_width,
  size_t stride_height, size_t stride_width,
  DType* input) {}

#endif  // SRC_BACKEND_GPU_BACKEND_PACK_INL_H_
