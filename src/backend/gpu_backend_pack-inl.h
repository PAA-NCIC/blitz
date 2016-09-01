#ifndef SRC_BACKEND_GPU_BACKEND_PACK_INL_H_
#define SRC_BACKEND_GPU_BACKEND_PACK_INL_H_

// small kernel
template<typename DType>
__global__ void GPUUnpack1024Kernel(const DType* input,
  const int input_height, const int input_width,
  const int filter_height, const int filter_width,
  const int padding_height, const int padding_width,
  const int stride_height, const int stride_width,
  DType* unpack) {
  const int output_height_index = threadIdx.x;
  const int output_width_index = threadIdx.y;
  const int input_channel_index = blockIdx.x;
  const int output_width = blockDim.y;
  const int input_channel = gridDim.x;
  const int height_offset = output_height_index *
    stride_height - padding_height;
  const int width_offset = output_width_index *
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
  const int size, const int input_channel,
  const int input_height, const int input_width,
  const int filter_height, const int filter_width,
  const int output_height, const int output_width,
  const int padding_height, const int padding_width,
  const int stride_height, const int stride_width,
  DType* unpack) {
  BLITZ_CUDA_LOOP(index, size) {
    const int channel_output_offset = index / output_width;
    const int output_height_index = channel_output_offset % output_height;
    const int output_width_index = index % output_width;
    const int input_channel_index = channel_output_offset / output_height;
    const int height_offset = output_height_index * stride_height - padding_height;
    const int width_offset = output_width_index * stride_width - padding_width;
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
  const DType* input, const int channel,
  const int input_height, const int input_width,
  const int filter_height, const int filter_width,
  const int output_height, const int output_width,
  const int padding_height, const int padding_width,
  const int stride_height, const int stride_width,
  DType* unpack) {
  if (channel <= 64 && output_height * output_width <= 256) {
    dim3 thread_per_block(output_height, output_width);
    GPUUnpack1024Kernel<DType><<<channel, thread_per_block>>>(
      input, input_height, input_width,
      filter_height, filter_width,
      padding_height, padding_width,
      stride_height, stride_width, unpack);
  } else {
    const int size = channel * output_height * output_width;
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
  const int filter_height, const int filter_width,
  const int output_height, const int output_width,
  const int padding_height, const int padding_width,
  const int stride_height, const int stride_width,
  DType* input) {
  const int input_height_index = threadIdx.x;
  const int input_width_index = threadIdx.y;
  const int input_channel_index = blockIdx.x;
  const int input_height_padding = input_height_index + padding_height;
  const int input_width_padding = input_width_index + padding_width;
  const int input_height = blockDim.x;
  const int input_width = blockDim.y;
  const int input_channel = gridDim.x;
  const int pack_width =  filter_height * filter_width * input_channel;

  const int pack_height_start = input_height_padding < filter_height ?
    0 : (input_height_padding - filter_height) / stride_height + 1;
  const int pack_height_end =
    min(input_height_padding / stride_height + 1, output_height);
  const int pack_width_start = input_width_padding < filter_width ?
    0 : (input_width_padding - filter_width) / stride_width + 1;
  const int pack_width_end =
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
  const int size, const int input_channel,
  const int input_height, const int input_width,
  const int filter_height, const int filter_width,
  const int output_height, const int output_width,
  const int padding_height, const int padding_width,
  const int stride_height, const int stride_width,
  DType* input) {
  BLITZ_CUDA_LOOP(index, size) {
    const int channel_height_offset = index / input_width;
    const int input_height_index = channel_height_offset % input_height;
    const int input_width_index = index % input_width;
    const int input_channel_index = channel_height_offset % input_channel;
    const int input_height_padding = input_height_index + padding_height;
    const int input_width_padding = input_width_index + padding_width;
    const int pack_width =  filter_height * filter_width * input_channel;

    const int pack_height_start = input_height_padding < filter_height ?
      0 : (input_height_padding - filter_height) / stride_height + 1;
    const int pack_height_end =
      min(input_height_padding / stride_height + 1, output_height);
    const int pack_width_start = input_width_padding < filter_width ?
      0 : (input_width_padding - filter_width) / stride_width + 1;
    const int pack_width_end =
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
  const DType* pack, const int channel,
  const int input_height, const int input_width,
  const int filter_height, const int filter_width,
  const int output_height, const int output_width,
  const int padding_height, const int padding_width,
  const int stride_height, const int stride_width,
  DType* input) {
  if (channel <= 64 && input_height * input_width <= 256) {
    dim3 thread_per_block(input_height, input_width);
    GPUPack1024Kernel<DType><<<channel, thread_per_block>>>(
      pack, filter_height, filter_width, output_height, output_width,
      padding_height, padding_width, stride_height, stride_width, input);
  } else {
    const int size = channel * input_height * input_width;
    GPUPackKernel<DType><<<BlitzGPUGetBlocks(size), BLITZ_NUM_GPU_THREADS>>>(
      pack, size, channel, input_height, input_width,
      filter_height, filter_width, output_height, output_width,
      padding_height, padding_width, stride_height, stride_width, input);
  }
}

template<typename DType>
void Backend<GPUTensor, DType>::Unpack2DFunc(
  const DType* input, const int channel,
  const int input_height, const int input_width,
  const int filter_height, const int filter_width,
  const int output_height, const int output_width,
  const int padding_height, const int padding_width,
  const int stride_height, const int stride_width,
  DType* unpack) {}

template<typename DType>
void Backend<GPUTensor, DType>::Pack2DFunc(
  const DType* pack, const int channel,
  const int input_height, const int input_width,
  const int filter_height, const int filter_width,
  const int output_height, const int output_width,
  const int padding_height, const int padding_width,
  const int stride_height, const int stride_width,
  DType* input) {}

#endif  // SRC_BACKEND_GPU_BACKEND_PACK_INL_H_
