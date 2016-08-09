#include <iostream>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "backend/backends.h"

using namespace blitz;

void tensor_compare(CPUTensor<float>& left, CPUTensor<float>& right) {
  for (size_t i = 0; i < left.size(); ++i) {
    if (!(left[i] <= right[i] + 1e6 && left[i] >= right[i] - 1e6)) {
      std::cout << "wrong index " << i << " value " << left[i] << ": " << right[i] << std::endl;
    }
  }
}

void forward_compare() {
  Shape input_shape(4);
  // batch_size
  input_shape[0] = 128;
  // input channel
  input_shape[1] = 3;
  // input height
  input_shape[2] = 28;
  // input width
  input_shape[3] = 28;

  Shape filter_shape(4);
  // output channel
  filter_shape[0] = 16;
  // input channel
  filter_shape[1] = 3;
  // filter height
  filter_shape[2] = 5;
  // filter width
  filter_shape[3] = 5;

  Shape output_shape(4);
  // batch_size
  output_shape[0] = 128;
  // output channel
  output_shape[1] = 16;
  // output height
  output_shape[2] = 24;
  // output width
  output_shape[3] = 24;

  Shape unpack_shape(2);
  unpack_shape[0] = 24 * 24;
  unpack_shape[1] = 5 * 5 * 3;

  CPUTensor<float> input(input_shape);
  CPUTensor<float> filter(filter_shape);
  CPUTensor<float> output(output_shape);
  CPUTensor<float> output_copy(output_shape);
  CPUTensor<float> unpack(unpack_shape);
  GPUTensor<float> filter_gpu(filter_shape);
  GPUTensor<float> output_gpu(output_shape);
  GPUTensor<float> input_gpu(input_shape);
  GPUTensor<float> unpack_gpu(unpack_shape);

  Backend<CPUTensor, float>::NormalDistributionFunc(0, 1, &input);
  Backend<GPUTensor, float>::HostCopyToFunc(input.data(),
    input.size(), input_gpu.data());
  Backend<CPUTensor, float>::NormalDistributionFunc(0, 1, &filter);
  Backend<GPUTensor, float>::HostCopyToFunc(filter.data(),
    filter.size(), filter_gpu.data());

  Backend<CPUTensor, float>::Convolution2DForwardFunc(&input, &filter,
    0, 0, 1, 1, &unpack, &output);
  Backend<GPUTensor, float>::Convolution2DForwardFunc(&input_gpu, &filter_gpu,
    0, 0, 1, 1, &unpack_gpu, &output_gpu);
  cudaMemcpy(output_copy.data(), output_gpu.data(), output_copy.size() * sizeof(float),
    cudaMemcpyDeviceToHost);

  std::cout << "forward compare:" << std::endl;
  tensor_compare(output_copy, output);
}

void backward_compare() {
  Shape input_shape(4);
  // batch_size
  input_shape[0] = 128;
  // input channel
  input_shape[1] = 3;
  // input height
  input_shape[2] = 28;
  // input width
  input_shape[3] = 28;

  Shape filter_shape(4);
  // output channel
  filter_shape[0] = 16;
  // input channel
  filter_shape[1] = 3;
  // filter height
  filter_shape[2] = 5;
  // filter width
  filter_shape[3] = 5;

  Shape output_shape(4);
  // batch_size
  output_shape[0] = 128;
  // output channel
  output_shape[1] = 16;
  // output height
  output_shape[2] = 24;
  // output width
  output_shape[3] = 24;

  Shape unpack_shape(2);
  unpack_shape[0] = 24 * 24;
  unpack_shape[1] = 5 * 5 * 3;

  CPUTensor<float> input(input_shape);
  CPUTensor<float> input_copy(input_shape);
  CPUTensor<float> filter(filter_shape);
  CPUTensor<float> output(output_shape);
  CPUTensor<float> unpack(unpack_shape);
  GPUTensor<float> filter_gpu(filter_shape);
  GPUTensor<float> output_gpu(output_shape);
  GPUTensor<float> input_gpu(input_shape);
  GPUTensor<float> unpack_gpu(unpack_shape);

  Backend<CPUTensor, float>::NormalDistributionFunc(0, 1, &output);
  Backend<GPUTensor, float>::HostCopyToFunc(output.data(),
    output.size(), output_gpu.data());
  Backend<CPUTensor, float>::NormalDistributionFunc(0, 1, &filter);
  Backend<GPUTensor, float>::HostCopyToFunc(filter.data(),
    filter.size(), filter_gpu.data());

  Backend<CPUTensor, float>::Convolution2DBackwardFunc(&output, &filter,
    0, 0, 1, 1, &unpack, &input);
  Backend<GPUTensor, float>::Convolution2DBackwardFunc(&output_gpu, &filter_gpu,
    0, 0, 1, 1, &unpack_gpu, &input_gpu);
  cudaMemcpy(input_copy.data(), input_gpu.data(), input_copy.size() * sizeof(float),
    cudaMemcpyDeviceToHost);

  std::cout << "backward compare:" << std::endl;
  tensor_compare(input_copy, input);
}

void backward_weight_compare() {
  Shape input_shape(4);
  // batch_size
  input_shape[0] = 128;
  // input channel
  input_shape[1] = 4;
  // input height
  input_shape[2] = 28;
  // input width
  input_shape[3] = 28;

  Shape filter_shape(4);
  // output channel
  filter_shape[0] = 16;
  // input channel
  filter_shape[1] = 4;
  // filter height
  filter_shape[2] = 5;
  // filter width
  filter_shape[3] = 5;

  Shape output_shape(4);
  // batch_size
  output_shape[0] = 128;
  // output channel
  output_shape[1] = 16;
  // output height
  output_shape[2] = 24;
  // output width
  output_shape[3] = 24;

  Shape unpack_shape(2);
  unpack_shape[0] = 24 * 24;
  unpack_shape[1] = 5 * 5 * 4;

  CPUTensor<float> input(input_shape);
  CPUTensor<float> filter(filter_shape);
  CPUTensor<float> filter_copy(filter_shape);
  CPUTensor<float> output(output_shape);
  CPUTensor<float> unpack(unpack_shape);
  GPUTensor<float> filter_gpu(filter_shape);
  GPUTensor<float> output_gpu(output_shape);
  GPUTensor<float> input_gpu(input_shape);
  GPUTensor<float> unpack_gpu(unpack_shape);

  Backend<CPUTensor, float>::NormalDistributionFunc(0, 1, &input);
  Backend<GPUTensor, float>::HostCopyToFunc(input.data(),
    input.size(), input_gpu.data());
  Backend<CPUTensor, float>::NormalDistributionFunc(0, 1, &output);
  Backend<GPUTensor, float>::HostCopyToFunc(output.data(),
    output.size(), output_gpu.data());

  Backend<CPUTensor, float>::Convolution2DUpdateFunc(&input, &output,
    0, 0, 1, 1, &unpack, &filter);
  Backend<GPUTensor, float>::Convolution2DUpdateFunc(&input_gpu, &output_gpu,
    0, 0, 1, 1, &unpack_gpu, &filter_gpu);
  cudaMemcpy(filter_copy.data(), filter_gpu.data(), filter_copy.size() * sizeof(float),
    cudaMemcpyDeviceToHost);

  std::cout << "backward update compare:" << std::endl;
  tensor_compare(filter_copy, filter);
}

int main() {
  forward_compare();
  backward_compare();
  backward_weight_compare();
  return 0;
}
