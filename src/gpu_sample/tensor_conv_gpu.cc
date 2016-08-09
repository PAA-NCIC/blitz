#include <iostream>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "backend/backends.h"

using namespace blitz;

/*
 * 0 1 0
 * 1 1 0
 * 0 1 0
 */
void init_input(CPUTensor<float>& input) {
  input[0] = 0;
  input[1] = 1;
  input[2] = 0;
  input[3] = 1;
  input[4] = 1;
  input[5] = 0;
  input[6] = 0;
  input[7] = 1;
  input[8] = 0;
}

/*
 * (output_channel) * (input_channel *
 * filter_height * filter_width)
 * output_channel 0
 * 0 0
 * 1 0
 *
 * output_channel 1
 * 1 0
 * 0 0
 */
void init_filter(CPUTensor<float>& weight) {
  weight[0] = 0;
  weight[1] = 0;
  weight[2] = 1;
  weight[3] = 0;
  weight[4] = 1;
  weight[5] = 0;
  weight[6] = 0;
  weight[7] = 0;
}

/*
 * expect output:
 * 1 1 0 1 0 1 1 1
 */
void report_conv_forward(CPUTensor<float>& output) {
  std::cout << "conv forward result: " << std::endl;
  const Shape& output_shape = output.shape();
  for (size_t i = 0; i < output_shape[0]; ++i) {
    for (size_t j = 0; j < output_shape[1]; ++j) {
      for (size_t k = 0; k < output_shape[2]; ++k) {
        for (size_t v = 0; v < output_shape[3]; ++v) {
          const int index = i * output_shape[1] * output_shape[2] *
            output_shape[3] + j * output_shape[2] * output_shape[3] +
            k * output_shape[3] + v;
          std::cout << "index: " << " i " << i << " j " << j << " k " << k << " v " << v << std::endl;
          std::cout << "value: " << output[index] << std::endl;
        }
      }
    }
  }
}

/*
 * expect output:
 * 1 1 0 1 0 1 1 1
 */
void report_conv_unpack(CPUTensor<float>& unpack) {
  std::cout << "conv unpack: " << std::endl;
  const Shape& unpack_shape = unpack.shape();
  for (size_t i = 0; i < unpack_shape[0]; ++i) {
    for (size_t j = 0; j < unpack_shape[1]; ++j) {
      const int index = i * unpack_shape[1] + j;
      std::cout << "index: " << " i " << i << " j " << j << std::endl;
      std::cout << "value: " << unpack[index] << std::endl;
    }
  }
}

/*
 * expect output:
 * 0 1 0 1 1 0 1 1 0 0 1 0
 */
void report_conv_backward(CPUTensor<float>& input) {
  std::cout << "conv backward result: " << std::endl;
  const Shape& input_shape = input.shape();
  for (size_t i = 0; i < input_shape[0]; ++i) {
    for (size_t j = 0; j < input_shape[1]; ++j) {
      for (size_t k = 0; k < input_shape[2]; ++k) {
        for (size_t v = 0; v < input_shape[3]; ++v) {
          const int index = i * input_shape[1] * input_shape[2] *
            input_shape[3] + j * input_shape[2] * input_shape[3] +
            k * input_shape[3] + v;
          std::cout << "index: " << " i " << i << " j " << j << " k " << k << " v " << v << std::endl;
          std::cout << "value: " << input[index] << std::endl;
        }
      }
    }
  }
}

/*
 * expect output:
 * 2 1 3 1 3 1 2 1
 */
void report_conv_update(CPUTensor<float>& update) {
  std::cout << "conv update result: " << std::endl;
  const Shape& update_shape = update.shape();
  for (size_t i = 0; i < update_shape[0]; ++i) {
    for (size_t j = 0; j < update_shape[1]; ++j) {
      for (size_t k = 0; k < update_shape[2]; ++k) {
        for (size_t v = 0; v < update_shape[3]; ++v) {
          const int index = i * update_shape[1] * update_shape[2] *
            update_shape[3] + j * update_shape[2] * update_shape[3] +
            k * update_shape[3] + v;
          std::cout << "index: " << " i " << i << " j " << j << " k " << k << " v " << v << std::endl;
          std::cout << "value: " << update[index] << std::endl;
        }
      }
    }
  }
}

int main() {
  Shape input_shape(4);
  // batch_size
  input_shape[0] = 1;
  // input channel
  input_shape[1] = 1;
  // input height
  input_shape[2] = 3;
  // input width
  input_shape[3] = 3;
  CPUTensor<float> input(input_shape);
  init_input(input);
  GPUTensor<float> input_gpu(input_shape);
  cudaMemcpy(input_gpu.data(), input.data(), input.size() * sizeof(float),
    cudaMemcpyHostToDevice);

  Shape filter_shape(4);
  // output channel
  filter_shape[0] = 2;
  // input channel
  filter_shape[1] = 1;
  // filter height
  filter_shape[2] = 2;
  // filter width
  filter_shape[3] = 2;
  CPUTensor<float> weight(filter_shape);
  init_filter(weight);
  GPUTensor<float> weight_gpu(filter_shape);
  cudaMemcpy(weight_gpu.data(), weight.data(), weight.size() * sizeof(float),
    cudaMemcpyHostToDevice);

  Shape unpack_shape(2);
  unpack_shape[0] = 1 * 2 * 2;
  unpack_shape[1] = 2 * 2;
  CPUTensor<float> unpack(unpack_shape);
  unpack.Fill(0);
  GPUTensor<float> unpack_gpu(unpack_shape);

  Shape output_shape(4);
  // batch size
  output_shape[0] = 1;
  // output channel
  output_shape[1] = 2;
  // output height
  output_shape[2] = 2;
  // output width
  output_shape[3] = 2;
  CPUTensor<float> output(output_shape);
  GPUTensor<float> output_gpu(output_shape);

  int stride_height = 1;
  int stride_width = 1;
  int padding_height = 0;
  int padding_width = 0;

  // pack
  Backend<GPUTensor, float>::Convolution2DForwardFunc(
    &input_gpu, &weight_gpu, padding_height, padding_width,
    stride_height, stride_width, &unpack_gpu, &output_gpu);
  cudaMemcpy(output.data(), output_gpu.data(), output.size() * sizeof(float),
    cudaMemcpyDeviceToHost);

  std::cout << "pack:" << std::endl;
  report_conv_forward(output);

  cudaMemcpy(unpack.data(), unpack_gpu.data(), unpack.size() * sizeof(float),
    cudaMemcpyDeviceToHost);
  report_conv_unpack(unpack);
  // backward input
  CPUTensor<float> backward_input(input_shape);
  backward_input.Fill(0);
  GPUTensor<float> backward_input_gpu(input_shape);
  unpack_gpu.Fill(0);

  Backend<GPUTensor, float>::Convolution2DBackwardFunc(
    &output_gpu, &weight_gpu, padding_height, padding_width,
    stride_height, stride_width, &unpack_gpu, &backward_input_gpu);

  cudaMemcpy(backward_input.data(), backward_input_gpu.data(), backward_input.size() * sizeof(float),
    cudaMemcpyDeviceToHost);
  report_conv_backward(backward_input);

  unpack_gpu.Fill(0);
  // update
  CPUTensor<float> update(filter_shape);
  update.Fill(0);
  GPUTensor<float> update_gpu(filter_shape);
  update_gpu.Fill(0);

  Backend<GPUTensor, float>::Convolution2DUpdateFunc(
    &input_gpu, &output_gpu, padding_height, padding_width,
    stride_height, stride_width, &unpack_gpu, &update_gpu);
  cudaMemcpy(update.data(), update_gpu.data(), update.size() * sizeof(float),
    cudaMemcpyDeviceToHost);

  cudaMemcpy(unpack.data(), unpack_gpu.data(), unpack.size() * sizeof(float),
    cudaMemcpyDeviceToHost);
  report_conv_unpack(unpack);

  report_conv_update(update);

  return 0;
}
