#include <iostream>

#include <cudnn.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "backend/backends.h"
#include "util/blitz_gpu_function.h"

using namespace blitz;

cudnnHandle_t cudnn_handle_;
cudaStream_t cudnn_stream_;

// algorithms for forward and backwards convolutions
cudnnConvolutionFwdAlgo_t forward_algorithm_;
cudnnConvolutionBwdFilterAlgo_t backward_filter_algorithm_;
cudnnConvolutionBwdDataAlgo_t backward_data_algorithm_;

cudnnTensorDescriptor_t input_desc_, output_desc_;
cudnnFilterDescriptor_t filter_desc_;
cudnnConvolutionDescriptor_t conv_desc_;

float *cudnn_alpha_, *cudnn_beta_;

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

int main() {
  const int batch_size = 1;
  const int input_channel = 1;
  const int input_height = 3;
  const int input_width = 3;
  const int filter_height = 2;
  const int filter_width = 2;
  const int output_channel = 2;
  const int output_height = 2;
  const int output_width = 2;

  Shape input_shape(4);
  // batch_size
  input_shape[0] = batch_size;
  // input channel
  input_shape[1] = input_channel;
  // input height
  input_shape[2] = input_height;
  // input width
  input_shape[3] = input_width;
  CPUTensor<float> input(input_shape);
  init_input(input);
  GPUTensor<float> input_gpu(input_shape);
  cudaMemcpy(input_gpu.data(), input.data(), input.size() * sizeof(float),
    cudaMemcpyHostToDevice);

  Shape filter_shape(4);
  // output channel
  filter_shape[0] = output_channel;
  // input channel
  filter_shape[1] = input_channel;
  // filter height
  filter_shape[2] = filter_height;
  // filter width
  filter_shape[3] = filter_width;
  CPUTensor<float> weight(filter_shape);
  init_filter(weight);
  GPUTensor<float> weight_gpu(filter_shape);
  cudaMemcpy(weight_gpu.data(), weight.data(), weight.size() * sizeof(float),
    cudaMemcpyHostToDevice);

  Shape output_shape(4);
  // batch size
  output_shape[0] = batch_size;
  // output channel
  output_shape[1] = output_channel;
  // output height
  output_shape[2] = output_height;
  // output width
  output_shape[3] = output_width;
  CPUTensor<float> output(output_shape);
  GPUTensor<float> output_gpu(output_shape);
  
  // create val
  cudnn_alpha_ = new float(1.0);
  cudnn_beta_ = new float(0.0);

  // create handle
  CUDNN_CHECK(cudnnCreate(&cudnn_handle_));
  cudaStreamCreate(&cudnn_stream_);
  cudnnSetStream(cudnn_handle_, cudnn_stream_);

  // create descriptors
  cudnn::createTensor4dDesc<float>(&input_desc_);
  cudnn::createTensor4dDesc<float>(&output_desc_);
  cudnn::createFilterDesc<float>(&filter_desc_);
  cudnn::createConvolution2DDesc<float>(&conv_desc_);

  // set descriptors
  cudnn::setTensor4dDesc<float>(&input_desc_,
    batch_size, input_channel, input_height, input_width);
  cudnn::setTensor4dDesc<float>(&output_desc_,
    batch_size, output_channel, output_height, output_width);
  cudnn::setFilterDesc<float>(&filter_desc_, output_channel,
    input_channel, filter_height, filter_width);
  cudnn::setConvolution2DDesc<float>(&conv_desc_, 0, 0, 1, 1);

  // set algorithms
  forward_algorithm_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
  backward_filter_algorithm_ = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
  backward_data_algorithm_ = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;

  cudaDeviceSynchronize();
  float aa = 1.0f, bb = 0.0f;
  CUDNN_CHECK(cudnnConvolutionForward(cudnn_handle_, &aa,
    input_desc_, input_gpu.data(), filter_desc_, weight_gpu.data(),
    conv_desc_, forward_algorithm_, NULL, 0, &bb,
    output_desc_, output_gpu.data()));
  //Shape unpack_shape(2);
  //unpack_shape[0] = 1 * 2 * 2;
  //unpack_shape[1] = 2 * 2;
  //CPUTensor<float> unpack(unpack_shape);
  //unpack.Fill(0);
  //GPUTensor<float> unpack_gpu(unpack_shape);
  //Backend<GPUTensor, float>::Convolution2DForwardFunc(
  //  &input_gpu, &weight_gpu, 0, 0, 1, 1, &unpack_gpu, &output_gpu);
  cudaDeviceSynchronize();
  cudaMemcpy(output.data(), output_gpu.data(), output.size() * sizeof(float),
    cudaMemcpyDeviceToHost);
  report_conv_forward(output);

  // backward input
  cudnnConvolutionBackwardData(cudnn_handle_, (void*)cudnn_alpha_,
    filter_desc_, weight_gpu.data(), output_desc_, output_gpu.data(),
    conv_desc_, backward_data_algorithm_, NULL, 0,
    (void*)cudnn_beta_, input_desc_, input_gpu.data());
  cudaMemcpy(input.data(), input_gpu.data(), input.size() * sizeof(float),
    cudaMemcpyDeviceToHost);
  report_conv_backward(input);

  cudnnDestroy(cudnn_handle_);
}
