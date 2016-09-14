#include <iostream>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "backend/backends.h"

using namespace blitz;

void forward1() {
  std::cout << "forward1" << std::endl;
  Shape input_shape(4);
  // batch_size
  input_shape[0] = 128;
  // input channel
  input_shape[1] = 3;
  // input height
  input_shape[2] = 224;
  // input width
  input_shape[3] = 224;

  Shape filter_shape(4);
  // output channel
  filter_shape[0] = 64;
  // input channel
  filter_shape[1] = 3;
  // filter height
  filter_shape[2] = 11;
  // filter width
  filter_shape[3] = 11;

  Shape output_shape(4);
  // batch_size
  output_shape[0] = 128;
  // output channel
  output_shape[1] = 64;
  // output height
  output_shape[2] = 55;
  // output width
  output_shape[3] = 55;

  Shape unpack_shape(2);
  unpack_shape[0] = 55 * 55;
  unpack_shape[1] = 11 * 11 * 3;

  GPUTensor<float> input(input_shape);
  GPUTensor<float> filter(filter_shape);
  GPUTensor<float> output(output_shape);
  GPUTensor<float> output_copy(output_shape);
  GPUTensor<float> unpack(unpack_shape);
  GPUTensor<float> filter_gpu(filter_shape);
  GPUTensor<float> output_gpu(output_shape);
  GPUTensor<float> input_gpu(input_shape);
  GPUTensor<float> unpack_gpu(unpack_shape);

  Backend<GPUTensor, float>::NormalDistributionFunc(0, 1, &input);
  Backend<GPUTensor, float>::NormalDistributionFunc(0, 1, &filter);

  const int iter = 3;
  string kernel = "asm";
  for (int i = 0; i < iter; ++i) {
    Backend<GPUTensor, float>::Convolution2DForwardFunc(&input, &filter,
      3, 3, 4, 4, &unpack, &output, kernel);
  }
  kernel = "blas";
  for (int i = 0; i < iter; ++i) {
    Backend<GPUTensor, float>::Convolution2DForwardFunc(&input_gpu, &filter_gpu,
      3, 3, 4, 4, &unpack_gpu, &output_gpu, kernel);
  }
}

void forward2() {
  std::cout << "forward2" << std::endl;
  Shape input_shape(4);
  // batch_size
  input_shape[0] = 128;
  // input channel
  input_shape[1] = 64;
  // input height
  input_shape[2] = 27;
  // input width
  input_shape[3] = 27;

  Shape filter_shape(4);
  // output channel
  filter_shape[0] = 192;
  // input channel
  filter_shape[1] = 64;
  // filter height
  filter_shape[2] = 5;
  // filter width
  filter_shape[3] = 5;

  Shape output_shape(4);
  // batch_size
  output_shape[0] = 128;
  // output channel
  output_shape[1] = 192;
  // output height
  output_shape[2] = 27;
  // output width
  output_shape[3] = 27;

  Shape unpack_shape(2);
  unpack_shape[0] = 27 * 27;
  unpack_shape[1] = 5 * 5 * 64;

  GPUTensor<float> input(input_shape);
  GPUTensor<float> filter(filter_shape);
  GPUTensor<float> output(output_shape);
  GPUTensor<float> output_copy(output_shape);
  GPUTensor<float> unpack(unpack_shape);
  GPUTensor<float> filter_gpu(filter_shape);
  GPUTensor<float> output_gpu(output_shape);
  GPUTensor<float> input_gpu(input_shape);
  GPUTensor<float> unpack_gpu(unpack_shape);

  Backend<GPUTensor, float>::NormalDistributionFunc(0, 1, &input);
  Backend<GPUTensor, float>::NormalDistributionFunc(0, 1, &filter);

  const int iter = 3;
  string kernel = "asm";
  for (int i = 0; i < iter; ++i) {
    Backend<GPUTensor, float>::Convolution2DForwardFunc(&input, &filter,
      2, 2, 1, 1, &unpack, &output, kernel);
  }
  kernel = "blas";
  for (int i = 0; i < iter; ++i) {
    Backend<GPUTensor, float>::Convolution2DForwardFunc(&input_gpu, &filter_gpu,
      2, 2, 1, 1, &unpack_gpu, &output_gpu, kernel);
  }
}

void forward4() {
  std::cout << "forward4" << std::endl;
  Shape input_shape(4);
  // batch_size
  input_shape[0] = 128;
  // input channel
  input_shape[1] = 384;
  // input height
  input_shape[2] = 13;
  // input width
  input_shape[3] = 13;

  Shape filter_shape(4);
  // output channel
  filter_shape[0] = 256;
  // input channel
  filter_shape[1] = 384;
  // filter height
  filter_shape[2] = 3;
  // filter width
  filter_shape[3] = 3;

  Shape output_shape(4);
  // batch_size
  output_shape[0] = 128;
  // output channel
  output_shape[1] = 256;
  // output height
  output_shape[2] = 13;
  // output width
  output_shape[3] = 13;

  Shape unpack_shape(2);
  unpack_shape[0] = 13 * 13;
  unpack_shape[1] = 3 * 3 * 384;

  GPUTensor<float> input(input_shape);
  GPUTensor<float> filter(filter_shape);
  GPUTensor<float> output(output_shape);
  GPUTensor<float> output_copy(output_shape);
  GPUTensor<float> unpack(unpack_shape);
  GPUTensor<float> filter_gpu(filter_shape);
  GPUTensor<float> output_gpu(output_shape);
  GPUTensor<float> input_gpu(input_shape);
  GPUTensor<float> unpack_gpu(unpack_shape);

  Backend<GPUTensor, float>::NormalDistributionFunc(0, 1, &input);
  Backend<GPUTensor, float>::NormalDistributionFunc(0, 1, &filter);

  const int iter = 3;
  string kernel = "asm";
  for (int i = 0; i < iter; ++i) {
    Backend<GPUTensor, float>::Convolution2DForwardFunc(&input, &filter,
      1, 1, 1, 1, &unpack, &output, kernel);
  }
  kernel = "blas";
  for (int i = 0; i < iter; ++i) {
    Backend<GPUTensor, float>::Convolution2DForwardFunc(&input_gpu, &filter_gpu,
      1, 1, 1, 1, &unpack_gpu, &output_gpu, kernel);
  }
}

void backward2() {
  std::cout << "backward2" << std::endl;
  Shape input_shape(4);
  // batch_size
  input_shape[0] = 128;
  // input channel
  input_shape[1] = 64;
  // input height
  input_shape[2] = 27;
  // input width
  input_shape[3] = 27;

  Shape filter_shape(4);
  // output channel
  filter_shape[0] = 192;
  // input channel
  filter_shape[1] = 64;
  // filter height
  filter_shape[2] = 5;
  // filter width
  filter_shape[3] = 5;

  Shape output_shape(4);
  // batch_size
  output_shape[0] = 128;
  // output channel
  output_shape[1] = 192;
  // output height
  output_shape[2] = 27;
  // output width
  output_shape[3] = 27;

  Shape unpack_shape(2);
  unpack_shape[0] = 27 * 27;
  unpack_shape[1] = 5 * 5 * 64;

  GPUTensor<float> input(input_shape);
  GPUTensor<float> input_copy(input_shape);
  GPUTensor<float> filter(filter_shape);
  GPUTensor<float> output(output_shape);
  GPUTensor<float> unpack(unpack_shape);
  GPUTensor<float> filter_gpu(filter_shape);
  GPUTensor<float> output_gpu(output_shape);
  GPUTensor<float> input_gpu(input_shape);
  GPUTensor<float> unpack_gpu(unpack_shape);

  Backend<GPUTensor, float>::NormalDistributionFunc(0, 1, &output);
  Backend<GPUTensor, float>::NormalDistributionFunc(0, 1, &filter);

  const int iter = 3;
  string kernel = "blas";
  for (int i = 0; i < iter; ++i) {
    Backend<GPUTensor, float>::Convolution2DBackwardFunc(&output_gpu, &filter_gpu,
      2, 2, 1, 1, &unpack_gpu, &input_gpu, kernel);
  }

  kernel = "blas";
  for (int i = 0; i < iter; ++i) {
    Backend<GPUTensor, float>::Convolution2DBackwardFunc(&output_gpu, &filter_gpu,
      2, 2, 1, 1, &unpack_gpu, &input_gpu, kernel);
  }
}

void backward_weight() {
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

  GPUTensor<float> input(input_shape);
  GPUTensor<float> filter(filter_shape);
  GPUTensor<float> filter_copy(filter_shape);
  GPUTensor<float> output(output_shape);
  GPUTensor<float> unpack(unpack_shape);
  GPUTensor<float> filter_gpu(filter_shape);
  GPUTensor<float> output_gpu(output_shape);
  GPUTensor<float> input_gpu(input_shape);
  GPUTensor<float> unpack_gpu(unpack_shape);

  Backend<GPUTensor, float>::NormalDistributionFunc(0, 1, &input);
  Backend<GPUTensor, float>::NormalDistributionFunc(0, 1, &output);

  Backend<GPUTensor, float>::Convolution2DUpdateFunc(&input, &output,
    0, 0, 1, 1, &unpack, &filter);
  Backend<GPUTensor, float>::Convolution2DUpdateFunc(&input_gpu, &output_gpu,
    0, 0, 1, 1, &unpack_gpu, &filter_gpu);
}

int main() {
  forward1();
  forward2();
  forward4();
  backward2();
  //backward_weight();
  return 0;
}
