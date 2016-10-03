#include <iostream>
#include "backend/backends.h"

using namespace blitz;
const int ITER = 5;

void forward_profile(
  const CPUTensor<float>& input, const CPUTensor<float>& weight,
  const int stride_height, const int stride_width,
  const int padding_height, const int padding_width,
  CPUTensor<float>& unpack, CPUTensor<float>& output) {
  // gemm parallel
  // std::cout << "gemm parallel:" << std::endl;
  // Backend<CPUTensor, float>::Convolution2DForwardFunc(
  //   &input, &weight, padding_height, padding_width,
  //   stride_height, stride_width, &unpack, &output);
  // output.Fill(0);
  // naive parallel
  // batch gemm parallel
  std::cout << "batch gemm parallel:" << std::endl;
  for (int i = 0; i < ITER; ++i) {
    time_point<system_clock> start, end;
    duration<double> time = duration<double>::zero();
    start = system_clock::now();
    Backend<CPUTensor, float>::Convolution2DForwardFunc(
        &input, &weight, padding_height, padding_width,
        stride_height, stride_width, &unpack, &output);
    end = system_clock::now();
    time = end - start;
    std::cout << time.count() << std::endl;
  }
  //output.Fill(0);
  
}

void mnist1()
{
  // MNIST first
  Shape input_shape(4);
  // batch_size
  input_shape[0] = 128;
  // input channel
  input_shape[1] = 1;
  // input height
  input_shape[2] = 28;
  // input width
  input_shape[3] = 28;
  CPUTensor<float> input(input_shape);
  Backend<CPUTensor, float>::UniformDistributionFunc(0.0f, 100.0f, &input);

  Shape filter_shape(4);
  // output channel
  filter_shape[0] = 16;
  // input channel
  filter_shape[1] = 1;
  // filter height
  filter_shape[2] = 5;
  // filter width
  filter_shape[3] = 5;
  CPUTensor<float> weight(filter_shape);
  Backend<CPUTensor, float>::NormalDistributionFunc(0.0f, 1.0f, &weight);

  Shape output_shape(4);
  // batch size
  output_shape[0] = 128;
  // output channel
  output_shape[1] = 16;
  // output height
  output_shape[2] = 24;
  // output width
  output_shape[3] = 24;
  CPUTensor<float> output(output_shape);

  Shape unpack_shape(1);
  unpack_shape[0] = BLITZ_NUM_THREADS *
    filter_shape[1] * filter_shape[2] * filter_shape[3] *
    output_shape[2] * output_shape[3];
  CPUTensor<float> unpack(unpack_shape);

  int stride_height = 1;
  int stride_width = 1;
  int padding_height = 0;
  int padding_width = 0;

  forward_profile(input, weight, stride_height, stride_width, padding_height,
    padding_width, unpack, output);
}

void mnist2()
{
  // MNIST first
  Shape input_shape(4);
  // batch_size
  input_shape[0] = 128;
  // input channel
  input_shape[1] = 16;
  // input height
  input_shape[2] = 8;
  // input width
  input_shape[3] = 8;
  CPUTensor<float> input(input_shape);
  Backend<CPUTensor, float>::UniformDistributionFunc(0.0f, 100.0f, &input);

  Shape filter_shape(4);
  // output channel
  filter_shape[0] = 32;
  // input channel
  filter_shape[1] = 16;
  // filter height
  filter_shape[2] = 5;
  // filter width
  filter_shape[3] = 5;
  CPUTensor<float> weight(filter_shape);
  Backend<CPUTensor, float>::NormalDistributionFunc(0.0f, 1.0f, &weight);

  Shape output_shape(4);
  // batch size
  output_shape[0] = 128;
  // output channel
  output_shape[1] = 32;
  // output height
  output_shape[2] = 8;
  // output width
  output_shape[3] = 8;
  CPUTensor<float> output(output_shape);

  Shape unpack_shape(1);
  unpack_shape[0] = BLITZ_NUM_THREADS *
    filter_shape[1] * filter_shape[2] * filter_shape[3] *
    output_shape[2] * output_shape[3];
  CPUTensor<float> unpack(unpack_shape);

  int stride_height = 1;
  int stride_width = 1;
  int padding_height = 0;
  int padding_width = 0;

  forward_profile(input, weight, stride_height, stride_width, padding_height,
    padding_width, unpack, output);
}

void cifar1()
{
  // MNIST first
  Shape input_shape(4);
  // batch_size
  input_shape[0] = 128;
  // input channel
  input_shape[1] = 3;
  // input height
  input_shape[2] = 32;
  // input width
  input_shape[3] = 32;
  CPUTensor<float> input(input_shape);
  Backend<CPUTensor, float>::UniformDistributionFunc(0.0f, 100.0f, &input);

  Shape filter_shape(4);
  // output channel
  filter_shape[0] = 16;
  // input channel
  filter_shape[1] = 3;
  // filter height
  filter_shape[2] = 5;
  // filter width
  filter_shape[3] = 5;
  CPUTensor<float> weight(filter_shape);
  Backend<CPUTensor, float>::NormalDistributionFunc(0.0f, 1.0f, &weight);

  Shape output_shape(4);
  // batch size
  output_shape[0] = 128;
  // output channel
  output_shape[1] = 16;
  // output height
  output_shape[2] = 28;
  // output width
  output_shape[3] = 28;
  CPUTensor<float> output(output_shape);

  Shape unpack_shape(1);
  unpack_shape[0] = BLITZ_NUM_THREADS *
    filter_shape[1] * filter_shape[2] * filter_shape[3] *
    output_shape[2] * output_shape[3];
  CPUTensor<float> unpack(unpack_shape);

  int stride_height = 1;
  int stride_width = 1;
  int padding_height = 0;
  int padding_width = 0;

  forward_profile(input, weight, stride_height, stride_width, padding_height,
    padding_width, unpack, output);
}

void alexnet1()
{
  // Alexnet first
  Shape input_shape(4);
  // batch_size
  input_shape[0] = 128;
  // input channel
  input_shape[1] = 3;
  // input height
  input_shape[2] = 227;
  // input width
  input_shape[3] = 227;
  CPUTensor<float> input(input_shape);
  Backend<CPUTensor, float>::UniformDistributionFunc(0.0f, 100.0f, &input);

  Shape filter_shape(4);
  // output channel
  filter_shape[0] = 64;
  // input channel
  filter_shape[1] = 3;
  // filter height
  filter_shape[2] = 11;
  // filter width
  filter_shape[3] = 11;
  CPUTensor<float> weight(filter_shape);
  Backend<CPUTensor, float>::NormalDistributionFunc(0.0f, 1.0f, &weight);

  Shape output_shape(4);
  // batch size
  output_shape[0] = 128;
  // output channel
  output_shape[1] = 64;
  // output height
  output_shape[2] = 55;
  // output width
  output_shape[3] = 55;
  CPUTensor<float> output(output_shape);

  Shape unpack_shape(1);
  unpack_shape[0] = BLITZ_NUM_THREADS *
    filter_shape[1] * filter_shape[2] * filter_shape[3] *
    output_shape[2] * output_shape[3];
  CPUTensor<float> unpack(unpack_shape);

  int stride_height = 4;
  int stride_width = 4;
  int padding_height = 3;
  int padding_width = 3;

  forward_profile(input, weight, stride_height, stride_width, padding_height,
    padding_width, unpack, output);
}

void alexnet2()
{
  // Alexnet first
  Shape input_shape(4);
  // batch_size
  input_shape[0] = 128;
  // input channel
  input_shape[1] = 64;
  // input height
  input_shape[2] = 27;
  // input width
  input_shape[3] = 27;
  CPUTensor<float> input(input_shape);
  Backend<CPUTensor, float>::UniformDistributionFunc(0.0f, 100.0f, &input);

  Shape filter_shape(4);
  // output channel
  filter_shape[0] = 192;
  // input channel
  filter_shape[1] = 64;
  // filter height
  filter_shape[2] = 5;
  // filter width
  filter_shape[3] = 5;
  CPUTensor<float> weight(filter_shape);
  Backend<CPUTensor, float>::NormalDistributionFunc(0.0f, 1.0f, &weight);

  Shape output_shape(4);
  // batch size
  output_shape[0] = 128;
  // output channel
  output_shape[1] = 192;
  // output height
  output_shape[2] = 27;
  // output width
  output_shape[3] = 27;
  CPUTensor<float> output(output_shape);

  Shape unpack_shape(1);
  unpack_shape[0] = BLITZ_NUM_THREADS *
    filter_shape[1] * filter_shape[2] * filter_shape[3] *
    output_shape[2] * output_shape[3];
  CPUTensor<float> unpack(unpack_shape);

  int stride_height = 1;
  int stride_width = 1;
  int padding_height = 2;
  int padding_width = 2;

  forward_profile(input, weight, stride_height, stride_width, padding_height,
    padding_width, unpack, output);
}

void alexnet3()
{
  // Alexnet first
  Shape input_shape(4);
  // batch_size
  input_shape[0] = 128;
  // input channel
  input_shape[1] = 192;
  // input height
  input_shape[2] = 13;
  // input width
  input_shape[3] = 13;
  CPUTensor<float> input(input_shape);
  Backend<CPUTensor, float>::UniformDistributionFunc(0.0f, 100.0f, &input);

  Shape filter_shape(4);
  // output channel
  filter_shape[0] = 192;
  // input channel
  filter_shape[1] = 384;
  // filter height
  filter_shape[2] = 3;
  // filter width
  filter_shape[3] = 3;
  CPUTensor<float> weight(filter_shape);
  Backend<CPUTensor, float>::NormalDistributionFunc(0.0f, 1.0f, &weight);

  Shape output_shape(4);
  // batch size
  output_shape[0] = 128;
  // output channel
  output_shape[1] = 384;
  // output height
  output_shape[2] = 13;
  // output width
  output_shape[3] = 13;
  CPUTensor<float> output(output_shape);

  Shape unpack_shape(1);
  unpack_shape[0] = BLITZ_NUM_THREADS *
    filter_shape[1] * filter_shape[2] * filter_shape[3] *
    output_shape[2] * output_shape[3];
  CPUTensor<float> unpack(unpack_shape);

  int stride_height = 1;
  int stride_width = 1;
  int padding_height = 1;
  int padding_width = 1;

  forward_profile(input, weight, stride_height, stride_width, padding_height,
    padding_width, unpack, output);
}

void alexnet4()
{
  // Alexnet first
  Shape input_shape(4);
  // batch_size
  input_shape[0] = 128;
  // input channel
  input_shape[1] = 384;
  // input height
  input_shape[2] = 13;
  // input width
  input_shape[3] = 13;
  CPUTensor<float> input(input_shape);
  Backend<CPUTensor, float>::UniformDistributionFunc(0.0f, 100.0f, &input);

  Shape filter_shape(4);
  // output channel
  filter_shape[0] = 256;
  // input channel
  filter_shape[1] = 384;
  // filter height
  filter_shape[2] = 3;
  // filter width
  filter_shape[3] = 3;
  CPUTensor<float> weight(filter_shape);
  Backend<CPUTensor, float>::NormalDistributionFunc(0.0f, 1.0f, &weight);

  Shape output_shape(4);
  // batch size
  output_shape[0] = 128;
  // output channel
  output_shape[1] = 256;
  // output height
  output_shape[2] = 13;
  // output width
  output_shape[3] = 13;
  CPUTensor<float> output(output_shape);

  Shape unpack_shape(1);
  unpack_shape[0] = BLITZ_NUM_THREADS *
    filter_shape[1] * filter_shape[2] * filter_shape[3] *
    output_shape[2] * output_shape[3];
  CPUTensor<float> unpack(unpack_shape);

  int stride_height = 1;
  int stride_width = 1;
  int padding_height = 1;
  int padding_width = 1;

  forward_profile(input, weight, stride_height, stride_width, padding_height,
    padding_width, unpack, output);
}

void alexnet5()
{
  // Alexnet first
  Shape input_shape(4);
  // batch_size
  input_shape[0] = 128;
  // input channel
  input_shape[1] = 256;
  // input height
  input_shape[2] = 13;
  // input width
  input_shape[3] = 13;
  CPUTensor<float> input(input_shape);
  Backend<CPUTensor, float>::UniformDistributionFunc(0.0f, 100.0f, &input);

  Shape filter_shape(4);
  // output channel
  filter_shape[0] = 256;
  // input channel
  filter_shape[1] = 256;
  // filter height
  filter_shape[2] = 3;
  // filter width
  filter_shape[3] = 3;
  CPUTensor<float> weight(filter_shape);
  Backend<CPUTensor, float>::NormalDistributionFunc(0.0f, 1.0f, &weight);

  Shape output_shape(4);
  // batch size
  output_shape[0] = 128;
  // output channel
  output_shape[1] = 256;
  // output height
  output_shape[2] = 13;
  // output width
  output_shape[3] = 13;
  CPUTensor<float> output(output_shape);

  Shape unpack_shape(1);
  unpack_shape[0] = BLITZ_NUM_THREADS *
    filter_shape[1] * filter_shape[2] * filter_shape[3] *
    output_shape[2] * output_shape[3];
  CPUTensor<float> unpack(unpack_shape);

  int stride_height = 1;
  int stride_width = 1;
  int padding_height = 1;
  int padding_width = 1;

  forward_profile(input, weight, stride_height, stride_width, padding_height,
    padding_width, unpack, output);
}

int main() {

  //mnist1();
  //mnist2();
  //cifar1();
  alexnet1();
  alexnet2();
  alexnet3();
  alexnet4();
  alexnet5();

  return 0;
}
