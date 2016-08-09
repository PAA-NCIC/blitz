#include <iostream>
#include "backend/backends.h"

using namespace blitz;

/*
 * batch1:
 * 0 1 0
 * 1 1 0
 * 0 1 0
 * batch2:
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
  input[9] = 0;
  input[10] = 1;
  input[11] = 0;
  input[12] = 1;
  input[13] = 1;
  input[14] = 0;
  input[15] = 0;
  input[16] = 1;
  input[17] = 0;
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

void report(CPUTensor<float>& output) {
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

int main() {
  Shape input_shape(4);
  // batch_size
  input_shape[0] = 2;
  // input channel
  input_shape[1] = 1;
  // input height
  input_shape[2] = 3;
  // input width
  input_shape[3] = 3;
  CPUTensor<float> input(input_shape);
  init_input(input);

  Shape filter_shape(4);
  // output channel
  filter_shape[0] = 2;
  // input channel
  filter_shape[1] = 2;
  // filter height
  filter_shape[2] = 2;
  // filter width
  filter_shape[3] = 2;
  CPUTensor<float> weight(filter_shape);
  init_filter(weight);

  Shape unpack_shape(2);
  unpack_shape[0] = 1 * 2 * 2;
  unpack_shape[1] = 2 * 2;
  CPUTensor<float> unpack(unpack_shape);
  unpack.Fill(0);

  Shape output_shape(4);
  // batch size
  output_shape[0] = 2;
  // output channel
  output_shape[1] = 2;
  // output height
  output_shape[2] = 2;
  // output width
  output_shape[3] = 2;
  CPUTensor<float> output(output_shape);

  Shape bias_shape(1);
  bias_shape[0] = output_shape[0] * output_shape[1] *
    output_shape[2];
  CPUTensor<float> bias(bias_shape);
  bias.Fill(1);

  CPUTensor<float> mask(output_shape);

  int stride_height = 1;
  int stride_width = 1;
  int padding_height = 0;
  int padding_width = 0;

  // forward
  // pack
  Backend<CPUTensor, float>::Convolution2DForwardFunc(
    &input, &weight, padding_height, padding_width,
    stride_height, stride_width, &unpack, &output);

  std::cout << "conv forward result: " << std::endl;
  /*
   * expect output:
   * 1 1 0 1 0 1 1 1
   * 1 1 0 1 0 1 1 1
   */
  report(output);

  // bias
  std::cout << "bias forward result: " << std::endl;
  Backend<CPUTensor, float>::BiasForwardFunc(&output, &bias, &output);
  /*
   * expect output:
   * 2 2 1 2 1 2 2 2
   * 2 2 1 2 1 2 2 2
   */
  report(output);

  // dropout
  Backend<CPUTensor, float>::MakeBinaryMaskFunc(0.0, 1.0, 0.5, &mask);
  Backend<CPUTensor, float>::MultiplyFunc(&output, &mask, &output);
  std::cout << "dropout forward result: " << std::endl;
  report(output);

  // softmax
  std::cout << "softmax forward result: " << std::endl;
  Backend<CPUTensor, float>::SoftmaxApplyFunc(&output, &output);
  /*
   * expect output:
   * 0.148 0.148 0.05 0.148 0.05 0.148 0.148 0.148
   * 0.148 0.148 0.05 0.148 0.05 0.148 0.148 0.148
   */
  report(output);

  CPUTensor<float> target(output.shape());
  /*
   * 1 0 0 0 0 0 0 0
   * 1 0 0 0 0 0 0 0
   */
  target.Fill(0);
  target[0] = 1;
  target[8] = 1;
  // cross entropy multi
  std::cout << "cross entropy forward result: " << std::endl;
  float loss = Backend<CPUTensor, float>::CrossEntropyMultiApplyFunc(&output, &target);
  /*
   * expect loss:
   * 1.90743
   */
  std::cout << loss << std::endl;

  // backward
  // cross entorpy multi
  std::cout << "cross entropy backward result: " << std::endl;
  Backend<CPUTensor, float>::CrossEntropyMultiDerivativeFunc(&output, &target, &output);
  /*
   * expect output:
   * -0.85 0.148 0.05 0.148 0.05 0.148 0.148 0.148
   * -0.85 0.148 0.05 0.148 0.05 0.148 0.148 0.148
   */
  report(output);

  Shape error_shape(4);
  error_shape[0] = 1;
  error_shape[1] = 1;
  error_shape[2] = 1;
  error_shape[3] = 8;
  CPUTensor<float> error(error_shape);
  // bias
  std::cout << "bias backward update result: " << std::endl;
  Backend<CPUTensor, float>::BiasBackwardUpdateFunc(&output, &error);
  /*
   * expect output:
   * -1.7 0.296 0.1 0.296 0.1 0.296 0.296 0.296
   */
  report(error);

  return 0;
}
