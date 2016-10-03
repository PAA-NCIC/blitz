#include <iostream>

#include "backend/backends.h"

using namespace blitz;

/*
 * 1 0 0 1
 * 2 3 1 0
 * 2 1 0 3
 * 5 2 -1 2
 */
void init_input(CPUTensor<float>& input) {
  input[0] = 1;
  input[1] = 0;
  input[2] = 0;
  input[3] = 1;
  input[4] = 2;
  input[5] = 3;
  input[6] = 1;
  input[7] = 0;
  input[8] = 2;
  input[9] = 1;
  input[10] = 0;
  input[11] = 3;
  input[12] = 5;
  input[13] = 2;
  input[14] = -1;
  input[15] = 2;
}

/*
 * expect:
 * 3 1
 * 5 3
 */
void report_pool_forward(CPUTensor<float>& output) {
  std::cout << "pool forward result: " << std::endl;
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
 * expect:
 * 5 3
 * 12 11
 */
void report_pool_max_index(CPUTensor<size_t>& max_index) {
  std::cout << "pool max_index result: " << std::endl;
  const Shape& max_index_shape = max_index.shape();
  for (size_t i = 0; i < max_index_shape[0]; ++i) {
    for (size_t j = 0; j < max_index_shape[1]; ++j) {
      for (size_t k = 0; k < max_index_shape[2]; ++k) {
        for (size_t v = 0; v < max_index_shape[3]; ++v) {
          const int index = i * max_index_shape[1] * max_index_shape[2] *
            max_index_shape[3] + j * max_index_shape[2] * max_index_shape[3] +
            k * max_index_shape[3] + v;
          std::cout << "index: " << " i " << i << " j " << j << " k " << k << " v " << v << std::endl;
          std::cout << "value: " << max_index[index] << std::endl;
        }
      }
    }
  }
}

/*
 * expect:
 * 0 0 0 1
 * 0 3 0 0
 * 0 0 0 3
 * 5 0 0 0
 */
void report_pool_backward(CPUTensor<float>& input) {
  std::cout << "pool backward result: " << std::endl;
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
  Shape input_shape(4);
  // batch_size
  input_shape[0] = 1;
  // input channel
  input_shape[1] = 1;
  // input height
  input_shape[2] = 4;
  // input width
  input_shape[3] = 4;
  CPUTensor<float> input(input_shape);
  init_input(input);

  Shape max_index_shape(4);
  // batch_size
  max_index_shape[0] = 1;
  // channel
  max_index_shape[1] = 1;
  // output height
  max_index_shape[2] = 2;
  // output width
  max_index_shape[3] = 2;
  CPUTensor<size_t> max_index(max_index_shape);

  Shape output_shape(4);
  // batch_size
  output_shape[0] = 1;
  // input channel
  output_shape[1] = 1;
  // input height
  output_shape[2] = 2;
  // input width
  output_shape[3] = 2;
  CPUTensor<float> output(output_shape);

  int filter_height = 2;
  int filter_width = 2;
  int stride_height = 2;
  int stride_width = 2;

  Backend<CPUTensor, float>::MaxPooling2DForwardFunc(&input, filter_height, filter_width,
    stride_width, stride_height, &max_index, &output);

  report_pool_forward(output);

  report_pool_max_index(max_index);

  Backend<CPUTensor, float>::MaxPooling2DBackwardFunc(&output, &max_index,
    filter_height, filter_width, stride_height, stride_width, &input);

  report_pool_backward(input);

  return 0;
}
