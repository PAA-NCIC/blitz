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

  Shape gamma_shape(3);
  // input channel
  gamma_shape[0] = 1;
  // input height
  gamma_shape[1] = 3;
  // input width
  gamma_shape[2] = 3;
  CPUTensor<float> gamma(gamma_shape);
  gamma.Fill(1);
  CPUTensor<float> beta(gamma_shape);
  gamma.Fill(0);
  CPUTensor<float> input_var(gamma_shape);
  CPUTensor<float> input_hat(gamma_shape);

  CPUTensor<float> output(input_shape);

  Backend<CPUTensor, float>::BatchNormForwardFunc(&input, &gamma,
    &beta, 1e-6, &input_var, &input_hat, &output);

  /* 
   * expect output:
   */

  report(output);

  return 0;
}
