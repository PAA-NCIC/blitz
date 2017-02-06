#include <iostream>
#include <blitz.h>

using namespace blitz;
// M N
Shape input_shape(2);
// N M
Shape output_shape(2);

void output_matrix(size_t dim_left, size_t dim_right, float* matrix) {
  for (size_t i = 0; i < dim_left; ++i) {
    for (size_t j = 0; j < dim_right; ++j) {
      std::cout << matrix[i * dim_right + j] << " ";
    }
    std::cout << std::endl;
  }
}

void transpose(size_t m, size_t n) {
  // set up cpu
  CPUTensor<float> input_cpu(input_shape);
  CPUTensor<float> output_cpu(output_shape);
  // init values
  std::cout << "Init:" << std::endl;
  Backend<CPUTensor, float>::UniformDistributionFunc(&input_cpu, 0.0, 1.0);
  output_matrix(m, n, input_cpu.data());
  std::cout << "Transposed:" << std::endl;
  // transpose
  Backend<CPUTensor, float>::Transpose2DFunc(&input_cpu, &output_cpu);
  output_matrix(n, m, output_cpu.data());
}

int main(int argc, char** argv) {
  const size_t NUM_ARGS = 2;
  // M N
  if (argc != NUM_ARGS + 1) {
    std::cerr << "Not matchable args!" << std::endl;
    exit(1);
  }
  const size_t M = atoi(argv[1]);
  const size_t N = atoi(argv[2]);
  // set shapes
  input_shape[0] = M;
  input_shape[1] = N;
  output_shape[0] = N;
  output_shape[1] = M;
  // run
  transpose(M, N);
  return 0;
}
