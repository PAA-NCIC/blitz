#include <iostream>
#include "backends/backends.h"

using namespace blitz;

// N C H W
Shape input_shape(4);
// K C R S
Shape filter_shape(4);
// N K P Q
Shape output_shape(4);
// cpu workspace
Shape workspace_shape_cpu(1);

void output_convolution_transform(size_t N, size_t CHW, float* output) {
  for (size_t j = 0; j < CHW; ++j) {
    for (size_t i = 0; i < N; ++i) {
      std::cout << output[i * CHW + j] << " ";
    }
  }
  std::cout << std::endl;
}

void init_output(size_t N, size_t K, size_t P, size_t Q, float* output) {
  size_t value = 0;
  for (size_t i = 0; i < K * P * Q; ++i) {
    for (size_t j = 0; j < N; ++j) {
      output[j * K * P * Q + i] = value++;
    } 
  }
}

void init_filter(size_t K, size_t C, size_t R, size_t S, float* filter) {
  for (size_t i = 0; i < K * C * R * S; ++i) {
    filter[i] = i;
  }
}

void set_input_shape_nchw(size_t N, size_t C, size_t H, size_t W) {
  input_shape[0] = N;
  input_shape[1] = C;
  input_shape[2] = H;
  input_shape[3] = W;
}

void set_filter_shape_kcrs(size_t K, size_t C, size_t R, size_t S) {
  filter_shape[0] = K;
  filter_shape[1] = C;
  filter_shape[2] = R;
  filter_shape[3] = S;
}

void set_output_shape_nkpq(size_t N, size_t K, size_t P, size_t Q) {
  output_shape[0] = N;
  output_shape[1] = K;
  output_shape[2] = P;
  output_shape[3] = Q;
}

void convolution_forward(
  const string& kernel,
  size_t pad_h, size_t pad_w,
  size_t str_h, size_t str_w) {
}

void convolution_backward(
  const string& kernel,
  size_t pad_h, size_t pad_w,
  size_t str_h, size_t str_w) {
  // set up cpu
  CPUTensor<float> input_cpu(input_shape);
  CPUTensor<float> filter_cpu(filter_shape);
  CPUTensor<float> output_cpu(output_shape);
  CPUTensor<float> workspace_cpu(workspace_shape_cpu);
  // init values
  init_filter(filter_shape[0], filter_shape[1], filter_shape[2], filter_shape[3],
    filter_cpu.data());
  init_output(output_shape[0], output_shape[1], output_shape[2], output_shape[3],
    output_cpu.data());
  // cpu convolution 
  Backend<CPUTensor, float>::Convolution2DBackwardFunc(
    &output_cpu,
    &filter_cpu,
    &input_cpu,
    &workspace_cpu,
    pad_h, pad_w, 
    str_h, str_w,
    kernel);
  output_convolution_transform(
    input_shape[0],
    input_shape.size() / input_shape[0],
    input_cpu.data());
}

void convolution_update(
  const string& kernel,
  size_t pad_h, size_t pad_w,
  size_t str_h, size_t str_w) {
}

int main(int argc, char** argv) {
  const size_t NUM_ARGS = 15;
  // phase kernel N C H W R S K P Q pad_h pad_w str_h str_w
  if (argc != NUM_ARGS + 1) {
    std::cerr << "Not enough args!" << std::endl;
    exit(1);
  }
  // get args
  const std::string phase = std::string(argv[1]); 
  const std::string kernel = std::string(argv[2]); 
  const size_t N = atoi(argv[3]);
  const size_t C = atoi(argv[4]);
  const size_t H = atoi(argv[5]);
  const size_t W = atoi(argv[6]);
  const size_t R = atoi(argv[7]);
  const size_t S = atoi(argv[8]);
  const size_t K = atoi(argv[9]);
  const size_t P = atoi(argv[10]);
  const size_t Q = atoi(argv[11]);
  const size_t pad_h = atoi(argv[12]);
  const size_t pad_w = atoi(argv[13]);
  const size_t str_h = atoi(argv[14]);
  const size_t str_w = atoi(argv[15]);
  // set shapes
  set_input_shape_nchw(N, C, H, W);
  set_filter_shape_kcrs(K, C, R, S);
  set_output_shape_nkpq(N, K, P, Q);
  // set workspace shape
  workspace_shape_cpu[0] = C * H * W * P * Q;
  // run convolution
  if (phase == "forward") {
    convolution_forward(kernel, pad_h, pad_w, str_h, str_w);
  } else if (phase == "backward") {
    convolution_backward(kernel, pad_h, pad_w, str_h, str_w);
  } else if (phase == "update") {
    convolution_update(kernel, pad_h, pad_w, str_h, str_w);
  }
  return 0;
}
