#include <omp.h>
#include "backends/backends.h"
#include "utils/blitz_algorithm_function.h"
#include "utils/blitz_shape_function.h"

using namespace blitz;

// N C H W
Shape input_shape(4);
// K C R S
Shape filter_shape(4);
// N K P Q
Shape output_shape(4);
// cpu workspace
Shape workspace_shape_cpu(1);

void compare(float* algo1, float* algo2, size_t size, float precision = 1e-3) {
  for (size_t i = 0; i < size; ++i) {
    if (algo1[i] > algo2[i] + precision || algo1[i] < algo2[i] - precision) {
      LOG(FATAL) << "Index: " << i << " algo1: " << algo1[i] << " algo2: " << algo2[i];
    }
  }
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
  input_shape.set_data_layout(BLITZ_BUFFER_NCHW);
}

void set_input_shape_nhwc(size_t N, size_t C, size_t H, size_t W) {
  input_shape[0] = N;
  input_shape[1] = H;
  input_shape[2] = W;
  input_shape[3] = C;
  input_shape.set_data_layout(BLITZ_BUFFER_NHWC);
}

void set_filter_shape_kcrs(size_t K, size_t C, size_t R, size_t S) {
  filter_shape[0] = K;
  filter_shape[1] = C;
  filter_shape[2] = R;
  filter_shape[3] = S;
  filter_shape.set_data_layout(BLITZ_FILTER_KCRS);
}

void set_output_shape_nkpq(size_t N, size_t K, size_t P, size_t Q) {
  output_shape[0] = N;
  output_shape[1] = K;
  output_shape[2] = P;
  output_shape[3] = Q;
  output_shape.set_data_layout(BLITZ_BUFFER_NCHW);
}

void set_output_shape_npqk(size_t N, size_t K, size_t P, size_t Q) {
  output_shape[0] = N;
  output_shape[1] = P;
  output_shape[2] = Q;
  output_shape[3] = K;
  output_shape.set_data_layout(BLITZ_BUFFER_NHWC);
}

void convolution_forward(
  BLITZ_ALGORITHM algorithm,
  size_t pad_h, size_t pad_w,
  size_t str_h, size_t str_w,
  size_t iter) {
  // set up cpu
  CPUTensor<float> input_cpu(input_shape);
  CPUTensor<float> filter_cpu(filter_shape);
  CPUTensor<float> output_cpu(output_shape);
  CPUTensor<float> output_cpu_algorithm(output_shape);
  CPUTensor<float> workspace_cpu(workspace_shape_cpu);
  size_t workspace_size = workspace_shape_cpu.size() * omp_get_max_threads();
  Shape workspace_shape_algorithm(1);
  workspace_shape_algorithm[0] = workspace_size;
  CPUTensor<float> workspace_cpu_algorithm(workspace_shape_algorithm);
  // init values
  Backend<CPUTensor, float>::UniformDistributionFunc(&filter_cpu, 0.0, 1.0);
  Backend<CPUTensor, float>::UniformDistributionFunc(&input_cpu, 0.0, 1.0);
  // cpu convolution 
  Backend<CPUTensor, float>::Convolution2DForwardFunc(
    &input_cpu,
    &filter_cpu,
    &output_cpu,
    &workspace_cpu,
    pad_h, pad_w, 
    str_h, str_w);
  // different algorithm
  for (size_t i = 0; i < iter; ++i) {
    Backend<CPUTensor, float>::Convolution2DForwardFunc(
      &input_cpu,
      &filter_cpu,
      &output_cpu_algorithm,
      &workspace_cpu_algorithm,
      pad_h, pad_w, 
      str_h, str_w,
      algorithm);
  }
  compare(output_cpu.data(), output_cpu_algorithm.data(), output_cpu.size());
}

void convolution_backward(
  BLITZ_ALGORITHM algorithm,
  size_t pad_h, size_t pad_w,
  size_t str_h, size_t str_w,
  size_t iter) {
  // set up cpu
  CPUTensor<float> input_cpu(input_shape);
  CPUTensor<float> input_cpu_algorithm(input_shape);
  CPUTensor<float> filter_cpu(filter_shape);
  CPUTensor<float> output_cpu(output_shape);
  CPUTensor<float> workspace_cpu(workspace_shape_cpu);
  size_t workspace_size = workspace_shape_cpu.size() * omp_get_max_threads();
  Shape workspace_shape_algorithm(1);
  workspace_shape_algorithm[0] = workspace_size;
  CPUTensor<float> workspace_cpu_algorithm(workspace_shape_algorithm);
  // init values
  Backend<CPUTensor, float>::UniformDistributionFunc(&filter_cpu, 0.0, 1.0);
  Backend<CPUTensor, float>::UniformDistributionFunc(&output_cpu, 0.0, 1.0);
  // cpu convolution 
  Backend<CPUTensor, float>::Convolution2DBackwardFunc(
    &output_cpu,
    &filter_cpu,
    &input_cpu,
    &workspace_cpu,
    pad_h, pad_w, 
    str_h, str_w);
  // different algorithm
  for (size_t i = 0; i < iter; ++i) {
    Backend<CPUTensor, float>::Convolution2DBackwardFunc(
      &output_cpu,
      &filter_cpu,
      &input_cpu_algorithm,
      &workspace_cpu_algorithm,
      pad_h, pad_w, 
      str_h, str_w,
      algorithm);
  }
  compare(input_cpu.data(), input_cpu_algorithm.data(), input_cpu.size());
}

void convolution_update(
  BLITZ_ALGORITHM algorithm,
  size_t pad_h, size_t pad_w,
  size_t str_h, size_t str_w,
  size_t iter) {
  // set up cpu
  CPUTensor<float> input_cpu(input_shape);
  CPUTensor<float> filter_cpu(filter_shape);
  CPUTensor<float> filter_cpu_algorithm(filter_shape);
  CPUTensor<float> output_cpu(output_shape);
  CPUTensor<float> workspace_cpu(workspace_shape_cpu);
  size_t workspace_size = workspace_shape_cpu.size() * omp_get_max_threads();
  Shape workspace_shape_algorithm(1);
  workspace_shape_algorithm[0] = workspace_size;
  CPUTensor<float> workspace_cpu_algorithm(workspace_shape_algorithm);
  // init values
  Backend<CPUTensor, float>::UniformDistributionFunc(&output_cpu, 0.0, 1.0);
  Backend<CPUTensor, float>::UniformDistributionFunc(&input_cpu, 0.0, 1.0);
  // cpu convolution 
  Backend<CPUTensor, float>::Convolution2DUpdateFunc(
    &input_cpu,
    &output_cpu,
    &filter_cpu,
    &workspace_cpu,
    pad_h, pad_w, 
    str_h, str_w);
  // different algorithm
  for (size_t i = 0; i < iter; ++i) {
    Backend<CPUTensor, float>::Convolution2DUpdateFunc(
      &input_cpu,
      &output_cpu,
      &filter_cpu_algorithm,
      &workspace_cpu_algorithm,
      pad_h, pad_w, 
      str_h, str_w,
      algorithm);
  }
  compare(filter_cpu.data(), filter_cpu_algorithm.data(), filter_cpu.size(), 1);
}

int main(int argc, char** argv) {
  const size_t NUM_ARGS = 18;
  // phase kernel N C H W R S K P Q pad_h pad_w str_h str_w iter
  if (argc != NUM_ARGS + 1) {
    LOG(FATAL) << "Not matchable args!";
  }
  FLAGS_logtostderr = true;
  google::InitGoogleLogging(argv[0]);
  // get args
  const std::string phase = std::string(argv[1]); 
  const std::string kernel = std::string(argv[2]); 
  const std::string input_layout = std::string(argv[3]); 
  const std::string output_layout = std::string(argv[4]); 
  const size_t N = atoi(argv[5]);
  const size_t C = atoi(argv[6]);
  const size_t H = atoi(argv[7]);
  const size_t W = atoi(argv[8]);
  const size_t R = atoi(argv[9]);
  const size_t S = atoi(argv[10]);
  const size_t K = atoi(argv[11]);
  const size_t P = atoi(argv[12]);
  const size_t Q = atoi(argv[13]);
  const size_t pad_h = atoi(argv[14]);
  const size_t pad_w = atoi(argv[15]);
  const size_t str_h = atoi(argv[16]);
  const size_t str_w = atoi(argv[17]);
  const size_t iter = atoi(argv[18]);
  // set shapes
  if (input_layout == "nhwc") {
    set_input_shape_nhwc(N, C, H, W);
  } else {
    set_input_shape_nchw(N, C, H, W);
  }
  if (output_layout == "nhwc") {
    set_output_shape_nkpq(N, K, P, Q);
  } else {
    set_output_shape_npqk(N, K, P, Q);
  }
  set_filter_shape_kcrs(K, C, R, S);
  // set workspace shape
  // run convolution
  if (phase == "forward") {
    workspace_shape_cpu[0] = C * R * S * P * Q;
    convolution_forward(BlitzParseAlgorithm(kernel), pad_h, pad_w, str_h, str_w, iter);
  } else if (phase == "backward") {
    workspace_shape_cpu[0] = C * R * S * P * Q;
    convolution_backward(BlitzParseAlgorithm(kernel), pad_h, pad_w, str_h, str_w, iter);
  } else if (phase == "update") {
    workspace_shape_cpu[0] = C * R * S * P * Q + K * C * R * S;
    convolution_update(BlitzParseAlgorithm(kernel), pad_h, pad_w, str_h, str_w, iter);
  }
  return 0;
}
