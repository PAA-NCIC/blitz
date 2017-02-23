#include <omp.h>
#include <blitz.h>

using namespace blitz;

// N C H W
Shape input_shape(4);
Shape input_shape_algorithm(4);
// K C R S
Shape filter_shape(4);
Shape filter_shape_algorithm(4);
// N K P Q
Shape output_shape(4);
Shape output_shape_algorithm(4);

void compare(float* algo1, float* algo2, size_t size, float precision = 1e-3) {
  for (size_t i = 0; i < size; ++i) {
    if (algo1[i] > algo2[i] + precision || algo1[i] < algo2[i] - precision) {
      LOG(INFO) << "Index: " << i << " algo1: " << algo1[i] << " algo2: " << algo2[i];
    }
  }
}

void set_shape_nchw(Shape& shape, size_t N, size_t C, size_t H, size_t W) {
  shape[0] = N;
  shape[1] = C;
  shape[2] = H;
  shape[3] = W;
  shape.set_data_layout(BLITZ_BUFFER_NCHW);
}

void set_shape_nhwc(Shape& shape, size_t N, size_t C, size_t H, size_t W) {
  shape[0] = N;
  shape[1] = H;
  shape[2] = W;
  shape[3] = C;
  shape.set_data_layout(BLITZ_BUFFER_NHWC);
}

void set_shape_kcrs(Shape& shape, size_t K, size_t C, size_t R, size_t S) {
  shape[0] = K;
  shape[1] = C;
  shape[2] = R;
  shape[3] = S;
  shape.set_data_layout(BLITZ_FILTER_KCRS);
}

void set_shape_rsck(Shape& shape, size_t K, size_t C, size_t R, size_t S) {
  shape[0] = R;
  shape[1] = S;
  shape[2] = C;
  shape[3] = K;
  shape.set_data_layout(BLITZ_FILTER_RSCK);
}

void convolution_forward(
  BLITZ_ALGORITHM algorithm,
  size_t pad_h, size_t pad_w,
  size_t str_h, size_t str_w,
  size_t iter) {
  // algorithm tensors
  CPUTensor<float> input_cpu_algorithm(input_shape_algorithm);
  CPUTensor<float> filter_cpu_algorithm(filter_shape_algorithm);
  CPUTensor<float> output_cpu_algorithm(output_shape_algorithm);
  ConvolutionContext<CPUTensor, float> context_algorithm(
    input_shape_algorithm, filter_shape_algorithm,
    pad_h, pad_w, str_h, str_w);
  context_algorithm.InitAlgorithmForUser(algorithm);
  // initial tensors
  CPUTensor<float> input_cpu(input_shape);
  CPUTensor<float> filter_cpu(filter_shape);
  CPUTensor<float> output_cpu(output_shape);
  ConvolutionContext<CPUTensor, float> context(
    input_shape, filter_shape,
    pad_h, pad_w, str_h, str_w);
  context.InitAlgorithmForUser(BLITZ_CONVOLUTION_NAIVE_DIRECT);
  // init values
  Backend<CPUTensor, float>::UniformDistributionFunc(&input_cpu, 0.0, 1.0);
  Backend<CPUTensor, float>::UniformDistributionFunc(&filter_cpu, 0.0, 1.0);
  Backend<CPUTensor, float>::TransformCopyFunc(&input_cpu, &input_cpu_algorithm);
  Backend<CPUTensor, float>::TransformCopyFunc(&filter_cpu, &filter_cpu_algorithm);
  // direct convolution 
  for (size_t i = 0; i < iter; ++i) {
    Backend<CPUTensor, float>::Convolution2DForwardFunc(
      &input_cpu,
      &filter_cpu,
      &output_cpu,
      &context);
  }
  // algorithm convolution
  for (size_t i = 0; i < iter; ++i) {
    Backend<CPUTensor, float>::Convolution2DForwardFunc(
      &input_cpu_algorithm,
      &filter_cpu_algorithm,
      &output_cpu_algorithm,
      &context_algorithm);
  }
  if (output_cpu_algorithm.data_layout() != input_cpu_algorithm.data_layout()) {
    CPUTensor<float> output_cpu_transform(output_shape);
    Backend<CPUTensor, float>::TransformCopyFunc(&output_cpu_algorithm, &output_cpu_transform);
    compare(output_cpu.data(), output_cpu_transform.data(), output_cpu.size(), 1e-2);
  } else {
    compare(output_cpu.data(), output_cpu_algorithm.data(), output_cpu.size(), 1e-2);
  }
}

void convolution_backward(
  BLITZ_ALGORITHM algorithm,
  size_t pad_h, size_t pad_w,
  size_t str_h, size_t str_w,
  size_t iter) {
  // algorithm tensors
  CPUTensor<float> input_cpu_algorithm(input_shape_algorithm);
  CPUTensor<float> filter_cpu_algorithm(filter_shape_algorithm);
  CPUTensor<float> output_cpu_algorithm(output_shape_algorithm);
  ConvolutionContext<CPUTensor, float> context_algorithm(
    input_shape_algorithm, filter_shape_algorithm,
    pad_h, pad_w, str_h, str_w);
  context_algorithm.InitAlgorithmForUser(algorithm);
  // initial tensors
  CPUTensor<float> input_cpu(input_shape);
  CPUTensor<float> filter_cpu(filter_shape);
  CPUTensor<float> output_cpu(output_shape);
  ConvolutionContext<CPUTensor, float> context(
    input_shape, filter_shape,
    pad_h, pad_w, str_h, str_w);
  context.InitAlgorithmForUser(BLITZ_CONVOLUTION_NAIVE_DIRECT);
  // init values
  Backend<CPUTensor, float>::UniformDistributionFunc(&filter_cpu, 0.0, 1.0);
  Backend<CPUTensor, float>::UniformDistributionFunc(&output_cpu, 0.0, 1.0);
  Backend<CPUTensor, float>::TransformCopyFunc(&filter_cpu, &filter_cpu_algorithm);
  Backend<CPUTensor, float>::TransformCopyFunc(&output_cpu, &output_cpu_algorithm);
  // cpu convolution 
  for (size_t i = 0; i < iter; ++i) {
    Backend<CPUTensor, float>::Convolution2DBackwardFunc(
      &output_cpu,
      &filter_cpu,
      &input_cpu,
      &context);
  }
  // different algorithm
  for (size_t i = 0; i < iter; ++i) {
    Backend<CPUTensor, float>::Convolution2DBackwardFunc(
      &output_cpu_algorithm,
      &filter_cpu_algorithm,
      &input_cpu_algorithm,
      &context_algorithm);
  }
  if (input_cpu_algorithm.data_layout() != output_cpu_algorithm.data_layout()) {
    CPUTensor<float> input_cpu_transform(input_shape);
    Backend<CPUTensor, float>::TransformCopyFunc(&input_cpu_algorithm, &input_cpu_transform);
    compare(input_cpu.data(), input_cpu_transform.data(), input_cpu.size(), 1e-2);
  } else {
    compare(input_cpu.data(), input_cpu_algorithm.data(), input_cpu.size(), 1e-2);
  }
}

void convolution_update(
  BLITZ_ALGORITHM algorithm,
  size_t pad_h, size_t pad_w,
  size_t str_h, size_t str_w,
  size_t iter) {
  // algorithm tensors
  CPUTensor<float> input_cpu_algorithm(input_shape_algorithm);
  CPUTensor<float> filter_cpu_algorithm(filter_shape_algorithm);
  CPUTensor<float> output_cpu_algorithm(output_shape_algorithm);
  ConvolutionContext<CPUTensor, float> context_algorithm(
    input_shape_algorithm, filter_shape_algorithm,
    pad_h, pad_w, str_h, str_w);
  context_algorithm.InitAlgorithmForUser(algorithm);
  // initial tensors
  CPUTensor<float> input_cpu(input_shape);
  CPUTensor<float> filter_cpu(filter_shape);
  CPUTensor<float> output_cpu(output_shape);
  ConvolutionContext<CPUTensor, float> context(
    input_shape, filter_shape,
    pad_h, pad_w, str_h, str_w);
  context.InitAlgorithmForUser(BLITZ_CONVOLUTION_NAIVE_DIRECT);
  // init values
  Backend<CPUTensor, float>::UniformDistributionFunc(&output_cpu, 0.0, 1.0);
  Backend<CPUTensor, float>::UniformDistributionFunc(&input_cpu, 0.0, 1.0);
  Backend<CPUTensor, float>::TransformCopyFunc(&output_cpu, &output_cpu_algorithm);
  Backend<CPUTensor, float>::TransformCopyFunc(&input_cpu, &input_cpu_algorithm);
  // cpu convolution 
  for (size_t i = 0; i < iter; ++i) {
    Backend<CPUTensor, float>::Convolution2DUpdateFunc(
      &input_cpu,
      &output_cpu,
      &filter_cpu,
      &context);
  }
  // different algorithm
  for (size_t i = 0; i < iter; ++i) {
    Backend<CPUTensor, float>::Convolution2DUpdateFunc(
      &input_cpu_algorithm,
      &output_cpu_algorithm,
      &filter_cpu_algorithm,
      &context_algorithm);
  }
  if (input_cpu_algorithm.data_layout() != output_cpu_algorithm.data_layout()) {
    CPUTensor<float> filter_cpu_transform(filter_shape);
    Backend<CPUTensor, float>::TransformCopyFunc(&filter_cpu_algorithm, &filter_cpu_transform);
    memcpy(filter_cpu.data(), filter_cpu_transform.data(), sizeof(float) * filter_cpu.size());
  } else {
    compare(filter_cpu.data(), filter_cpu_algorithm.data(), filter_cpu.size(), 1);
  }
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
  if (input_layout == "nchw") {
    set_shape_nchw(input_shape, N, C, H, W);
    set_shape_kcrs(filter_shape, K, C, R, S);
    set_shape_nchw(output_shape, N, K, P, Q);
    set_shape_nchw(input_shape_algorithm, N, C, H, W);
    set_shape_kcrs(filter_shape_algorithm, K, C, R, S);
  } else {
    set_shape_nhwc(input_shape, N, C, H, W);
    set_shape_rsck(filter_shape, K, C, R, S);
    set_shape_nhwc(output_shape, N, K, P, Q);
    set_shape_nhwc(input_shape_algorithm, N, C, H, W);
    set_shape_rsck(filter_shape_algorithm, K, C, R, S);
  }
  if (output_layout == "nchw") {
    set_shape_nchw(output_shape_algorithm, N, K, P, Q);
  } else {
    set_shape_nhwc(output_shape_algorithm, N, K, P, Q);
  }
  // run convolution
  if (phase == "forward") {
    convolution_forward(BlitzParseAlgorithm(kernel), pad_h, pad_w, str_h, str_w, iter);
  } else if (phase == "backward") {
    convolution_backward(BlitzParseAlgorithm(kernel), pad_h, pad_w, str_h, str_w, iter);
  } else if (phase == "update") {
    convolution_update(BlitzParseAlgorithm(kernel), pad_h, pad_w, str_h, str_w, iter);
  }
  return 0;
}
