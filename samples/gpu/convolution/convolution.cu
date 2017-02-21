#include <cuda.h>
#include <cuda_runtime_api.h>
#include <blitz.h>

using namespace blitz;

// N C H W
Shape input_shape(4);
// K C R S
Shape filter_shape(4);
// N K P Q
Shape output_shape(4);

void compare(float* algo1, float* algo2, size_t size, float precision = 1e-3) {
  for (size_t i = 0; i < size; ++i) {
    if (algo1[i] > algo2[i] + precision || algo1[i] < algo2[i] - precision) {
      LOG(FATAL) << "Index: " << i << " algo1: " << algo1[i] << " algo2: " << algo2[i];
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
  // gpu tensors
  GPUTensor<float> input_gpu(input_shape);
  GPUTensor<float> filter_gpu(filter_shape);
  GPUTensor<float> output_gpu(output_shape);
  ConvolutionContext<GPUTensor, float> context_gpu(
    input_shape, filter_shape,
    pad_h, pad_w, str_h, str_w);
  context_gpu.InitAlgorithmForUser(algorithm);
  // initial tensors
  CPUTensor<float> input_cpu(input_shape);
  CPUTensor<float> filter_cpu(filter_shape);
  CPUTensor<float> output_cpu(output_shape);
  ConvolutionContext<CPUTensor, float> context(
    input_shape, filter_shape,
    pad_h, pad_w, str_h, str_w);
  context.InitAlgorithmForUser(BLITZ_CONVOLUTION_NAIVE_DIRECT);
  // init values
  CPUTensor<float> output_cpu_copy(output_shape);
  Backend<CPUTensor, float>::UniformDistributionFunc(&input_cpu, 0.0, 1.0);
  Backend<CPUTensor, float>::UniformDistributionFunc(&filter_cpu, 0.0, 1.0);
  cudaMemcpy(input_gpu.data(), input_cpu.data(), sizeof(float) * input_gpu.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(filter_gpu.data(), filter_cpu.data(), sizeof(float) * filter_gpu.size(), cudaMemcpyHostToDevice);
  // direct convolution 
  for (size_t i = 0; i < iter; ++i) {
    Backend<CPUTensor, float>::Convolution2DForwardFunc(
      &input_cpu,
      &filter_cpu,
      &output_cpu,
      &context);
  }
  // gpu convolution
  for (size_t i = 0; i < iter; ++i) {
    Backend<GPUTensor, float>::Convolution2DForwardFunc(
      &input_gpu,
      &filter_gpu,
      &output_gpu,
      &context_gpu);
  }
  cudaMemcpy(output_cpu_copy.data(), output_gpu.data(), sizeof(float) * output_gpu.size(), cudaMemcpyDeviceToHost);
  compare(output_cpu.data(), output_cpu_copy.data(), output_cpu.size(), 1e-2);
}

void convolution_backward(
  BLITZ_ALGORITHM algorithm,
  size_t pad_h, size_t pad_w,
  size_t str_h, size_t str_w,
  size_t iter) {
  // algorithm tensors
  GPUTensor<float> input_gpu(input_shape);
  GPUTensor<float> filter_gpu(filter_shape);
  GPUTensor<float> output_gpu(output_shape);
  ConvolutionContext<GPUTensor, float> context_gpu(
    input_shape, filter_shape,
    pad_h, pad_w, str_h, str_w);
  context_gpu.InitAlgorithmForUser(algorithm);
  // initial tensors
  CPUTensor<float> input_cpu(input_shape);
  CPUTensor<float> filter_cpu(filter_shape);
  CPUTensor<float> output_cpu(output_shape);
  ConvolutionContext<CPUTensor, float> context(
    input_shape, filter_shape,
    pad_h, pad_w, str_h, str_w);
  context.InitAlgorithmForUser(BLITZ_CONVOLUTION_NAIVE_DIRECT);
  // init values
  CPUTensor<float> input_cpu_copy(input_shape);
  Backend<CPUTensor, float>::UniformDistributionFunc(&filter_cpu, 0.0, 1.0);
  Backend<CPUTensor, float>::UniformDistributionFunc(&output_cpu, 0.0, 1.0);
  cudaMemcpy(filter_gpu.data(), filter_cpu.data(), sizeof(float) * filter_gpu.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(output_gpu.data(), output_cpu.data(), sizeof(float) * output_gpu.size(), cudaMemcpyHostToDevice);
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
    Backend<GPUTensor, float>::Convolution2DBackwardFunc(
      &output_gpu,
      &filter_gpu,
      &input_gpu,
      &context_gpu);
  }
  cudaMemcpy(input_cpu_copy.data(), input_gpu.data(), sizeof(float) * input_gpu.size(), cudaMemcpyDeviceToHost);
  compare(input_cpu.data(), input_cpu_copy.data(), input_cpu.size(), 1e-2);
}

void convolution_update(
  BLITZ_ALGORITHM algorithm,
  size_t pad_h, size_t pad_w,
  size_t str_h, size_t str_w,
  size_t iter) {
  // algorithm tensors
  GPUTensor<float> input_gpu(input_shape);
  GPUTensor<float> filter_gpu(filter_shape);
  GPUTensor<float> output_gpu(output_shape);
  ConvolutionContext<GPUTensor, float> context_gpu(
    input_shape, filter_shape,
    pad_h, pad_w, str_h, str_w);
  context_gpu.InitAlgorithmForUser(algorithm);
  // initial tensors
  CPUTensor<float> input_cpu(input_shape);
  CPUTensor<float> filter_cpu(filter_shape);
  CPUTensor<float> output_cpu(output_shape);
  ConvolutionContext<CPUTensor, float> context(
    input_shape, filter_shape,
    pad_h, pad_w, str_h, str_w);
  context.InitAlgorithmForUser(BLITZ_CONVOLUTION_NAIVE_DIRECT);
  // init values
  CPUTensor<float> filter_cpu_copy(filter_shape);
  Backend<CPUTensor, float>::UniformDistributionFunc(&output_cpu, 0.0, 1.0);
  Backend<CPUTensor, float>::UniformDistributionFunc(&input_cpu, 0.0, 1.0);
  cudaMemcpy(output_gpu.data(), output_cpu.data(), sizeof(float) * output_gpu.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(input_gpu.data(), input_cpu.data(), sizeof(float) * input_gpu.size(), cudaMemcpyHostToDevice);
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
    Backend<GPUTensor, float>::Convolution2DUpdateFunc(
      &input_gpu,
      &output_gpu,
      &filter_gpu,
      &context_gpu);
  }
  cudaMemcpy(filter_cpu_copy.data(), filter_gpu.data(), sizeof(float) * filter_gpu.size(), cudaMemcpyDeviceToHost);
  compare(filter_cpu.data(), filter_cpu_copy.data(), filter_cpu.size(), 10);
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
  } else {
    set_shape_nhwc(input_shape, N, C, H, W);
    set_shape_rsck(filter_shape, K, C, R, S);
    set_shape_nhwc(output_shape, N, K, P, Q);
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
