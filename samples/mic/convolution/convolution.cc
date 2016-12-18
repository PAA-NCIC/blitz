#include <iostream>
#include "backends/backends.h"
#include "utils/blitz_shape_function.h"
#include "utils/blitz_algorithm_function.h"

using namespace blitz;

// N C H W
Shape input_shape(4);
// K C R S
Shape filter_shape(4);
// N K P Q
Shape output_shape(4);
// cpu workspace
Shape workspace_shape_cpu(1);

void compare(float* algo1, float* algo2, size_t size) {
    size_t i = 0;
	for (i = 0; i < size; ++i) {
		if ((algo1[i] - algo2[i] > 0.001f )|| (algo1[i] - algo2[i] < -0.001f)) {
		//	std::cout << "Index: " << i << " value1: " << algo1[i] << " value2: " << algo2[i] << std::endl;
            std::cout << "Index: " << i << "  diff: " << algo1[i] - algo2[i] << std::endl;
		}
	}
}

void output_convolution_transform(size_t N, size_t CHW, float* output) {
  for (size_t j = 0; j < CHW; ++j) {
    for (size_t i = 0; i < N; ++i) {
      std::cout << output[i * CHW + j] << " ";
    }
  }
  std::cout << std::endl;
}

void init_input(size_t N, size_t C, size_t H, size_t W, float* input) {
  size_t value = 0;
  for (size_t i = 0; i < C * H * W; ++i) {
    for (size_t j = 0; j < N; ++j) {
      input[j * C * H * W + i] = value++;
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

void convolution_forward(
  BLITZ_ALGORITHM algorithm,
  size_t pad_h, size_t pad_w,
  size_t str_h, size_t str_w) {
  // set up cpu
    CPUTensor<float> input_cpu(input_shape);
    CPUTensor<float> filter_cpu(filter_shape);
    CPUTensor<float> output_cpu(output_shape);
    CPUTensor<float> workspace_cpu(workspace_shape_cpu);
  
    // set up mic
    MICTensor<float> input_mic(input_shape);
    MICTensor<float> filter_mic(filter_shape);
    MICTensor<float> output_mic(output_shape);
    MICTensor<float> workspace_mic(workspace_shape_cpu);
  
    // init values
  	Backend<CPUTensor, float>::UniformDistributionFunc(&filter_cpu, 0.0, 1.0);
  	Backend<CPUTensor, float>::UniformDistributionFunc(&input_cpu, 0.0, 1.0);
  	memcpy(filter_mic.data(), filter_cpu.data(), sizeof(float) * filter_cpu.size());
  	memcpy(input_mic.data(), input_cpu.data(), sizeof(float) * input_cpu.size());
//    std::cout << "input and filter difference" << std::endl;
//    compare(input_cpu.data(), input_mic.data(), input_mic.size());
//    compare(filter_cpu.data(), filter_mic.data(), filter_mic.size());
  
    // cpu convolution 
    Backend<CPUTensor, float>::Convolution2DForwardFunc(
      &input_cpu,
      &filter_cpu,
      &output_cpu,
      &workspace_cpu,
      pad_h, pad_w, 
      str_h, str_w,
      algorithm);
    std::cout << "cpu fwd finished" << std::endl;

  	// mic convolution
    Backend<MICTensor, float>::Convolution2DForwardFunc(
      &input_mic,
      &filter_mic,
      &output_mic,
      &workspace_mic,
      pad_h, pad_w, 
      str_h, str_w,
      BLITZ_CONVOLUTION_XSMM_DIRECT);
    std::cout << "mic fwd finished" << std:: endl;

    compare(output_cpu.data(), output_mic.data(), output_mic.size());
}

void convolution_backward(
	BLITZ_ALGORITHM algorithm,
  size_t pad_h, size_t pad_w,
  size_t str_h, size_t str_w) {
  // set up cpu
    CPUTensor<float> input_cpu(input_shape);
    CPUTensor<float> filter_cpu(filter_shape);
    CPUTensor<float> output_cpu(output_shape);
    CPUTensor<float> workspace_cpu(workspace_shape_cpu);
  
    // set up mic
    MICTensor<float> input_mic(input_shape);
    MICTensor<float> filter_mic(filter_shape);
    MICTensor<float> output_mic(output_shape);
    MICTensor<float> workspace_mic(workspace_shape_cpu);
  
    // init values
  	Backend<CPUTensor, float>::UniformDistributionFunc(&filter_cpu, 0.0, 1.0);
  	Backend<CPUTensor, float>::UniformDistributionFunc(&output_cpu, 0.0, 1.0);
  	memcpy(filter_mic.data(), filter_cpu.data(), sizeof(float) * filter_cpu.size());
  	memcpy(output_mic.data(), output_cpu.data(), sizeof(float) * output_cpu.size());
//    std::cout << "input and output difference" << std::endl;
//    compare(filter_cpu.data(), filter_mic.data(), filter_mic.size());
//    compare(output_cpu.data(), output_mic.data(), output_mic.size());
  
    // cpu convolution 
    Backend<CPUTensor, float>::Convolution2DBackwardFunc(
      &output_cpu,
      &filter_cpu,
      &input_cpu,
      &workspace_cpu,
      pad_h, pad_w, 
      str_h, str_w,
      algorithm);
    std::cout << "cpu bwd finished" << std::endl;

  	// mic convolution
    Backend<MICTensor, float>::Convolution2DBackwardFunc(
      &output_mic,
      &filter_mic,
      &input_mic,
      &workspace_mic,
      pad_h, pad_w, 
      str_h, str_w,
      BLITZ_CONVOLUTION_XSMM_DIRECT);
    std::cout << "mic bwd finished" << std:: endl;

    compare(output_cpu.data(), output_mic.data(), output_mic.size());
}


void convolution_update(
	BLITZ_ALGORITHM algorithm,
  size_t pad_h, size_t pad_w,
  size_t str_h, size_t str_w) {
  // set up cpu
    CPUTensor<float> input_cpu(input_shape);
    CPUTensor<float> filter_cpu(filter_shape);
    CPUTensor<float> output_cpu(output_shape);
    CPUTensor<float> workspace_cpu(workspace_shape_cpu);
  
    // set up mic
    MICTensor<float> input_mic(input_shape);
    MICTensor<float> filter_mic(filter_shape);
    MICTensor<float> output_mic(output_shape);
    MICTensor<float> workspace_mic(workspace_shape_cpu);
  
    // init values
  	Backend<CPUTensor, float>::UniformDistributionFunc(&input_cpu, 0.0, 1.0);
  	Backend<CPUTensor, float>::UniformDistributionFunc(&output_cpu, 0.0, 1.0);
  	memcpy(input_mic.data(), input_cpu.data(), sizeof(float) * input_cpu.size());
  	memcpy(output_mic.data(), output_cpu.data(), sizeof(float) * output_cpu.size());
//    compare(input_cpu.data(), input_mic.data(), input_mic.size());
//    compare(output_cpu.data(), output_mic.data(), output_mic.size());
  
    // cpu convolution 
    Backend<CPUTensor, float>::Convolution2DUpdateFunc(
      &input_cpu,
      &output_cpu,
      &filter_cpu,
      &workspace_cpu,
      pad_h, pad_w, 
      str_h, str_w,
      algorithm);
    std::cout << "cpu pwd finished" << std::endl;

  	// mic convolution
    Backend<MICTensor, float>::Convolution2DUpdateFunc(
      &input_mic,
      &output_mic,
      &filter_mic,
      &workspace_mic,
      pad_h, pad_w, 
      str_h, str_w,
      BLITZ_CONVOLUTION_XSMM_DIRECT);
    std::cout << "mic pwd finished" << std:: endl;

    compare(output_cpu.data(), output_mic.data(), output_mic.size());
}


int main(int argc, char** argv) {
  const size_t NUM_ARGS = 16;
  // phase kernel N C H W R S K P Q pad_h pad_w str_h str_w iter
  // get args
  const std::string phase = std::string(argv[1]); 
  const std::string kernel = std::string(argv[2]); 
  const size_t iter = atoi(argv[3]);
  const size_t H = atoi(argv[4]);
  const size_t W = atoi(argv[5]);
  const size_t N = atoi(argv[6]);
  const size_t C = atoi(argv[7]);
  const size_t K = atoi(argv[8]);
  const size_t R = atoi(argv[9]);
  const size_t S = atoi(argv[10]);
  const size_t pad_h = atoi(argv[11]);
  const size_t pad_w = pad_h;
  const size_t str_h = atoi(argv[12]);
  const size_t str_w = str_h;
  const size_t P = atoi(argv[13]);
  const size_t Q = atoi(argv[14]);
   //set shapes
  set_input_shape_nchw(N, C, H, W);
  set_filter_shape_kcrs(K, C, R, S);
  set_output_shape_nkpq(N, K, P, Q);
  // set workspace shape
  //  workspace_shape_cpu[0] = BLITZ_NUM_THREADS * C * H * W * P * Q;
  workspace_shape_cpu[0] = C * H * W * P * Q;
  std::cout << phase << std::endl;
  // run convolution
  if (phase == "forward") 
      convolution_forward(BLITZ_CONVOLUTION_BLAS_GEMM, pad_h, pad_w, str_h, str_w);
  else if (phase == "backward")
      convolution_backward(BLITZ_CONVOLUTION_BLAS_GEMM, pad_h, pad_w, str_h, str_w);
  else if (phase == "update")
      convolution_update(BLITZ_CONVOLUTION_BLAS_GEMM, pad_h, pad_w, str_h, str_w);
  else
      std::cout << "wrong phase" << std::endl;
return 0;
}
