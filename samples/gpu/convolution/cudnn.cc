#include <cudnn.h>
#include "utils/common.h"
#include "utils/blitz_gpu_function.h"
#include "utils/blitz_cpu_function.h"
#include "backends/backends.h"

using namespace blitz;

// algorithms for forward and backwards convolutions
cudnnHandle_t cudnn_handle;
cudnnConvolutionFwdAlgo_t forward_algorithm;
cudnnConvolutionBwdFilterAlgo_t backward_filter_algorithm;
cudnnConvolutionBwdDataAlgo_t backward_data_algorithm;
cudnnTensorDescriptor_t input_desc, output_desc;
cudnnFilterDescriptor_t filter_desc;
cudnnConvolutionDescriptor_t conv_desc;
float *cudnn_alpha, *cudnn_beta;
// N C H W
Shape input_shape(4);
// K C R S
Shape filter_shape(4);
// N K P Q
Shape output_shape(4);
// cpu workspace
Shape workspace_shape_cpu(1);
// gpu workspace
Shape workspace_shape_gpu(1);
// init timer
float elapsed_time_gpu;
CUevent event_start, event_stop;

void set_forward_algorithm(const string& name) {
	if (name == "direct") {
		forward_algorithm = CUDNN_CONVOLUTION_FWD_ALGO_DIRECT;
	} else if (name == "fft_tiling") {
		forward_algorithm = CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING;
	} else if (name == "fft") {
		forward_algorithm = CUDNN_CONVOLUTION_FWD_ALGO_FFT;
	} else if (name == "gemm_pre") {
		forward_algorithm = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
	} else if (name == "gemm_implicit") {
		forward_algorithm = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
	} else if (name == "gemm") {
		forward_algorithm = CUDNN_CONVOLUTION_FWD_ALGO_GEMM;
	} else if (name == "winograd") {
		forward_algorithm = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD;
	}
}

void parse_forward_algorithm() {
	switch (forward_algorithm) {
		case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM:
			std::cout << "Implicit GEMM" << std::endl;
			break;
		case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM:
			std::cout << "Implicit precomputed GEMM" << std::endl;
			break;
		case CUDNN_CONVOLUTION_FWD_ALGO_GEMM:
			std::cout << "Explicit GEMM" << std::endl;
			break;
		case CUDNN_CONVOLUTION_FWD_ALGO_DIRECT:
			std::cout << "Direct" << std::endl;
			break;
		case CUDNN_CONVOLUTION_FWD_ALGO_FFT:
			std::cout << "FFT" << std::endl;
			break;
		case CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING:
			std::cout << "FFT tiling" << std::endl;
			break;
		case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD:
			std::cout << "Winograd" << std::endl;
			break;
		default:
			break;
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

void convolution_backward(const string& kernel, size_t iter) {
}

void convolution_update(const string& kernel, size_t iter) {
}

void convolution_forward(const string& kernel, size_t iter) {
  // set up gpu
  GPUTensor<float> input_gpu(input_shape);
  GPUTensor<float> filter_gpu(filter_shape);
  GPUTensor<float> output_gpu(output_shape);
	size_t workspace_size = 0;
	// set algorithms
	if (kernel == "auto") {  // fastest
		CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(cudnn_handle,
			input_desc,
			filter_desc,
			conv_desc,
			output_desc,
			CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
			2e9,  // memoryLimitInBytes,
			&forward_algorithm));
	}	else {
		set_forward_algorithm(kernel);
	}
	parse_forward_algorithm();
	// config workspace
	CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle,
		input_desc,
		filter_desc,
		conv_desc,
		output_desc,
		forward_algorithm,
		&workspace_size));
	std::cout << "workspace_size: " << workspace_size << std::endl;
	Shape workspace_shape(1);
	workspace_shape[0] = workspace_size;
	GPUTensor<float> workspace_gpu(workspace_shape);
	CUDNN_CHECK(cudnnConvolutionForward(cudnn_handle,
		reinterpret_cast<void*>(cudnn_alpha),
		input_desc,
		input_gpu.data(),
		filter_desc,
		filter_gpu.data(),
		conv_desc,
		forward_algorithm,
		workspace_gpu.data(),
		workspace_size,
		reinterpret_cast<void*>(cudnn_beta),
		output_desc,
		output_gpu.data()));
	BLITZ_GPU_TIMER_START(elapsed_time_gpu, event_start, event_stop);
	for (size_t i = 1; i < iter; ++i) {
		CUDNN_CHECK(cudnnConvolutionForward(cudnn_handle,
			reinterpret_cast<void*>(cudnn_alpha),
			input_desc,
			input_gpu.data(),
			filter_desc,
			filter_gpu.data(),
			conv_desc,
			forward_algorithm,
			workspace_gpu.data(),
			workspace_size,
			reinterpret_cast<void*>(cudnn_beta),
			output_desc,
			output_gpu.data()));
	}
	BLITZ_GPU_TIMER_END(elapsed_time_gpu, event_start, event_stop);
	BLITZ_GPU_TIMER_INFO((iter - 1) * 2 * filter_shape.size() * output_shape[0] * output_shape[2] * output_shape[3], elapsed_time_gpu);
}

int main(int argc, char** argv) {
  const size_t NUM_ARGS = 16;
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
  const size_t iterations = atoi(argv[16]);
  // set shapes
  set_input_shape_nchw(N, C, H, W);
  set_filter_shape_kcrs(K, C, R, S);
  set_output_shape_nkpq(N, K, P, Q);
	// create val
	cudnn_alpha = new float(1.0);
	cudnn_beta = new float(0.0);
	// create handle
	cudnnCreate(&cudnn_handle);
	// create descriptors
	cudnn::createTensor4dDesc<float>(&input_desc);
	cudnn::createTensor4dDesc<float>(&output_desc);
	cudnn::createFilterDesc<float>(&filter_desc);
	cudnn::createConvolution2DDesc<float>(&conv_desc);
	// set descriptors
	cudnn::setTensor4dDesc<float>(&input_desc, N, C, H, W);
	cudnn::setTensor4dDesc<float>(&output_desc, N, K, P, Q);
	cudnn::setFilterDesc<float>(&filter_desc, K, C, R, S);
	cudnn::setConvolution2DDesc<float>(&conv_desc, pad_h, pad_w, str_h, str_w);
  // run convolution
	if (phase == "forward") {
		convolution_forward(kernel, iterations);
	} else if (phase == "backward") {
		convolution_backward(kernel, iterations);
	} else if (phase == "update") {
		convolution_update(kernel, iterations);
	}
	return 0;
}
