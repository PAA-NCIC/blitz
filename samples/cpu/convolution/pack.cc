#include <iostream>
#include "backends/backends.h"
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

void set_input_shape_nchw(size_t N, size_t C, size_t H, size_t W, BLITZ_DATA_LAYOUT data_layout) {
  input_shape[0] = N;
  input_shape[1] = C;
  input_shape[2] = H;
  input_shape[3] = W;
	input_shape.set_data_layout(data_layout);
}

void set_input_shape_nhwc(size_t N, size_t C, size_t H, size_t W, BLITZ_DATA_LAYOUT data_layout) {
  input_shape[0] = N;
  input_shape[1] = H;
  input_shape[2] = W;
  input_shape[3] = C; 
	input_shape.set_data_layout(data_layout);
}

void set_filter_shape_kcrs(size_t K, size_t C, size_t R, size_t S, BLITZ_DATA_LAYOUT data_layout) {
  filter_shape[0] = K;
  filter_shape[1] = C;
  filter_shape[2] = R;
  filter_shape[3] = S;
	filter_shape.set_data_layout(data_layout);
}

void set_output_shape_nkpq(size_t N, size_t K, size_t P, size_t Q, BLITZ_DATA_LAYOUT data_layout) {
  output_shape[0] = N;
  output_shape[1] = K;
  output_shape[2] = P;
  output_shape[3] = Q;
	output_shape.set_data_layout(data_layout);
}

void compare(const float* algo1, const float* algo2, size_t size) {
	for (size_t i = 0; i < size; ++i) {
		if (algo1[i] >= algo2[i] + 1e-5 || algo1[i] <= algo2[i] - 1e-5) {
			std::cout << "index: " << i << " value1: " << algo1[i] << " value2: " << algo2[i] << std::endl;
		}
	}
}

void unpack_stride_multi(
  const float* input,
  float* unpack,
  size_t channel,
  size_t input_height,
  size_t input_width,
  size_t filter_height,
  size_t filter_width,
  size_t output_height,
  size_t output_width,
  size_t padding_height,
  size_t padding_width,
  size_t stride_height,
  size_t stride_width) {
	// (input_channel * filter_height * filter_width) *
	// (output_width * output_height)
	size_t unpack_index = 0;
	for (size_t channel_index = 0; channel_index < channel; ++channel_index) {
		const size_t channel_offset = channel_index * input_height * input_width;
		const float* input_slice = input + channel_offset;
		for (size_t filter_height_index = 0; filter_height_index < filter_height; ++filter_height_index) {
			for (size_t filter_width_index = 0; filter_width_index < filter_width; ++filter_width_index) {
				int filter_height_offset = -padding_height + filter_height_index;
				for (size_t output_height_index = 0; output_height_index < output_height; ++output_height_index) {
					if (filter_height_offset < 0 || filter_height_offset >= static_cast<int>(input_height)) {
						for (size_t output_width_index = 0; output_width_index < output_width; ++output_width_index) {
							unpack[unpack_index++] = 0;
						}
					} else {
						int filter_width_offset = -padding_width + filter_width_index;
						for (size_t output_width_index = 0; output_width_index < output_width; ++output_width_index) {
							if (filter_width_offset < 0 || filter_width_offset >= static_cast<int>(input_width)) {
								unpack[unpack_index++] = 0;
							} else {
								unpack[unpack_index++] = input_slice[filter_height_offset * input_width + filter_width_offset];
							}
							filter_width_offset += stride_width;
						}
					}
					filter_height_offset += stride_height;
				}
			}
		}
	}
}

void input_hwc2chw(const float* hwc, float* chw, size_t channel, size_t input_height, size_t input_width) {
	for (size_t i = 0; i < channel; ++i) {
		for (size_t j = 0; j < input_height; ++j) {
			for (size_t k = 0; k < input_width; ++k) {
				chw[i * input_height * input_width + j * input_width + k] = hwc[j * input_width * channel + k * channel + i];
			}
		}
	}
}

void input_chw2hwc(const float* chw, float* hwc, size_t channel, size_t input_height, size_t input_width) {
	for (size_t i = 0; i < channel; ++i) {
		for (size_t j = 0; j < input_height; ++j) {
			for (size_t k = 0; k < input_width; ++k) {
				hwc[j * input_width * channel + k * channel + i] = chw[i * input_height * input_width + j * input_width + k];
			}
		}
	}
}

void workspace_hwc2chw(const float* hwc, float* chw, size_t output_height, size_t output_width,
	size_t channel, size_t filter_height, size_t filter_width) {
	for (size_t i = 0; i < output_height * output_width; ++i) {
		for (size_t j = 0; j < filter_height; ++j) {
			for (size_t k = 0; k < filter_width; ++k) {
				for (size_t v = 0; v < channel; ++v) {
					chw[i * filter_height * filter_width * channel + v * filter_height * filter_width + j * filter_width + k] = 
						hwc[i * filter_height * filter_width * channel + j * filter_width * channel + k * channel + v];
				}
			}
		}
	}
}

void unpack(size_t pad_h, size_t pad_w, size_t str_h, size_t str_w, size_t iterations) {
	// shape decode
	size_t N, H, W, C, R, S, K, P, Q;
	Blitz2DBuffer(input_shape.data_layout(), &input_shape, &N, &C, &H, &W);
	Blitz2DFilter(filter_shape.data_layout(), &filter_shape, &K, &C, &R, &S);
	Blitz2DBuffer(output_shape.data_layout(), &output_shape, &N, &K, &P, &Q);
  // set up cpu
  CPUTensor<float> input_cpu(input_shape);
  CPUTensor<float> input_cpu_transform(input_shape);
  CPUTensor<float> filter_cpu(filter_shape);
  CPUTensor<float> output_cpu(output_shape);
  CPUTensor<float> workspace_cpu(workspace_shape_cpu);
  CPUTensor<float> workspace_cpu_optimize(workspace_shape_cpu);
	CPUTensor<float> workspace_cpu_transform(workspace_shape_cpu);
	Backend<CPUTensor, float>::UniformDistributionFunc(&input_cpu, 0.0, 1.0);
	BLITZ_DATA_LAYOUT data_layout;

	timeval t1, t2; 
	double elapsed_time = 0.0f;

	if (input_shape.data_layout() == BLITZ_BUFFER_NHWC) {
		input_hwc2chw(input_cpu.data(), input_cpu_transform.data(), C, H, W);
	} else {
		memcpy(input_cpu_transform.data(), input_cpu.data(), sizeof(float) * input_cpu.size());	
	}
	gettimeofday(&t1, NULL);
	for (size_t i = 0; i < iterations; ++i) {
		unpack_stride_multi(
			input_cpu_transform.data(), workspace_cpu.data(),
			C, H, W, R, S, P, Q,
			pad_h, pad_w, str_h, str_w);
	}
	gettimeofday(&t2, NULL);
	elapsed_time += (t2.tv_sec - t1.tv_sec) * 1000.0; 
	elapsed_time += (t2.tv_usec - t1.tv_usec) / 1000.0;
	elapsed_time /= 1000.0;
	std::cout << "general pack time: " << elapsed_time << std::endl;

	gettimeofday(&t1, NULL);
	for (size_t i = 0; i < iterations; ++i) {
		data_layout = Backend<CPUTensor, float>::Unpack2DFunc(
			input_cpu.data(), workspace_cpu_optimize.data(),
			C, H, W, R, S, P, Q,
			pad_h, pad_w, str_h, str_w,
			input_shape.data_layout());
	}
	gettimeofday(&t2, NULL);
	elapsed_time += (t2.tv_sec - t1.tv_sec) * 1000.0; 
	elapsed_time += (t2.tv_usec - t1.tv_usec) / 1000.0;
	elapsed_time /= 1000.0;
	std::cout << "optimize pack time: " << elapsed_time << std::endl;
	bool hwc = false;
	bool transform = false;

	if (data_layout == BLITZ_PACK_CRSPQ) {
	} else if (data_layout == BLITZ_PACK_PQRSC) {
		transform = true;
		hwc = true;
	} else if (data_layout == BLITZ_PACK_PQCRS) {
		transform = true;
	}

	if (hwc == true) {
		workspace_hwc2chw(workspace_cpu_optimize.data(), workspace_cpu_transform.data(), P, Q, C, R, S);
		memcpy(workspace_cpu_optimize.data(), workspace_cpu_transform.data(), sizeof(float) * workspace_cpu_optimize.size());	
	}

	if (transform == true) {
		Shape shape(2);
		shape[0] = P * Q;
		shape[1] = workspace_shape_cpu[0] / (P * Q);
		workspace_cpu_optimize.set_shape(shape);
		shape[0] = workspace_shape_cpu[0] / (P * Q);
		shape[1] = P * Q;
		workspace_cpu_transform.set_shape(shape);
		Backend<CPUTensor, float>::Transpose2DFunc(&workspace_cpu_optimize, &workspace_cpu_transform);
		memcpy(workspace_cpu_optimize.data(), workspace_cpu_transform.data(), sizeof(float) * workspace_cpu_optimize.size());	
	}	

	compare(workspace_cpu.data(), workspace_cpu_optimize.data(), workspace_shape_cpu.size());
}

void pack(size_t pad_h, size_t pad_w, size_t str_h, size_t str_w, size_t iterations) {
}

int main(int argc, char** argv) {
  const size_t NUM_ARGS = 17;
  // phase C H W R S K P Q pad_h pad_w str_h str_w iterations
  if (argc != NUM_ARGS + 1) {
    std::cerr << "Not enough args!" << std::endl;
    exit(1);
  }
  // get args
  const std::string phase = std::string(argv[1]); 
	const std::string input_data_layout = std::string(argv[2]);
	const std::string output_data_layout = std::string(argv[3]);
	const std::string filter_data_layout = std::string(argv[4]);
  const size_t C = atoi(argv[5]);
  const size_t H = atoi(argv[6]);
  const size_t W = atoi(argv[7]);
  const size_t R = atoi(argv[8]);
  const size_t S = atoi(argv[9]);
  const size_t K = atoi(argv[10]);
  const size_t P = atoi(argv[11]);
  const size_t Q = atoi(argv[12]);
  const size_t pad_h = atoi(argv[13]);
  const size_t pad_w = atoi(argv[14]);
  const size_t str_h = atoi(argv[15]);
  const size_t str_w = atoi(argv[16]);
	const size_t iterations = atoi(argv[17]);
  // set shapes
	if (BlitzParseShape(input_data_layout) == BLITZ_BUFFER_NCHW) {
		set_input_shape_nchw(1, C, H, W, BlitzParseShape(input_data_layout));
	} else {
		set_input_shape_nhwc(1, C, H, W, BlitzParseShape(input_data_layout));
	}
  set_filter_shape_kcrs(K, C, R, S, BlitzParseShape(filter_data_layout));
  set_output_shape_nkpq(1, K, P, Q, BlitzParseShape(output_data_layout));
  // set workspace shape
  workspace_shape_cpu[0] = C * R * S * P * Q;
  // run pack
  if (phase == "pack") {
    pack(pad_h, pad_w, str_h, str_w, iterations);
  } else if (phase == "unpack") {
    unpack(pad_h, pad_w, str_h, str_w, iterations);
  }
	return 0;
}
