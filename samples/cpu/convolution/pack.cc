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

void compare(const float* algo1, const float* algo2, size_t size) {
	for (size_t i = 0; i < size; ++i) {
		if (algo1[i] >= algo2[i] + 1e-5 || algo1[i] <= algo2[i] - 1e-5) {
			std::cout << "index: " << i << " value1: " << algo1[i] << " value2: " << algo2[i] << std::endl;
		}
	}
}

void unpack_stride_multi_pad(
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
	for (size_t channel_index = 0; channel_index < channel; ++channel_index) {
		for (size_t output_height_index = 0; output_height_index < output_height; ++output_height_index) {
			for (size_t filter_height_index = 0; filter_height_index < filter_height; ++filter_height_index) {
				float* unpack_slice = unpack + output_height_index * output_width * filter_height * filter_width * channel + channel_index * filter_height * filter_width + filter_height_index * filter_width;
				if (filter_height_index + output_height_index * stride_height < padding_height ||
					filter_height_index + output_height_index * stride_height >= padding_height + input_height) {
					for (size_t output_width_index = 0; output_width_index < output_width; ++output_width_index) {
						float* unpack_slice_slice = unpack_slice + output_width_index * filter_height * filter_width * channel;
						#pragma simd vectorlength(4)
						#pragma vector aligned
						for (size_t filter_width_index = 0; filter_width_index < filter_width; ++filter_width_index) {
							unpack_slice_slice[filter_width_index] = 0;
						}
					}
				} else {
					const float* input_slice = input + channel_index * input_height * input_width + (output_height_index * stride_height + filter_height_index - padding_height) * input_width;
					for (size_t output_width_index = 0; output_width_index < output_width; ++output_width_index) {
						const float* input_slice_slice = input_slice + output_width_index * stride_width;
						float* unpack_slice_slice = unpack_slice + output_width_index * filter_height * filter_width * channel;
						int filter_width_index = 0; 
						#pragma unroll
						for (; filter_width_index + output_width_index * stride_width < padding_width; ++filter_width_index) {
							unpack_slice_slice[filter_width_index] = 0;
						}
						size_t output_end = std::min(input_width + padding_width - output_width_index * stride_width, filter_width);
						size_t padding_end = std::min(input_width + 2 * padding_width, filter_width);
						#pragma simd vectorlength(4)
						#pragma vector aligned
						for (; filter_width_index < output_end; ++filter_width_index) {
							unpack_slice_slice[filter_width_index] = input_slice_slice[filter_width_index - padding_width];
						}
						#pragma unroll
						for (; filter_width_index < padding_end; ++filter_width_index) {
							unpack_slice_slice[filter_width_index] = 0;
						}
					}
				}
			}
		}
	}
}

void unpack_stride_multi_pad_hwc(
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
	for (size_t output_height_index = 0; output_height_index < output_height; ++output_height_index) {
		for (size_t filter_height_index = 0; filter_height_index < filter_height; ++filter_height_index) {
			float* unpack_slice = unpack + output_height_index * output_width * filter_height * filter_width * channel + filter_height_index * filter_width * channel;
			if (filter_height_index + output_height_index * stride_height < padding_height ||
				filter_height_index + output_height_index * stride_height >= padding_height + input_height) {
				for (size_t output_width_index = 0; output_width_index < output_width; ++output_width_index) {
					float* unpack_slice_slice = unpack_slice + output_width_index * filter_height * filter_width * channel;
					#pragma simd
					#pragma vector aligned
					for (size_t filter_width_index = 0; filter_width_index < filter_width * channel; ++filter_width_index) {
						unpack_slice_slice[filter_width_index] = 0;
					}
				}
			} else {
				const float* input_slice = input + (output_height_index * stride_height + filter_height_index - padding_height) * input_width * channel;
				for (size_t output_width_index = 0; output_width_index < output_width; ++output_width_index) {
					const float* input_slice_slice = input_slice + output_width_index * stride_width * channel;
					float* unpack_slice_slice = unpack_slice + output_width_index * filter_height * filter_width * channel;
					int filter_width_index = 0; 
					#pragma unroll
					for (; filter_width_index + output_width_index * stride_width < padding_width; ++filter_width_index) {
						#pragma simd
						#pragma vector aligned
						for (size_t channel_index = 0; channel_index < channel; ++channel_index) {
							unpack_slice_slice[filter_width_index * channel + channel_index] = 0;
						}
					}
					size_t output_end = std::min(input_width + padding_width - output_width_index * stride_width, filter_width);
					size_t padding_end = std::min(input_width + 2 * padding_width, filter_width);
					#pragma unroll
					for (; filter_width_index < output_end; ++filter_width_index) {
						#pragma simd
						#pragma vector aligned
						for (size_t channel_index = 0; channel_index < channel; ++channel_index) {
							unpack_slice_slice[filter_width_index * channel + channel_index] = input_slice_slice[(filter_width_index - padding_width) * channel + channel_index];
						}
					}
					#pragma unroll
					for (; filter_width_index < padding_end; ++filter_width_index) {
						#pragma simd
						#pragma vector aligned
						for (size_t channel_index = 0; channel_index < channel; ++channel_index) {
							unpack_slice_slice[filter_width_index * channel + channel_index] = 0;
						}
					}
				}
			}
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
	for (size_t channel_index = 0; channel_index < channel; ++channel_index) {
		for (size_t output_height_index = 0; output_height_index < output_height; ++output_height_index) {
			for (size_t filter_height_index = 0; filter_height_index < filter_height; ++filter_height_index) {
				const float* input_slice = input + channel_index * input_height * input_width + (output_height_index * stride_height + filter_height_index) * input_width;
				float* unpack_slice = unpack + output_height_index * output_width * filter_height * filter_width * channel + channel_index * filter_height * filter_width + filter_height_index * filter_width;
				for (size_t output_width_index = 0; output_width_index < output_width; ++output_width_index) {
					#pragma simd vectorlength(4)
					#pragma vector aligned
					for (size_t filter_width_index = 0; filter_width_index < filter_width; ++filter_width_index) {
						unpack_slice[filter_width_index] = input_slice[filter_width_index];
					}
					unpack_slice += filter_height * filter_width * channel;
					input_slice += stride_width;
				}
			}
		}
	}
}

void unpack_stride_multi_hwc(
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
	for (size_t output_height_index = 0; output_height_index < output_height; ++output_height_index) {
		for (size_t filter_height_index = 0; filter_height_index < filter_height; ++filter_height_index) {
			const float* input_slice = input + (output_height_index * stride_height + filter_height_index) * input_width * channel;
			float* unpack_slice = unpack + output_height_index * output_width * filter_height * filter_width * channel + filter_height_index * filter_width * channel;
			for (size_t output_width_index = 0; output_width_index < output_width; ++output_width_index) {
				const float* input_slice_slice = input_slice + output_width_index * stride_width * channel;
				float* unpack_slice_slice = unpack_slice + output_width_index * filter_height * filter_width * channel;
				#pragma simd
				#pragma vector aligned
				for (size_t filter_width_index = 0; filter_width_index < filter_width * channel; ++filter_width_index) {
					unpack_slice_slice[filter_width_index] = input_slice_slice[filter_width_index];
				}
			}
		}
	}
}

void unpack_stride_one(
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
	float* raw_unpack = unpack;
	float* prev_unpack = unpack;
	const float* raw_input = input;
	const float* prev_input = input;
  for (size_t channel_index = 0; channel_index < channel; ++channel_index) {
		prev_input = raw_input + channel_index * input_height * input_width;
		input = prev_input;
		prev_unpack = raw_unpack + channel_index * filter_height * filter_width * output_height * output_width;
		unpack = prev_unpack;
		for (size_t output_height_index = 0; output_height_index < output_height; ++output_height_index) {
			for (size_t filter_height_index = 0; filter_height_index < filter_height; ++filter_height_index) {
				const float* input_slice = prev_input + filter_height_index * input_width;
				float* unpack_slice = prev_unpack + filter_height_index * filter_width * output_height * output_width;
				for (size_t filter_width_index = 0; filter_width_index < filter_width; ++filter_width_index) {
					unpack = unpack_slice + filter_width_index * output_height * output_width;
					input = input_slice + filter_width_index;
					#pragma simd
					#pragma vector aligned
					for (size_t output_width_index = 0; output_width_index < output_width; ++output_width_index) {
						unpack[output_width_index] = input[output_width_index];
					}
				}
			}
			prev_unpack += output_width;
			prev_input += input_width;
			unpack = prev_unpack;
			input = prev_input;
    }
  }
}

void unpack_stride_one_pad(
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
	float* raw_unpack = unpack;
	float* prev_unpack = unpack;
	const float* raw_input = input;
	const float* prev_input = input;
  for (size_t channel_index = 0; channel_index < channel; ++channel_index) {
		prev_input = raw_input + channel_index * input_height * input_width;
		input = prev_input;
		prev_unpack = raw_unpack + channel_index * filter_height * filter_width * output_height * output_width;
		unpack = prev_unpack;
		for (size_t output_height_index = 0; output_height_index < output_height; ++output_height_index) {
			for (size_t filter_height_index = 0; filter_height_index < filter_height; ++filter_height_index) {
				if (filter_height_index + output_height_index < padding_height ||
					filter_height_index + output_height_index >= padding_height + input_height) {
					for (size_t filter_width_index = 0; filter_width_index < filter_width; ++filter_width_index) {
						#pragma simd
						#pragma vector aligned
						for (size_t output_width_index = 0; output_width_index < output_width; ++output_width_index) {
							unpack[output_width_index] = 0;
						}
						unpack += output_height * output_width;
					}
				} else {
					for (size_t filter_width_index = 0; filter_width_index < filter_width; ++filter_width_index) {
						size_t output_width_index = 0;
						#pragma unroll
						for (; filter_width_index + output_width_index < padding_width; ++output_width_index) {
							unpack[output_width_index] = 0;
						}
						const size_t output_end = std::min(padding_width + input_width - filter_width_index, output_width);
						const size_t padding_end = std::min(input_width + 2 * padding_width - filter_width_index, output_width);
						#pragma simd
						#pragma vector aligned
						for (; output_width_index < output_end; ++output_width_index) {
							unpack[output_width_index] = input[output_width_index - padding_width];
						}
						#pragma unroll
						for (; output_width_index < padding_end; ++output_width_index) {
							unpack[output_width_index] = 0;
						}
						unpack += output_height * output_width;
						input++;	
					}
					input += input_width - filter_width;
				}
			}
			prev_unpack += output_width;
			if (output_height_index >= padding_height) {
				prev_input += input_width;
			}
			unpack = prev_unpack;
			input = prev_input;
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
	size_t H = input_shape[2];
	size_t W = input_shape[3];
	size_t C = filter_shape[1];
	size_t R = filter_shape[2];
	size_t S = filter_shape[3];
	size_t P = output_shape[2];
	size_t Q = output_shape[3];
  // set up cpu
  CPUTensor<float> input_cpu(input_shape);
  CPUTensor<float> input_cpu_transform(input_shape);
  CPUTensor<float> filter_cpu(filter_shape);
  CPUTensor<float> output_cpu(output_shape);
  CPUTensor<float> workspace_cpu(workspace_shape_cpu);
  CPUTensor<float> workspace_cpu_optimize(workspace_shape_cpu);
	CPUTensor<float> workspace_cpu_transform(workspace_shape_cpu);
	Backend<CPUTensor, float>::UniformDistributionFunc(&input_cpu, 0.0, 1.0);
	bool transform = false;
	bool hwc = false;

	timeval t1, t2; 
	double elapsed_time = 0.0f;

	gettimeofday(&t1, NULL);
	for (size_t i = 0; i < iterations; ++i) {
		Backend<CPUTensor, float>::Unpack2DFunc(
			input_cpu.data(), workspace_cpu.data(),
			C, H, W, R, S, P, Q,
			pad_h, pad_w, str_h, str_w);
	}
	gettimeofday(&t2, NULL);
	elapsed_time += (t2.tv_sec - t1.tv_sec) * 1000.0; 
	elapsed_time += (t2.tv_usec - t1.tv_usec) / 1000.0;
	elapsed_time /= 1000.0;
	std::cout << "general pack time: " << elapsed_time << std::endl;

	if (input_shape[1] >= 8) {
		hwc = true;
		input_chw2hwc(input_cpu.data(), input_cpu_transform.data(), C, H, W);
		memcpy(input_cpu.data(), input_cpu_transform.data(), sizeof(float) * input_cpu_transform.size());	
	}

	gettimeofday(&t1, NULL);
	for (size_t i = 0; i < iterations; ++i) {
		if (input_shape[1] >= 8) {
			transform = true;
			if (pad_h == 0 && pad_w == 0) {
				unpack_stride_multi_hwc(
					input_cpu.data(), workspace_cpu_optimize.data(),
					C, H, W, R, S, P, Q,
					pad_h, pad_w, str_h, str_w);
			} else {
				unpack_stride_multi_pad_hwc(
					input_cpu.data(), workspace_cpu_optimize.data(),
					C, H, W, R, S, P, Q,
					pad_h, pad_w, str_h, str_w);
			}
		} else {
			if (str_h == 1 && str_w == 1) {
				if (pad_h == 0 && pad_w == 0) {
					unpack_stride_one_pad(
						input_cpu.data(), workspace_cpu_optimize.data(),
						C, H, W, R, S, P, Q,
						pad_h, pad_w, str_h, str_w);
				} else {
					unpack_stride_one_pad(
						input_cpu.data(), workspace_cpu_optimize.data(),
						C, H, W, R, S, P, Q,
						pad_h, pad_w, str_h, str_w);
				}
			} else {
				transform = true;
				if (pad_h == 0 && pad_w == 0) {
					unpack_stride_multi(
						input_cpu.data(), workspace_cpu_optimize.data(),
						C, H, W, R, S, P, Q,
						pad_h, pad_w, str_h, str_w);
				} else {
					unpack_stride_multi_pad(
						input_cpu.data(), workspace_cpu_optimize.data(),
						C, H, W, R, S, P, Q,
						pad_h, pad_w, str_h, str_w);
				}
			}
		}
	}
	gettimeofday(&t2, NULL);
	elapsed_time += (t2.tv_sec - t1.tv_sec) * 1000.0; 
	elapsed_time += (t2.tv_usec - t1.tv_usec) / 1000.0;
	elapsed_time /= 1000.0;
	std::cout << "optimize pack time: " << elapsed_time << std::endl;

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

void pack(size_t pad_h, size_t pad_w, size_t str_h, size_t str_w, size_t iterations, const int a) {
}

int main(int argc, char** argv) {
  const size_t NUM_ARGS = 14;
  // phase C H W R S K P Q pad_h pad_w str_h str_w iterations
  if (argc != NUM_ARGS + 1) {
    std::cerr << "Not enough args!" << std::endl;
    exit(1);
  }
  // get args
  const std::string phase = std::string(argv[1]); 
  const size_t C = atoi(argv[2]);
  const size_t H = atoi(argv[3]);
  const size_t W = atoi(argv[4]);
  const size_t R = atoi(argv[5]);
  const size_t S = atoi(argv[6]);
  const size_t K = atoi(argv[7]);
  const size_t P = atoi(argv[8]);
  const size_t Q = atoi(argv[9]);
  const size_t pad_h = atoi(argv[10]);
  const size_t pad_w = atoi(argv[11]);
  const size_t str_h = atoi(argv[12]);
  const size_t str_w = atoi(argv[13]);
	const size_t iterations = atoi(argv[14]);
  // set shapes
  set_input_shape_nchw(1, C, H, W);
  set_filter_shape_kcrs(K, C, R, S);
  set_output_shape_nkpq(1, K, P, Q);
  // set workspace shape
  workspace_shape_cpu[0] = C * R * S * P * Q;
  // run pack
  if (phase == "pack") {
    pack(pad_h, pad_w, str_h, str_w, iterations, 1);
  } else if (phase == "unpack") {
    unpack(pad_h, pad_w, str_h, str_w, iterations);
  }
	return 0;
}
