#include <omp.h>
#include <blitz.h>

using namespace blitz;

#define ACCESS_INPUT(data, i, j, k, v) *(data + ((i * input_channel + j) * input_height + k) * input_width + v)
#define ACCESS_OUTPUT(data, i, j, k, v) *(data + ((i * output_channel + j) * output_height + k) * output_width + v)
#define ACCESS_FILTER(data, i, j, k, v) *(data + ((i * input_channel + j) * filter_height + k) * filter_width + v)

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

void nhwc2nchw(const float* nhwc, float* nchw, size_t batch_size, size_t channel, size_t input_height, size_t input_width) {
  for (size_t b = 0; b < batch_size; ++b) {
    const float *hwc = nhwc + b * channel * input_height * input_width;
    float *chw = nchw + b * channel * input_height * input_width;
    for (size_t i = 0; i < channel; ++i) {
      for (size_t j = 0; j < input_height; ++j) {
        for (size_t k = 0; k < input_width; ++k) {
          chw[i * input_height * input_width + j * input_width + k] = hwc[j * input_width * channel + k * channel + i];
        }
      }
    }
  }
}

void nchw2nhwc(const float* nchw, float* nhwc, size_t batch_size, size_t channel, size_t input_height, size_t input_width) {
  for (size_t b = 0; b < batch_size; ++b) {
    const float *chw = nchw + b * channel * input_height * input_width;
    float *hwc = nhwc + b * channel * input_height * input_width;
    for (size_t i = 0; i < channel; ++i) {
      for (size_t j = 0; j < input_height; ++j) {
        for (size_t k = 0; k < input_width; ++k) {
          hwc[j * input_width * channel + k * channel + i] = chw[i * input_height * input_width + j * input_width + k];
        }
      }
    }
  }
}

void pqrsc2crspq(const float* hwc, float* chw, size_t output_height, size_t output_width,
  size_t channel, size_t filter_height, size_t filter_width) {
  for (size_t i = 0; i < output_height * output_width; ++i) {
    for (size_t j = 0; j < filter_height; ++j) {
      for (size_t k = 0; k < filter_width; ++k) {
        for (size_t v = 0; v < channel; ++v) {
          chw[((v * filter_height + j) * filter_width + k) * output_height * output_width + i] = 
            hwc[i * filter_height * filter_width * channel + j * filter_width * channel + k * channel + v];
        }
      }
    }
  }
}

void crspq2pqrsc(const float* chw, float* hwc, size_t output_height, size_t output_width,
  size_t channel, size_t filter_height, size_t filter_width) {
  for (size_t i = 0; i < output_height * output_width; ++i) {
    for (size_t j = 0; j < filter_height; ++j) {
      for (size_t k = 0; k < filter_width; ++k) {
        for (size_t v = 0; v < channel; ++v) {
          hwc[i * filter_height * filter_width * channel + j * filter_width * channel + k * channel + v] =
            chw[((v * filter_height + k) * filter_width + j) * output_height * output_width + i];
        }
      }
    }
  }
}

void kcrs2krsc(const float* kcrs, float* krsc, size_t output_channel, size_t input_channel,
  size_t filter_height, size_t filter_width) {
  for (size_t i = 0; i < output_channel; ++i) {
    for (size_t j = 0; j < input_channel; ++j) {
      for (size_t k = 0; k < filter_height; ++k) {
        for (size_t v = 0; v < filter_width; ++v) {
          krsc[((i * filter_height + k) * filter_width + v) * input_channel + j] =
	    kcrs[((i * input_channel + j) * filter_height + k) * filter_width + v];
	}
      }
    }
  }
}

void krsc2kcrs(const float* krsc, float* kcrs, size_t output_channel, size_t input_channel,
  size_t filter_height, size_t filter_width) {
  for (size_t i = 0; i < output_channel; ++i) {
    for (size_t j = 0; j < input_channel; ++j) {
      for (size_t k = 0; k < filter_height; ++k) {
        for (size_t v = 0; v < filter_width; ++v) {
	  kcrs[((i * input_channel + j) * filter_height + k) * filter_width + v] = 
            krsc[((i * filter_height + k) * filter_width + v) * input_channel + j];
	}
      }
    }
  }
}

void forward_base(
  const float* input,
  const float* filter,
  float* output,
  size_t batch_size,
  size_t input_channel,
  size_t input_height,
  size_t input_width,
  size_t filter_height,
  size_t filter_width,
  size_t output_channel,
  size_t output_height,
  size_t output_width,
  size_t padding_height,
  size_t padding_width,
  size_t stride_height,
  size_t stride_width) {
  // borrow from libxsmm
  #pragma omp parallel for
  for (size_t b = 0; b < batch_size; ++b) {
    for (size_t oc = 0; oc < output_channel; ++oc) {
      for (size_t ic = 0; ic < input_channel; ++ic) {
        for (size_t p = 0; p < output_height; ++p) {
          int ih = p * stride_height - padding_height;
          for (size_t q = 0; q < output_width; ++q) {
            int iw = q * stride_width - padding_width;
            for (size_t r = 0; r < filter_height; ++r) {
              if (ih + static_cast<int>(r) >= 0 && ih + static_cast<int>(r) < static_cast<int>(input_height)) {
                for (size_t s = 0; s < filter_width; ++s) {
	          if (iw + static_cast<int>(s) >= 0 && iw + static_cast<int>(s) < static_cast<int>(input_width)) {
                    ACCESS_OUTPUT(output, b, oc, p, q) += ACCESS_INPUT(input, b, ic, ih + r, iw + s) *
	              ACCESS_FILTER(filter, oc, ic, r, s); 
                  }
                }
	      }
            }
          }
        }
      }
    }
  }
}

void backward_base(
  const float* output,
  const float* filter,
  float* input,
  size_t batch_size,
  size_t input_channel,
  size_t input_height,
  size_t input_width,
  size_t filter_height,
  size_t filter_width,
  size_t output_channel,
  size_t output_height,
  size_t output_width,
  size_t padding_height,
  size_t padding_width,
  size_t stride_height,
  size_t stride_width) {
  // borrow from libxsmm
  #pragma omp parallel for
  for (size_t b = 0; b < batch_size; ++b) {
    for (size_t ic = 0; ic < input_channel; ++ic) {
      for (size_t oc = 0; oc < output_channel; ++oc) {
        for (size_t oh = 0; oh < output_height; ++oh) {
          int ih = oh * stride_height - padding_height;
          for (size_t ow = 0; ow < output_width; ++ow) {
	    int iw = ow * stride_width - padding_width;
            for (size_t r = 0; r < filter_height; ++r) {
              if (ih + static_cast<int>(r) >= 0 && ih + static_cast<int>(r) < static_cast<int>(input_height)) {
                for (size_t s = 0; s < filter_width; ++s) {
	          if (iw + static_cast<int>(s) >= 0 && iw + static_cast<int>(s) < static_cast<int>(input_width)) {
                    ACCESS_INPUT(input, b, ic, ih + r, iw + s) += ACCESS_OUTPUT(output, b, oc, oh, ow) *
	              ACCESS_FILTER(filter, oc, ic, r, s); 
                  }
                }
	      }
            }
          }
        }
      }
    }
  }
}

void update_base(
  const float* input,
  const float* output,
  float* update,
  size_t batch_size,
  size_t input_channel,
  size_t input_height,
  size_t input_width,
  size_t filter_height,
  size_t filter_width,
  size_t output_channel,
  size_t output_height,
  size_t output_width,
  size_t padding_height,
  size_t padding_width,
  size_t stride_height,
  size_t stride_width) {
  // borrow from libxsmm
  #pragma omp parallel for
  for (size_t ic = 0; ic < input_channel; ++ic) {
    for (size_t oc = 0; oc < output_channel; ++oc) {
      for (size_t b = 0; b < batch_size; ++b) {
        for (size_t oh = 0; oh < output_height; ++oh) {
          int ih = oh * stride_height - padding_height;
          for (size_t ow = 0; ow < output_width; ++ow) {
	    int iw = ow * stride_width - padding_width;
            for (size_t r = 0; r < filter_height; ++r) {
              if (ih + static_cast<int>(r) >= 0 && ih + static_cast<int>(r) < static_cast<int>(input_height)) {
                for (size_t s = 0; s < filter_width; ++s) {
	          if (iw + static_cast<int>(s) >= 0 && iw + static_cast<int>(s) < static_cast<int>(input_width)) {
                    ACCESS_FILTER(update, oc, ic, r, s) += ACCESS_INPUT(input, b, ic, ih + r, iw + s) *
	              ACCESS_OUTPUT(output, b, oc, oh, ow); 
                  }
                }
	      }
            }
          }
        }
      }
    }
  }
}

void convolution_forward(
  BLITZ_ALGORITHM algorithm,
  size_t pad_h, size_t pad_w,
  size_t str_h, size_t str_w,
  size_t iter) {
  // set up cpu
  CPUTensor<float> input_cpu(input_shape);
  CPUTensor<float> input_cpu_algorithm(input_shape);
  CPUTensor<float> filter_cpu(filter_shape);
  CPUTensor<float> filter_cpu_algorithm(filter_shape);
  CPUTensor<float> output_cpu(output_shape);
  CPUTensor<float> output_cpu_algorithm(output_shape);
  CPUTensor<float> workspace_cpu(workspace_shape_cpu);
  size_t workspace_size = workspace_shape_cpu.size() * omp_get_max_threads();
  Shape workspace_shape_algorithm(1);
  workspace_shape_algorithm[0] = workspace_size;
  CPUTensor<float> workspace_cpu_algorithm(workspace_shape_algorithm);
  size_t NIN, C, H, W;
  size_t KF, CF, R, S;
  size_t NOUT, K, P, Q;
  Blitz2DBuffer(input_cpu.data_layout(), input_cpu.shape_ptr(), &NIN, &C, &H, &W);
  Blitz2DFilter(filter_cpu.data_layout(), filter_cpu.shape_ptr(), &KF, &CF, &R, &S);
  Blitz2DBuffer(output_cpu.data_layout(), output_cpu.shape_ptr(), &NOUT, &K, &P, &Q);
  // init values
  Backend<CPUTensor, float>::UniformDistributionFunc(&filter_cpu, 0.0, 1.0);
  Backend<CPUTensor, float>::UniformDistributionFunc(&input_cpu, 0.0, 1.0);
  if (input_cpu.data_layout() == BLITZ_BUFFER_NHWC) {
    nchw2nhwc(input_cpu.data(), input_cpu_algorithm.data(), NIN, C, H, W);
    kcrs2krsc(filter_cpu.data(), filter_cpu_algorithm.data(), KF, CF, R, S);
  } else {
    memcpy(input_cpu_algorithm.data(), input_cpu.data(), sizeof(float) * input_cpu.size());
    memcpy(filter_cpu_algorithm.data(), filter_cpu.data(), sizeof(float) * filter_cpu.size());
  }
  // cpu convolution 
  for (size_t i = 0; i < iter; ++i) {
    forward_base(input_cpu.data(),
      filter_cpu.data(),
      output_cpu.data(),
      NIN, C, H, W,
      R, S,
      K, P, Q,
      pad_h, pad_w,
      str_h, str_w);
  }
  // different algorithm
  for (size_t i = 0; i < iter; ++i) {
    Backend<CPUTensor, float>::Convolution2DForwardFunc(
      &input_cpu_algorithm,
      &filter_cpu_algorithm,
      &output_cpu_algorithm,
      &workspace_cpu_algorithm,
      pad_h, pad_w, 
      str_h, str_w,
      algorithm);
  }
  if (output_cpu_algorithm.data_layout() == BLITZ_BUFFER_NHWC) {
    CPUTensor<float> output_cpu_transform(output_shape);
    nhwc2nchw(output_cpu_algorithm.data(), output_cpu_transform.data(), NOUT, K, P, Q);
    memcpy(output_cpu_algorithm.data(), output_cpu_transform.data(), sizeof(float) * output_cpu_transform.size());
  }
  compare(output_cpu.data(), output_cpu_algorithm.data(), output_cpu.size(), 1e-2);
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
  CPUTensor<float> filter_cpu_algorithm(filter_shape);
  CPUTensor<float> output_cpu(output_shape);
  CPUTensor<float> output_cpu_algorithm(output_shape);
  CPUTensor<float> workspace_cpu(workspace_shape_cpu);
  size_t workspace_size = workspace_shape_cpu.size() * omp_get_max_threads();
  Shape workspace_shape_algorithm(1);
  workspace_shape_algorithm[0] = workspace_size;
  CPUTensor<float> workspace_cpu_algorithm(workspace_shape_algorithm);
  size_t NIN, C, H, W;
  size_t KF, CF, R, S;
  size_t NOUT, K, P, Q;
  Blitz2DBuffer(input_cpu.data_layout(), input_cpu.shape_ptr(), &NIN, &C, &H, &W);
  Blitz2DFilter(filter_cpu.data_layout(), filter_cpu.shape_ptr(), &KF, &CF, &R, &S);
  Blitz2DBuffer(output_cpu.data_layout(), output_cpu.shape_ptr(), &NOUT, &K, &P, &Q);
  // init values
  Backend<CPUTensor, float>::UniformDistributionFunc(&filter_cpu, 0.0, 1.0);
  Backend<CPUTensor, float>::UniformDistributionFunc(&output_cpu, 0.0, 1.0);
  if (output_cpu.data_layout() == BLITZ_BUFFER_NHWC) {
    nchw2nhwc(output_cpu.data(), output_cpu_algorithm.data(), NOUT, K, P, Q);
  } else {
    memcpy(output_cpu_algorithm.data(), output_cpu.data(), sizeof(float) * output_cpu.size());
  }
  if (input_cpu.data_layout() == BLITZ_BUFFER_NHWC) {
    kcrs2krsc(filter_cpu.data(), filter_cpu_algorithm.data(), KF, CF, R, S);
  } else {
    memcpy(filter_cpu_algorithm.data(), filter_cpu.data(), sizeof(float) * filter_cpu.size());
  }
  // cpu convolution 
  for (size_t i = 0; i < iter; ++i) {
    backward_base(output_cpu.data(),
      filter_cpu.data(),
      input_cpu.data(),
      NIN, C, H, W,
      R, S,
      K, P, Q,
      pad_h, pad_w,
      str_h, str_w);
  }
  // different algorithm
  for (size_t i = 0; i < iter; ++i) {
    Backend<CPUTensor, float>::Convolution2DBackwardFunc(
      &output_cpu_algorithm,
      &filter_cpu_algorithm,
      &input_cpu_algorithm,
      &workspace_cpu_algorithm,
      pad_h, pad_w, 
      str_h, str_w,
      algorithm);
  }
  if (input_cpu_algorithm.data_layout() == BLITZ_BUFFER_NHWC) {
    CPUTensor<float> input_cpu_transform(input_shape);
    nhwc2nchw(input_cpu_algorithm.data(), input_cpu_transform.data(), NIN, C, H, W);
    memcpy(input_cpu_algorithm.data(), input_cpu_transform.data(), sizeof(float) * input_cpu_transform.size());
  }
  compare(input_cpu.data(), input_cpu_algorithm.data(), input_cpu.size(), 1e-2);
}

void convolution_update(
  BLITZ_ALGORITHM algorithm,
  size_t pad_h, size_t pad_w,
  size_t str_h, size_t str_w,
  size_t iter) {
  // set up cpu
  CPUTensor<float> input_cpu(input_shape);
  CPUTensor<float> input_cpu_algorithm(input_shape);
  CPUTensor<float> filter_cpu(filter_shape);
  CPUTensor<float> filter_cpu_transform(filter_shape);
  CPUTensor<float> filter_cpu_algorithm(filter_shape);
  CPUTensor<float> output_cpu(output_shape);
  CPUTensor<float> output_cpu_algorithm(output_shape);
  CPUTensor<float> workspace_cpu(workspace_shape_cpu);
  size_t workspace_size = workspace_shape_cpu.size() * omp_get_max_threads();
  Shape workspace_shape_algorithm(1);
  workspace_shape_algorithm[0] = workspace_size;
  CPUTensor<float> workspace_cpu_algorithm(workspace_shape_algorithm);
  size_t NIN, C, H, W;
  size_t KF, CF, R, S;
  size_t NOUT, K, P, Q;
  Blitz2DBuffer(input_cpu.data_layout(), input_cpu.shape_ptr(), &NIN, &C, &H, &W);
  Blitz2DFilter(filter_cpu.data_layout(), filter_cpu.shape_ptr(), &KF, &CF, &R, &S);
  Blitz2DBuffer(output_cpu.data_layout(), output_cpu.shape_ptr(), &NOUT, &K, &P, &Q);
  // init values
  Backend<CPUTensor, float>::UniformDistributionFunc(&output_cpu, 0.0, 1.0);
  Backend<CPUTensor, float>::UniformDistributionFunc(&input_cpu, 0.0, 1.0);
  if (output_cpu.data_layout() == BLITZ_BUFFER_NHWC) {
    nchw2nhwc(output_cpu.data(), output_cpu_algorithm.data(), NOUT, K, P, Q);
  } else {
    memcpy(output_cpu_algorithm.data(), output_cpu.data(), sizeof(float) * output_cpu.size());
  }
  if (input_cpu.data_layout() == BLITZ_BUFFER_NHWC) {
    nchw2nhwc(input_cpu.data(), input_cpu_algorithm.data(), NIN, C, H, W);
  } else {
    memcpy(input_cpu_algorithm.data(), input_cpu.data(), sizeof(float) * input_cpu.size());
  }
  // cpu convolution 
  for (size_t i = 0; i < iter; ++i) {
    update_base(input_cpu.data(),
      output_cpu.data(),
      filter_cpu.data(),
      NIN, C, H, W,
      R, S,
      K, P, Q,
      pad_h, pad_w,
      str_h, str_w);
  }
  // different algorithm
  for (size_t i = 0; i < iter; ++i) {
    Backend<CPUTensor, float>::Convolution2DUpdateFunc(
      &input_cpu_algorithm,
      &output_cpu_algorithm,
      &filter_cpu_algorithm,
      &workspace_cpu_algorithm,
      pad_h, pad_w, 
      str_h, str_w,
      algorithm);
  }
  if (input_cpu_algorithm.data_layout() == BLITZ_BUFFER_NHWC) {
    krsc2kcrs(filter_cpu_algorithm.data(), filter_cpu_transform.data(), K, C, R, S);
    memcpy(filter_cpu_algorithm.data(), filter_cpu_transform.data(), sizeof(float) * filter_cpu_transform.size());
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
