#include <iostream>
#include "utils/blitz_shape_function.h"
#include "backends/backends.h"

using namespace blitz;

// N C H W
Shape input_shape_nchw(4);
// N H W C
Shape input_shape_nhwc(4);
// N K P Q
Shape output_shape_nkpq(4);
// N Q K P
Shape output_shape_npqk(4);

void compare(float* algo1, float* algo2, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    if (algo1[i] > algo2[i] + 1e-5 || algo1[i] < algo2[i] - 1e-5) {
      std::cout << "Index: " << i << " value1: " << algo1[i] << " value2: " << algo2[i] << std::endl;
    }
  }
}

void init_max_index(size_t* max_index, size_t window_size, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    max_index[i] = i % window_size;
  }
}

void set_input_shape_nhwc(size_t N, size_t C, size_t H, size_t W) {
  input_shape_nhwc[0] = N;
  input_shape_nhwc[1] = H;
  input_shape_nhwc[2] = W;
  input_shape_nhwc[3] = C;
  input_shape_nhwc.set_data_layout(BLITZ_POOLING_BUFFER_NHWC);
}

void set_output_shape_npqk(size_t N, size_t K, size_t P, size_t Q) {
  output_shape_npqk[0] = N;
  output_shape_npqk[1] = P;
  output_shape_npqk[2] = Q;
  output_shape_npqk[3] = K;
  output_shape_npqk.set_data_layout(BLITZ_POOLING_BUFFER_NHWC);
}

void set_input_shape_nchw(size_t N, size_t C, size_t H, size_t W) {
  input_shape_nchw[0] = N;
  input_shape_nchw[1] = C;
  input_shape_nchw[2] = H;
  input_shape_nchw[3] = W;
  input_shape_nchw.set_data_layout(BLITZ_POOLING_BUFFER_NCHW);
}

void set_output_shape_nkpq(size_t N, size_t K, size_t P, size_t Q) {
  output_shape_nkpq[0] = N;
  output_shape_nkpq[1] = K;
  output_shape_nkpq[2] = P;
  output_shape_nkpq[3] = Q;
  output_shape_nkpq.set_data_layout(BLITZ_POOLING_BUFFER_NCHW);
}

void nchw2nhwc(const float* nchw, float* nhwc, size_t batch, size_t channel, size_t height, size_t width) {
  for (size_t i = 0; i < batch; ++i) {
    for (size_t j = 0; j < channel; ++j) {
      for (size_t k = 0; k < height; ++k) {
        for (size_t v = 0; v < width; ++v) {
          nhwc[i * channel * height * width + k * width * channel + v * channel + j] = 
            nchw[i * channel * height * width + j * height * width + k * width + v];
        }
      }
    }
  }
}

void nchw2nhwc(const size_t* nchw, size_t* nhwc, size_t batch, size_t channel, size_t height, size_t width) {
  for (size_t i = 0; i < batch; ++i) {
    for (size_t j = 0; j < channel; ++j) {
      for (size_t k = 0; k < height; ++k) {
        for (size_t v = 0; v < width; ++v) {
          size_t hh = nchw[i * channel * height * width + j * height * width + k * width + v] / width;
          size_t ww = nchw[i * channel * height * width + j * height * width + k * width + v] % width;
          nhwc[i * channel * height * width + k * width * channel + v * channel + j] = (hh * width + ww) * channel + j;
        }
      }
    }
  }
}

void nhwc2nchw(const float* nhwc, float* nchw, size_t batch, size_t channel, size_t height, size_t width) {
  for (size_t i = 0; i < batch; ++i) {
    for (size_t j = 0; j < channel; ++j) {
      for (size_t k = 0; k < height; ++k) {
        for (size_t v = 0; v < width; ++v) {
          nchw[i * channel * height * width + j * height * width + k * width + v] = 
            nhwc[i * channel * height * width + k * width * channel + v * channel + j];
        }
      }
    }
  }
}

void pooling_forward_nchw(size_t filter_h, size_t filter_w, size_t str_h, size_t str_w) {
  // set up cpu
  CPUTensor<float> input_cpu(input_shape_nchw);
  CPUTensor<size_t> max_index_cpu(output_shape_nkpq);
  CPUTensor<float> output_cpu(output_shape_nkpq);
  // set up mic
  MICTensor<float> input_mic(input_shape_nchw);
  MICTensor<size_t> max_index_mic(output_shape_nkpq);
  MICTensor<float> output_mic(output_shape_nkpq);
  // init values
  Backend<CPUTensor, float>::UniformDistributionFunc(&input_cpu, 0.0, 1.0);
  memcpy(input_mic.data(), input_cpu.data(), sizeof(float) * input_cpu.size());
  // cpu pooling 
  Backend<CPUTensor, float>::MaxPooling2DForwardFunc(
    &input_cpu,
    &output_cpu,
    &max_index_cpu,
    filter_h, filter_w,
    str_h, str_w);
  // mic pooling
  Backend<MICTensor, float>::MaxPooling2DForwardFunc(
    &input_mic,
    &output_mic,
    &max_index_mic,
    filter_h, filter_w,
    str_h, str_w);
  compare(output_cpu.data(), output_mic.data(), output_mic.size());
}

void pooling_forward_nhwc(size_t filter_h, size_t filter_w, size_t str_h, size_t str_w) {
  // set up cpu
  CPUTensor<float> input_cpu(input_shape_nchw);
  CPUTensor<size_t> max_index_cpu(output_shape_nkpq);
  CPUTensor<float> output_cpu(output_shape_nkpq);
  // set up mic
  MICTensor<float> input_mic(input_shape_nhwc);
  MICTensor<size_t> max_index_mic(output_shape_npqk);
  MICTensor<float> output_mic(output_shape_npqk);
  MICTensor<float> output_mic_transform(output_shape_nkpq);
  // init values
  Backend<CPUTensor, float>::UniformDistributionFunc(&input_cpu, 0.0, 1.0);
  memcpy(input_mic.data(), input_cpu.data(), sizeof(float) * input_cpu.size());
  // cpu pooling 
  Backend<CPUTensor, float>::MaxPooling2DForwardFunc(
    &input_cpu,
    &output_cpu,
    &max_index_cpu,
    filter_h, filter_w,
    str_h, str_w);
  // mic pooling
  nchw2nhwc(input_cpu.data(), input_mic.data(),
    input_shape_nchw[0], input_shape_nchw[1], input_shape_nchw[2], input_shape_nchw[3]);
  Backend<MICTensor, float>::MaxPooling2DForwardFunc(
    &input_mic,
    &output_mic,
    &max_index_mic,
    filter_h, filter_w,
    str_h, str_w);
  nhwc2nchw(output_mic.data(), output_mic_transform.data(),
    output_shape_nkpq[0],  output_shape_nkpq[1], output_shape_nkpq[2], output_shape_nkpq[3]);
  compare(output_cpu.data(), output_mic_transform.data(), output_mic.size());
}

void pooling_backward_nchw(size_t filter_h, size_t filter_w, size_t str_h, size_t str_w) {
  // set up cpu
  CPUTensor<float> input_cpu(input_shape_nchw);
  CPUTensor<size_t> max_index_cpu(output_shape_nkpq);
  CPUTensor<float> output_cpu(output_shape_nkpq);
  // set up mic
  MICTensor<float> input_mic(input_shape_nchw);
  MICTensor<size_t> max_index_mic(output_shape_nkpq);
  MICTensor<float> output_mic(output_shape_nkpq);
  // init values
  Backend<CPUTensor, float>::UniformDistributionFunc(&output_cpu, 0.0, 1.0);
  memcpy(output_mic.data(), output_cpu.data(), sizeof(float) * output_cpu.size());
  init_max_index(max_index_cpu.data(), input_shape_nchw[2] * input_shape_nchw[3], max_index_cpu.size());
  memcpy(max_index_mic.data(), max_index_cpu.data(), sizeof(size_t) * max_index_cpu.size());
  // cpu pooling 
  Backend<CPUTensor, float>::MaxPooling2DBackwardFunc(
    &output_cpu,
    &input_cpu,
    &max_index_cpu,
    filter_h, filter_w,
    str_h, str_w);
  // mic pooling
  Backend<MICTensor, float>::MaxPooling2DBackwardFunc(
    &output_mic,
    &input_mic,
    &max_index_mic,
    filter_h, filter_w,
    str_h, str_w);
  compare(input_cpu.data(), input_mic.data(), input_cpu.size());
}

void pooling_backward_nhwc(size_t filter_h, size_t filter_w, size_t str_h, size_t str_w) {
  // set up cpu
  CPUTensor<float> input_cpu(input_shape_nchw);
  CPUTensor<size_t> max_index_cpu(output_shape_nkpq);
  CPUTensor<float> output_cpu(output_shape_nkpq);
  // set up mic
  MICTensor<float> input_mic(input_shape_nhwc);
  MICTensor<float> input_mic_transform(input_shape_nchw);
  MICTensor<size_t> max_index_mic(output_shape_npqk);
  MICTensor<float> output_mic(output_shape_npqk);
  // init values
  Backend<CPUTensor, float>::UniformDistributionFunc(&output_cpu, 0.0, 1.0);
  nchw2nhwc(output_cpu.data(), output_mic.data(),
    output_shape_nkpq[0], output_shape_nkpq[1], output_shape_nkpq[2], output_shape_nkpq[3]);
  init_max_index(max_index_cpu.data(), input_shape_nchw[2] * input_shape_nchw[3], max_index_cpu.size());
  nchw2nhwc(max_index_cpu.data(), max_index_mic.data(),
    output_shape_nkpq[0], output_shape_nkpq[1], output_shape_nkpq[2], output_shape_nkpq[3]);
  // cpu pooling 
  Backend<CPUTensor, float>::MaxPooling2DBackwardFunc(
    &output_cpu,
    &input_cpu,
    &max_index_cpu,
    filter_h, filter_w,
    str_h, str_w);
  // mic pooling
  Backend<MICTensor, float>::MaxPooling2DBackwardFunc(
    &output_mic,
    &input_mic,
    &max_index_mic,
    filter_h, filter_w,
    str_h, str_w);
  nhwc2nchw(input_mic.data(), input_mic_transform.data(),
    input_shape_nchw[0], input_shape_nchw[1], input_shape_nchw[2], input_shape_nchw[3]);
  compare(input_cpu.data(), input_mic_transform.data(), input_cpu.size());
}

int main(int argc, char** argv) {
  const size_t NUM_ARGS = 14;
  // phase layout N C H W R S K P Q pad_h pad_w str_h str_w iteration
  if (argc != NUM_ARGS + 1) {
    std::cerr << "Not enough args!" << std::endl;
    exit(1);
  }
  // get args
  const std::string phase = std::string(argv[1]); 
  const std::string layout = std::string(argv[2]);
  const size_t N = atoi(argv[3]);
  const size_t C = atoi(argv[4]);
  const size_t H = atoi(argv[5]);
  const size_t W = atoi(argv[6]);
  const size_t R = atoi(argv[7]);
  const size_t S = atoi(argv[8]);
  const size_t K = atoi(argv[9]);
  const size_t P = atoi(argv[10]);
  const size_t Q = atoi(argv[11]);
  const size_t str_h = atoi(argv[12]);
  const size_t str_w = atoi(argv[13]);
  const size_t iterations = atoi(argv[14]);
  // set shapes
  set_input_shape_nchw(N, C, H, W);
  set_input_shape_nhwc(N, C, H, W);
  set_output_shape_nkpq(N, K, P, Q);
  set_output_shape_npqk(N, K, P, Q);
  std::cout << phase << std::endl;
  // run pooling
  for (size_t i = 0; i < iterations; ++i) {
    if (phase == "forward" && layout == "pooling_nchw") {
      pooling_forward_nchw(R, S, str_h, str_w);
    } else if (phase == "backward" && layout == "pooling_nchw") {
      pooling_backward_nchw(R, S, str_h, str_w);
    } else if (phase == "forward" && layout == "pooling_nhwc") {
      pooling_forward_nhwc(R, S, str_h, str_w);
    } else if (phase == "backward" && layout == "pooling_nhwc") {
      pooling_backward_nhwc(R, S, str_h, str_w);
    }
  }
  std::cout << "end" << std::endl;
  return 0;
}
