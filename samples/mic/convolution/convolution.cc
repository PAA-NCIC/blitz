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
//N H W C
Shape input_shape_nhwc(4);
//R S C K
Shape filter_shape_rsck(4);
//N H W C
Shape output_shape_nhwc(4);

#define EPSILON 0.001f
#define ACCESS(psrc, i, j, k, m, J, K, M) (*(psrc + m + k * M + j * K * M + i * J * K * M))

void zero_buf(float* buf, long size) {
  int i;
  for (i = 0; i < size; ++i) {
    buf[i] = 0.0f;
  }
}

void init_buf(float* buf, long size, int initPos, int initOne) {
  int i;
  zero_buf(buf, size);
  for (i = 0; i < size; ++i) {
    buf[i] = (float)((initOne != 0) ? 1.0 : ((initPos != 0) ? drand48() : (0.05 - drand48()/10.0)));
  }
}

void compare(float* algo1, float* algo2, size_t size) {
  size_t i = 0;
  for (i = 0; i < size; ++i) {
    if ((algo1[i] - algo2[i] > EPSILON )|| (algo1[i] - algo2[i] < -EPSILON)) {
      std::cout << "Index: " << i << " diff:" << algo1[i] - algo2[i] << std::endl;
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

void set_input_shape(size_t N, size_t C, size_t H, size_t W) {
  //set nchw
  input_shape[0] = N;
  input_shape[1] = C;
  input_shape[2] = H;
  input_shape[3] = W;
  input_shape.set_data_layout(BLITZ_BUFFER_NCHW);
  //set nhwc
  input_shape_nhwc[0] = N;
  input_shape_nhwc[1] = H;
  input_shape_nhwc[2] = W;
  input_shape_nhwc[3] = C;
  input_shape_nhwc.set_data_layout(BLITZ_BUFFER_NHWC);
}

void set_filter_shape(size_t K, size_t C, size_t R, size_t S) {
  //set kcrs
  filter_shape[0] = K;
  filter_shape[1] = C;
  filter_shape[2] = R;
  filter_shape[3] = S;
  filter_shape.set_data_layout(BLITZ_FILTER_KCRS);
  //set rsck
  filter_shape_rsck[0] = R;
  filter_shape_rsck[1] = S;
  filter_shape_rsck[2] = C;
  filter_shape_rsck[3] = K;
  filter_shape_rsck.set_data_layout(BLITZ_FILTER_RSCK);
}

void set_output_shape(size_t N, size_t K, size_t P, size_t Q) {
  //set nkpq
  output_shape[0] = N;
  output_shape[1] = K;
  output_shape[2] = P;
  output_shape[3] = Q;
  output_shape.set_data_layout(BLITZ_BUFFER_NCHW);
  //set npqk
  output_shape_nhwc[0] = N;
  output_shape_nhwc[1] = P;
  output_shape_nhwc[2] = Q;
  output_shape_nhwc[3] = K;
  output_shape_nhwc.set_data_layout(BLITZ_BUFFER_NHWC);
}

void copy_NCHW_to_NHWC(const float* nchw, float* nhwc, int N, int H, int W, int C) {
  int n, h, w, c;
  for ( n = 0; n < N; n++ ) {
    for ( h = 0; h < H; h++ ) {
      for ( w = 0; w < W; w++ ) {
        for ( c = 0; c < C; c++ ) {
          ACCESS(nhwc, n, h, w, c, H, W, C) = ACCESS(nchw, n, c, h, w, C, H, W);
        }
      }
    }
  }
}

void copy_NHWC_to_NCHW(const float* nhwc, float* nchw, int N, int H, int W, int C) {
  int n, h, w, c;
  for(n = 0; n < N; n++) {
    for(h = 0; h < H; h++){
      for(w = 0; w < W; w++){
        for(c = 0; c < C; c++){
          ACCESS(nchw, n, c, h, w, C, H, W) = ACCESS(nhwc, n, h, w, c, H, W, C);
        }
      }
    }
  }
}

void copy_KCRS_to_RSCK(const float *kcrs, float *rsck, int R, int S, int C, int K) {
  int r, s, c, k;
  for ( r = 0; r < R; r++ ) {
    for ( s = 0; s < S; s++ ) {
      for ( c = 0; c < C; c++ ) {
        for ( k = 0; k < K; k++ ) {
          ACCESS(rsck, r, s, c, k, S, C, K) = ACCESS(kcrs, k, c, r, s, C, R, S);
        }
      }
    }
  }
}

void copy_RSCK_to_KCRS(const float *rsck, float *kcrs, int R, int S, int C, int K) {
  int r, s, c, k;
  for ( r = 0; r < R; r++ ) {
    for ( s = 0; s < S; s++ ) {
      for ( c = 0; c < C; c++ ) {
        for ( k = 0; k < K; k++ ) {
          ACCESS(kcrs, k, c, r, s, C, R, S) = ACCESS(rsck, r, s, c, k, S, C, K); 
        }
      }
    }
  }
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
  // init values
  Backend<CPUTensor, float>::UniformDistributionFunc(&filter_cpu, 0.0, 1.0);
  Backend<CPUTensor, float>::UniformDistributionFunc(&input_cpu, 0.0, 1.0);
  //LIBXSMM example init
  // set up mic
  MICTensor<float> input_mic(input_shape_nhwc);
  MICTensor<float> filter_mic(filter_shape_rsck);
  MICTensor<float> output_mic(output_shape_nhwc);
  MICTensor<float> workspace_mic(workspace_shape_cpu);
  copy_KCRS_to_RSCK(filter_cpu.data(), filter_mic.data(), filter_shape_rsck[0], filter_shape_rsck[1], filter_shape_rsck[2], filter_shape_rsck[3]);
  copy_NCHW_to_NHWC(input_cpu.data(), input_mic.data(), input_shape_nhwc[0], input_shape_nhwc[1], input_shape_nhwc[2], input_shape_nhwc[3] );
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
  float *nchw = (float *)malloc(output_cpu.size() * sizeof(float));
  copy_NHWC_to_NCHW(output_mic.data(), nchw, output_shape_nhwc[0], output_shape_nhwc[1],output_shape_nhwc[2], output_shape_nhwc[3]);
  compare(output_cpu.data(), nchw, output_mic.size());
  free(nchw);
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
  // init values
  Backend<CPUTensor, float>::UniformDistributionFunc(&filter_cpu, 0.0, 1.0);
  Backend<CPUTensor, float>::UniformDistributionFunc(&output_cpu, 0.0, 1.0);
  // set up mic
  MICTensor<float> input_mic(input_shape_nhwc);
  MICTensor<float> filter_mic(filter_shape_rsck);
  MICTensor<float> output_mic(output_shape_nhwc);
  MICTensor<float> workspace_mic(workspace_shape_cpu);
  copy_KCRS_to_RSCK(filter_cpu.data(), filter_mic.data(), filter_shape_rsck[0], filter_shape_rsck[1], filter_shape_rsck[2], filter_shape_rsck[3]);
  copy_NCHW_to_NHWC(output_cpu.data(), output_mic.data(),output_shape_nhwc[0], output_shape_nhwc[1], output_shape_nhwc[2], output_shape_nhwc[3] );
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
  float *nchw = (float *)malloc(input_cpu.size() * sizeof(float));
  copy_NHWC_to_NCHW(input_mic.data(), nchw, input_shape_nhwc[0], input_shape_nhwc[1],input_shape_nhwc[2], input_shape_nhwc[3]);
  compare(input_cpu.data(), nchw, input_mic.size());
  free(nchw);
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
  // init values
  Backend<CPUTensor, float>::UniformDistributionFunc(&input_cpu, 0.0, 1.0);
  Backend<CPUTensor, float>::UniformDistributionFunc(&output_cpu, 0.0, 1.0);

  // set up mic
  MICTensor<float> input_mic(input_shape_nhwc);
  MICTensor<float> filter_mic(filter_shape_rsck);
  MICTensor<float> output_mic(output_shape_nhwc);
  MICTensor<float> workspace_mic(workspace_shape_cpu);
  copy_NCHW_to_NHWC(input_cpu.data(), input_mic.data(),input_shape_nhwc[0], input_shape_nhwc[1], input_shape_nhwc[2], input_shape_nhwc[3] );
  copy_NCHW_to_NHWC(output_cpu.data(), output_mic.data(),output_shape_nhwc[0], output_shape_nhwc[1], output_shape_nhwc[2], output_shape_nhwc[3] );
  
  // cpu convolution 
  Backend<CPUTensor, float>::Convolution2DUpdateFunc(
    &input_cpu,
    &output_cpu,
    &filter_cpu,
    &workspace_cpu,
    pad_h, pad_w, 
    str_h, str_w,
    algorithm);
  std::cout << "cpu upd finished" << std::endl;

  // mic convolution
  Backend<MICTensor, float>::Convolution2DUpdateFunc(
    &input_mic,
    &output_mic,
    &filter_mic,
    &workspace_mic,
    pad_h, pad_w, 
    str_h, str_w,
    BLITZ_CONVOLUTION_XSMM_DIRECT);
  std::cout << "mic upd finished" << std::endl;

  float *kcrs = (float *)malloc(filter_cpu.size() * sizeof(float));
  copy_RSCK_to_KCRS(filter_mic.data(), kcrs, filter_shape_rsck[0], filter_shape_rsck[1],filter_shape_rsck[2], filter_shape_rsck[3]);
  compare(filter_cpu.data(), kcrs, filter_mic.size());
  free(kcrs);
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
  set_input_shape(N, C, H, W);
  set_filter_shape(K, C, R, S);
  set_output_shape(N, K, P, Q);
  // set workspace shape
  //  workspace_shape_cpu[0] = BLITZ_NUM_THREADS * C * R * S * P * Q;
  workspace_shape_cpu[0] = C * R * S * P * Q;
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
