#include <iostream>
#include "backends/backends.h"
#include<fstream>
#include "utils/blitz_shape_function.h"
#include "utils/blitz_algorithm_function.h"

using namespace blitz;
using std::cout;
using std::endl;
using std::scientific;

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

#define EPSILON 0.000001f
#define ACCESS(psrc, i, j, k, m, J, K, M) (*(psrc + (m) + (k) * M + (j) * K * M + (i) * J * K * M))

void zero_buf(float* buf, long size) {
  int i;
  for (i = 0; i < size; ++i) {
    buf[i] = 0.0;
  }
}
void zero_buf(double* buf, long size) {
  int i;
  for (i = 0; i < size; ++i) {
    buf[i] = 0.0;
  }
}

void init_buf(float* buf, long size, int initPos, int initOne) {
  int i;
  zero_buf(buf, size);
  for (i = 0; i < size; ++i) {
    buf[i] = (float)((initOne != 0) ? 1.0 : ((initPos != 0) ? drand48() : (0.05 - drand48()/10.0)));
  }
}
void init_buf(double* buf, long size, int initPos, int initOne)
{
  int i;
  zero_buf(buf, size);
  for (i = 0; i < size; ++i) {
    buf[i] = (double)((initOne != 0) ? 1.0 : ((initPos != 0) ? drand48() : (0.05 - drand48()/10.0)));
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
void compare(double* algo1, double* algo2, size_t size) {
    size_t i = 0;
    cout.precision(2);
	for (i = 0; i < size; ++i) {
		if ((algo1[i] - algo2[i] > EPSILON )|| (algo1[i] - algo2[i] < -EPSILON)) {
            cout << "Index: " << i << " " << scientific << algo1[i] - algo2[i] << endl;
		}
	}
}

void compareThree(float *algo1, double *benchmark, float *algo2, size_t size)
{
    size_t i = 0;
    float diff1, diff2;
    cout.precision(2);
    for(i = 0; i < size; i++){
       diff1 = algo1[i] - benchmark[i];
       diff2 = algo2[i] - benchmark[i];
       if((diff1 - diff2 > EPSILON) || (diff1 - diff2 < -EPSILON)){
           cout << "Index: " << i << " " << scientific << diff1 << " " << scientific << diff2 << endl;
       }
    }
}

void cpy(float *desc, float *src, size_t size)
{
    size_t i = 0;
    for(i = 0; i < size; ++i){
        desc[i] = src[i];
    }
}
void cpy(double *desc, float *src, size_t size)
{
    size_t i = 0;
    for(i = 0; i < size; ++i){
        desc[i] = static_cast<double>(src[i]);
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

typedef struct {
  int nImg;
  int nIfm;
  int nOfm;
  int ifhp;
  int ifwp;
  int ofhp;
  int ofwp;
  int ofh;
  int ofw;
  int pad_h_in;
  int pad_w_in;
  int pad_h_out;
  int pad_w_out;
  int kh;
  int kw;
  int stride_h;
  int stride_w;
} naive_conv_t;

void naive_conv_fp(naive_conv_t* param, const float* input, float* output, const float* filter)
{
  int nImg      = param->nImg;
  int nIfm      = param->nIfm;
  int nOfm      = param->nOfm;
  int ifhp      = param->ifhp;
  int ifwp      = param->ifwp;
  int ofhp      = param->ofhp;
  int ofwp      = param->ofwp;
  int ofh       = param->ofh;
  int ofw       = param->ofw;
  int pad_h_out = param->pad_h_out;
  int pad_w_out = param->pad_w_out;
  int kh        = param->kh;
  int kw        = param->kw;
  int stride_h  = param->stride_h;
  int stride_w  = param->stride_w;
  /* loop counters */
  int img, ofm, ifm, oj, oi, ij, ii, kj, ki;

#if defined(_OPENMP)
# pragma omp parallel for collapse(2) private(img, ofm, ifm, oj, oi, ij, ii, kj, ki)
#endif
  for (img = 0; img < nImg; ++img) {
    for (ofm = 0; ofm < nOfm; ++ofm) {
      for (ifm = 0; ifm < nIfm; ++ifm) {
        for (oj = 0; oj < ofh; ++oj) {
          ij = oj * stride_h;
          for (oi = 0; oi < ofw; ++oi) {
            ii = oi * stride_w;
            for (kj = 0; kj < kh; ++kj) {
              for (ki = 0; ki < kw; ++ki) {
                ACCESS(output, img, ofm, oj, oi, nOfm, ofhp, ofwp) +=
                  ACCESS(input, img, ifm, ij + kj, ii + ki, nIfm, ifhp, ifwp)
                  * ACCESS(filter, ofm, ifm, kj, ki, nIfm, kh, kw);
              }
            }
          }
        }
      }
    }
  }
}
void naive_conv_fp(naive_conv_t* param, const double* input, double* output, const double* filter)
{
  int nImg      = param->nImg;
  int nIfm      = param->nIfm;
  int nOfm      = param->nOfm;
  int ifhp      = param->ifhp;
  int ifwp      = param->ifwp;
  int ofhp      = param->ofhp;
  int ofwp      = param->ofwp;
  int ofh       = param->ofh;
  int ofw       = param->ofw;
  int pad_h_out = param->pad_h_out;
  int pad_w_out = param->pad_w_out;
  int kh        = param->kh;
  int kw        = param->kw;
  int stride_h  = param->stride_h;
  int stride_w  = param->stride_w;
  /* loop counters */
  int img, ofm, ifm, oj, oi, ij, ii, kj, ki;

#if defined(_OPENMP)
# pragma omp parallel for collapse(2) private(img, ofm, ifm, oj, oi, ij, ii, kj, ki)
#endif
  for (img = 0; img < nImg; ++img) {
    for (ofm = 0; ofm < nOfm; ++ofm) {
      for (ifm = 0; ifm < nIfm; ++ifm) {
        for (oj = 0; oj < ofh; ++oj) {
          ij = oj * stride_h;
          for (oi = 0; oi < ofw; ++oi) {
            ii = oi * stride_w;
            for (kj = 0; kj < kh; ++kj) {
              for (ki = 0; ki < kw; ++ki) {
                ACCESS(output, img, ofm, oj, oi, nOfm, ofhp, ofwp) +=
                  ACCESS(input, img, ifm, ij + kj, ii + ki, nIfm, ifhp, ifwp)
                  * ACCESS(filter, ofm, ifm, kj, ki, nIfm, kh, kw);
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
  size_t str_h, size_t str_w, naive_conv_t *naive_param) {
  // set up cpu
  CPUTensor<float> input_cpu(input_shape);
  CPUTensor<float> filter_cpu(filter_shape);
  CPUTensor<float> output_cpu(output_shape);
  CPUTensor<float> workspace_cpu(workspace_shape_cpu);

  // init values
  Backend<CPUTensor, float>::UniformDistributionFunc(&filter_cpu, 0.0, 1.0);
  Backend<CPUTensor, float>::UniformDistributionFunc(&input_cpu, 0.0, 1.0);
  //LIBXSMM example init
//  init_buf(filter_cpu.data(), filter_cpu.size(), 0, 0);
//  init_buf(input_cpu.data(), input_cpu.size(), 0, 0);
  // set up mic
  MICTensor<float> input_mic(input_shape_nhwc);
  MICTensor<float> filter_mic(filter_shape_rsck);
  MICTensor<float> output_mic(output_shape_nhwc);
  MICTensor<float> workspace_mic(workspace_shape_cpu);
  copy_KCRS_to_RSCK(filter_cpu.data(), filter_mic.data(), filter_shape_rsck[0], filter_shape_rsck[1], filter_shape_rsck[2], filter_shape_rsck[3]);
  copy_NCHW_to_NHWC(input_cpu.data(), input_mic.data(), input_shape_nhwc[0], input_shape_nhwc[1], input_shape_nhwc[2], input_shape_nhwc[3] );

  //run naive convolution
  float *naive_input = (float *)malloc(input_cpu.size() * sizeof(float));
  float *naive_filter = (float *)malloc(filter_cpu.size() * sizeof(float));
  float *naive_output = (float *)malloc(output_cpu.size() * sizeof(float));
  cpy(naive_input, input_cpu.data(), input_cpu.size());
  cpy(naive_filter, filter_cpu.data(), filter_cpu.size());
  zero_buf(naive_output, output_cpu.size());
  naive_conv_fp(naive_param, naive_input, naive_output, naive_filter);

  // cpu convolution 
  Backend<CPUTensor, float>::Convolution2DForwardFunc(
    &input_cpu,
    &filter_cpu,
    &output_cpu,
    &workspace_cpu,
    pad_h, pad_w, 
    str_h, str_w,
    algorithm);
  cout << "cpu fwd finished" << endl;

  // mic convolution
  Backend<MICTensor, float>::Convolution2DForwardFunc(
    &input_mic,
    &filter_mic,
    &output_mic,
    &workspace_mic,
    pad_h, pad_w, 
    str_h, str_w,
    BLITZ_CONVOLUTION_XSMM_DIRECT);
    cout << "mic fwd finished" << endl;
  float *nchw = (float *)malloc(output_cpu.size() * sizeof(float));
  copy_NHWC_to_NCHW(output_mic.data(), nchw, output_shape_nhwc[0], output_shape_nhwc[1],output_shape_nhwc[2], output_shape_nhwc[3]);

  compare(naive_output, output_cpu.data(), output_cpu.size());

  free(nchw);
  free(naive_input);
  free(naive_output);
  free(naive_filter);
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
//	memcpy(filter_mic.data(), filter_cpu.data(), sizeof(float) * filter_cpu.size());
//  	memcpy(output_mic.data(), output_cpu.data(), sizeof(float) * output_cpu.size());
//    cout << "input and output difference" << endl;
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
  cout << "cpu bwd finished" << endl;

  // mic convolution
  Backend<MICTensor, float>::Convolution2DBackwardFunc(
    &output_mic,
    &filter_mic,
    &input_mic,
    &workspace_mic,
    pad_h, pad_w, 
    str_h, str_w,
    BLITZ_CONVOLUTION_XSMM_DIRECT);
  cout << "mic bwd finished" << endl;

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
  cout << "cpu upd finished" << endl;
  // mic convolution
  Backend<MICTensor, float>::Convolution2DUpdateFunc(
    &input_mic,
    &output_mic,
    &filter_mic,
    &workspace_mic,
    pad_h, pad_w, 
    str_h, str_w,
    BLITZ_CONVOLUTION_XSMM_DIRECT);
  cout << "mic upd finished" << endl;

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
  //set struct for naive convolution
  naive_conv_t naive_param;
  naive_param.nImg = N;
  naive_param.nIfm = C;
  naive_param.nOfm = K;
  naive_param.ifhp = H;
  naive_param.ifwp = W;
  naive_param.ofhp = P;
  naive_param.ofwp = Q;
  naive_param.ofh = P;
  naive_param.ofw = Q;
  naive_param.pad_h_in = 0;
  naive_param.pad_w_in = 0;
  naive_param.pad_h_out = pad_h;
  naive_param.pad_w_out = pad_w;
  naive_param.kh = R;
  naive_param.kw = S;
  naive_param.stride_h = str_h;
  naive_param.stride_w = str_w;

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
