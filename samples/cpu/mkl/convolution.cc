#include <glog/logging.h>
#include "sys/time.h"
#include "blitz_mkl.h"

const size_t dimension = 4;
dnnLayout_t lt_user_input = NULL, lt_user_filt = NULL, lt_user_output = NULL;
dnnLayout_t lt_conv_input = NULL, lt_conv_filt = NULL, lt_conv_output = NULL;
dnnPrimitive_t conv = NULL;
dnnPrimitive_t cv_user_to_conv_input = NULL, cv_user_to_conv_filt = NULL;
dnnPrimitiveAttributes_t attributes = NULL;
dnnError_t err;
float *user_i = NULL, *user_f = NULL, *user_o = NULL;
float* res_conv[dnnResourceNumber] = {0};
double computations = 0;

void convolution_forward(
  size_t *input_size, size_t *input_strides,
  size_t *filter_size, size_t *filter_strides,
  size_t *output_size, size_t *output_strides,
  size_t *strides, int *input_offset, size_t iters) {
  MKL_CHECK_ERR(dnnConvolutionCreateForward_F32(&conv, attributes, dnnAlgorithmConvolutionDirect,
    dimension, input_size, output_size, filter_size, strides, input_offset, dnnBorderZeros), err);
  MKL_CHECK_ERR(dnnLayoutCreateFromPrimitive_F32(&lt_conv_input, conv, dnnResourceSrc) , err );
  MKL_CHECK_ERR(dnnLayoutCreateFromPrimitive_F32(&lt_conv_filt, conv, dnnResourceFilter), err );
  MKL_CHECK_ERR(dnnLayoutCreateFromPrimitive_F32(&lt_conv_output, conv, dnnResourceDst) , err );
  MKL_CHECK_ERR(init_conversion(&cv_user_to_conv_input, &res_conv[dnnResourceSrc],
    lt_conv_input, lt_user_input, user_i) , err);
  MKL_CHECK_ERR(init_conversion(&cv_user_to_conv_filt, &res_conv[dnnResourceFilter],
    lt_conv_filt, lt_user_filt, user_f), err);
  MKL_CHECK_ERR(dnnAllocateBuffer_F32((void**)&res_conv[dnnResourceDst], lt_conv_output), err );
  if (cv_user_to_conv_filt)
    MKL_CHECK_ERR(dnnConversionExecute_F32(cv_user_to_conv_filt, user_f, res_conv[dnnResourceFilter]), err );

  struct timeval t1, t2;
  double elapse;
  for (size_t i = 0; i < iters; ++i) {
    BLITZ_CPU_TIMER_START(elapse, t1);
    if (cv_user_to_conv_input)
      MKL_CHECK_ERR(dnnConversionExecute_F32(cv_user_to_conv_input, user_i, res_conv[dnnResourceSrc]), err );
    MKL_CHECK_ERR(dnnExecute_F32(conv, (void**)res_conv), err);
    BLITZ_CPU_TIMER_END(elapse, t1, t2);
    BLITZ_CPU_TIMER_INFO(computations, elapse);
  }

  dnnDelete_F32(conv);
  dnnDelete_F32(cv_user_to_conv_input);
  dnnDelete_F32(cv_user_to_conv_filt);
  dnnLayoutDelete_F32(lt_conv_input);
  dnnLayoutDelete_F32(lt_conv_filt);
  dnnLayoutDelete_F32(lt_conv_output);
  dnnPrimitiveAttributesDestroy_F32(attributes);
  if (res_conv[dnnResourceSrc] != (void*)user_i) dnnReleaseBuffer_F32(res_conv[dnnResourceSrc]);
  if (res_conv[dnnResourceFilter] != (void*)user_f) dnnReleaseBuffer_F32(res_conv[dnnResourceFilter]);
  dnnReleaseBuffer_F32(res_conv[dnnResourceDst]);
}

int main(int argc, char **argv) {
  const size_t NUM_ARGS = 15;
  // phase kernel N C H W R S K P Q pad_h pad_w str_h str_w iter
  if (argc != NUM_ARGS + 1) {
    LOG(FATAL) << "Not matchable args!";
  }
  FLAGS_logtostderr = true;
  google::InitGoogleLogging(argv[0]);
  // get args
  const std::string phase = std::string(argv[1]); 
  const size_t N = atoi(argv[2]);
  const size_t C = atoi(argv[3]);
  const size_t H = atoi(argv[4]);
  const size_t W = atoi(argv[5]);
  const size_t R = atoi(argv[6]);
  const size_t S = atoi(argv[7]);
  const size_t K = atoi(argv[8]);
  const size_t P = atoi(argv[9]);
  const size_t Q = atoi(argv[10]);
  const size_t pad_h = atoi(argv[11]);
  const size_t pad_w = atoi(argv[12]);
  const size_t str_h = atoi(argv[13]);
  const size_t str_w = atoi(argv[14]);
  const size_t iters = atoi(argv[15]);

  size_t input_size[dimension] = {W, H, C, N};
  size_t input_strides[dimension] = {1, W, H * W, C * W * H};

  size_t output_size[dimension] = {Q, P, K, N};
  size_t output_strides[dimension] = {1, Q, Q * P, K * Q * P};

  size_t filter_size[dimension] = {S, R, C, K};
  size_t filter_strides[dimension] = {1, S, S * R, S * R * C};

  size_t strides[dimension - 2] = {str_h, str_w};
  int input_offset[dimension - 2] = {-static_cast<int>(pad_h), -static_cast<int>(pad_w)};

  computations = 2 * static_cast<double>(N * K * P * Q) * static_cast<double>(C * R * S);
  
  user_i = (float*)malloc(sizeof(float)*(N * C * H * W));
  user_f = (float*)malloc(sizeof(float)*(K * C * R * S));
  user_o = (float*)malloc(sizeof(float)*(N * K * P * Q));

  MKL_CHECK_ERR(dnnLayoutCreate_F32(&lt_user_input, dimension, input_size, input_strides) , err);
  MKL_CHECK_ERR(dnnLayoutCreate_F32(&lt_user_filt, dimension, filter_size, filter_strides), err);
  MKL_CHECK_ERR(dnnLayoutCreate_F32(&lt_user_output, dimension, output_size, output_strides), err);
  MKL_CHECK_ERR(dnnPrimitiveAttributesCreate_F32(&attributes), err);

  convolution_forward(input_size, input_strides,
    filter_size, filter_strides,
    output_size, output_strides,
    strides, input_offset,
    iters);

  dnnPrimitiveAttributesDestroy_F32(attributes);
  dnnLayoutDelete_F32(lt_user_input);
  dnnLayoutDelete_F32(lt_user_filt);
  dnnLayoutDelete_F32(lt_user_output);

  return 0;
}
