#include "kernels/xsmm_function.h"

namespace blitz {

scoped_ptr<XsmmLoadBuffer> Xsmm::instance_(0);
boost::once_flag Xsmm::flag_ = BOOST_ONCE_INIT;

template<>
void BlitzXsmmConvolution2D(
  float* I,
  float* O,
  float* F,
  size_t N,
  size_t C, size_t H, size_t W,
  size_t R, size_t S,
  size_t K, size_t P, size_t Q,
  size_t str_h, size_t str_w,
  size_t pad_h, size_t pad_w,
  size_t tid_batch, size_t tid, size_t num_threads,
  const string& phase) {
  libxsmm_dnn_conv_desc conv_desc;
  conv_desc.N = N;
  conv_desc.C = C;
  conv_desc.H = H;
  conv_desc.W = W;
  conv_desc.K = K;
  conv_desc.R = R; 
  conv_desc.S = S; 
  conv_desc.u = str_h;
  conv_desc.v = str_w;
  // because pad_h_in is not used, we use it to indicate tid_batch, whic is default 0
  conv_desc.pad_h_in = tid_batch;
  conv_desc.pad_w_in = 0;
  conv_desc.pad_h_out = pad_h;
  conv_desc.pad_w_out = pad_w;
  conv_desc.splits = 1;
  conv_desc.threads = num_threads;
  conv_desc.algo = LIBXSMM_DNN_CONV_ALGO_AUTO;
  conv_desc.buffer_format = LIBXSMM_DNN_CONV_FORMAT_LIBXSMM;
  conv_desc.filter_format = LIBXSMM_DNN_CONV_FORMAT_LIBXSMM;
  conv_desc.fuse_ops = LIBXSMM_DNN_CONV_FUSE_NONE;
  conv_desc.datatype = LIBXSMM_DNN_DATATYPE_F32;
  conv_desc.options = LIBXSMM_DNN_CONV_OPTION_NONE;
  // get handle, only adds once
  // TODO(keren) double check optimization
  #pragma omp critical
  {
    if (!Xsmm::HasBuffer(conv_desc)) { 
      Xsmm::AddBuffer(conv_desc);
    }
  }
  XsmmBuffer buffer = Xsmm::GetBuffer(conv_desc);
  // run convolution
  if (phase == "forward") {
    CHKERR_LIBXSMM_DNN(libxsmm_dnn_copyin_buffer(
      buffer.libxsmm_input, static_cast<void*>(I), LIBXSMM_DNN_CONV_FORMAT_NCHW));
    CHKERR_LIBXSMM_DNN(libxsmm_dnn_copyin_filter(
      buffer.libxsmm_filter, static_cast<void*>(F), LIBXSMM_DNN_CONV_FORMAT_KCRS));
    CHKERR_LIBXSMM_DNN(libxsmm_dnn_convolve_st(buffer.libxsmm_handle, LIBXSMM_DNN_CONV_KIND_FWD, 0, tid));
    CHKERR_LIBXSMM_DNN(libxsmm_dnn_copyout_buffer(
      buffer.libxsmm_output, static_cast<void*>(O), LIBXSMM_DNN_CONV_FORMAT_NCHW));
  } else if (phase == "backward") {
  } else if (phase == "update") {
  } else {
    LOG(FATAL) << "Phase: " << phase << " not exist";
  }
}

template<>
void BlitzXsmmConvolution2D(
  double* I,
  double* O,
  double* F,
  size_t N,
  size_t C, size_t H, size_t W,
  size_t R, size_t S,
  size_t K, size_t P, size_t Q,
  size_t str_h, size_t str_w,
  size_t pad_h, size_t pad_w,
  size_t tid_batch, size_t tid, size_t num_threads,
  const string& phase) {
  LOG(FATAL) << "xsmm kernel dost not support double precision";
}

}  // namespace blitz
