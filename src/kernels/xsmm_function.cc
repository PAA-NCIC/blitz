#include "kernels/xsmm_function.h"

namespace blitz {

scoped_ptr<XsmmLoadBuffer> Xsmm::instance_(0);
boost::once_flag Xsmm::flag_ = BOOST_ONCE_INIT;


template<>
XsmmBuffer BlitzXsmmPrepare2D(
    float* input,
    float* output,
    float* filter,
    BLITZ_DATA_LAYOUT buffer_layout,
    BLITZ_DATA_LAYOUT filter_layout,
    const size_t N, const size_t H, const size_t W,
    const size_t C, const size_t K, const size_t R,
    const size_t S, 
    const size_t stride_h, const size_t stride_w,
    const size_t padding_h, const size_t padding_w){
    libxsmm_dnn_conv_desc conv_desc;
    libxsmm_dnn_err_t status;
#if defined(_OPENMP)
    int num_threads = omp_get_num_threads();
#else
    int num_threads = 1;
#endif
       //setup libxsmm handle
    if(buffer_layout == BLITZ_BUFFER_NCHW) {
         conv_desc.buffer_format = LIBXSMM_DNN_CONV_FORMAT_LIBXSMM;
         if(filter_layout == BLITZ_FILTER_KCRS) {
             conv_desc.filter_format = LIBXSMM_DNN_CONV_FORMAT_LIBXSMM;
         } else if(filter_layout == BLITZ_FILTER_RSCK) {
              LOG(FATAL) << "xsmm kernel does not support LIBXSMM--RSCK convolution";
//             conv_desc.filter_format = LIBXSMM_DNN_CONV_FORMAT_RSCK;
         } else {
              //@TODO filter in other format
            LOG(FATAL) << "xsmm kernel does not support undefined format convolution";
         }
    } else if(buffer_layout == BLITZ_BUFFER_NHWC) {
        conv_desc.buffer_format = LIBXSMM_DNN_CONV_FORMAT_NHWC;
        if(filter_layout == BLITZ_FILTER_KCRS) {
            conv_desc.filter_format = LIBXSMM_DNN_CONV_FORMAT_LIBXSMM;
        } else if(filter_layout == BLITZ_FILTER_RSCK) {
            conv_desc.filter_format = LIBXSMM_DNN_CONV_FORMAT_RSCK;
        } else {
            //@TODO filter in other format
            LOG(FATAL) << "xsmm kernel does not support undefined format convolution";
        }
    } else {
        //@TODO input in other format
    }
    conv_desc.N = N;
    conv_desc.C = C;
    conv_desc.H = H;
    conv_desc.W = W;
    conv_desc.K = K;
    conv_desc.R = R;
    conv_desc.S = S;
    conv_desc.u = stride_h;
    conv_desc.v = stride_w;
    conv_desc.pad_h_in = 0;
    conv_desc.pad_w_in = 0;
    conv_desc.pad_h_out = padding_h;
    conv_desc.pad_w_out = padding_w;
    conv_desc.threads = num_threads;
    conv_desc.algo = LIBXSMM_DNN_CONV_ALGO_AUTO;
    conv_desc.fuse_ops = LIBXSMM_DNN_CONV_FUSE_NONE;
    conv_desc.options = LIBXSMM_DNN_CONV_OPTION_NONE;
    conv_desc.datatype_in = LIBXSMM_DNN_DATATYPE_F32;
    conv_desc.datatype_out = LIBXSMM_DNN_DATATYPE_F32;
   
    //a handle add only once
   if(!Xsmm::HasBuffer(conv_desc)) {
       Xsmm::AddBuffer(conv_desc, static_cast<void *>(input), static_cast<void *>(output), static_cast<void *>(filter));
   }
    //return buffer
    return Xsmm::GetBuffer(conv_desc);
}

template<>
XsmmBuffer BlitzXsmmPrepare2D(
    double* input,
    double* output,
    double* filter,
    BLITZ_DATA_LAYOUT buffer_layout,
    BLITZ_DATA_LAYOUT filter_layout,
    const size_t N, const size_t H, const size_t W,
    const size_t C, const size_t K, const size_t R,
    const size_t S,
    const size_t stride_h, const size_t stride_w,
    const size_t padding_h, const size_t padding_w){
	LOG(FATAL) << "xsmm kernel dost not support double precision";
}
}  // namespace blitz
