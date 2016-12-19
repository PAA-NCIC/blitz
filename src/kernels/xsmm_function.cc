#include "kernels/xsmm_function.h"

namespace blitz {

scoped_ptr<XsmmLoadBuffer> Xsmm::instance_(0);
boost::once_flag Xsmm::flag_ = BOOST_ONCE_INIT;


template<>
XsmmBuffer BlitzXsmmPrepare2D(
    float* input,
    float* output,
    float* filter,
    const Shape& input_shape,
    const Shape& filter_shape,
    const Shape& output_shape,
    size_t stride_h, size_t stride_w,
    size_t padding_h, size_t padding_w){
    libxsmm_dnn_conv_desc conv_desc;
    libxsmm_dnn_err_t status;
#if defined(_OPENMP)
    int nThreads = omp_get_max_threads();
#else
    int nThreads = 1;
#endif
       //setup libxsmm handle
    if(input_shape.data_layout() == BLITZ_BUFFER_NCHW) {
       conv_desc.buffer_format = LIBXSMM_DNN_CONV_FORMAT_LIBXSMM;
       conv_desc.N = input_shape[0];
       conv_desc.C = input_shape[1];
       conv_desc.H = input_shape[2];
       conv_desc.W = input_shape[3];
       conv_desc.K = output_shape[1];
       if(filter_shape.data_layout() == BLITZ_FILTER_KCRS) {
           conv_desc.filter_format = LIBXSMM_DNN_CONV_FORMAT_LIBXSMM;
           conv_desc.R = filter_shape[2];
           conv_desc.S = filter_shape[3];
       } else if(filter_shape.data_layout() == BLITZ_FILTER_RSCK) {
            LOG(FATAL) << "xsmm kernel does not support LIBXSMM--RSCK convolution";
//           conv_desc.filter_format = LIBXSMM_DNN_CONV_FORMAT_RSCK;
//           conv_desc.R = filter_shape[0];
//           conv_desc.S = filter_shape[1];
       } else {
            //@TODO filter in other format
       }
    } else if(input_shape.data_layout() == BLITZ_BUFFER_NHWC) {
        conv_desc.buffer_format = LIBXSMM_DNN_CONV_FORMAT_NHWC;
        conv_desc.N = input_shape[0];
        conv_desc.C = input_shape[3];
        conv_desc.H = input_shape[1];
        conv_desc.W = input_shape[2];
        conv_desc.K = output_shape[3];
        if(filter_shape.data_layout() == BLITZ_FILTER_KCRS) {
           conv_desc.filter_format = LIBXSMM_DNN_CONV_FORMAT_LIBXSMM;
           conv_desc.R = filter_shape[2];
           conv_desc.S = filter_shape[3];
        } else if(filter_shape.data_layout() == BLITZ_FILTER_RSCK) {
            conv_desc.filter_format = LIBXSMM_DNN_CONV_FORMAT_RSCK;
            conv_desc.R = filter_shape[0];
            conv_desc.S = filter_shape[1];
        } else {
            //@TODO filter in other format
        }
    } else {
        //@TODO input in other format
    }
    conv_desc.u = stride_h;
    conv_desc.v = stride_w;
    conv_desc.pad_h_in = 0;
    conv_desc.pad_w_in = 0;
    conv_desc.pad_h_out = padding_h;
    conv_desc.pad_w_out = padding_w;
    conv_desc.threads = nThreads;
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
    const Shape& input_shape,
    const Shape& filter_shape,
    const Shape& output_shape,
    size_t stride_h, size_t stride_w,
    size_t padding_h, size_t padding_w){
	LOG(FATAL) << "xsmm kernel dost not support double precision";
}
    
}  // namespace blitz
