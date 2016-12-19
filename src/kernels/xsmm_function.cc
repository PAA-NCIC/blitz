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
    
#if 0
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
	const string& input_format,
	const string& output_format,
	const string& filter_format,
	const string& phase) {
    //create libxsmm_dnn_conv_desc
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
	conv_desc.pad_h_in = 0;
	conv_desc.pad_w_in = 0;
	conv_desc.pad_h_out = pad_h;
	conv_desc.pad_w_out = pad_w;

	conv_desc.threads = BLITZ_NUM_THREADS;
	conv_desc.algo = LIBXSMM_DNN_CONV_ALGO_AUTO;
    conv_desc.buffer_format = LIBXSMM_DNN_CONV_FORMAT_LIBXSMM;
    conv_desc.filter_format = LIBXSMM_DNN_CONV_FORMAT_LIBXSMM;
	conv_desc.fuse_ops = LIBXSMM_DNN_CONV_FUSE_NONE;
	conv_desc.options = LIBXSMM_DNN_CONV_OPTION_NONE;
    conv_desc.datatype_in = LIBXSMM_DNN_DATATYPE_F32;
    conv_desc.datatype_out = LIBXSMM_DNN_DATATYPE_F32;
    
	if (input_format == "nchw") {
		conv_desc.buffer_format = LIBXSMM_DNN_CONV_FORMAT_LIBXSMM;
	} else if (input_format == "nhwc") {
		conv_desc.buffer_format = LIBXSMM_DNN_CONV_FORMAT_NHWC;
	} else {
		LOG(FATAL) << "xsmm kernel does not support format: " << input_format;
	}
	if (filter_format == "kcrs") {
		conv_desc.filter_format = LIBXSMM_DNN_CONV_FORMAT_LIBXSMM;
	} else if (input_format == "rsck") {
		conv_desc.filter_format = LIBXSMM_DNN_CONV_FORMAT_RSCK;
	} else {
		LOG(FATAL) << "xsmm kernel does not support format: " << filter_format;
	}

	// get handle, only adds once
	if (!Xsmm::HasBuffer(conv_desc)) { 
		Xsmm::AddBuffer(conv_desc, I, O, F);
	}
	XsmmBuffer buffer = Xsmm::GetBuffer(conv_desc);

	// bind formats
	// input
	if (input_format == "nchw") {
		CHKERR_LIBXSMM_DNN(libxsmm_dnn_copyin_buffer(buffer.libxsmm_input, 
			static_cast<void*>(I), LIBXSMM_DNN_CONV_FORMAT_NCHW));
	}

	// kernel
	if (filter_format == "kcrs") {
		CHKERR_LIBXSMM_DNN(libxsmm_dnn_copyin_filter(buffer.libxsmm_filter,
			static_cast<void*>(F), LIBXSMM_DNN_CONV_FORMAT_KCRS));
	}

    //output
	CHKERR_LIBXSMM_DNN(libxsmm_dnn_zero_buffer(buffer.libxsmm_output));

	// run convolution
	if (phase == "forward") {
		#pragma omp parallel
		{
			CHKERR_LIBXSMM_DNN(libxsmm_dnn_convolve_st(buffer.libxsmm_handle,
				LIBXSMM_DNN_CONV_KIND_FWD, 0, omp_get_thread_num()));
		}
		// output copy
		if (output_format == "nkpq") {
			CHKERR_LIBXSMM_DNN(libxsmm_dnn_copyout_buffer(buffer.libxsmm_output,
				static_cast<void*>(O), LIBXSMM_DNN_CONV_FORMAT_NCHW));
		}
	} else if (phase == "backward") {
		#pragma omp parallel
		{
			CHKERR_LIBXSMM_DNN(libxsmm_dnn_convolve_st(buffer.libxsmm_handle,
				LIBXSMM_DNN_CONV_KIND_BWD, 0, omp_get_thread_num()));
		}
		// input copy
	if (input_format == "nchw") {
			CHKERR_LIBXSMM_DNN(libxsmm_dnn_copyout_buffer(buffer.libxsmm_input,
				static_cast<void*>(O), LIBXSMM_DNN_CONV_FORMAT_NCHW));
		}
	} else if (phase == "update") {
		#pragma omp parallel
		{
			CHKERR_LIBXSMM_DNN(libxsmm_dnn_convolve_st(buffer.libxsmm_handle,
				LIBXSMM_DNN_CONV_KIND_UPD, 0, omp_get_thread_num()));
		}
		// filter copy
		if (filter_format == "kcrs") {
			CHKERR_LIBXSMM_DNN(libxsmm_dnn_copyout_filter(buffer.libxsmm_filter,
				static_cast<void*>(F), LIBXSMM_DNN_CONV_FORMAT_KCRS));
		}
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
	const string& input_format,
	const string& output_format,
	const string& filter_format,
	const string& phase) {
	LOG(FATAL) << "xsmm kernel dost not support double precision";
}
#endif

}  // namespace blitz
