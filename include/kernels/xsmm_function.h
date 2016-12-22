#ifndef INCLUDE_KERNELS_XSMM_FUNCTION_H_
#define INCLUDE_KERNELS_XSMM_FUNCTION_H_

#include <libxsmm.h>
#include <omp.h>

#include <boost/thread/once.hpp>
#include <boost/noncopyable.hpp>

#include "backends/shape.h"
#include "utils/common.h"

#include "../backends/shape.h"
namespace blitz {

#define CHKERR_LIBXSMM_DNN(A) if (A != LIBXSMM_DNN_SUCCESS ) std::cerr << libxsmm_dnn_get_error(A);


typedef struct {
  libxsmm_dnn_conv_handle* libxsmm_handle;
  libxsmm_dnn_buffer* libxsmm_input;
  libxsmm_dnn_buffer* libxsmm_output;
  libxsmm_dnn_filter* libxsmm_filter;
} XsmmBuffer;

class XsmmLoadBuffer {
 public:
  XsmmLoadBuffer() {}

  ~XsmmLoadBuffer() {
    typedef map<string, XsmmBuffer>::iterator BufferIterator;
    for (BufferIterator it = buffers_.begin(); it != buffers_.end(); ++it) {
      libxsmm_dnn_destroy_buffer((it->second).libxsmm_input);
      libxsmm_dnn_destroy_buffer((it->second).libxsmm_output);
      libxsmm_dnn_destroy_filter((it->second).libxsmm_filter);
      libxsmm_dnn_destroy_conv_handle((it->second).libxsmm_handle);
    }
  }

  void AddBuffer(libxsmm_dnn_conv_desc conv_desc, void* input, void* output, void* filter) {
    libxsmm_dnn_err_t status;
    //create handle
    libxsmm_dnn_conv_handle *libxsmm_handle = libxsmm_dnn_create_conv_handle_check(conv_desc, &status);
    //create buffer
    libxsmm_dnn_buffer* libxsmm_input;
    libxsmm_dnn_buffer* libxsmm_output;
    libxsmm_dnn_filter* libxsmm_filter;
    //input
    if(conv_desc.buffer_format == LIBXSMM_DNN_CONV_FORMAT_LIBXSMM) {
      libxsmm_input = libxsmm_dnn_create_input_buffer_check(libxsmm_handle, &status);
      CHKERR_LIBXSMM_DNN(status);
      CHKERR_LIBXSMM_DNN(libxsmm_dnn_copyin_buffer(libxsmm_input, input, LIBXSMM_DNN_CONV_FORMAT_NCHW));
    } else if(conv_desc.buffer_format == LIBXSMM_DNN_CONV_FORMAT_NHWC) {
      libxsmm_input = libxsmm_dnn_link_input_buffer_check(libxsmm_handle, input, LIBXSMM_DNN_CONV_FORMAT_NHWC_PTR, &status);
      CHKERR_LIBXSMM_DNN(status);
    } else {
      //@TODO input in other format
    }
    //filter
    if(conv_desc.filter_format == LIBXSMM_DNN_CONV_FORMAT_LIBXSMM) {
      libxsmm_filter = libxsmm_dnn_create_filter_check(libxsmm_handle, &status);
      CHKERR_LIBXSMM_DNN(status);
      CHKERR_LIBXSMM_DNN(libxsmm_dnn_copyin_filter(libxsmm_filter, filter, LIBXSMM_DNN_CONV_FORMAT_KCRS));
    } else if(conv_desc.filter_format == LIBXSMM_DNN_CONV_FORMAT_RSCK) {
      libxsmm_filter = libxsmm_dnn_link_filter_check(libxsmm_handle, filter, LIBXSMM_DNN_CONV_FORMAT_RSCK_PTR, &status);
      CHKERR_LIBXSMM_DNN(status);
    } else {
      //@TODO filter in other format
    }
    //output
    if(conv_desc.buffer_format == LIBXSMM_DNN_CONV_FORMAT_LIBXSMM) {
      libxsmm_output = libxsmm_dnn_create_output_buffer_check(libxsmm_handle, &status);
      CHKERR_LIBXSMM_DNN(status);
      CHKERR_LIBXSMM_DNN(libxsmm_dnn_zero_buffer(libxsmm_output));
    } else if (conv_desc.buffer_format == LIBXSMM_DNN_CONV_FORMAT_NHWC) {
      libxsmm_output = libxsmm_dnn_link_output_buffer_check(libxsmm_handle, output, LIBXSMM_DNN_CONV_FORMAT_NHWC_PTR, &status);
      CHKERR_LIBXSMM_DNN(status);
    } else {
      //@TODO output in other format
    }
    // bind buffers
    CHKERR_LIBXSMM_DNN(libxsmm_dnn_bind_input_buffer(libxsmm_handle, libxsmm_input));
    CHKERR_LIBXSMM_DNN(libxsmm_dnn_bind_output_buffer(libxsmm_handle, libxsmm_output));
    CHKERR_LIBXSMM_DNN(libxsmm_dnn_bind_filter(libxsmm_handle, libxsmm_filter));
    //add XsmmBuffer 
    XsmmBuffer buffer;
    buffer.libxsmm_handle = libxsmm_handle;
    buffer.libxsmm_input = libxsmm_input;
    buffer.libxsmm_output = libxsmm_output;
    buffer.libxsmm_filter = libxsmm_filter;
    std::stringstream ss;
    ss << conv_desc.N << "_" << conv_desc.C << "_" << conv_desc.H <<
      "_" << conv_desc.W << "_" << conv_desc.K << "_" << conv_desc.R <<
      "_" << conv_desc.S << "_" << conv_desc.u << "_" << conv_desc.v <<
      "_" << conv_desc.pad_h_out << "_" << conv_desc.pad_w_out <<
      "_" << conv_desc.buffer_format << "_" << conv_desc.filter_format;
    buffers_[ss.str()] = buffer;
  }

  bool HasBuffer(libxsmm_dnn_conv_desc conv_desc) {
    std::stringstream ss;
    ss << conv_desc.N << "_" << conv_desc.C << "_" << conv_desc.H <<
      "_" << conv_desc.W << "_" << conv_desc.K << "_" << conv_desc.R <<
      "_" << conv_desc.S << "_" << conv_desc.u << "_" << conv_desc.v <<
      "_" << conv_desc.pad_h_out << "_" << conv_desc.pad_w_out <<
      "_" << conv_desc.buffer_format << "_" << conv_desc.filter_format;
    if (buffers_.find(ss.str()) == buffers_.end()) {
      return false;
    }   
    return true;
  }

  XsmmBuffer GetBuffer(libxsmm_dnn_conv_desc conv_desc) {
    std::stringstream ss;
    ss << conv_desc.N << "_" << conv_desc.C << "_" << conv_desc.H <<
      "_" << conv_desc.W << "_" << conv_desc.K << "_" << conv_desc.R <<
      "_" << conv_desc.S << "_" << conv_desc.u << "_" << conv_desc.v <<
      "_" << conv_desc.pad_h_out << "_" << conv_desc.pad_w_out <<
      "_" << conv_desc.buffer_format << "_" << conv_desc.filter_format;
    if (buffers_.find(ss.str()) == buffers_.end()) {
      LOG(FATAL) << "Cannot find buffer: ";
    }   
    return buffers_[ss.str()];
  }

 private:
  map<string, XsmmBuffer> buffers_;

  DISABLE_COPY_AND_ASSIGN(XsmmLoadBuffer);
};

class Xsmm {
 public:
  static XsmmLoadBuffer& GetInstance() {
    // thread safe
    boost::call_once(&Xsmm::Create, flag_);
    return *(Xsmm::instance_);
  }

  static void Create() {
    Xsmm::instance_.reset(new XsmmLoadBuffer());
  }

  static bool HasBuffer(libxsmm_dnn_conv_desc desc) {
    XsmmLoadBuffer& xsmm_load_handle = Xsmm::GetInstance();
    return xsmm_load_handle.HasBuffer(desc);
  }

  static XsmmBuffer GetBuffer(libxsmm_dnn_conv_desc desc) {
    XsmmLoadBuffer& xsmm_load_handle = Xsmm::GetInstance();
    return xsmm_load_handle.GetBuffer(desc);
  }

  static void AddBuffer(libxsmm_dnn_conv_desc desc, void* input, void* output, void* filter) {
    XsmmLoadBuffer& xsmm_load_handle = Xsmm::GetInstance();
    return xsmm_load_handle.AddBuffer(desc, input, output, filter);
  }

  virtual ~Xsmm();

 private:
  static scoped_ptr<XsmmLoadBuffer> instance_;
  static boost::once_flag flag_;

  Xsmm();
  DISABLE_COPY_AND_ASSIGN(Xsmm);
};

/*=============================================================================
 *  do some prepare work for using libxsmm convolution, includeding:
 *  #transfom data format to proper format if needed
 *  #create a handle for libxsmm convolution
 *=============================================================================*/
template<typename DType>
XsmmBuffer BlitzXsmmPrepare2D(
    DType* input,
    DType* output,
    DType* filter,
    BLITZ_DATA_LAYOUT buffer_layout,
    BLITZ_DATA_LAYOUT filter_layout,
    const size_t N, const size_t H, const size_t W,
    const size_t C, const size_t K, const size_t R, 
    const size_t S, 
    const size_t stride_h, const size_t stride_w,
    const size_t padding_h, const size_t padding_w);

}  // namespace blitz
#endif  // INCLUDE_KERNELS_XSMM_FUNCTION_H_
