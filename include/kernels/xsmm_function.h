#ifndef INCLUDE_KERNELS_XSMM_FUNCTION_H_
#define INCLUDE_KERNELS_XSMM_FUNCTION_H_

#include <libxsmm.h>
#include <omp.h>

#include <boost/thread/once.hpp>
#include <boost/noncopyable.hpp>

#include "utils/common.h"

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

  void AddBuffer(libxsmm_dnn_conv_desc conv_desc) {
    libxsmm_dnn_err_t status;
    libxsmm_dnn_buffer* libxsmm_input;
    libxsmm_dnn_buffer* libxsmm_output;
    libxsmm_dnn_filter* libxsmm_filter;
    libxsmm_dnn_conv_handle* libxsmm_handle =
      libxsmm_dnn_create_conv_handle_check(conv_desc, &status);;
    // create buffers
    libxsmm_input = libxsmm_dnn_create_input_buffer_check(libxsmm_handle, &status);
    CHKERR_LIBXSMM_DNN(status);
    libxsmm_output = libxsmm_dnn_create_output_buffer_check(libxsmm_handle, &status);
    CHKERR_LIBXSMM_DNN(status);
    libxsmm_filter = libxsmm_dnn_create_filter_check(libxsmm_handle, &status);
    CHKERR_LIBXSMM_DNN(status);
    // bind buffers
    CHKERR_LIBXSMM_DNN(libxsmm_dnn_bind_input_buffer(libxsmm_handle, libxsmm_input));
    CHKERR_LIBXSMM_DNN(libxsmm_dnn_bind_output_buffer(libxsmm_handle, libxsmm_output));
    CHKERR_LIBXSMM_DNN(libxsmm_dnn_bind_filter(libxsmm_handle, libxsmm_filter));
    // add into map
    XsmmBuffer buffer;
    buffer.libxsmm_handle = libxsmm_handle;
    buffer.libxsmm_input = libxsmm_input;
    buffer.libxsmm_output = libxsmm_output;
    buffer.libxsmm_filter = libxsmm_filter;
    std::stringstream ss;
    ss << conv_desc.N << "_" << conv_desc.C << "_" << conv_desc.H <<
      "_" << conv_desc.W << "_" << conv_desc.K << "_" << conv_desc.R <<
      "_" << conv_desc.S << "_" << conv_desc.u << "_" << conv_desc.v <<
      "_" << conv_desc.pad_h_in << "_" << conv_desc.pad_h_out << "_" << conv_desc.pad_w_out;
    buffers_[ss.str()] = buffer;
  }

  bool HasBuffer(libxsmm_dnn_conv_desc conv_desc) {
    std::stringstream ss;
    ss << conv_desc.N << "_" << conv_desc.C << "_" << conv_desc.H <<
      "_" << conv_desc.W << "_" << conv_desc.K << "_" << conv_desc.R <<
      "_" << conv_desc.S << "_" << conv_desc.u << "_" << conv_desc.v <<
      "_" << conv_desc.pad_h_in << "_" << conv_desc.pad_h_out << "_" << conv_desc.pad_w_out;
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
      "_" << conv_desc.pad_h_in << "_" << conv_desc.pad_h_out << "_" << conv_desc.pad_w_out;
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
    XsmmLoadBuffer& xsmm_load_buffer = Xsmm::GetInstance();
    return xsmm_load_buffer.HasBuffer(desc);
  }

  static XsmmBuffer GetBuffer(libxsmm_dnn_conv_desc desc) {
    XsmmLoadBuffer& xsmm_load_buffer = Xsmm::GetInstance();
    return xsmm_load_buffer.GetBuffer(desc);
  }

  static void AddBuffer(libxsmm_dnn_conv_desc desc) {
    XsmmLoadBuffer& xsmm_load_buffer = Xsmm::GetInstance();
    return xsmm_load_buffer.AddBuffer(desc);
  }

  virtual ~Xsmm();

 private:
  Xsmm();

  static scoped_ptr<XsmmLoadBuffer> instance_;
  static boost::once_flag flag_;

  DISABLE_COPY_AND_ASSIGN(Xsmm);
};

template<typename DType>
void BlitzXsmmConvolution2D(
  DType* input,
  DType* output,
  DType* filter,
  size_t batch_size,
  size_t input_channel,
  size_t input_height, size_t input_width,
  size_t filter_height, size_t filter_width,
  size_t output_channel,
  size_t output_height, size_t output_width,
  size_t stride_height, size_t stride_width,
  size_t padding_height, size_t padding_width,
  size_t tid_batch, size_t tid, size_t num_threads,
  const string& phase);

}  // namespace blitz

#endif  // INCLUDE_KERNELS_XSMM_FUNCTION_H_
