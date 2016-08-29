#include "layer/conv.h"

#include "backend/backends.h"
#ifndef BLITZ_CPU_ONLY
#include "util/blitz_gpu_function.h"
#endif

namespace blitz {

template<template <typename> class TensorType, typename DType>
void Conv<TensorType, DType>::InitImpl(const Shape& input_shape) {
  // input shape decode
  int batch_size = input_shape[0];
  int input_channel = input_shape[1];
  int input_height = input_shape[2];
  int input_width = input_shape[3];
  // filter shape decode
  int filter_height = filter_shape_[2];
  int filter_width = filter_shape_[3];
  // output shape encode
  int output_channel = filter_shape_[0];
  int output_height = (input_height + 2 * padding_height_ - filter_height) /
    stride_height_ + 1;
  int output_width = (input_width + 2 * padding_width_ - filter_width) /
    stride_width_ + 1;

  Shape output_shape(4);
  output_shape[0] = batch_size;
  output_shape[1] = output_channel;
  output_shape[2] = output_height;
  output_shape[3] = output_width;

  // forward and backward output
  this->forward_output_ = make_shared<TensorType<DType> >(output_shape);
  this->backward_output_ = make_shared<TensorType<DType> >(input_shape);

  // weight
  Shape shape_weight(4);
  shape_weight[0] = output_channel;
  shape_weight[1] = input_channel;
  shape_weight[2] = filter_height;
  shape_weight[3] = filter_width;

  this->weight_ = make_shared<TensorType<DType> >(shape_weight);
  this->update_ = make_shared<TensorType<DType> >(shape_weight);

  // unpack one image in every iteration

  if (this->kernel_ == "asm" || this->kernel_ == "blas") {
    Shape unpack_shape(2);
    unpack_shape[0] = input_channel * filter_height * filter_width;
    unpack_shape[1] = output_height * output_width;
    this->unpack_ = make_shared<TensorType<DType> >(unpack_shape);
  } 
#ifndef BLITZ_CPU_ONLY
  else if (this->kernel_ == "cudnn") {
    // create val
    cudnn_alpha_ = new DType(1.0);
    cudnn_beta_ = new DType(0.0);

    // create handle
    cudnnCreate(&cudnn_handle_);

    // create descriptors
    cudnn::createTensor4dDesc<DType>(&input_desc_);
    cudnn::createTensor4dDesc<DType>(&output_desc_);
    cudnn::createFilterDesc<DType>(&filter_desc_);
    cudnn::createConvolution2DDesc<DType>(&conv_desc_);

    // set descriptors
    cudnn::setTensor4dDesc<DType>(&input_desc_,
      batch_size, input_channel, input_height, input_width);
    cudnn::setTensor4dDesc<DType>(&output_desc_,
      batch_size, output_channel, output_height, output_width);
    cudnn::setFilterDesc<DType>(&filter_desc_, output_channel,
      input_channel, filter_height, filter_width);
    cudnn::setConvolution2DDesc<DType>(&conv_desc_,
      padding_height_, padding_width_,
      stride_height_, stride_width_);

    // set algorithms
    forward_algorithm_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    backward_filter_algorithm_ = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
    backward_data_algorithm_ = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
  }
#endif

  LOG(INFO) << "Conv Layer: " << this->name_;
  LOG(INFO) << "input shape: " << input_channel << " * " << input_height <<
    " * " << input_width;
  LOG(INFO) << "weight shape: " << output_channel << " * " << input_channel <<
    " * " << input_height << " * " << input_width;
  LOG(INFO) << "output shape: " << output_channel << " * " << output_height <<
    " * " << output_width;
}

template<template <typename> class TensorType, typename DType>
void Conv<TensorType, DType>::ForwardPropImpl(
  shared_ptr<TensorType<DType> > forward_input) {
  // TODO(keren) fusing
#ifndef BLITZ_CPU_ONLY
  if (this->kernel_ == "cudnn") {
    // start cudnn directly from the layer, not throught backend
    // because backend is a general engine
    cudnnConvolutionForward(cudnn_handle_, (void*)cudnn_alpha_,
      input_desc_, forward_input->data(), filter_desc_, (this->weight_)->data(),
      conv_desc_, forward_algorithm_, NULL, 0, (void*)cudnn_beta_,
      output_desc_, (this->forward_output_)->data());
  } else {
    Backend<TensorType, DType>::Convolution2DForwardFunc(
      forward_input.get(), (this->weight_).get(),
      padding_height_, padding_width_, stride_height_, stride_width_,
      (this->unpack_).get(), (this->forward_output_).get(), this->kernel_);
  }
#else
  Backend<TensorType, DType>::Convolution2DForwardFunc(
    forward_input.get(), (this->weight_).get(),
    padding_height_, padding_width_, stride_height_, stride_width_,
    (this->unpack_).get(), (this->forward_output_).get());
#endif
}

template<template <typename> class TensorType, typename DType>
void Conv<TensorType, DType>::BackwardPropImpl(
  shared_ptr<TensorType<DType> > backward_input) {
  if (this->backward_prop_) {
#ifndef BLITZ_CPU_ONLY
    if (this->kernel_ == "cudnn") {
      cudnnConvolutionBackwardData(cudnn_handle_, (void*)cudnn_alpha_,
        filter_desc_, (this->weight_)->data(), output_desc_, backward_input->data(),
        conv_desc_, backward_data_algorithm_, NULL, 0,
        (void*)cudnn_beta_, input_desc_, (this->backward_output_)->data());
    } else {
      Backend<TensorType, DType>::Convolution2DBackwardFunc(
      backward_input.get(), (this->weight_).get(),
      padding_height_, padding_width_, stride_height_, stride_width_,
      (this->unpack_).get(), (this->backward_output_).get(), this->kernel_);
    }
#else
    Backend<TensorType, DType>::Convolution2DBackwardFunc(
    backward_input.get(), (this->weight_).get(),
    padding_height_, padding_width_, stride_height_, stride_width_,
    (this->unpack_).get(), (this->backward_output_).get());
#endif
  }
#ifndef BLITZ_CPU_ONLY
  if (this->kernel_ == "cudnn") {
    cudnnConvolutionBackwardFilter(cudnn_handle_, (void*)cudnn_alpha_,
      input_desc_, (this->forward_input_)->data(),
      output_desc_, backward_input->data(),
      conv_desc_, backward_filter_algorithm_, NULL, 0,
      (void*)cudnn_alpha_, filter_desc_, (this->update_)->data());
  } else {
    Backend<TensorType, DType>::Convolution2DUpdateFunc(
      (this->forward_input_).get(), backward_input.get(),
      padding_height_, padding_width_, stride_height_, stride_width_,
      (this->unpack_).get(), (this->update_).get(), this->kernel_);
  }
#else
  Backend<TensorType, DType>::Convolution2DUpdateFunc(
    (this->forward_input_).get(), backward_input.get(),
    padding_height_, padding_width_, stride_height_, stride_width_,
    (this->unpack_).get(), (this->update_).get());
#endif
}

INSTANTIATE_CLASS(Conv);

}  // namespace blitz
