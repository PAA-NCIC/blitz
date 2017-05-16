#include "layers/conv.h"

#include "blitz.h"

namespace blitz {

template<template <typename> class TensorType, typename DType>
void Conv<TensorType, DType>::InitImpl(const Shape& input_shape) {
  // input shape decode
  size_t batch_size = input_shape[0];
  size_t input_channel = input_shape[1];
  size_t input_height = input_shape[2];
  size_t input_width = input_shape[3];
  // filter shape decode
  size_t filter_height = filter_shape_[2];
  size_t filter_width = filter_shape_[3];
  // output shape encode
  size_t output_channel = filter_shape_[0];
  size_t output_height = (input_height + 2 * padding_height_ - filter_height) /
    stride_height_ + 1;
  size_t output_width = (input_width + 2 * padding_width_ - filter_width) /
    stride_width_ + 1;
  Shape output_shape(4, BLITZ_BUFFER_NCHW);
  output_shape[0] = batch_size;
  output_shape[1] = output_channel;
  output_shape[2] = output_height;
  output_shape[3] = output_width;
  // forward and backward output
  this->forward_output_ = make_shared<TensorType<DType> >(output_shape);
  this->backward_output_ = make_shared<TensorType<DType> >(input_shape);
  // weight
  Shape weight_shape(4, BLITZ_FILTER_KCRS);
  weight_shape[0] = output_channel;
  weight_shape[1] = input_channel;
  weight_shape[2] = filter_height;
  weight_shape[3] = filter_width;
  this->weight_ = make_shared<TensorType<DType> >(weight_shape);
  this->update_ = make_shared<TensorType<DType> >(weight_shape);
  this->backward_computations_ = this->backward_update_computations_ =
    this->forward_computations_ = static_cast<double>(batch_size) *
    static_cast<double>(output_channel * output_height * output_width) *
    static_cast<double>(input_channel * filter_height * filter_width * 2);

  if (this->algorithm_ == BLITZ_CONVOLUTION_CUDNN) {
  #ifdef BLITZ_USE_CUDNN
    Shape workspace_shape(1);
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
    this->workspace_ = make_shared<TensorType<DType> >(workspace_shape);
  #endif
  } else {
    context_ = make_shared<ConvolutionContext<TensorType, DType> >(
      input_shape, weight_shape,
      padding_height_, padding_width_,
      stride_height_, stride_width_);
    context_->InitAlgorithmForUser(this->algorithm_);
  }

  LOG(INFO) << "Conv Layer: " << this->name_;
  LOG(INFO) << "input shape: " << input_channel << " * " << input_height <<
    " * " << input_width;
  LOG(INFO) << "weight shape: " << output_channel << " * " << input_channel <<
    " * " << filter_height << " * " << filter_width;
  LOG(INFO) << "output shape: " << output_channel << " * " << output_height <<
    " * " << output_width;
}

template<template <typename> class TensorType, typename DType>
void Conv<TensorType, DType>::ForwardPropImpl(
  shared_ptr<TensorType<DType> > forward_input) {
  // TODO(keren) fusing
  #ifdef BLITZ_USE_CUDNN
  if (this->algorithm_ == BLITZ_CONVOLUTION_CUDNN) {
    // start cudnn directly from the layer, not throught backend
    // because backend is a general engine
    cudnnConvolutionForward(cudnn_handle_,
      reinterpret_cast<void*>(cudnn_alpha_), input_desc_,
      forward_input->data(), filter_desc_, (this->weight_)->data(),
      conv_desc_, forward_algorithm_, NULL, 0,
      reinterpret_cast<void*>(cudnn_beta_),
      output_desc_, (this->forward_output_)->data());
  } else {
    Backend<TensorType, DType>::Convolution2DForwardFunc(
      forward_input.get(),
      (this->weight_).get(),
      (this->forward_output_).get(),
      this->context_.get());
  }
  #else
  Backend<TensorType, DType>::Convolution2DForwardFunc(
    forward_input.get(),
    (this->weight_).get(),
    (this->forward_output_).get(), 
    this->context_.get());
  #endif
}

template<template <typename> class TensorType, typename DType>
void Conv<TensorType, DType>::BackwardPropImpl(
  shared_ptr<TensorType<DType> > backward_input) {
  if (this->backward_prop_) {
#ifdef BLITZ_USE_CUDNN
    if (this->algorithm_ == BLITZ_CONVOLUTION_CUDNN) {
      cudnnConvolutionBackwardData(cudnn_handle_,
        reinterpret_cast<void*>(cudnn_alpha_),
        filter_desc_, (this->weight_)->data(),
        output_desc_, backward_input->data(),
        conv_desc_, backward_data_algorithm_, NULL, 0,
        reinterpret_cast<void*>(cudnn_beta_), input_desc_,
        (this->backward_output_)->data());
    } else {
      Backend<TensorType, DType>::Convolution2DBackwardFunc(
      backward_input.get(),
      (this->weight_).get(),
      (this->backward_output_).get(), 
      this->context_.get());
    }
#else
    Backend<TensorType, DType>::Convolution2DBackwardFunc(
      backward_input.get(),
      (this->weight_).get(),
      (this->backward_output_).get(),
      this->context_.get());
#endif
  }
#ifdef BLITZ_USE_CUDNN
  if (this->algorithm_ == BLITZ_CONVOLUTION_CUDNN) {
    cudnnConvolutionBackwardFilter(cudnn_handle_,
      reinterpret_cast<void*>(cudnn_alpha_),
      input_desc_, (this->forward_input_)->data(),
      output_desc_, backward_input->data(),
      conv_desc_, backward_filter_algorithm_, NULL, 0,
      reinterpret_cast<void*>(cudnn_alpha_),
      filter_desc_, (this->update_)->data());
  } else {
    Backend<TensorType, DType>::Convolution2DUpdateFunc(
      (this->forward_input_).get(),
      backward_input.get(),
      (this->update_).get(),
      this->context_.get());
  }
#else
  Backend<TensorType, DType>::Convolution2DUpdateFunc(
    (this->forward_input_).get(),
    backward_input.get(),
    (this->update_).get(),
    this->context_.get());
#endif
}

INSTANTIATE_CLASS_CPU(Conv);
#ifdef BLITZ_USE_GPU
  INSTANTIATE_CLASS_GPU(Conv);
#endif

}  // namespace blitz
