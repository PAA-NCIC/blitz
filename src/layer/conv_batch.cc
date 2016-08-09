#include "layer/conv_batch.h"

#include "backend/backends.h"

namespace blitz {

template<template <typename> class TensorType, typename DType>
void ConvBatch<TensorType, DType>::InitImpl(const Shape& input_shape) {
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
  Shape unpack_shape(2);
  unpack_shape[0] = input_channel * filter_height * filter_width;
  unpack_shape[1] = output_height * output_width;

  // batch parallel buffer
  update_batch_.resize(BLITZ_NUM_THREADS);
  for (size_t i = 0; i < update_batch_.size(); ++i) {
    update_batch_[i] = make_shared<TensorType<DType> >(shape_weight);
  }

  unpack_batch_.resize(BLITZ_NUM_THREADS);
  for (size_t i = 0; i < unpack_batch_.size(); ++i) {
    unpack_batch_[i] = make_shared<TensorType<DType> >(unpack_shape);
  }

  LOG(INFO) << "Conv Layer: " << this->name_;
  LOG(INFO) << "input shape: " << input_channel << " * " << input_height <<
    " * " << input_width;
  LOG(INFO) << "weight shape: " << output_channel << " * " << input_channel <<
    " * " << input_height << " * " << input_width;
  LOG(INFO) << "output shape: " << output_channel << " * " << output_height <<
    " * " << output_width;
}

template<template <typename> class TensorType, typename DType>
void ConvBatch<TensorType, DType>::ForwardPropImpl(
  shared_ptr<TensorType<DType> > forward_input) {
  // TODO(keren) fusing
  Backend<TensorType, DType>::Convolution2DForwardFunc(
    forward_input.get(), (this->weight_).get(),
    padding_height_, padding_width_, stride_height_, stride_width_,
    &unpack_batch_, (this->forward_output_).get());
}

template<template <typename> class TensorType, typename DType>
void ConvBatch<TensorType, DType>::BackwardPropImpl(
  shared_ptr<TensorType<DType> > backward_input) {
  if (this->backward_prop_) {
    Backend<TensorType, DType>::Convolution2DBackwardFunc(
    backward_input.get(), (this->weight_).get(),
    padding_height_, padding_width_, stride_height_, stride_width_,
    &unpack_batch_, (this->backward_output_).get());
  }
  for (size_t i = 0; i < update_batch_.size(); ++i) {
    update_batch_[i]->Fill(0);
  }
  Backend<TensorType, DType>::Convolution2DUpdateFunc(
    (this->forward_input_).get(), backward_input.get(),
    padding_height_, padding_width_, stride_height_, stride_width_,
    &unpack_batch_, &update_batch_, (this->update_).get());
}

INSTANTIATE_CLASS(ConvBatch);

}  // namespace blitz
