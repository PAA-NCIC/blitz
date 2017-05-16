#include "layers/dropout.h"

#include "blitz.h"

namespace blitz {

template<template <typename> class TensorType, typename DType>
void Dropout<TensorType, DType>::InitImpl(const Shape& input_shape) {
  // forward and backward output
  this->forward_output_ = make_shared<TensorType<DType> >(input_shape);
  this->backward_output_ = make_shared<TensorType<DType> >(input_shape);

  // mask
  mask_ = make_shared<TensorType<DType> >(input_shape);

  LOG(INFO) << "Dropout Layer: " << this->name_;
  LOG(INFO) << "Keep: " << keep_;
}

template<template <typename> class TensorType, typename DType>
void Dropout<TensorType, DType>::ForwardPropImpl(
  shared_ptr<TensorType<DType> > forward_input) {
  if (this->train_) {
    Backend<TensorType, DType>::MakeBinaryMaskFunc(mask_.get(),
      0.0, 1.0, keep_);
    Backend<TensorType, DType>::MultiplyFunc(forward_input.get(),
      mask_.get(), (this->forward_output_).get());
  } else {
    Backend<TensorType, DType>::MultiplyFunc(forward_input.get(),
      (this->forward_output_).get(), keep_);
  }
}

template<template <typename> class TensorType, typename DType>
void Dropout<TensorType, DType>::BackwardPropImpl(
  shared_ptr<TensorType<DType> > backward_input) {
  Backend<TensorType, DType>::MultiplyFunc(backward_input.get(),
    mask_.get(), (this->backward_output_).get());
}

INSTANTIATE_CLASS_CPU(Dropout);
#ifdef BLITZ_USE_GPU
  INSTANTIATE_CLASS_GPU(Dropout);
#endif

}  // namespace blitz
