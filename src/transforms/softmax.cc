#include "transforms/softmax.h"

#include "backends/backends.h"

namespace blitz {

template<template <typename> class TensorType, typename DType>
void Softmax<TensorType, DType>::Apply(
  const shared_ptr<TensorType<DType> > input,
  shared_ptr<TensorType<DType> > output) {
  Backend<TensorType, DType>::SoftmaxApplyFunc(
    input.get(), output.get());
}

template<template <typename> class TensorType, typename DType>
void Softmax<TensorType, DType>::Derivative(
  const shared_ptr<TensorType<DType> > input,
  shared_ptr<TensorType<DType> > output) {
  if (!short_cut_) {
    Backend<TensorType, DType>::SoftmaxDerivativeFunc(
      input.get(), output.get());
  }
}

INSTANTIATE_CLASS(Softmax);

}  // namespace blitz
