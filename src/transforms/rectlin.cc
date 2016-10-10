#include "transforms/rectlin.h"

#include "backends/backends.h"

namespace blitz {

template<template <typename> class TensorType, typename DType>
void Rectlin<TensorType, DType>::Apply(
  const shared_ptr<TensorType<DType> > input,
  shared_ptr<TensorType<DType> > output) {
  Backend<TensorType, DType>::RectlinApplyFunc(
    input.get(), slope_, output.get());
}

template<template <typename> class TensorType, typename DType>
void Rectlin<TensorType, DType>::Derivative(
  const shared_ptr<TensorType<DType> > input,
  shared_ptr<TensorType<DType> > output) {
  Backend<TensorType, DType>::RectlinDerivativeFunc(
    input.get(), slope_, output.get());
}

INSTANTIATE_CLASS(Rectlin);

}  // namespace blitz
