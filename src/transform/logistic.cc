#include "transform/logistic.h"
#include "backend/backends.h"

namespace blitz {

template<template <typename> class TensorType, typename DType>
void Logistic<TensorType, DType>::Apply(
  const shared_ptr<TensorType<DType> > input,
  shared_ptr<TensorType<DType> > output) {
  Backend<TensorType, DType>::LogisticApplyFunc(input.get(), output.get());
}

template<template <typename> class TensorType, typename DType>
void Logistic<TensorType, DType>::Derivative(
  const shared_ptr<TensorType<DType> > input,
  shared_ptr<TensorType<DType> > output) {
  if (!short_cut_) {
    Backend<TensorType, DType>::LogisticDerivativeFunc(
      input.get(), output.get());
  }
}

INSTANTIATE_CLASS(Logistic);

}  // namespace blitz
