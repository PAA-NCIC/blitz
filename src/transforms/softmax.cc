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

INSTANTIATE_CLASS_CPU(Softmax);
#ifdef BLITZ_USE_MIC
  INSTANTIATE_CLASS_MIC(Softmax);
#endif
#ifdef BLITZ_USE_GPU
  INSTANTIATE_CLASS_GPU(Softmax);
#endif

}  // namespace blitz
