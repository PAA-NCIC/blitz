#include "transforms/logistic.h"

#include "backends/backends.h"

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

INSTANTIATE_CLASS_CPU(Logistic);
#ifdef BLITZ_USE_MIC
  INSTANTIATE_CLASS_MIC(Logistic);
#endif
#ifdef BLITZ_USE_GPU
  INSTANTIATE_CLASS_GPU(Logistic);
#endif

}  // namespace blitz
