#include "transforms/rectlin.h"

#include "backends/backends.h"

namespace blitz {

template<template <typename> class TensorType, typename DType>
void Rectlin<TensorType, DType>::Apply(
  const shared_ptr<TensorType<DType> > input,
  shared_ptr<TensorType<DType> > output) {
  Backend<TensorType, DType>::RectlinApplyFunc(
    input.get(), output.get(), slope_);
}

template<template <typename> class TensorType, typename DType>
void Rectlin<TensorType, DType>::Derivative(
  const shared_ptr<TensorType<DType> > input,
  shared_ptr<TensorType<DType> > output) {
  Backend<TensorType, DType>::RectlinDerivativeFunc(
    input.get(), output.get(), slope_);
}

INSTANTIATE_CLASS_CPU(Rectlin);
#ifdef BLITZ_USE_MIC
  INSTANTIATE_CLASS_MIC(Rectlin);
#endif
#ifdef BLITZ_USE_GPU
  INSTANTIATE_CLASS_GPU(Rectlin);
#endif

}  // namespace blitz
