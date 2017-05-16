#include "transforms/square_mean.h"

#include "blitz.h"

namespace blitz {

template<template <typename> class TensorType, typename DType>
DType SquareMean<TensorType, DType>::Apply(
  const shared_ptr<TensorType<DType> > output,
  const shared_ptr<TensorType<DType> > target) {
  return Backend<TensorType, DType>::SquareMeanApplyFunc(
    output.get(), target.get());
}

template<template <typename> class TensorType, typename DType>
void SquareMean<TensorType, DType>::Derivative(
  const shared_ptr<TensorType<DType> > output,
  const shared_ptr<TensorType<DType> > target,
  shared_ptr<TensorType<DType> > result) {
  Backend<TensorType, DType>::SquareMeanDerivativeFunc(
    output.get(), target.get(), result.get());
}

INSTANTIATE_CLASS_CPU(SquareMean);
#ifdef BLITZ_USE_GPU
  INSTANTIATE_CLASS_GPU(SquareMean);
#endif

}  // namespace blitz
