#include "transform/cross_entropy_binary.h"
#include "backend/backends.h"

namespace blitz {

template<template <typename> class TensorType, typename DType>
DType CrossEntropyBinary<TensorType, DType>::Apply(
  const shared_ptr<TensorType<DType> > output,
  const shared_ptr<TensorType<DType> > target) {
  return Backend<TensorType, DType>::CrossEntropyBinaryApplyFunc(
      output.get(), target.get());
}

template<template <typename> class TensorType, typename DType>
void CrossEntropyBinary<TensorType, DType>::Derivative(
  const shared_ptr<TensorType<DType> > output,
  const shared_ptr<TensorType<DType> > target,
  shared_ptr<TensorType<DType> > result) {
  Backend<TensorType, DType>::CrossEntropyBinaryDerivativeFunc(
      output.get(), target.get(), result.get());
}

INSTANTIATE_CLASS(CrossEntropyBinary);

}  // namespace blitz
