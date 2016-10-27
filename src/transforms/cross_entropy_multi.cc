#include "transforms/cross_entropy_multi.h"

#include "backends/backends.h"

namespace blitz {

template<template <typename> class TensorType, typename DType>
DType CrossEntropyMulti<TensorType, DType>::Apply(
  const shared_ptr<TensorType<DType> > output,
  const shared_ptr<TensorType<DType> > target) {
  return Backend<TensorType, DType>::CrossEntropyMultiApplyFunc(
    output.get(), target.get());
}

template<template <typename> class TensorType, typename DType>
void CrossEntropyMulti<TensorType, DType>::Derivative(
  const shared_ptr<TensorType<DType> > output,
  const shared_ptr<TensorType<DType> > target,
  shared_ptr<TensorType<DType> > result) {
  Backend<TensorType, DType>::CrossEntropyMultiDerivativeFunc(
    output.get(), target.get(), result.get());
}

INSTANTIATE_CLASS_CPU(CrossEntropyMulti);
#ifdef BLITZ_USE_MIC
  INSTANTIATE_CLASS_MIC(CrossEntropyMulti);
#endif
#ifdef BLITZ_USE_GPU
  INSTANTIATE_CLASS_GPU(CrossEntropyMulti);
#endif

}  // namespace blitz
