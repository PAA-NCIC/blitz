#include "fillers/gaussian.h"

#include "blitz.h"

namespace blitz {

template<template <typename> class TensorType, typename DType>
void Gaussian<TensorType, DType>::FillImpl(LayerWeightIterator
  layer_weight_it) {
  shared_ptr<TensorType<DType> > weight = layer_weight_it->second;
  Backend<TensorType, DType>::NormalDistributionFunc(weight.get(), loc_, scale_);
}

INSTANTIATE_CLASS_CPU(Gaussian);
#ifdef BLITZ_USE_GPU
  INSTANTIATE_CLASS_GPU(Gaussian);
#endif

}  // namespace blitz
