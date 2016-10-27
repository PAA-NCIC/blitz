#include "fillers/uniform.h"

#include "backends/backends.h"

namespace blitz {

template<template <typename> class TensorType, typename DType>
void Uniform<TensorType, DType>::FillImpl(LayerWeightIterator
  layer_weight_it) {
  shared_ptr<TensorType<DType> > weight = layer_weight_it->second;
  Backend<TensorType, DType>::UniformDistributionFunc(weight.get(), low_, high_);
}

INSTANTIATE_CLASS_CPU(Uniform);
#ifdef BLITZ_USE_MIC
  INSTANTIATE_CLASS_MIC(Uniform);
#endif
#ifdef BLITZ_USE_GPU
  INSTANTIATE_CLASS_GPU(Uniform);
#endif

}  // namespace blitz
