#include "filler/uniform.h"
#include "backend/backends.h"

namespace blitz {

template<template <typename> class TensorType, typename DType>
void Uniform<TensorType, DType>::FillImpl(LayerWeightIterator
  layer_weight_it) {
#ifdef BLITZ_DEVELOP
  LOG(INFO) << "Uniform Fill: " << layer_weight_it->first;
#endif
  shared_ptr<TensorType<DType> > weight = layer_weight_it->second;
  Backend<TensorType, DType>::UniformDistributionFunc(low_, high_,
    weight.get());
}

INSTANTIATE_CLASS(Uniform);

}  // namespace blitz
