#include "filler/gaussian.h"
#include "backend/backends.h"

namespace blitz {

template<template <typename> class TensorType, typename DType>
void Gaussian<TensorType, DType>::FillImpl(LayerWeightIterator
  layer_weight_it) {
#ifdef BLITZ_DEVELOP
  LOG(INFO) << "Gaussian Fill: " << layer_weight_it->first;
#endif
  shared_ptr<TensorType<DType> > weight = layer_weight_it->second;
  Backend<TensorType, DType>::NormalDistributionFunc(loc_, scale_,
    weight.get());
}

INSTANTIATE_CLASS(Gaussian);

}  // namespace blitz
