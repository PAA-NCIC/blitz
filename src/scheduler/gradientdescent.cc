#include "scheduler/gradientdescent.h"
#include "backends/backends.h"

namespace blitz {

template<template <typename> class TensorType, typename DType>
void Gradientdescent<TensorType, DType>::OptimizeImpl(
  const size_t epoch, const size_t batch_size,
  const DType learning_rate, LayerParamIterator layer_param_it) {
#ifdef BLITZ_DEVELOP
  LOG(INFO) << "Optimize: " << layer_param_it->first;
  LOG(INFO) << "batch_size: " << batch_size;
  LOG(INFO) << "epoch: " << epoch;
  LOG(INFO) << "learning_rate: " << learning_rate;
#endif
  shared_ptr<LayerParam> layer_param = layer_param_it->second;
  shared_ptr<TensorType<DType> > weight = layer_param->weight();
  shared_ptr<TensorType<DType> > gradient = layer_param->update();
  shared_ptr<TensorType<DType> > velocity = layer_param->state();

  Backend<TensorType, DType>::GradientdescentFunc(
    weight.get(), gradient.get(), velocity.get(),
    momentum_coef_, learning_rate, decay_, batch_size);
}

INSTANTIATE_CLASS_CPU(Gradientdescent);
#ifdef BLITZ_USE_MIC
  INSTANTIATE_CLASS_MIC(Gradientdescent);
#endif
#ifdef BLITZ_USE_GPU
  INSTANTIATE_CLASS_GPU(Gradientdescent);
#endif

}  // namespace blitz

