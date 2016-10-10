#include "layers/layer_wrapper.h"

#include <list>
#include <string>

#include "backends/backends.h"

namespace blitz {

template<template <typename> class TensorType, typename DType>
void LayerWrapper<TensorType, DType>::Init(const Shape& input_shape,
  shared_ptr<FillerWrapper<TensorType, DType> > filler_wrapper,
  shared_ptr<Scheduler<TensorType, DType> > scheduler) {
  // first input layer
  LOG(INFO) << "Init layers: ";
  LayerIterator layer_it = begin();
  (*layer_it)->Init(input_shape, filler_wrapper, scheduler);

  shared_ptr<Layer<TensorType, DType> > prev_layer = *layer_it;
  ++layer_it;

  while (layer_it != layers_.end()) {
    (*layer_it)->Init(prev_layer->forward_output_shape(), filler_wrapper,
      scheduler);
    prev_layer = *layer_it;
    ++layer_it;
  }

  // init error_
  const Shape& output_shape = (*layers_.rbegin())->forward_output_shape();
  error_ = make_shared<TensorType<DType> >(output_shape);

  // set first layer not bprop
  (*layers_.begin())->set_backward_prop(false);
}

template<template <typename> class TensorType, typename DType>
void LayerWrapper<TensorType, DType>::ForwardProp(
  shared_ptr<TensorType<DType> > input) {
  LayerIterator it = begin();
  while (it != end()) {
    (*it)->ForwardProp(input);
    input = (*it)->forward_output();
    ++it;
  }
}

template<template <typename> class TensorType, typename DType>
void LayerWrapper<TensorType, DType>::BackwardProp() {
  LayerReverseIterator it = rbegin();
  shared_ptr<TensorType<DType> > input = error_;
  while (it != rend()) {
    (*it)->BackwardProp(input);
    input = (*it)->backward_output();
    ++it;
  }
}

template<template <typename> class TensorType, typename DType>
DType LayerWrapper<TensorType, DType>::ApplyCost(
    const shared_ptr<TensorType<DType> > target) {
  shared_ptr<TensorType<DType> > output = (*layers_.rbegin())->forward_output();
  return cost_->Apply(output, target);
}

template<template <typename> class TensorType, typename DType>
void LayerWrapper<TensorType, DType>::DerivativeCost(
    const shared_ptr<TensorType<DType> > target) {
  shared_ptr<TensorType<DType> > output = (*layers_.rbegin())->forward_output();
  cost_->Derivative(output, target, error_);
}

template<template <typename> class TensorType, typename DType>
DType LayerWrapper<TensorType, DType>::Evaluate(
    const shared_ptr<TensorType<DType> > target, const string& eval_type) {
  shared_ptr<TensorType<DType> > output = (*layers_.rbegin())->forward_output();
  DType accuracy = 0.0;
  if (eval_type == "classify") {
    accuracy = Backend<TensorType, DType>::EvaluateClassifyFunc(
      output.get(), target.get());
  } else if (eval_type == "regress") {
    accuracy = Backend<TensorType, DType>::EvaluateRegressFunc(
      output.get(), target.get());
  }
  return accuracy;
}

template<template <typename> class TensorType, typename DType>
void LayerWrapper<TensorType, DType>::SetTrainMode() {
  LayerIterator it = begin();
  while (it != end()) {
    (*it)->SetTrainMode();
    ++it;
  }
}

template<template <typename> class TensorType, typename DType>
void LayerWrapper<TensorType, DType>::SetInferenceMode() {
  LayerIterator it = begin();
  while (it != end()) {
    (*it)->SetInferenceMode();
    ++it;
  }
}

INSTANTIATE_CLASS(LayerWrapper);

}  // namespace blitz
