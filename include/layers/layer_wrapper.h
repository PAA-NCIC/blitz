#ifndef INCLUDE_LAYERS_LAYER_WRAPPER_H_
#define INCLUDE_LAYERS_LAYER_WRAPPER_H_

#include <list>
#include <string>

#include "fillers/filler_wrapper.h"
#include "layers/layer.h"
#include "scheduler/scheduler.h"
#include "transforms/cost.h"
#include "utils/common.h"

namespace blitz {

// list structure
// TODO(keren) dag structure
template<template <typename> class TensorType, typename DType>
class LayerWrapper {
 public:
  typedef typename list<shared_ptr<Layer<TensorType, DType> > >::iterator
    LayerIterator;
  typedef typename list<shared_ptr<Layer<TensorType, DType> > >::reverse_iterator
    LayerReverseIterator;

 public:
  explicit LayerWrapper(
    const list<shared_ptr<Layer<TensorType, DType> > >& layers,
    shared_ptr<Cost<TensorType, DType> > cost) :
    layers_(layers), cost_(cost) {}

  // STL like function
  void push_back(shared_ptr<Layer<TensorType, DType> > layer) {
    layers_.push_back(layer);
  }

  // setters
  void set_layers(const list<shared_ptr<Layer<TensorType, DType> > >& layers) {
    layers_ = layers;
  }

  void set_cost(shared_ptr<Cost<TensorType, DType> > cost) {
    cost_ = cost;
  }

  // iterators
  LayerIterator begin() {
    return layers_.begin();
  }

  LayerReverseIterator rbegin() {
    return layers_.rbegin();
  }

  LayerIterator end() {
    return layers_.end();
  }

  LayerReverseIterator rend() {
    return layers_.rend();
  }

  shared_ptr<TensorType<DType> > forward_output() {
    return (*layers_.rbegin())->forward_output();
  }

  DType ApplyCost(const shared_ptr<TensorType<DType> > target);

  void DerivativeCost(const shared_ptr<TensorType<DType> > target);

  DType Evaluate(const shared_ptr<TensorType<DType> > target,
    const string& eval_type);

  void Init(const Shape& data_shape,
    shared_ptr<FillerWrapper<TensorType, DType> > filler_wrapper,
    shared_ptr<Scheduler<TensorType, DType> > scheduler);

  void ForwardProp(shared_ptr<TensorType<DType> > input);

  void BackwardProp();

  void SetTrainMode();

  void SetInferenceMode();

 private:
  list<shared_ptr<Layer<TensorType, DType> > > layers_;
  shared_ptr<Cost<TensorType, DType> > cost_;
  shared_ptr<TensorType<DType> > error_;

  DISABLE_COPY_AND_ASSIGN(LayerWrapper);
};

}  // namespace blitz

#endif  // INCLUDE_LAYERS_LAYER_WRAPPER_H_
