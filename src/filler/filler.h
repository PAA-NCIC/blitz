#ifndef SRC_FILLER_FILLER_H_
#define SRC_FILLER_FILLER_H_

#include <map>
#include <string>

#include "util/common.h"

namespace blitz {

template<template <typename> class TensorType, typename DType>
class Filler {
 public:
  typedef typename map<string, shared_ptr<TensorType<DType> > >::iterator
    LayerWeightIterator;

 public:
  explicit Filler(const string& name) : name_(name) {}
  virtual ~Filler() {}

  void Fill() {
    LayerWeightIterator layer_weight_it = this->layer_weights_.begin();

    for ( ;layer_weight_it != this->layer_weights_.end(); ++layer_weight_it) {
      this->FillImpl(layer_weight_it);
    }
  }

  virtual void FillImpl(LayerWeightIterator layer_weight_it) = 0;

  void AddLayer(const string& layer_name, shared_ptr<TensorType<DType> >
    weight) {
    if (this->layer_weights_.find(layer_name) == this->layer_weights_.end()) {
      this->layer_weights_[layer_name] = weight;
    } else {
      LOG(ERROR) << "layer: " << layer_name << " exists";
    }
  }

  const string& name() {
    return name_;
  }

 protected:
  map<string, shared_ptr<TensorType<DType> > > layer_weights_;

  const string name_;

  DISABLE_COPY_AND_ASSIGN(Filler);
};

}  // namespace blitz

#endif  // SRC_FILLER_FILLER_H_
