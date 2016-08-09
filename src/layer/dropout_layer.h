#ifndef SRC_LAYER_DROPOUT_LAYER_H_
#define SRC_LAYER_DROPOUT_LAYER_H_

#include <list>
#include <string>

#include "layer/layer.h"
#include "util/common.h"

namespace blitz {

template<template <typename> class TensorType, typename DType>
class DropoutLayer : public Layer<TensorType, DType> {
 public:
  explicit DropoutLayer(const string& name, const DType keep) :
    Layer<TensorType, DType>(name), keep_(keep) {}
  ~DropoutLayer() {}

  virtual void InitImpl(const Shape& input_shape);
  virtual void ForwardPropImpl(shared_ptr<TensorType<DType> > forward_input);
  virtual void BackwardPropImpl(shared_ptr<TensorType<DType> > backward_input);

 private:
  const DType keep_;

  shared_ptr<TensorType<DType> > mask_;
};

}  // namespace blitz

#endif  // SRC_LAYER_DROPOUT_LAYER_H_
