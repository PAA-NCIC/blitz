#ifndef SRC_LAYER_POOLING_LAYER_H_
#define SRC_LAYER_POOLING_LAYER_H_

#include <list>
#include <string>

#include "layer/layer.h"
#include "util/common.h"

namespace blitz {

template<template <typename> class TensorType, typename DType>
class PoolingLayer : public Layer<TensorType, DType> {
 public:
  explicit PoolingLayer(
    const string& name, const int filter, const int stride,
    const string& op = "max") :
    Layer<TensorType, DType>(name), filter_(filter),
    stride_(stride), op_(op) {}
  ~PoolingLayer() {}

  virtual void InitImpl(const Shape& input_shape);
  virtual void ForwardPropImpl(shared_ptr<TensorType<DType> > forward_input);
  virtual void BackwardPropImpl(shared_ptr<TensorType<DType> > backward_input);

 private:
  const int filter_;
  const int stride_;

  string op_;

  // according to different op
  shared_ptr<TensorType<size_t> > max_index_;
};

}  // namespace blitz

#endif  // SRC_LAYER_POOLING_LAYER_H_
