#ifndef INCLUDE_LAYERS_POOLING_H_
#define INCLUDE_LAYERS_POOLING_H_

#include <list>
#include <string>

#include "layers/layer.h"
#include "utils/common.h"

namespace blitz {

template<template <typename> class TensorType, typename DType>
class Pooling : public Layer<TensorType, DType> {
 public:
  explicit Pooling(
    const string& name, const size_t filter, const size_t stride,
    const string& op = "max") :
    Layer<TensorType, DType>(name), filter_(filter),
    stride_(stride), op_(op) {}
  ~Pooling() {}

  virtual void InitImpl(const Shape& input_shape);
  virtual void ForwardPropImpl(shared_ptr<TensorType<DType> > forward_input);
  virtual void BackwardPropImpl(shared_ptr<TensorType<DType> > backward_input);

 private:
  const size_t filter_;
  const size_t stride_;

  const string op_;

  // according to different op
  shared_ptr<TensorType<size_t> > max_index_;

  DISABLE_COPY_AND_ASSIGN(Pooling);
};

}  // namespace blitz

#endif  // INCLUDE_LAYERS_POOLING_H_
