#ifndef SRC_LAYER_CONV_BATCH_H_
#define SRC_LAYER_CONV_BATCH_H_

#include <map>
#include <string>
#include <vector>

#include "layer/param_layer.h"
#include "util/common.h"
#include "transform/activation.h"

namespace blitz {

template<template <typename> class TensorType, typename DType>
class ConvBatch : public ParamLayer<TensorType, DType> {
 public:
  explicit ConvBatch(
    const string& name, const string& filler_name,
    const string& optimizer_name,
    shared_ptr<Activation<TensorType, DType> > activation,
    const Shape& filter_shape,
    const int stride_height = 1, const int stride_width = 1,
    const int padding_height = 0, const int padding_width = 0,
    const string& kernel = "blas") :
    ParamLayer<TensorType, DType>(name, filler_name,
    optimizer_name, activation), filter_shape_(filter_shape),
    stride_height_(stride_height), stride_width_(stride_width),
    padding_height_(padding_height), padding_width_(padding_width),
    kernel_(kernel) {}
  ~ConvBatch() {}

  virtual void InitImpl(const Shape& input_shape);
  virtual void ForwardPropImpl(shared_ptr<TensorType<DType> > forward_input);
  virtual void BackwardPropImpl(shared_ptr<TensorType<DType> > backward_input);

 private:
  const Shape filter_shape_;

  vector<shared_ptr<TensorType<DType> > > unpack_batch_;
  vector<shared_ptr<TensorType<DType> > > update_batch_;

  const int stride_height_;
  const int stride_width_;
  const int padding_height_;
  const int padding_width_;

  const string kernel_;
};

}  // namespace blitz

#endif  // SRC_LAYER_CONV_BATCH_H_
