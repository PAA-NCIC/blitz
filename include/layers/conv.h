#ifndef INCLUDE_LAYERS_CONV_H_
#define INCLUDE_LAYERS_CONV_H_

#include <string>

#ifndef BLITZ_CPU_ONLY
#include <cudnn.h>
#endif
#include "layers/param_layer.h"
#include "transforms/activation.h"
#include "utils/common.h"

namespace blitz {

template<template <typename> class TensorType, typename DType>
class Conv : public ParamLayer<TensorType, DType> {
 public:
  explicit Conv(
    const string& name, const string& filler_name,
    const string& optimizer_name,
    shared_ptr<Activation<TensorType, DType> > activation,
    const Shape& filter_shape,
    const size_t stride_height = 1, const size_t stride_width = 1,
    const size_t padding_height = 0, const size_t padding_width = 0,
    const string& kernel = "blas") :
    ParamLayer<TensorType, DType>(name, filler_name,
    optimizer_name, activation), filter_shape_(filter_shape),
    stride_height_(stride_height), stride_width_(stride_width),
    padding_height_(padding_height), padding_width_(padding_width),
    kernel_(kernel) {}
  ~Conv() {}

  virtual void InitImpl(const Shape& input_shape);
  virtual void ForwardPropImpl(shared_ptr<TensorType<DType> > forward_input);
  virtual void BackwardPropImpl(shared_ptr<TensorType<DType> > backward_input);

 private:
  // TODO(keren) bias
  const Shape filter_shape_;

  shared_ptr<TensorType<DType> > workspace_;

  const size_t stride_height_;
  const size_t stride_width_;
  const size_t padding_height_;
  const size_t padding_width_;

  const string kernel_;

  double forward_computations_;
  double backward_computations_;
  double backward_update_computations_;
#ifndef BLITZ_CPU_ONLY
  cudnnHandle_t cudnn_handle_;

  // algorithms for forward and backwards convolutions
  cudnnConvolutionFwdAlgo_t forward_algorithm_;
  cudnnConvolutionBwdFilterAlgo_t backward_filter_algorithm_;
  cudnnConvolutionBwdDataAlgo_t backward_data_algorithm_;

  cudnnTensorDescriptor_t input_desc_, output_desc_;
  cudnnFilterDescriptor_t filter_desc_;
  cudnnConvolutionDescriptor_t conv_desc_;

  DType *cudnn_alpha_, *cudnn_beta_;
#endif

  DISABLE_COPY_AND_ASSIGN(Conv);
};

}  // namespace blitz

#endif  // INCLUDE_LAYERS_CONV_H_
