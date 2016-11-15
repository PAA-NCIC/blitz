#include "layers/affine.h"

#include "backends/backends.h"

namespace blitz {

template<template <typename> class TensorType, typename DType>
void Affine<TensorType, DType>::InitImpl(const Shape& input_shape) {
  // input and output
  size_t batch_size = input_shape[0];
  size_t nin = input_shape.size() / batch_size;

  Shape output_shape(2);
  output_shape[0] = batch_size;
  output_shape[1] = nout_;

  // forward and backward output
  this->forward_output_ = make_shared<TensorType<DType> >(output_shape);
  this->backward_output_ = make_shared<TensorType<DType> >(input_shape);

  // weight and update
  Shape shape_weight(2);
  shape_weight[0] = nin;
  shape_weight[1] = nout_;

  this->weight_ = make_shared<TensorType<DType> >(shape_weight);
  this->update_ = make_shared<TensorType<DType> >(shape_weight);

  LOG(INFO) << "Affine Layer: " << this->name_;
  LOG(INFO) << "nin: " << nin;
  LOG(INFO) << "weight shape: " << nin << " * " << nout_;
  LOG(INFO) << "nout: " << nout_;
}

template<template <typename> class TensorType, typename DType>
void Affine<TensorType, DType>::ForwardPropImpl(
  shared_ptr<TensorType<DType> > forward_input) {
  Backend<TensorType, DType>::MatrixMultiplyFunc(
    forward_input.get(),
    (this->weight_).get(),
    (this->forward_output_).get(),
    false, false, 1, 0,
    this->algorithm_);
}

template<template <typename> class TensorType, typename DType>
void Affine<TensorType, DType>::BackwardPropImpl(
  shared_ptr<TensorType<DType> > backward_input) {
  #ifdef BLITZ_PERFORMANCE
  time_point<system_clock> start;
  time_point<system_clock> end;
  duration<double> time =
    duration<double>::zero();
  start = system_clock::now();
  #endif  // BLITZ_PERFORMANCE
  if (this->backward_prop_) {
    Backend<TensorType, DType>::MatrixMultiplyFunc(
      backward_input.get(),
      (this->weight_).get(),
      (this->backward_output_).get(), 
      false, true, 1, 0,
      this->algorithm_);
  }
  #ifdef BLITZ_PERFORMANCE
  end = system_clock::now();
  time = end - start;
  LOG(INFO) << "Backward affine time: " << time.count();
  #endif  // BLITZ_PERFORMANCE

  #ifdef BLITZ_PERFORMANCE
  start = system_clock::now();
  #endif  // BLITZ_PERFORMANCE
  Backend<TensorType, DType>::MatrixMultiplyFunc(
    (this->forward_input_).get(),
    backward_input.get(),
    (this->update_).get(),
    true, false, 1, 0,
    this->algorithm_);
  #ifdef BLITZ_PERFORMANCE
  end = system_clock::now();
  time = end - start;
  LOG(INFO) << "Backward affine weight time: " << time.count();
  #endif  // BLITZ_PERFORMANCE
}

INSTANTIATE_CLASS_CPU(Affine);
#ifdef BLITZ_USE_MIC
  INSTANTIATE_CLASS_MIC(Affine);
#endif
#ifdef BLITZ_USE_GPU
  INSTANTIATE_CLASS_GPU(Affine);
#endif

}  // namespace blitz
