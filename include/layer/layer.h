#ifndef INCLUDE_LAYER_LAYER_H_
#define INCLUDE_LAYER_LAYER_H_

#include <string>

#include "util/common.h"
#include "scheduler/scheduler.h"
#include "filler/filler_wrapper.h"
#include "backend/shape.h"

namespace blitz {

template<template <typename> class TensorType, typename DType>
class Layer {
 public:
  explicit Layer(const string& name) :
    name_(name), train_(true), backward_prop_(true) {}  // indicate pure virtual
  virtual ~Layer() {}  // ensure pure virtual

  virtual void Init(const Shape& input_shape,
    shared_ptr<FillerWrapper<TensorType, DType> > filler_wrapper,
    shared_ptr<Scheduler<TensorType, DType> > scheduler) {
    this->InitImpl(input_shape);
  }
  virtual void InitImpl(const Shape& input_shape) = 0;

  // forward
  virtual void ForwardProp(shared_ptr<TensorType<DType> > forward_input) {
    #ifdef BLITZ_PERFORMANCE
    time_point<system_clock> start, end;
    duration<double> core_time =
      duration<double>::zero();
    LOG(INFO) << this->name_ << ":";
    start = system_clock::now();
    #endif  // BLITZ_PERFORMANCE

    this->ForwardPropImpl(forward_input);

    #ifdef BLITZ_PERFORMANCE
    end = system_clock::now();
    core_time = end - start;
    LOG(INFO) << "Forward core time: " <<
      core_time.count();
    #endif  // BLITZ_PERFORMANCE
  }
  virtual void ForwardPropImpl(shared_ptr<TensorType<DType> >
    forward_input) = 0;

  // backward
  virtual void BackwardProp(shared_ptr<TensorType<DType> > backward_input) {
    #ifdef BLITZ_PERFORMANCE
    time_point<system_clock> start, end;
    duration<double> core_time = duration<double>::zero();
    start = system_clock::now();
    LOG(INFO) << this->name_ << ":";
    #endif  // BLITZ_PERFORMANCE

    this->BackwardPropImpl(backward_input);

    #ifdef BLITZ_PERFORMANCE
    end = system_clock::now();
    core_time = end - start;
    LOG(INFO) << "Backward core time: " <<
      core_time.count();
    #endif  // BLITZ_PERFORMANCE
  }

  virtual void BackwardPropImpl(shared_ptr<TensorType<DType> >
    backward_input) = 0;

  shared_ptr<TensorType<DType> > forward_output() const {
    return this->forward_output_;
  }

  void set_forward_output(shared_ptr<TensorType<DType> > forward_output) {
    this->forward_output_ = forward_output;
  }

  const Shape& forward_output_shape() const {
    return this->forward_output_->shape();
  }

  shared_ptr<TensorType<DType> > backward_output() const {
    return this->backward_output_;
  }

  void set_backward_output(shared_ptr<TensorType<DType> > backward_output) {
    this->backward_output_ = backward_output;
  }

  const Shape& backward_output_shape() const {
    return this->backward_output_->shape();
  }

  bool backward_prop() const {
    return this->backward_prop;
  }

  void set_backward_prop(const bool backward_prop) {
    this->backward_prop_ = backward_prop;
  }

  // Two modes
  void SetTrainMode() {
    this->train_ = true;
  }

  void SetInferenceMode() {
    this->train_ = false;
  }

 protected:
  shared_ptr<TensorType<DType> > forward_input_;
  shared_ptr<TensorType<DType> > forward_output_;
  shared_ptr<TensorType<DType> > backward_output_;

  const string name_;
  bool train_;
  bool backward_prop_;

  DISABLE_COPY_AND_ASSIGN(Layer);
};

}  // namespace blitz

#endif  // INCLUDE_LAYER_LAYER_H_
