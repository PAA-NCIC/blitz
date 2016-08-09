#ifndef SRC_LAYER_PARAM_LAYER_H_
#define SRC_LAYER_PARAM_LAYER_H_

#include <string>

#include "backend/backend.h"
#include "layer/layer.h"
#include "util/common.h"
#include "filler/filler.h"
#include "transform/activation.h"

namespace blitz {

template<template <typename> class TensorType, typename DType>
class ParamLayer : public Layer<TensorType, DType> {
 public:
  // BatchNorm:
  // Y = gamma * X + beta
  class BatchNorm {
   public:
    // default
    explicit BatchNorm(const string& name, const string& gamma_filler_name,
      const string& gamma_optimizer_name, const string& beta_filler_name,
      const string& beta_optimizer_name) : name_(name),
      gamma_optimizer_name_(gamma_optimizer_name),
      gamma_filler_name_(gamma_filler_name),
      beta_optimizer_name_(beta_optimizer_name),
      beta_filler_name_(beta_filler_name), epsilon_(1e-6) {}

    const string& name() const {
      return name_;
    }

    const string& gamma_optimizer_name() const {
      return gamma_optimizer_name_;
    }

    const string& gamma_filler_name() const {
      return gamma_filler_name_;
    }

    const string& beta_optimizer_name() const {
      return beta_optimizer_name_;
    }

    const string& beta_filler_name() {
      return beta_filler_name_;
    }

    void set_gamma_weight(shared_ptr<TensorType<DType> > weight) {
      gamma_weight_ = weight;
    }

    shared_ptr<TensorType<DType> > gamma_weight() {
      return gamma_weight_;
    }

    void set_beta_weight(shared_ptr<TensorType<DType> > weight) {
      beta_weight_ = weight;
    }

    shared_ptr<TensorType<DType> > beta_weight() {
      return beta_weight_;
    }

    void set_gamma_update(shared_ptr<TensorType<DType> > update) {
      gamma_update_ = update;
    }

    shared_ptr<TensorType<DType> > gamma_update() {
      return gamma_update_;
    }

    void set_beta_update(shared_ptr<TensorType<DType> > update) {
      beta_update_ = update;
    }

    shared_ptr<TensorType<DType> > beta_update() {
      return beta_update_;
    }

    void set_input_var(shared_ptr<TensorType<DType> > input_var) {
      input_var_ = input_var;
    }

    shared_ptr<TensorType<DType> > input_var() {
      return input_var_;
    }

    void set_input_hat(shared_ptr<TensorType<DType> > input_hat) {
      input_hat_ = input_hat;
    }

    shared_ptr<TensorType<DType> > input_hat() {
      return input_hat_;
    }

    DType epsilon() {
      return epsilon_;
    }

   private:
    const string name_;
    const string gamma_optimizer_name_;
    const string gamma_filler_name_;
    const string beta_optimizer_name_;
    const string beta_filler_name_;
    const DType epsilon_;
    shared_ptr<TensorType<DType> > gamma_weight_;
    shared_ptr<TensorType<DType> > beta_weight_;
    shared_ptr<TensorType<DType> > gamma_update_;
    shared_ptr<TensorType<DType> > beta_update_;
    shared_ptr<TensorType<DType> > input_var_;
    shared_ptr<TensorType<DType> > input_hat_;
  };

  // Bias:
  // Y = X + beta
  class Bias {
   public:
    // default
    explicit Bias(const string& name, const string& filler_name,
      const string& optimizer_name) : name_(name),
      optimizer_name_(optimizer_name), filler_name_(filler_name) {}

    const string& name() const {
      return name_;
    }

    const string& optimizer_name() const {
      return optimizer_name_;
    }

    const string& filler_name() const {
      return filler_name_;
    }

    void set_weight(shared_ptr<TensorType<DType> > weight) {
      weight_ = weight;
    }

    shared_ptr<TensorType<DType> > weight() {
      return weight_;
    }

    void set_update(shared_ptr<TensorType<DType> > update) {
      update_ = update;
    }

    shared_ptr<TensorType<DType> > update() {
      return update_;
    }

   private:
    const string name_;
    const string optimizer_name_;
    const string filler_name_;
    shared_ptr<TensorType<DType> > weight_;
    shared_ptr<TensorType<DType> > update_;
  };

 public:
  explicit ParamLayer(const string& name,
    const string& filler_name, const string& optimizer_name,
    shared_ptr<Activation<TensorType, DType> > activation) :
    Layer<TensorType, DType>(name), filler_name_(filler_name),
    optimizer_name_(optimizer_name), activation_(activation) {}
  virtual ~ParamLayer() {}

  virtual void Init(const Shape& input_shape,
    shared_ptr<FillerWrapper<TensorType, DType> > filler_wrapper,
    shared_ptr<Scheduler<TensorType, DType> > scheduler) {
    Layer<TensorType, DType>::Init(input_shape, filler_wrapper,
      scheduler);
    // set filler
    filler_wrapper->AddLayer(this->filler_name_, this->name_,
      this->weight_);
    // set update state
    scheduler->AddLayer(this->optimizer_name_, this->name_,
      this->weight_, this->update_);

    const Shape& output_shape = (this->forward_output_)->shape();
    Shape shape(1);
    shape[0] = output_shape.size() / output_shape[0];

    if (bias_ != 0) {
      // make beta tensor
      (this->bias_)->set_weight(
        make_shared<TensorType<DType> >(shape));
      (this->bias_)->set_update(
        make_shared<TensorType<DType> >(shape));
      // set filler
      filler_wrapper->AddLayer((this->bias_)->filler_name(),
        (this->bias_)->name(), (this->bias_)->weight());
      // set update state
      scheduler->AddLayer((this->bias_)->optimizer_name(),
        (this->bias_)->name(), (this->bias_)->weight(),
        (this->bias_)->update());
    }

    if (batch_norm_ != 0) {
      // make beta tensor
      (this->batch_norm_)->set_beta_weight(
        make_shared<TensorType<DType> >(shape));
      (this->batch_norm_)->set_beta_update(
        make_shared<TensorType<DType> >(shape));
      // make gamma tensor
      (this->batch_norm_)->set_gamma_weight(
        make_shared<TensorType<DType> >(shape));
      (this->batch_norm_)->set_gamma_update(
        make_shared<TensorType<DType> >(shape));
      // input mean and variance
      (this->batch_norm_)->set_input_hat(
        make_shared<TensorType<DType> >(output_shape));
      (this->batch_norm_)->set_input_var(
        make_shared<TensorType<DType> >(shape));
      // set filler
      filler_wrapper->AddLayer((this->batch_norm_)->beta_filler_name(),
        (this->batch_norm_)->name() + "_beta",
        (this->batch_norm_)->beta_weight());
      filler_wrapper->AddLayer((this->batch_norm_)->gamma_filler_name(),
        (this->batch_norm_)->name() + "_gamma",
        (this->batch_norm_)->gamma_weight());
      // set update state
      scheduler->AddLayer((this->batch_norm_)->beta_optimizer_name(),
        (this->batch_norm_)->name() + "_beta",
        (this->batch_norm_)->beta_weight(),
        (this->batch_norm_)->beta_update());
      scheduler->AddLayer((this->batch_norm_)->gamma_optimizer_name(),
        (this->batch_norm_)->name() + "_gamma",
        (this->batch_norm_)->gamma_weight(),
        (this->batch_norm_)->gamma_update());
    }
  }

  virtual void InitImpl(const Shape& input_shape) = 0;

  // forward
  virtual void ForwardProp(shared_ptr<TensorType<DType> > forward_input) {
    Layer<TensorType, DType>::ForwardProp(forward_input);
    #ifdef BLITZ_PERFORMANCE
    time_point<system_clock> start, end;
    duration<double> bias_time =
      duration<double>::zero();
    duration<double> batch_norm_time =
      duration<double>::zero();
    duration<double> activation_time =
      duration<double>::zero();
    start = system_clock::now();
    #endif  // BLITZ_PERFORMANCE
    if (bias_ != 0) {
      Backend<TensorType, DType>::BiasForwardFunc(
        (this->forward_output_).get(),
        ((this->bias_)->weight()).get(),
        (this->forward_output_).get());
    }
    #ifdef BLITZ_PERFORMANCE
    end = system_clock::now();
    bias_time = end - start;
    LOG(INFO) << "Forward bias time: " <<
      bias_time.count();
    #endif  // BLITZ_PERFORMANCE

    #ifdef BLITZ_PERFORMANCE
    start = system_clock::now();
    #endif  // BLITZ_PERFORMANCE
    if (batch_norm_ != 0) {
      (this->batch_norm_)->input_var()->Fill(0);
      (this->batch_norm_)->input_hat()->Fill(0);
      Backend<TensorType, DType>::BatchNormForwardFunc(
        this->forward_output_.get(),
        (this->batch_norm_)->gamma_weight().get(),
        (this->batch_norm_)->beta_weight().get(),
        (this->batch_norm_)->epsilon(),
        (this->batch_norm_)->input_var().get(),
        (this->batch_norm_)->input_hat().get(),
        this->forward_output_.get());
    }
    #ifdef BLITZ_PERFORMANCE
    end = system_clock::now();
    batch_norm_time = end - start;
    LOG(INFO) << "Forward batch_norm time: " <<
      batch_norm_time.count();
    #endif  // BLITZ_PERFORMANCE

    #ifdef BLITZ_PERFORMANCE
    start = system_clock::now();
    #endif  // BLITZ_PERFORMANCE
    // activation
    if (this->activation_ != 0)
      this->activation_->Apply(this->forward_output_, this->forward_output_);
    #ifdef BLITZ_PERFORMANCE
    end = system_clock::now();
    activation_time = end - start;
    LOG(INFO) << "Forward activation time: " <<
      activation_time.count();
    #endif  // BLITZ_PERFORMANCE
    // update forward_input
    this->forward_input_ = forward_input;
  }
  virtual void ForwardPropImpl(shared_ptr<TensorType<DType> >
    forward_input) = 0;

  // backward
  virtual void BackwardProp(shared_ptr<TensorType<DType> > backward_input) {
    #ifdef BLITZ_PERFORMANCE
    time_point<system_clock> start, end;
    duration<double> bias_time =
      duration<double>::zero();
    duration<double> batch_norm_time =
      duration<double>::zero();
    duration<double> activation_time =
      duration<double>::zero();
    start = system_clock::now();
    #endif  // BLITZ_PERFORMANCE
    if (this->activation_ != 0)
      this->activation_->Derivative(this->forward_output_, backward_input);
    #ifdef BLITZ_PERFORMANCE
    end = system_clock::now();
    activation_time = end - start;
    LOG(INFO) << "Backward activation time: " <<
      activation_time.count();
    #endif  // BLITZ_PERFORMANCE

    #ifdef BLITZ_PERFORMANCE
    start = system_clock::now();
    #endif  // BLITZ_PERFORMANCE
    if (bias_ != 0) {
      (this->bias_)->update()->Fill(0);
      Backend<TensorType, DType>::BiasBackwardUpdateFunc(
        backward_input.get(),
        (this->bias_)->update().get());
    }
    #ifdef BLITZ_PERFORMANCE
    end = system_clock::now();
    bias_time = end - start;
    LOG(INFO) << "Backward bias time: " <<
      bias_time.count();
    #endif  // BLITZ_PERFORMANCE

    #ifdef BLITZ_PERFORMANCE
    start = system_clock::now();
    #endif  // BLITZ_PERFORMANCE
    if (batch_norm_ != 0) {
      (this->batch_norm_)->beta_update()->Fill(0);
      (this->batch_norm_)->gamma_update()->Fill(0);
      Backend<TensorType, DType>::BatchNormBackwardFunc(
        backward_input.get(),
        (this->batch_norm_)->input_hat().get(),
        (this->batch_norm_)->input_var().get(),
        (this->batch_norm_)->gamma_weight().get(),
        (this->batch_norm_)->epsilon(),
        (this->batch_norm_)->gamma_update().get(),
        (this->batch_norm_)->beta_update().get(), backward_input.get());
    }
    #ifdef BLITZ_PERFORMANCE
    end = system_clock::now();
    batch_norm_time = end - start;
    LOG(INFO) << "Backward batch_norm time: " <<
      batch_norm_time.count();
    #endif  // BLITZ_PERFORMANCE

    (this->update_)->Fill(0);
    Layer<TensorType, DType>::BackwardProp(backward_input);
  }

  virtual void BackwardPropImpl(shared_ptr<TensorType<DType> >
    backward_input) = 0;

  void set_bias(shared_ptr<Bias> bias) {
    this->bias_ = bias;
  }

  void set_batch_norm(shared_ptr<BatchNorm> batch_norm) {
    this->batch_norm_ = batch_norm;
  }

 protected:
  const string filler_name_;
  const string optimizer_name_;

  shared_ptr<TensorType<DType> > weight_;
  shared_ptr<TensorType<DType> > update_;

  shared_ptr<Bias> bias_;
  shared_ptr<BatchNorm> batch_norm_;
  shared_ptr<Activation<TensorType, DType> > activation_;
};

}  // namespace blitz

#endif  // SRC_LAYER_PARAM_LAYER_H_
