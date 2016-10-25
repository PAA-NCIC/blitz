#ifndef INCLUDE_SCHEDULER_OPTIMIZER_H_
#define INCLUDE_SCHEDULER_OPTIMIZER_H_

#include <map>
#include <string>

#include "backends/tensor.h"
#include "utils/common.h"

namespace blitz {

template<template <typename> class TensorType, typename DType>
class Optimizer {
 public:
  class LayerParam {
   public:
    LayerParam(shared_ptr<TensorType<DType> > weight,
      shared_ptr<TensorType<DType> > update) : weight_(weight),
      update_(update) {
      state_ = make_shared<TensorType<DType> >(weight->shape());
    }

    shared_ptr<TensorType<DType> > weight() {
      return weight_;
    }

    shared_ptr<TensorType<DType> > update() {
      return update_;
    }

    shared_ptr<TensorType<DType> > state() {
      return state_;
    }

   private:
    shared_ptr<TensorType<DType> > weight_;
    shared_ptr<TensorType<DType> > update_;
    shared_ptr<TensorType<DType> > state_;
  };

  typedef typename map<string, shared_ptr<LayerParam> >::iterator
    LayerParamIterator;

 public:
  Optimizer(const string& name, const DType learning_rate,
    const DType change, const size_t step) :
    name_(name), learning_rate_(learning_rate),
    change_(change), step_(step) {}
  virtual ~Optimizer() {}

  void Optimize(const size_t epoch, const size_t batch_size) {
    const DType learning_rate = this->ChangeLearningRate(epoch);

    LayerParamIterator layer_param_it =
      this->layer_params_.begin();

    for (; layer_param_it != this->layer_params_.end();
      ++layer_param_it) {
      #ifdef BLITZ_PERFORMANCE
      time_point<system_clock> start, end;
      duration<double> core_time = duration<double>::zero();
      LOG(INFO) << layer_param_it->first << ":";
      start = system_clock::now();
      #endif  // BLITZ_PERFORMANCE
      this->OptimizeImpl(epoch, batch_size, learning_rate,
        layer_param_it);
      #ifdef BLITZ_PERFORMANCE
      end = system_clock::now();
      core_time = end - start;
      LOG(INFO) << "Optimize core time: " <<
        core_time.count();
      #endif  // BLITZ_PERFORMANCE
    }
  }

  virtual void OptimizeImpl(const size_t epoch, const size_t batch_size,
    const DType learning_rate, LayerParamIterator layer_param_it) = 0;

  void AddLayer(const string& name,
    shared_ptr<TensorType<DType> > weight,
    shared_ptr<TensorType<DType> > update) {
    if (this->layer_params_.find(name) == this->layer_params_.end()) {
      // move constructor
      this->layer_params_[name] = make_shared<LayerParam>(weight, update);
    } else {
      LOG(ERROR) << "layer: " << name << " exists";
    }
  }

 protected:
  DType ChangeLearningRate(const size_t epoch) {
    if (step_ != 0 && change_ != DType(0)) {
      return learning_rate_ * pow(change_, (epoch / step_));
    }
    return learning_rate_;
  }

  map<string, shared_ptr<LayerParam> > layer_params_;

  const string name_;
  const DType learning_rate_;
  const DType change_;
  const size_t step_;

  DISABLE_COPY_AND_ASSIGN(Optimizer);
};

}  // namespace blitz

#endif  // INCLUDE_SCHEDULER_OPTIMIZER_H_
