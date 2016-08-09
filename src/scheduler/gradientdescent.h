#ifndef SRC_SCHEDULER_GRADIENTDESCENT_H_
#define SRC_SCHEDULER_GRADIENTDESCENT_H_

#include <string>

#include "util/common.h"
#include "scheduler/optimizer.h"

namespace blitz {

template<template <typename> class TensorType, typename DType>
class Gradientdescent : public Optimizer<TensorType, DType> {
 public:
  typedef typename Optimizer<TensorType, DType>::LayerParam
    LayerParam;
  typedef typename Optimizer<TensorType, DType>::LayerParamIterator
    LayerParamIterator;

 public:
  explicit Gradientdescent(const string& name, const DType learning_rate,
    const DType change, const int step,
    DType momentum_coef = 1.0, DType decay = 1.0) :
    Optimizer<TensorType, DType>(name, learning_rate, change, step),
    momentum_coef_(momentum_coef), decay_(decay) {}
  virtual ~Gradientdescent() {}

  virtual void OptimizeImpl(const int epoch, const int batch_size,
    const DType learning_rate, LayerParamIterator layer_param_it);

 private:
  DType momentum_coef_;
  DType decay_;
};

}  // namespace blitz

#endif  // SRC_SCHEDULER_GRADIENTDESCENT_H_
