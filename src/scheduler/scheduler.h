#ifndef SRC_SCHEDULER_SCHEDULER_H_
#define SRC_SCHEDULER_SCHEDULER_H_

#include <list>
#include <map>
#include <string>

#include "util/common.h"
#include "scheduler/optimizer.h"

namespace blitz {

template<template <typename> class TensorType, typename DType>
class Scheduler {
 public:
  explicit Scheduler(
    map<string, shared_ptr<Optimizer<TensorType, DType> > > optimizers) :
    optimizers_(optimizers) {}

  void Run(const int epoch, const int batch_size);

  void AddLayer(const string& optimizer_name,
    const string& layer_name,
    shared_ptr<TensorType<DType> > weight,
    shared_ptr<TensorType<DType> > update);

 private:
  map<string, shared_ptr<Optimizer<TensorType, DType> > > optimizers_;
};

}  // namespace blitz

#endif  // SRC_SCHEDULER_SCHEDULER_H_
