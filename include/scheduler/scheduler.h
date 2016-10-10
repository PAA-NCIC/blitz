#ifndef INCLUDE_SCHEDULER_SCHEDULER_H_
#define INCLUDE_SCHEDULER_SCHEDULER_H_

#include <list>
#include <map>
#include <string>

#include "scheduler/optimizer.h"
#include "utils/common.h"

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

  DISABLE_COPY_AND_ASSIGN(Scheduler);
};

}  // namespace blitz

#endif  // INCLUDE_SCHEDULER_SCHEDULER_H_
