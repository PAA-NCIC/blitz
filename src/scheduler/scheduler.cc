#include "scheduler/scheduler.h"

#include "blitz.h"

namespace blitz {

template<template <typename> class TensorType, typename DType>
void Scheduler<TensorType, DType>::Run(const size_t epoch, const size_t batch_size) {
  typename map<string, shared_ptr<Optimizer<TensorType, DType> > >::iterator
    optimizer_it = optimizers_.begin();

  for (; optimizer_it != optimizers_.end(); ++optimizer_it) {
#ifdef BLITZ_DEVELOP
    LOG(INFO) << optimizer_it->first;
#endif
#ifdef BLITZ_PERFORMANCE
    LOG(INFO) << optimizer_it->first;
    time_point<system_clock> start, end;
    duration<double> core_time = duration<double>::zero();
    start = system_clock::now();
#endif
    shared_ptr<Optimizer<TensorType, DType> > optimizer =
      optimizer_it->second;
    optimizer->Optimize(epoch, batch_size);
#ifdef BLITZ_PERFORMANCE
    end = system_clock::now();
    core_time = end - start;
    LOG(INFO) << "Optimize time: " <<
      core_time.count();
#endif
  }
}

template<template <typename> class TensorType, typename DType>
void Scheduler<TensorType, DType>::AddLayer(const string& optimizer_name,
  const string& layer_name, shared_ptr<TensorType<DType> > weight,
  shared_ptr<TensorType<DType> > update) {
  if (optimizers_.find(optimizer_name) == optimizers_.end()) {
    LOG(ERROR) << "no such optimizer: " << optimizer_name;
  }
  shared_ptr<Optimizer<TensorType, DType> > optimizer =
    optimizers_[optimizer_name];
  optimizer->AddLayer(layer_name, weight, update);
}

INSTANTIATE_CLASS_CPU(Scheduler);
#ifdef BLITZ_USE_GPU
  INSTANTIATE_CLASS_GPU(Scheduler);
#endif

}  // namespace blitz

