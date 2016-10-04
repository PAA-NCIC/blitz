#ifndef INCLUDE_MODEL_MODEL_H_
#define INCLUDE_MODEL_MODEL_H_

#include <string>

#include "util/common.h"
#include "callback/callback_wrapper.h"
#include "data/data_iterator.h"
#include "scheduler/scheduler.h"
#include "layer/layer_wrapper.h"

namespace blitz {

template<template <typename> class TensorType, typename DType>
class Model {
 public:
  explicit Model(const int epoches) :
    epoches_(epoches) {}

  void Inference(
    shared_ptr<DataIterator<TensorType, DType> > inference_set,
    shared_ptr<DataIterator<TensorType, DType> > inference_label,
    shared_ptr<LayerWrapper<TensorType, DType> > layer_wrapper,
    const string& eval_type);

  // fit and evaluate
  void Fit(
    shared_ptr<DataIterator<TensorType, DType> > data_set,
    shared_ptr<DataIterator<TensorType, DType> > data_label,
    shared_ptr<DataIterator<TensorType, DType> > eval_set,
    shared_ptr<DataIterator<TensorType, DType> > eval_label,
    shared_ptr<FillerWrapper<TensorType, DType> > filler_wrapper,
    shared_ptr<LayerWrapper<TensorType, DType> > layer_wrapper,
    shared_ptr<CallbackWrapper> callback_wrapper,
    shared_ptr<Scheduler<TensorType, DType> > scheduler,
    const string& eval_type);

  // fit without evaluate
  void Fit(
    shared_ptr<DataIterator<TensorType, DType> > data_set,
    shared_ptr<DataIterator<TensorType, DType> > data_label,
    shared_ptr<FillerWrapper<TensorType, DType> > filler_wrapper,
    shared_ptr<LayerWrapper<TensorType, DType> > layer_wrapper,
    shared_ptr<CallbackWrapper> callback_wrapper,
    shared_ptr<Scheduler<TensorType, DType> > scheduler);

 private:
  void EpochFit(
    int epoch,
    shared_ptr<DataIterator<TensorType, DType> > data_set,
    shared_ptr<DataIterator<TensorType, DType> > data_label,
    shared_ptr<LayerWrapper<TensorType, DType> > layer_wrapper,
    shared_ptr<CallbackWrapper> callback_wrapper,
    shared_ptr<Scheduler<TensorType, DType> > scheduler);

  void Evaluate(
    shared_ptr<DataIterator<TensorType, DType> > eval_set,
    shared_ptr<DataIterator<TensorType, DType> > eval_label,
    shared_ptr<LayerWrapper<TensorType, DType> > layer_wrapper,
    const string& eval_type);

  DType ForwardProp(shared_ptr<LayerWrapper<TensorType, DType> > layer_wrapper,
    shared_ptr<TensorType<DType> > input,
    const shared_ptr<TensorType<DType> > target);
  void BackwardProp(shared_ptr<LayerWrapper<TensorType, DType> > layer_wrapper,
    const shared_ptr<TensorType<DType> > target);

  const int epoches_;

  DISABLE_COPY_AND_ASSIGN(Model);
};

}  // namespace blitz

#endif  // INCLUDE_MODEL_MODEL_H_
