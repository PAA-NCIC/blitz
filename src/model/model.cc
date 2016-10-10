#include "model/model.h"

#include <string>

#include "backends/backends.h"

namespace blitz {

template<template <typename> class TensorType, typename DType>
void Model<TensorType, DType>::Inference(
  shared_ptr<DataIterator<TensorType, DType> > inference_set,
  shared_ptr<DataIterator<TensorType, DType> > inference_label,
  shared_ptr<LayerWrapper<TensorType, DType> > layer_wrapper,
  const string& eval_type) {
  inference_set->Init();
  inference_label->Init();
  layer_wrapper->SetInferenceMode();

  int niteration = inference_set->total() / inference_set->batch_size();
  float accuracy = 0.0f;

  ptime todayUtc(day_clock::universal_day(),
    second_clock::universal_time().time_of_day());
  string output_file = "./result/" + to_simple_string(todayUtc) + ".csv";
  ofstream os(output_file.c_str(), ofstream::out);

  for (int i = 0; i < niteration; ++i) {
    shared_ptr<TensorType<DType> > input = inference_set->GenerateTensor(i);
    shared_ptr<TensorType<DType> > target = inference_label->GenerateTensor(i);

    ForwardProp(layer_wrapper, input, target);

    accuracy += layer_wrapper->Evaluate(target, eval_type);

    (layer_wrapper->forward_output())->OutputCSV(&os);
  }

  accuracy /= niteration;
  LOG(INFO) << "Accuracy: " << accuracy;

  os.close();
}

template<template <typename> class TensorType, typename DType>
void Model<TensorType, DType>::Fit(
  shared_ptr<DataIterator<TensorType, DType> > data_set,
  shared_ptr<DataIterator<TensorType, DType> > data_label,
  shared_ptr<DataIterator<TensorType, DType> > eval_set,
  shared_ptr<DataIterator<TensorType, DType> > eval_label,
  shared_ptr<FillerWrapper<TensorType, DType> > filler_wrapper,
  shared_ptr<LayerWrapper<TensorType, DType> > layer_wrapper,
  shared_ptr<CallbackWrapper> callback_wrapper,
  shared_ptr<Scheduler<TensorType, DType> > scheduler,
  const string& eval_type) {
  // init data set and label
  data_set->Init();
  data_label->Init();
  eval_set->Init();
  eval_label->Init();
  const Shape& input_shape = data_set->input_shape();
  layer_wrapper->Init(input_shape, filler_wrapper, scheduler);

  filler_wrapper->Fill();

  for (int i = 0; i < epoches_; ++i) {
    layer_wrapper->SetTrainMode();

    callback_wrapper->OnEpochBegin(i);

    EpochFit(i, data_set, data_label, layer_wrapper,
      callback_wrapper, scheduler);

    callback_wrapper->OnEpochEnd(i);

    layer_wrapper->SetInferenceMode();
    Evaluate(eval_set, eval_label, layer_wrapper, eval_type);
  }
}

template<template <typename> class TensorType, typename DType>
void Model<TensorType, DType>::Fit(
  shared_ptr<DataIterator<TensorType, DType> > data_set,
  shared_ptr<DataIterator<TensorType, DType> > data_label,
  shared_ptr<FillerWrapper<TensorType, DType> > filler_wrapper,
  shared_ptr<LayerWrapper<TensorType, DType> > layer_wrapper,
  shared_ptr<CallbackWrapper> callback_wrapper,
  shared_ptr<Scheduler<TensorType, DType> > scheduler) {
  // init data set and label
  data_set->Init();
  data_label->Init();
  const Shape& input_shape = data_set->input_shape();
  layer_wrapper->Init(input_shape, filler_wrapper, scheduler);
  layer_wrapper->SetTrainMode();

  filler_wrapper->Fill();

  for (int i = 0; i < epoches_; ++i) {
    callback_wrapper->OnEpochBegin(i);

    EpochFit(i, data_set, data_label, layer_wrapper,
      callback_wrapper, scheduler);

    callback_wrapper->OnEpochEnd(i);
  }
}


template<template <typename> class TensorType, typename DType>
void Model<TensorType, DType>::EpochFit(
  int epoch,
  shared_ptr<DataIterator<TensorType, DType> > data_set,
  shared_ptr<DataIterator<TensorType, DType> > data_label,
  shared_ptr<LayerWrapper<TensorType, DType> > layer_wrapper,
  shared_ptr<CallbackWrapper> callback_wrapper,
  shared_ptr<Scheduler<TensorType, DType> > scheduler) {
  int niteration = data_set->total() / data_set->batch_size();
  time_point<system_clock> start, end;
  start = system_clock::now();

  for (int i = 0; i < niteration; ++i) {
    callback_wrapper->OnBatchBegin(i);

    shared_ptr<TensorType<DType> > input = data_set->GenerateTensor(i);
    shared_ptr<TensorType<DType> > target = data_label->GenerateTensor(i);

    DType loss = ForwardProp(layer_wrapper, input, target);

    BackwardProp(layer_wrapper, target);

    scheduler->Run(epoch, data_set->batch_size());

    callback_wrapper->OnBatchEnd(i, loss);
  }
  end = system_clock::now();

  duration<double> elapsed_seconds = end-start;
  LOG(INFO) << "Elapsed second: " << elapsed_seconds.count();
}

template<template <typename> class TensorType, typename DType>
void Model<TensorType, DType>::Evaluate(
  shared_ptr<DataIterator<TensorType, DType> > eval_set,
  shared_ptr<DataIterator<TensorType, DType> > eval_label,
  shared_ptr<LayerWrapper<TensorType, DType> > layer_wrapper,
  const string& eval_type) {
  int niteration = eval_set->total() / eval_set->batch_size();
  float accuracy = 0.0f;

  for (int i = 0; i < niteration; ++i) {
    shared_ptr<TensorType<DType> > input = eval_set->GenerateTensor(i);
    shared_ptr<TensorType<DType> > target = eval_label->GenerateTensor(i);

    ForwardProp(layer_wrapper, input, target);

    accuracy += layer_wrapper->Evaluate(target, eval_type);
  }

  accuracy /= niteration;
  LOG(INFO) << "Accuracy: " << accuracy;
}

template<template <typename> class TensorType, typename DType>
DType Model<TensorType, DType>::ForwardProp(
  shared_ptr<LayerWrapper<TensorType, DType> > layer_wrapper,
  shared_ptr<TensorType<DType> > input,
  const shared_ptr<TensorType<DType> > target) {
  layer_wrapper->ForwardProp(input);
  return layer_wrapper->ApplyCost(target);
}

template<template <typename> class TensorType, typename DType>
void Model<TensorType, DType>::BackwardProp(
  shared_ptr<LayerWrapper<TensorType, DType> > layer_wrapper,
  const shared_ptr<TensorType<DType> > target) {
  layer_wrapper->DerivativeCost(target);
  layer_wrapper->BackwardProp();
}

INSTANTIATE_CLASS(Model);

}  // namespace blitz

