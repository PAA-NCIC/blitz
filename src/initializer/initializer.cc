#include "initializer/initializer.h"

#include "initializer/parser.h"
#include "model/model.h"
#include "blitz.h"

namespace blitz {

scoped_ptr<Initializer::Factory> Initializer::instance_(0);
boost::once_flag Initializer::flag_ = BOOST_ONCE_INIT;

template<template <typename> class TensorType, typename DType>
void Initializer::Initialize(const Parser& parser) {
  LOG(INFO) << "Model type: " << parser.model_type();
  LOG(INFO) << "Evaluation type: " << parser.eval_type();
  LOG(INFO) << "Backend type: " << parser.backend_type();
  LOG(INFO) << "Data path: " << parser.data_path();
  LOG(INFO) << "Data type: " << parser.data_type();
  LOG(INFO) << "Label size: " << parser.label_size();
  LOG(INFO) << "Batch size: " << parser.batch_size();
  LOG(INFO) << "Epoches: " << parser.epoches();

  const int epoches = parser.epoches();
  Model<TensorType, DType> model(epoches);
  shared_ptr<DataIterator<TensorType, DType> > data_set =
    parser.data_set<TensorType, DType>();
  shared_ptr<DataIterator<TensorType, DType> > data_label =
    parser.data_label<TensorType, DType>();
  shared_ptr<FillerWrapper<TensorType, DType> > filler_wrapper =
    parser.filler_wrapper<TensorType, DType>();
  shared_ptr<LayerWrapper<TensorType, DType> > layer_wrapper =
    parser.layer_wrapper<TensorType, DType>();
  shared_ptr<CallbackWrapper> callback_wrapper =
    parser.callback_wrapper();

  if (parser.model_type() == "train") {
    LOG(INFO) << "Training";
    shared_ptr<Scheduler<TensorType, DType> > scheduler =
      parser.scheduler<TensorType, DType>();

    if (parser.eval() == true) {
      shared_ptr<DataIterator<TensorType, DType> > eval_set =
        parser.eval_set<TensorType, DType>();
      shared_ptr<DataIterator<TensorType, DType> > eval_label =
        parser.eval_label<TensorType, DType>();
      const string& eval_type = parser.eval_type();

      model.Fit(data_set, data_label, eval_set, eval_label, filler_wrapper,
        layer_wrapper, callback_wrapper, scheduler, eval_type);
    } else {
      model.Fit(data_set, data_label, filler_wrapper,
        layer_wrapper, callback_wrapper, scheduler);
    }
  }

  if (parser.inference() == true) {
    LOG(INFO) << "Inference";
    shared_ptr<DataIterator<TensorType, DType> > inference_set =
      parser.inference_set<TensorType, DType>();
    shared_ptr<DataIterator<TensorType, DType> > inference_label =
      parser.inference_label<TensorType, DType>();
    const string& eval_type = parser.eval_type();

    model.Inference(inference_set, inference_label, layer_wrapper,
      eval_type);
  }
}

REGISTER_INITIALIZER_CPU;
#ifdef BLITZ_USE_GPU
  REGISTER_INITIALIZER_GPU;
#endif

}  // namespace blitz
