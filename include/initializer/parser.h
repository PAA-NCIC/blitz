#ifndef INCLUDE_INITIALIZER_PARSER_H_
#define INCLUDE_INITIALIZER_PARSER_H_

#include <yaml-cpp/yaml.h>

#include <list>
#include <map>
#include <string>
#include <vector>

#include "callbacks/callback.h"
#include "callbacks/callback_wrapper.h"
#include "data/data_iterator.h"
#include "layers/layer_wrapper.h"
#include "scheduler/scheduler.h"
#include "scheduler/optimizer.h"
#include "transforms/cost.h"
#include "transforms/activation.h"
#include "fillers/filler.h"
#include "fillers/filler_wrapper.h"
#include "utils/common.h"
#include "utils/blitz_shape_function.h"

namespace blitz {

class Parser {
 public:
  explicit Parser(const YAML::Node& config) : config_(config) {}

  void SetDefaultArgs();

  // getters
  const string& data_path() const {
    if (data_path_ == 0) {
      if (config_["data_path"]) {
        data_path_ = make_shared<string>(config_["data_path"].as<string>());
      } else {
        LOG(FATAL) << "'data_path' parameter missing";
      }
    }
    return *data_path_;
  }

  const string& data_type() const {
    if (data_type_ == 0) {
      if (config_["data_type"]) {
        data_type_ = make_shared<string>(config_["data_type"].as<string>());
      } else {
        LOG(FATAL) << "'data_type' parameter missing";
      }
    }
    return *data_type_;
  }

  const string& model_type() const {
    if (model_type_ == 0) {
      if (config_["model_type"]) {
        model_type_ = make_shared<string>(config_["model_type"].as<string>());
      } else {
        LOG(FATAL) << "'model_type' parameter missing";
      }
    }
    return *model_type_;
  }

  const string& eval_type() const {
    if (eval_type_ == 0) {
      if (config_["eval_type"]) {
        eval_type_ = make_shared<string>(config_["eval_type"].as<string>());
      } else {
        eval_type_ = make_shared<string>("classify");
        LOG(WARNING) << "'eval_type' parameter missing";
      }
    }
    return *eval_type_;
  }

  const string& backend_type() const {
    if (backend_type_ == 0) {
      if (config_["backend_type"]) {
        backend_type_ = make_shared<string>(config_["backend_type"].
          as<string>());
      } else {
        LOG(WARNING) << "'backend_type' parameter missing";
      }
    }
    return *backend_type_;
  }

	BLITZ_DATA_LAYOUT data_layout() const {
		if (data_layout_ == 0) {
			if (config_["data_layout"]) {
        data_layout_ = make_shared<BLITZ_DATA_LAYOUT>(
					BlitzParseShape(config_["data_layout"].as<string>()));
			} else {
        data_layout_ = make_shared<BLITZ_DATA_LAYOUT>(BLITZ_BUFFER_NCHW);
			}
		}
		return *data_layout_;
	}

  const Shape& data_shape() const {
    if (data_shape_ == 0) {
      if (config_["data_shape"]) {
        data_shape_ = make_shared<Shape>(
          config_["data_shape"].as<vector<size_t> >());
      } else {
        LOG(FATAL) << "'data_shape' parameter missing";
      }
    }
    return *data_shape_;
  }

  size_t label_size() const {
    if (label_size_ == 0) {
      if (config_["label_size"]) {
        label_size_ = make_shared<size_t>(config_["label_size"].as<size_t>());
      } else {
        LOG(FATAL) << "'label_size' parameter missing";
      }
    }
    return *label_size_;
  }

  size_t epoches() const {
    if (epoches_ == 0) {
      if (config_["epoches"]) {
        epoches_ = make_shared<size_t>(config_["epoches"].as<size_t>());
      } else {
        LOG(WARNING) << "'epoches' parameter missing";
      }
    }
    return *epoches_;
  }

  size_t batch_size() const {
    if (batch_size_ == 0) {
      if (config_["batch_size"]) {
        batch_size_ = make_shared<size_t>(config_["batch_size"].as<size_t>());
      } else {
        LOG(WARNING) << "'batch_size' parameter missing";
      }
    }
    return *batch_size_;
  }

  size_t pool_size() const {
    if (pool_size_ == 0) {
      if (config_["pool_size"]) {
        pool_size_ = make_shared<size_t>(config_["pool_size"].as<size_t>());
      } else {
        pool_size_ = make_shared<size_t>(3000);
        LOG(WARNING) << "'pool_size' parameter missing";
      }
    }
    return *pool_size_;
  }

  bool eval() const {
    if (eval_ == 0) {
      if (config_["eval"]) {
        eval_ = make_shared<bool>(config_["eval"].as<bool>());
      } else {
        eval_ = make_shared<bool>(false);
        LOG(WARNING) << "'eval' parameter missing";
      }
    }
    return *eval_;
  }

  bool inference() const {
    if (inference_ == 0) {
      if (config_["inference"]) {
        inference_ = make_shared<bool>(config_["inference"].as<bool>());
      } else {
        inference_ = make_shared<bool>(false);
        LOG(WARNING) << "'inference' parameter missing";
      }
    }
    return *inference_;
  }

  // [batch_size, data_shape] = input_shape
  const Shape& input_shape() const {
    if (input_shape_ == 0) {
      const Shape& shape = data_shape();
      input_shape_ = make_shared<Shape>(shape.dimension() + 1);
      (*input_shape_)[0] = batch_size();
      for (size_t i = 0; i < shape.dimension(); ++i) {
        (*input_shape_)[i + 1] = shape[i];
      }
			input_shape_->set_data_layout(data_layout());
    }
    return *input_shape_;
  }

  // [batch_size, label_size] = label_shape
  const Shape& label_shape() const {
    if (label_shape_ == 0) {
      label_shape_ = make_shared<Shape>(2);
      (*label_shape_)[0] = batch_size();
      (*label_shape_)[1] = label_size();
    }
    return *label_shape_;
  }

  // composite types
  shared_ptr<CallbackWrapper> callback_wrapper() const {
    if (!config_["callbacks"])
      LOG(FATAL) << "'callbacks' parameter missing";

    const YAML::Node node = config_["callbacks"];

    list<shared_ptr<Callback> > callbacks;
    for (size_t i = 0 ; i < node.size(); ++i) {
      callbacks.push_back(SetCallback(node[i]));
    }
    shared_ptr<CallbackWrapper> callback_wrapper =
      make_shared<CallbackWrapper>(callbacks);
    return callback_wrapper;
  }

  template<template <typename> class TensorType, typename DType>
  shared_ptr<LayerWrapper<TensorType, DType> > layer_wrapper() const {
    if (!config_["layers"])
      LOG(FATAL) << "'layers' parameter missing";

    const YAML::Node node = config_["layers"];

    list<shared_ptr<Layer<TensorType, DType> > > layers;
    for (size_t i = 0; i < node.size() - 1; ++i) {
      layers.push_back(SetLayer<TensorType, DType>(node[i]));
    }

    shared_ptr<LayerWrapper<TensorType, DType> > layer_wrapper =
      make_shared<LayerWrapper<TensorType, DType> >(layers,
      SetCost<TensorType, DType>(node[node.size() - 1]["cost"]));
    return layer_wrapper;
  }

  template<template <typename> class TensorType, typename DType>
  shared_ptr<Scheduler<TensorType, DType> > scheduler() const {
    if (!config_["scheduler"])
      LOG(FATAL) << "'scheduler' parameter missing";

    const YAML::Node node = config_["scheduler"];

    map<string, shared_ptr<Optimizer<TensorType, DType> > > optimizers;
    for (size_t i = 0; i < node.size(); ++i) {
      const YAML::Node optimizer_node = node[i];
      shared_ptr<Optimizer<TensorType, DType> > optimizer =
        SetOptimizer<TensorType, DType>(optimizer_node);
      optimizers[optimizer_node["name"].as<string>()] = optimizer;
    }

    shared_ptr<Scheduler<TensorType, DType> > scheduler =
      make_shared<Scheduler<TensorType, DType> >(optimizers);
    return scheduler;
  }

  template<template <typename> class TensorType, typename DType>
  shared_ptr<FillerWrapper<TensorType, DType> > filler_wrapper() const {
    if (!config_["fillers"])
      LOG(FATAL) << "'fillers' parameter missing";

    const YAML::Node node = config_["fillers"];

    map<string, shared_ptr<Filler<TensorType, DType> > > fillers;
    for (size_t i = 0; i < node.size(); ++i) {
      const YAML::Node filler_node = node[i];
      shared_ptr<Filler<TensorType, DType> > filler =
        SetFiller<TensorType, DType>(filler_node);
      fillers[filler_node["name"].as<string>()] = filler;
    }

    shared_ptr<FillerWrapper<TensorType, DType> > filler_wrapper =
      make_shared<FillerWrapper<TensorType, DType> >(fillers);
    return filler_wrapper;
  }

  template<template <typename> class TensorType, typename DType>
  shared_ptr<DataIterator<TensorType, DType> > data_set() const {
    string path = data_path();
    if (model_type() == "train") {
      path.append("_train_data.log");
    } else if (model_type() == "inference") {
      path.append("_test_data.log");
    } else {
      LOG(FATAL) << "Unknown model type: " << model_type();
    }
    shared_ptr<DataIterator<TensorType, DType> > data_set =
      make_shared<DataIterator<TensorType, DType> >(
      path, input_shape(), batch_size(), pool_size());
    return data_set;
  }

  template<template <typename> class TensorType, typename DType>
  shared_ptr<DataIterator<TensorType, DType> > data_label() const {
    string path = data_path();
    if (model_type() == "train") {
      path.append("_train_label.log");
    } else if (model_type() == "inference") {
      path.append("_test_label.log");
    } else {
      LOG(FATAL) << "Unknown model type: " << model_type();
    }

    shared_ptr<DataIterator<TensorType, DType> > data_label =
      make_shared<DataIterator<TensorType, DType> >(
      path, label_shape(), batch_size(), pool_size());
    return data_label;
  }

  template<template <typename> class TensorType, typename DType>
  shared_ptr<DataIterator<TensorType, DType> > eval_set() const {
    string path = data_path();
    path.append("_eval_data.log");
    shared_ptr<DataIterator<TensorType, DType> > eval_set =
      make_shared<DataIterator<TensorType, DType> >(
      path, input_shape(), batch_size());
    return eval_set;
  }

  template<template <typename> class TensorType, typename DType>
  shared_ptr<DataIterator<TensorType, DType> > eval_label() const {
    string path = data_path();
    path.append("_eval_label.log");

    shared_ptr<DataIterator<TensorType, DType> > eval_label =
      make_shared<DataIterator<TensorType, DType> >(
      path, label_shape(), batch_size());
    return eval_label;
  }

  template<template <typename> class TensorType, typename DType>
  shared_ptr<DataIterator<TensorType, DType> > inference_set() const {
    string path = data_path();
    path.append("_inference_data.log");
    shared_ptr<DataIterator<TensorType, DType> > inference_set =
      make_shared<DataIterator<TensorType, DType> >(
      path, input_shape(), batch_size());
    return inference_set;
  }

  template<template <typename> class TensorType, typename DType>
  shared_ptr<DataIterator<TensorType, DType> > inference_label() const {
    string path = data_path();
    path.append("_inference_label.log");
    shared_ptr<DataIterator<TensorType, DType> > inference_label =
      make_shared<DataIterator<TensorType, DType> >(
      path, label_shape(), batch_size());
    return inference_label;
  }

 private:
  // subsetters
  shared_ptr<Callback> SetCallback(const YAML::Node& node) const;

  template<template <typename> class TensorType, typename DType>
  shared_ptr<Activation<TensorType, DType> > SetActivation(
    const YAML::Node& node) const;

  template<template <typename> class TensorType, typename DType>
  shared_ptr<Cost<TensorType, DType> > SetCost(
    const YAML::Node& node) const;

  template<template <typename> class TensorType, typename DType>
  shared_ptr<Layer<TensorType, DType> > SetLayer(
    const YAML::Node& node) const;

  template<template <typename> class TensorType, typename DType>
  shared_ptr<Filler<TensorType, DType> > SetFiller(
    const YAML::Node& node) const;

  template<template <typename> class TensorType, typename DType>
  shared_ptr<Optimizer<TensorType, DType> > SetOptimizer(
    const YAML::Node& node) const;

  // root node
  YAML::Node config_;

  // composite types
  mutable shared_ptr<Shape> data_shape_;
  mutable shared_ptr<Shape> input_shape_;
  mutable shared_ptr<Shape> label_shape_;

  // basic types
  mutable shared_ptr<string> data_type_;
  mutable shared_ptr<string> data_path_;
  mutable shared_ptr<string> model_type_;
  mutable shared_ptr<string> eval_type_;
  mutable shared_ptr<string> backend_type_;

  mutable shared_ptr<size_t> epoches_;
  mutable shared_ptr<size_t> batch_size_;
  mutable shared_ptr<size_t> label_size_;
  mutable shared_ptr<size_t> pool_size_;

	mutable shared_ptr<BLITZ_DATA_LAYOUT> data_layout_;

  mutable shared_ptr<bool> eval_;
  mutable shared_ptr<bool> inference_;

  DISABLE_COPY_AND_ASSIGN(Parser);
};

}  //  namespace blitz

#endif  //  INCLUDE_INITIALIZERS_PARSER_H_
