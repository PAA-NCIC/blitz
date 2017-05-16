#include "initializer/parser.h"

#include "blitz.h"
#include "layers/affine.h"
#include "layers/conv.h"
#include "layers/pooling.h"
#include "layers/dropout.h"
#include "layers/param_layer.h"
#include "utils/blitz_algorithm_function.h"

namespace blitz {

template<template <typename> class TensorType, typename DType>
shared_ptr<Layer<TensorType, DType> > Parser::SetLayer(const YAML::Node& node) const {
  if (!node["name"])
    LOG(FATAL) << "'name' parameter missing";

  if (!node["type"])
    LOG(FATAL) << "'type' parameter missing";

  shared_ptr<Layer<TensorType, DType> > layer;
  string type = node["type"].as<string>();
  string name = node["name"].as<string>();

  if (type == "Affine" || type == "Conv") {
    shared_ptr<ParamLayer<TensorType, DType> > param_layer;

    if (type == "Affine") {
      if (!node["nout"])
        LOG(FATAL) << "'nout' parameter missing";

      if (!node["filler"])
        LOG(FATAL) << "'filler' parameter missing";

      if (!node["optimizer"])
        LOG(FATAL) << "'optimizer' parameter missing";

      int nout = node["nout"].as<int>();
      string filler_name = node["filler"].as<string>();
      string optimizer_name = node["optimizer"].as<string>();
      BLITZ_ALGORITHM algorithm = node["kernel"] ?
        BlitzParseAlgorithm(node["kernel"].as<string>()) : BLITZ_BLAS_GEMM;

      shared_ptr<Activation<TensorType, DType> > activation;
      if (node["activation"])
        activation = SetActivation<TensorType, DType>(node["activation"]);
      param_layer = static_pointer_cast<ParamLayer<TensorType, DType> >(
        make_shared<Affine<TensorType, DType> >(name, filler_name,
          optimizer_name, activation, nout, algorithm));
    } else if (type == "Conv") {
      int stride = 1;
      int padding = 0;

      if (node["stride"]) {
        stride = node["stride"].as<int>();
      }

      if (node["padding"]) {
        padding = node["padding"].as<int>();
      }

      if (!node["fshape"])
        LOG(FATAL) << "'fshape' parameter missing";

      if (!node["filler"])
        LOG(FATAL) << "'filler' parameter missing";

      if (!node["optimizer"])
        LOG(FATAL) << "'optimizer' parameter missing";

      Shape shape(node["fshape"].as<vector<size_t> >());
      string filler_name = node["filler"].as<string>();
      string optimizer_name = node["optimizer"].as<string>();
      BLITZ_ALGORITHM algorithm = node["kernel"] ?
        BlitzParseAlgorithm(node["kernel"].as<string>()) : BLITZ_CONVOLUTION_BLAS_GEMM;

      shared_ptr<Activation<TensorType, DType> > activation;
      if (node["activation"])
        activation = SetActivation<TensorType, DType>(node["activation"]);

      param_layer = shared_ptr<ParamLayer<TensorType, DType> >(
        static_pointer_cast<ParamLayer<TensorType, DType> >(
          new Conv<TensorType, DType>(name, filler_name,
            optimizer_name, activation, shape, stride, stride, padding,
            padding, algorithm)));
    }

    if (node["bias"]) {
      typedef typename ParamLayer<TensorType, DType>::Bias Bias;
      const YAML::Node bias_node = node["bias"];

      if (!bias_node["name"])
        LOG(FATAL) << "'name' parameter missing";

      if (!bias_node["filler"])
        LOG(FATAL) << "'filler' parameter missing";

      if (!bias_node["optimizer"])
        LOG(FATAL) << "'optimizer' parameter missing";

      string bias_name = bias_node["name"].as<string>();
      string bias_filler_name = bias_node["filler"].as<string>();
      string bias_optimizer_name = bias_node["optimizer"].as<string>();
      shared_ptr<Bias> bias = make_shared<Bias>(bias_name,
        bias_filler_name, bias_optimizer_name);
      param_layer->set_bias(bias);
    }

    if (node["batch_norm"]) {
      typedef typename ParamLayer<TensorType, DType>::BatchNorm BatchNorm;
      const YAML::Node batch_norm_node = node["batch_norm"];

      if (!batch_norm_node["name"])
        LOG(FATAL) << "'name' parameter missing";

      if (!batch_norm_node["gamma_filler"])
        LOG(FATAL) << "'beta_filler' parameter missing";

      if (!batch_norm_node["gamma_optimizer"])
        LOG(FATAL) << "'beta_optimizer' parameter missing";

      if (!batch_norm_node["beta_filler"])
        LOG(FATAL) << "'beta_filler' parameter missing";

      if (!batch_norm_node["beta_optimizer"])
        LOG(FATAL) << "'beta_optimizer' parameter missing";

      string batch_norm_name = batch_norm_node["name"].as<string>();
      string beta_filler_name = batch_norm_node["beta_filler"].
        as<string>();
      string beta_optimizer_name = batch_norm_node["beta_optimizer"].
        as<string>();
      string gamma_filler_name = batch_norm_node["gamma_filler"].
        as<string>();
      string gamma_optimizer_name = batch_norm_node["gamma_optimizer"].
        as<string>();
      shared_ptr<BatchNorm> batch_norm = make_shared<BatchNorm>(
        batch_norm_name, gamma_filler_name, gamma_optimizer_name,
        beta_filler_name, beta_optimizer_name);
      param_layer->set_batch_norm(batch_norm);
    }

    layer = static_pointer_cast<Layer<TensorType, DType> >(param_layer);
  } else if (type == "Pooling") {
    if (!node["fshape"])
      LOG(FATAL) << "'fshape' parameter missing";

    if (!node["stride"])
      LOG(FATAL) << "'stride' parameter missing";

    if (!node["op"])
      LOG(FATAL) << "'op' parameter missing";

    int shape = node["fshape"].as<int>();
    int stride = node["stride"].as<int>();
    string op = node["op"].as<string>();

    layer = static_pointer_cast<Layer<TensorType, DType> >(
      make_shared<Pooling<TensorType, DType> >(name,
        shape, stride, op));
  } else if (type == "Dropout") {
    if (!node["keep"])
      LOG(FATAL) << "'keep' parameter missing";

    DType keep = node["keep"].as<DType>();
    layer = static_pointer_cast<Layer<TensorType, DType> >(
      make_shared<Dropout<TensorType, DType> >(name,
        keep));
  } else {
    LOG(FATAL) << "Unkown layer type: " << type;
  }

  return layer;
}

INSTANTIATE_SETTER_CPU(Layer);
#ifdef BLITZ_USE_GPU
  INSTANTIATE_SETTER_GPU(Layer);
#endif

}  // namespace blitz
