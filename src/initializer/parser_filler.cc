#include "initializer/parser.h"

#include "backends/backends.h"
#include "fillers/gaussian.h"
#include "fillers/uniform.h"
#include "fillers/constant.h"

namespace blitz {

template<template <typename> class TensorType, typename DType>
shared_ptr<Filler<TensorType, DType> > Parser::SetFiller(const YAML::Node& node) const {
  shared_ptr<Filler<TensorType, DType> > filler;
  if (!node["name"])
    LOG(FATAL) << "'name' parameter missing";

  if (!node["type"])
    LOG(FATAL) << "'type' parameter missing";

  string name = node["name"].as<string>();
  string type = node["type"].as<string>();

  if (type == "Gaussian") {
    if (!node["loc"])
      LOG(FATAL) << "'loc' parameter missing";

    if (!node["scale"])
      LOG(FATAL) << "'scale' parameter missing";

    DType loc = node["loc"].as<DType>();
    DType scale = node["scale"].as<DType>();
    filler = static_pointer_cast<Filler<TensorType, DType> >(
      make_shared<Gaussian<TensorType, DType> >(name, loc, scale));
  } else if (type == "Uniform") {
    if (!node["low"])
      LOG(FATAL) << "'low' parameter missing";

    if (!node["high"])
      LOG(FATAL) << "'high' parameter missing";

    DType low = node["low"].as<DType>();
    DType high = node["high"].as<DType>();
    filler = static_pointer_cast<Filler<TensorType, DType> >(
      make_shared<Uniform<TensorType, DType> >(name, low, high));
  } else if (type == "Constant") {
    if (!node["val"])
      LOG(FATAL) << "'val' parameter missing";

    DType val = node["val"].as<DType>();
    filler = static_pointer_cast<Filler<TensorType, DType> >(
      make_shared<Constant<TensorType, DType> >(name, val));
  } else {
    LOG(FATAL) << "Unkown param_filler type: " << type;
  }

  return filler;
}

INSTANTIATE_SETTER_CPU(Filler);
#ifdef BLITZ_USE_GPU
  INSTANTIATE_SETTER_GPU(Filler);
#endif
#ifdef BLITZ_USE_MIC
  INSTANTIATE_SETTER_MIC(Filler);
#endif

}  // namespace blitz
