#include "initializer/parser.h"

#include "backends/backends.h"
#include "scheduler/gradientdescent.h"

namespace blitz {

template<template <typename> class TensorType, typename DType>
shared_ptr<Optimizer<TensorType, DType> > Parser::SetOptimizer(const YAML::Node& node) const {
  if (!node["name"]) {
    LOG(FATAL) << "'name' parameter missing";
  }
  shared_ptr<Optimizer<TensorType, DType> > optimizer;
  string type = node["type"].as<string>();
  string name = node["name"].as<string>();

  if (type == "GradientDescent") {
    if (!node["learning_rate"])
      LOG(FATAL) << "'learning_rate' parameter missing";

    if (!node["step"])
      LOG(FATAL) << "'step' parameter missing";

    if (!node["change"])
      LOG(FATAL) << "'change' parameter missing";

    if (!node["momentum_coef"])
      LOG(FATAL) << "'momentum_coef' parameter missing";

    if (!node["decay"])
      LOG(FATAL) << "'decay' parameter missing";

    DType learning_rate = node["learning_rate"].as<DType>();
    DType change = node["change"].as<DType>();
    int step = node["step"].as<int>();
    DType momentum_coef = node["momentum_coef"].as<DType>();
    DType decay = node["decay"].as<DType>();
    optimizer = static_pointer_cast<Optimizer<TensorType, DType> >(
      make_shared<Gradientdescent<TensorType, DType> >(name,
        learning_rate, change, step, momentum_coef, decay));
  } else {
    LOG(FATAL) << "Unkown optimizer type: " << type;
  }

  return optimizer;
}

INSTANTIATE_SETTER_CPU(Optimizer);
#ifdef BLITZ_USE_GPU
  INSTANTIATE_SETTER_GPU(Optimizer);
#endif
#ifdef BLITZ_USE_MIC
  INSTANTIATE_SETTER_MIC(Optimizer);
#endif

}  // namespace blitz
