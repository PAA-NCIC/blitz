#include "initializer/parser.h"

#include "backend/backends.h"
#include "transform/cross_entropy_binary.h"
#include "transform/cross_entropy_multi.h"
#include "transform/square_mean.h"
#include "transform/abs_mean.h"

namespace blitz {

template<template <typename> class TensorType, typename DType>
shared_ptr<Cost<TensorType, DType> >
  Parser::SetCost(const YAML::Node& node) const {
  shared_ptr<Cost<TensorType, DType> > cost;
  string type = node["type"].as<string>();

  if (type == "CrossEntropyBinary") {
    cost = static_pointer_cast<Cost<TensorType, DType> >(
        make_shared<CrossEntropyBinary<TensorType, DType> >());
  } else if (type == "CrossEntropyMulti") {
    cost = static_pointer_cast<Cost<TensorType, DType> >(
        make_shared<CrossEntropyMulti<TensorType, DType> >());
  } else if (type == "SquareMean") {
    cost = static_pointer_cast<Cost<TensorType, DType> >(
        make_shared<SquareMean<TensorType, DType> >());
  } else if (type == "AbsMean") {
    cost = static_pointer_cast<Cost<TensorType, DType> >(
        make_shared<AbsMean<TensorType, DType> >());
  } else {
    LOG(FATAL) << "Unkown cost type: " << type;
  }

  return cost;
}

INSTANTIATE_SETTER(Cost);

}  // namespace blitz
